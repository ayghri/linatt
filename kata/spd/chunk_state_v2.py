"""O(T) chunked-STATE SPD delta, v2 -- built for LONG context (T>=16K).

Design (fixes the v1 negatives):
  * NO full-psi materialization: psi is built in FEATURE-BLOCKS of BI features-i at a time
    (psi_block = C x BI*E), so Ec = sum_blocks psi_block(K) @ S_block -- a few big
    (C x BI*E)@(BI*E x DVB) matmuls, not M*E tiny per-feature dots. Unlocks M=1 (E^2=4096)
    without the C x E^2 tile that OOMs SRAM.
  * State S lives in HBM (E^2 x DV per head).
  * dv-tile INSIDE one program: A_kk / A_qk / inverse T computed ONCE per chunk (DV-independent)
    and reused across DV-tiles -> no redundant recompute (v1's dv-split-across-programs redid them).
  * A_kk, A_qk from the scalar (k.k)^2 kernel, double (g,h) loop = SPD sum over all group pairs.

Cost O(T * E^2 * DV) ; wins over flash O(T^2 * DV) once E^2 < T. Grid = (B*H,).
in-place state update -> autotune MUST reset_to_zero=['S'] or S accumulates across trials.
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl

try:
    from .chunk_state_delta import spd_delta_state_ref
except ImportError:
    from chunk_state_delta import spd_delta_state_ref

_CFG = [triton.Config({}, num_warps=w, num_stages=1) for w in (2, 4, 8)]


@triton.autotune(configs=_CFG, key=['T', 'C', 'M', 'DV', 'DVB', 'E', 'BI'], reset_to_zero=['S'])
@triton.jit
def _state_v2_kernel(q, k, v, beta, o, Uout, S, s_scale, T,
                     H: tl.constexpr, D: tl.constexpr, E: tl.constexpr, M: tl.constexpr,
                     DV: tl.constexpr, DVB: tl.constexpr, C: tl.constexpr, E2: tl.constexpr,
                     BI: tl.constexpr, WRITE_U: tl.constexpr):
    i_bh = tl.program_id(0).to(tl.int64)
    bqk = i_bh * T * D
    bv = i_bh * T * DV
    bs = i_bh * E2 * DV
    NC = tl.cdiv(T, C)
    o_row = tl.arange(0, C)
    rdv = tl.arange(0, DVB)
    rbe = tl.arange(0, BI * E)                       # features within one i-block
    eye = (o_row[:, None] == o_row[None, :]).to(tl.float32)
    strict = (o_row[:, None] > o_row[None, :]).to(tl.float32)
    blk = o_row // 16
    for c in range(NC):
        c0 = c * C
        b_beta = tl.load(tl.make_block_ptr(beta + i_bh * T, (T,), (1,), (c0,), (C,), (0,)), boundary_check=(0,)).to(tl.float32)
        # --- scalar Grams A_kk, A_qk (no psi) : sum over ALL group pairs (g,h) ---
        Akk = tl.zeros([C, C], dtype=tl.float32)
        Aqk = tl.zeros([C, C], dtype=tl.float32)
        for g in range(M):
            kg = tl.load(tl.make_block_ptr(k + bqk, (T, D), (D, 1), (c0, g * E), (C, E), (1, 0)), boundary_check=(0, 1)) * s_scale
            qg = tl.load(tl.make_block_ptr(q + bqk, (T, D), (D, 1), (c0, g * E), (C, E), (1, 0)), boundary_check=(0, 1)) * s_scale
            for hh in range(M):
                khh = tl.load(tl.make_block_ptr(k + bqk, (T, D), (D, 1), (c0, hh * E), (C, E), (1, 0)), boundary_check=(0, 1)) * s_scale
                ip = tl.dot(kg, tl.trans(khh)); Akk += ip * ip
                ipq = tl.dot(qg, tl.trans(khh)); Aqk += ipq * ipq
        Aqk = tl.where(o_row[:, None] >= o_row[None, :], Aqk, 0.0)      # tril
        # --- block-inverse T = (I + beta*tril(A_kk,-1))^-1  (FLA-style, C=64) ---
        N = b_beta[:, None] * (Akk * strict)
        N_bd = tl.where(blk[:, None] == blk[None, :], N, 0.0)         # 16x16 diagonal blocks
        Tm = eye
        for i in range(1, 16):                                        # forward-sub within 16-blocks
            corr = tl.dot(tl.where((o_row % 16 == i)[:, None], N_bd, 0.0), Tm, allow_tf32=True)
            Tm = tl.where((o_row % 16 == i)[:, None], eye - corr, Tm)
        L = 16                                                        # block-recursive merges 16->32->..->C
        while L < C:
            mL = (o_row[:, None] // (2 * L) == o_row[None, :] // (2 * L)) & (o_row[:, None] % (2 * L) >= L) & (o_row[None, :] % (2 * L) < L)
            Tm = Tm - tl.where(mL, tl.dot(tl.dot(Tm, tl.where(mL, N, 0.0), allow_tf32=True), Tm, allow_tf32=True), 0.0)
            L = L * 2
        Tmb = Tm.to(tl.bfloat16)
        Aqkb = Aqk.to(tl.bfloat16)
        # --- DV-tiles (inside the program; A_kk/A_qk/T shared) ---
        for dv0 in range(0, DV, DVB):
            rowd = (c0 + o_row)[:, None] * DV + (dv0 + rdv)[None, :]
            b_v = tl.load(v + bv + rowd, mask=(c0 + o_row)[:, None] < T, other=0.0).to(tl.float32)
            # Ec = psi(K) @ S  (feature-blocked)
            Ec = tl.zeros([C, DVB], dtype=tl.float32)
            for i0 in range(0, E, BI):
                pK = tl.zeros([C, BI * E], dtype=tl.float32)
                for g in range(M):
                    kg = tl.load(tl.make_block_ptr(k + bqk, (T, D), (D, 1), (c0, g * E), (C, E), (1, 0)), boundary_check=(0, 1)) * s_scale
                    ksl = tl.load(tl.make_block_ptr(k + bqk, (T, D), (D, 1), (c0, g * E + i0), (C, BI), (1, 0)), boundary_check=(0, 1)) * s_scale
                    pK += tl.reshape(ksl[:, :, None] * kg[:, None, :], (C, BI * E))
                S_blk = tl.load(tl.make_block_ptr(S + bs, (E2, DV), (DV, 1), (i0 * E, dv0), (BI * E, DVB), (1, 0)), boundary_check=(0, 1)).to(tl.bfloat16)
                Ec += tl.dot(pK.to(tl.bfloat16), S_blk)
            Vp = b_v - Ec
            U = tl.dot(Tmb, (b_beta[:, None] * Vp).to(tl.bfloat16))       # (C, DVB)
            if WRITE_U:
                tl.store(Uout + bv + rowd, U.to(Uout.dtype.element_ty), mask=(c0 + o_row)[:, None] < T)
            # readout oc = psi(Q) @ S + tril(A_qk) @ U
            oQS = tl.zeros([C, DVB], dtype=tl.float32)
            for i0 in range(0, E, BI):
                pQ = tl.zeros([C, BI * E], dtype=tl.float32)
                for g in range(M):
                    qg = tl.load(tl.make_block_ptr(q + bqk, (T, D), (D, 1), (c0, g * E), (C, E), (1, 0)), boundary_check=(0, 1)) * s_scale
                    qsl = tl.load(tl.make_block_ptr(q + bqk, (T, D), (D, 1), (c0, g * E + i0), (C, BI), (1, 0)), boundary_check=(0, 1)) * s_scale
                    pQ += tl.reshape(qsl[:, :, None] * qg[:, None, :], (C, BI * E))
                S_blk = tl.load(tl.make_block_ptr(S + bs, (E2, DV), (DV, 1), (i0 * E, dv0), (BI * E, DVB), (1, 0)), boundary_check=(0, 1)).to(tl.bfloat16)
                oQS += tl.dot(pQ.to(tl.bfloat16), S_blk)
            oc = oQS + tl.dot(Aqkb, U.to(tl.bfloat16))
            tl.store(o + bv + rowd, oc.to(o.dtype.element_ty), mask=(c0 + o_row)[:, None] < T)
            # state update S_blk += psi(K)^T @ U
            for i0 in range(0, E, BI):
                pK = tl.zeros([C, BI * E], dtype=tl.float32)
                for g in range(M):
                    kg = tl.load(tl.make_block_ptr(k + bqk, (T, D), (D, 1), (c0, g * E), (C, E), (1, 0)), boundary_check=(0, 1)) * s_scale
                    ksl = tl.load(tl.make_block_ptr(k + bqk, (T, D), (D, 1), (c0, g * E + i0), (C, BI), (1, 0)), boundary_check=(0, 1)) * s_scale
                    pK += tl.reshape(ksl[:, :, None] * kg[:, None, :], (C, BI * E))
                upd = tl.dot(tl.trans(pK.to(tl.bfloat16)), U.to(tl.bfloat16))   # (BI*E, DVB)
                pS = tl.make_block_ptr(S + bs, (E2, DV), (DV, 1), (i0 * E, dv0), (BI * E, DVB), (1, 0))
                tl.store(pS, (tl.load(pS, boundary_check=(0, 1)) + upd).to(S.dtype.element_ty), boundary_check=(0, 1))


def spd_delta_state_v2(q, k, v, beta, M, scale=None, C=64, DVB=None, BI=None, return_us=False):
    B, H, T, D = q.shape
    DV = v.shape[-1]; E = D // M; E2 = E * E
    if scale is None:
        scale = 1.0 / E
    if DVB is None:            # full DV = single tile: N=DV matmuls + psi materialized once
        DVB = DV
    DVB = min(DVB, DV)
    while DV % DVB != 0:
        DVB //= 2
    if BI is None:                        # aim BI*E ~ 256 (good matmul K-dim), BI | E
        BI = max(1, 256 // E)
    while E % BI != 0:
        BI //= 2
    q, k, v, beta = (x.contiguous() for x in (q, k, v, beta))
    o = torch.empty(B, H, T, DV, device=q.device, dtype=torch.float32)
    S = torch.zeros(B * H, E2, DV, device=q.device, dtype=torch.float32)
    U = torch.empty(B, H, T, DV, device=q.device, dtype=torch.float32) if return_us else o
    _state_v2_kernel[(B * H,)](q, k, v, beta, o, U, S, scale ** 0.5, T,
                               H=H, D=D, E=E, M=M, DV=DV, DVB=DVB, C=C, E2=E2, BI=BI,
                               WRITE_U=return_us)
    return (o, U, S) if return_us else o


class _StateDelta(torch.autograd.Function):
    """O(T) chunked-state forward; backward reuses flash_delta_bwd (identical delta -> identical
    grad). Backward is still O(T^2) (recomputes U + grams) -- a stopgap until the O(T) state
    backward exists. Inference/no-grad runs the O(T) forward only."""
    @staticmethod
    def forward(ctx, q, k, v, beta, M, scale, C):
        o = spd_delta_state_v2(q, k, v, beta, M, scale=scale, C=C)
        ctx.save_for_backward(q, k, v, beta)
        ctx.M, ctx.scale, ctx.C = M, scale, C
        return o

    @staticmethod
    def backward(ctx, do):
        from .flash_delta import flash_delta_U, flash_delta_bwd
        q, k, v, beta = ctx.saved_tensors
        U = flash_delta_U(q, k, v, beta, ctx.M, scale=ctx.scale, C=ctx.C)     # O(T^2) recompute
        dq, dk, dv, db = flash_delta_bwd(q, k, v, beta, U, do, ctx.M, ctx.scale, ctx.C)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), db.to(beta.dtype), None, None, None


def flash_delta_state(q, k, v, beta, M, scale=None, C=64):
    """Trainable O(T) chunked-state SPD delta (raw numerator, no normalization)."""
    return _StateDelta.apply(q, k, v, beta, M, scale, C)


# ============================ O(T) BACKWARD kernel ============================
@triton.jit
def _state_bwd_kernel(q, k, v, beta, do, U, S, dS, dq, dk, dv, dbeta, s_scale, T,
                      H: tl.constexpr, D: tl.constexpr, E: tl.constexpr, M: tl.constexpr,
                      DV: tl.constexpr, C: tl.constexpr, E2: tl.constexpr, BI: tl.constexpr):
    """Reverse-scan backward. S starts = S_final and is decremented in-place to S_c each step.
    dS carries the state gradient. psi symmetric -> dq/dk via 2 s^2 dPmat @ x_g (feature-blocked)."""
    i_bh = tl.program_id(0).to(tl.int64)
    bqk = i_bh * T * D; bv = i_bh * T * DV; bs = i_bh * E2 * DV
    NC = tl.cdiv(T, C)
    o_row = tl.arange(0, C); rdv = tl.arange(0, DV); rbe = tl.arange(0, BI * E)
    eye = (o_row[:, None] == o_row[None, :]).to(tl.float32)
    strict = (o_row[:, None] > o_row[None, :]).to(tl.float32)
    blk = o_row // 16
    ss2 = 2.0 * s_scale * s_scale
    for cc in range(NC):
        c = NC - 1 - cc
        c0 = c * C
        rowv = (c0 + o_row)[:, None] * DV + rdv[None, :]
        msk = (c0 + o_row)[:, None] < T
        b_beta = tl.load(tl.make_block_ptr(beta + i_bh * T, (T,), (1,), (c0,), (C,), (0,)), boundary_check=(0,)).to(tl.float32)
        b_do = tl.load(tl.make_block_ptr(do + bv, (T, DV), (DV, 1), (c0, 0), (C, DV), (1, 0)), boundary_check=(0, 1)).to(tl.float32)
        b_v = tl.load(tl.make_block_ptr(v + bv, (T, DV), (DV, 1), (c0, 0), (C, DV), (1, 0)), boundary_check=(0, 1)).to(tl.float32)
        b_U = tl.load(tl.make_block_ptr(U + bv, (T, DV), (DV, 1), (c0, 0), (C, DV), (1, 0)), boundary_check=(0, 1)).to(tl.float32)
        # (1) reverse-recompute S_c = S - psi(K)^T U
        for i0 in range(0, E, BI):
            pK = tl.zeros([C, BI * E], dtype=tl.float32)
            for g in range(M):
                kg = tl.load(tl.make_block_ptr(k + bqk, (T, D), (D, 1), (c0, g * E), (C, E), (1, 0)), boundary_check=(0, 1)) * s_scale
                ksl = tl.load(tl.make_block_ptr(k + bqk, (T, D), (D, 1), (c0, g * E + i0), (C, BI), (1, 0)), boundary_check=(0, 1)) * s_scale
                pK += tl.reshape(ksl[:, :, None] * kg[:, None, :], (C, BI * E))
            pS = tl.make_block_ptr(S + bs, (E2, DV), (DV, 1), (i0 * E, 0), (BI * E, DV), (1, 0))
            tl.store(pS, (tl.load(pS, boundary_check=(0, 1)) - tl.dot(tl.trans(pK.to(tl.bfloat16)), b_U.to(tl.bfloat16))).to(S.dtype.element_ty), boundary_check=(0, 1))
        # (2) scalar grams + block inverse
        Akk = tl.zeros([C, C], dtype=tl.float32); Aqk = tl.zeros([C, C], dtype=tl.float32)
        for g in range(M):
            kg = tl.load(tl.make_block_ptr(k + bqk, (T, D), (D, 1), (c0, g * E), (C, E), (1, 0)), boundary_check=(0, 1)) * s_scale
            qg = tl.load(tl.make_block_ptr(q + bqk, (T, D), (D, 1), (c0, g * E), (C, E), (1, 0)), boundary_check=(0, 1)) * s_scale
            for hh in range(M):
                khh = tl.load(tl.make_block_ptr(k + bqk, (T, D), (D, 1), (c0, hh * E), (C, E), (1, 0)), boundary_check=(0, 1)) * s_scale
                ip = tl.dot(kg, tl.trans(khh)); Akk += ip * ip
                ipq = tl.dot(qg, tl.trans(khh)); Aqk += ipq * ipq
        Lkk = Akk * strict
        Lqk = tl.where(o_row[:, None] >= o_row[None, :], Aqk, 0.0)
        N = b_beta[:, None] * Lkk
        N_bd = tl.where(blk[:, None] == blk[None, :], N, 0.0)
        Tm = eye
        for i in range(1, 16):
            corr = tl.dot(tl.where((o_row % 16 == i)[:, None], N_bd, 0.0), Tm, allow_tf32=True)
            Tm = tl.where((o_row % 16 == i)[:, None], eye - corr, Tm)
        L = 16
        while L < C:
            mL = (o_row[:, None] // (2 * L) == o_row[None, :] // (2 * L)) & (o_row[:, None] % (2 * L) >= L) & (o_row[None, :] % (2 * L) < L)
            Tm = Tm - tl.where(mL, tl.dot(tl.dot(Tm, tl.where(mL, N, 0.0), allow_tf32=True), Tm, allow_tf32=True), 0.0)
            L = L * 2
        # (3) Ec = psi(K) S_c ; dU_state = psi(K) dS  (feature-blocked)
        Ec = tl.zeros([C, DV], dtype=tl.float32); dU = tl.zeros([C, DV], dtype=tl.float32)
        for i0 in range(0, E, BI):
            pK = tl.zeros([C, BI * E], dtype=tl.float32)
            for g in range(M):
                kg = tl.load(tl.make_block_ptr(k + bqk, (T, D), (D, 1), (c0, g * E), (C, E), (1, 0)), boundary_check=(0, 1)) * s_scale
                ksl = tl.load(tl.make_block_ptr(k + bqk, (T, D), (D, 1), (c0, g * E + i0), (C, BI), (1, 0)), boundary_check=(0, 1)) * s_scale
                pK += tl.reshape(ksl[:, :, None] * kg[:, None, :], (C, BI * E))
            S_blk = tl.load(tl.make_block_ptr(S + bs, (E2, DV), (DV, 1), (i0 * E, 0), (BI * E, DV), (1, 0)), boundary_check=(0, 1))
            dS_blk = tl.load(tl.make_block_ptr(dS + bs, (E2, DV), (DV, 1), (i0 * E, 0), (BI * E, DV), (1, 0)), boundary_check=(0, 1))
            Ec += tl.dot(pK.to(tl.bfloat16), S_blk.to(tl.bfloat16))
            dU += tl.dot(pK.to(tl.bfloat16), dS_blk.to(tl.bfloat16))
        Vp = b_v - Ec
        dU += tl.dot(tl.trans(Lqk).to(tl.bfloat16), b_do.to(tl.bfloat16))     # + Lqk^T do
        # (4) dbVp = Tm^-T dU ; dTm = -dbVp U^T
        dbVp = tl.dot(tl.trans(Tm).to(tl.bfloat16), dU.to(tl.bfloat16))
        dTm = -tl.dot(dbVp.to(tl.bfloat16), tl.trans(b_U).to(tl.bfloat16))
        dbeta_c = tl.sum(Vp * dbVp, 1) + tl.sum(Lkk * dTm, 1)
        dVp = b_beta[:, None] * dbVp
        dE = -dVp
        dAkk = b_beta[:, None] * (dTm * strict)
        dAkk_s = dAkk + tl.trans(dAkk)
        dLqk = tl.where(o_row[:, None] >= o_row[None, :], tl.dot(b_do.to(tl.bfloat16), tl.trans(b_U).to(tl.bfloat16)), 0.0)
        # (5) dq, dk (feature-blocked, symmetric psi VJP) + (6) dS_c accumulate
        for i0 in range(0, E, BI):
            pK = tl.zeros([C, BI * E], dtype=tl.float32); pQ = tl.zeros([C, BI * E], dtype=tl.float32)
            for g in range(M):
                kg = tl.load(tl.make_block_ptr(k + bqk, (T, D), (D, 1), (c0, g * E), (C, E), (1, 0)), boundary_check=(0, 1)) * s_scale
                qg = tl.load(tl.make_block_ptr(q + bqk, (T, D), (D, 1), (c0, g * E), (C, E), (1, 0)), boundary_check=(0, 1)) * s_scale
                ksl = tl.load(tl.make_block_ptr(k + bqk, (T, D), (D, 1), (c0, g * E + i0), (C, BI), (1, 0)), boundary_check=(0, 1)) * s_scale
                qsl = tl.load(tl.make_block_ptr(q + bqk, (T, D), (D, 1), (c0, g * E + i0), (C, BI), (1, 0)), boundary_check=(0, 1)) * s_scale
                pK += tl.reshape(ksl[:, :, None] * kg[:, None, :], (C, BI * E))
                pQ += tl.reshape(qsl[:, :, None] * qg[:, None, :], (C, BI * E))
            S_blk = tl.load(tl.make_block_ptr(S + bs, (E2, DV), (DV, 1), (i0 * E, 0), (BI * E, DV), (1, 0)), boundary_check=(0, 1))
            dS_blk = tl.load(tl.make_block_ptr(dS + bs, (E2, DV), (DV, 1), (i0 * E, 0), (BI * E, DV), (1, 0)), boundary_check=(0, 1))
            # dPq_block = do S_c^T + dLqk Pk ; dPk_block = U dS^T + dE S_c^T + dAkk_s Pk + dLqk^T Pq
            dPq = tl.dot(b_do.to(tl.bfloat16), tl.trans(S_blk).to(tl.bfloat16)) + tl.dot(dLqk.to(tl.bfloat16), pK.to(tl.bfloat16))
            dPk = (tl.dot(b_U.to(tl.bfloat16), tl.trans(dS_blk).to(tl.bfloat16)) + tl.dot(dE.to(tl.bfloat16), tl.trans(S_blk).to(tl.bfloat16))
                   + tl.dot(dAkk_s.to(tl.bfloat16), pK.to(tl.bfloat16)) + tl.dot(tl.trans(dLqk).to(tl.bfloat16), pQ.to(tl.bfloat16)))
            dPq3 = tl.reshape(dPq, (C, BI, E)); dPk3 = tl.reshape(dPk, (C, BI, E))
            for g in range(M):
                qg = tl.load(tl.make_block_ptr(q + bqk, (T, D), (D, 1), (c0, g * E), (C, E), (1, 0)), boundary_check=(0, 1))
                kg = tl.load(tl.make_block_ptr(k + bqk, (T, D), (D, 1), (c0, g * E), (C, E), (1, 0)), boundary_check=(0, 1))
                dqg = ss2 * tl.sum(dPq3 * qg[:, None, :], axis=2)     # (C, BI)
                dkg = ss2 * tl.sum(dPk3 * kg[:, None, :], axis=2)
                tl.store(tl.make_block_ptr(dq + bqk, (T, D), (D, 1), (c0, g * E + i0), (C, BI), (1, 0)), dqg.to(dq.dtype.element_ty), boundary_check=(0, 1))
                tl.store(tl.make_block_ptr(dk + bqk, (T, D), (D, 1), (c0, g * E + i0), (C, BI), (1, 0)), dkg.to(dk.dtype.element_ty), boundary_check=(0, 1))
            # dS_c += psi(Q)^T do + psi(K)^T dE
            upd = tl.dot(tl.trans(pQ.to(tl.bfloat16)), b_do.to(tl.bfloat16)) + tl.dot(tl.trans(pK.to(tl.bfloat16)), dE.to(tl.bfloat16))
            pdS = tl.make_block_ptr(dS + bs, (E2, DV), (DV, 1), (i0 * E, 0), (BI * E, DV), (1, 0))
            tl.store(pdS, (tl.load(pdS, boundary_check=(0, 1)) + upd).to(dS.dtype.element_ty), boundary_check=(0, 1))
        tl.store(tl.make_block_ptr(dv + bv, (T, DV), (DV, 1), (c0, 0), (C, DV), (1, 0)), dVp.to(dv.dtype.element_ty), boundary_check=(0, 1))
        tl.store(tl.make_block_ptr(dbeta + i_bh * T, (T,), (1,), (c0,), (C,), (0,)), dbeta_c.to(dbeta.dtype.element_ty), boundary_check=(0,))


def spd_delta_state_bwd(q, k, v, beta, do, U, S_final, M, scale=None, C=64, BI=None):
    B, H, T, D = q.shape
    DV = v.shape[-1]; E = D // M; E2 = E * E
    if scale is None:
        scale = 1.0 / E
    if BI is None:                        # backward holds ~6 big tiles; E=64 (M=1) needs BI=1 to fit SRAM
        BI = 1 if E >= 64 else max(1, 128 // E)
    while E % BI != 0:
        BI //= 2
    q, k, v, beta, do, U = (x.contiguous() for x in (q, k, v, beta, do, U))
    dq = torch.empty(B, H, T, D, device=q.device, dtype=torch.float32)
    dk = torch.empty(B, H, T, D, device=q.device, dtype=torch.float32)
    dv = torch.empty(B, H, T, DV, device=q.device, dtype=torch.float32)
    dbeta = torch.empty(B, H, T, device=q.device, dtype=torch.float32)
    S = S_final.clone()
    dS = torch.zeros(B * H, E2, DV, device=q.device, dtype=torch.float32)
    _state_bwd_kernel[(B * H,)](q, k, v, beta, do, U, S, dS, dq, dk, dv, dbeta, scale ** 0.5, T,
                                H=H, D=D, E=E, M=M, DV=DV, C=C, E2=E2, BI=BI, num_warps=4)
    return dq, dk, dv, dbeta
