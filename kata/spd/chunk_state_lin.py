"""O(T) chunked-STATE LINEAR SPD attention (no delta) -- INDEPENDENT-CHAINS design.

Raw numerator: o[t] = sum_{s<=t} kappa(q_t, k_s) v_s, kappa = sum_{g,h} (scale * q_g . k_h)^2.
Chunked-state form gives, per chunk c, THREE independent chains (measured: the serial
psi->outer->accumulate->psi.S chain is latency-bound on the elementwise k(x)k build, so
overlapping independent chains hides that latency -- ablation showed grams/inverse are ~4%):
  A cross : oQS  = psi(Q_c) . S_{c-1}          (reads old state, feature-blocked)
  B intra : oI   = tril((Q_c K_c^T)^2) . V_c   (chunk-local, tensor-core, no state)
  C update: S_c  = S_{c-1} + psi(K_c)^T . V_c   (writes new state, feature-blocked)
  o_c = oQS + oI
A reads S before C writes it -> independent; B is state-free. State S is (B*H, E2, DV) in HBM.
psi(x) = sum_g x_g (x) x_g  (symmetric E x E); grams/state sum over all group pairs (g,h).
"""

from __future__ import annotations
import torch
import triton
import triton.language as tl

_CFG = [triton.Config({}, num_warps=w, num_stages=s) for w in (4, 8) for s in (1, 2, 3)]


@triton.autotune(
    configs=_CFG, key=["T", "C", "M", "DV", "E", "BI"], reset_to_zero=["S"]
)
@triton.jit
def _state_lin_kernel(
    q_addr,
    k_addr,
    v_addr,
    o_addr,
    state_addr,
    state_scale_addr,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    M: tl.constexpr,
    DV: tl.constexpr,
    C: tl.constexpr,
    E2: tl.constexpr,
    BI: tl.constexpr,
):
    idx_bh = tl.program_id(0)
    bqk = idx_bh * T * D
    bv = idx_bh * T * DV
    bs = idx_bh * E2 * DV
    num_chunks = tl.cdiv(T, C)
    o_row = tl.arange(0, C)
    rdv = tl.arange(0, DV)
    causal = o_row[:, None] >= o_row[None, :]

    for c in range(num_chunks):
        c0 = c * C
        rowd = (c0 + o_row)[:, None] * DV + rdv[None, :]
        m_row = (c0 + o_row)[:, None] < T
        b_v = tl.load(v_addr + bv + rowd, mask=m_row, other=0.0).to(tl.bfloat16)

        # ---- Chain B (intra): A_qk = sum_{g,h} (scale q_g . k_h)^2 ; oI = tril(A_qk) @ V ----
        Aqk = tl.zeros([C, C], dtype=tl.float32)
        for g in range(M):
            qg = (
                tl.load(
                    tl.make_block_ptr(
                        q_addr + bqk, (T, D), (D, 1), (c0, g * E), (C, E), (1, 0)
                    ),
                    boundary_check=(0, 1),
                )
                * state_scale_addr
            )
            for hh in range(M):
                khh = (
                    tl.load(
                        tl.make_block_ptr(
                            k_addr + bqk, (T, D), (D, 1), (c0, hh * E), (C, E), (1, 0)
                        ),
                        boundary_check=(0, 1),
                    )
                    * state_scale_addr
                )
                ipq = tl.dot(qg, tl.trans(khh))
                Aqk += ipq * ipq
        Aqk = tl.where(causal, Aqk, 0.0)
        oI = tl.dot(Aqk.to(tl.bfloat16), b_v)  # intra output, no state

        # ---- Chain A (cross): oQS = psi(Q_c) . S_{c-1}  (feature-blocked over i) ----
        # ---- Chain C (update): S += psi(K_c)^T . V_c    (same feature blocks) ----
        oQS = tl.zeros([C, DV], dtype=tl.float32)
        for i0 in range(0, E, BI):
            pQ = tl.zeros([C, BI * E], dtype=tl.float32)
            pK = tl.zeros([C, BI * E], dtype=tl.float32)
            for g in range(M):
                qg = (
                    tl.load(
                        tl.make_block_ptr(
                            q_addr + bqk, (T, D), (D, 1), (c0, g * E), (C, E), (1, 0)
                        ),
                        boundary_check=(0, 1),
                    )
                    * state_scale_addr
                )
                qsl = (
                    tl.load(
                        tl.make_block_ptr(
                            q_addr + bqk, (T, D), (D, 1), (c0, g * E + i0), (C, BI), (1, 0)
                        ),
                        boundary_check=(0, 1),
                    )
                    * state_scale_addr
                )
                kg = (
                    tl.load(
                        tl.make_block_ptr(
                            k_addr + bqk, (T, D), (D, 1), (c0, g * E), (C, E), (1, 0)
                        ),
                        boundary_check=(0, 1),
                    )
                    * state_scale_addr
                )
                ksl = (
                    tl.load(
                        tl.make_block_ptr(
                            k_addr + bqk, (T, D), (D, 1), (c0, g * E + i0), (C, BI), (1, 0)
                        ),
                        boundary_check=(0, 1),
                    )
                    * state_scale_addr
                )
                pQ += tl.reshape(qsl[:, :, None] * qg[:, None, :], (C, BI * E))
                pK += tl.reshape(ksl[:, :, None] * kg[:, None, :], (C, BI * E))
            pS = tl.make_block_ptr(
                state_addr + bs, (E2, DV), (DV, 1), (i0 * E, 0), (BI * E, DV), (1, 0)
            )
            S_blk = tl.load(pS, boundary_check=(0, 1))  # read old state (fp32)
            # bf16 matmul inputs, fp32 accumulate; S_blk+upd both fp32 -> state stays fp32-accumulated
            oQS += tl.dot(pQ.to(tl.bfloat16), S_blk.to(tl.bfloat16))  # chain A
            upd = tl.dot(tl.trans(pK.to(tl.bfloat16)), b_v)  # chain C (fp32 accum)
            tl.store(pS, (S_blk + upd).to(state_addr.dtype.element_ty), boundary_check=(0, 1))

        tl.store(o_addr + bv + rowd, (oQS + oI).to(o_addr.dtype.element_ty), mask=m_row)


# =====================================================================
# O(T) BACKWARD: 2-pass chunked (verified vs autograd, rel ~1e-16).
#   pass 1 (forward scan, rebuild H): dq (full) + intra dk, dv
#   pass 2 (reverse scan, R=sum_{c'>=c} psi(Q)^T do): inter dk, dv
# H and psi(k) are symmetric in (a,b) -> every dpsi is symmetric -> feat-bwd is 2*dpsi.x~.
# =====================================================================
_BCFG = [triton.Config({}, num_warps=w, num_stages=s) for w in (4, 8) for s in (1, 2)]


@triton.autotune(
    configs=_BCFG, key=["T", "C", "M", "DV", "E", "BI"], reset_to_zero=["Hst"]
)
@triton.jit
def _lin_bwd_p1_kernel(
    q,
    k,
    v,
    do,
    dq,
    dk,
    dv,
    Hst,
    s_scale,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    M: tl.constexpr,
    DV: tl.constexpr,
    C: tl.constexpr,
    E2: tl.constexpr,
    BI: tl.constexpr,
):
    i_bh = tl.program_id(0).to(tl.int64)
    bqk = i_bh * T * D
    bv = i_bh * T * DV
    bs = i_bh * E2 * DV
    NC = tl.cdiv(T, C)
    o_row = tl.arange(0, C)
    causal = o_row[:, None] >= o_row[None, :]
    for c in range(NC):
        c0 = c * C
        rowd = (c0 + o_row)[:, None] * DV + tl.arange(0, DV)[None, :]
        m_row = (c0 + o_row)[:, None] < T
        b_v = tl.load(v + bv + rowd, mask=m_row, other=0.0).to(tl.bfloat16)
        b_do = tl.load(do + bv + rowd, mask=m_row, other=0.0).to(tl.bfloat16)
        dA = tl.dot(b_do, tl.trans(b_v))  # (C,C) = do @ V^T
        dA = tl.where(causal, dA, 0.0).to(tl.bfloat16)
        A = tl.zeros([C, C], dtype=tl.float32)
        for i0 in range(0, E, BI):
            pQ = tl.zeros([C, BI * E], dtype=tl.float32)
            pK = tl.zeros([C, BI * E], dtype=tl.float32)
            for g in range(M):
                qg = (
                    tl.load(
                        tl.make_block_ptr(
                            q + bqk, (T, D), (D, 1), (c0, g * E), (C, E), (1, 0)
                        ),
                        boundary_check=(0, 1),
                    )
                    * s_scale
                )
                qs = (
                    tl.load(
                        tl.make_block_ptr(
                            q + bqk, (T, D), (D, 1), (c0, g * E + i0), (C, BI), (1, 0)
                        ),
                        boundary_check=(0, 1),
                    )
                    * s_scale
                )
                kg = (
                    tl.load(
                        tl.make_block_ptr(
                            k + bqk, (T, D), (D, 1), (c0, g * E), (C, E), (1, 0)
                        ),
                        boundary_check=(0, 1),
                    )
                    * s_scale
                )
                ks = (
                    tl.load(
                        tl.make_block_ptr(
                            k + bqk, (T, D), (D, 1), (c0, g * E + i0), (C, BI), (1, 0)
                        ),
                        boundary_check=(0, 1),
                    )
                    * s_scale
                )
                pQ += tl.reshape(qs[:, :, None] * qg[:, None, :], (C, BI * E))
                pK += tl.reshape(ks[:, :, None] * kg[:, None, :], (C, BI * E))
            pQb = pQ.to(tl.bfloat16)
            pKb = pK.to(tl.bfloat16)
            A += tl.dot(pQb, tl.trans(pKb))
            pH = tl.make_block_ptr(
                Hst + bs, (E2, DV), (DV, 1), (i0 * E, 0), (BI * E, DV), (1, 0)
            )
            Hb = tl.load(pH, boundary_check=(0, 1))
            # dpsiQ (C,BI*E) = do H_block^T + dA pK   (symmetric) -> dq_g += 2 s (dpsiQ reshaped . q~_g)
            dpsiQ = tl.dot(b_do, tl.trans(Hb.to(tl.bfloat16))) + tl.dot(dA, pKb)
            dpsiK = tl.dot(tl.trans(dA), pQb)  # dA^T pQ  (intra dk)
            dqr = tl.reshape(dpsiQ, (C, BI, E))
            dkr = tl.reshape(dpsiK, (C, BI, E))
            for g in range(M):
                qg = (
                    tl.load(
                        tl.make_block_ptr(
                            q + bqk, (T, D), (D, 1), (c0, g * E), (C, E), (1, 0)
                        ),
                        boundary_check=(0, 1),
                    )
                    * s_scale
                )
                kg = (
                    tl.load(
                        tl.make_block_ptr(
                            k + bqk, (T, D), (D, 1), (c0, g * E), (C, E), (1, 0)
                        ),
                        boundary_check=(0, 1),
                    )
                    * s_scale
                )
                dqg = 2.0 * tl.sum(dqr * qg[:, None, :], axis=2) * s_scale  # (C,BI)
                dkg = 2.0 * tl.sum(dkr * kg[:, None, :], axis=2) * s_scale
                tl.store(
                    tl.make_block_ptr(
                        dq + bqk, (T, D), (D, 1), (c0, g * E + i0), (C, BI), (1, 0)
                    ),
                    dqg.to(dq.dtype.element_ty),
                    boundary_check=(0, 1),
                )
                # dk intra: accumulate (pass2 adds inter). initialize here.
                tl.store(
                    tl.make_block_ptr(
                        dk + bqk, (T, D), (D, 1), (c0, g * E + i0), (C, BI), (1, 0)
                    ),
                    dkg.to(dk.dtype.element_ty),
                    boundary_check=(0, 1),
                )
            tl.store(
                pH,
                (Hb + tl.dot(tl.trans(pKb), b_v).to(tl.float32)).to(
                    Hst.dtype.element_ty
                ),
                boundary_check=(0, 1),
            )
        # intra dv = tril(A)^T @ do
        Ab = tl.where(causal, A, 0.0).to(tl.bfloat16)
        dvi = tl.dot(tl.trans(Ab), b_do)
        tl.store(dv + bv + rowd, dvi.to(dv.dtype.element_ty), mask=m_row)


@triton.autotune(
    configs=_BCFG, key=["T", "C", "M", "DV", "E", "BI"], reset_to_zero=["Rst"]
)
@triton.jit
def _lin_bwd_p2_kernel(
    q,
    k,
    v,
    do,
    dk2,
    dv2,
    Rst,
    s_scale,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    M: tl.constexpr,
    DV: tl.constexpr,
    C: tl.constexpr,
    E2: tl.constexpr,
    BI: tl.constexpr,
):
    i_bh = tl.program_id(0).to(tl.int64)
    bqk = i_bh * T * D
    bv = i_bh * T * DV
    bs = i_bh * E2 * DV
    NC = tl.cdiv(T, C)
    o_row = tl.arange(0, C)
    for cc in range(NC):
        c = NC - 1 - cc
        c0 = c * C
        rowd = (c0 + o_row)[:, None] * DV + tl.arange(0, DV)[None, :]
        m_row = (c0 + o_row)[:, None] < T
        b_v = tl.load(v + bv + rowd, mask=m_row, other=0.0).to(tl.bfloat16)
        b_do = tl.load(do + bv + rowd, mask=m_row, other=0.0).to(tl.bfloat16)
        dvo = tl.zeros([C, DV], dtype=tl.float32)
        for i0 in range(0, E, BI):
            pQ = tl.zeros([C, BI * E], dtype=tl.float32)
            pK = tl.zeros([C, BI * E], dtype=tl.float32)
            for g in range(M):
                qg = (
                    tl.load(
                        tl.make_block_ptr(
                            q + bqk, (T, D), (D, 1), (c0, g * E), (C, E), (1, 0)
                        ),
                        boundary_check=(0, 1),
                    )
                    * s_scale
                )
                qs = (
                    tl.load(
                        tl.make_block_ptr(
                            q + bqk, (T, D), (D, 1), (c0, g * E + i0), (C, BI), (1, 0)
                        ),
                        boundary_check=(0, 1),
                    )
                    * s_scale
                )
                kg = (
                    tl.load(
                        tl.make_block_ptr(
                            k + bqk, (T, D), (D, 1), (c0, g * E), (C, E), (1, 0)
                        ),
                        boundary_check=(0, 1),
                    )
                    * s_scale
                )
                ks = (
                    tl.load(
                        tl.make_block_ptr(
                            k + bqk, (T, D), (D, 1), (c0, g * E + i0), (C, BI), (1, 0)
                        ),
                        boundary_check=(0, 1),
                    )
                    * s_scale
                )
                pQ += tl.reshape(qs[:, :, None] * qg[:, None, :], (C, BI * E))
                pK += tl.reshape(ks[:, :, None] * kg[:, None, :], (C, BI * E))
            pQb = pQ.to(tl.bfloat16)
            pKb = pK.to(tl.bfloat16)
            pR = tl.make_block_ptr(
                Rst + bs, (E2, DV), (DV, 1), (i0 * E, 0), (BI * E, DV), (1, 0)
            )
            Rb = tl.load(pR, boundary_check=(0, 1))  # R_{c+1} (future, exclusive)
            dvo += tl.dot(pKb, Rb.to(tl.bfloat16))  # psi(K) R
            dpsiK = tl.dot(
                b_v, tl.trans(Rb.to(tl.bfloat16))
            )  # V R^T  (inter dk), (C,BI*E)
            dkr = tl.reshape(dpsiK, (C, BI, E))
            for g in range(M):
                kg = (
                    tl.load(
                        tl.make_block_ptr(
                            k + bqk, (T, D), (D, 1), (c0, g * E), (C, E), (1, 0)
                        ),
                        boundary_check=(0, 1),
                    )
                    * s_scale
                )
                dkg = 2.0 * tl.sum(dkr * kg[:, None, :], axis=2) * s_scale  # (C,BI)
                tl.store(
                    tl.make_block_ptr(
                        dk2 + bqk, (T, D), (D, 1), (c0, g * E + i0), (C, BI), (1, 0)
                    ),
                    dkg.to(dk2.dtype.element_ty),
                    boundary_check=(0, 1),
                )  # overwrite (inter only)
            tl.store(
                pR,
                (Rb + tl.dot(tl.trans(pQb), b_do).to(tl.float32)).to(
                    Rst.dtype.element_ty
                ),
                boundary_check=(0, 1),
            )
        tl.store(
            dv2 + bv + rowd, dvo.to(dv2.dtype.element_ty), mask=m_row
        )  # overwrite (inter only)


def _lin_params(D, M, BI):
    E = D // M
    if BI is None:
        BI = max(1, 256 // E)
    while E % BI != 0:
        BI //= 2
    return E, E * E, BI


class _StateLin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, M, scale, C, BI):
        B, H, T, D = q.shape
        DV = v.shape[-1]
        E, E2, BI = _lin_params(D, M, BI)
        if scale is None:
            scale = 1.0 / E
        q, k, v = (x.contiguous() for x in (q, k, v))
        o = torch.empty(B, H, T, DV, device=q.device, dtype=torch.float32)
        S = torch.zeros(B * H, E2, DV, device=q.device, dtype=torch.float32)
        _state_lin_kernel[(B * H,)](
            q, k, v, o, S, scale**0.5, T, H=H, D=D, E=E, M=M, DV=DV, C=C, E2=E2, BI=BI
        )
        ctx.save_for_backward(q, k, v)
        ctx.M, ctx.scale, ctx.C, ctx.BI, ctx.E, ctx.E2 = M, scale, C, BI, E, E2
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v = ctx.saved_tensors
        B, H, T, D = q.shape
        DV = v.shape[-1]
        M, scale, C, BI, E, E2 = ctx.M, ctx.scale, ctx.C, ctx.BI, ctx.E, ctx.E2
        do = do.contiguous()
        dq = torch.empty_like(q)
        dk1 = torch.empty_like(k)  # pass1 intra dk
        dv1 = torch.empty_like(v)  # pass1 intra dv
        dk2 = torch.empty_like(k)  # pass2 inter dk
        dv2 = torch.empty_like(v)  # pass2 inter dv
        Hst = torch.zeros(B * H, E2, DV, device=q.device, dtype=torch.float32)
        Rst = torch.zeros(B * H, E2, DV, device=q.device, dtype=torch.float32)
        ss = scale**0.5
        _lin_bwd_p1_kernel[(B * H,)](
            q,
            k,
            v,
            do,
            dq,
            dk1,
            dv1,
            Hst,
            ss,
            T,
            H=H,
            D=D,
            E=E,
            M=M,
            DV=DV,
            C=C,
            E2=E2,
            BI=BI,
        )
        _lin_bwd_p2_kernel[(B * H,)](
            q,
            k,
            v,
            do,
            dk2,
            dv2,
            Rst,
            ss,
            T,
            H=H,
            D=D,
            E=E,
            M=M,
            DV=DV,
            C=C,
            E2=E2,
            BI=BI,
        )
        return dq, dk1 + dk2, dv1 + dv2, None, None, None, None


def spd_state_lin(q, k, v, M, scale=None, C=64, BI=None):
    """Linear SPD chunked-state (no delta), differentiable. q,k,v: (B,H,T,*) -> o (B,H,T,DV).
    O(T) forward + O(T) 2-pass backward. State fp32-accumulated, bf16 matmul inputs."""
    return _StateLin.apply(q, k, v, M, scale, C, BI)
