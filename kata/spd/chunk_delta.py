"""KATA-SPD delta rule: chunked forward.

Delta recurrence (un-normalized; psi-normalized elsewhere):
    u_t = beta_t ( v_t  -  sum_{s<t}  <psi(k_s), psi(k_t)>  u_s )
    o_t =              sum_{s<=t}      <psi(q_t), psi(k_s)>  u_s
with the sum-mode SPD feature psi(x) = sum_g vec(x_g (x) x_g),  P = E^2,
so  <psi(a), psi(b)> = sum_{ij} (a_i . b_j)^2   (i,j over the M groups).

Chunked form carries state S (P x DV) = sum_s psi(k_s)^T u_s across chunks. Per chunk:
    V'   = V - psi(K) @ S                     # inter-chunk erase (C x DV)
    T    = (I + tril(beta * A_kk, -1))^{-1}    # A_kk[t,s] = <psi(k_t),psi(k_s)>  (C x C)
    vnew = T @ (beta * V')                     # erase-corrected pseudo-values (C x DV)
    o    = psi(Q) @ S  +  tril(A_qk) @ vnew    # inter + intra causal   (C x DV)
    S   += psi(K)^T @ vnew                     # state update (P x DV)
Crucially A_kk, A_qk and T are C x C and come from the RAW q,k (KK^T squared) -- no
C x P WY factor is ever formed -- so the only P-sized object is the state S, which the
Triton kernel keeps in HBM and streams in P-tiles. Reference (this file) is dense/pure-torch.
"""

from __future__ import annotations

import torch


def _psi(x, M, s):
    """sum-mode SPD feature: (..., D) -> (..., P=E^2). psi(x)=sum_g vec(x_g (x) x_g)."""
    *lead, D = x.shape
    E = D // M
    xg = (x * s).reshape(*lead, M, E)
    outer = xg[..., :, None] * xg[..., None, :]  # (..., M, E, E)
    return outer.reshape(*lead, M, E * E).sum(-2)  # (..., P)


def spd_delta_ref(q, k, v, beta, M, scale=None):
    """Dense reference via a single triangular solve over the whole sequence."""
    B, H, T, D = q.shape
    E = D // M
    s = (1.0 / E if scale is None else scale) ** 0.5
    pk, pq = _psi(k, M, s), _psi(q, M, s)  # (B,H,T,P)
    Akk = torch.einsum("bhtp,bhsp->bhts", pk, pk)  # <psi(k_t),psi(k_s)>
    Aqk = torch.einsum("bhtp,bhsp->bhts", pq, pk)  # <psi(q_t),psi(k_s)>
    I = torch.eye(T, device=q.device, dtype=q.dtype)
    Tm = I + beta[..., None] * torch.tril(Akk, -1)  # (I + beta tril(Akk,-1))
    U = torch.linalg.solve_triangular(Tm, beta[..., None] * v, upper=False)
    o = torch.einsum("bhts,bhsd->bhtd", torch.tril(Aqk), U)
    return o


def spd_delta_chunked_ref(q, k, v, beta, M, scale=None, C=32):
    """Chunked reference carrying the P x DV state S -- mirrors the kernel structure."""
    B, H, T, D = q.shape
    E = D // M
    P = E * E
    DV = v.shape[-1]
    s = (1.0 / E if scale is None else scale) ** 0.5
    assert T % C == 0
    NC = T // C
    S = torch.zeros(B, H, P, DV, device=q.device, dtype=torch.float32)
    Ic = torch.eye(C, device=q.device, dtype=torch.float32)
    o = torch.empty(B, H, T, DV, device=q.device, dtype=torch.float32)
    for c in range(NC):
        sl = slice(c * C, (c + 1) * C)
        pk = _psi(k[:, :, sl], M, s).float()  # (B,H,C,P)
        pq = _psi(q[:, :, sl], M, s).float()
        bc = beta[:, :, sl].float()  # (B,H,C)
        vc = v[:, :, sl].float()
        Akk = torch.einsum("bhtp,bhsp->bhts", pk, pk)  # C x C
        Vp = vc - torch.einsum("bhtp,bhpd->bhtd", pk, S)  # inter-chunk erase
        Tm = Ic + bc[..., None] * torch.tril(Akk, -1)
        vnew = torch.linalg.solve_triangular(Tm, bc[..., None] * Vp, upper=False)
        Aqk = torch.einsum("bhtp,bhsp->bhts", pq, pk)
        o_inter = torch.einsum("bhtp,bhpd->bhtd", pq, S)
        o_intra = torch.einsum("bhts,bhsd->bhtd", torch.tril(Aqk), vnew)
        o[:, :, sl] = o_inter + o_intra
        S = S + torch.einsum("bhtp,bhtd->bhpd", pk, vnew)  # state update
    return o


# ------------------------------------------------------------------ Triton kernel
import triton
import triton.language as tl


@triton.jit
def _chunk_delta_fwd_kernel(
    q,
    k,
    v,
    beta,
    o,
    S_all,
    s_scale,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    M: tl.constexpr,
    P: tl.constexpr,
    DV: tl.constexpr,
    C: tl.constexpr,
    LOGC: tl.constexpr,
    SAVE: tl.constexpr,
):
    """Sum-mode SPD delta, state S (P x DV) carried in registers (fits for M>=4, P<=256).
    All C x C work (A_kk, A_qk, the Neumann-log inverse T) uses fp32/tf32; the P-sized
    dots (psi@S, psi^T@vnew) use bf16 tensor cores. SAVE writes S_c (state entering chunk c)
    to S_all for the backward."""
    i_bh = tl.program_id(0).to(tl.int64)
    bos_qk = i_bh * T * D
    bos_v = i_bh * T * DV
    NC = tl.cdiv(T, C)
    S = tl.zeros([P, DV], dtype=tl.float32)
    o_row = tl.arange(0, C)
    eye = (o_row[:, None] == o_row[None, :]).to(tl.float32)
    strict = (o_row[:, None] > o_row[None, :]).to(tl.float32)
    causal = o_row[:, None] >= o_row[None, :]

    for i_c in range(NC):
        c0 = i_c * C
        row_ok = (c0 + o_row) < T
        if SAVE:
            tl.store(
                tl.make_block_ptr(
                    S_all + (i_bh * NC + i_c) * P * DV,
                    (P, DV),
                    (DV, 1),
                    (0, 0),
                    (P, DV),
                    (1, 0),
                ),
                S,
            )
        b_v = tl.load(
            tl.make_block_ptr(v + bos_v, (T, DV), (DV, 1), (c0, 0), (C, DV), (1, 0)),
            boundary_check=(0, 1),
        ).to(tl.float32)
        b_beta = tl.load(
            tl.make_block_ptr(beta + i_bh * T, (T,), (1,), (c0,), (C,), (0,)),
            boundary_check=(0,),
        ).to(tl.float32)
        psi_q = tl.zeros([C, P], dtype=tl.bfloat16)
        psi_k = tl.zeros([C, P], dtype=tl.bfloat16)
        for g in range(M):
            b_qg = (
                tl.load(
                    tl.make_block_ptr(
                        q + bos_qk, (T, D), (D, 1), (c0, g * E), (C, E), (1, 0)
                    ),
                    boundary_check=(0, 1),
                )
                * s_scale
            )
            b_kg = (
                tl.load(
                    tl.make_block_ptr(
                        k + bos_qk, (T, D), (D, 1), (c0, g * E), (C, E), (1, 0)
                    ),
                    boundary_check=(0, 1),
                )
                * s_scale
            )
            psi_q += tl.reshape(b_qg[:, :, None] * b_qg[:, None, :], (C, P)).to(
                tl.bfloat16
            )
            psi_k += tl.reshape(b_kg[:, :, None] * b_kg[:, None, :], (C, P)).to(
                tl.bfloat16
            )
        psi_k = tl.where(row_ok[:, None], psi_k, 0.0)

        Akk = tl.dot(psi_k, tl.trans(psi_k))  # (C,C) fp32
        Vp = b_v - tl.dot(psi_k, S.to(tl.bfloat16))  # inter erase (C,DV)
        # T = (I + beta*strict(Akk))^{-1} via Neumann doubling
        Pm = -(b_beta[:, None] * (Akk * strict))
        Tm = eye + Pm
        for _i in tl.static_range(1, LOGC):
            Pm = tl.dot(Pm, Pm, allow_tf32=True)
            Tm = Tm + tl.dot(Tm, Pm, allow_tf32=True)
        vnew = tl.dot(Tm, b_beta[:, None] * Vp, allow_tf32=True)  # (C,DV) fp32

        o_inter = tl.dot(psi_q, S.to(tl.bfloat16))  # (C,DV)
        Aqk = tl.dot(psi_q, tl.trans(psi_k))
        Aqk = tl.where(causal & row_ok[None, :], Aqk, 0.0)
        o_intra = tl.dot(Aqk.to(tl.bfloat16), vnew.to(tl.bfloat16))
        tl.store(
            tl.make_block_ptr(o + bos_v, (T, DV), (DV, 1), (c0, 0), (C, DV), (1, 0)),
            (o_inter + o_intra).to(o.dtype.element_ty),
            boundary_check=(0, 1),
        )

        vnew_m = tl.where(row_ok[:, None], vnew, 0.0)
        S += tl.dot(tl.trans(psi_k), vnew_m.to(tl.bfloat16))


def spd_chunk_delta(q, k, v, beta, M, scale=None, C=64, warps=4, save_state=False):
    """SRAM-resident SPD delta chunked forward (sum mode). q,k:(B,H,T,D) v:(B,H,T,DV)
    beta:(B,H,T). Returns o (B,H,T,DV), or (o, S_all) if save_state (S_all: per-chunk
    entering state for the backward). Valid where P=E^2 state fits registers (M>=4)."""
    import math

    B, H, T, D = q.shape
    DV = v.shape[-1]
    E = D // M
    P = E * E
    if scale is None:
        scale = 1.0 / E
    assert T % C == 0 and (C & (C - 1)) == 0, "C power of two dividing T"
    q, k, v, beta = (x.contiguous() for x in (q, k, v, beta))
    NC = T // C
    o = torch.empty(B, H, T, DV, device=q.device, dtype=torch.float32)
    S_all = (
        torch.empty(B, H, NC, P, DV, device=q.device, dtype=torch.float32)
        if save_state
        else torch.empty(1, device=q.device, dtype=torch.float32)
    )
    _chunk_delta_fwd_kernel[(B * H,)](
        q,
        k,
        v,
        beta,
        o,
        S_all,
        scale**0.5,
        T,
        H=H,
        D=D,
        E=E,
        M=M,
        P=P,
        DV=DV,
        C=C,
        LOGC=int(math.log2(C)),
        SAVE=save_state,
        num_warps=warps,
        num_stages=2,
    )
    if save_state:
        return o, S_all
    return o


# ------------------------------------------------------------ analytical backward (ref)
def _psi_bwd(dpsi, x, M, s):
    """grad of psi(x)=sum_g vec((s x_g)(x)(s x_g)) w.r.t x. dpsi (...,P) -> dx (...,D)."""
    *lead, D = x.shape
    E = D // M
    xg = (x * s).reshape(*lead, M, E)  # (...,M,E)
    dO = dpsi.reshape(*lead, E, E)  # (...,E,E), shared across groups (sum mode)
    # d(x_g x_g^T)/dx_g -> (dO + dO^T) @ x_g ; times s (outer used s x_g twice -> chain gives s each)
    dxg = torch.einsum("...ef,...gf->...ge", dO + dO.transpose(-1, -2), xg) * s
    return dxg.reshape(*lead, D)


def spd_delta_chunked_bwd_ref(q, k, v, beta, do, M, scale=None, C=32):
    """Analytical chunked backward. Returns dq,dk,dv,dbeta. Pure torch, validates the math."""
    B, H, T, D = q.shape
    E = D // M
    P = E * E
    DV = v.shape[-1]
    s = (1.0 / E if scale is None else scale) ** 0.5
    NC = T // C
    Ic = torch.eye(C, device=q.device, dtype=torch.float32)
    tril_strict = torch.tril(torch.ones(C, C, device=q.device), -1)
    tril_causal = torch.tril(torch.ones(C, C, device=q.device), 0)

    # forward pass, cache S_c (state entering chunk c) + U_c, T_c
    S = torch.zeros(B, H, P, DV, device=q.device, dtype=torch.float32)
    cache = []
    for c in range(NC):
        sl = slice(c * C, (c + 1) * C)
        pk = _psi(k[:, :, sl], M, s).float()
        pq = _psi(q[:, :, sl], M, s).float()
        bc = beta[:, :, sl].float()
        vc = v[:, :, sl].float()
        Akk = torch.einsum("bhtp,bhsp->bhts", pk, pk)
        Vp = vc - torch.einsum("bhtp,bhpd->bhtd", pk, S)
        N = bc[..., None] * (Akk * tril_strict)
        Tm = torch.linalg.inv(Ic + N)
        U = torch.einsum("bhts,bhsd->bhtd", Tm, bc[..., None] * Vp)
        cache.append((S.clone(), pk, pq, bc, vc, Akk, Vp, N, Tm, U))
        S = S + torch.einsum("bhtp,bhtd->bhpd", pk, U)

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    dbeta = torch.zeros_like(beta)
    dS = torch.zeros(B, H, P, DV, device=q.device, dtype=torch.float32)  # grad of state
    for c in reversed(range(NC)):
        sl = slice(c * C, (c + 1) * C)
        Sc, pk, pq, bc, vc, Akk, Vp, N, Tm, U = cache[c]
        doc = do[:, :, sl].float()
        Aqk = torch.einsum("bhtp,bhsp->bhts", pq, pk)
        Aqkc = Aqk * tril_causal
        # o = pq@S + Aqkc@U
        dpq = torch.einsum("bhtd,bhpd->bhtp", doc, Sc)
        dSc = torch.einsum("bhtp,bhtd->bhpd", pq, doc)
        dAqkc = torch.einsum("bhtd,bhsd->bhts", doc, U)
        dU = torch.einsum("bhts,bhtd->bhsd", Aqkc, doc)
        dAqk = dAqkc * tril_causal
        # S_{c+1} = S + pk^T@U
        dSc = dSc + dS
        dpk = torch.einsum("bhtd,bhpd->bhtp", U, dS)
        dU = dU + torch.einsum("bhtp,bhpd->bhtd", pk, dS)
        # U = T@(b*Vp)
        bVp = bc[..., None] * Vp
        dTm = torch.einsum("bhtd,bhsd->bhts", dU, bVp)
        dbVp = torch.einsum("bhts,bhtd->bhsd", Tm, dU)
        dVp = bc[..., None] * dbVp
        dbeta_c = (Vp * dbVp).sum(-1)
        # Vp = V - pk@S
        dv[:, :, sl] += dVp.to(dv.dtype)
        dpk = dpk - torch.einsum("bhtd,bhpd->bhtp", dVp, Sc)
        dSc = dSc - torch.einsum("bhtp,bhtd->bhpd", pk, dVp)
        # T=(I+N)^{-1}
        dN = -torch.einsum(
            "bhtu,bhuv,bhvs->bhts", Tm.transpose(-1, -2), dTm, Tm.transpose(-1, -2)
        )
        dAkk = bc[..., None] * (dN * tril_strict)
        dbeta_c = dbeta_c + ((Akk * tril_strict) * dN).sum(-1)
        # Akk=pk@pk^T ; Aqk=pq@pk^T
        dpk = dpk + torch.einsum("bhts,bhsp->bhtp", dAkk + dAkk.transpose(-1, -2), pk)
        dpq = dpq + torch.einsum("bhts,bhsp->bhtp", dAqk, pk)
        dpk = dpk + torch.einsum("bhts,bhtp->bhsp", dAqk, pq)
        # psi backward + beta
        dq[:, :, sl] += _psi_bwd(dpq, q[:, :, sl].float(), M, s).to(dq.dtype)
        dk[:, :, sl] += _psi_bwd(dpk, k[:, :, sl].float(), M, s).to(dk.dtype)
        dbeta[:, :, sl] += dbeta_c.to(dbeta.dtype)
        dS = dSc
    return dq, dk, dv, dbeta


# --------------------------------------------------- trainable autograd.Function
class _SPDDelta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, beta, M, scale, C):
        o, S_all = spd_chunk_delta(q, k, v, beta, M, scale=scale, C=C, save_state=True)
        ctx.save_for_backward(q, k, v, beta, S_all)
        ctx.M, ctx.scale, ctx.C = M, scale, C
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, beta, S_all = ctx.saved_tensors
        dq, dk, dv, db = spd_chunk_delta_bwd(
            q, k, v, beta, S_all, do, ctx.M, ctx.scale, ctx.C
        )
        return (
            dq.to(q.dtype),
            dk.to(k.dtype),
            dv.to(v.dtype),
            db.to(beta.dtype),
            None,
            None,
            None,
        )


def spd_delta(q, k, v, beta, M, scale=None, C=64):
    """Trainable SPD delta (sum mode). q,k:(B,H,T,D) v:(B,H,T,DV) beta:(B,H,T) -> o (B,H,T,DV)."""
    return _SPDDelta.apply(q, k, v, beta, M, scale, C)


# ------------------------------------------------------------ Triton backward kernel
@triton.jit
def _chunk_delta_bwd_kernel(
    q,
    k,
    v,
    beta,
    S_all,
    do,
    dq,
    dk,
    dv,
    dbeta,
    s_scale,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    M: tl.constexpr,
    P: tl.constexpr,
    DV: tl.constexpr,
    C: tl.constexpr,
    LOGC: tl.constexpr,
):
    """Reverse chunk scan. Assumes T % C == 0 (row_ok all true). One program per (b,h).
    Recomputes the forward per chunk from the checkpointed state S_c, then applies the
    analytically-derived VJPs; carries dS (grad of the entering state) backward."""
    i_bh = tl.program_id(0).to(tl.int64)
    bos_qk = i_bh * T * D
    bos_v = i_bh * T * DV
    NC = tl.cdiv(T, C)
    o_row = tl.arange(0, C)
    eye = (o_row[:, None] == o_row[None, :]).to(tl.float32)
    strict = (o_row[:, None] > o_row[None, :]).to(tl.float32)
    causal = (o_row[:, None] >= o_row[None, :]).to(tl.float32)
    dS = tl.zeros([P, DV], dtype=tl.float32)

    for i_cr in range(NC):
        i_c = NC - 1 - i_cr
        c0 = i_c * C
        # ---- recompute forward for chunk i_c ----
        b_v = tl.load(
            tl.make_block_ptr(v + bos_v, (T, DV), (DV, 1), (c0, 0), (C, DV), (1, 0)),
            boundary_check=(0, 1),
        ).to(tl.float32)
        b_beta = tl.load(
            tl.make_block_ptr(beta + i_bh * T, (T,), (1,), (c0,), (C,), (0,)),
            boundary_check=(0,),
        ).to(tl.float32)
        psi_q = tl.zeros([C, P], dtype=tl.bfloat16)
        psi_k = tl.zeros([C, P], dtype=tl.bfloat16)
        for g in range(M):
            b_qg = (
                tl.load(
                    tl.make_block_ptr(
                        q + bos_qk, (T, D), (D, 1), (c0, g * E), (C, E), (1, 0)
                    ),
                    boundary_check=(0, 1),
                )
                * s_scale
            )
            b_kg = (
                tl.load(
                    tl.make_block_ptr(
                        k + bos_qk, (T, D), (D, 1), (c0, g * E), (C, E), (1, 0)
                    ),
                    boundary_check=(0, 1),
                )
                * s_scale
            )
            psi_q += tl.reshape(b_qg[:, :, None] * b_qg[:, None, :], (C, P)).to(
                tl.bfloat16
            )
            psi_k += tl.reshape(b_kg[:, :, None] * b_kg[:, None, :], (C, P)).to(
                tl.bfloat16
            )
        Sc = tl.load(
            tl.make_block_ptr(
                S_all + (i_bh * NC + i_c) * P * DV,
                (P, DV),
                (DV, 1),
                (0, 0),
                (P, DV),
                (1, 0),
            )
        )
        Akk = tl.dot(psi_k, tl.trans(psi_k))
        Aqk = tl.dot(psi_q, tl.trans(psi_k))
        Aqkc = tl.where(causal > 0, Aqk, 0.0)
        Vp = b_v - tl.dot(psi_k, Sc.to(tl.bfloat16))
        Pm = -(b_beta[:, None] * (Akk * strict))
        Tm = eye + Pm
        for _i in tl.static_range(1, LOGC):
            Pm = tl.dot(Pm, Pm, allow_tf32=True)
            Tm = Tm + tl.dot(Tm, Pm, allow_tf32=True)
        bVp = b_beta[:, None] * Vp
        U = tl.dot(Tm, bVp, allow_tf32=True)

        # ---- backward ----
        b_do = tl.load(
            tl.make_block_ptr(do + bos_v, (T, DV), (DV, 1), (c0, 0), (C, DV), (1, 0)),
            boundary_check=(0, 1),
        ).to(tl.float32)
        # o = pq@Sc + Aqkc@U
        dpq = tl.dot(b_do.to(tl.bfloat16), tl.trans(Sc.to(tl.bfloat16)))  # (C,P)
        dSc = tl.dot(tl.trans(psi_q), b_do.to(tl.bfloat16))  # (P,DV)
        dAqkc = tl.dot(b_do.to(tl.bfloat16), tl.trans(U.to(tl.bfloat16)))  # (C,C)
        dU = tl.dot(tl.trans(Aqkc.to(tl.bfloat16)), b_do.to(tl.bfloat16))  # (C,DV)
        dAqk = tl.where(causal > 0, dAqkc, 0.0)
        # S' = Sc + pk^T@U
        dSc += dS
        dpk = tl.dot(U.to(tl.bfloat16), tl.trans(dS.to(tl.bfloat16)))  # (C,P)
        dU += tl.dot(psi_k, dS.to(tl.bfloat16))  # (C,DV)
        # U = Tm@bVp
        dTm = tl.dot(dU.to(tl.bfloat16), tl.trans(bVp.to(tl.bfloat16)))  # (C,C)
        dbVp = tl.dot(tl.trans(Tm), dU, allow_tf32=True)  # (C,DV)
        dVp = b_beta[:, None] * dbVp
        dbeta_c = tl.sum(Vp * dbVp, axis=1)  # (C,)
        # Vp = b_v - pk@Sc
        tl.store(
            tl.make_block_ptr(dv + bos_v, (T, DV), (DV, 1), (c0, 0), (C, DV), (1, 0)),
            dVp.to(dv.dtype.element_ty),
            boundary_check=(0, 1),
        )
        dpk -= tl.dot(dVp.to(tl.bfloat16), tl.trans(Sc.to(tl.bfloat16)))  # (C,P)
        dSc -= tl.dot(tl.trans(psi_k), dVp.to(tl.bfloat16))  # (P,DV)
        # Tm = (I+N)^{-1}
        dN = -tl.dot(
            tl.trans(Tm), tl.dot(dTm, tl.trans(Tm), allow_tf32=True), allow_tf32=True
        )
        dAkk = b_beta[:, None] * (dN * strict)
        dbeta_c += tl.sum((Akk * strict) * dN, axis=1)
        # Akk=pk pk^T ; Aqk=pq pk^T
        dpk += tl.dot((dAkk + tl.trans(dAkk)).to(tl.bfloat16), psi_k)  # (C,P)
        dpq += tl.dot(dAqk.to(tl.bfloat16), psi_k)  # (C,P)
        dpk += tl.dot(tl.trans(dAqk.to(tl.bfloat16)), psi_q)  # (C,P)

        # psi backward -> dq, dk (per group), + store dbeta
        dOq = tl.reshape(dpq, (C, E, E))
        dOk = tl.reshape(dpk, (C, E, E))
        for g in range(M):
            xq = (
                tl.load(
                    tl.make_block_ptr(
                        q + bos_qk, (T, D), (D, 1), (c0, g * E), (C, E), (1, 0)
                    ),
                    boundary_check=(0, 1),
                ).to(tl.float32)
                * s_scale
            )
            xk = (
                tl.load(
                    tl.make_block_ptr(
                        k + bos_qk, (T, D), (D, 1), (c0, g * E), (C, E), (1, 0)
                    ),
                    boundary_check=(0, 1),
                ).to(tl.float32)
                * s_scale
            )
            dxq = (
                tl.sum(dOq * xq[:, None, :], axis=2)
                + tl.sum(dOq * xq[:, :, None], axis=1)
            ) * s_scale
            dxk = (
                tl.sum(dOk * xk[:, None, :], axis=2)
                + tl.sum(dOk * xk[:, :, None], axis=1)
            ) * s_scale
            tl.store(
                tl.make_block_ptr(
                    dq + bos_qk, (T, D), (D, 1), (c0, g * E), (C, E), (1, 0)
                ),
                dxq.to(dq.dtype.element_ty),
                boundary_check=(0, 1),
            )
            tl.store(
                tl.make_block_ptr(
                    dk + bos_qk, (T, D), (D, 1), (c0, g * E), (C, E), (1, 0)
                ),
                dxk.to(dk.dtype.element_ty),
                boundary_check=(0, 1),
            )
        tl.store(
            tl.make_block_ptr(dbeta + i_bh * T, (T,), (1,), (c0,), (C,), (0,)),
            dbeta_c.to(dbeta.dtype.element_ty),
            boundary_check=(0,),
        )
        dS = dSc


def spd_chunk_delta_bwd(q, k, v, beta, S_all, do, M, scale=None, C=64, warps=8):
    import math

    B, H, T, D = q.shape
    DV = v.shape[-1]
    E = D // M
    P = E * E
    if scale is None:
        scale = 1.0 / E
    q, k, v, beta, do = (x.contiguous() for x in (q, k, v, beta, do))
    dq = torch.empty_like(q, dtype=torch.float32)
    dk = torch.empty_like(k, dtype=torch.float32)
    dv = torch.empty_like(v, dtype=torch.float32)
    dbeta = torch.empty_like(beta, dtype=torch.float32)
    _chunk_delta_bwd_kernel[(B * H,)](
        q,
        k,
        v,
        beta,
        S_all,
        do,
        dq,
        dk,
        dv,
        dbeta,
        scale**0.5,
        T,
        H=H,
        D=D,
        E=E,
        M=M,
        P=P,
        DV=DV,
        C=C,
        LOGC=int(math.log2(C)),
        num_warps=warps,
        num_stages=1,
    )
    return dq, dk, dv, dbeta
