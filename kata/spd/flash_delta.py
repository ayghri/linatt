"""Flash-attention-style KATA-SPD delta: scalar kernel trick, no psi / no E^2 state.

Kernel:  kappa(x,y) = <x,y>^2  ;  <psi(a),psi(b)> = sum_{ij} (a_i . b_j)^2  (M groups).
Delta (per the residual recurrence, batched-efficient two-pass form):
    A_kk[t,s] = kappa-Gram(k_t, k_s),  A_qk[t,s] = kappa-Gram(q_t, k_s)   (from RAW q,k)
    U = (I + diag(beta) tril(A_kk,-1))^{-1} diag(beta) V                   (pseudo-values)
    o = tril(A_qk) U
Never forms psi(k) (E^2), the E^2 x d_v state S, or the lifted residual r_i -- only the
scalar (q.k)^2 / (k.k)^2 kernels. Wins when E^2 >> T (i.e. M=1, full SPD).
"""

from __future__ import annotations
import torch


def _kappa_gram(a, b, M, s):
    """Gram[t,s] = sum_{ij} ((s a_{t,i}) . (s b_{s,j}))^2 from raw a,b -- no psi."""
    B, H, Ta, D = a.shape
    E = D // M
    ag = (a * s).reshape(B, H, Ta, M, E)
    bg = (b * s).reshape(B, H, b.shape[2], M, E)
    ip = torch.einsum("bhtme,bhsne->bhtsmn", ag, bg)  # (a_{t,i} . b_{s,j})
    return (ip * ip).sum((-1, -2))  # (B,H,Ta,Tb)


def spd_delta_kernel_ref(q, k, v, beta, M, scale=None):
    B, H, T, D = q.shape
    E = D // M
    s = (1.0 / E if scale is None else scale) ** 0.5
    Akk = _kappa_gram(k, k, M, s)
    Aqk = _kappa_gram(q, k, M, s)
    Tm = torch.eye(T, device=q.device, dtype=torch.float32) + beta[
        ..., None
    ] * torch.tril(Akk, -1)
    U = torch.linalg.solve_triangular(Tm, beta[..., None] * v, upper=False)
    return torch.einsum("bhts,bhsd->bhtd", torch.tril(Aqk), U)


# ------------------------------------------------------------ flash U kernel (no E^2 state)
import triton
import triton.language as tl

_AUTO_CFGS = [triton.Config({}, num_warps=w, num_stages=st)
              for w in (2, 4, 8) for st in (1, 2, 3)]   # full sweep for the matmul-heavy block-inverse
_AUTO_CFGS_U = [triton.Config({}, num_warps=4, num_stages=1)]  # DV-split: 4 warps only (H100 wgmma)


@triton.autotune(configs=_AUTO_CFGS, key=['T', 'C', 'M', 'DV'])
@triton.jit
def _flash_U_kernel(q, k, v, beta, U, s_scale, T,
                    H: tl.constexpr, D: tl.constexpr, E: tl.constexpr, M: tl.constexpr,
                    DV: tl.constexpr, C: tl.constexpr, LOGC: tl.constexpr):
    """Pseudo-values U via the all-pairs scalar erase -- NO psi, NO E^2 state.
    Sequential over chunks c; each streams previous chunks c'<c for the inter-chunk erase."""
    i_bh = tl.program_id(0).to(tl.int64)
    bos_qk = i_bh * T * D
    bos_v = i_bh * T * DV
    NC = tl.cdiv(T, C)
    o_row = tl.arange(0, C)
    eye = (o_row[:, None] == o_row[None, :]).to(tl.float32)
    strict = (o_row[:, None] > o_row[None, :]).to(tl.float32)
    for c in range(NC):
        c0 = c * C
        b_v = tl.load(tl.make_block_ptr(v + bos_v, (T, DV), (DV, 1), (c0, 0), (C, DV), (1, 0)), boundary_check=(0, 1)).to(tl.float32)
        b_beta = tl.load(tl.make_block_ptr(beta + i_bh * T, (T,), (1,), (c0,), (C,), (0,)), boundary_check=(0,)).to(tl.float32)
        Ec = tl.zeros([C, DV], dtype=tl.float32)
        for cp in range(0, c):
            cp0 = cp * C
            Akk = tl.zeros([C, C], dtype=tl.float32)
            for g in range(M):
                b_kg = tl.load(tl.make_block_ptr(k + bos_qk, (T, D), (D, 1), (c0, g * E), (C, E), (1, 0)), boundary_check=(0, 1)) * s_scale
                for hh in range(M):
                    b_kph = tl.load(tl.make_block_ptr(k + bos_qk, (T, D), (D, 1), (cp0, hh * E), (C, E), (1, 0)), boundary_check=(0, 1)) * s_scale
                    ip = tl.dot(b_kg, tl.trans(b_kph)); Akk += ip * ip
            b_Ucp = tl.load(tl.make_block_ptr(U + bos_v, (T, DV), (DV, 1), (cp0, 0), (C, DV), (1, 0)), boundary_check=(0, 1)).to(tl.bfloat16)
            Ec += tl.dot(Akk.to(tl.bfloat16), b_Ucp)
        Vp = b_v - Ec
        Akk_cc = tl.zeros([C, C], dtype=tl.float32)
        for g in range(M):
            b_kg = tl.load(tl.make_block_ptr(k + bos_qk, (T, D), (D, 1), (c0, g * E), (C, E), (1, 0)), boundary_check=(0, 1)) * s_scale
            for hh in range(M):
                b_kh = tl.load(tl.make_block_ptr(k + bos_qk, (T, D), (D, 1), (c0, hh * E), (C, E), (1, 0)), boundary_check=(0, 1)) * s_scale
                ip = tl.dot(b_kg, tl.trans(b_kh)); Akk_cc += ip * ip
        # FLA-style block inverse T=(I+N)^-1 (N=beta*tril(A_kk,-1)): 16-block-diag forward
        # substitution (stable, no Neumann series) + block-recursive merges (matmuls). C=64.
        N = b_beta[:, None] * (Akk_cc * strict)
        blk = o_row // 16
        N_bd = tl.where(blk[:, None] == blk[None, :], N, 0.0)            # 16x16 diagonal blocks
        Tm = eye
        for i in range(1, 16):                                          # forward-sub within 16-blocks
            corr = tl.dot(tl.where((o_row % 16 == i)[:, None], N_bd, 0.0), Tm, allow_tf32=True)
            Tm = tl.where((o_row % 16 == i)[:, None], eye - corr, Tm)
        m1 = (blk[:, None] // 2 == blk[None, :] // 2) & (o_row % 32 >= 16)[:, None] & (o_row % 32 < 16)[None, :]
        Tm = Tm - tl.where(m1, tl.dot(tl.dot(Tm, tl.where(m1, N, 0.0), allow_tf32=True), Tm, allow_tf32=True), 0.0)
        m2 = (o_row >= 32)[:, None] & (o_row < 32)[None, :]
        Tm = Tm - tl.where(m2, tl.dot(tl.dot(Tm, tl.where(m2, N, 0.0), allow_tf32=True), Tm, allow_tf32=True), 0.0)
        Uc = tl.dot(Tm, b_beta[:, None] * Vp, allow_tf32=True)
        tl.store(tl.make_block_ptr(U + bos_v, (T, DV), (DV, 1), (c0, 0), (C, DV), (1, 0)), Uc.to(U.dtype.element_ty), boundary_check=(0, 1))


def flash_delta_U(q, k, v, beta, M, scale=None, C=64, warps=4):
    import math
    B, H, T, D = q.shape
    DV = v.shape[-1]; E = D // M
    if scale is None: scale = 1.0 / E
    q, k, v, beta = (x.contiguous() for x in (q, k, v, beta))
    U = torch.empty(B, H, T, DV, device=q.device, dtype=torch.float32)
    _flash_U_kernel[(B * H,)](q, k, v, beta, U, scale ** 0.5, T,
                              H=H, D=D, E=E, M=M, DV=DV, C=C, LOGC=int(math.log2(C)))
    return U


# ------------------------------------------------------ scalar-kernel backward (reference)
def _kappa_gram_bwd(dG, a, b, M, s):
    """VJP of Gram[t,s]=sum_gh (s a_{t,g} . s b_{s,h})^2. Uses grad_x <x,y>^2 = 2<x,y> y."""
    B, H, Ta, D = a.shape
    E = D // M
    ag = (a * s).reshape(B, H, Ta, M, E)
    bg = (b * s).reshape(B, H, b.shape[2], M, E)
    ip = torch.einsum("bhtme,bhsne->bhtsmn", ag, bg)  # <a_{t,g}, b_{s,h}>
    dip = 2.0 * dG[..., None, None] * ip  # d/dip of dG.(ip^2)
    da = torch.einsum("bhtsmn,bhsne->bhtme", dip, bg).reshape(B, H, Ta, D) * s
    db = torch.einsum("bhtsmn,bhtme->bhsne", dip, ag).reshape(B, H, b.shape[2], D) * s
    return da, db


def spd_delta_kernel_bwd_ref(q, k, v, beta, do, M, scale=None):
    """Two-pass scalar-kernel backward (no psi, no E^2 state). Returns dq,dk,dv,dbeta."""
    B, H, T, D = q.shape
    E = D // M
    s = (1.0 / E if scale is None else scale) ** 0.5
    Akk = _kappa_gram(k, k, M, s)
    Aqk = _kappa_gram(q, k, M, s)
    Lqk = torch.tril(Aqk)
    Lkk = torch.tril(Akk, -1)
    Tm = torch.eye(T, device=q.device) + beta[..., None] * Lkk
    U = torch.linalg.solve_triangular(Tm, beta[..., None] * v, upper=False)
    # readout: o = Lqk U
    dAqk = torch.tril(torch.einsum("bhtd,bhsd->bhts", do, U))
    dU = torch.einsum("bhts,bhtd->bhsd", Lqk, do)
    # U = Tm^{-1}(beta v):  dbV = Tm^{-T} dU ;  dTm = -dbV U^T
    dbV = torch.linalg.solve_triangular(Tm.transpose(-1, -2), dU, upper=True)
    dTm = -torch.einsum("bhsd,bhtd->bhst", dbV, U)
    dv = beta[..., None] * dbV
    dbeta = (v * dbV).sum(-1)
    dAkk = beta[..., None] * torch.tril(dTm, -1)
    dbeta = dbeta + (Lkk * dTm).sum(-1)
    # kernel grads
    dq, dk1 = _kappa_gram_bwd(dAqk, q, k, M, s)
    dk2a, dk2b = _kappa_gram_bwd(dAkk, k, k, M, s)
    dk = dk1 + dk2a + dk2b
    return dq, dk, dv, dbeta


# ================================================ flash backward (scalar, no state)
@triton.jit
def _gram_blk(
    src,
    other,
    bqk,
    r0,
    c0,
    s_scale,
    C: tl.constexpr,
    E: tl.constexpr,
    M: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
):
    """A[t,s] = sum_gh (src_{r0+t,g} . other_{c0+s,h})^2  (C x C), scaled."""
    A = tl.zeros([C, C], dtype=tl.float32)
    for g in range(M):
        a = (
            tl.load(
                tl.make_block_ptr(
                    src + bqk, (T, D), (D, 1), (r0, g * E), (C, E), (1, 0)
                ),
                boundary_check=(0, 1),
            )
            * s_scale
        )
        for hh in range(M):
            b = (
                tl.load(
                    tl.make_block_ptr(
                        other + bqk, (T, D), (D, 1), (c0, hh * E), (C, E), (1, 0)
                    ),
                    boundary_check=(0, 1),
                )
                * s_scale
            )
            ip = tl.dot(a, tl.trans(b))
            A += ip * ip
    return A


@triton.autotune(configs=_AUTO_CFGS, key=['T', 'C', 'M', 'DV'])
@triton.jit
def _dU_kernel(
    q,
    k,
    do,
    U,
    dU,
    s_scale,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    M: tl.constexpr,
    DV: tl.constexpr,
    C: tl.constexpr,
    NC: tl.constexpr,
):
    """Readout VJP: dU_s = sum_{t>=s} A_qk[t,s] do_t. Program per (bh, key-block s)."""
    i_bh = tl.program_id(0).to(tl.int64)
    i_cs = tl.program_id(1)
    bqk = i_bh * T * D
    bv = i_bh * T * DV
    cs0 = i_cs * C
    o_row = tl.arange(0, C)
    acc = tl.zeros([C, DV], dtype=tl.float32)
    for i_ct in range(i_cs, NC):
        ct0 = i_ct * C
        Aqk = _gram_blk(
            q, k, bqk, ct0, cs0, s_scale, C, E, M, T, D
        )  # rows=t(query), cols=s(key)
        if i_ct == i_cs:
            Aqk = tl.where(o_row[:, None] >= o_row[None, :], Aqk, 0.0)
        b_do = tl.load(
            tl.make_block_ptr(do + bv, (T, DV), (DV, 1), (ct0, 0), (C, DV), (1, 0)),
            boundary_check=(0, 1),
        ).to(tl.bfloat16)
        acc += tl.dot(tl.trans(Aqk).to(tl.bfloat16), b_do)
    tl.store(
        tl.make_block_ptr(dU + bv, (T, DV), (DV, 1), (cs0, 0), (C, DV), (1, 0)),
        acc.to(dU.dtype.element_ty),
        boundary_check=(0, 1),
    )


def _dU(q, k, do, U, M, s, C):
    import math

    B, H, T, D = q.shape
    DV = do.shape[-1]
    E = D // M
    NC = T // C
    dU = torch.empty(B, H, T, DV, device=q.device, dtype=torch.float32)
    _dU_kernel[(B * H, NC)](
        q,
        k,
        do,
        U,
        dU,
        s,
        T,
        H=H,
        D=D,
        E=E,
        M=M,
        DV=DV,
        C=C,
        NC=NC,
    )
    return dU


@triton.autotune(configs=_AUTO_CFGS, key=['T', 'C', 'M', 'DV'])
@triton.jit
def _dbV_kernel(
    k,
    beta,
    dU,
    dbV,
    s_scale,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    M: tl.constexpr,
    DV: tl.constexpr,
    C: tl.constexpr,
    LOGC: tl.constexpr,
    NC: tl.constexpr,
):
    """Reverse solve dbV = Tm^{-T} dU (Tm^T upper). Program per (bh); reverse chunk scan."""
    i_bh = tl.program_id(0).to(tl.int64)
    bqk = i_bh * T * D
    bv = i_bh * T * DV
    o_row = tl.arange(0, C)
    eye = (o_row[:, None] == o_row[None, :]).to(tl.float32)
    strict = (o_row[:, None] > o_row[None, :]).to(tl.float32)
    for cr in range(NC):
        c = NC - 1 - cr
        c0 = c * C
        inter = tl.zeros([C, DV], dtype=tl.float32)
        for cp in range(c + 1, NC):
            cp0 = cp * C
            Akk = _gram_blk(
                k, k, bqk, c0, cp0, s_scale, C, E, M, T, D
            )  # rows=t(c), cols=s(c'>c)
            b_beta = tl.load(
                tl.make_block_ptr(beta + i_bh * T, (T,), (1,), (cp0,), (C,), (0,)),
                boundary_check=(0,),
            ).to(tl.float32)
            b_dbV = tl.load(
                tl.make_block_ptr(
                    dbV + bv, (T, DV), (DV, 1), (cp0, 0), (C, DV), (1, 0)
                ),
                boundary_check=(0, 1),
            ).to(tl.float32)
            inter += tl.dot(Akk.to(tl.bfloat16), (b_beta[:, None] * b_dbV).to(tl.bfloat16))
        b_dU = tl.load(
            tl.make_block_ptr(dU + bv, (T, DV), (DV, 1), (c0, 0), (C, DV), (1, 0)),
            boundary_check=(0, 1),
        ).to(tl.float32)
        rhs = b_dU - inter
        Akk_cc = _gram_blk(k, k, bqk, c0, c0, s_scale, C, E, M, T, D)
        b_beta_c = tl.load(
            tl.make_block_ptr(beta + i_bh * T, (T,), (1,), (c0,), (C,), (0,)),
            boundary_check=(0,),
        ).to(tl.float32)
        # FLA block-inverse Tm=(I+N)^-1 (same as fwd), then dbV = Tm^-T @ rhs = trans(Tm) @ rhs.
        N = b_beta_c[:, None] * (Akk_cc * strict)
        blk = o_row // 16
        N_bd = tl.where(blk[:, None] == blk[None, :], N, 0.0)
        Tm = eye
        for i in range(1, 16):
            corr = tl.dot(tl.where((o_row % 16 == i)[:, None], N_bd, 0.0), Tm, allow_tf32=True)
            Tm = tl.where((o_row % 16 == i)[:, None], eye - corr, Tm)
        m1 = (blk[:, None] // 2 == blk[None, :] // 2) & (o_row % 32 >= 16)[:, None] & (o_row % 32 < 16)[None, :]
        Tm = Tm - tl.where(m1, tl.dot(tl.dot(Tm, tl.where(m1, N, 0.0), allow_tf32=True), Tm, allow_tf32=True), 0.0)
        m2 = (o_row >= 32)[:, None] & (o_row < 32)[None, :]
        Tm = Tm - tl.where(m2, tl.dot(tl.dot(Tm, tl.where(m2, N, 0.0), allow_tf32=True), Tm, allow_tf32=True), 0.0)
        dbV_c = tl.dot(tl.trans(Tm), rhs, allow_tf32=True)
        tl.store(
            tl.make_block_ptr(dbV + bv, (T, DV), (DV, 1), (c0, 0), (C, DV), (1, 0)),
            dbV_c.to(dbV.dtype.element_ty),
            boundary_check=(0, 1),
        )


def _dbV_solve(k, beta, dU, M, s, C):
    import math

    B, H, T, D = k.shape
    DV = dU.shape[-1]
    E = D // M
    NC = T // C
    dbV = torch.zeros(B, H, T, DV, device=k.device, dtype=torch.float32)
    _dbV_kernel[(B * H,)](
        k,
        beta,
        dU,
        dbV,
        s,
        T,
        H=H,
        D=D,
        E=E,
        M=M,
        DV=DV,
        C=C,
        LOGC=int(math.log2(C)),
        NC=NC,
    )
    return dbV


@triton.autotune(configs=_AUTO_CFGS, key=['T', 'C', 'M', 'DV'])
@triton.jit
def _dq_kernel(
    q,
    k,
    U,
    do,
    z,
    dq,
    s_scale,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    M: tl.constexpr,
    DV: tl.constexpr,
    C: tl.constexpr,
    NC: tl.constexpr,
):
    """dq_t from readout: dA_qk[t,s]=<do_t,U_s>-z_t (causal), then grad<q,k>^2. Program per (bh,query-block).
    z_t=<do_t,o_t>/den_t centers for the normalized backward (z=0 -> raw un-normalized)."""
    i_bh = tl.program_id(0).to(tl.int64)
    i_ct = tl.program_id(1)
    bqk = i_bh * T * D
    bv = i_bh * T * DV
    ct0 = i_ct * C
    o_row = tl.arange(0, C)
    b_do = tl.load(
        tl.make_block_ptr(do + bv, (T, DV), (DV, 1), (ct0, 0), (C, DV), (1, 0)),
        boundary_check=(0, 1),
    ).to(tl.bfloat16)
    b_z = tl.load(
        tl.make_block_ptr(z + i_bh * T, (T,), (1,), (ct0,), (C,), (0,)),
        boundary_check=(0,),
    ).to(tl.float32)
    for g in range(M):
        b_qg = (
            tl.load(
                tl.make_block_ptr(
                    q + bqk, (T, D), (D, 1), (ct0, g * E), (C, E), (1, 0)
                ),
                boundary_check=(0, 1),
            )
            * s_scale
        )
        dqg = tl.zeros([C, E], dtype=tl.float32)
        for i_cs in range(0, i_ct + 1):
            cs0 = i_cs * C
            b_U = tl.load(
                tl.make_block_ptr(U + bv, (T, DV), (DV, 1), (cs0, 0), (C, DV), (1, 0)),
                boundary_check=(0, 1),
            ).to(tl.bfloat16)
            dAqk = tl.dot(b_do, tl.trans(b_U)) - b_z[:, None]  # <do'_t,U_s> - z_t
            if i_cs == i_ct:
                dAqk = tl.where(o_row[:, None] >= o_row[None, :], dAqk, 0.0)
            for hh in range(M):
                b_kh = (
                    tl.load(
                        tl.make_block_ptr(
                            k + bqk, (T, D), (D, 1), (cs0, hh * E), (C, E), (1, 0)
                        ),
                        boundary_check=(0, 1),
                    )
                    * s_scale
                )
                ip = tl.dot(b_qg, tl.trans(b_kh))  # (C,C) = q_g . k_h
                dip = 2.0 * dAqk * ip
                dqg += tl.dot(dip.to(tl.bfloat16), b_kh.to(tl.bfloat16))
        tl.store(
            tl.make_block_ptr(dq + bqk, (T, D), (D, 1), (ct0, g * E), (C, E), (1, 0)),
            (dqg * s_scale).to(dq.dtype.element_ty),
            boundary_check=(0, 1),
        )


def _dq(q, k, U, do, z, M, s, C):
    B, H, T, D = q.shape
    DV = do.shape[-1]
    E = D // M
    NC = T // C
    dq = torch.zeros(B, H, T, D, device=q.device, dtype=torch.float32)
    _dq_kernel[(B * H, NC)](
        q,
        k,
        U,
        do,
        z,
        dq,
        s,
        T,
        H=H,
        D=D,
        E=E,
        M=M,
        DV=DV,
        C=C,
        NC=NC,
    )
    return dq


@triton.autotune(configs=_AUTO_CFGS, key=['T', 'C', 'M', 'DV'])
@triton.jit
def _dvdb_kernel(
    k,
    v,
    beta,
    U,
    dbV,
    dv,
    dbeta,
    s_scale,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    M: tl.constexpr,
    DV: tl.constexpr,
    C: tl.constexpr,
    NC: tl.constexpr,
):
    """dv = beta*dbV ; dbeta = rowsum(v*dbV) - rowsum(dbV*E), E_t = sum_{s<t} A_kk[t,s] U_s."""
    i_bh = tl.program_id(0).to(tl.int64)
    i_ct = tl.program_id(1)
    bqk = i_bh * T * D
    bv = i_bh * T * DV
    ct0 = i_ct * C
    o_row = tl.arange(0, C)
    b_dbV = tl.load(
        tl.make_block_ptr(dbV + bv, (T, DV), (DV, 1), (ct0, 0), (C, DV), (1, 0)),
        boundary_check=(0, 1),
    ).to(tl.float32)
    b_v = tl.load(
        tl.make_block_ptr(v + bv, (T, DV), (DV, 1), (ct0, 0), (C, DV), (1, 0)),
        boundary_check=(0, 1),
    ).to(tl.float32)
    b_beta = tl.load(
        tl.make_block_ptr(beta + i_bh * T, (T,), (1,), (ct0,), (C,), (0,)),
        boundary_check=(0,),
    ).to(tl.float32)
    tl.store(
        tl.make_block_ptr(dv + bv, (T, DV), (DV, 1), (ct0, 0), (C, DV), (1, 0)),
        (b_beta[:, None] * b_dbV).to(dv.dtype.element_ty),
        boundary_check=(0, 1),
    )
    Ee = tl.zeros([C, DV], dtype=tl.float32)
    for i_cs in range(0, i_ct + 1):
        cs0 = i_cs * C
        Akk = _gram_blk(k, k, bqk, ct0, cs0, s_scale, C, E, M, T, D)
        if i_cs == i_ct:
            Akk = tl.where(o_row[:, None] > o_row[None, :], Akk, 0.0)
        b_U = tl.load(
            tl.make_block_ptr(U + bv, (T, DV), (DV, 1), (cs0, 0), (C, DV), (1, 0)),
            boundary_check=(0, 1),
        ).to(tl.bfloat16)
        Ee += tl.dot(Akk.to(tl.bfloat16), b_U)
    dbeta_c = tl.sum(b_v * b_dbV, axis=1) - tl.sum(b_dbV * Ee, axis=1)
    tl.store(
        tl.make_block_ptr(dbeta + i_bh * T, (T,), (1,), (ct0,), (C,), (0,)),
        dbeta_c.to(dbeta.dtype.element_ty),
        boundary_check=(0,),
    )


@triton.autotune(configs=_AUTO_CFGS, key=['T', 'C', 'M', 'DV', 'BN'])
@triton.jit
def _dk_kernel(q, k, v, beta, U, dbV, do, z, dk, dv, dbeta, s_scale, T,
              H: tl.constexpr, D: tl.constexpr, E: tl.constexpr, M: tl.constexpr,
              DV: tl.constexpr, C: tl.constexpr, BN: tl.constexpr, FOLD: tl.constexpr):
    """dk over a BN-key tile (amortize the O(T^2) query reloads). Global-index causal masks.
    readout (t>=j, dA_qk) + erase-col (t>j, dA_kk[t,j]) + erase-row (s<j, dA_kk[j,s]); FOLD -> dv,dbeta."""
    i_bh = tl.program_id(0).to(tl.int64); i_n = tl.program_id(1)
    bqk = i_bh * T * D; bv = i_bh * T * DV; n0 = i_n * BN
    k_idx = n0 + tl.arange(0, BN)
    b_Uj = tl.load(tl.make_block_ptr(U + bv, (T, DV), (DV, 1), (n0, 0), (BN, DV), (1, 0)), boundary_check=(0, 1)).to(tl.bfloat16)
    b_dbVj = tl.load(tl.make_block_ptr(dbV + bv, (T, DV), (DV, 1), (n0, 0), (BN, DV), (1, 0)), boundary_check=(0, 1)).to(tl.bfloat16)
    b_betaj = tl.load(tl.make_block_ptr(beta + i_bh * T, (T,), (1,), (n0,), (BN,), (0,)), boundary_check=(0,)).to(tl.float32)
    Efold = tl.zeros([BN, DV], dtype=tl.float32)
    for g in range(M):
        kjg = tl.load(tl.make_block_ptr(k + bqk, (T, D), (D, 1), (n0, g * E), (BN, E), (1, 0)), boundary_check=(0, 1)) * s_scale
        dkg = tl.zeros([BN, E], dtype=tl.float32)
        for ct0 in range(n0, T, C):                                       # queries t >= tile keys
            t_idx = ct0 + tl.arange(0, C)
            do_t = tl.load(tl.make_block_ptr(do + bv, (T, DV), (DV, 1), (ct0, 0), (C, DV), (1, 0)), boundary_check=(0, 1)).to(tl.bfloat16)
            dbV_t = tl.load(tl.make_block_ptr(dbV + bv, (T, DV), (DV, 1), (ct0, 0), (C, DV), (1, 0)), boundary_check=(0, 1)).to(tl.bfloat16)
            beta_t = tl.load(tl.make_block_ptr(beta + i_bh * T, (T,), (1,), (ct0,), (C,), (0,)), boundary_check=(0,)).to(tl.float32)
            z_t = tl.load(tl.make_block_ptr(z + i_bh * T, (T,), (1,), (ct0,), (C,), (0,)), boundary_check=(0,)).to(tl.float32)
            dAqk = tl.dot(do_t, tl.trans(b_Uj)) - z_t[:, None]            # <do'_t,U_s> - z_t  (C, BN)
            dAkk = -beta_t[:, None] * tl.dot(dbV_t, tl.trans(b_Uj))
            dAqk = tl.where(t_idx[:, None] >= k_idx[None, :], dAqk, 0.0)
            dAkk = tl.where(t_idx[:, None] > k_idx[None, :], dAkk, 0.0)
            for i in range(M):
                qti = tl.load(tl.make_block_ptr(q + bqk, (T, D), (D, 1), (ct0, i * E), (C, E), (1, 0)), boundary_check=(0, 1)) * s_scale
                kti = tl.load(tl.make_block_ptr(k + bqk, (T, D), (D, 1), (ct0, i * E), (C, E), (1, 0)), boundary_check=(0, 1)) * s_scale
                dip_ro = 2.0 * dAqk * tl.dot(qti, tl.trans(kjg))          # (C, BN)
                dip_er = 2.0 * dAkk * tl.dot(kti, tl.trans(kjg))
                dkg += tl.dot(tl.trans(dip_ro).to(tl.bfloat16), qti.to(tl.bfloat16))   # (BN, E)
                dkg += tl.dot(tl.trans(dip_er).to(tl.bfloat16), kti.to(tl.bfloat16))
        for cs0 in range(0, n0 + BN, C):                                  # keys s < tile keys
            s_idx = cs0 + tl.arange(0, C)
            U_s = tl.load(tl.make_block_ptr(U + bv, (T, DV), (DV, 1), (cs0, 0), (C, DV), (1, 0)), boundary_check=(0, 1)).to(tl.bfloat16)
            dAkk_r = -b_betaj[:, None] * tl.dot(b_dbVj, tl.trans(U_s))     # (BN, C)
            dAkk_r = tl.where(k_idx[:, None] > s_idx[None, :], dAkk_r, 0.0)
            for i in range(M):
                ksi = tl.load(tl.make_block_ptr(k + bqk, (T, D), (D, 1), (cs0, i * E), (C, E), (1, 0)), boundary_check=(0, 1)) * s_scale
                ip = tl.dot(kjg, tl.trans(ksi))                           # (BN, C)
                dkg += tl.dot((2.0 * dAkk_r * ip).to(tl.bfloat16), ksi.to(tl.bfloat16))
                if FOLD:
                    akk = tl.where(k_idx[:, None] > s_idx[None, :], ip * ip, 0.0)
                    Efold += tl.dot(akk.to(tl.bfloat16), U_s)             # (BN, DV)
        tl.store(tl.make_block_ptr(dk + bqk, (T, D), (D, 1), (n0, g * E), (BN, E), (1, 0)),
                 (dkg * s_scale).to(dk.dtype.element_ty), boundary_check=(0, 1))
    if FOLD:
        b_vj = tl.load(tl.make_block_ptr(v + bv, (T, DV), (DV, 1), (n0, 0), (BN, DV), (1, 0)), boundary_check=(0, 1)).to(tl.float32)
        b_dbVj_f = tl.load(tl.make_block_ptr(dbV + bv, (T, DV), (DV, 1), (n0, 0), (BN, DV), (1, 0)), boundary_check=(0, 1)).to(tl.float32)
        tl.store(tl.make_block_ptr(dv + bv, (T, DV), (DV, 1), (n0, 0), (BN, DV), (1, 0)),
                 (b_betaj[:, None] * b_dbVj_f).to(dv.dtype.element_ty), boundary_check=(0, 1))
        dbeta_j = tl.sum(b_vj * b_dbVj_f, axis=1) - tl.sum(b_dbVj_f * Efold, axis=1)
        tl.store(tl.make_block_ptr(dbeta + i_bh * T, (T,), (1,), (n0,), (BN,), (0,)),
                 dbeta_j.to(dbeta.dtype.element_ty), boundary_check=(0,))


def flash_delta_bwd(q, k, v, beta, U, do, M, scale=None, C=64, z=None):
    import math

    B, H, T, D = q.shape
    DV = v.shape[-1]
    E = D // M
    NC = T // C
    s = (1.0 / E if scale is None else scale) ** 0.5
    q, k, v, beta, U, do = (x.contiguous() for x in (q, k, v, beta, U, do))
    # z_t = <do_t,o_t>/den_t centers dA_qk for the normalized backward; z=0 -> raw (do already scaled by 1/den)
    z = torch.zeros(B, H, T, device=q.device, dtype=torch.float32) if z is None else z.contiguous().float()
    dU = _dU(q, k, do, U, M, s, C)
    dbV = _dbV_solve(k, beta, dU, M, s, C)
    dq = _dq(q, k, U, do, z, M, s, C)
    dv = torch.empty(B, H, T, DV, device=q.device, dtype=torch.float32)
    dbeta = torch.empty(B, H, T, device=q.device, dtype=torch.float32)
    fold = M == 1                        # M=1: fold dv/dbeta into _dk (E_j is free there)
    if not fold:
        _dvdb_kernel[(B * H, NC)](
            k, v, beta, U, dbV, dv, dbeta, s, T,
            H=H, D=D, E=E, M=M, DV=DV, C=C, NC=NC,
        )
    dk = torch.zeros(B, H, T, D, device=q.device, dtype=torch.float32)
    BN = C          # key-tile == chunk for now (correctness focus)
    _dk_kernel[(B * H, triton.cdiv(T, BN))](
        q, k, v, beta, U, dbV, do, z, dk, dv, dbeta, s, T,
        H=H, D=D, E=E, M=M, DV=DV, C=C, BN=BN, FOLD=fold,
    )
    return dq, dk, dv, dbeta


@triton.autotune(configs=_AUTO_CFGS, key=['T', 'C', 'M', 'DV', 'BM'])
@triton.jit
def _readout_kernel(q, k, U, o, s_scale, T, H: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
                    M: tl.constexpr, DV: tl.constexpr, C: tl.constexpr, BM: tl.constexpr):
    """o_t = sum_{s<=t} A_qk[t,s] U_s. Query tile of BM rows -> each k_s/U_s load serves BM
    queries (fewer HBM reloads; the readout is bandwidth-bound at B>=32)."""
    i_bh = tl.program_id(0).to(tl.int64); i_m = tl.program_id(1)
    bqk = i_bh * T * D; bv = i_bh * T * DV; m0 = i_m * BM
    q_idx = m0 + tl.arange(0, BM)
    acc = tl.zeros([BM, DV], dtype=tl.float32)
    n_kb = tl.cdiv(m0 + BM, C)                       # key-blocks with any key <= a query in the tile
    for i_cs in range(0, n_kb):
        cs0 = i_cs * C
        k_idx = cs0 + tl.arange(0, C)
        Aqk = tl.zeros([BM, C], dtype=tl.float32)
        for g in range(M):
            qg = tl.load(tl.make_block_ptr(q + bqk, (T, D), (D, 1), (m0, g * E), (BM, E), (1, 0)),
                         boundary_check=(0, 1)) * s_scale
            for hh in range(M):
                kh = tl.load(tl.make_block_ptr(k + bqk, (T, D), (D, 1), (cs0, hh * E), (C, E), (1, 0)),
                             boundary_check=(0, 1)) * s_scale
                ip = tl.dot(qg, tl.trans(kh))
                Aqk += ip * ip
        Aqk = tl.where(q_idx[:, None] >= k_idx[None, :], Aqk, 0.0)   # causal (global indices)
        b_U = tl.load(tl.make_block_ptr(U + bv, (T, DV), (DV, 1), (cs0, 0), (C, DV), (1, 0)),
                      boundary_check=(0, 1)).to(tl.bfloat16)
        acc += tl.dot(Aqk.to(tl.bfloat16), b_U)
    tl.store(tl.make_block_ptr(o + bv, (T, DV), (DV, 1), (m0, 0), (BM, DV), (1, 0)),
             acc.to(o.dtype.element_ty), boundary_check=(0, 1))


def flash_delta_readout(q, k, U, M, scale=None, C=64, BM=128):
    B, H, T, D = q.shape; DV = U.shape[-1]; E = D // M
    s = (1.0 / E if scale is None else scale) ** 0.5
    o = torch.empty(B, H, T, DV, device=q.device, dtype=torch.float32)
    grid = (B * H, triton.cdiv(T, BM))
    _readout_kernel[grid](q, k, U, o, s, T, H=H, D=D, E=E, M=M, DV=DV, C=C, BM=BM)
    return o


@triton.autotune(configs=_AUTO_CFGS, key=['T', 'C', 'M', 'BM'])
@triton.jit
def _den_kernel(q, k, den, s_scale, T, H: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
                M: tl.constexpr, C: tl.constexpr, BM: tl.constexpr):
    """den_t = sum_{s<=t} A_qk[t,s]  (KATA normalization denominator)."""
    i_bh = tl.program_id(0).to(tl.int64); i_m = tl.program_id(1)
    bqk = i_bh * T * D; m0 = i_m * BM
    q_idx = m0 + tl.arange(0, BM)
    acc = tl.zeros([BM], dtype=tl.float32)
    n_kb = tl.cdiv(m0 + BM, C)
    for i_cs in range(0, n_kb):
        cs0 = i_cs * C
        k_idx = cs0 + tl.arange(0, C)
        Aqk = tl.zeros([BM, C], dtype=tl.float32)
        for g in range(M):
            qg = tl.load(tl.make_block_ptr(q + bqk, (T, D), (D, 1), (m0, g * E), (BM, E), (1, 0)), boundary_check=(0, 1)) * s_scale
            for hh in range(M):
                kh = tl.load(tl.make_block_ptr(k + bqk, (T, D), (D, 1), (cs0, hh * E), (C, E), (1, 0)), boundary_check=(0, 1)) * s_scale
                ip = tl.dot(qg, tl.trans(kh)); Aqk += ip * ip
        Aqk = tl.where(q_idx[:, None] >= k_idx[None, :], Aqk, 0.0)
        acc += tl.sum(Aqk, axis=1)
    tl.store(den + i_bh * T + m0 + tl.arange(0, BM), acc, mask=q_idx < T)


def flash_delta_den(q, k, M, scale=None, C=64, BM=128):
    B, H, T, D = q.shape; E = D // M
    s = (1.0 / E if scale is None else scale) ** 0.5
    den = torch.empty(B, H, T, device=q.device, dtype=torch.float32)
    _den_kernel[(B * H, triton.cdiv(T, BM))](q, k, den, s, T, H=H, D=D, E=E, M=M, C=C, BM=BM)
    return den


class _FlashDelta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, beta, M, scale, C, normalize, eps):
        U = flash_delta_U(q, k, v, beta, M, scale=scale, C=C)
        num = flash_delta_readout(q, k, U, M, scale=scale, C=C)   # numerator = tril(A_qk) U
        if normalize:
            den = flash_delta_den(q, k, M, scale=scale, C=C)
            o = num / (den[..., None] + eps)
            ctx.save_for_backward(q, k, v, beta, U, den, o)
        else:
            o = num
            ctx.save_for_backward(q, k, v, beta, U)
        ctx.M, ctx.scale, ctx.C, ctx.normalize, ctx.eps = M, scale, C, normalize, eps
        return o

    @staticmethod
    def backward(ctx, do):
        if ctx.normalize:
            q, k, v, beta, U, den, o = ctx.saved_tensors
            inv = 1.0 / (den[..., None] + ctx.eps)            # 1/den_t  (B,H,T,1)
            dop = do * inv                                    # do'_t = do_t / den_t
            z = (do * o).sum(-1) * inv[..., 0]                # z_t = <do_t,o_t>/den_t  (B,H,T)
            dq, dk, dv, db = flash_delta_bwd(q, k, v, beta, U, dop, ctx.M, ctx.scale, ctx.C, z=z)
        else:
            q, k, v, beta, U = ctx.saved_tensors
            dq, dk, dv, db = flash_delta_bwd(q, k, v, beta, U, do, ctx.M, ctx.scale, ctx.C)
        return (dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), db.to(beta.dtype),
                None, None, None, None, None)


def flash_delta(q, k, v, beta, M, scale=None, C=64, normalize=False, eps=1e-5):
    """Trainable full-SPD delta. normalize=True -> o = num/(den+eps), den_t = sum_{s<=t} A_qk[t,s],
    with a CONSISTENT backward: den propagates its gradient too (do' = do/den, dA_qk gets -z_t,
    z_t = <do_t,o_t>/den_t). normalize=False -> raw numerator. No psi / no E^2 state."""
    return _FlashDelta.apply(q, k, v, beta, M, scale, C, normalize, eps)
