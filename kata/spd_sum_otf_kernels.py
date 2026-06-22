"""ψ-on-the-fly chunked linear attention for SPD-sum-packed.

Forward-only. Reads raw Q, K, V (last dim = d_head, e.g. 64), expands

    psi(x) = sum_i x_g_i x_g_i^T    in R^{E×E},   E = d_head / M

inside SRAM per chunk; never materializes ψ(K) or ψ(Q) in HBM.
HBM contains:
- raw Q, K (B, T, H, d) bf16   ← inputs (no ψ)
- V (B, T, H, dv) bf16
- chunk-boundary state S_local, S_prefix (B, H, NC, QD=E², DV) — stored bf16
- chunk-boundary z_local, z_prefix (B, H, NC, QD) — stored bf16
- output O (B, T, H, DV) bf16

Math (same as `feature_maps.spd_sum`, no eps / no 1/M scaling):
    psi(x) = vec(sum_i x_g_i x_g_i^T)               # flat E² vector
    <psi(x), psi(y)> = sum_{i,j} (x_g_i · y_g_j)²    # SPD-sum identity
    S_t = sum_{s<=t} psi(k_s) v_s^T
    z_t = sum_{s<=t} psi(k_s)
    o_t = (psi(q_t)^T S_t) / (psi(q_t)^T z_t + eps)

Three sub-kernels, matching the existing tree-scan dispatch:
1. chunk_state_otf : per-chunk S_loc = psi(K_chunk)^T @ V_chunk, z_loc = sum_t psi(k_t)
2. chunk_scan_tree : Hillis–Steele tree-scan over chunks (parallel cumsum)
3. chunk_output_otf: o = inter (psi(Q) @ S_prefix) + intra (causal psi(Q) @ psi(K)^T @ V), normalized

Constraints:
- d_head = D = M * E  (D divisible by M)
- E ≥ 2; QD = E² is pow-2 when E is pow-2 (E=16, 8 etc.)
- T % C == 0, NC = T/C, NC must be pow-2 for tree-scan
- DV % DV_BLK == 0
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# =====================================================================
# Kernel 1: chunk_state (psi on-the-fly)
# =====================================================================


@triton.jit
def _otf_chunk_state_kernel(
    K_ptr,           # (B, T, H, D=64) bf16  raw keys
    V_ptr,           # (B, T, H, DV)  bf16
    S_ptr,           # (B, H, NC, QD, DV) state-dtype  out
    Z_ptr,           # (B, H, NC, QD) state-dtype  out
    H, T, NC,
    C: tl.constexpr,
    D: tl.constexpr,   # raw key dim
    E: tl.constexpr,   # per-group dim, D = M*E
    M: tl.constexpr,
    QD: tl.constexpr,  # = E*E
    DV: tl.constexpr,
    DV_BLK: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_nc = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_c = tl.arange(0, C)
    offs_e = tl.arange(0, E)
    offs_qd = tl.arange(0, QD)
    t_idx = pid_nc * C + offs_c

    # Accumulate combined[c, l*E + m] = sum_i K_g[c, i, l] * K_g[c, i, m]
    combined = tl.zeros((C, QD), dtype=tl.float32)
    for m_id in tl.static_range(M):
        k_off = (
            b * T * H * D
            + t_idx[:, None] * H * D
            + h * D
            + m_id * E
            + offs_e[None, :]
        )
        K_g = tl.load(K_ptr + k_off).to(tl.float32)  # (C, E)
        # outer product (C, E, E) -> flat (C, QD=E²)
        K_outer = K_g[:, :, None] * K_g[:, None, :]
        combined += tl.reshape(K_outer, (C, QD))

    # combined IS psi(K_chunk) — (C, QD) with QD = E² flat layout.
    psi_K = combined

    # Z_local = sum over chunk axis (C)
    Z_loc = tl.sum(psi_K, axis=0)  # (QD,)
    z_off = b * H * NC * QD + h * NC * QD + pid_nc * QD + offs_qd
    tl.store(Z_ptr + z_off, Z_loc.to(Z_ptr.dtype.element_ty))

    # S_local = psi_K^T @ V, tiled along DV.
    psi_K_T = tl.trans(psi_K)  # (QD, C)
    for dv_id in tl.static_range(DV // DV_BLK):
        offs_dv = dv_id * DV_BLK + tl.arange(0, DV_BLK)
        v_off = (
            b * T * H * DV
            + t_idx[:, None] * H * DV
            + h * DV
            + offs_dv[None, :]
        )
        V_tile = tl.load(V_ptr + v_off).to(tl.float32)  # (C, DV_BLK)
        S_tile = tl.dot(psi_K_T, V_tile, allow_tf32=False)  # (QD, DV_BLK)
        s_off = (
            b * H * NC * QD * DV
            + h * NC * QD * DV
            + pid_nc * QD * DV
            + offs_qd[:, None] * DV
            + offs_dv[None, :]
        )
        tl.store(S_ptr + s_off, S_tile.to(S_ptr.dtype.element_ty))


# =====================================================================
# Kernel 2: chunk_scan_tree (parallel prefix sum)
# =====================================================================


@triton.jit
def _otf_chunk_scan_tree_kernel(
    S_local_ptr,
    Z_local_ptr,
    S_prefix_ptr,
    Z_prefix_ptr,
    H,
    NC: tl.constexpr,
    QD: tl.constexpr,
    DV: tl.constexpr,
):
    """Grid: (B*H, QD). Exclusive prefix sum over NC via tl.cumsum."""
    pid_bh = tl.program_id(0)
    q_idx = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_nc = tl.arange(0, NC)
    offs_dv = tl.arange(0, DV)

    z_off = b * H * NC * QD + h * NC * QD + offs_nc * QD + q_idx
    Z = tl.load(Z_local_ptr + z_off).to(tl.float32)
    tl.store(
        Z_prefix_ptr + z_off,
        (tl.cumsum(Z, axis=0) - Z).to(Z_prefix_ptr.dtype.element_ty),
    )

    s_off = (
        b * H * NC * QD * DV
        + h * NC * QD * DV
        + offs_nc[:, None] * QD * DV
        + q_idx * DV
        + offs_dv[None, :]
    )
    S = tl.load(S_local_ptr + s_off).to(tl.float32)
    tl.store(
        S_prefix_ptr + s_off,
        (tl.cumsum(S, axis=0) - S).to(S_prefix_ptr.dtype.element_ty),
    )


# =====================================================================
# Kernel 3: chunk_output (psi on-the-fly for Q and K, full readout)
# =====================================================================


@triton.jit
def _otf_chunk_output_kernel(
    Q_ptr,            # (B, T, H, D)  raw bf16
    K_ptr,            # (B, T, H, D)  raw bf16
    V_ptr,            # (B, T, H, DV) bf16
    S_pref_ptr,       # (B, H, NC, QD, DV) state-dtype
    Z_pref_ptr,       # (B, H, NC, QD) state-dtype
    O_ptr,            # (B, T, H, DV) bf16
    D_out_ptr,        # (B, T, H) fp32  (the denominator value per token)
    H, T, NC,
    C: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    M: tl.constexpr,
    QD: tl.constexpr,
    DV: tl.constexpr,
    DV_BLK: tl.constexpr,
    EPS: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_nc = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_c = tl.arange(0, C)
    offs_e = tl.arange(0, E)
    offs_qd = tl.arange(0, QD)
    t_idx = pid_nc * C + offs_c

    # Build psi(Q_chunk), psi(K_chunk) on-the-fly: shape (C, QD=E²).
    psi_Q = tl.zeros((C, QD), dtype=tl.float32)
    psi_K = tl.zeros((C, QD), dtype=tl.float32)
    for m_id in tl.static_range(M):
        qk_off = (
            b * T * H * D
            + t_idx[:, None] * H * D
            + h * D
            + m_id * E
            + offs_e[None, :]
        )
        Q_g = tl.load(Q_ptr + qk_off).to(tl.float32)  # (C, E)
        K_g = tl.load(K_ptr + qk_off).to(tl.float32)  # (C, E)
        psi_Q += tl.reshape(Q_g[:, :, None] * Q_g[:, None, :], (C, QD))
        psi_K += tl.reshape(K_g[:, :, None] * K_g[:, None, :], (C, QD))

    # Z_inter[t] = <psi(q_t), Z_pre>
    z_off = b * H * NC * QD + h * NC * QD + pid_nc * QD + offs_qd
    Zp = tl.load(Z_pref_ptr + z_off).to(tl.float32)  # (QD,)
    Z_inter = tl.sum(psi_Q * Zp[None, :], axis=1)  # (C,)

    # Intra-chunk attention pattern A[t, s] = <psi(q_t), psi(k_s)> for s<=t.
    A = tl.dot(psi_Q, tl.trans(psi_K), allow_tf32=False)  # (C, C)
    causal = offs_c[:, None] >= offs_c[None, :]
    A_masked = tl.where(causal, A, 0.0)
    Z_intra = tl.sum(A_masked, axis=1)  # (C,)
    D_t = Z_inter + Z_intra + EPS  # (C,)

    d_off = b * T * H + t_idx * H + h
    tl.store(D_out_ptr + d_off, D_t)

    # Output, DV-tiled.
    for dv_id in tl.static_range(DV // DV_BLK):
        offs_dv = dv_id * DV_BLK + tl.arange(0, DV_BLK)
        v_off = (
            b * T * H * DV
            + t_idx[:, None] * H * DV
            + h * DV
            + offs_dv[None, :]
        )
        V_tile = tl.load(V_ptr + v_off).to(tl.float32)  # (C, DV_BLK)

        sp_off = (
            b * H * NC * QD * DV
            + h * NC * QD * DV
            + pid_nc * QD * DV
            + offs_qd[:, None] * DV
            + offs_dv[None, :]
        )
        Sp = tl.load(S_pref_ptr + sp_off).to(tl.float32)  # (QD, DV_BLK)

        O_inter = tl.dot(psi_Q, Sp, allow_tf32=False)  # (C, DV_BLK)
        O_intra = tl.dot(A_masked, V_tile, allow_tf32=False)  # (C, DV_BLK)
        O_tile = (O_inter + O_intra) / D_t[:, None]

        o_off = v_off
        tl.store(O_ptr + o_off, O_tile.to(O_ptr.dtype.element_ty))


# =====================================================================
# Python wrapper
# =====================================================================


def chunk_linattn_spd_sum_otf_fwd(
    q: torch.Tensor,        # (B, T, H, D)
    k: torch.Tensor,        # (B, T, H, D)
    v: torch.Tensor,        # (B, T, H, DV)
    num_groups: int = 4,
    chunk_size: int = 64,
    eps: float = 1e-6,
    state_dtype: torch.dtype = torch.float32,
):
    """Forward chunked linear attention with psi-on-the-fly SPD-sum.

    Args:
        q, k: (B, T, H, D=d_head). num_groups must divide D.
        v: (B, T, H, DV).
        num_groups: M (split D into M groups of E = D/M).
        chunk_size: C, must divide T; NC = T/C must be pow-2.
        state_dtype: dtype for HBM-stored chunk-boundary state (S, z).
          float32 = safe accumulator. bfloat16 = half the HBM bytes, may
          lose some precision over long T.

    Returns:
        o: (B, T, H, DV) same dtype as v.
    """
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    B, T, H, Dh = q.shape
    DV = v.shape[-1]
    assert k.shape == q.shape, (q.shape, k.shape)
    assert Dh % num_groups == 0, f"D={Dh} not divisible by M={num_groups}"
    M = num_groups
    E = Dh // M
    assert E >= 2, f"E={E} too small (need E>=2 for outer product)"
    if E & (E - 1):
        raise ValueError(f"E={E} must be pow-2 for triton tile (currently)")
    QD = E * E

    C = chunk_size
    if T % C != 0:
        raise ValueError(f"T={T} not divisible by C={C}")
    NC = T // C
    if NC & (NC - 1):
        raise ValueError(f"NC=T/C={NC} must be pow-2")

    DV_BLK = 16 if QD >= 256 else DV
    if DV % DV_BLK:
        raise ValueError(f"DV={DV} not divisible by DV_BLK={DV_BLK}")

    dev = q.device
    S_local = torch.empty(B, H, NC, QD, DV, device=dev, dtype=state_dtype)
    Z_local = torch.empty(B, H, NC, QD, device=dev, dtype=state_dtype)
    S_prefix = torch.empty_like(S_local)
    Z_prefix = torch.empty_like(Z_local)
    O = torch.empty_like(v)
    D_out = torch.empty(B, T, H, device=dev, dtype=torch.float32)

    _otf_chunk_state_kernel[(B * H, NC)](
        k, v, S_local, Z_local, H, T, NC,
        C=C, D=Dh, E=E, M=M, QD=QD, DV=DV, DV_BLK=DV_BLK,
        num_warps=4, num_stages=1,
    )
    _otf_chunk_scan_tree_kernel[(B * H, QD)](
        S_local, Z_local, S_prefix, Z_prefix, H,
        NC=NC, QD=QD, DV=DV,
        num_warps=4, num_stages=1,
    )
    _otf_chunk_output_kernel[(B * H, NC)](
        q, k, v, S_prefix, Z_prefix, O, D_out, H, T, NC,
        C=C, D=Dh, E=E, M=M, QD=QD, DV=DV, DV_BLK=DV_BLK,
        EPS=eps,
        num_warps=4, num_stages=1,
    )
    return O
