"""Chunked linear attention on PRE-MATERIALIZED psi(q), psi(k) tensors with
TREE-SCAN inter-chunk reduction (parallel `tl.cumsum`, log-depth Kogge-Stone).

This is a stripped-down version of the SPD-on-the-fly kernel. The feature map
is applied in pytorch *before* the kernel is called, so q, k arrive already
in their projected (B, T, H, q_dim) form. The kernel only does:

  1. chunk_state: S_local = K_chunk^T @ V_chunk  (per chunk, parallel)
  2. chunk_scan : exclusive prefix sum over NC via tl.cumsum   (TREE scan)
  3. chunk_output: o = (Q @ S_prefix + intra_causal_attention) / (Q . Z_prefix + intra_Z)

Drop-in for fla.ops.linear_attn.chunk_linear_attn(normalize=True). Used as
the substrate for the KATA SPD-packed feature map AND for positive/lorentz
feature maps (the same code path applies; only psi differs).

Two scan modes are exposed:
  - 'tree'   (default): tl.cumsum, parallel reduction inside program (O(log NC))
  - 'linear':           explicit `for c in tl.range(NC)` accumulation (O(NC))
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _ts_chunk_state_kernel(
    K_ptr,
    V_ptr,
    S_ptr,
    Z_ptr,
    H,
    T,
    NC,
    C: tl.constexpr,
    QD: tl.constexpr,
    DV: tl.constexpr,
    DV_BLK: tl.constexpr,
):
    """S_local[b,h,nc,:,:] = K_chunk^T @ V_chunk   ;  Z_local[..] = sum K
    DV-tiled: K is loaded once, then V_tile of (C, DV_BLK) at a time;
    S_local is written per DV tile. Grid: (B*H, NC)."""
    pid_bh = tl.program_id(0)
    pid_nc = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_c = tl.arange(0, C)
    offs_q = tl.arange(0, QD)
    t_idx = pid_nc * C + offs_c

    k_off = b * T * H * QD + t_idx[:, None] * H * QD + h * QD + offs_q[None, :]
    K = tl.load(K_ptr + k_off).to(tl.float32)  # (C, QD)
    Z = tl.sum(K, axis=0)  # (QD,)
    z_off = b * H * NC * QD + h * NC * QD + pid_nc * QD + offs_q
    tl.store(Z_ptr + z_off, Z)

    K_T = tl.trans(K)  # (QD, C)

    for dv_id in tl.static_range(DV // DV_BLK):
        offs_dv = dv_id * DV_BLK + tl.arange(0, DV_BLK)
        v_off = (
            b * T * H * DV + t_idx[:, None] * H * DV + h * DV + offs_dv[None, :]
        )
        V_tile = tl.load(V_ptr + v_off).to(tl.float32)  # (C, DV_BLK)
        S_tile = tl.dot(K_T, V_tile, allow_tf32=False)  # (QD, DV_BLK)
        s_off = (
            b * H * NC * QD * DV
            + h * NC * QD * DV
            + pid_nc * QD * DV
            + offs_q[:, None] * DV
            + offs_dv[None, :]
        )
        tl.store(S_ptr + s_off, S_tile)


@triton.jit
def _ts_chunk_scan_tree_kernel(
    S_local_ptr,
    Z_local_ptr,
    S_prefix_ptr,
    Z_prefix_ptr,
    H,
    NC: tl.constexpr,
    QD: tl.constexpr,
    DV: tl.constexpr,
):
    """Exclusive prefix sum over NC via tl.cumsum (parallel tree reduction).
    Grid: (B*H, QD)."""
    pid_bh = tl.program_id(0)
    q_idx = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_nc = tl.arange(0, NC)
    offs_dv = tl.arange(0, DV)

    z_off = b * H * NC * QD + h * NC * QD + offs_nc * QD + q_idx
    Z = tl.load(Z_local_ptr + z_off)
    tl.store(Z_prefix_ptr + z_off, tl.cumsum(Z, axis=0) - Z)

    s_off = (
        b * H * NC * QD * DV
        + h * NC * QD * DV
        + offs_nc[:, None] * QD * DV
        + q_idx * DV
        + offs_dv[None, :]
    )
    S = tl.load(S_local_ptr + s_off)
    tl.store(S_prefix_ptr + s_off, tl.cumsum(S, axis=0) - S)


@triton.jit
def _ts_chunk_scan_linear_kernel(
    S_local_ptr,
    Z_local_ptr,
    S_prefix_ptr,
    Z_prefix_ptr,
    H,
    NC: tl.constexpr,
    QD: tl.constexpr,
    DV: tl.constexpr,
):
    """Sequential left-to-right accumulation. Grid: (B*H, QD)."""
    pid_bh = tl.program_id(0)
    q_idx = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_dv = tl.arange(0, DV)
    z_acc = tl.zeros([], dtype=tl.float32)
    s_acc = tl.zeros((DV,), dtype=tl.float32)

    for c in tl.range(0, NC):
        z_off = b * H * NC * QD + h * NC * QD + c * QD + q_idx
        s_off = (
            b * H * NC * QD * DV
            + h * NC * QD * DV
            + c * QD * DV
            + q_idx * DV
            + offs_dv
        )
        tl.store(Z_prefix_ptr + z_off, z_acc)
        tl.store(S_prefix_ptr + s_off, s_acc)
        z_acc = z_acc + tl.load(Z_local_ptr + z_off)
        s_acc = s_acc + tl.load(S_local_ptr + s_off)


@triton.jit
def _ts_chunk_output_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    S_pref_ptr,
    Z_pref_ptr,
    O_ptr,
    D_ptr,
    H,
    T,
    NC,
    C: tl.constexpr,
    QD: tl.constexpr,
    DV: tl.constexpr,
    DV_BLK: tl.constexpr,
    EPS: tl.constexpr,
):
    """Output for one chunk = inter (Q @ S_prefix) + intra (Q @ K^T causal) @ V.
    DV-tiled: A_total / Z_inter / D computed once outside DV loop; per DV tile
    we load V_tile, S_prefix_tile and write one (C, DV_BLK) slice of O.
    Grid: (B*H, NC)."""
    pid_bh = tl.program_id(0)
    pid_nc = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_c = tl.arange(0, C)
    offs_q = tl.arange(0, QD)
    t_idx = pid_nc * C + offs_c

    qk_off = b * T * H * QD + t_idx[:, None] * H * QD + h * QD + offs_q[None, :]
    Q = tl.load(Q_ptr + qk_off).to(tl.float32)  # (C, QD)
    K = tl.load(K_ptr + qk_off).to(tl.float32)  # (C, QD)

    z_off = b * H * NC * QD + h * NC * QD + pid_nc * QD + offs_q
    Zp = tl.load(Z_pref_ptr + z_off).to(tl.float32)  # (QD,)

    # === DV-independent ====
    Z_inter = tl.sum(Q * Zp[None, :], axis=1)  # (C,)
    A = tl.dot(Q, tl.trans(K), allow_tf32=False)  # (C, C)
    causal = offs_c[:, None] >= offs_c[None, :]
    A_masked = tl.where(causal, A, 0.0)
    Z_intra = tl.sum(A_masked, axis=1)  # (C,)
    D = Z_inter + Z_intra + EPS  # (C,)

    # Save D once
    d_off = b * T * H + t_idx * H + h
    tl.store(D_ptr + d_off, D)

    for dv_id in tl.static_range(DV // DV_BLK):
        offs_dv = dv_id * DV_BLK + tl.arange(0, DV_BLK)
        v_off = (
            b * T * H * DV + t_idx[:, None] * H * DV + h * DV + offs_dv[None, :]
        )
        V_tile = tl.load(V_ptr + v_off).to(tl.float32)  # (C, DV_BLK)

        sp_off = (
            b * H * NC * QD * DV
            + h * NC * QD * DV
            + pid_nc * QD * DV
            + offs_q[:, None] * DV
            + offs_dv[None, :]
        )
        Sp_tile = tl.load(S_pref_ptr + sp_off).to(tl.float32)  # (QD, DV_BLK)

        O_inter_tile = tl.dot(Q, Sp_tile, allow_tf32=False)
        O_intra_tile = tl.dot(A_masked, V_tile, allow_tf32=False)
        O_tile = (O_inter_tile + O_intra_tile) / D[:, None]

        o_off = v_off
        tl.store(O_ptr + o_off, O_tile.to(O_ptr.dtype.element_ty))


# =====================================================================
# BACKWARD KERNELS
# Math (linear attention with normalization, no SPD identity needed since
# psi is already materialized in q, k):
#     A[t, m]  = q[t] · k[m]                              (scalar per (t,m))
#     N[t, d]  = sum_{m≤t} A[t,m] · v[m, d]
#     D[t]     = sum_{m≤t} A[t,m] + eps
#     o[t, d]  = N / D
#     dN[t, d] = do[t, d] / D[t]
#     dD[t]    = -<do[t], o[t]> / D[t]
#     dq[t, l] = sum_{m≤t} (<dN[t], v[m]> + dD[t]) · k[m, l]
#     dk[m, l] = sum_{t≥m} (<dN[t], v[m]> + dD[t]) · q[t, l]
#     dv[m, d] = sum_{t≥m} A[t, m] · dN[t, d]
# =====================================================================


@triton.jit
def _ts_bwd_chunk_state_kernel(
    Q_ptr,
    dN_ptr,
    dD_ptr,
    RS_ptr,
    RZ_ptr,
    H,
    T,
    NC,
    C: tl.constexpr,
    QD: tl.constexpr,
    DV: tl.constexpr,
    DV_BLK: tl.constexpr,
):
    """Per-chunk reverse-state local statistics:
        RS_local[c, l, d] = sum_{t in chunk c} q[t, l] · dN[t, d]
        RZ_local[c, l]    = sum_{t in chunk c} q[t, l] · dD[t]
    Mirror of fwd state with (q, dN, dD) instead of (k, v, 1).
    Grid: (B*H, NC).
    """
    pid_bh = tl.program_id(0)
    pid_nc = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_c = tl.arange(0, C)
    offs_q = tl.arange(0, QD)
    t_idx = pid_nc * C + offs_c

    q_off = b * T * H * QD + t_idx[:, None] * H * QD + h * QD + offs_q[None, :]
    Q = tl.load(Q_ptr + q_off).to(tl.float32)  # (C, QD)

    dd_off = b * T * H + t_idx * H + h
    dD = tl.load(dD_ptr + dd_off).to(tl.float32)  # (C,)

    RZ = tl.sum(Q * dD[:, None], axis=0)  # (QD,)
    rz_off = b * H * NC * QD + h * NC * QD + pid_nc * QD + offs_q
    tl.store(RZ_ptr + rz_off, RZ)

    Q_T = tl.trans(Q)  # (QD, C)
    for dv_id in tl.static_range(DV // DV_BLK):
        offs_dv = dv_id * DV_BLK + tl.arange(0, DV_BLK)
        dn_off = (
            b * T * H * DV + t_idx[:, None] * H * DV + h * DV + offs_dv[None, :]
        )
        dN_tile = tl.load(dN_ptr + dn_off).to(tl.float32)  # (C, DV_BLK)
        RS_tile = tl.dot(Q_T, dN_tile, allow_tf32=False)  # (QD, DV_BLK)
        rs_off = (
            b * H * NC * QD * DV
            + h * NC * QD * DV
            + pid_nc * QD * DV
            + offs_q[:, None] * DV
            + offs_dv[None, :]
        )
        tl.store(RS_ptr + rs_off, RS_tile)


@triton.jit
def _ts_bwd_chunk_scan_kernel(
    RS_local_ptr,
    RZ_local_ptr,
    RS_suffix_ptr,
    RZ_suffix_ptr,
    H,
    NC: tl.constexpr,
    QD: tl.constexpr,
    DV: tl.constexpr,
):
    """Reverse exclusive cumsum (suffix sum) over NC via tl.cumsum on
    flipped index. Grid: (B*H, QD)."""
    pid_bh = tl.program_id(0)
    q_idx = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_nc = tl.arange(0, NC)
    rev_nc = NC - 1 - offs_nc
    offs_dv = tl.arange(0, DV)

    z_off = b * H * NC * QD + h * NC * QD + rev_nc * QD + q_idx
    Z = tl.load(RZ_local_ptr + z_off)
    tl.store(RZ_suffix_ptr + z_off, tl.cumsum(Z, axis=0) - Z)

    s_off = (
        b * H * NC * QD * DV
        + h * NC * QD * DV
        + rev_nc[:, None] * QD * DV
        + q_idx * DV
        + offs_dv[None, :]
    )
    S = tl.load(RS_local_ptr + s_off)
    tl.store(RS_suffix_ptr + s_off, tl.cumsum(S, axis=0) - S)


@triton.jit
def _ts_bwd_chunk_output_dq_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    dN_ptr,
    dD_ptr,
    S_pref_ptr,
    Z_pref_ptr,
    dQ_ptr,
    H,
    T,
    NC,
    C: tl.constexpr,
    QD: tl.constexpr,
    DV: tl.constexpr,
    DV_BLK: tl.constexpr,
):
    """dq for one chunk = inter (uses fwd S_prefix, Z_prefix) + intra causal.

        dq_inter[t, l] = sum_d dN[t, d] · S_prefix[l, d]
                       + dD[t] · Z_prefix[l]
        dq_intra[t, l] = sum_{m≤t in chunk} (<dN[t],v[m]> + dD[t]) · k[m, l]
                       = (beta_causal @ k_chunk)[t, l]
    Grid: (B*H, NC).
    """
    pid_bh = tl.program_id(0)
    pid_nc = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_c = tl.arange(0, C)
    offs_q = tl.arange(0, QD)
    t_idx = pid_nc * C + offs_c
    causal = offs_c[:, None] >= offs_c[None, :]

    qk_off = b * T * H * QD + t_idx[:, None] * H * QD + h * QD + offs_q[None, :]
    K = tl.load(K_ptr + qk_off).to(tl.float32)  # (C, QD)

    dd_off = b * T * H + t_idx * H + h
    dD = tl.load(dD_ptr + dd_off).to(tl.float32)  # (C,)

    z_off = b * H * NC * QD + h * NC * QD + pid_nc * QD + offs_q
    Zp = tl.load(Z_pref_ptr + z_off).to(tl.float32)  # (QD,)

    # === Compute dN_v (C,C) DV-tiled and accumulate inter contribution ===
    dN_v = tl.zeros((C, C), dtype=tl.float32)
    dq_inter_N = tl.zeros((C, QD), dtype=tl.float32)

    for dv_id in tl.static_range(DV // DV_BLK):
        offs_dv = dv_id * DV_BLK + tl.arange(0, DV_BLK)
        v_off = (
            b * T * H * DV + t_idx[:, None] * H * DV + h * DV + offs_dv[None, :]
        )
        V_tile = tl.load(V_ptr + v_off).to(tl.float32)
        dN_tile = tl.load(dN_ptr + v_off).to(tl.float32)
        dN_v += tl.dot(dN_tile, tl.trans(V_tile), allow_tf32=False)

        sp_off = (
            b * H * NC * QD * DV
            + h * NC * QD * DV
            + pid_nc * QD * DV
            + offs_q[:, None] * DV
            + offs_dv[None, :]
        )
        Sp_tile = tl.load(S_pref_ptr + sp_off).to(tl.float32)  # (QD, DV_BLK)
        dq_inter_N += tl.dot(dN_tile, tl.trans(Sp_tile), allow_tf32=False)

    # Inter dq from D
    dq_inter_D = dD[:, None] * Zp[None, :]  # (C, QD)

    # Intra
    beta = dN_v + dD[:, None]  # (C, C)
    beta_c = tl.where(causal, beta, 0.0)
    dq_intra = tl.dot(beta_c, K, allow_tf32=False)  # (C, QD)

    dq = dq_inter_N + dq_inter_D + dq_intra

    dq_off = qk_off
    tl.store(dQ_ptr + dq_off, dq.to(dQ_ptr.dtype.element_ty))


@triton.jit
def _ts_bwd_chunk_output_dkdv_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    dN_ptr,
    dD_ptr,
    RS_suff_ptr,
    RZ_suff_ptr,
    dK_ptr,
    dV_ptr,
    H,
    T,
    NC,
    C: tl.constexpr,
    QD: tl.constexpr,
    DV: tl.constexpr,
    DV_BLK: tl.constexpr,
):
    """dk and dv for one chunk = inter (uses RS_suffix, RZ_suffix) + intra anti-causal.

        dk_inter[m, l] = sum_d v[m, d] · RS_suffix[l, d]  +  RZ_suffix[l]
        dv_inter[m, d] = sum_l k[m, l] · RS_suffix[l, d]   = (k_chunk @ RS_suffix^T)[m, d]
            but stored with RS shape (QD, DV); dv_inter = (k_chunk · RS_suffix^T) is awkward.
            Equivalent and cleaner: RS_suffix already has axes (QD, DV);
            sum_l k[m,l] RS[l,d] = (k_chunk @ RS_suffix)[m, d].

        dk_intra[m, l] = (beta_causal^T @ q_chunk)[m, l]
        dv_intra[m, d] = (A_causal^T @ dN)[m, d]   where A = q @ k^T causal
    Grid: (B*H, NC).
    """
    pid_bh = tl.program_id(0)
    pid_nc = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_c = tl.arange(0, C)
    offs_q = tl.arange(0, QD)
    t_idx = pid_nc * C + offs_c
    causal = offs_c[:, None] >= offs_c[None, :]

    qk_off = b * T * H * QD + t_idx[:, None] * H * QD + h * QD + offs_q[None, :]
    Q = tl.load(Q_ptr + qk_off).to(tl.float32)  # (C, QD)
    K = tl.load(K_ptr + qk_off).to(tl.float32)  # (C, QD)

    dd_off = b * T * H + t_idx * H + h
    dD = tl.load(dD_ptr + dd_off).to(tl.float32)  # (C,)

    rz_off = b * H * NC * QD + h * NC * QD + pid_nc * QD + offs_q
    RZ = tl.load(RZ_suff_ptr + rz_off).to(tl.float32)  # (QD,)

    # === DV-independent intra: A = Q @ K^T, A_causal ===
    A = tl.dot(Q, tl.trans(K), allow_tf32=False)  # (C, C)
    A_causal = tl.where(causal, A, 0.0)

    # === dN_v (DV-tiled) + dk_inter accumulation ===
    dN_v = tl.zeros((C, C), dtype=tl.float32)
    # dk_inter = RZ[None, :]  # broadcast (C, QD), m-independent term
    # dk_inter accumulator has shape (C, QD)
    dk_inter_acc = tl.zeros((C, QD), dtype=tl.float32) + RZ[None, :]

    for dv_id in tl.static_range(DV // DV_BLK):
        offs_dv = dv_id * DV_BLK + tl.arange(0, DV_BLK)
        v_off = (
            b * T * H * DV + t_idx[:, None] * H * DV + h * DV + offs_dv[None, :]
        )
        V_tile = tl.load(V_ptr + v_off).to(tl.float32)
        dN_tile = tl.load(dN_ptr + v_off).to(tl.float32)
        dN_v += tl.dot(dN_tile, tl.trans(V_tile), allow_tf32=False)

        rs_off = (
            b * H * NC * QD * DV
            + h * NC * QD * DV
            + pid_nc * QD * DV
            + offs_q[:, None] * DV
            + offs_dv[None, :]
        )
        RS_tile = tl.load(RS_suff_ptr + rs_off).to(tl.float32)  # (QD, DV_BLK)

        # dk_inter_N[m, l] += sum_d v[m, d] · RS[l, d] over the DV_BLK slice
        dk_inter_acc += tl.dot(V_tile, tl.trans(RS_tile), allow_tf32=False)

        # dv intra+inter for this DV slice
        dv_inter_tile = tl.dot(K, RS_tile, allow_tf32=False)  # (C, DV_BLK)
        dv_intra_tile = tl.dot(tl.trans(A_causal), dN_tile, allow_tf32=False)
        dv_tile = dv_inter_tile + dv_intra_tile
        tl.store(dV_ptr + v_off, dv_tile.to(dV_ptr.dtype.element_ty))

    # Intra dk
    beta = dN_v + dD[:, None]
    beta_c = tl.where(causal, beta, 0.0)
    dk_intra = tl.dot(tl.trans(beta_c), Q, allow_tf32=False)  # (C, QD)

    dk = dk_inter_acc + dk_intra
    dk_off = qk_off
    tl.store(dK_ptr + dk_off, dk.to(dK_ptr.dtype.element_ty))


def chunk_linattn_treescan_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    o: torch.Tensor,
    D: torch.Tensor,
    S_prefix: torch.Tensor,
    Z_prefix: torch.Tensor,
    chunk_size: int = 32,
    dv_block: int | None = None,
):
    B, T, H, QD = q.shape
    DV = v.shape[-1]
    C = chunk_size
    NC = T // C
    if dv_block is None:
        dv_block = 16 if QD >= 256 else DV

    Df = D.float().clamp_min(1e-30)
    dN = (do.float() / Df.unsqueeze(-1)).contiguous()
    dD = (-(do.float() * o.float()).sum(dim=-1) / Df).contiguous()

    dev = q.device
    RS_local = torch.empty(B, H, NC, QD, DV, device=dev, dtype=torch.float32)
    RZ_local = torch.empty(B, H, NC, QD, device=dev, dtype=torch.float32)
    RS_suffix = torch.empty_like(RS_local)
    RZ_suffix = torch.empty_like(RZ_local)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    _ts_bwd_chunk_state_kernel[(B * H, NC)](
        q,
        dN,
        dD,
        RS_local,
        RZ_local,
        H,
        T,
        NC,
        C=C,
        QD=QD,
        DV=DV,
        DV_BLK=dv_block,
        num_warps=4,
        num_stages=1,
    )
    _ts_bwd_chunk_scan_kernel[(B * H, QD)](
        RS_local,
        RZ_local,
        RS_suffix,
        RZ_suffix,
        H,
        NC=NC,
        QD=QD,
        DV=DV,
        num_warps=4,
        num_stages=1,
    )
    _ts_bwd_chunk_output_dq_kernel[(B * H, NC)](
        q,
        k,
        v,
        dN,
        dD,
        S_prefix,
        Z_prefix,
        dq,
        H,
        T,
        NC,
        C=C,
        QD=QD,
        DV=DV,
        DV_BLK=dv_block,
        num_warps=4,
        num_stages=1,
    )
    _ts_bwd_chunk_output_dkdv_kernel[(B * H, NC)](
        q,
        k,
        v,
        dN,
        dD,
        RS_suffix,
        RZ_suffix,
        dk,
        dv,
        H,
        T,
        NC,
        C=C,
        QD=QD,
        DV=DV,
        DV_BLK=dv_block,
        num_warps=4,
        num_stages=1,
    )
    return dq, dk, dv


class _TreescanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, chunk_size, dv_block, eps):
        from kata.treescan_kernels import chunk_linattn_treescan_fwd_full

        O, D, S_prefix, Z_prefix = chunk_linattn_treescan_fwd_full(
            q,
            k,
            v,
            chunk_size=chunk_size,
            eps=eps,
            dv_block=dv_block,
        )
        ctx.save_for_backward(q, k, v, O, D, S_prefix, Z_prefix)
        ctx.chunk_size = chunk_size
        ctx.dv_block = dv_block
        return O

    @staticmethod
    def backward(ctx, do):
        q, k, v, O, D, S_prefix, Z_prefix = ctx.saved_tensors
        dq, dk, dv = chunk_linattn_treescan_bwd(
            q,
            k,
            v,
            do,
            O,
            D,
            S_prefix,
            Z_prefix,
            chunk_size=ctx.chunk_size,
            dv_block=ctx.dv_block,
        )
        return dq, dk, dv, None, None, None


def chunk_linattn_treescan_fwd_full(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int = 32,
    eps: float = 1e-6,
    dv_block: int | None = None,
):
    """Forward variant that ALSO returns (D, S_prefix, Z_prefix) for backward.
    Same compute as `chunk_linattn_treescan` (tree-scan only)."""
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    B, T, H, QD = q.shape
    DV = v.shape[-1]
    C = chunk_size
    NC = T // C
    if T % C or NC & (NC - 1):
        raise ValueError(f"T={T} / C={C}: NC must be a power of 2")

    if dv_block is None:
        dv_block = 16 if QD >= 256 else DV
    if DV % dv_block:
        raise ValueError(f"DV={DV} not divisible by dv_block={dv_block}")

    dev = q.device
    S_local = torch.empty(B, H, NC, QD, DV, device=dev, dtype=torch.float32)
    Z_local = torch.empty(B, H, NC, QD, device=dev, dtype=torch.float32)
    S_prefix = torch.empty_like(S_local)
    Z_prefix = torch.empty_like(Z_local)
    O = torch.empty_like(v)
    D = torch.empty(B, T, H, device=dev, dtype=torch.float32)

    _ts_chunk_state_kernel[(B * H, NC)](
        k,
        v,
        S_local,
        Z_local,
        H,
        T,
        NC,
        C=C,
        QD=QD,
        DV=DV,
        DV_BLK=dv_block,
        num_warps=4,
        num_stages=1,
    )
    _ts_chunk_scan_tree_kernel[(B * H, QD)](
        S_local,
        Z_local,
        S_prefix,
        Z_prefix,
        H,
        NC=NC,
        QD=QD,
        DV=DV,
        num_warps=4,
        num_stages=1,
    )
    _ts_chunk_output_kernel[(B * H, NC)](
        q,
        k,
        v,
        S_prefix,
        Z_prefix,
        O,
        D,
        H,
        T,
        NC,
        C=C,
        QD=QD,
        DV=DV,
        DV_BLK=dv_block,
        EPS=eps,
        num_warps=4,
        num_stages=1,
    )
    return O, D, S_prefix, Z_prefix


def chunk_linattn_treescan_autograd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int = 32,
    eps: float = 1e-6,
    dv_block: int | None = None,
):
    """Autograd-wrapped version of chunk_linattn_treescan with bespoke triton
    backward. For training. Tree-scan only (linear-scan is bench-only)."""
    return _TreescanFunction.apply(q, k, v, chunk_size, dv_block, eps)


def chunk_linattn_treescan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int = 32,
    eps: float = 1e-6,
    scan_mode: str = "tree",
    dv_block: int | None = None,
):
    """Chunked normalized linear attention on pre-materialized q, k, v.

    Args:
        q: (B, T, H, QD)  pre-feature-mapped queries
        k: (B, T, H, QD)  pre-feature-mapped keys (psi(k_hat))
        v: (B, T, H, DV)  values
        scan_mode: 'tree' (parallel cumsum) or 'linear' (sequential)
    Returns:
        o: (B, T, H, DV)
    """
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    B, T, H, QD = q.shape
    DV = v.shape[-1]
    C = chunk_size
    if T % C != 0:
        raise ValueError(f"T={T} not divisible by C={C}")
    NC = T // C
    if NC & (NC - 1):
        raise ValueError(f"NC=T/C={NC} must be a power of 2")

    # DV tile size: defaults to a sensible per-shape value. Smaller tiles
    # reduce SRAM use (needed when QD is large, e.g., SPD-packed feature
    # maps with QD ≥ 1024).
    if dv_block is None:
        dv_block = 16 if QD >= 256 else DV
    if DV % dv_block:
        raise ValueError(f"DV={DV} not divisible by dv_block={dv_block}")

    dev = q.device
    S_local = torch.empty(B, H, NC, QD, DV, device=dev, dtype=torch.float32)
    Z_local = torch.empty(B, H, NC, QD, device=dev, dtype=torch.float32)
    S_prefix = torch.empty_like(S_local)
    Z_prefix = torch.empty_like(Z_local)
    O = torch.empty_like(v)
    D = torch.empty(B, T, H, device=dev, dtype=torch.float32)

    _ts_chunk_state_kernel[(B * H, NC)](
        k,
        v,
        S_local,
        Z_local,
        H,
        T,
        NC,
        C=C,
        QD=QD,
        DV=DV,
        DV_BLK=dv_block,
        num_warps=4,
        num_stages=1,
    )
    scan_kernel = (
        _ts_chunk_scan_tree_kernel
        if scan_mode == "tree"
        else _ts_chunk_scan_linear_kernel
    )
    scan_kernel[(B * H, QD)](
        S_local,
        Z_local,
        S_prefix,
        Z_prefix,
        H,
        NC=NC,
        QD=QD,
        DV=DV,
        num_warps=4,
        num_stages=1,
    )
    _ts_chunk_output_kernel[(B * H, NC)](
        q,
        k,
        v,
        S_prefix,
        Z_prefix,
        O,
        D,
        H,
        T,
        NC,
        C=C,
        QD=QD,
        DV=DV,
        DV_BLK=dv_block,
        EPS=eps,
        num_warps=4,
        num_stages=1,
    )
    return O


# =====================================================================
# PACKED-QD VARIANT (forward-only): the feature map's output dim QD_REAL
# need not be a power of two. We tile QD as the next power of two
# QD_BLK = next_pow2(QD_REAL) and mask off lanes [QD_REAL, QD_BLK) on every
# QD-axis load/store. HBM strides use QD_REAL, so HBM size of K, S_local,
# S_prefix shrinks by QD_BLK/QD_REAL vs zero-padding.
# Used for spd_sum_packed (qd_real = E(E+1)/2) and spd_concat.
# =====================================================================


@triton.jit
def _ts_chunk_state_kernel_packed(
    K_ptr,
    V_ptr,
    S_ptr,
    Z_ptr,
    H,
    T,
    NC,
    C: tl.constexpr,
    QD: tl.constexpr,        # actual qd in HBM (stride)
    QD_BLK: tl.constexpr,    # tile size, pow2, >= QD
    DV: tl.constexpr,
    DV_BLK: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_nc = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_c = tl.arange(0, C)
    offs_q = tl.arange(0, QD_BLK)
    q_mask = offs_q < QD
    t_idx = pid_nc * C + offs_c

    k_off = b * T * H * QD + t_idx[:, None] * H * QD + h * QD + offs_q[None, :]
    K = tl.load(K_ptr + k_off, mask=q_mask[None, :], other=0.0).to(tl.float32)
    Z = tl.sum(K, axis=0)
    z_off = b * H * NC * QD + h * NC * QD + pid_nc * QD + offs_q
    tl.store(Z_ptr + z_off, Z, mask=q_mask)

    K_T = tl.trans(K)  # (QD_BLK, C)

    for dv_id in tl.static_range(DV // DV_BLK):
        offs_dv = dv_id * DV_BLK + tl.arange(0, DV_BLK)
        v_off = (
            b * T * H * DV + t_idx[:, None] * H * DV + h * DV + offs_dv[None, :]
        )
        V_tile = tl.load(V_ptr + v_off).to(tl.float32)  # (C, DV_BLK)
        S_tile = tl.dot(K_T, V_tile, allow_tf32=False)  # (QD_BLK, DV_BLK)
        s_off = (
            b * H * NC * QD * DV
            + h * NC * QD * DV
            + pid_nc * QD * DV
            + offs_q[:, None] * DV
            + offs_dv[None, :]
        )
        tl.store(S_ptr + s_off, S_tile, mask=q_mask[:, None])


@triton.jit
def _ts_chunk_scan_tree_kernel_packed(
    S_local_ptr,
    Z_local_ptr,
    S_prefix_ptr,
    Z_prefix_ptr,
    H,
    NC: tl.constexpr,
    QD: tl.constexpr,
    QD_BLK: tl.constexpr,
    DV: tl.constexpr,
):
    """Grid: (B*H, QD). Each program handles ONE real qd index."""
    pid_bh = tl.program_id(0)
    q_idx = tl.program_id(1)  # 0 .. QD-1
    b = pid_bh // H
    h = pid_bh % H

    offs_nc = tl.arange(0, NC)
    offs_dv = tl.arange(0, DV)

    z_off = b * H * NC * QD + h * NC * QD + offs_nc * QD + q_idx
    Z = tl.load(Z_local_ptr + z_off)
    tl.store(Z_prefix_ptr + z_off, tl.cumsum(Z, axis=0) - Z)

    s_off = (
        b * H * NC * QD * DV
        + h * NC * QD * DV
        + offs_nc[:, None] * QD * DV
        + q_idx * DV
        + offs_dv[None, :]
    )
    S = tl.load(S_local_ptr + s_off)
    tl.store(S_prefix_ptr + s_off, tl.cumsum(S, axis=0) - S)


@triton.jit
def _ts_chunk_output_kernel_packed(
    Q_ptr,
    K_ptr,
    V_ptr,
    S_pref_ptr,
    Z_pref_ptr,
    O_ptr,
    D_ptr,
    H,
    T,
    NC,
    C: tl.constexpr,
    QD: tl.constexpr,
    QD_BLK: tl.constexpr,
    DV: tl.constexpr,
    DV_BLK: tl.constexpr,
    EPS: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_nc = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_c = tl.arange(0, C)
    offs_q = tl.arange(0, QD_BLK)
    q_mask = offs_q < QD
    t_idx = pid_nc * C + offs_c

    qk_off = b * T * H * QD + t_idx[:, None] * H * QD + h * QD + offs_q[None, :]
    Q = tl.load(Q_ptr + qk_off, mask=q_mask[None, :], other=0.0).to(tl.float32)
    K = tl.load(K_ptr + qk_off, mask=q_mask[None, :], other=0.0).to(tl.float32)

    z_off = b * H * NC * QD + h * NC * QD + pid_nc * QD + offs_q
    Zp = tl.load(Z_pref_ptr + z_off, mask=q_mask, other=0.0).to(tl.float32)

    Z_inter = tl.sum(Q * Zp[None, :], axis=1)
    A = tl.dot(Q, tl.trans(K), allow_tf32=False)
    causal = offs_c[:, None] >= offs_c[None, :]
    A_masked = tl.where(causal, A, 0.0)
    Z_intra = tl.sum(A_masked, axis=1)
    D = Z_inter + Z_intra + EPS

    d_off = b * T * H + t_idx * H + h
    tl.store(D_ptr + d_off, D)

    for dv_id in tl.static_range(DV // DV_BLK):
        offs_dv = dv_id * DV_BLK + tl.arange(0, DV_BLK)
        v_off = (
            b * T * H * DV + t_idx[:, None] * H * DV + h * DV + offs_dv[None, :]
        )
        V_tile = tl.load(V_ptr + v_off).to(tl.float32)

        sp_off = (
            b * H * NC * QD * DV
            + h * NC * QD * DV
            + pid_nc * QD * DV
            + offs_q[:, None] * DV
            + offs_dv[None, :]
        )
        Sp_tile = tl.load(
            S_pref_ptr + sp_off, mask=q_mask[:, None], other=0.0
        ).to(tl.float32)

        O_inter_tile = tl.dot(Q, Sp_tile, allow_tf32=False)
        O_intra_tile = tl.dot(A_masked, V_tile, allow_tf32=False)
        O_tile = (O_inter_tile + O_intra_tile) / D[:, None]

        o_off = v_off
        tl.store(O_ptr + o_off, O_tile.to(O_ptr.dtype.element_ty))


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return p


def chunk_linattn_treescan_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int = 32,
    eps: float = 1e-6,
    dv_block: int | None = None,
):
    """Tree-scan forward on materialized psi(q), psi(k) with arbitrary qd.

    QD_REAL = q.shape[-1] (no pow2 requirement); the kernel tiles along the
    QD axis with QD_BLK = next_pow2(QD_REAL) and masks the padded lanes on
    load/store. HBM strides use QD_REAL so S_local / S_prefix scale with the
    real qd, not the padded one.
    """
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    B, T, H, QD = q.shape
    DV = v.shape[-1]
    C = chunk_size
    if T % C:
        raise ValueError(f"T={T} not divisible by C={C}")
    NC = T // C
    if NC & (NC - 1):
        raise ValueError(f"NC=T/C={NC} must be a power of 2")
    QD_BLK = _next_pow2(QD)

    if dv_block is None:
        dv_block = 16 if QD_BLK >= 256 else DV
    if DV % dv_block:
        raise ValueError(f"DV={DV} not divisible by dv_block={dv_block}")

    dev = q.device
    S_local = torch.empty(B, H, NC, QD, DV, device=dev, dtype=torch.float32)
    Z_local = torch.empty(B, H, NC, QD, device=dev, dtype=torch.float32)
    S_prefix = torch.empty_like(S_local)
    Z_prefix = torch.empty_like(Z_local)
    O = torch.empty_like(v)
    D = torch.empty(B, T, H, device=dev, dtype=torch.float32)

    _ts_chunk_state_kernel_packed[(B * H, NC)](
        k, v, S_local, Z_local, H, T, NC,
        C=C, QD=QD, QD_BLK=QD_BLK, DV=DV, DV_BLK=dv_block,
        num_warps=4, num_stages=1,
    )
    _ts_chunk_scan_tree_kernel_packed[(B * H, QD)](
        S_local, Z_local, S_prefix, Z_prefix, H,
        NC=NC, QD=QD, QD_BLK=QD_BLK, DV=DV,
        num_warps=4, num_stages=1,
    )
    _ts_chunk_output_kernel_packed[(B * H, NC)](
        q, k, v, S_prefix, Z_prefix, O, D, H, T, NC,
        C=C, QD=QD, QD_BLK=QD_BLK, DV=DV, DV_BLK=dv_block,
        EPS=eps, num_warps=4, num_stages=1,
    )
    return O
