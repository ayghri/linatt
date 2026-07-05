import math

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------
# Triton autotune configs. Tune (BT, BS, num_warps, num_stages) for both
# fwd and bwd kernels. Small configs first so they're tried before larger
# ones that OOR on smaller SRAM (e.g., RTX 3090's 100KB shared).
# ---------------------------------------------------------------------

# NOTE: do NOT key on T (sequence length). The optimal tile (BT/BS/warps/stages)
from .utils import FWD_CONFIGS, BWD_CONFIGS, FWD_KEY, BWD_KEY_M1, BWD_KEY_ME

# the kernels below reference these with a leading underscore
_FWD_CONFIGS, _BWD_CONFIGS = FWD_CONFIGS, BWD_CONFIGS
_FWD_KEY, _BWD_KEY_M1, _BWD_KEY_ME = FWD_KEY, BWD_KEY_M1, BWD_KEY_ME


@triton.autotune(configs=FWD_CONFIGS, key=FWD_KEY)
@triton.jit
def parallel_kata_attn_fwd_kernel(
    q,  # (B, T, HQ, K)
    k,  # (B, T, H, K)
    v,  # (B, T, H, V)
    o,  # (B, T, HQ, V)
    den,  # (B, T, HQ)
    c,  # (B, T, HQ)
    scale,  # softmax_scale
    T,
    H: tl.constexpr,
    M: tl.constexpr,  # SPD num_groups
    K_d: tl.constexpr,  # head_k dim (= M * E)
    V_d: tl.constexpr,  # head_v dim
    HQ: tl.constexpr,
    G: tl.constexpr,  # = HQ // H (GQA grouping)
    BV: tl.constexpr,
    BT: tl.constexpr,  # query-block rows  (autotuned)
    BS: tl.constexpr,  # key-block rows    (autotuned)
    HAS_DECAY: tl.constexpr,  # multiply score by exp(c_t - c_s) (GDN-style decay)
):
    E: tl.constexpr = K_d // M    # per-group head dim (refactor dropped it from the sig)
    i_v = tl.program_id(0)
    i_t = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    bos = i_b * T

    p_o = tl.make_block_ptr(
        o + (bos * HQ + i_hq) * V_d,
        (T, V_d),
        (HQ * V_d, 1),
        (i_t * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )

    p_den = tl.make_block_ptr(
        den + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,)
    )

    # Running accumulators
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_acc = tl.zeros([BT], dtype=tl.float32)

    o_q = i_t * BT + tl.arange(0, BT)

    if HAS_DECAY:
        p_cq = tl.make_block_ptr(
            c + bos * HQ + i_hq,
            (T,),
            (HQ,),
            (i_t * BT,),
            (BT,),
            (0,),
        )
        # (BT,) cumulative decay at query rows
        b_cq = tl.load(p_cq, boundary_check=(0,))

    # Phase 1: strictly earlier key blocks (no causal mask needed)
    for i_s in range(0, i_t * BT, BS):
        b_s = tl.zeros([BT, BS], dtype=tl.float32)
        for m_id in tl.static_range(M):
            p_q_g = tl.make_block_ptr(
                q + (bos * HQ + i_hq) * K_d,
                (T, K_d),
                (HQ * K_d, 1),
                (i_t * BT, m_id * E),
                (BT, E),
                (1, 0),
            )
            p_k_g = tl.make_block_ptr(
                k + (bos * H + i_h) * K_d,
                (K_d, T),
                (1, H * K_d),
                (m_id * E, i_s),
                (E, BS),
                (0, 1),
            )
            b_q_g = tl.load(p_q_g, boundary_check=(0, 1))
            b_k_g = tl.load(p_k_g, boundary_check=(0, 1))
            qk_g = tl.dot(b_q_g, b_k_g) * scale
            b_s += qk_g * qk_g

        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        b_s = tl.where(m_k[None, :], b_s, 0.0)

        if HAS_DECAY:
            p_cs = tl.make_block_ptr(
                c + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,)
            )
            b_cs = tl.load(p_cs, boundary_check=(0,))
            b_s = b_s * tl.exp(tl.minimum(b_cq[:, None] - b_cs[None, :], 0.0))

        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V_d,
            (T, V_d),
            (H * V_d, 1),
            (i_s, i_v * BV),
            (BS, BV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))

        b_acc += tl.sum(b_s, axis=1)
        b_o += tl.dot(b_s.to(b_v.dtype), b_v)

    # Phase 2: on-diagonal block (causal mask)
    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        b_s = tl.zeros([BT, BS], dtype=tl.float32)
        for m_id in tl.static_range(M):
            p_q_g = tl.make_block_ptr(
                q + (bos * HQ + i_hq) * K_d,
                (T, K_d),
                (HQ * K_d, 1),
                (i_t * BT, m_id * E),
                (BT, E),
                (1, 0),
            )
            p_k_g = tl.make_block_ptr(
                k + (bos * H + i_h) * K_d,
                (K_d, T),
                (1, H * K_d),
                (m_id * E, i_s),
                (E, BS),
                (0, 1),
            )
            b_q_g = tl.load(p_q_g, boundary_check=(0, 1))
            b_k_g = tl.load(p_k_g, boundary_check=(0, 1))
            qk_g = tl.dot(b_q_g, b_k_g) * scale
            b_s += qk_g * qk_g

        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        m_s = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
        b_s = tl.where(m_s, b_s, 0.0)

        if HAS_DECAY:
            p_cs = tl.make_block_ptr(
                c + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,)
            )
            b_cs = tl.load(p_cs, boundary_check=(0,))
            b_s = b_s * tl.exp(tl.minimum(b_cq[:, None] - b_cs[None, :], 0.0))

        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V_d,
            (T, V_d),
            (H * V_d, 1),
            (i_s, i_v * BV),
            (BS, BV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))

        b_acc += tl.sum(b_s, axis=1)
        b_o += tl.dot(b_s.to(b_v.dtype), b_v)

    # Normalize
    b_acc_safe = tl.where(b_acc > 0, b_acc, 1.0)
    b_o = b_o / b_acc_safe[:, None]

    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_den, b_acc.to(p_den.dtype.element_ty), boundary_check=(0,))


@triton.autotune(configs=_FWD_CONFIGS, key=_FWD_KEY)
@triton.jit
def parallel_kata_attn_sum_fwd_kernel(
    q,
    k,
    v,
    o,
    den,
    scale,
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K_d: tl.constexpr,
    V_d: tl.constexpr,
    BV: tl.constexpr,
    M: tl.constexpr,
    E: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
):
    """Sum-SPD score: A[t,s] = Sum_{i,j} (scale * q_i[t] . k_j[s])^2.

    M^2 per-group dot products per (t,s) tile (vs the concat kernel's M).
    Implicit psi(q) = vec(Sum_i q_i q_i^T) of dim d_head^2 / M^2 (smaller
    state than concat). Matches the math used by paper KATA-SPD-4
    (psi_packed variant='four_rank' in kata/reference.py).
    """
    i_v = tl.program_id(0)
    i_t = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    bos = i_b * T

    p_o = tl.make_block_ptr(
        o + (bos * HQ + i_hq) * V_d,
        (T, V_d),
        (HQ * V_d, 1),
        (i_t * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    p_den = tl.make_block_ptr(
        den + bos * HQ + i_hq,
        (T,),
        (HQ,),
        (i_t * BT,),
        (BT,),
        (0,),
    )

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_acc = tl.zeros([BT], dtype=tl.float32)
    o_q = i_t * BT + tl.arange(0, BT)

    # Phase 1: strictly earlier key blocks
    for i_s in range(0, i_t * BT, BS):
        b_s = tl.zeros([BT, BS], dtype=tl.float32)
        for i in tl.static_range(M):
            p_q_i = tl.make_block_ptr(
                q + (bos * HQ + i_hq) * K_d,
                (T, K_d),
                (HQ * K_d, 1),
                (i_t * BT, i * E),
                (BT, E),
                (1, 0),
            )
            b_q_i = tl.load(p_q_i, boundary_check=(0, 1))
            for j in tl.static_range(M):
                p_k_j = tl.make_block_ptr(
                    k + (bos * H + i_h) * K_d,
                    (K_d, T),
                    (1, H * K_d),
                    (j * E, i_s),
                    (E, BS),
                    (0, 1),
                )
                b_k_j = tl.load(p_k_j, boundary_check=(0, 1))
                qk_ij = tl.dot(b_q_i, b_k_j) * scale
                b_s += qk_ij * qk_ij

        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        b_s = tl.where(m_k[None, :], b_s, 0.0)

        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V_d,
            (T, V_d),
            (H * V_d, 1),
            (i_s, i_v * BV),
            (BS, BV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_acc += tl.sum(b_s, axis=1)
        b_o += tl.dot(b_s.to(b_v.dtype), b_v)

    # Phase 2: on-diagonal block (causal)
    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        b_s = tl.zeros([BT, BS], dtype=tl.float32)
        for i in tl.static_range(M):
            p_q_i = tl.make_block_ptr(
                q + (bos * HQ + i_hq) * K_d,
                (T, K_d),
                (HQ * K_d, 1),
                (i_t * BT, i * E),
                (BT, E),
                (1, 0),
            )
            b_q_i = tl.load(p_q_i, boundary_check=(0, 1))
            for j in tl.static_range(M):
                p_k_j = tl.make_block_ptr(
                    k + (bos * H + i_h) * K_d,
                    (K_d, T),
                    (1, H * K_d),
                    (j * E, i_s),
                    (E, BS),
                    (0, 1),
                )
                b_k_j = tl.load(p_k_j, boundary_check=(0, 1))
                qk_ij = tl.dot(b_q_i, b_k_j) * scale
                b_s += qk_ij * qk_ij

        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        m_s = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
        b_s = tl.where(m_s, b_s, 0.0)

        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V_d,
            (T, V_d),
            (H * V_d, 1),
            (i_s, i_v * BV),
            (BS, BV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_acc += tl.sum(b_s, axis=1)
        b_o += tl.dot(b_s.to(b_v.dtype), b_v)

    b_acc_safe = tl.where(b_acc > 0, b_acc, 1.0)
    b_o = b_o / b_acc_safe[:, None]
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_den, b_acc.to(p_den.dtype.element_ty), boundary_check=(0,))


@triton.jit
def parallel_kata_attn_bwd_preprocess(
    o,
    do,
    delta,
    NEL: tl.constexpr,
    V: tl.constexpr,
):
    """delta[i] = <o[i,:], do[i,:]>  for each row i  (grid: total rows)."""
    i_n = tl.program_id(0)
    o_d = tl.arange(0, NEL)
    m_d = o_d < V
    b_o = tl.load(o + i_n * V + o_d, mask=m_d, other=0).to(tl.float32)
    b_do = tl.load(do + i_n * V + o_d, mask=m_d, other=0).to(tl.float32)
    tl.store(delta + i_n, tl.sum(b_o * b_do).to(delta.dtype.element_ty))


# =====================================================================
# Working M=4-unrolled backward kernels.
# Avoids the unsupported slice-add idiom by keeping 4 separate per-group
# accumulators (b_dq0..b_dq3 and b_dk0..b_dk3) and emitting 4 stores at
# the end. M is hardcoded to 4. For other M values, write a separate kernel
# or use the pytorch reference path in `parallel_kata_attn_bwd_torch`.
# =====================================================================








# =====================================================================
# M=1 backward kernels (degenerate: K_d == E, no grouping)
# =====================================================================






# =====================================================================
# M=2 backward kernels (two per-group accumulators)
# =====================================================================






# =====================================================================
# Working M=4-unrolled backward kernels.
# Avoids the unsupported slice-add idiom by keeping 4 separate per-group
# accumulators (b_dq0..b_dq3 and b_dk0..b_dk3) and emitting 4 stores at
# the end. M is hardcoded to 4. For other M values, write a separate kernel
# or use the pytorch reference path in `parallel_kata_attn_bwd_torch`.
# =====================================================================








# =====================================================================
# M=1 backward kernels (degenerate: K_d == E, no grouping)
# =====================================================================






# =====================================================================
# M=2 backward kernels (two per-group accumulators)
# =====================================================================






# =====================================================================
# Working M=4-unrolled backward kernels.
# Avoids the unsupported slice-add idiom by keeping 4 separate per-group
# accumulators (b_dq0..b_dq3 and b_dk0..b_dk3) and emitting 4 stores at
# the end. M is hardcoded to 4. For other M values, write a separate kernel
# or use the pytorch reference path in `parallel_kata_attn_bwd_torch`.
# =====================================================================








# =====================================================================
# M=1 backward kernels (degenerate: K_d == E, no grouping)
# =====================================================================






# =====================================================================
# M=2 backward kernels (two per-group accumulators)
# =====================================================================






# =====================================================================
# Working M=4-unrolled backward kernels.
# Avoids the unsupported slice-add idiom by keeping 4 separate per-group
# accumulators (b_dq0..b_dq3 and b_dk0..b_dk3) and emitting 4 stores at
# the end. M is hardcoded to 4. For other M values, write a separate kernel
# or use the pytorch reference path in `parallel_kata_attn_bwd_torch`.
# =====================================================================


@triton.autotune(configs=_BWD_CONFIGS, key=_BWD_KEY_ME)
@triton.jit
def parallel_kata_attn_bwd_kernel_dq_M4(
    q,
    k,
    v,
    den,
    delta,
    do,
    dq,
    scale,
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K_d: tl.constexpr,
    V_d: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BV: tl.constexpr,
    E: tl.constexpr,
):
    """dQ for kata-quadratic attention, M=4. Grid: (1, NT, B*HQ)."""
    i_t = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    bos = i_b * T

    p_do = tl.make_block_ptr(
        do + (bos * HQ + i_hq) * V_d,
        (T, V_d),
        (HQ * V_d, 1),
        (i_t * BT, 0),
        (BT, BV),
        (1, 0),
    )
    p_den = tl.make_block_ptr(
        den + bos * HQ + i_hq,
        (T,),
        (HQ,),
        (i_t * BT,),
        (BT,),
        (0,),
    )
    p_delta = tl.make_block_ptr(
        delta + bos * HQ + i_hq,
        (T,),
        (HQ,),
        (i_t * BT,),
        (BT,),
        (0,),
    )
    b_do = tl.load(p_do, boundary_check=(0, 1)).to(tl.float32)
    b_D = tl.load(p_den, boundary_check=(0,))
    b_delta = tl.load(p_delta, boundary_check=(0,))
    b_D_safe = tl.where(b_D > 0, b_D, 1.0)

    # M=4 per-group dq accumulators
    b_dq0 = tl.zeros([BT, E], dtype=tl.float32)
    b_dq1 = tl.zeros([BT, E], dtype=tl.float32)
    b_dq2 = tl.zeros([BT, E], dtype=tl.float32)
    b_dq3 = tl.zeros([BT, E], dtype=tl.float32)

    o_q = i_t * BT + tl.arange(0, BT)

    for i_s in range(0, min((i_t + 1) * BT, T), BS):
        # Per-group Q (BT, E) and K (E, BS).
        p_q0 = tl.make_block_ptr(
            q + (bos * HQ + i_hq) * K_d,
            (T, K_d),
            (HQ * K_d, 1),
            (i_t * BT, 0 * E),
            (BT, E),
            (1, 0),
        )
        p_q1 = tl.make_block_ptr(
            q + (bos * HQ + i_hq) * K_d,
            (T, K_d),
            (HQ * K_d, 1),
            (i_t * BT, 1 * E),
            (BT, E),
            (1, 0),
        )
        p_q2 = tl.make_block_ptr(
            q + (bos * HQ + i_hq) * K_d,
            (T, K_d),
            (HQ * K_d, 1),
            (i_t * BT, 2 * E),
            (BT, E),
            (1, 0),
        )
        p_q3 = tl.make_block_ptr(
            q + (bos * HQ + i_hq) * K_d,
            (T, K_d),
            (HQ * K_d, 1),
            (i_t * BT, 3 * E),
            (BT, E),
            (1, 0),
        )
        p_k0 = tl.make_block_ptr(
            k + (bos * H + i_h) * K_d,
            (K_d, T),
            (1, H * K_d),
            (0 * E, i_s),
            (E, BS),
            (0, 1),
        )
        p_k1 = tl.make_block_ptr(
            k + (bos * H + i_h) * K_d,
            (K_d, T),
            (1, H * K_d),
            (1 * E, i_s),
            (E, BS),
            (0, 1),
        )
        p_k2 = tl.make_block_ptr(
            k + (bos * H + i_h) * K_d,
            (K_d, T),
            (1, H * K_d),
            (2 * E, i_s),
            (E, BS),
            (0, 1),
        )
        p_k3 = tl.make_block_ptr(
            k + (bos * H + i_h) * K_d,
            (K_d, T),
            (1, H * K_d),
            (3 * E, i_s),
            (E, BS),
            (0, 1),
        )

        b_q0 = tl.load(p_q0, boundary_check=(0, 1))
        b_q1 = tl.load(p_q1, boundary_check=(0, 1))
        b_q2 = tl.load(p_q2, boundary_check=(0, 1))
        b_q3 = tl.load(p_q3, boundary_check=(0, 1))
        b_k0 = tl.load(p_k0, boundary_check=(0, 1))
        b_k1 = tl.load(p_k1, boundary_check=(0, 1))
        b_k2 = tl.load(p_k2, boundary_check=(0, 1))
        b_k3 = tl.load(p_k3, boundary_check=(0, 1))

        qk0 = tl.dot(b_q0, b_k0) * scale
        qk1 = tl.dot(b_q1, b_k1) * scale
        qk2 = tl.dot(b_q2, b_k2) * scale
        qk3 = tl.dot(b_q3, b_k3) * scale

        # b_A not needed in dq kernel (we use dA directly).

        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V_d,
            (V_d, T),
            (1, H * V_d),
            (0, i_s),
            (BV, BS),
            (0, 1),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
        b_dN_V = tl.dot(b_do, b_v)  # (BT, BS)
        b_dA = (b_dN_V - b_delta[:, None]) / b_D_safe[:, None]

        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        causal = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
        b_dA = tl.where(causal, b_dA, 0.0)

        da0 = 2.0 * qk0 * b_dA
        da1 = 2.0 * qk1 * b_dA
        da2 = 2.0 * qk2 * b_dA
        da3 = 2.0 * qk3 * b_dA

        # b_k_g is (E, BS); transpose to (BS, E) for the dot
        b_dq0 += tl.dot(da0.to(b_k0.dtype), tl.trans(b_k0)) * scale
        b_dq1 += tl.dot(da1.to(b_k1.dtype), tl.trans(b_k1)) * scale
        b_dq2 += tl.dot(da2.to(b_k2.dtype), tl.trans(b_k2)) * scale
        b_dq3 += tl.dot(da3.to(b_k3.dtype), tl.trans(b_k3)) * scale

    # Store the four (BT, E) slices of dQ.
    p_dq0 = tl.make_block_ptr(
        dq + (bos * HQ + i_hq) * K_d,
        (T, K_d),
        (HQ * K_d, 1),
        (i_t * BT, 0 * E),
        (BT, E),
        (1, 0),
    )
    p_dq1 = tl.make_block_ptr(
        dq + (bos * HQ + i_hq) * K_d,
        (T, K_d),
        (HQ * K_d, 1),
        (i_t * BT, 1 * E),
        (BT, E),
        (1, 0),
    )
    p_dq2 = tl.make_block_ptr(
        dq + (bos * HQ + i_hq) * K_d,
        (T, K_d),
        (HQ * K_d, 1),
        (i_t * BT, 2 * E),
        (BT, E),
        (1, 0),
    )
    p_dq3 = tl.make_block_ptr(
        dq + (bos * HQ + i_hq) * K_d,
        (T, K_d),
        (HQ * K_d, 1),
        (i_t * BT, 3 * E),
        (BT, E),
        (1, 0),
    )
    tl.store(p_dq0, b_dq0.to(p_dq0.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dq1, b_dq1.to(p_dq1.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dq2, b_dq2.to(p_dq2.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dq3, b_dq3.to(p_dq3.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(configs=_BWD_CONFIGS, key=_BWD_KEY_ME)
@triton.jit
def parallel_kata_attn_bwd_kernel_dkdv_M4(
    q,
    k,
    v,
    den,
    delta,
    do,
    dk,
    dv,
    scale,
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,  # assumed = 1 (MHA); for GQA, post-sum in python
    K_d: tl.constexpr,
    V_d: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BV: tl.constexpr,
    E: tl.constexpr,
):
    """dK + dV for kata-quadratic attention, M=4. Grid: (1, NS, B*HQ).

    Note: G=1 assumption. For GQA (G>1) we'd accumulate one dK/dV per
    query head and sum after — the wrapper handles that path in pytorch.
    """
    i_s = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    bos = i_b * T

    b_dk0 = tl.zeros([BS, E], dtype=tl.float32)
    b_dk1 = tl.zeros([BS, E], dtype=tl.float32)
    b_dk2 = tl.zeros([BS, E], dtype=tl.float32)
    b_dk3 = tl.zeros([BS, E], dtype=tl.float32)
    b_dv = tl.zeros([BS, BV], dtype=tl.float32)

    o_k = i_s * BS + tl.arange(0, BS)

    t_start = (i_s * BS) // BT
    NT = tl.cdiv(T, BT)
    for i_t in range(t_start, NT):
        p_do = tl.make_block_ptr(
            do + (bos * HQ + i_hq) * V_d,
            (T, V_d),
            (HQ * V_d, 1),
            (i_t * BT, 0),
            (BT, BV),
            (1, 0),
        )
        p_den = tl.make_block_ptr(
            den + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,)
        )
        p_delta = tl.make_block_ptr(
            delta + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,)
        )
        b_do = tl.load(p_do, boundary_check=(0, 1)).to(tl.float32)
        b_D = tl.load(p_den, boundary_check=(0,))
        b_delta = tl.load(p_delta, boundary_check=(0,))
        b_D_safe = tl.where(b_D > 0, b_D, 1.0)

        # Per-group Q (BT, E) and K (E, BS).
        p_q0 = tl.make_block_ptr(
            q + (bos * HQ + i_hq) * K_d,
            (T, K_d),
            (HQ * K_d, 1),
            (i_t * BT, 0 * E),
            (BT, E),
            (1, 0),
        )
        p_q1 = tl.make_block_ptr(
            q + (bos * HQ + i_hq) * K_d,
            (T, K_d),
            (HQ * K_d, 1),
            (i_t * BT, 1 * E),
            (BT, E),
            (1, 0),
        )
        p_q2 = tl.make_block_ptr(
            q + (bos * HQ + i_hq) * K_d,
            (T, K_d),
            (HQ * K_d, 1),
            (i_t * BT, 2 * E),
            (BT, E),
            (1, 0),
        )
        p_q3 = tl.make_block_ptr(
            q + (bos * HQ + i_hq) * K_d,
            (T, K_d),
            (HQ * K_d, 1),
            (i_t * BT, 3 * E),
            (BT, E),
            (1, 0),
        )
        p_k0 = tl.make_block_ptr(
            k + (bos * H + i_h) * K_d,
            (K_d, T),
            (1, H * K_d),
            (0 * E, i_s * BS),
            (E, BS),
            (0, 1),
        )
        p_k1 = tl.make_block_ptr(
            k + (bos * H + i_h) * K_d,
            (K_d, T),
            (1, H * K_d),
            (1 * E, i_s * BS),
            (E, BS),
            (0, 1),
        )
        p_k2 = tl.make_block_ptr(
            k + (bos * H + i_h) * K_d,
            (K_d, T),
            (1, H * K_d),
            (2 * E, i_s * BS),
            (E, BS),
            (0, 1),
        )
        p_k3 = tl.make_block_ptr(
            k + (bos * H + i_h) * K_d,
            (K_d, T),
            (1, H * K_d),
            (3 * E, i_s * BS),
            (E, BS),
            (0, 1),
        )

        b_q0 = tl.load(p_q0, boundary_check=(0, 1))
        b_q1 = tl.load(p_q1, boundary_check=(0, 1))
        b_q2 = tl.load(p_q2, boundary_check=(0, 1))
        b_q3 = tl.load(p_q3, boundary_check=(0, 1))
        b_k0 = tl.load(p_k0, boundary_check=(0, 1))
        b_k1 = tl.load(p_k1, boundary_check=(0, 1))
        b_k2 = tl.load(p_k2, boundary_check=(0, 1))
        b_k3 = tl.load(p_k3, boundary_check=(0, 1))

        qk0 = tl.dot(b_q0, b_k0) * scale
        qk1 = tl.dot(b_q1, b_k1) * scale
        qk2 = tl.dot(b_q2, b_k2) * scale
        qk3 = tl.dot(b_q3, b_k3) * scale
        b_A = qk0 * qk0 + qk1 * qk1 + qk2 * qk2 + qk3 * qk3

        o_q = i_t * BT + tl.arange(0, BT)
        m_k = o_k < T
        causal = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]

        # dV first.  P[t,s] = A[t,s] / D[t]
        b_P = tl.where(causal, b_A / b_D_safe[:, None], 0.0)
        b_dv += tl.dot(tl.trans(b_P.to(b_do.dtype)), b_do)

        # dA = (<dO, V> - delta) / D
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V_d,
            (V_d, T),
            (1, H * V_d),
            (0, i_s * BS),
            (BV, BS),
            (0, 1),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
        b_dN_V = tl.dot(b_do, b_v)
        b_dA = (b_dN_V - b_delta[:, None]) / b_D_safe[:, None]
        b_dA = tl.where(causal, b_dA, 0.0)

        # da per group, then dk_g += scale * sum_t da[t,s] * q_g[t,:]  ⇒  scale * da^T @ q_g (BS, E)
        da0 = 2.0 * qk0 * b_dA
        da1 = 2.0 * qk1 * b_dA
        da2 = 2.0 * qk2 * b_dA
        da3 = 2.0 * qk3 * b_dA

        b_dk0 += tl.dot(tl.trans(da0.to(b_q0.dtype)), b_q0) * scale
        b_dk1 += tl.dot(tl.trans(da1.to(b_q1.dtype)), b_q1) * scale
        b_dk2 += tl.dot(tl.trans(da2.to(b_q2.dtype)), b_q2) * scale
        b_dk3 += tl.dot(tl.trans(da3.to(b_q3.dtype)), b_q3) * scale

    # Stores
    p_dk0 = tl.make_block_ptr(
        dk + (bos * H + i_h) * K_d,
        (T, K_d),
        (H * K_d, 1),
        (i_s * BS, 0 * E),
        (BS, E),
        (1, 0),
    )
    p_dk1 = tl.make_block_ptr(
        dk + (bos * H + i_h) * K_d,
        (T, K_d),
        (H * K_d, 1),
        (i_s * BS, 1 * E),
        (BS, E),
        (1, 0),
    )
    p_dk2 = tl.make_block_ptr(
        dk + (bos * H + i_h) * K_d,
        (T, K_d),
        (H * K_d, 1),
        (i_s * BS, 2 * E),
        (BS, E),
        (1, 0),
    )
    p_dk3 = tl.make_block_ptr(
        dk + (bos * H + i_h) * K_d,
        (T, K_d),
        (H * K_d, 1),
        (i_s * BS, 3 * E),
        (BS, E),
        (1, 0),
    )
    p_dv = tl.make_block_ptr(
        dv + (bos * H + i_h) * V_d,
        (T, V_d),
        (H * V_d, 1),
        (i_s * BS, 0),
        (BS, BV),
        (1, 0),
    )
    tl.store(p_dk0, b_dk0.to(p_dk0.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk1, b_dk1.to(p_dk1.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk2, b_dk2.to(p_dk2.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk3, b_dk3.to(p_dk3.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


def parallel_kata_attn_bwd_triton_M4(
    q,
    k,
    v,
    o,
    den,
    do,
    scale,
    BT=64,
    BS=64,
):
    """Triton bwd, M=4 hardcoded, MHA only (G=1).

    Returns (dq, dk, dv) in input dtypes.
    """
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    do = do.contiguous()
    B, T, HQ, K_d = q.shape
    H = k.shape[2]
    V_d = v.shape[-1]
    assert H == HQ, "M=4 Triton bwd assumes MHA (G=1); use pytorch ref for GQA"
    M = 4
    E = K_d // M
    assert K_d == M * E, f"K_d={K_d} not divisible by M={M}"

    dev = q.device
    # Preprocess: delta[t] = <o[t], do[t]>
    delta = torch.empty(B * T * HQ, device=dev, dtype=torch.float32)
    NEL = triton.next_power_of_2(V_d)
    parallel_kata_attn_bwd_preprocess[(B * T * HQ,)](
        o.contiguous().view(-1, V_d),
        do.view(-1, V_d),
        delta,
        NEL=NEL,
        V=V_d,
        num_warps=4,
        num_stages=1,
    )
    delta = delta.view(B, T, HQ)

    dq = torch.empty(B, T, HQ, K_d, device=dev, dtype=q.dtype)
    dk = torch.empty(B, T, H, K_d, device=dev, dtype=k.dtype)
    dv = torch.empty(B, T, H, V_d, device=dev, dtype=v.dtype)

    parallel_kata_attn_bwd_kernel_dq_M4[
        lambda meta: (1, triton.cdiv(T, meta["BT"]), B * HQ)
    ](
        q,
        k,
        v,
        den,
        delta,
        do,
        dq,
        scale,
        T,
        H=H,
        HQ=HQ,
        G=1,
        K_d=K_d,
        V_d=V_d,
        BV=V_d,
        E=E,
    )
    parallel_kata_attn_bwd_kernel_dkdv_M4[
        lambda meta: (1, triton.cdiv(T, meta["BS"]), B * HQ)
    ](
        q,
        k,
        v,
        den,
        delta,
        do,
        dk,
        dv,
        scale,
        T,
        H=H,
        HQ=HQ,
        G=1,
        K_d=K_d,
        V_d=V_d,
        BV=V_d,
        E=E,
    )
    return dq, dk, dv


# =====================================================================
# M=1 backward kernels (degenerate: K_d == E, no grouping)
# =====================================================================


@triton.autotune(configs=_BWD_CONFIGS, key=_BWD_KEY_M1)
@triton.jit
def parallel_kata_attn_bwd_kernel_dq_M1(
    q,
    k,
    v,
    den,
    delta,
    do,
    dq,
    scale,
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K_d: tl.constexpr,
    V_d: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BV: tl.constexpr,
):
    i_t = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    bos = i_b * T

    p_do = tl.make_block_ptr(
        do + (bos * HQ + i_hq) * V_d,
        (T, V_d),
        (HQ * V_d, 1),
        (i_t * BT, 0),
        (BT, BV),
        (1, 0),
    )
    p_den = tl.make_block_ptr(
        den + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,)
    )
    p_delta = tl.make_block_ptr(
        delta + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,)
    )
    b_do = tl.load(p_do, boundary_check=(0, 1))   # keep bf16 for the do.v tensor-core dot
    b_D = tl.load(p_den, boundary_check=(0,))
    b_delta = tl.load(p_delta, boundary_check=(0,))
    b_D_safe = tl.where(b_D > 0, b_D, 1.0)

    b_dq = tl.zeros([BT, K_d], dtype=tl.float32)
    o_q = i_t * BT + tl.arange(0, BT)
    # q-block is loop-invariant -> load ONCE, not per key-block.
    p_q = tl.make_block_ptr(
        q + (bos * HQ + i_hq) * K_d, (T, K_d), (HQ * K_d, 1),
        (i_t * BT, 0), (BT, K_d), (1, 0),
    )
    b_q = tl.load(p_q, boundary_check=(0, 1))

    for i_s in range(0, min((i_t + 1) * BT, T), BS):
        p_k = tl.make_block_ptr(
            k + (bos * H + i_h) * K_d,
            (K_d, T),
            (1, H * K_d),
            (0, i_s),
            (K_d, BS),
            (0, 1),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        qk = tl.dot(b_q, b_k) * scale

        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V_d,
            (V_d, T),
            (1, H * V_d),
            (0, i_s),
            (BV, BS),
            (0, 1),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))   # bf16 -> tensor-core do.v (fp32 accum)
        b_dN_V = tl.dot(b_do, b_v)
        b_dA = (b_dN_V - b_delta[:, None]) / b_D_safe[:, None]
        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        causal = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
        b_dA = tl.where(causal, b_dA, 0.0)

        da = 2.0 * qk * b_dA
        b_dq += tl.dot(da.to(b_k.dtype), tl.trans(b_k)) * scale

    p_dq = tl.make_block_ptr(
        dq + (bos * HQ + i_hq) * K_d,
        (T, K_d),
        (HQ * K_d, 1),
        (i_t * BT, 0),
        (BT, K_d),
        (1, 0),
    )
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(configs=_BWD_CONFIGS, key=_BWD_KEY_M1)
@triton.jit
def parallel_kata_attn_bwd_kernel_dkdv_M1(
    q,
    k,
    v,
    den,
    delta,
    do,
    dk,
    dv,
    scale,
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K_d: tl.constexpr,
    V_d: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BV: tl.constexpr,
):
    i_s = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    bos = i_b * T

    b_dk = tl.zeros([BS, K_d], dtype=tl.float32)
    b_dv = tl.zeros([BS, BV], dtype=tl.float32)
    o_k = i_s * BS + tl.arange(0, BS)
    t_start = (i_s * BS) // BT
    NT = tl.cdiv(T, BT)
    # k,v blocks are loop-invariant (fixed i_s) -> load ONCE, bf16 for tensor-core dots.
    p_k = tl.make_block_ptr(
        k + (bos * H + i_h) * K_d, (K_d, T), (1, H * K_d),
        (0, i_s * BS), (K_d, BS), (0, 1),
    )
    p_v = tl.make_block_ptr(
        v + (bos * H + i_h) * V_d, (V_d, T), (1, H * V_d),
        (0, i_s * BS), (BV, BS), (0, 1),
    )
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_v = tl.load(p_v, boundary_check=(0, 1))

    for i_t in range(t_start, NT):
        p_do = tl.make_block_ptr(
            do + (bos * HQ + i_hq) * V_d,
            (T, V_d),
            (HQ * V_d, 1),
            (i_t * BT, 0),
            (BT, BV),
            (1, 0),
        )
        p_den = tl.make_block_ptr(
            den + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,)
        )
        p_delta = tl.make_block_ptr(
            delta + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,)
        )
        b_do = tl.load(p_do, boundary_check=(0, 1))   # bf16 for the dv / do.v tensor-core dots
        b_D = tl.load(p_den, boundary_check=(0,))
        b_delta = tl.load(p_delta, boundary_check=(0,))
        b_D_safe = tl.where(b_D > 0, b_D, 1.0)

        p_q = tl.make_block_ptr(
            q + (bos * HQ + i_hq) * K_d,
            (T, K_d),
            (HQ * K_d, 1),
            (i_t * BT, 0),
            (BT, K_d),
            (1, 0),
        )
        b_q = tl.load(p_q, boundary_check=(0, 1))
        qk = tl.dot(b_q, b_k) * scale
        b_A = qk * qk

        o_q = i_t * BT + tl.arange(0, BT)
        m_k = o_k < T
        causal = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
        b_P = tl.where(causal, b_A / b_D_safe[:, None], 0.0)
        b_dv += tl.dot(tl.trans(b_P.to(b_do.dtype)), b_do)

        b_dN_V = tl.dot(b_do, b_v)
        b_dA = (b_dN_V - b_delta[:, None]) / b_D_safe[:, None]
        b_dA = tl.where(causal, b_dA, 0.0)
        da = 2.0 * qk * b_dA
        b_dk += tl.dot(tl.trans(da.to(b_q.dtype)), b_q) * scale

    p_dk = tl.make_block_ptr(
        dk + (bos * H + i_h) * K_d,
        (T, K_d),
        (H * K_d, 1),
        (i_s * BS, 0),
        (BS, K_d),
        (1, 0),
    )
    p_dv = tl.make_block_ptr(
        dv + (bos * H + i_h) * V_d,
        (T, V_d),
        (H * V_d, 1),
        (i_s * BS, 0),
        (BS, BV),
        (1, 0),
    )
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


# =====================================================================
# M=2 backward kernels (two per-group accumulators)
# =====================================================================


@triton.autotune(configs=_BWD_CONFIGS, key=_BWD_KEY_ME)
@triton.jit
def parallel_kata_attn_bwd_kernel_dq_M2(
    q,
    k,
    v,
    den,
    delta,
    do,
    dq,
    scale,
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K_d: tl.constexpr,
    V_d: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BV: tl.constexpr,
    E: tl.constexpr,
):
    i_t = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = i_b * T

    p_do = tl.make_block_ptr(
        do + (bos * HQ + i_hq) * V_d,
        (T, V_d),
        (HQ * V_d, 1),
        (i_t * BT, 0),
        (BT, BV),
        (1, 0),
    )
    p_den = tl.make_block_ptr(
        den + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,)
    )
    p_delta = tl.make_block_ptr(
        delta + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,)
    )
    b_do = tl.load(p_do, boundary_check=(0, 1)).to(tl.float32)
    b_D = tl.load(p_den, boundary_check=(0,))
    b_delta = tl.load(p_delta, boundary_check=(0,))
    b_D_safe = tl.where(b_D > 0, b_D, 1.0)

    b_dq0 = tl.zeros([BT, E], dtype=tl.float32)
    b_dq1 = tl.zeros([BT, E], dtype=tl.float32)
    o_q = i_t * BT + tl.arange(0, BT)

    for i_s in range(0, min((i_t + 1) * BT, T), BS):
        p_q0 = tl.make_block_ptr(
            q + (bos * HQ + i_hq) * K_d,
            (T, K_d),
            (HQ * K_d, 1),
            (i_t * BT, 0 * E),
            (BT, E),
            (1, 0),
        )
        p_q1 = tl.make_block_ptr(
            q + (bos * HQ + i_hq) * K_d,
            (T, K_d),
            (HQ * K_d, 1),
            (i_t * BT, 1 * E),
            (BT, E),
            (1, 0),
        )
        p_k0 = tl.make_block_ptr(
            k + (bos * H + i_h) * K_d,
            (K_d, T),
            (1, H * K_d),
            (0 * E, i_s),
            (E, BS),
            (0, 1),
        )
        p_k1 = tl.make_block_ptr(
            k + (bos * H + i_h) * K_d,
            (K_d, T),
            (1, H * K_d),
            (1 * E, i_s),
            (E, BS),
            (0, 1),
        )
        b_q0 = tl.load(p_q0, boundary_check=(0, 1))
        b_q1 = tl.load(p_q1, boundary_check=(0, 1))
        b_k0 = tl.load(p_k0, boundary_check=(0, 1))
        b_k1 = tl.load(p_k1, boundary_check=(0, 1))
        qk0 = tl.dot(b_q0, b_k0) * scale
        qk1 = tl.dot(b_q1, b_k1) * scale

        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V_d,
            (V_d, T),
            (1, H * V_d),
            (0, i_s),
            (BV, BS),
            (0, 1),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
        b_dN_V = tl.dot(b_do, b_v)
        b_dA = (b_dN_V - b_delta[:, None]) / b_D_safe[:, None]
        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        causal = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
        b_dA = tl.where(causal, b_dA, 0.0)
        da0 = 2.0 * qk0 * b_dA
        da1 = 2.0 * qk1 * b_dA
        b_dq0 += tl.dot(da0.to(b_k0.dtype), tl.trans(b_k0)) * scale
        b_dq1 += tl.dot(da1.to(b_k1.dtype), tl.trans(b_k1)) * scale

    p_dq0 = tl.make_block_ptr(
        dq + (bos * HQ + i_hq) * K_d,
        (T, K_d),
        (HQ * K_d, 1),
        (i_t * BT, 0 * E),
        (BT, E),
        (1, 0),
    )
    p_dq1 = tl.make_block_ptr(
        dq + (bos * HQ + i_hq) * K_d,
        (T, K_d),
        (HQ * K_d, 1),
        (i_t * BT, 1 * E),
        (BT, E),
        (1, 0),
    )
    tl.store(p_dq0, b_dq0.to(p_dq0.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dq1, b_dq1.to(p_dq1.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(configs=_BWD_CONFIGS, key=_BWD_KEY_ME)
@triton.jit
def parallel_kata_attn_bwd_kernel_dkdv_M2(
    q,
    k,
    v,
    den,
    delta,
    do,
    dk,
    dv,
    scale,
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K_d: tl.constexpr,
    V_d: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BV: tl.constexpr,
    E: tl.constexpr,
):
    i_s = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = i_b * T

    b_dk0 = tl.zeros([BS, E], dtype=tl.float32)
    b_dk1 = tl.zeros([BS, E], dtype=tl.float32)
    b_dv = tl.zeros([BS, BV], dtype=tl.float32)
    o_k = i_s * BS + tl.arange(0, BS)
    t_start = (i_s * BS) // BT
    NT = tl.cdiv(T, BT)

    for i_t in range(t_start, NT):
        p_do = tl.make_block_ptr(
            do + (bos * HQ + i_hq) * V_d,
            (T, V_d),
            (HQ * V_d, 1),
            (i_t * BT, 0),
            (BT, BV),
            (1, 0),
        )
        p_den = tl.make_block_ptr(
            den + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,)
        )
        p_delta = tl.make_block_ptr(
            delta + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,)
        )
        b_do = tl.load(p_do, boundary_check=(0, 1)).to(tl.float32)
        b_D = tl.load(p_den, boundary_check=(0,))
        b_delta = tl.load(p_delta, boundary_check=(0,))
        b_D_safe = tl.where(b_D > 0, b_D, 1.0)

        p_q0 = tl.make_block_ptr(
            q + (bos * HQ + i_hq) * K_d,
            (T, K_d),
            (HQ * K_d, 1),
            (i_t * BT, 0 * E),
            (BT, E),
            (1, 0),
        )
        p_q1 = tl.make_block_ptr(
            q + (bos * HQ + i_hq) * K_d,
            (T, K_d),
            (HQ * K_d, 1),
            (i_t * BT, 1 * E),
            (BT, E),
            (1, 0),
        )
        p_k0 = tl.make_block_ptr(
            k + (bos * H + i_h) * K_d,
            (K_d, T),
            (1, H * K_d),
            (0 * E, i_s * BS),
            (E, BS),
            (0, 1),
        )
        p_k1 = tl.make_block_ptr(
            k + (bos * H + i_h) * K_d,
            (K_d, T),
            (1, H * K_d),
            (1 * E, i_s * BS),
            (E, BS),
            (0, 1),
        )
        b_q0 = tl.load(p_q0, boundary_check=(0, 1))
        b_q1 = tl.load(p_q1, boundary_check=(0, 1))
        b_k0 = tl.load(p_k0, boundary_check=(0, 1))
        b_k1 = tl.load(p_k1, boundary_check=(0, 1))
        qk0 = tl.dot(b_q0, b_k0) * scale
        qk1 = tl.dot(b_q1, b_k1) * scale
        b_A = qk0 * qk0 + qk1 * qk1

        o_q = i_t * BT + tl.arange(0, BT)
        m_k = o_k < T
        causal = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
        b_P = tl.where(causal, b_A / b_D_safe[:, None], 0.0)
        b_dv += tl.dot(tl.trans(b_P.to(b_do.dtype)), b_do)

        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V_d,
            (V_d, T),
            (1, H * V_d),
            (0, i_s * BS),
            (BV, BS),
            (0, 1),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
        b_dN_V = tl.dot(b_do, b_v)
        b_dA = (b_dN_V - b_delta[:, None]) / b_D_safe[:, None]
        b_dA = tl.where(causal, b_dA, 0.0)
        da0 = 2.0 * qk0 * b_dA
        da1 = 2.0 * qk1 * b_dA
        b_dk0 += tl.dot(tl.trans(da0.to(b_q0.dtype)), b_q0) * scale
        b_dk1 += tl.dot(tl.trans(da1.to(b_q1.dtype)), b_q1) * scale

    p_dk0 = tl.make_block_ptr(
        dk + (bos * H + i_h) * K_d,
        (T, K_d),
        (H * K_d, 1),
        (i_s * BS, 0 * E),
        (BS, E),
        (1, 0),
    )
    p_dk1 = tl.make_block_ptr(
        dk + (bos * H + i_h) * K_d,
        (T, K_d),
        (H * K_d, 1),
        (i_s * BS, 1 * E),
        (BS, E),
        (1, 0),
    )
    p_dv = tl.make_block_ptr(
        dv + (bos * H + i_h) * V_d,
        (T, V_d),
        (H * V_d, 1),
        (i_s * BS, 0),
        (BS, BV),
        (1, 0),
    )
    tl.store(p_dk0, b_dk0.to(p_dk0.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk1, b_dk1.to(p_dk1.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
