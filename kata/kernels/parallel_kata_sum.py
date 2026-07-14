import triton
import triton.language as tl

from .utils import BWD_CONFIGS, BWD_KEY_ME


@triton.autotune(configs=BWD_CONFIGS, key=BWD_KEY_ME)
@triton.jit
def parallel_kata_attn_sum_bwd_kernel_dkdv_M2(
    q_addr,
    k_addr,
    v_addr,
    den_addr,
    delta_addr,
    do_addr,
    dk_addr,
    dv_addr,
    scale,
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    d_k: tl.constexpr,
    d_v: tl.constexpr,
    bV: tl.constexpr,
    E: tl.constexpr,
    bT: tl.constexpr,
    bK: tl.constexpr,
):
    i_s = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = i_b * T

    b_dk0 = tl.zeros([bK, E], dtype=tl.float32)
    b_dk1 = tl.zeros([bK, E], dtype=tl.float32)
    b_dv = tl.zeros([bK, bV], dtype=tl.float32)
    o_k = i_s * bK + tl.arange(0, bK)
    t_start = (i_s * bK) // bT
    NT = tl.cdiv(T, bT)

    for i_t in range(t_start, NT):
        p_do = tl.make_block_ptr(
            do_addr + (bos * HQ + i_hq) * d_v,
            (T, d_v),
            (HQ * d_v, 1),
            (i_t * bT, 0),
            (bT, bV),
            (1, 0),
        )
        p_den = tl.make_block_ptr(
            den_addr + bos * HQ + i_hq, (T,), (HQ,), (i_t * bT,), (bT,), (0,)
        )
        p_delta = tl.make_block_ptr(
            delta_addr + bos * HQ + i_hq, (T,), (HQ,), (i_t * bT,), (bT,), (0,)
        )
        b_do = tl.load(p_do, boundary_check=(0, 1)).to(tl.float32)
        b_D = tl.load(p_den, boundary_check=(0,))
        b_delta = tl.load(p_delta, boundary_check=(0,))
        b_D_safe = tl.where(b_D > 0, b_D, 1.0)

        p_q0 = tl.make_block_ptr(
            q_addr + (bos * HQ + i_hq) * d_k,
            (T, d_k),
            (HQ * d_k, 1),
            (i_t * bT, 0 * E),
            (bT, E),
            (1, 0),
        )
        p_q1 = tl.make_block_ptr(
            q_addr + (bos * HQ + i_hq) * d_k,
            (T, d_k),
            (HQ * d_k, 1),
            (i_t * bT, 1 * E),
            (bT, E),
            (1, 0),
        )
        p_k0 = tl.make_block_ptr(
            k_addr + (bos * H + i_h) * d_k,
            (d_k, T),
            (1, H * d_k),
            (0 * E, i_s * bK),
            (E, bK),
            (0, 1),
        )
        p_k1 = tl.make_block_ptr(
            k_addr + (bos * H + i_h) * d_k,
            (d_k, T),
            (1, H * d_k),
            (1 * E, i_s * bK),
            (E, bK),
            (0, 1),
        )
        b_q0 = tl.load(p_q0, boundary_check=(0, 1))
        b_q1 = tl.load(p_q1, boundary_check=(0, 1))
        b_k0 = tl.load(p_k0, boundary_check=(0, 1))
        b_k1 = tl.load(p_k1, boundary_check=(0, 1))

        qk_00 = tl.dot(b_q0, b_k0) * scale
        qk_01 = tl.dot(b_q0, b_k1) * scale
        qk_10 = tl.dot(b_q1, b_k0) * scale
        qk_11 = tl.dot(b_q1, b_k1) * scale
        b_A = qk_00 * qk_00 + qk_01 * qk_01 + qk_10 * qk_10 + qk_11 * qk_11

        o_q = i_t * bT + tl.arange(0, bT)
        m_k = o_k < T
        causal = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
        b_P = tl.where(causal, b_A / b_D_safe[:, None], 0.0)
        b_dv += tl.dot(tl.trans(b_P.to(b_do.dtype)), b_do)

        p_v = tl.make_block_ptr(
            v_addr + (bos * H + i_h) * d_v,
            (d_v, T),
            (1, H * d_v),
            (0, i_s * bK),
            (bV, bK),
            (0, 1),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
        b_dN_V = tl.dot(b_do, b_v)
        b_dA = (b_dN_V - b_delta[:, None]) / b_D_safe[:, None]
        b_dA = tl.where(causal, b_dA, 0.0)

        # j=0: sum over i=0,1 of (2 * qk_i0 * dA)^T @ q_i  → b_dk0
        da = 2.0 * qk_00 * b_dA
        b_dk0 += tl.dot(tl.trans(da.to(b_q0.dtype)), b_q0) * scale
        da = 2.0 * qk_10 * b_dA
        b_dk0 += tl.dot(tl.trans(da.to(b_q1.dtype)), b_q1) * scale
        # j=1
        da = 2.0 * qk_01 * b_dA
        b_dk1 += tl.dot(tl.trans(da.to(b_q0.dtype)), b_q0) * scale
        da = 2.0 * qk_11 * b_dA
        b_dk1 += tl.dot(tl.trans(da.to(b_q1.dtype)), b_q1) * scale

    p_dk0 = tl.make_block_ptr(
        dk_addr + (bos * H + i_h) * d_k,
        (T, d_k),
        (H * d_k, 1),
        (i_s * bK, 0 * E),
        (bK, E),
        (1, 0),
    )
    p_dk1 = tl.make_block_ptr(
        dk_addr + (bos * H + i_h) * d_k,
        (T, d_k),
        (H * d_k, 1),
        (i_s * bK, 1 * E),
        (bK, E),
        (1, 0),
    )
    p_dv = tl.make_block_ptr(
        dv_addr + (bos * H + i_h) * d_v,
        (T, d_v),
        (H * d_v, 1),
        (i_s * bK, 0),
        (bK, bV),
        (1, 0),
    )
    tl.store(p_dk0, b_dk0.to(p_dk0.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk1, b_dk1.to(p_dk1.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


# =====================================================================
# Triton backward kernels for sum-SPD attention, M=2.
# 4 (i, j) pairs per (t, s) tile.
# =====================================================================


@triton.autotune(configs=BWD_CONFIGS, key=BWD_KEY_ME)
@triton.jit
def parallel_kata_attn_sum_bwd_kernel_dq_M2(
    q_addr,
    k_addr,
    v_addr,
    den_addr,
    delta_addr,
    do_addr,
    dq_addr,
    scale,
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    d_k: tl.constexpr,
    d_v: tl.constexpr,
    bV: tl.constexpr,
    E: tl.constexpr,
    bT: tl.constexpr,
    bK: tl.constexpr,
):
    i_t = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = i_b * T

    p_do = tl.make_block_ptr(
        do_addr + (bos * HQ + i_hq) * d_v,
        (T, d_v),
        (HQ * d_v, 1),
        (i_t * bT, 0),
        (bT, bV),
        (1, 0),
    )
    p_den = tl.make_block_ptr(
        den_addr + bos * HQ + i_hq, (T,), (HQ,), (i_t * bT,), (bT,), (0,)
    )
    p_delta = tl.make_block_ptr(
        delta_addr + bos * HQ + i_hq, (T,), (HQ,), (i_t * bT,), (bT,), (0,)
    )
    b_do = tl.load(p_do, boundary_check=(0, 1)).to(tl.float32)
    b_D = tl.load(p_den, boundary_check=(0,))
    b_delta = tl.load(p_delta, boundary_check=(0,))
    b_D_safe = tl.where(b_D > 0, b_D, 1.0)

    b_dq0 = tl.zeros([bT, E], dtype=tl.float32)
    b_dq1 = tl.zeros([bT, E], dtype=tl.float32)
    o_q = i_t * bT + tl.arange(0, bT)

    for i_s in range(0, min((i_t + 1) * bT, T), bK):
        p_q0 = tl.make_block_ptr(
            q_addr + (bos * HQ + i_hq) * d_k,
            (T, d_k),
            (HQ * d_k, 1),
            (i_t * bT, 0 * E),
            (bT, E),
            (1, 0),
        )
        p_q1 = tl.make_block_ptr(
            q_addr + (bos * HQ + i_hq) * d_k,
            (T, d_k),
            (HQ * d_k, 1),
            (i_t * bT, 1 * E),
            (bT, E),
            (1, 0),
        )
        p_k0 = tl.make_block_ptr(
            k_addr + (bos * H + i_h) * d_k,
            (d_k, T),
            (1, H * d_k),
            (0 * E, i_s),
            (E, bK),
            (0, 1),
        )
        p_k1 = tl.make_block_ptr(
            k_addr + (bos * H + i_h) * d_k,
            (d_k, T),
            (1, H * d_k),
            (1 * E, i_s),
            (E, bK),
            (0, 1),
        )
        b_q0 = tl.load(p_q0, boundary_check=(0, 1))
        b_q1 = tl.load(p_q1, boundary_check=(0, 1))
        b_k0 = tl.load(p_k0, boundary_check=(0, 1))
        b_k1 = tl.load(p_k1, boundary_check=(0, 1))

        p_v = tl.make_block_ptr(
            v_addr + (bos * H + i_h) * d_v,
            (d_v, T),
            (1, H * d_v),
            (0, i_s),
            (bV, bK),
            (0, 1),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
        b_dN_V = tl.dot(b_do, b_v)
        b_dA = (b_dN_V - b_delta[:, None]) / b_D_safe[:, None]
        o_k = i_s + tl.arange(0, bK)
        m_k = o_k < T
        causal = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
        b_dA = tl.where(causal, b_dA, 0.0)

        # i=0, j=0..1 → b_dq0
        qk = tl.dot(b_q0, b_k0) * scale
        da = 2.0 * qk * b_dA
        b_dq0 += tl.dot(da.to(b_k0.dtype), tl.trans(b_k0)) * scale
        qk = tl.dot(b_q0, b_k1) * scale
        da = 2.0 * qk * b_dA
        b_dq0 += tl.dot(da.to(b_k1.dtype), tl.trans(b_k1)) * scale
        # i=1
        qk = tl.dot(b_q1, b_k0) * scale
        da = 2.0 * qk * b_dA
        b_dq1 += tl.dot(da.to(b_k0.dtype), tl.trans(b_k0)) * scale
        qk = tl.dot(b_q1, b_k1) * scale
        da = 2.0 * qk * b_dA
        b_dq1 += tl.dot(da.to(b_k1.dtype), tl.trans(b_k1)) * scale

    p_dq0 = tl.make_block_ptr(
        dq_addr + (bos * HQ + i_hq) * d_k,
        (T, d_k),
        (HQ * d_k, 1),
        (i_t * bT, 0 * E),
        (bT, E),
        (1, 0),
    )
    p_dq1 = tl.make_block_ptr(
        dq_addr + (bos * HQ + i_hq) * d_k,
        (T, d_k),
        (HQ * d_k, 1),
        (i_t * bT, 1 * E),
        (bT, E),
        (1, 0),
    )
    tl.store(p_dq0, b_dq0.to(p_dq0.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dq1, b_dq1.to(p_dq1.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(configs=BWD_CONFIGS, key=BWD_KEY_ME)
@triton.jit
def parallel_kata_attn_sum_bwd_kernel_dkdv_M4(
    q_addr,
    k_addr,
    v_addr,
    den_addr,
    delta_addr,
    do_addr,
    dk_addr,
    dv_addr,
    scale,
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    d_k: tl.constexpr,
    d_v: tl.constexpr,
    bV: tl.constexpr,
    E: tl.constexpr,
    bT: tl.constexpr,
    bK: tl.constexpr,
):
    """Sum-SPD dK+dV kernel, M=4. Grid: (1, NS, B*HQ)."""
    i_s = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = i_b * T

    b_dk0 = tl.zeros([bK, E], dtype=tl.float32)
    b_dk1 = tl.zeros([bK, E], dtype=tl.float32)
    b_dk2 = tl.zeros([bK, E], dtype=tl.float32)
    b_dk3 = tl.zeros([bK, E], dtype=tl.float32)
    b_dv = tl.zeros([bK, bV], dtype=tl.float32)
    o_k = i_s * bK + tl.arange(0, bK)
    t_start = (i_s * bK) // bT
    NT = tl.cdiv(T, bT)

    for i_t in range(t_start, NT):
        p_do = tl.make_block_ptr(
            do_addr + (bos * HQ + i_hq) * d_v,
            (T, d_v),
            (HQ * d_v, 1),
            (i_t * bT, 0),
            (bT, bV),
            (1, 0),
        )
        p_den = tl.make_block_ptr(
            den_addr + bos * HQ + i_hq, (T,), (HQ,), (i_t * bT,), (bT,), (0,)
        )
        p_delta = tl.make_block_ptr(
            delta_addr + bos * HQ + i_hq, (T,), (HQ,), (i_t * bT,), (bT,), (0,)
        )
        b_do = tl.load(p_do, boundary_check=(0, 1)).to(tl.float32)
        b_D = tl.load(p_den, boundary_check=(0,))
        b_delta = tl.load(p_delta, boundary_check=(0,))
        b_D_safe = tl.where(b_D > 0, b_D, 1.0)

        p_q0 = tl.make_block_ptr(
            q_addr + (bos * HQ + i_hq) * d_k,
            (T, d_k),
            (HQ * d_k, 1),
            (i_t * bT, 0 * E),
            (bT, E),
            (1, 0),
        )
        p_q1 = tl.make_block_ptr(
            q_addr + (bos * HQ + i_hq) * d_k,
            (T, d_k),
            (HQ * d_k, 1),
            (i_t * bT, 1 * E),
            (bT, E),
            (1, 0),
        )
        p_q2 = tl.make_block_ptr(
            q_addr + (bos * HQ + i_hq) * d_k,
            (T, d_k),
            (HQ * d_k, 1),
            (i_t * bT, 2 * E),
            (bT, E),
            (1, 0),
        )
        p_q3 = tl.make_block_ptr(
            q_addr + (bos * HQ + i_hq) * d_k,
            (T, d_k),
            (HQ * d_k, 1),
            (i_t * bT, 3 * E),
            (bT, E),
            (1, 0),
        )
        p_k0 = tl.make_block_ptr(
            k_addr + (bos * H + i_h) * d_k,
            (d_k, T),
            (1, H * d_k),
            (0 * E, i_s * bK),
            (E, bK),
            (0, 1),
        )
        p_k1 = tl.make_block_ptr(
            k_addr + (bos * H + i_h) * d_k,
            (d_k, T),
            (1, H * d_k),
            (1 * E, i_s * bK),
            (E, bK),
            (0, 1),
        )
        p_k2 = tl.make_block_ptr(
            k_addr + (bos * H + i_h) * d_k,
            (d_k, T),
            (1, H * d_k),
            (2 * E, i_s * bK),
            (E, bK),
            (0, 1),
        )
        p_k3 = tl.make_block_ptr(
            k_addr + (bos * H + i_h) * d_k,
            (d_k, T),
            (1, H * d_k),
            (3 * E, i_s * bK),
            (E, bK),
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

        # A and dv_addr
        qk_00 = tl.dot(b_q0, b_k0) * scale
        qk_01 = tl.dot(b_q0, b_k1) * scale
        qk_02 = tl.dot(b_q0, b_k2) * scale
        qk_03 = tl.dot(b_q0, b_k3) * scale
        qk_10 = tl.dot(b_q1, b_k0) * scale
        qk_11 = tl.dot(b_q1, b_k1) * scale
        qk_12 = tl.dot(b_q1, b_k2) * scale
        qk_13 = tl.dot(b_q1, b_k3) * scale
        qk_20 = tl.dot(b_q2, b_k0) * scale
        qk_21 = tl.dot(b_q2, b_k1) * scale
        qk_22 = tl.dot(b_q2, b_k2) * scale
        qk_23 = tl.dot(b_q2, b_k3) * scale
        qk_30 = tl.dot(b_q3, b_k0) * scale
        qk_31 = tl.dot(b_q3, b_k1) * scale
        qk_32 = tl.dot(b_q3, b_k2) * scale
        qk_33 = tl.dot(b_q3, b_k3) * scale
        b_A = (
            qk_00 * qk_00
            + qk_01 * qk_01
            + qk_02 * qk_02
            + qk_03 * qk_03
            + qk_10 * qk_10
            + qk_11 * qk_11
            + qk_12 * qk_12
            + qk_13 * qk_13
            + qk_20 * qk_20
            + qk_21 * qk_21
            + qk_22 * qk_22
            + qk_23 * qk_23
            + qk_30 * qk_30
            + qk_31 * qk_31
            + qk_32 * qk_32
            + qk_33 * qk_33
        )

        o_q = i_t * bT + tl.arange(0, bT)
        m_k = o_k < T
        causal = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
        b_P = tl.where(causal, b_A / b_D_safe[:, None], 0.0)
        b_dv += tl.dot(tl.trans(b_P.to(b_do.dtype)), b_do)

        p_v = tl.make_block_ptr(
            v_addr + (bos * H + i_h) * d_v,
            (d_v, T),
            (1, H * d_v),
            (0, i_s * bK),
            (bV, bK),
            (0, 1),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
        b_dN_V = tl.dot(b_do, b_v)
        b_dA = (b_dN_V - b_delta[:, None]) / b_D_safe[:, None]
        b_dA = tl.where(causal, b_dA, 0.0)

        # dk_addr[j] += scale * sum_i (2 * qk_ij * dA)^T @ q_i
        # j=0: i=0..3
        da = 2.0 * qk_00 * b_dA
        b_dk0 += tl.dot(tl.trans(da.to(b_q0.dtype)), b_q0) * scale
        da = 2.0 * qk_10 * b_dA
        b_dk0 += tl.dot(tl.trans(da.to(b_q1.dtype)), b_q1) * scale
        da = 2.0 * qk_20 * b_dA
        b_dk0 += tl.dot(tl.trans(da.to(b_q2.dtype)), b_q2) * scale
        da = 2.0 * qk_30 * b_dA
        b_dk0 += tl.dot(tl.trans(da.to(b_q3.dtype)), b_q3) * scale
        # j=1
        da = 2.0 * qk_01 * b_dA
        b_dk1 += tl.dot(tl.trans(da.to(b_q0.dtype)), b_q0) * scale
        da = 2.0 * qk_11 * b_dA
        b_dk1 += tl.dot(tl.trans(da.to(b_q1.dtype)), b_q1) * scale
        da = 2.0 * qk_21 * b_dA
        b_dk1 += tl.dot(tl.trans(da.to(b_q2.dtype)), b_q2) * scale
        da = 2.0 * qk_31 * b_dA
        b_dk1 += tl.dot(tl.trans(da.to(b_q3.dtype)), b_q3) * scale
        # j=2
        da = 2.0 * qk_02 * b_dA
        b_dk2 += tl.dot(tl.trans(da.to(b_q0.dtype)), b_q0) * scale
        da = 2.0 * qk_12 * b_dA
        b_dk2 += tl.dot(tl.trans(da.to(b_q1.dtype)), b_q1) * scale
        da = 2.0 * qk_22 * b_dA
        b_dk2 += tl.dot(tl.trans(da.to(b_q2.dtype)), b_q2) * scale
        da = 2.0 * qk_32 * b_dA
        b_dk2 += tl.dot(tl.trans(da.to(b_q3.dtype)), b_q3) * scale
        # j=3
        da = 2.0 * qk_03 * b_dA
        b_dk3 += tl.dot(tl.trans(da.to(b_q0.dtype)), b_q0) * scale
        da = 2.0 * qk_13 * b_dA
        b_dk3 += tl.dot(tl.trans(da.to(b_q1.dtype)), b_q1) * scale
        da = 2.0 * qk_23 * b_dA
        b_dk3 += tl.dot(tl.trans(da.to(b_q2.dtype)), b_q2) * scale
        da = 2.0 * qk_33 * b_dA
        b_dk3 += tl.dot(tl.trans(da.to(b_q3.dtype)), b_q3) * scale

    p_dk0 = tl.make_block_ptr(
        dk_addr + (bos * H + i_h) * d_k,
        (T, d_k),
        (H * d_k, 1),
        (i_s * bK, 0 * E),
        (bK, E),
        (1, 0),
    )
    p_dk1 = tl.make_block_ptr(
        dk_addr + (bos * H + i_h) * d_k,
        (T, d_k),
        (H * d_k, 1),
        (i_s * bK, 1 * E),
        (bK, E),
        (1, 0),
    )
    p_dk2 = tl.make_block_ptr(
        dk_addr + (bos * H + i_h) * d_k,
        (T, d_k),
        (H * d_k, 1),
        (i_s * bK, 2 * E),
        (bK, E),
        (1, 0),
    )
    p_dk3 = tl.make_block_ptr(
        dk_addr + (bos * H + i_h) * d_k,
        (T, d_k),
        (H * d_k, 1),
        (i_s * bK, 3 * E),
        (bK, E),
        (1, 0),
    )
    p_dv = tl.make_block_ptr(
        dv_addr + (bos * H + i_h) * d_v,
        (T, d_v),
        (H * d_v, 1),
        (i_s * bK, 0),
        (bK, bV),
        (1, 0),
    )
    tl.store(p_dk0, b_dk0.to(p_dk0.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk1, b_dk1.to(p_dk1.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk2, b_dk2.to(p_dk2.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk3, b_dk3.to(p_dk3.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


# =====================================================================
# Triton backward kernels for sum-SPD attention, M=4 (production case).
# 16 (i, j) pairs per (t, s) tile: 4 dq_addr accumulators + 4 dk_addr accumulators
# + 1 dv_addr accumulator. Hardcoded M=4 unroll.
# =====================================================================


@triton.autotune(configs=BWD_CONFIGS, key=BWD_KEY_ME)
@triton.jit
def parallel_kata_attn_sum_bwd_kernel_dq_M4(
    q_addr,
    k_addr,
    v_addr,
    den_addr,
    delta_addr,
    do_addr,
    dq_addr,
    scale,
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    d_k: tl.constexpr,
    d_v: tl.constexpr,
    bV: tl.constexpr,
    E: tl.constexpr,
    bT: tl.constexpr,
    bK: tl.constexpr,
):
    """Sum-SPD dQ kernel, M=4. Grid: (1, NT, B*HQ)."""
    i_t = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = i_b * T

    p_do = tl.make_block_ptr(
        do_addr + (bos * HQ + i_hq) * d_v,
        (T, d_v),
        (HQ * d_v, 1),
        (i_t * bT, 0),
        (bT, bV),
        (1, 0),
    )
    p_den = tl.make_block_ptr(
        den_addr + bos * HQ + i_hq, (T,), (HQ,), (i_t * bT,), (bT,), (0,)
    )
    p_delta = tl.make_block_ptr(
        delta_addr + bos * HQ + i_hq, (T,), (HQ,), (i_t * bT,), (bT,), (0,)
    )
    b_do = tl.load(p_do, boundary_check=(0, 1)).to(tl.float32)
    b_D = tl.load(p_den, boundary_check=(0,))
    b_delta = tl.load(p_delta, boundary_check=(0,))
    b_D_safe = tl.where(b_D > 0, b_D, 1.0)

    b_dq0 = tl.zeros([bT, E], dtype=tl.float32)
    b_dq1 = tl.zeros([bT, E], dtype=tl.float32)
    b_dq2 = tl.zeros([bT, E], dtype=tl.float32)
    b_dq3 = tl.zeros([bT, E], dtype=tl.float32)
    o_q = i_t * bT + tl.arange(0, bT)

    for i_s in range(0, min((i_t + 1) * bT, T), bK):
        # Load Q groups (bT, E) and K groups (E, bK) for this chunk.
        p_q0 = tl.make_block_ptr(
            q_addr + (bos * HQ + i_hq) * d_k,
            (T, d_k),
            (HQ * d_k, 1),
            (i_t * bT, 0 * E),
            (bT, E),
            (1, 0),
        )
        p_q1 = tl.make_block_ptr(
            q_addr + (bos * HQ + i_hq) * d_k,
            (T, d_k),
            (HQ * d_k, 1),
            (i_t * bT, 1 * E),
            (bT, E),
            (1, 0),
        )
        p_q2 = tl.make_block_ptr(
            q_addr + (bos * HQ + i_hq) * d_k,
            (T, d_k),
            (HQ * d_k, 1),
            (i_t * bT, 2 * E),
            (bT, E),
            (1, 0),
        )
        p_q3 = tl.make_block_ptr(
            q_addr + (bos * HQ + i_hq) * d_k,
            (T, d_k),
            (HQ * d_k, 1),
            (i_t * bT, 3 * E),
            (bT, E),
            (1, 0),
        )
        p_k0 = tl.make_block_ptr(
            k_addr + (bos * H + i_h) * d_k,
            (d_k, T),
            (1, H * d_k),
            (0 * E, i_s),
            (E, bK),
            (0, 1),
        )
        p_k1 = tl.make_block_ptr(
            k_addr + (bos * H + i_h) * d_k,
            (d_k, T),
            (1, H * d_k),
            (1 * E, i_s),
            (E, bK),
            (0, 1),
        )
        p_k2 = tl.make_block_ptr(
            k_addr + (bos * H + i_h) * d_k,
            (d_k, T),
            (1, H * d_k),
            (2 * E, i_s),
            (E, bK),
            (0, 1),
        )
        p_k3 = tl.make_block_ptr(
            k_addr + (bos * H + i_h) * d_k,
            (d_k, T),
            (1, H * d_k),
            (3 * E, i_s),
            (E, bK),
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

        # dA from accumulated A is the same regardless of (i, j) — load V, dO once.
        p_v = tl.make_block_ptr(
            v_addr + (bos * H + i_h) * d_v,
            (d_v, T),
            (1, H * d_v),
            (0, i_s),
            (bV, bK),
            (0, 1),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
        b_dN_V = tl.dot(b_do, b_v)
        b_dA = (b_dN_V - b_delta[:, None]) / b_D_safe[:, None]
        o_k = i_s + tl.arange(0, bK)
        m_k = o_k < T
        causal = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
        b_dA = tl.where(causal, b_dA, 0.0)

        # Per (i, j): dq_addr[i] += scale * (2 * qk_ij * dA) @ k_j^T
        # i=0
        qk = tl.dot(b_q0, b_k0) * scale
        da = 2.0 * qk * b_dA
        b_dq0 += tl.dot(da.to(b_k0.dtype), tl.trans(b_k0)) * scale
        qk = tl.dot(b_q0, b_k1) * scale
        da = 2.0 * qk * b_dA
        b_dq0 += tl.dot(da.to(b_k1.dtype), tl.trans(b_k1)) * scale
        qk = tl.dot(b_q0, b_k2) * scale
        da = 2.0 * qk * b_dA
        b_dq0 += tl.dot(da.to(b_k2.dtype), tl.trans(b_k2)) * scale
        qk = tl.dot(b_q0, b_k3) * scale
        da = 2.0 * qk * b_dA
        b_dq0 += tl.dot(da.to(b_k3.dtype), tl.trans(b_k3)) * scale
        # i=1
        qk = tl.dot(b_q1, b_k0) * scale
        da = 2.0 * qk * b_dA
        b_dq1 += tl.dot(da.to(b_k0.dtype), tl.trans(b_k0)) * scale
        qk = tl.dot(b_q1, b_k1) * scale
        da = 2.0 * qk * b_dA
        b_dq1 += tl.dot(da.to(b_k1.dtype), tl.trans(b_k1)) * scale
        qk = tl.dot(b_q1, b_k2) * scale
        da = 2.0 * qk * b_dA
        b_dq1 += tl.dot(da.to(b_k2.dtype), tl.trans(b_k2)) * scale
        qk = tl.dot(b_q1, b_k3) * scale
        da = 2.0 * qk * b_dA
        b_dq1 += tl.dot(da.to(b_k3.dtype), tl.trans(b_k3)) * scale
        # i=2
        qk = tl.dot(b_q2, b_k0) * scale
        da = 2.0 * qk * b_dA
        b_dq2 += tl.dot(da.to(b_k0.dtype), tl.trans(b_k0)) * scale
        qk = tl.dot(b_q2, b_k1) * scale
        da = 2.0 * qk * b_dA
        b_dq2 += tl.dot(da.to(b_k1.dtype), tl.trans(b_k1)) * scale
        qk = tl.dot(b_q2, b_k2) * scale
        da = 2.0 * qk * b_dA
        b_dq2 += tl.dot(da.to(b_k2.dtype), tl.trans(b_k2)) * scale
        qk = tl.dot(b_q2, b_k3) * scale
        da = 2.0 * qk * b_dA
        b_dq2 += tl.dot(da.to(b_k3.dtype), tl.trans(b_k3)) * scale
        # i=3
        qk = tl.dot(b_q3, b_k0) * scale
        da = 2.0 * qk * b_dA
        b_dq3 += tl.dot(da.to(b_k0.dtype), tl.trans(b_k0)) * scale
        qk = tl.dot(b_q3, b_k1) * scale
        da = 2.0 * qk * b_dA
        b_dq3 += tl.dot(da.to(b_k1.dtype), tl.trans(b_k1)) * scale
        qk = tl.dot(b_q3, b_k2) * scale
        da = 2.0 * qk * b_dA
        b_dq3 += tl.dot(da.to(b_k2.dtype), tl.trans(b_k2)) * scale
        qk = tl.dot(b_q3, b_k3) * scale
        da = 2.0 * qk * b_dA
        b_dq3 += tl.dot(da.to(b_k3.dtype), tl.trans(b_k3)) * scale

    p_dq0 = tl.make_block_ptr(
        dq_addr + (bos * HQ + i_hq) * d_k,
        (T, d_k),
        (HQ * d_k, 1),
        (i_t * bT, 0 * E),
        (bT, E),
        (1, 0),
    )
    p_dq1 = tl.make_block_ptr(
        dq_addr + (bos * HQ + i_hq) * d_k,
        (T, d_k),
        (HQ * d_k, 1),
        (i_t * bT, 1 * E),
        (bT, E),
        (1, 0),
    )
    p_dq2 = tl.make_block_ptr(
        dq_addr + (bos * HQ + i_hq) * d_k,
        (T, d_k),
        (HQ * d_k, 1),
        (i_t * bT, 2 * E),
        (bT, E),
        (1, 0),
    )
    p_dq3 = tl.make_block_ptr(
        dq_addr + (bos * HQ + i_hq) * d_k,
        (T, d_k),
        (HQ * d_k, 1),
        (i_t * bT, 3 * E),
        (bT, E),
        (1, 0),
    )
    tl.store(p_dq0, b_dq0.to(p_dq0.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dq1, b_dq1.to(p_dq1.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dq2, b_dq2.to(p_dq2.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dq3, b_dq3.to(p_dq3.dtype.element_ty), boundary_check=(0, 1))
