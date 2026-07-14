import triton
import triton.language as tl


from .utils import FWD_CONFIGS, BWD_CONFIGS, FWD_KEY, BWD_KEY_M1, BWD_KEY_ME


@triton.autotune(configs=FWD_CONFIGS, key=FWD_KEY)
@triton.jit
def parallel_kata_attn_fwd_kernel(
    q_addr,  # (B, T, HQ, d_k)
    k_addr,  # (B, T, H, d_k)
    v_addr,  # (B, T, H, d_v)
    o_addr,  # (B, T, HQ, d_v)
    den_addr,  # (B, T, HQ)
    c_addr,  # (B, T, HQ) decay logits
    scale_addr,  # scores like
    T,
    d_k: tl.constexpr,  # head_k dim (= M * E)
    d_v: tl.constexpr,  # head_v dim
    H: tl.constexpr,
    M: tl.constexpr,  # SPD num_groups
    HQ: tl.constexpr,
    G: tl.constexpr,  # = HQ // H (GQA grouping)
    bT: tl.constexpr,  # query-block rows  (autotuned)
    bV: tl.constexpr,  # block on d_v (autotune)
    bK: tl.constexpr,  # key-block rows  (autotuned)
    HAS_DECAY: tl.constexpr,  # multiply score by exp(c_t - c_s), GLA-style decay
    VARLEN: tl.constexpr,
):
    # per-group head dim (E dropped from the sig; it is d_k // M)
    E: tl.constexpr = d_k // M
    idx_v = tl.program_id(0)
    idx_t = tl.program_id(1)
    idx_bh = tl.program_id(2)
    idx_b, idx_hq = idx_bh // HQ, idx_bh % HQ
    idx_h = idx_hq // G

    bos = idx_b * T
    offs_q = idx_t * bT + tl.arange(0, bT)

    ptr_o = tl.make_block_ptr(
        o_addr + (bos * HQ + idx_hq) * d_v,
        (T, d_v),
        (HQ * d_v, 1),
        (idx_t * bT, idx_v * bV),
        (bT, bV),
        (1, 0),
    )

    # (idx_t * bT, idx_v * bV),
    # ptr_o = tl.make_tensor_descriptor(
    #     o_addr + (bos * HQ + idx_hq) * d_v,
    #     shape=[T, d_v],
    #     strides=[HQ * d_v, 1],
    #     block_shape=[bT, bV],
    # )

    ptr_den = tl.make_block_ptr(
        den_addr + bos * HQ + idx_hq,
        (T,),
        (HQ,),
        (idx_t * bT,),
        (bT,),
        (0,),
    )

    ptr_v = tl.make_block_ptr(
        v_addr + (bos * H + idx_h) * d_v,
        (T, d_v),
        (H * d_v, 1),
        (0, idx_v * bV),
        (bK, bV),
        (1, 0),
    )
    ptr_q_g = tl.make_block_ptr(
        q_addr + (bos * HQ + idx_hq) * d_k,
        (T, d_k),
        (HQ * d_k, 1),
        (idx_t * bT, 0),
        (bT, E),
        (1, 0),
    )
    ptr_k_g = tl.make_block_ptr(
        k_addr + (bos * H + idx_h) * d_k,
        (d_k, T),
        (1, H * d_k),
        (0, 0),
        (E, bK),
        (0, 1),
    )

    b_o = tl.zeros([bT, bV], dtype=tl.float32)
    b_acc = tl.zeros([bT], dtype=tl.float32)

    if HAS_DECAY:
        ptr_cq = tl.make_block_ptr(
            c_addr + bos * HQ + idx_hq,
            (T,),
            (HQ,),
            (idx_t * bT,),
            (bT,),
            (0,),
        )
        b_cq = tl.load(ptr_cq, boundary_check=(0,))
        ptr_decay = tl.make_block_ptr(
            c_addr + bos * HQ + idx_hq,
            (T,),
            (HQ,),
            (0,),
            (bK,),
            (0,),
        )

    # Phase 1: strictly earlier key blocks (no causal mask)
    for idx_s in range(0, idx_t * bT, bK):
        b_scores = tl.zeros([bT, bK], dtype=tl.float32)
        for m_id in tl.static_range(M):
            q_g = tl.load(tl.advance(ptr_q_g, (0, m_id * E)), boundary_check=(0, 1))
            k_g = tl.load(tl.advance(ptr_k_g, (m_id * E, 0)), boundary_check=(0, 1))
            qk_g = tl.dot(q_g, k_g) * scale_addr
            b_scores += qk_g * qk_g
        ptr_k_g = tl.advance(ptr_k_g, (0, bK))

        offs_k = idx_s + tl.arange(0, bK)
        b_scores = tl.where((offs_k < T)[None, :], b_scores, 0.0)

        if HAS_DECAY:
            b_decay = tl.load(ptr_decay, boundary_check=(0,))
            b_scores = b_scores * tl.exp(
                tl.minimum(b_cq[:, None] - b_decay[None, :], 0.0)
            )
            ptr_decay = tl.advance(ptr_decay, (bK,))

        b_v = tl.load(ptr_v, boundary_check=(0, 1))
        ptr_v = tl.advance(ptr_v, (bK, 0))
        b_acc += tl.sum(b_scores, axis=1)
        b_o += tl.dot(b_scores.to(b_v.dtype), b_v)

    # Phase 2: on-diagonal blocks (causal mask) -- fresh ptrs
    for idx_s in range(idx_t * bT, min((idx_t + 1) * bT, T), bK):
        b_scores = tl.zeros([bT, bK], dtype=tl.float32)
        for m_id in tl.static_range(M):
            p_q_g = tl.make_block_ptr(
                q_addr + (bos * HQ + idx_hq) * d_k,
                (T, d_k),
                (HQ * d_k, 1),
                (idx_t * bT, m_id * E),
                (bT, E),
                (1, 0),
            )
            p_k_g = tl.make_block_ptr(
                k_addr + (bos * H + idx_h) * d_k,
                (d_k, T),
                (1, H * d_k),
                (m_id * E, idx_s),
                (E, bK),
                (0, 1),
            )
            b_q_g = tl.load(p_q_g, boundary_check=(0, 1))
            b_k_g = tl.load(p_k_g, boundary_check=(0, 1))
            qk_g = tl.dot(b_q_g, b_k_g) * scale_addr
            b_scores += qk_g * qk_g

        offs_k = idx_s + tl.arange(0, bK)
        m_s = (offs_q[:, None] >= offs_k[None, :]) & (offs_k < T)[None, :]
        b_scores = tl.where(m_s, b_scores, 0.0)

        if HAS_DECAY:
            p_cs = tl.make_block_ptr(
                c_addr + bos * HQ + idx_hq,
                (T,),
                (HQ,),
                (idx_s,),
                (bK,),
                (0,),
            )
            b_cs = tl.load(p_cs, boundary_check=(0,))
            b_scores = b_scores * tl.exp(tl.minimum(b_cq[:, None] - b_cs[None, :], 0.0))

        p_v = tl.make_block_ptr(
            v_addr + (bos * H + idx_h) * d_v,
            (T, d_v),
            (H * d_v, 1),
            (idx_s, idx_v * bV),
            (bK, bV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_acc += tl.sum(b_scores, axis=1)
        b_o += tl.dot(b_scores.to(b_v.dtype), b_v)

    # Normalize
    b_acc_safe = tl.where(b_acc > 0, b_acc, 1.0)
    b_o = b_o / b_acc_safe[:, None]
    tl.store(ptr_o, b_o.to(ptr_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(ptr_den, b_acc.to(ptr_den.dtype.element_ty), boundary_check=(0,))


@triton.autotune(configs=FWD_CONFIGS, key=FWD_KEY)
@triton.jit
def parallel_kata_attn_sum_fwd_kernel(
    q_addr,
    k_addr,
    v_addr,
    o_addr,
    den_addr,
    scale,
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    d_k: tl.constexpr,
    d_v: tl.constexpr,
    bV: tl.constexpr,
    M: tl.constexpr,
    E: tl.constexpr,
    bT: tl.constexpr,
    bK: tl.constexpr,
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
        o_addr + (bos * HQ + i_hq) * d_v,
        (T, d_v),
        (HQ * d_v, 1),
        (i_t * bT, i_v * bV),
        (bT, bV),
        (1, 0),
    )
    p_den = tl.make_block_ptr(
        den_addr + bos * HQ + i_hq,
        (T,),
        (HQ,),
        (i_t * bT,),
        (bT,),
        (0,),
    )

    b_o = tl.zeros([bT, bV], dtype=tl.float32)
    b_acc = tl.zeros([bT], dtype=tl.float32)
    o_q = i_t * bT + tl.arange(0, bT)

    # Phase 1: strictly earlier key blocks
    for i_s in range(0, i_t * bT, bK):
        b_s = tl.zeros([bT, bK], dtype=tl.float32)
        for i in tl.static_range(M):
            p_q_i = tl.make_block_ptr(
                q_addr + (bos * HQ + i_hq) * d_k,
                (T, d_k),
                (HQ * d_k, 1),
                (i_t * bT, i * E),
                (bT, E),
                (1, 0),
            )
            b_q_i = tl.load(p_q_i, boundary_check=(0, 1))
            for j in tl.static_range(M):
                p_k_j = tl.make_block_ptr(
                    k_addr + (bos * H + i_h) * d_k,
                    (d_k, T),
                    (1, H * d_k),
                    (j * E, i_s),
                    (E, bK),
                    (0, 1),
                )
                b_k_j = tl.load(p_k_j, boundary_check=(0, 1))
                qk_ij = tl.dot(b_q_i, b_k_j) * scale
                b_s += qk_ij * qk_ij

        o_k = i_s + tl.arange(0, bK)
        m_k = o_k < T
        b_s = tl.where(m_k[None, :], b_s, 0.0)

        p_v = tl.make_block_ptr(
            v_addr + (bos * H + i_h) * d_v,
            (T, d_v),
            (H * d_v, 1),
            (i_s, i_v * bV),
            (bK, bV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_acc += tl.sum(b_s, axis=1)
        b_o += tl.dot(b_s.to(b_v.dtype), b_v)

    # Phase 2: on-diagonal block (causal)
    for i_s in range(i_t * bT, min((i_t + 1) * bT, T), bK):
        b_s = tl.zeros([bT, bK], dtype=tl.float32)
        for i in tl.static_range(M):
            p_q_i = tl.make_block_ptr(
                q_addr + (bos * HQ + i_hq) * d_k,
                (T, d_k),
                (HQ * d_k, 1),
                (i_t * bT, i * E),
                (bT, E),
                (1, 0),
            )
            b_q_i = tl.load(p_q_i, boundary_check=(0, 1))
            for j in tl.static_range(M):
                p_k_j = tl.make_block_ptr(
                    k_addr + (bos * H + i_h) * d_k,
                    (d_k, T),
                    (1, H * d_k),
                    (j * E, i_s),
                    (E, bK),
                    (0, 1),
                )
                b_k_j = tl.load(p_k_j, boundary_check=(0, 1))
                qk_ij = tl.dot(b_q_i, b_k_j) * scale
                b_s += qk_ij * qk_ij

        o_k = i_s + tl.arange(0, bK)
        m_k = o_k < T
        m_s = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
        b_s = tl.where(m_s, b_s, 0.0)

        p_v = tl.make_block_ptr(
            v_addr + (bos * H + i_h) * d_v,
            (T, d_v),
            (H * d_v, 1),
            (i_s, i_v * bV),
            (bK, bV),
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
    o_addr,
    do_addr,
    delta_addr,
    NEL: tl.constexpr,
    V: tl.constexpr,
):
    """delta_addr[i] = <o_addr[i,:], do_addr[i,:]>  for each row i  (grid: total rows)."""
    i_n = tl.program_id(0)
    o_d = tl.arange(0, NEL)
    m_d = o_d < V
    b_o = tl.load(o_addr + i_n * V + o_d, mask=m_d, other=0).to(tl.float32)
    b_do = tl.load(do_addr + i_n * V + o_d, mask=m_d, other=0).to(tl.float32)
    tl.store(delta_addr + i_n, tl.sum(b_o * b_do).to(delta_addr.dtype.element_ty))


@triton.autotune(configs=BWD_CONFIGS, key=BWD_KEY_ME)
@triton.jit
def parallel_kata_attn_bwd_kernel_dq_M4(
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
    bT: tl.constexpr,
    bK: tl.constexpr,
    bV: tl.constexpr,
    E: tl.constexpr,
):
    """dQ for kata-quadratic attention, M=4. Grid: (1, NT, B*HQ)."""
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
        den_addr + bos * HQ + i_hq,
        (T,),
        (HQ,),
        (i_t * bT,),
        (bT,),
        (0,),
    )
    p_delta = tl.make_block_ptr(
        delta_addr + bos * HQ + i_hq,
        (T,),
        (HQ,),
        (i_t * bT,),
        (bT,),
        (0,),
    )
    b_do = tl.load(p_do, boundary_check=(0, 1)).to(tl.float32)
    b_D = tl.load(p_den, boundary_check=(0,))
    b_delta = tl.load(p_delta, boundary_check=(0,))
    b_D_safe = tl.where(b_D > 0, b_D, 1.0)

    # M=4 per-group dq_addr accumulators
    b_dq0 = tl.zeros([bT, E], dtype=tl.float32)
    b_dq1 = tl.zeros([bT, E], dtype=tl.float32)
    b_dq2 = tl.zeros([bT, E], dtype=tl.float32)
    b_dq3 = tl.zeros([bT, E], dtype=tl.float32)

    o_q = i_t * bT + tl.arange(0, bT)

    for i_s in range(0, min((i_t + 1) * bT, T), bK):
        # Per-group Q (bT, E) and K (E, bK).
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

        qk0 = tl.dot(b_q0, b_k0) * scale
        qk1 = tl.dot(b_q1, b_k1) * scale
        qk2 = tl.dot(b_q2, b_k2) * scale
        qk3 = tl.dot(b_q3, b_k3) * scale

        # b_A not needed in dq_addr kernel (we use dA directly).

        p_v = tl.make_block_ptr(
            v_addr + (bos * H + i_h) * d_v,
            (d_v, T),
            (1, H * d_v),
            (0, i_s),
            (bV, bK),
            (0, 1),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
        b_dN_V = tl.dot(b_do, b_v)  # (bT, bK)
        b_dA = (b_dN_V - b_delta[:, None]) / b_D_safe[:, None]

        o_k = i_s + tl.arange(0, bK)
        m_k = o_k < T
        causal = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
        b_dA = tl.where(causal, b_dA, 0.0)

        da0 = 2.0 * qk0 * b_dA
        da1 = 2.0 * qk1 * b_dA
        da2 = 2.0 * qk2 * b_dA
        da3 = 2.0 * qk3 * b_dA

        # b_k_g is (E, bK); transpose to (bK, E) for the dot
        b_dq0 += tl.dot(da0.to(b_k0.dtype), tl.trans(b_k0)) * scale
        b_dq1 += tl.dot(da1.to(b_k1.dtype), tl.trans(b_k1)) * scale
        b_dq2 += tl.dot(da2.to(b_k2.dtype), tl.trans(b_k2)) * scale
        b_dq3 += tl.dot(da3.to(b_k3.dtype), tl.trans(b_k3)) * scale

    # Store the four (bT, E) slices of dQ.
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


@triton.autotune(configs=BWD_CONFIGS, key=BWD_KEY_ME)
@triton.jit
def parallel_kata_attn_bwd_kernel_dkdv_M4(
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
    G: tl.constexpr,  # assumed = 1 (MHA); for GQA, post-sum in python
    d_k: tl.constexpr,
    d_v: tl.constexpr,
    bT: tl.constexpr,
    bK: tl.constexpr,
    bV: tl.constexpr,
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

        # Per-group Q (bT, E) and K (E, bK).
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

        qk0 = tl.dot(b_q0, b_k0) * scale
        qk1 = tl.dot(b_q1, b_k1) * scale
        qk2 = tl.dot(b_q2, b_k2) * scale
        qk3 = tl.dot(b_q3, b_k3) * scale
        b_A = qk0 * qk0 + qk1 * qk1 + qk2 * qk2 + qk3 * qk3

        o_q = i_t * bT + tl.arange(0, bT)
        m_k = o_k < T
        causal = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]

        # dV first.  P[t,s] = A[t,s] / D[t]
        b_P = tl.where(causal, b_A / b_D_safe[:, None], 0.0)
        b_dv += tl.dot(tl.trans(b_P.to(b_do.dtype)), b_do)

        # dA = (<dO, V> - delta_addr) / D
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

        # da per group, then dk_g += scale * sum_t da[t,s] * q_g[t,:]  ⇒  scale * da^T @ q_g (bK, E)
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
# M=1 backward kernels (degenerate: d_k == E, no grouping)
# =====================================================================


@triton.autotune(configs=BWD_CONFIGS, key=BWD_KEY_M1)
@triton.jit
def parallel_kata_attn_bwd_kernel_dq_M1(
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
    bT: tl.constexpr,
    bK: tl.constexpr,
    bV: tl.constexpr,
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
    b_do = tl.load(
        p_do, boundary_check=(0, 1)
    )  # keep bf16 for the do_addr.v_addr tensor-core dot
    b_D = tl.load(p_den, boundary_check=(0,))
    b_delta = tl.load(p_delta, boundary_check=(0,))
    b_D_safe = tl.where(b_D > 0, b_D, 1.0)

    b_dq = tl.zeros([bT, d_k], dtype=tl.float32)
    o_q = i_t * bT + tl.arange(0, bT)
    # q_addr-block is loop-invariant -> load ONCE, not per key-block.
    p_q = tl.make_block_ptr(
        q_addr + (bos * HQ + i_hq) * d_k,
        (T, d_k),
        (HQ * d_k, 1),
        (i_t * bT, 0),
        (bT, d_k),
        (1, 0),
    )
    b_q = tl.load(p_q, boundary_check=(0, 1))

    # persistent key/value ptrs advanced by bK per key-block (tl.advance is functional)
    p_k = tl.make_block_ptr(
        k_addr + (bos * H + i_h) * d_k,
        (d_k, T),
        (1, H * d_k),
        (0, 0),
        (d_k, bK),
        (0, 1),
    )
    p_v = tl.make_block_ptr(
        v_addr + (bos * H + i_h) * d_v,
        (d_v, T),
        (1, H * d_v),
        (0, 0),
        (bV, bK),
        (0, 1),
    )
    for i_s in range(0, min((i_t + 1) * bT, T), bK):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        qk = tl.dot(b_q, b_k) * scale

        b_v = tl.load(
            p_v, boundary_check=(0, 1)
        )  # bf16 -> tensor-core do_addr.v_addr (fp32 accum)
        b_dN_V = tl.dot(b_do, b_v)
        b_dA = (b_dN_V - b_delta[:, None]) / b_D_safe[:, None]
        o_k = i_s + tl.arange(0, bK)
        m_k = o_k < T
        causal = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
        b_dA = tl.where(causal, b_dA, 0.0)

        da = 2.0 * qk * b_dA
        b_dq += tl.dot(da.to(b_k.dtype), tl.trans(b_k)) * scale
        p_k = tl.advance(p_k, (0, bK))
        p_v = tl.advance(p_v, (0, bK))

    p_dq = tl.make_block_ptr(
        dq_addr + (bos * HQ + i_hq) * d_k,
        (T, d_k),
        (HQ * d_k, 1),
        (i_t * bT, 0),
        (bT, d_k),
        (1, 0),
    )
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(configs=BWD_CONFIGS, key=BWD_KEY_M1)
@triton.jit
def parallel_kata_attn_bwd_kernel_dkdv_M1(
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
    bT: tl.constexpr,
    bK: tl.constexpr,
    bV: tl.constexpr,
):
    i_s = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    bos = i_b * T

    b_dk = tl.zeros([bK, d_k], dtype=tl.float32)
    b_dv = tl.zeros([bK, bV], dtype=tl.float32)
    o_k = i_s * bK + tl.arange(0, bK)
    t_start = (i_s * bK) // bT
    NT = tl.cdiv(T, bT)
    # k_addr,v_addr blocks are loop-invariant (fixed i_s) -> load ONCE, bf16 for tensor-core dots.
    p_k = tl.make_block_ptr(
        k_addr + (bos * H + i_h) * d_k,
        (d_k, T),
        (1, H * d_k),
        (0, i_s * bK),
        (d_k, bK),
        (0, 1),
    )
    p_v = tl.make_block_ptr(
        v_addr + (bos * H + i_h) * d_v,
        (d_v, T),
        (1, H * d_v),
        (0, i_s * bK),
        (bV, bK),
        (0, 1),
    )
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_v = tl.load(p_v, boundary_check=(0, 1))

    # persistent query-side ptrs advanced by bT per query-block (start at t_start*bT)
    p_do = tl.make_block_ptr(
        do_addr + (bos * HQ + i_hq) * d_v,
        (T, d_v),
        (HQ * d_v, 1),
        (t_start * bT, 0),
        (bT, bV),
        (1, 0),
    )
    p_den = tl.make_block_ptr(
        den_addr + bos * HQ + i_hq, (T,), (HQ,), (t_start * bT,), (bT,), (0,)
    )
    p_delta = tl.make_block_ptr(
        delta_addr + bos * HQ + i_hq, (T,), (HQ,), (t_start * bT,), (bT,), (0,)
    )
    p_q = tl.make_block_ptr(
        q_addr + (bos * HQ + i_hq) * d_k,
        (T, d_k),
        (HQ * d_k, 1),
        (t_start * bT, 0),
        (bT, d_k),
        (1, 0),
    )
    for i_t in range(t_start, NT):
        b_do = tl.load(
            p_do, boundary_check=(0, 1)
        )  # bf16 for the dv_addr / do_addr.v_addr tensor-core dots
        b_D = tl.load(p_den, boundary_check=(0,))
        b_delta = tl.load(p_delta, boundary_check=(0,))
        b_D_safe = tl.where(b_D > 0, b_D, 1.0)

        b_q = tl.load(p_q, boundary_check=(0, 1))
        qk = tl.dot(b_q, b_k) * scale
        b_A = qk * qk

        o_q = i_t * bT + tl.arange(0, bT)
        m_k = o_k < T
        causal = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
        b_P = tl.where(causal, b_A / b_D_safe[:, None], 0.0)
        b_dv += tl.dot(tl.trans(b_P.to(b_do.dtype)), b_do)

        b_dN_V = tl.dot(b_do, b_v)
        b_dA = (b_dN_V - b_delta[:, None]) / b_D_safe[:, None]
        b_dA = tl.where(causal, b_dA, 0.0)
        da = 2.0 * qk * b_dA
        b_dk += tl.dot(tl.trans(da.to(b_q.dtype)), b_q) * scale
        p_do = tl.advance(p_do, (bT, 0))
        p_den = tl.advance(p_den, (bT,))
        p_delta = tl.advance(p_delta, (bT,))
        p_q = tl.advance(p_q, (bT, 0))

    p_dk = tl.make_block_ptr(
        dk_addr + (bos * H + i_h) * d_k,
        (T, d_k),
        (H * d_k, 1),
        (i_s * bK, 0),
        (bK, d_k),
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
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


# =====================================================================
# M=2 backward kernels (two per-group accumulators)
# =====================================================================


@triton.autotune(configs=BWD_CONFIGS, key=BWD_KEY_ME)
@triton.jit
def parallel_kata_attn_bwd_kernel_dq_M2(
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
    bT: tl.constexpr,
    bK: tl.constexpr,
    bV: tl.constexpr,
    E: tl.constexpr,
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

    # q_addr-groups are loop-invariant (fixed i_t) -> load ONCE, not per key-block.
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
    b_q0 = tl.load(p_q0, boundary_check=(0, 1))
    b_q1 = tl.load(p_q1, boundary_check=(0, 1))
    # persistent key/value ptrs advanced by bK per key-block
    p_k0 = tl.make_block_ptr(
        k_addr + (bos * H + i_h) * d_k,
        (d_k, T),
        (1, H * d_k),
        (0 * E, 0),
        (E, bK),
        (0, 1),
    )
    p_k1 = tl.make_block_ptr(
        k_addr + (bos * H + i_h) * d_k,
        (d_k, T),
        (1, H * d_k),
        (1 * E, 0),
        (E, bK),
        (0, 1),
    )
    p_v = tl.make_block_ptr(
        v_addr + (bos * H + i_h) * d_v,
        (d_v, T),
        (1, H * d_v),
        (0, 0),
        (bV, bK),
        (0, 1),
    )
    for i_s in range(0, min((i_t + 1) * bT, T), bK):
        b_k0 = tl.load(p_k0, boundary_check=(0, 1))
        b_k1 = tl.load(p_k1, boundary_check=(0, 1))
        qk0 = tl.dot(b_q0, b_k0) * scale
        qk1 = tl.dot(b_q1, b_k1) * scale

        b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
        b_dN_V = tl.dot(b_do, b_v)
        b_dA = (b_dN_V - b_delta[:, None]) / b_D_safe[:, None]
        o_k = i_s + tl.arange(0, bK)
        m_k = o_k < T
        causal = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
        b_dA = tl.where(causal, b_dA, 0.0)
        da0 = 2.0 * qk0 * b_dA
        da1 = 2.0 * qk1 * b_dA
        b_dq0 += tl.dot(da0.to(b_k0.dtype), tl.trans(b_k0)) * scale
        b_dq1 += tl.dot(da1.to(b_k1.dtype), tl.trans(b_k1)) * scale
        p_k0 = tl.advance(p_k0, (0, bK))
        p_k1 = tl.advance(p_k1, (0, bK))
        p_v = tl.advance(p_v, (0, bK))

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
def parallel_kata_attn_bwd_kernel_dkdv_M2(
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
    bT: tl.constexpr,
    bK: tl.constexpr,
    bV: tl.constexpr,
    E: tl.constexpr,
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

    # k_addr,v_addr blocks are loop-invariant (fixed i_s) -> load ONCE.
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
    p_v = tl.make_block_ptr(
        v_addr + (bos * H + i_h) * d_v,
        (d_v, T),
        (1, H * d_v),
        (0, i_s * bK),
        (bV, bK),
        (0, 1),
    )
    b_k0 = tl.load(p_k0, boundary_check=(0, 1))
    b_k1 = tl.load(p_k1, boundary_check=(0, 1))
    b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)

    # persistent query-side ptrs advanced by bT per query-block (start at t_start*bT)
    p_do = tl.make_block_ptr(
        do_addr + (bos * HQ + i_hq) * d_v,
        (T, d_v),
        (HQ * d_v, 1),
        (t_start * bT, 0),
        (bT, bV),
        (1, 0),
    )
    p_den = tl.make_block_ptr(
        den_addr + bos * HQ + i_hq, (T,), (HQ,), (t_start * bT,), (bT,), (0,)
    )
    p_delta = tl.make_block_ptr(
        delta_addr + bos * HQ + i_hq, (T,), (HQ,), (t_start * bT,), (bT,), (0,)
    )
    p_q0 = tl.make_block_ptr(
        q_addr + (bos * HQ + i_hq) * d_k,
        (T, d_k),
        (HQ * d_k, 1),
        (t_start * bT, 0 * E),
        (bT, E),
        (1, 0),
    )
    p_q1 = tl.make_block_ptr(
        q_addr + (bos * HQ + i_hq) * d_k,
        (T, d_k),
        (HQ * d_k, 1),
        (t_start * bT, 1 * E),
        (bT, E),
        (1, 0),
    )
    for i_t in range(t_start, NT):
        b_do = tl.load(p_do, boundary_check=(0, 1)).to(tl.float32)
        b_D = tl.load(p_den, boundary_check=(0,))
        b_delta = tl.load(p_delta, boundary_check=(0,))
        b_D_safe = tl.where(b_D > 0, b_D, 1.0)

        b_q0 = tl.load(p_q0, boundary_check=(0, 1))
        b_q1 = tl.load(p_q1, boundary_check=(0, 1))
        qk0 = tl.dot(b_q0, b_k0) * scale
        qk1 = tl.dot(b_q1, b_k1) * scale
        b_A = qk0 * qk0 + qk1 * qk1

        o_q = i_t * bT + tl.arange(0, bT)
        m_k = o_k < T
        causal = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
        b_P = tl.where(causal, b_A / b_D_safe[:, None], 0.0)
        b_dv += tl.dot(tl.trans(b_P.to(b_do.dtype)), b_do)

        b_dN_V = tl.dot(b_do, b_v)
        b_dA = (b_dN_V - b_delta[:, None]) / b_D_safe[:, None]
        b_dA = tl.where(causal, b_dA, 0.0)
        da0 = 2.0 * qk0 * b_dA
        da1 = 2.0 * qk1 * b_dA
        b_dk0 += tl.dot(tl.trans(da0.to(b_q0.dtype)), b_q0) * scale
        b_dk1 += tl.dot(tl.trans(da1.to(b_q1.dtype)), b_q1) * scale
        p_do = tl.advance(p_do, (bT, 0))
        p_den = tl.advance(p_den, (bT,))
        p_delta = tl.advance(p_delta, (bT,))
        p_q0 = tl.advance(p_q0, (bT, 0))
        p_q1 = tl.advance(p_q1, (bT, 0))

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
