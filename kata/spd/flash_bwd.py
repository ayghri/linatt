"""FlashAttention-style backward for SPD flash attention (no psi materialized).

FA recomputes the score in the backward (it doesn't store it); so does this.
Triton can't hold a list of per-group accumulators, and the two fusion routes
that would compute dP=do@v^T once both lost to this split on H100:
  - per-pair atomic_add (dq+dk)        -> 21.5ms  (atomic contention)
  - masked-wide (full-D) matmuls       -> 14.5ms  (M x wider matmuls)
This 3-kernel split puts the group in the GRID (one accumulator per program) and
recomputes dP per group; it was the fastest at 9.0ms (concat/M2, B=32, T=2048).

  dv : grid (key-block, B*H)        A=score, P=A/D, dv += P^T@do
  dk : grid (key-block, B*H, M)     group g -> dk_g accumulator
  dq : grid (query-block, B*H, M)   group g -> dq_g accumulator
dA[t,s]=(do[t].v[s]-delta[t])/D[t]; da=2*(scale*q_g.k_g)*dA;
dq_g=scale*da@k_g, dk_g=scale*da^T@q_g. CONCAT: g pairs with g; SUM: all i,j.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _dv_kernel(
    q,
    k,
    v,
    do,
    den,
    dv,
    scale,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    M: tl.constexpr,
    DV: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    MODE: tl.constexpr,
):
    i_s = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    bos_qk = i_bh * T * D
    bos_v = i_bh * T * DV
    o_k = i_s * BS + tl.arange(0, BS)
    sc2 = scale * scale
    b_dv = tl.zeros([BS, DV], dtype=tl.float32)
    for i_t in range((i_s * BS) // BT, tl.cdiv(T, BT)):
        b_A = tl.zeros([BT, BS], dtype=tl.float32)
        if MODE == 0:
            for g in range(M):
                qg = tl.load(
                    tl.make_block_ptr(
                        q + bos_qk, (T, D), (D, 1), (i_t * BT, g * E), (BT, E), (1, 0)
                    )
                )
                kg = tl.load(
                    tl.make_block_ptr(
                        k + bos_qk, (D, T), (1, D), (g * E, i_s * BS), (E, BS), (0, 1)
                    )
                )
                qk = tl.dot(qg, kg)
                b_A += qk * qk
        else:
            for gi in range(M):
                qgi = tl.load(
                    tl.make_block_ptr(
                        q + bos_qk, (T, D), (D, 1), (i_t * BT, gi * E), (BT, E), (1, 0)
                    )
                )
                for gj in range(M):
                    kgj = tl.load(
                        tl.make_block_ptr(
                            k + bos_qk,
                            (D, T),
                            (1, D),
                            (gj * E, i_s * BS),
                            (E, BS),
                            (0, 1),
                        )
                    )
                    qk = tl.dot(qgi, kgj)
                    b_A += qk * qk
        o_q = i_t * BT + tl.arange(0, BT)
        b_den = tl.load(
            tl.make_block_ptr(den + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,)),
            boundary_check=(0,),
        )
        b_den = tl.where(b_den > 0, b_den, 1.0)
        b_P = tl.where(
            (o_q[:, None] >= o_k[None, :]) & (o_k[None, :] < T),
            b_A * sc2 / b_den[:, None],
            0.0,
        )
        b_do = tl.load(
            tl.make_block_ptr(
                do + bos_v, (T, DV), (DV, 1), (i_t * BT, 0), (BT, DV), (1, 0)
            )
        )
        b_dv += tl.dot(tl.trans(b_P.to(b_do.dtype)), b_do)
    tl.store(
        tl.make_block_ptr(
            dv + bos_v, (T, DV), (DV, 1), (i_s * BS, 0), (BS, DV), (1, 0)
        ),
        b_dv.to(dv.dtype.element_ty),
    )


@triton.jit
def _dk_kernel(
    q,
    k,
    v,
    do,
    delta,
    den,
    dk,
    scale,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    M: tl.constexpr,
    DV: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    MODE: tl.constexpr,
):
    i_s = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    g = tl.program_id(2)
    bos_qk = i_bh * T * D
    bos_v = i_bh * T * DV
    o_k = i_s * BS + tl.arange(0, BS)
    b_dk = tl.zeros([BS, E], dtype=tl.float32)
    kg = tl.load(
        tl.make_block_ptr(
            k + bos_qk, (D, T), (1, D), (g * E, i_s * BS), (E, BS), (0, 1)
        )
    )
    for i_t in range((i_s * BS) // BT, tl.cdiv(T, BT)):
        b_do = tl.load(
            tl.make_block_ptr(
                do + bos_v, (T, DV), (DV, 1), (i_t * BT, 0), (BT, DV), (1, 0)
            )
        )
        b_v = tl.load(
            tl.make_block_ptr(
                v + bos_v, (T, DV), (DV, 1), (i_s * BS, 0), (BS, DV), (1, 0)
            )
        )
        b_del = tl.load(
            tl.make_block_ptr(delta + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,)),
            boundary_check=(0,),
        )
        b_den = tl.load(
            tl.make_block_ptr(den + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,)),
            boundary_check=(0,),
        )
        b_den = tl.where(b_den > 0, b_den, 1.0)
        dnv = tl.dot(b_do, tl.trans(b_v))
        o_q = i_t * BT + tl.arange(0, BT)
        dA = tl.where(
            (o_q[:, None] >= o_k[None, :]) & (o_k[None, :] < T),
            (dnv - b_del[:, None]) / b_den[:, None],
            0.0,
        )
        for gi in range(M):
            use = (gi == g) if MODE == 0 else True
            if use:
                qgi = tl.load(
                    tl.make_block_ptr(
                        q + bos_qk, (T, D), (D, 1), (i_t * BT, gi * E), (BT, E), (1, 0)
                    )
                )
                gij = tl.dot(qgi, kg) * scale
                da = 2.0 * gij * dA
                b_dk += tl.dot(tl.trans(da.to(qgi.dtype)), qgi) * scale
    tl.store(
        tl.make_block_ptr(
            dk + bos_qk, (T, D), (D, 1), (i_s * BS, g * E), (BS, E), (1, 0)
        ),
        b_dk.to(dk.dtype.element_ty),
    )


@triton.jit
def _dq_kernel(
    q,
    k,
    v,
    do,
    delta,
    den,
    dq,
    scale,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    M: tl.constexpr,
    DV: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    MODE: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    g = tl.program_id(2)
    bos_qk = i_bh * T * D
    bos_v = i_bh * T * DV
    o_q = i_t * BT + tl.arange(0, BT)
    b_do = tl.load(
        tl.make_block_ptr(do + bos_v, (T, DV), (DV, 1), (i_t * BT, 0), (BT, DV), (1, 0))
    )
    b_del = tl.load(
        tl.make_block_ptr(delta + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,)),
        boundary_check=(0,),
    )
    b_den = tl.load(
        tl.make_block_ptr(den + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,)),
        boundary_check=(0,),
    )
    b_den = tl.where(b_den > 0, b_den, 1.0)
    qg = tl.load(
        tl.make_block_ptr(
            q + bos_qk, (T, D), (D, 1), (i_t * BT, g * E), (BT, E), (1, 0)
        )
    )
    b_dq = tl.zeros([BT, E], dtype=tl.float32)
    for i_s in range(0, (i_t + 1) * BT, BS):
        b_v = tl.load(
            tl.make_block_ptr(v + bos_v, (T, DV), (DV, 1), (i_s, 0), (BS, DV), (1, 0))
        )
        dnv = tl.dot(b_do, tl.trans(b_v))
        o_k = i_s + tl.arange(0, BS)
        dA = tl.where(
            (o_q[:, None] >= o_k[None, :]) & (o_k[None, :] < T),
            (dnv - b_del[:, None]) / b_den[:, None],
            0.0,
        )
        for gj in range(M):
            use = (gj == g) if MODE == 0 else True
            if use:
                kgt = tl.load(
                    tl.make_block_ptr(
                        k + bos_qk, (D, T), (1, D), (gj * E, i_s), (E, BS), (0, 1)
                    )
                )
                kgn = tl.load(
                    tl.make_block_ptr(
                        k + bos_qk, (T, D), (D, 1), (i_s, gj * E), (BS, E), (1, 0)
                    )
                )
                gij = tl.dot(qg, kgt) * scale
                da = 2.0 * gij * dA
                b_dq += tl.dot(da.to(qg.dtype), kgn) * scale
    tl.store(
        tl.make_block_ptr(
            dq + bos_qk, (T, D), (D, 1), (i_t * BT, g * E), (BT, E), (1, 0)
        ),
        b_dq.to(dq.dtype.element_ty),
    )


@triton.jit
def _dkdv_concat(
    q,
    k,
    v,
    do,
    delta,
    den,
    dk,
    dv,
    scale,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    M: tl.constexpr,
    DV: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
):
    # CONCAT fused dk+dv, KV-block parallel, named per-group accumulators ->
    # dP=do@v^T computed ONCE per block-pair (no group-in-grid, no atomics).
    i_s = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    bo = i_bh * T * D
    bv = i_bh * T * DV
    o_k = i_s * BS + tl.arange(0, BS)
    b_dv = tl.zeros([BS, DV], dtype=tl.float32)
    dk0 = tl.zeros([BS, E], dtype=tl.float32)
    dk1 = tl.zeros([BS, E], dtype=tl.float32)
    dk2 = tl.zeros([BS, E], dtype=tl.float32)
    dk3 = tl.zeros([BS, E], dtype=tl.float32)
    k0t = tl.load(
        tl.make_block_ptr(k + bo, (D, T), (1, D), (0, i_s * BS), (E, BS), (0, 1))
    )
    if M >= 2:
        k1t = tl.load(
            tl.make_block_ptr(k + bo, (D, T), (1, D), (E, i_s * BS), (E, BS), (0, 1))
        )
    if M >= 4:
        k2t = tl.load(
            tl.make_block_ptr(
                k + bo, (D, T), (1, D), (2 * E, i_s * BS), (E, BS), (0, 1)
            )
        )
        k3t = tl.load(
            tl.make_block_ptr(
                k + bo, (D, T), (1, D), (3 * E, i_s * BS), (E, BS), (0, 1)
            )
        )

    for i_t in range((i_s * BS) // BT, tl.cdiv(T, BT)):
        b_do = tl.load(
            tl.make_block_ptr(
                do + bv, (T, DV), (DV, 1), (i_t * BT, 0), (BT, DV), (1, 0)
            )
        )
        b_v = tl.load(
            tl.make_block_ptr(v + bv, (T, DV), (DV, 1), (i_s * BS, 0), (BS, DV), (1, 0))
        )
        b_del = tl.load(
            tl.make_block_ptr(delta + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,)),
            boundary_check=(0,),
        )
        b_den = tl.load(
            tl.make_block_ptr(den + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,)),
            boundary_check=(0,),
        )
        b_den = tl.where(b_den > 0, b_den, 1.0)
        o_q = i_t * BT + tl.arange(0, BT)
        msk = (o_q[:, None] >= o_k[None, :]) & (o_k[None, :] < T)
        dnv = tl.dot(b_do, tl.trans(b_v))  # ONCE
        dA = tl.where(msk, (dnv - b_del[:, None]) / b_den[:, None], 0.0)
        b_A = tl.zeros([BT, BS], dtype=tl.float32)
        q0 = tl.load(
            tl.make_block_ptr(q + bo, (T, D), (D, 1), (i_t * BT, 0), (BT, E), (1, 0))
        )
        g0 = tl.dot(q0, k0t) * scale
        b_A += g0 * g0
        dk0 += tl.dot(tl.trans((2.0 * g0 * dA).to(b_do.dtype)), q0) * scale
        if M >= 2:
            q1 = tl.load(
                tl.make_block_ptr(
                    q + bo, (T, D), (D, 1), (i_t * BT, E), (BT, E), (1, 0)
                )
            )
            g1 = tl.dot(q1, k1t) * scale
            b_A += g1 * g1
            dk1 += tl.dot(tl.trans((2.0 * g1 * dA).to(b_do.dtype)), q1) * scale
        if M >= 4:
            q2 = tl.load(
                tl.make_block_ptr(
                    q + bo, (T, D), (D, 1), (i_t * BT, 2 * E), (BT, E), (1, 0)
                )
            )
            g2 = tl.dot(q2, k2t) * scale
            b_A += g2 * g2
            dk2 += tl.dot(tl.trans((2.0 * g2 * dA).to(b_do.dtype)), q2) * scale
            q3 = tl.load(
                tl.make_block_ptr(
                    q + bo, (T, D), (D, 1), (i_t * BT, 3 * E), (BT, E), (1, 0)
                )
            )
            g3 = tl.dot(q3, k3t) * scale
            b_A += g3 * g3
            dk3 += tl.dot(tl.trans((2.0 * g3 * dA).to(b_do.dtype)), q3) * scale
        b_P = tl.where(msk, b_A / b_den[:, None], 0.0)
        b_dv += tl.dot(tl.trans(b_P.to(b_do.dtype)), b_do)

    tl.store(
        tl.make_block_ptr(dk + bo, (T, D), (D, 1), (i_s * BS, 0), (BS, E), (1, 0)),
        dk0.to(dk.dtype.element_ty),
    )
    if M >= 2:
        tl.store(
            tl.make_block_ptr(dk + bo, (T, D), (D, 1), (i_s * BS, E), (BS, E), (1, 0)),
            dk1.to(dk.dtype.element_ty),
        )
    if M >= 4:
        tl.store(
            tl.make_block_ptr(
                dk + bo, (T, D), (D, 1), (i_s * BS, 2 * E), (BS, E), (1, 0)
            ),
            dk2.to(dk.dtype.element_ty),
        )
        tl.store(
            tl.make_block_ptr(
                dk + bo, (T, D), (D, 1), (i_s * BS, 3 * E), (BS, E), (1, 0)
            ),
            dk3.to(dk.dtype.element_ty),
        )
    tl.store(
        tl.make_block_ptr(dv + bv, (T, DV), (DV, 1), (i_s * BS, 0), (BS, DV), (1, 0)),
        b_dv.to(dv.dtype.element_ty),
    )


@triton.jit
def _dq_concat(
    q,
    k,
    v,
    do,
    delta,
    den,
    dq,
    scale,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    M: tl.constexpr,
    DV: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    bo = i_bh * T * D
    bv = i_bh * T * DV
    o_q = i_t * BT + tl.arange(0, BT)
    b_do = tl.load(
        tl.make_block_ptr(do + bv, (T, DV), (DV, 1), (i_t * BT, 0), (BT, DV), (1, 0))
    )
    b_del = tl.load(
        tl.make_block_ptr(delta + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,)),
        boundary_check=(0,),
    )
    b_den = tl.load(
        tl.make_block_ptr(den + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,)),
        boundary_check=(0,),
    )
    b_den = tl.where(b_den > 0, b_den, 1.0)
    q0 = tl.load(
        tl.make_block_ptr(q + bo, (T, D), (D, 1), (i_t * BT, 0), (BT, E), (1, 0))
    )
    if M >= 2:
        q1 = tl.load(
            tl.make_block_ptr(q + bo, (T, D), (D, 1), (i_t * BT, E), (BT, E), (1, 0))
        )
    if M >= 4:
        q2 = tl.load(
            tl.make_block_ptr(
                q + bo, (T, D), (D, 1), (i_t * BT, 2 * E), (BT, E), (1, 0)
            )
        )
        q3 = tl.load(
            tl.make_block_ptr(
                q + bo, (T, D), (D, 1), (i_t * BT, 3 * E), (BT, E), (1, 0)
            )
        )
    dq0 = tl.zeros([BT, E], dtype=tl.float32)
    dq1 = tl.zeros([BT, E], dtype=tl.float32)
    dq2 = tl.zeros([BT, E], dtype=tl.float32)
    dq3 = tl.zeros([BT, E], dtype=tl.float32)

    for i_s in range(0, (i_t + 1) * BT, BS):
        b_v = tl.load(
            tl.make_block_ptr(v + bv, (T, DV), (DV, 1), (i_s, 0), (BS, DV), (1, 0))
        )
        dnv = tl.dot(b_do, tl.trans(b_v))  # ONCE
        o_k = i_s + tl.arange(0, BS)
        dA = tl.where(
            (o_q[:, None] >= o_k[None, :]) & (o_k[None, :] < T),
            (dnv - b_del[:, None]) / b_den[:, None],
            0.0,
        )
        k0t = tl.load(
            tl.make_block_ptr(k + bo, (D, T), (1, D), (0, i_s), (E, BS), (0, 1))
        )
        k0n = tl.load(
            tl.make_block_ptr(k + bo, (T, D), (D, 1), (i_s, 0), (BS, E), (1, 0))
        )
        g0 = tl.dot(q0, k0t) * scale
        dq0 += tl.dot((2.0 * g0 * dA).to(b_do.dtype), k0n) * scale
        if M >= 2:
            k1t = tl.load(
                tl.make_block_ptr(k + bo, (D, T), (1, D), (E, i_s), (E, BS), (0, 1))
            )
            k1n = tl.load(
                tl.make_block_ptr(k + bo, (T, D), (D, 1), (i_s, E), (BS, E), (1, 0))
            )
            g1 = tl.dot(q1, k1t) * scale
            dq1 += tl.dot((2.0 * g1 * dA).to(b_do.dtype), k1n) * scale
        if M >= 4:
            k2t = tl.load(
                tl.make_block_ptr(k + bo, (D, T), (1, D), (2 * E, i_s), (E, BS), (0, 1))
            )
            k2n = tl.load(
                tl.make_block_ptr(k + bo, (T, D), (D, 1), (i_s, 2 * E), (BS, E), (1, 0))
            )
            g2 = tl.dot(q2, k2t) * scale
            dq2 += tl.dot((2.0 * g2 * dA).to(b_do.dtype), k2n) * scale
            k3t = tl.load(
                tl.make_block_ptr(k + bo, (D, T), (1, D), (3 * E, i_s), (E, BS), (0, 1))
            )
            k3n = tl.load(
                tl.make_block_ptr(k + bo, (T, D), (D, 1), (i_s, 3 * E), (BS, E), (1, 0))
            )
            g3 = tl.dot(q3, k3t) * scale
            dq3 += tl.dot((2.0 * g3 * dA).to(b_do.dtype), k3n) * scale

    tl.store(
        tl.make_block_ptr(dq + bo, (T, D), (D, 1), (i_t * BT, 0), (BT, E), (1, 0)),
        dq0.to(dq.dtype.element_ty),
    )
    if M >= 2:
        tl.store(
            tl.make_block_ptr(dq + bo, (T, D), (D, 1), (i_t * BT, E), (BT, E), (1, 0)),
            dq1.to(dq.dtype.element_ty),
        )
    if M >= 4:
        tl.store(
            tl.make_block_ptr(
                dq + bo, (T, D), (D, 1), (i_t * BT, 2 * E), (BT, E), (1, 0)
            ),
            dq2.to(dq.dtype.element_ty),
        )
        tl.store(
            tl.make_block_ptr(
                dq + bo, (T, D), (D, 1), (i_t * BT, 3 * E), (BT, E), (1, 0)
            ),
            dq3.to(dq.dtype.element_ty),
        )


@triton.jit
def _dkdv_sum(
    q,
    k,
    v,
    do,
    delta,
    den,
    dk,
    dv,
    scale,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    M: tl.constexpr,
    DV: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
):
    # SUM fused dk+dv via group-folded tensor-core dots (no M^2 small dots):
    # G=(q rows folded to BT*M, E) @ (k rows folded to BS*M, E)^T -> (BT*M, BS*M);
    # A=sum over the two group axes of G^2; DA_folded = 2*G*dA; dk = DA_folded^T @ q_r.
    i_s = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    bo = i_bh * T * D
    bv = i_bh * T * DV
    o_k = i_s * BS + tl.arange(0, BS)
    b_dv = tl.zeros([BS, DV], dtype=tl.float32)
    b_dk = tl.zeros([BS, D], dtype=tl.float32)
    b_k = tl.load(
        tl.make_block_ptr(k + bo, (T, D), (D, 1), (i_s * BS, 0), (BS, D), (1, 0))
    )
    k_r = tl.reshape(b_k, (BS * M, E))  # rows (s,j)
    for i_t in range((i_s * BS) // BT, tl.cdiv(T, BT)):
        b_do = tl.load(
            tl.make_block_ptr(
                do + bv, (T, DV), (DV, 1), (i_t * BT, 0), (BT, DV), (1, 0)
            )
        )
        b_v = tl.load(
            tl.make_block_ptr(v + bv, (T, DV), (DV, 1), (i_s * BS, 0), (BS, DV), (1, 0))
        )
        b_del = tl.load(
            tl.make_block_ptr(delta + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,)),
            boundary_check=(0,),
        )
        b_den = tl.load(
            tl.make_block_ptr(den + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,)),
            boundary_check=(0,),
        )
        b_den = tl.where(b_den > 0, b_den, 1.0)
        o_q = i_t * BT + tl.arange(0, BT)
        msk = (o_q[:, None] >= o_k[None, :]) & (o_k[None, :] < T)
        dnv = tl.dot(b_do, tl.trans(b_v))  # (BT,BS) ONCE
        dA = tl.where(msk, (dnv - b_del[:, None]) / b_den[:, None], 0.0)
        b_q = tl.load(
            tl.make_block_ptr(q + bo, (T, D), (D, 1), (i_t * BT, 0), (BT, D), (1, 0))
        )
        q_r = tl.reshape(b_q, (BT * M, E))  # rows (t,i)
        G = tl.dot(q_r, tl.trans(k_r)) * scale  # (BT*M, BS*M) = scale*(q_i.k_j)
        G4 = tl.reshape(G, (BT, M, BS, M))
        A = tl.sum(tl.sum(G4 * G4, axis=3), axis=1)  # (BT,BS) score
        b_P = tl.where(msk, A / b_den[:, None], 0.0)
        b_dv += tl.dot(tl.trans(b_P.to(b_do.dtype)), b_do)
        dA_bc = tl.reshape(
            tl.broadcast_to(dA[:, None, :, None], (BT, M, BS, M)), (BT * M, BS * M)
        )
        DAf = (2.0 * G * dA_bc).to(b_do.dtype)  # (BT*M, BS*M)
        b_dk += tl.reshape(tl.dot(tl.trans(DAf), q_r) * scale, (BS, D))
    tl.store(
        tl.make_block_ptr(dk + bo, (T, D), (D, 1), (i_s * BS, 0), (BS, D), (1, 0)),
        b_dk.to(dk.dtype.element_ty),
    )
    tl.store(
        tl.make_block_ptr(dv + bv, (T, DV), (DV, 1), (i_s * BS, 0), (BS, DV), (1, 0)),
        b_dv.to(dv.dtype.element_ty),
    )


@triton.jit
def _dq_sum(
    q,
    k,
    v,
    do,
    delta,
    den,
    dq,
    scale,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    M: tl.constexpr,
    DV: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    bo = i_bh * T * D
    bv = i_bh * T * DV
    o_q = i_t * BT + tl.arange(0, BT)
    b_do = tl.load(
        tl.make_block_ptr(do + bv, (T, DV), (DV, 1), (i_t * BT, 0), (BT, DV), (1, 0))
    )
    b_del = tl.load(
        tl.make_block_ptr(delta + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,)),
        boundary_check=(0,),
    )
    b_den = tl.load(
        tl.make_block_ptr(den + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,)),
        boundary_check=(0,),
    )
    b_den = tl.where(b_den > 0, b_den, 1.0)
    b_q = tl.load(
        tl.make_block_ptr(q + bo, (T, D), (D, 1), (i_t * BT, 0), (BT, D), (1, 0))
    )
    q_r = tl.reshape(b_q, (BT * M, E))
    b_dq = tl.zeros([BT, D], dtype=tl.float32)
    for i_s in range(0, (i_t + 1) * BT, BS):
        b_v = tl.load(
            tl.make_block_ptr(v + bv, (T, DV), (DV, 1), (i_s, 0), (BS, DV), (1, 0))
        )
        dnv = tl.dot(b_do, tl.trans(b_v))  # ONCE
        o_k = i_s + tl.arange(0, BS)
        dA = tl.where(
            (o_q[:, None] >= o_k[None, :]) & (o_k[None, :] < T),
            (dnv - b_del[:, None]) / b_den[:, None],
            0.0,
        )
        b_k = tl.load(
            tl.make_block_ptr(k + bo, (T, D), (D, 1), (i_s, 0), (BS, D), (1, 0))
        )
        k_r = tl.reshape(b_k, (BS * M, E))
        G = tl.dot(q_r, tl.trans(k_r)) * scale  # (BT*M, BS*M)
        dA_bc = tl.reshape(
            tl.broadcast_to(dA[:, None, :, None], (BT, M, BS, M)), (BT * M, BS * M)
        )
        DAf = (2.0 * G * dA_bc).to(b_do.dtype)
        b_dq += tl.reshape(tl.dot(DAf, k_r) * scale, (BT, D))
    tl.store(
        tl.make_block_ptr(dq + bo, (T, D), (D, 1), (i_t * BT, 0), (BT, D), (1, 0)),
        b_dq.to(dq.dtype.element_ty),
    )


def spd_flash_bwd(q, k, v, o, den, do, M, mode, scale, BT=64, BS=64):
    B, H, T, D = q.shape
    DV = v.shape[-1]
    E = D // M
    MODE = 0 if mode == "concat" else 1
    do = do.contiguous()
    delta = (o.float() * do.float()).sum(-1).contiguous()
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    kw = dict(H=H, D=D, E=E, M=M, DV=DV, BT=BT, BS=BS, num_warps=4, num_stages=2)
    if MODE == 0:  # CONCAT: fused named-accumulator kernels (dP once, no atomics)
        _dkdv_concat[(triton.cdiv(T, BS), B * H)](
            q, k, v, do, delta, den, dk, dv, scale, T, **kw
        )
        _dq_concat[(triton.cdiv(T, BT), B * H)](
            q, k, v, do, delta, den, dq, scale, T, **kw
        )
    else:  # SUM: group-folded tensor-core dots (one big dot, no M^2)
        BL = 64 // M  # folded G is BL*M x BL*M = 64x64
        while T % BL != 0:
            BL //= 2
        kw["BT"] = BL
        kw["BS"] = BL  # num_stages stays 2
        _dkdv_sum[(triton.cdiv(T, BL), B * H)](
            q, k, v, do, delta, den, dk, dv, scale, T, **kw
        )
        _dq_sum[(triton.cdiv(T, BL), B * H)](
            q, k, v, do, delta, den, dq, scale, T, **kw
        )
    return dq, dk, dv
