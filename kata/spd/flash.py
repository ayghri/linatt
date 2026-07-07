"""Triton flash-like (parallel) forward kernel for KATA-SPD attention.

FLA parallel-attention structure: one program per (query-block, batch-head);
loop over causal key blocks, build the SPD score block A[BT,BS], causal-mask the
diagonal block, accumulate num += A @ V and den += rowsum(A); finally o = num/den.

Score block, with q,k split into M groups of width E = d_head/M (scale per dot):
  CONCAT (MODE=0): A = sum_i  (scale * q_i @ k_i^T)^2
  SUM    (MODE=1): A = sum_ij (scale * q_i @ k_j^T)^2

Validated against kata.spd_attn.spd_parallel_ref. Forward only (bwd is a follow-up).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# autotune sweep: block sizes / warps / stages. key on shapes that change the
# best config so each (T, D, DV, E, MODE) is tuned once and cached.
# Single Hopper-safe config (dq/dkdv backward). Larger tiles/warps faulted on
# sm90 in autotune (illegal memory access), same as the forward sweep.
_CONFIGS = [
    triton.Config({"BT": 64, "BS": 64}, num_warps=4, num_stages=2),
]
_KEY = ["T", "D", "DV", "E", "MODE"]


@triton.jit
def _spd_fwd_kernel(
    q,
    k,
    v,
    o,
    den,
    scale,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    M: tl.constexpr,
    DV: tl.constexpr,
    BLK: tl.constexpr,
    MODE: tl.constexpr,
    BOUND: tl.constexpr,
):
    # FlashAttention-style: one query block per program; sweep key blocks
    # 0..diagonal accumulating num += A@V, den += rowsum(A). SPD is sum-normalized
    # (no online max / exp). Assumes T % BLK == 0; no boundary checks (D, DV, T
    # block-aligned). BOUND: optional per-row running-max rescale to keep the
    # accumulator bounded for long T (costs a max+rescale per block).
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    bos_qk = i_bh * T * D
    bos_v = i_bh * T * DV
    o_q = i_t * BLK + tl.arange(0, BLK)

    sc2 = scale * scale
    b_num = tl.zeros([BLK, DV], dtype=tl.float32)
    b_den = tl.zeros([BLK], dtype=tl.float32)
    b_c = tl.zeros([BLK], dtype=tl.float32)  # per-row running max of A

    for i_s in range(0, (i_t + 1) * BLK, BLK):
        # per-group 2D MMAs: Triton's tuned tensor-core path. Both the 3D batched
        # tl.dot (~6x slower) and the group-folded single 2D dot + 4D reshape/sum
        # (~1.7x slower) were measured worse on H100, so the group loop stays.
        b_A = tl.zeros([BLK, BLK], dtype=tl.float32)
        if MODE == 0:  # CONCAT: same-group dots
            for g in range(M):
                b_qg = tl.load(
                    tl.make_block_ptr(
                        q + bos_qk, (T, D), (D, 1), (i_t * BLK, g * E), (BLK, E), (1, 0)
                    )
                )
                b_kg = tl.load(
                    tl.make_block_ptr(
                        k + bos_qk, (D, T), (1, D), (g * E, i_s), (E, BLK), (0, 1)
                    )
                )
                qk = tl.dot(b_qg, b_kg)
                b_A += qk * qk
        else:  # SUM: all cross-group pairs
            for gi in range(M):
                b_qi = tl.load(
                    tl.make_block_ptr(
                        q + bos_qk,
                        (T, D),
                        (D, 1),
                        (i_t * BLK, gi * E),
                        (BLK, E),
                        (1, 0),
                    )
                )
                for gj in range(M):
                    b_kj = tl.load(
                        tl.make_block_ptr(
                            k + bos_qk, (D, T), (1, D), (gj * E, i_s), (E, BLK), (0, 1)
                        )
                    )
                    qk = tl.dot(b_qi, b_kj)
                    b_A += qk * qk
        b_A = b_A * sc2
        # causal where (identity for full blocks below diagonal; masks diagonal)
        o_k = i_s + tl.arange(0, BLK)
        b_A = tl.where(o_q[:, None] >= o_k[None, :], b_A, 0.0)

        b_v = tl.load(
            tl.make_block_ptr(v + bos_v, (T, DV), (DV, 1), (i_s, 0), (BLK, DV), (1, 0))
        )
        if BOUND:
            # num,den held divided by running max b_c; o=num/den invariant.
            new_c = tl.maximum(tl.maximum(b_c, tl.max(b_A, axis=1)), 1e-12)
            rs = b_c / new_c
            b_num *= rs[:, None]
            b_den *= rs
            b_A = b_A / new_c[:, None]  # entries in [0, 1]
            b_c = new_c
        b_num += tl.dot(b_A.to(b_v.dtype), b_v)
        b_den += tl.sum(b_A, axis=1)

    b_o = b_num / tl.maximum(b_den, 1e-12)[:, None]  # b_c cancels
    b_den = b_den * b_c if BOUND else b_den  # true row-sum for bwd
    tl.store(
        tl.make_block_ptr(
            o + bos_v, (T, DV), (DV, 1), (i_t * BLK, 0), (BLK, DV), (1, 0)
        ),
        b_o.to(o.dtype.element_ty),
    )
    tl.store(
        tl.make_block_ptr(den + i_bh * T, (T,), (1,), (i_t * BLK,), (BLK,), (0,)),
        b_den.to(den.dtype.element_ty),
    )


def spd_attn_parallel_fwd(
    q,
    k,
    v,
    M: int,
    mode: str,
    scale: float | None = None,
    BLK: int | None = None,
    bound: bool = True,
):
    """q,k: (B,H,T,D)  v: (B,H,T,DV) -> (o, den). Forward only (flash-like).

    bound: per-row running-max rescale (keeps the accumulator bounded for long T;
    small cost). Safe to disable for short T where overflow can't occur.
    """
    B, H, T, D = q.shape
    DV = v.shape[-1]
    E = D // M
    if scale is None:
        scale = 1.0 / E
    q, k, v = (x.contiguous() for x in (q, k, v))
    o = torch.empty(B, H, T, DV, device=q.device, dtype=q.dtype)
    den = torch.empty(B, H, T, device=q.device, dtype=torch.float32)
    if BLK is None:
        BLK = 64
    while T % BLK != 0:
        BLK //= 2
    _spd_fwd_kernel[(triton.cdiv(T, BLK), B * H)](
        q,
        k,
        v,
        o,
        den,
        scale,
        T,
        H=H,
        D=D,
        E=E,
        M=M,
        DV=DV,
        BLK=BLK,
        MODE=0 if mode == "concat" else 1,
        BOUND=bound,
        num_warps=4,
        num_stages=2,
    )
    return o, den


@triton.autotune(configs=_CONFIGS, key=_KEY)
@triton.jit
def _spd_dq_kernel(
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
    bos_qk = i_bh * T * D
    bos_v = i_bh * T * DV
    o_q = i_t * BT + tl.arange(0, BT)

    p_do = tl.make_block_ptr(
        do + bos_v, (T, DV), (DV, 1), (i_t * BT, 0), (BT, DV), (1, 0)
    )
    b_do = tl.load(p_do, boundary_check=(0, 1))
    p_del = tl.make_block_ptr(delta + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    b_del = tl.load(p_del, boundary_check=(0,))
    p_den = tl.make_block_ptr(den + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    b_den = tl.load(p_den, boundary_check=(0,))
    b_den = tl.where(b_den > 0, b_den, 1.0)

    # one query group i at a time -> independent (BT, E) accumulator -> store.
    for gi in range(M):
        b_dq = tl.zeros([BT, E], dtype=tl.float32)
        for i_s in range(0, (i_t + 1) * BT, BS):
            p_v = tl.make_block_ptr(
                v + bos_v, (T, DV), (DV, 1), (i_s, 0), (BS, DV), (1, 0)
            )
            b_v = tl.load(p_v, boundary_check=(0, 1))
            dnv = tl.dot(b_do, tl.trans(b_v))  # (BT, BS) = <do, v>
            dA = (dnv - b_del[:, None]) / b_den[:, None]
            o_k = i_s + tl.arange(0, BS)
            dA = tl.where((o_q[:, None] >= o_k[None, :]) & (o_k[None, :] < T), dA, 0.0)

            p_qi = tl.make_block_ptr(
                q + bos_qk, (T, D), (D, 1), (i_t * BT, gi * E), (BT, E), (1, 0)
            )
            b_qi = tl.load(p_qi, boundary_check=(0, 1))
            # CONCAT: only group gi pairs with key-group gi. SUM: gi pairs with all gj.
            j0 = gi if MODE == 0 else 0
            j1 = gi + 1 if MODE == 0 else M
            for gj in range(j0, j1):
                p_kt = tl.make_block_ptr(
                    k + bos_qk, (D, T), (1, D), (gj * E, i_s), (E, BS), (0, 1)
                )
                p_kn = tl.make_block_ptr(
                    k + bos_qk, (T, D), (D, 1), (i_s, gj * E), (BS, E), (1, 0)
                )
                qk = (
                    tl.dot(b_qi, tl.load(p_kt, boundary_check=(0, 1))) * scale
                )  # g_ij (BT,BS)
                da = 2.0 * qk * dA
                b_dq += (
                    tl.dot(da.to(b_qi.dtype), tl.load(p_kn, boundary_check=(0, 1)))
                    * scale
                )
        p_dq = tl.make_block_ptr(
            dq + bos_qk, (T, D), (D, 1), (i_t * BT, gi * E), (BT, E), (1, 0)
        )
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(configs=_CONFIGS, key=_KEY)
@triton.jit
def _spd_dkdv_kernel(
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
    MODE: tl.constexpr,
):
    i_s = tl.program_id(0)  # key block
    i_bh = tl.program_id(1).to(tl.int64)
    bos_qk = i_bh * T * D
    bos_v = i_bh * T * DV
    o_k = i_s * BS + tl.arange(0, BS)

    b_dv = tl.zeros([BS, DV], dtype=tl.float32)
    t_start = (i_s * BS) // BT  # query blocks t >= key block (causal)
    NT = tl.cdiv(T, BT)

    for gj in range(M):  # key group gj -> independent (BS, E) dk accumulator
        b_dk = tl.zeros([BS, E], dtype=tl.float32)
        for i_t in range(t_start, NT):
            p_do = tl.make_block_ptr(
                do + bos_v, (T, DV), (DV, 1), (i_t * BT, 0), (BT, DV), (1, 0)
            )
            b_do = tl.load(p_do, boundary_check=(0, 1))
            p_del = tl.make_block_ptr(
                delta + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,)
            )
            b_del = tl.load(p_del, boundary_check=(0,))
            p_den = tl.make_block_ptr(
                den + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,)
            )
            b_den = tl.where(
                tl.load(p_den, boundary_check=(0,)) > 0,
                tl.load(p_den, boundary_check=(0,)),
                1.0,
            )
            p_v = tl.make_block_ptr(
                v + bos_v, (T, DV), (DV, 1), (i_s * BS, 0), (BS, DV), (1, 0)
            )
            b_v = tl.load(p_v, boundary_check=(0, 1))
            dnv = tl.dot(b_do, tl.trans(b_v))  # (BT, BS)
            dA = (dnv - b_del[:, None]) / b_den[:, None]
            o_q = i_t * BT + tl.arange(0, BT)
            causal = (o_q[:, None] >= o_k[None, :]) & (o_k[None, :] < T)
            dA = tl.where(causal, dA, 0.0)

            p_kt = tl.make_block_ptr(
                k + bos_qk, (D, T), (1, D), (gj * E, i_s * BS), (E, BS), (0, 1)
            )
            b_ktj = tl.load(p_kt, boundary_check=(0, 1))  # (E, BS)
            i0 = gj if MODE == 0 else 0
            i1 = gj + 1 if MODE == 0 else M
            for gi in range(i0, i1):
                p_qi = tl.make_block_ptr(
                    q + bos_qk, (T, D), (D, 1), (i_t * BT, gi * E), (BT, E), (1, 0)
                )
                b_qi = tl.load(p_qi, boundary_check=(0, 1))  # (BT, E)
                qk = tl.dot(b_qi, b_ktj) * scale  # g_ij (BT, BS)
                da = 2.0 * qk * dA
                b_dk += tl.dot(tl.trans(da.to(b_qi.dtype)), b_qi) * scale  # (BS, E)
            # dv: accumulate once (use the last group's pass; A recomputed below)
            if gj == 0:
                # recompute full A for P = A/D
                b_A = tl.zeros([BT, BS], dtype=tl.float32)
                if MODE == 0:
                    for g in range(M):
                        p_q = tl.make_block_ptr(
                            q + bos_qk,
                            (T, D),
                            (D, 1),
                            (i_t * BT, g * E),
                            (BT, E),
                            (1, 0),
                        )
                        p_kk = tl.make_block_ptr(
                            k + bos_qk,
                            (D, T),
                            (1, D),
                            (g * E, i_s * BS),
                            (E, BS),
                            (0, 1),
                        )
                        s = (
                            tl.dot(
                                tl.load(p_q, boundary_check=(0, 1)),
                                tl.load(p_kk, boundary_check=(0, 1)),
                            )
                            * scale
                        )
                        b_A += s * s
                else:
                    for gi2 in range(M):
                        p_q = tl.make_block_ptr(
                            q + bos_qk,
                            (T, D),
                            (D, 1),
                            (i_t * BT, gi2 * E),
                            (BT, E),
                            (1, 0),
                        )
                        b_q2 = tl.load(p_q, boundary_check=(0, 1))
                        for gj2 in range(M):
                            p_kk = tl.make_block_ptr(
                                k + bos_qk,
                                (D, T),
                                (1, D),
                                (gj2 * E, i_s * BS),
                                (E, BS),
                                (0, 1),
                            )
                            s = (
                                tl.dot(b_q2, tl.load(p_kk, boundary_check=(0, 1)))
                                * scale
                            )
                            b_A += s * s
                b_P = tl.where(causal, b_A / b_den[:, None], 0.0)
                b_dv += tl.dot(tl.trans(b_P.to(b_do.dtype)), b_do)
        p_dk = tl.make_block_ptr(
            dk + bos_qk, (T, D), (D, 1), (i_s * BS, gj * E), (BS, E), (1, 0)
        )
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    p_dv = tl.make_block_ptr(
        dv + bos_v, (T, DV), (DV, 1), (i_s * BS, 0), (BS, DV), (1, 0)
    )
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


class SPDParallelAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, M, mode, scale):
        o, den = spd_attn_parallel_fwd(q, k, v, M, mode, scale)
        ctx.save_for_backward(q, k, v, o, den)
        ctx.M, ctx.mode, ctx.scale = M, mode, scale
        return o

    @staticmethod
    def backward(ctx, do):
        from .flash_bwd import spd_flash_bwd
        q, k, v, o, den = ctx.saved_tensors
        M, mode, scale = ctx.M, ctx.mode, ctx.scale
        if scale is None:
            scale = 1.0 / (q.shape[-1] // M)
        dq, dk, dv = spd_flash_bwd(q, k, v, o, den, do, M, mode, scale)
        return dq, dk, dv, None, None, None


def spd_attn_parallel(q, k, v, M, mode, scale=None):
    """Differentiable flash-like SPD attention (fwd + bwd Triton kernels)."""
    return SPDParallelAttn.apply(q, k, v, M, mode, scale)
