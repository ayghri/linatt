"""Tree-scan (parallel-prefix) chunked SPD forward -- CONCAT mode (cat-M).

Concat kernel:  kappa(q,k) = sum_g (q_g . k_g)^2 over M groups of width E=d/M.
Unlike sum mode (chunk_scan.py: one E^2 state aggregating all M groups -> all-pairs
sum_gh (q_g.k_h)^2), concat keeps NG=M *independent* E^2 sub-states and sums the
per-group (num_g, den_g) at readout -- the numerator/denominator sum over groups, so
o = (sum_g num_g) / (sum_g den_g).

Same associative-scan structure as chunk_scan.py:
  phase 1 (NC*BH programs): per chunk, per group g: h_c[g]=psi(K_g)^T V, z_c[g]=sum psi(K_g),
           + intra-chunk causal A = sum_g psi(Q_g) psi(K_g)^T -> (n_intra, d_intra).
  phase 2 (prefix scan, ~log NC): exclusive cumsum of the (M*E^2, Dv) states h, z.
  phase 3 (NC*BH programs): o = (sum_g psi(Q_g) @ H_c[g] + intra) / (sum_g psi(Q_g).Z_c[g] + d_intra).

Rationale vs the sequential linear chunk: the sequential scan has only BH programs and
O(NC) depth; the tree-scan exposes NC*BH programs at ~log NC depth. At M=2 the per-group
E^2=1024 state is small enough that the prefix-scan bandwidth does not eat the parallelism
win, so on H100 this should beat flash_delta_state at long context.
"""

import torch
import triton
import triton.language as tl

from kata.spd.chunk_scan import (
    _excl_scan_kernel,
    _scan_blocked_A,
    _scan_blocked_B,
)


@triton.jit
def _scan_local_cat_kernel(
    q_addr,
    k_addr,
    v_addr,
    h,
    z,
    num_i,
    den_i,
    s_scale,
    T,
    # H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    M: tl.constexpr,
    P: tl.constexpr,
    DV: tl.constexpr,
    C: tl.constexpr,
    NC: tl.constexpr,
    LINEAR: tl.constexpr = tl.constexpr(False),
):
    i_c = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    c0 = i_c * C
    bos_qk = i_bh * T * D
    bos_v = i_bh * T * DV
    bos_h = (i_bh * NC + i_c) * (M * P) * DV  # NG=M sub-states of P rows each
    bos_z = (i_bh * NC + i_c) * (M * P)
    o_row = c0 + tl.arange(0, C)
    row_ok = o_row < T

    b_v = tl.load(
        tl.make_block_ptr(
            v_addr + bos_v, (T, DV), (DV, 1), (c0, 0), (C, DV), (1, 0)
        ),
        boundary_check=(0, 1),
    ).to(tl.bfloat16)
    b_vm = tl.where(row_ok[:, None], b_v, 0.0)

    A = tl.zeros([C, C], dtype=tl.float32)
    for g in range(M):
        b_qg = (
            tl.load(
                tl.make_block_ptr(
                    q_addr + bos_qk, (T, D), (D, 1), (c0, g * E), (C, E), (1, 0)
                ),
                boundary_check=(0, 1),
            )
            * s_scale
        )
        b_kg = (
            tl.load(
                tl.make_block_ptr(
                    k_addr + bos_qk, (T, D), (D, 1), (c0, g * E), (C, E), (1, 0)
                ),
                boundary_check=(0, 1),
            )
            * s_scale
        )
        if LINEAR:  # psi(x)=x, identity feature -> P == E
            psi_qg = b_qg.to(tl.bfloat16)
            psi_kg = b_kg.to(tl.bfloat16)
        else:  # psi(x)=vec(x⊗x), SPD feature -> P == E^2
            psi_qg = tl.reshape(b_qg[:, :, None] * b_qg[:, None, :], (C, P)).to(
                tl.bfloat16
            )
            psi_kg = tl.reshape(b_kg[:, :, None] * b_kg[:, None, :], (C, P)).to(
                tl.bfloat16
            )
        psi_kg = tl.where(row_ok[:, None], psi_kg, 0.0)

        h_cg = tl.dot(tl.trans(psi_kg), b_vm)  # (P, Dv) fp32
        z_cg = tl.sum(psi_kg.to(tl.float32), axis=0)  # (P,)
        tl.store(
            tl.make_block_ptr(
                h + bos_h, (M * P, DV), (DV, 1), (g * P, 0), (P, DV), (1, 0)
            ),
            h_cg.to(h.dtype.element_ty),
        )
        tl.store(
            tl.make_block_ptr(z + bos_z, (M * P,), (1,), (g * P,), (P,), (0,)),
            z_cg.to(z.dtype.element_ty),
        )
        A += tl.dot(psi_qg, tl.trans(psi_kg))  # intra, same-group -> sum_g

    A = tl.where((o_row[:, None] >= o_row[None, :]) & row_ok[None, :], A, 0.0)
    n_intra = tl.dot(A.to(tl.bfloat16), b_v)
    d_intra = tl.sum(A, axis=1)
    tl.store(
        tl.make_block_ptr(
            num_i + bos_v, (T, DV), (DV, 1), (c0, 0), (C, DV), (1, 0)
        ),
        n_intra.to(num_i.dtype.element_ty),
        boundary_check=(0, 1),
    )
    tl.store(
        tl.make_block_ptr(den_i + i_bh * T, (T,), (1,), (c0,), (C,), (0,)),
        d_intra.to(den_i.dtype.element_ty),
        boundary_check=(0,),
    )


@triton.jit
def _scan_output_cat_kernel(
    q,
    Hx,
    Zx,
    num_i,
    den_i,
    o,
    den,
    s_scale,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    M: tl.constexpr,
    P: tl.constexpr,
    DV: tl.constexpr,
    C: tl.constexpr,
    NC: tl.constexpr,
    EPS: tl.constexpr,
    LINEAR: tl.constexpr = tl.constexpr(False),
):
    i_c = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    c0 = i_c * C
    bos_qk = i_bh * T * D
    bos_v = i_bh * T * DV
    bos_h = (i_bh * NC + i_c) * (M * P) * DV
    bos_z = (i_bh * NC + i_c) * (M * P)

    n_inter = tl.zeros([C, DV], dtype=tl.float32)
    d_inter = tl.zeros([C], dtype=tl.float32)
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
        if LINEAR:
            psi_qg = b_qg.to(tl.bfloat16)
        else:
            psi_qg = tl.reshape(b_qg[:, :, None] * b_qg[:, None, :], (C, P)).to(
                tl.bfloat16
            )
        H_cg = tl.load(
            tl.make_block_ptr(
                Hx + bos_h, (M * P, DV), (DV, 1), (g * P, 0), (P, DV), (1, 0)
            )
        ).to(tl.bfloat16)
        Z_cg = tl.load(
            tl.make_block_ptr(Zx + bos_z, (M * P,), (1,), (g * P,), (P,), (0,))
        )
        n_inter += tl.dot(psi_qg, H_cg)
        d_inter += tl.sum(psi_qg.to(tl.float32) * Z_cg[None, :], axis=1)

    n_intra = tl.load(
        tl.make_block_ptr(
            num_i + bos_v, (T, DV), (DV, 1), (c0, 0), (C, DV), (1, 0)
        ),
        boundary_check=(0, 1),
    )
    d_intra = tl.load(
        tl.make_block_ptr(den_i + i_bh * T, (T,), (1,), (c0,), (C,), (0,)),
        boundary_check=(0,),
    )
    num = n_inter + n_intra
    d = tl.maximum(d_inter + d_intra, EPS)
    tl.store(
        tl.make_block_ptr(
            o + bos_v, (T, DV), (DV, 1), (c0, 0), (C, DV), (1, 0)
        ),
        (num / d[:, None]).to(o.dtype.element_ty),
        boundary_check=(0, 1),
    )
    tl.store(
        tl.make_block_ptr(den + i_bh * T, (T,), (1,), (c0,), (C,), (0,)),
        d.to(den.dtype.element_ty),
        boundary_check=(0,),
    )


def spd_chunk_scan_cat(
    q, k, v, M, scale=None, C=32, eps=1e-6, warps=4, stages=None, feature="quad"
):
    """Tree-scan chunked forward, CONCAT mode. q,k:(B,H,T,D) v:(B,H,T,DV) -> o,(den).

    feature="quad" (default): psi(x)=vec(x⊗x), the SPD degree-2 kernel; per-group state is E^2 x Dv.
    feature="linear": psi(x)=x, ordinary linear attention (Katharopoulos); per-group state is E x Dv,
    so no E^2 blow-up and no SMEM OOM -- a cheap way to exercise the tree-scan mechanism itself.
    State is NG=M sub-states of P each (total M*P rows). scale cancels in the Nadaraya-Watson ratio.
    """
    B, H, T, D = q.shape
    DV = v.shape[-1]
    E = D // M
    LINEAR = feature == "linear"
    P = E if LINEAR else E * E
    if scale is None:
        scale = 1.0 / (E**0.5)
    assert T % C == 0, "T must be divisible by C"
    NC = T // C
    q, k, v = (x.contiguous() for x in (q, k, v))
    s = scale**0.5
    Pt = M * P  # total state rows
    if stages is None:  # big P*Dv tiles blow SMEM at num_stages>1
        stages = 1 if P * DV >= 32768 else 2

    h = torch.empty(B, H, NC, Pt, DV, device=q.device, dtype=torch.float32)
    zc = torch.empty(B, H, NC, Pt, device=q.device, dtype=torch.float32)
    num_i = torch.empty(B, H, T, DV, device=q.device, dtype=torch.float32)
    den_i = torch.empty(B, H, T, device=q.device, dtype=torch.float32)
    grid = (NC, B * H)
    _scan_local_cat_kernel[grid](
        q,
        k,
        v,
        h,
        zc,
        num_i,
        den_i,
        s,
        T,
        D=D,
        E=E,
        M=M,
        P=P,
        DV=DV,
        C=C,
        NC=NC,
        LINEAR=LINEAR,
        num_warps=warps,
        num_stages=stages,
    )

    # phase 2: exclusive prefix scan over the (Pt, DV) states (reuse chunk_scan.py scan kernels).
    Hx = torch.empty_like(h)
    BPs, BVs = min(64, Pt), min(64, DV)
    pgrid = (triton.cdiv(Pt, BPs), triton.cdiv(DV, BVs))
    if NC <= 256 or (NC & (NC - 1)) != 0:
        _excl_scan_kernel[(B * H, *pgrid)](
            h, Hx, NC=NC, P=Pt, DV=DV, BP=BPs, BV=BVs, num_warps=4
        )
    else:
        BC = 1 << ((NC.bit_length() - 1) // 2)
        G = NC // BC
        NV = pgrid[1]
        NPV = pgrid[0] * pgrid[1]
        bsum = torch.empty(
            B, H, G, Pt, DV, device=q.device, dtype=torch.float32
        )
        _scan_blocked_A[(B * H, G, NPV)](
            h,
            Hx,
            bsum,
            NC=NC,
            BC=BC,
            G=G,
            P=Pt,
            DV=DV,
            BP=BPs,
            BV=BVs,
            NV=NV,
            num_warps=4,
        )
        bpref = torch.empty_like(bsum)
        _excl_scan_kernel[(B * H, *pgrid)](
            bsum, bpref, NC=G, P=Pt, DV=DV, BP=BPs, BV=BVs, num_warps=4
        )
        _scan_blocked_B[(B * H, G, NPV)](
            Hx,
            bpref,
            NC=NC,
            BC=BC,
            G=G,
            P=Pt,
            DV=DV,
            BP=BPs,
            BV=BVs,
            NV=NV,
            num_warps=4,
        )
    Zx = zc.cumsum(2) - zc

    o = torch.empty(B, H, T, DV, device=q.device, dtype=q.dtype)
    den = torch.empty(B, H, T, device=q.device, dtype=torch.float32)
    _scan_output_cat_kernel[grid](
        q,
        Hx,
        Zx,
        num_i,
        den_i,
        o,
        den,
        s,
        T,
        H=H,
        D=D,
        E=E,
        M=M,
        P=P,
        DV=DV,
        C=C,
        NC=NC,
        EPS=eps,
        LINEAR=LINEAR,
        num_warps=warps,
        num_stages=stages,
    )
    return o, den
