"""Tree-scan (parallel-prefix) chunked PSD forward.

The sequential chunk kernel walks chunks left->right carrying the state S
(O(NC) dependency depth, only B*H programs of parallelism). Here the inter-chunk
recurrence S_c = S_{c-1} + psi(K_c)^T V_c is an ASSOCIATIVE scan (addition), so:

  phase 1 (parallel, NC*B*H programs): per chunk local state h_c = psi(K_c)^T V_c,
           z_c = sum psi(K_c), and the intra-chunk causal output.
  phase 2 (parallel prefix scan, ~log NC depth): exclusive cumsum of h_c, z_c.
  phase 3 (parallel, NC*B*H programs): o = (psi(Q_c) @ H_c + intra) / (psi(Q_c) @ Z_c + intra_den).

Parallelism is NC*B*H (vs B*H sequential) -- the win at low batch / long context
(decoding). Cost: the per-chunk local states h_c live in HBM (O(NC*P*Dv)).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _scan_local_kernel(
    q_addr,
    k_addr,
    v_addr,
    h_addr,
    z_addr,
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
):
    idx_bh = tl.program_id(1)
    idx_c = tl.program_id(0)
    c0 = idx_c * C
    bos_qk = idx_bh * T * D
    bos_v = idx_bh * T * DV
    bos_h = (idx_bh * NC + idx_c) * P * DV
    bos_z = (idx_bh * NC + idx_c) * P
    o_row = c0 + tl.arange(0, C)
    row_ok = o_row < T

    b_v = tl.load(
        tl.make_block_ptr(v_addr + bos_v, (T, DV), (DV, 1), (c0, 0), (C, DV), (1, 0)),
        boundary_check=(0, 1),
    )
    psi_q = tl.zeros([C, P], dtype=tl.bfloat16)
    psi_k = tl.zeros([C, P], dtype=tl.bfloat16)

    for g in range(M):  # sum mode: sum over groups
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
        psi_q += tl.reshape(b_qg[:, :, None] * b_qg[:, None, :], (C, P)).to(tl.bfloat16)
        psi_k += tl.reshape(b_kg[:, :, None] * b_kg[:, None, :], (C, P)).to(tl.bfloat16)
    psi_k = tl.where(row_ok[:, None], psi_k, 0.0)
    b_vm = tl.where(row_ok[:, None], b_v, 0.0)

    # local state (P, Dv) and z (P,)
    h_c = tl.dot(tl.trans(psi_k), b_vm)  # fp32
    z_c = tl.sum(psi_k.to(tl.float32), axis=0)
    tl.store(
        tl.make_block_ptr(h_addr + bos_h, (P, DV), (DV, 1), (0, 0), (P, DV), (1, 0)),
        h_c.to(h_addr.dtype.element_ty),
    )
    tl.store(
        tl.make_block_ptr(z_addr + bos_z, (P,), (1,), (0,), (P,), (0,)),
        z_c.to(z_addr.dtype.element_ty),
    )

    # intra-chunk causal output
    A = tl.dot(psi_q, tl.trans(psi_k))
    A = tl.where((o_row[:, None] >= o_row[None, :]) & row_ok[None, :], A, 0.0)
    n_intra = tl.dot(A.to(tl.bfloat16), b_v)  # (C, Dv)
    d_intra = tl.sum(A, axis=1)
    tl.store(
        tl.make_block_ptr(num_i + bos_v, (T, DV), (DV, 1), (c0, 0), (C, DV), (1, 0)),
        n_intra.to(num_i.dtype.element_ty),
        boundary_check=(0, 1),
    )
    tl.store(
        tl.make_block_ptr(den_i + idx_bh * T, (T,), (1,), (c0,), (C,), (0,)),
        d_intra.to(den_i.dtype.element_ty),
        boundary_check=(0,),
    )


@triton.jit
def _scan_output_kernel(
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
):
    i_c = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    c0 = i_c * C
    bos_qk = i_bh * T * D
    bos_v = i_bh * T * DV
    bos_h = (i_bh * NC + i_c) * P * DV
    bos_z = (i_bh * NC + i_c) * P

    psi_q = tl.zeros([C, P], dtype=tl.bfloat16)
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
        psi_q += tl.reshape(b_qg[:, :, None] * b_qg[:, None, :], (C, P)).to(tl.bfloat16)

    H_c = tl.load(
        tl.make_block_ptr(Hx + bos_h, (P, DV), (DV, 1), (0, 0), (P, DV), (1, 0))
    ).to(tl.bfloat16)
    Z_c = tl.load(tl.make_block_ptr(Zx + bos_z, (P,), (1,), (0,), (P,), (0,)))
    n_inter = tl.dot(psi_q, H_c)  # (C, Dv)
    d_inter = tl.sum(psi_q.to(tl.float32) * Z_c[None, :], axis=1)

    n_intra = tl.load(
        tl.make_block_ptr(num_i + bos_v, (T, DV), (DV, 1), (c0, 0), (C, DV), (1, 0)),
        boundary_check=(0, 1),
    )
    d_intra = tl.load(
        tl.make_block_ptr(den_i + i_bh * T, (T,), (1,), (c0,), (C,), (0,)),
        boundary_check=(0,),
    )
    num = n_inter + n_intra
    d = tl.maximum(d_inter + d_intra, EPS)
    tl.store(
        tl.make_block_ptr(o + bos_v, (T, DV), (DV, 1), (c0, 0), (C, DV), (1, 0)),
        (num / d[:, None]).to(o.dtype.element_ty),
        boundary_check=(0, 1),
    )
    tl.store(
        tl.make_block_ptr(den + i_bh * T, (T,), (1,), (c0,), (C,), (0,)),
        d.to(den.dtype.element_ty),
        boundary_check=(0,),
    )


@triton.jit
def _excl_scan_kernel(
    h,
    Hx,
    NC: tl.constexpr,
    P: tl.constexpr,
    DV: tl.constexpr,
    BP: tl.constexpr,
    BV: tl.constexpr,
):
    """Exclusive prefix sum of h over the chunk axis: Hx[c] = sum_{j<c} h[j].

    One program owns a (BP,BV) tile of the (P,DV) state and streams the NC chunks,
    carrying the running sum in registers. Replaces torch's `cumsum(2) - h` (which
    round-trips the O(NC*P*DV) tensor ~5x) with exactly 2 HBM passes (read h, write Hx),
    and folds in the exclusive shift for free. Parallelism = BH * ceil(P/BP) * ceil(DV/BV).
    """
    i_bh = tl.program_id(0).to(tl.int64)
    i_p = tl.program_id(1)
    i_v = tl.program_id(2)
    acc = tl.zeros([BP, BV], dtype=tl.float32)
    base = i_bh * NC * P * DV
    for i_c in range(NC):
        off = base + i_c * P * DV
        ip = tl.make_block_ptr(
            h + off, (P, DV), (DV, 1), (i_p * BP, i_v * BV), (BP, BV), (1, 0)
        )
        op = tl.make_block_ptr(
            Hx + off, (P, DV), (DV, 1), (i_p * BP, i_v * BV), (BP, BV), (1, 0)
        )
        tl.store(op, acc)  # exclusive: store prefix BEFORE adding
        acc += tl.load(ip).to(tl.float32)


@triton.jit
def _scan_blocked_A(
    h,
    Hx,
    bsum,
    NC: tl.constexpr,
    BC: tl.constexpr,
    G: tl.constexpr,
    P: tl.constexpr,
    DV: tl.constexpr,
    BP: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
):
    """Level-1 of the 2-level scan: block-local exclusive scan of BC chunks + block total.
    Grid (BH, G, (P/BP)*(DV/BV)) -> G*BH*tiles programs (vs BH*tiles for the flat scan),
    each of depth O(BC). Requires BC*G == NC so no boundary guard is needed."""
    i_bh = tl.program_id(0).to(tl.int64)
    i_blk = tl.program_id(1)
    i_pv = tl.program_id(2)
    i_p = i_pv // NV
    i_v = i_pv % NV
    acc = tl.zeros([BP, BV], dtype=tl.float32)
    base = i_bh * NC * P * DV
    c0 = i_blk * BC
    for j in range(BC):
        off = base + (c0 + j) * P * DV
        op = tl.make_block_ptr(
            Hx + off, (P, DV), (DV, 1), (i_p * BP, i_v * BV), (BP, BV), (1, 0)
        )
        ip = tl.make_block_ptr(
            h + off, (P, DV), (DV, 1), (i_p * BP, i_v * BV), (BP, BV), (1, 0)
        )
        tl.store(op, acc)
        acc += tl.load(ip).to(tl.float32)
    ob = (i_bh * G + i_blk) * P * DV
    tl.store(
        tl.make_block_ptr(
            bsum + ob, (P, DV), (DV, 1), (i_p * BP, i_v * BV), (BP, BV), (1, 0)
        ),
        acc,
    )


@triton.jit
def _scan_blocked_B(
    Hx,
    bpref,
    NC: tl.constexpr,
    BC: tl.constexpr,
    G: tl.constexpr,
    P: tl.constexpr,
    DV: tl.constexpr,
    BP: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
):
    """Level-3: add each block's exclusive-over-blocks prefix onto its BC local prefixes."""
    i_bh = tl.program_id(0).to(tl.int64)
    i_blk = tl.program_id(1)
    i_pv = tl.program_id(2)
    i_p = i_pv // NV
    i_v = i_pv % NV
    ob = (i_bh * G + i_blk) * P * DV
    pref = tl.load(
        tl.make_block_ptr(
            bpref + ob, (P, DV), (DV, 1), (i_p * BP, i_v * BV), (BP, BV), (1, 0)
        )
    )
    base = i_bh * NC * P * DV
    c0 = i_blk * BC
    for j in range(BC):
        off = base + (c0 + j) * P * DV
        hp = tl.make_block_ptr(
            Hx + off, (P, DV), (DV, 1), (i_p * BP, i_v * BV), (BP, BV), (1, 0)
        )
        tl.store(hp, tl.load(hp) + pref)


def spd_chunk_scan(q, k, v, M, mode, scale=None, C=32, eps=1e-6, warps=4):
    """Tree-scan chunked SPD forward (sum mode). q,k:(B,H,T,D) v:(B,H,T,DV).

    warps: num_warps for the local/output kernels. More warps spreads the
    (C,P)/(C,C) tiles over more threads -> fewer registers/thread, which lets C
    grow past the single-warp-group register cap.
    """
    assert mode == "sum", "tree-scan implemented for sum mode"
    B, H, T, D = q.shape
    DV = v.shape[-1]
    E = D // M
    P = E * E
    if scale is None:
        scale = 1.0 / E
    assert T % C == 0, "T must be divisible by C"
    NC = T // C
    q, k, v = (x.contiguous() for x in (q, k, v))
    s = scale**0.5

    h = torch.empty(B, H, NC, P, DV, device=q.device, dtype=torch.float32)
    zc = torch.empty(B, H, NC, P, device=q.device, dtype=torch.float32)
    num_i = torch.empty(B, H, T, DV, device=q.device, dtype=torch.float32)
    den_i = torch.empty(B, H, T, device=q.device, dtype=torch.float32)
    grid = (NC, B * H)
    _scan_local_kernel[grid](
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
        num_warps=warps,
        num_stages=2,
    )

    # phase 2: exclusive prefix scan over chunk states h (O(NC*P*DV)). Two regimes:
    #  - few chunks: flat sequential scan (BH*tiles programs, O(NC) depth) is enough.
    #  - many chunks (B=1 long context): the flat scan is latency-bound (few programs, one
    #    long serial chain), so use a 2-LEVEL blocked scan -> depth O(NC/G + G) ~ O(sqrt NC)
    #    with G*BH*tiles programs, realizing the O(log(T/C))-ish tree depth the flat scan
    #    threw away. (Recurse the block-sum scan for full O(log NC).)
    Hx = torch.empty_like(h)
    BPs, BVs = min(64, P), min(64, DV)
    pgrid = (triton.cdiv(P, BPs), triton.cdiv(DV, BVs))
    if NC <= 256 or (NC & (NC - 1)) != 0:  # few chunks, or non-power-of-2 NC
        _excl_scan_kernel[(B * H, *pgrid)](
            h, Hx, NC=NC, P=P, DV=DV, BP=BPs, BV=BVs, num_warps=4
        )
    else:
        BC = 1 << ((NC.bit_length() - 1) // 2)  # ~sqrt(NC), divides NC (power of 2)
        G = NC // BC
        NV = pgrid[1]  # number of DV tiles
        NPV = pgrid[0] * pgrid[1]  # flattened (P,DV) tiles -> axis 2
        bsum = torch.empty(B, H, G, P, DV, device=q.device, dtype=torch.float32)
        _scan_blocked_A[(B * H, G, NPV)](
            h,
            Hx,
            bsum,
            NC=NC,
            BC=BC,
            G=G,
            P=P,
            DV=DV,
            BP=BPs,
            BV=BVs,
            NV=NV,
            num_warps=4,
        )
        bpref = torch.empty_like(bsum)
        _excl_scan_kernel[(B * H, *pgrid)](
            bsum, bpref, NC=G, P=P, DV=DV, BP=BPs, BV=BVs, num_warps=4
        )
        _scan_blocked_B[(B * H, G, NPV)](
            Hx, bpref, NC=NC, BC=BC, G=G, P=P, DV=DV, BP=BPs, BV=BVs, NV=NV, num_warps=4
        )
    Zx = zc.cumsum(2) - zc

    o = torch.empty(B, H, T, DV, device=q.device, dtype=q.dtype)
    den = torch.empty(B, H, T, device=q.device, dtype=torch.float32)
    _scan_output_kernel[grid](
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
        num_warps=warps,
        num_stages=2,
    )
    return o, den
