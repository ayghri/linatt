"""Triton kernels for SPD chunked linear attention with on-the-fly SRAM
expansion. Supports multi-group SPD (M groups, each of dim E = head_k / M)
and d_v tiling.

State representation: full per-group outer products, no sqrt(2) packing.
For each group m, psi_full(g_m) = vec(g_m g_m^T) of dim E^2. Multi-group concat
gives total q-dim Q_total = M * E^2. Inner products satisfy
    <psi_M(x), psi_M(y)> = sum_m (x_m . y_m)^2
matching the packed form in kata/feature_maps.py.

Memory budgets that drove block-size choices on 3090 (100 KB shared / SM):
    chunk_output: tiles per program kept under ~80 KB:
        Q_full (C, ME)     C=32, M=4, E=16 -> 8 KB
        K_full (C, ME)                     -> 8 KB
        V_tile (C, d_v_BLK) d_v_BLK=16       -> 2 KB
        S_tile (E*E, d_v_BLK)               -> 16 KB
        Q_outer_g (C, E*E) (per-group)     -> 32 KB
        QK + A2 (C, C)                     -> 8 KB
        ...
    Looping over (M groups) and (d_v tiles) keeps largest live tile to a
    single group's Q_outer + S_tile.

Constraints:
- T % C == 0, NC = T/C must be power of 2
- E >= 16 (Triton tl.dot inner-dim min) -> head_k / M >= 16
- d_v % d_v_BLK == 0
- ME = M*E (= head_k_dim) loaded as (C, ME); subselected per group with E
  constexpr offsets.

Math (uses SPD identity (q.k)^2 throughout):
    Forward
        A[t, m] = sum_g (q_g[t] . k_g[m])^2     (per-group squared inner)
        N[t, d] = sum_{m<=t} A[t, m] v[m, d]
        D[t]    = sum_{m<=t} A[t, m] + eps
        o[t, d] = N / D
    Backward (per chunk; vectorized over groups with constexpr loops)
        dN[t, d] = do[t, d] / D[t]
        dD[t]    = -<do[t], o[t]> / D[t]
        For each group g:
            scalar prefactor for dq_g and dk_g sums = 2 (q_g . k_g)
            dq_g[t, l] = 2 sum_m (q_g[t]. k_g[m]) (<dN[t], v[m]> + dD[t]) k_g[m, l]
            dk_g[m, l] = 2 sum_t (q_g[t]. k_g[m]) (<dN[t], v[m]> + dD[t]) q_g[t, l]
        dv[m, d] = sum_t A[t, m] dN[t, d]                (sum across groups baked into A)
"""

import torch
import triton
import triton.language as tl

# Autotune configs. Triton picks the fastest per (key) combination at first
# call; subsequent calls with the same key reuse the cached choice.
# d_v_BLK is the inner-loop tile over the value dim; (num_warps, num_stages)
# control GPU resource use. Larger d_v_BLK -> fewer iterations / more SRAM;
# smaller d_v_BLK -> more iterations / less SRAM. Autotune walks both.

_SPD_CONFIGS_d_v = [
    # Single-config "no-op" autotune: avoids the slow first-call compile across
    # multiple configs while still letting us upgrade to a real sweep later.
    # d_v_BLK >= 16 (inner-dim min for tl.dot in bwd kernels).
    # num_stages=1 only — pipelining caused cross-call data corruption.
    triton.Config({"d_v_BLK": 16}, num_warps=4, num_stages=1),
]
_SPD_CONFIGS_SCAN = [
    triton.Config({}, num_warps=4, num_stages=1),
]
# Autotune key: anything that affects shape / generated code paths.
_SPD_KEY = ["T", "NC", "C", "ME", "M", "d_v"]
_SPD_KEY_SCAN = ["NC", "Q_TOTAL", "d_v"]


def _spd_dv_prune(configs, named_args, **kwargs):
    """Drop configs whose d_v_BLK > actual d_v (would produce 0-iter loops).

    d_v is a constexpr passed to the kernel; in early_config_prune it shows up
    in either named_args (positional/keyword) or kwargs (constexpr from
    autotune call site).
    """
    d_v = named_args.get("d_v", kwargs.get("d_v"))
    if d_v is None:
        return configs
    return [c for c in configs if c.kwargs.get("d_v_BLK", d_v) <= d_v]


# =====================================================================
# FORWARD KERNELS
# =====================================================================


@triton.autotune(
    configs=_SPD_CONFIGS_d_v,
    key=_SPD_KEY,
    prune_configs_by={"early_config_prune": _spd_dv_prune},
)
@triton.jit
def spd_chunk_state_kernel(
    K_addr,
    V_addr,
    S_addr,
    Z_addr,
    num_heads,
    T,
    num_chunks,
    chunk_size: tl.constexpr,
    ME: tl.constexpr,
    E: tl.constexpr,
    num_groups: tl.constexpr,
    d_v: tl.constexpr,
    Q_PER_G: tl.constexpr,
    d_v_BLK: tl.constexpr,
):
    """Per-chunk per-group state local statistics:
        S_local[c, m, q, d]  +=  K_outer_g[i, j]  *  V[c, d]
        Z_local[c, m, q]     +=  K_outer_g[i, j]
    where K_outer_g = K_g outer K_g flattened to E^2 = Q_PER_G.

    Loops: M groups (outer) -> d_v tiles (inner). Each program does one
    (b, h, nc) chunk.
    Grid: (B*H, NC).

    State layout: S[B, H, NC, M*Q_PER_G, d_v] flat-concat over M groups.
                  Z[B, H, NC, M*Q_PER_G] same.
    """
    pid_bh = tl.program_id(0)
    pid_nc = tl.program_id(1)
    b = pid_bh // num_heads
    h = pid_bh % num_heads

    offs_c = tl.arange(0, chunk_size)
    offs_e = tl.arange(0, E)
    offs_q_pg = tl.arange(0, Q_PER_G)
    t_idx = pid_nc * chunk_size + offs_c

    Q_TOTAL: tl.constexpr = num_groups * Q_PER_G

    for m_id in tl.static_range(num_groups):
        # K_g[c, e] = K[b, t_idx, h, m_id*E + e]
        k_off = (
            b * T * num_heads * ME
            + t_idx[:, None] * num_heads * ME
            + h * ME
            + m_id * E
            + offs_e[None, :]
        )
        K_g = tl.load(K_addr + k_off).to(tl.float32)  # (C, E)

        K_outer = K_g[:, :, None] * K_g[:, None, :]  # (C, E, E)
        K_flat = tl.reshape(K_outer, (chunk_size, Q_PER_G))  # (C, Q_PER_G)

        # Z_local for this group (E^2,)
        Z_g = tl.sum(K_flat, axis=0)
        z_off = (
            b * num_heads * num_chunks * Q_TOTAL
            + h * num_chunks * Q_TOTAL
            + pid_nc * Q_TOTAL
            + m_id * Q_PER_G
            + offs_q_pg
        )
        tl.store(Z_addr + z_off, Z_g)

        # d_v-tiled S_local: S_g_tile = K_flat^T @ V_tile, store per tile.
        for dv_id in tl.static_range(d_v // d_v_BLK):
            offs_dv = dv_id * d_v_BLK + tl.arange(0, d_v_BLK)
            v_off = (
                b * T * num_heads * d_v + t_idx[:, None] * num_heads * d_v + h * d_v + offs_dv[None, :]
            )
            V_tile = tl.load(V_addr + v_off).to(tl.float32)  # (C, d_v_BLK)

            S_tile = tl.dot(
                tl.trans(K_flat), V_tile, allow_tf32=False
            )  # (Q_PER_G, d_v_BLK)

            s_off = (
                b * num_heads * num_chunks * Q_TOTAL * d_v
                + h * num_chunks * Q_TOTAL * d_v
                + pid_nc * Q_TOTAL * d_v
                + (m_id * Q_PER_G + offs_q_pg)[:, None] * d_v
                + offs_dv[None, :]
            )
            tl.store(S_addr + s_off, S_tile)


@triton.autotune(configs=_SPD_CONFIGS_SCAN, key=_SPD_KEY_SCAN)
@triton.jit
def spd_chunk_scan_linear_kernel(
    S_local_addr,
    Z_local_addr,
    S_prefix_addr,
    Z_prefix_addr,
    S_init_addr,
    Z_init_addr,
    H,
    HAS_INIT: tl.constexpr,
    NC: tl.constexpr,
    Q_TOTAL: tl.constexpr,
    d_v: tl.constexpr,
):
    """Linear (sequential) scan: each program loops chunks 0..NC-1 in order,
    accumulating Z and S states. O(NC) work per program — what standard
    chunked linear attention uses for its inter-chunk reduction.
    Grid: (B*H, Q_TOTAL).
    """
    pid_bh = tl.program_id(0)
    q_idx = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_dv = tl.arange(0, d_v)

    if HAS_INIT:
        z_init_off = b * H * Q_TOTAL + h * Q_TOTAL + q_idx
        s_init_off = b * H * Q_TOTAL * d_v + h * Q_TOTAL * d_v + q_idx * d_v + offs_dv
        z_acc = tl.load(Z_init_addr + z_init_off)
        s_acc = tl.load(S_init_addr + s_init_off)
    else:
        z_acc = tl.zeros([], dtype=tl.float32)
        s_acc = tl.zeros((d_v,), dtype=tl.float32)

    # tl.range = non-unrolling loop; forces a true sequential O(NC) scan,
    # which is what we want to compare against the parallel tl.cumsum.
    for c in tl.range(0, NC):
        z_off = b * H * NC * Q_TOTAL + h * NC * Q_TOTAL + c * Q_TOTAL + q_idx
        s_off = (
            b * H * NC * Q_TOTAL * d_v
            + h * NC * Q_TOTAL * d_v
            + c * Q_TOTAL * d_v
            + q_idx * d_v
            + offs_dv
        )
        tl.store(Z_prefix_addr + z_off, z_acc)
        tl.store(S_prefix_addr + s_off, s_acc)
        z_acc = z_acc + tl.load(Z_local_addr + z_off)
        s_acc = s_acc + tl.load(S_local_addr + s_off)


@triton.autotune(configs=_SPD_CONFIGS_SCAN, key=_SPD_KEY_SCAN)
@triton.jit
def spd_chunk_scan_kernel(
    S_local_addr,
    Z_local_addr,
    S_prefix_addr,
    Z_prefix_addr,
    S_init_addr,
    Z_init_addr,
    H,
    HAS_INIT: tl.constexpr,
    NC: tl.constexpr,
    Q_TOTAL: tl.constexpr,
    d_v: tl.constexpr,
):
    """Exclusive prefix sum along NC. Grid: (B*H, Q_TOTAL).
    Each program handles one (b, h, q_idx); loops over d_v in scan kernel
    is small (d_v up to 64) so we don't tile here.
    """
    pid_bh = tl.program_id(0)
    q_idx = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_nc = tl.arange(0, NC)
    offs_dv = tl.arange(0, d_v)

    z_off = b * H * NC * Q_TOTAL + h * NC * Q_TOTAL + offs_nc * Q_TOTAL + q_idx
    Z = tl.load(Z_local_addr + z_off)
    Z_excl = tl.cumsum(Z, axis=0) - Z
    if HAS_INIT:
        z_init = tl.load(Z_init_addr + b * H * Q_TOTAL + h * Q_TOTAL + q_idx)
        Z_excl = Z_excl + z_init
    tl.store(Z_prefix_addr + z_off, Z_excl)

    s_off = (
        b * H * NC * Q_TOTAL * d_v
        + h * NC * Q_TOTAL * d_v
        + offs_nc[:, None] * Q_TOTAL * d_v
        + q_idx * d_v
        + offs_dv[None, :]
    )
    S = tl.load(S_local_addr + s_off)
    S_excl = tl.cumsum(S, axis=0) - S
    if HAS_INIT:
        s_init_off = b * H * Q_TOTAL * d_v + h * Q_TOTAL * d_v + q_idx * d_v + offs_dv
        s_init = tl.load(S_init_addr + s_init_off)
        S_excl = S_excl + s_init[None, :]
    tl.store(S_prefix_addr + s_off, S_excl)


@triton.autotune(configs=_SPD_CONFIGS_SCAN, key=_SPD_KEY)
@triton.jit
def spd_chunk_output_kernel(
    Q_addr,
    K_addr,
    V_addr,
    S_pref_addr,
    Z_pref_addr,
    O_addr,
    D_addr,
    H,
    T,
    NC,
    C: tl.constexpr,
    ME: tl.constexpr,
    E: tl.constexpr,
    M: tl.constexpr,
    d_v: tl.constexpr,
    Q_PER_G: tl.constexpr,
    EPS: tl.constexpr,
    NORMALIZE: tl.constexpr,
):
    """Output for one chunk = inter (psi(Q) @ S_prefix) + intra (causal).
    Inner: loop over M groups for the per-group (q.k)^2 contribution; outer
    loop over d_v tiles. Grid: (B*H, NC).
    """
    pid_bh = tl.program_id(0)
    pid_nc = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_c = tl.arange(0, C)
    offs_e = tl.arange(0, E)
    # offs_me = tl.arange(0, ME)
    offs_q_pg = tl.arange(0, Q_PER_G)
    t_idx = pid_nc * C + offs_c

    Q_TOTAL: tl.constexpr = M * Q_PER_G

    causal = offs_c[:, None] >= offs_c[None, :]
    A_total = tl.zeros((C, C), dtype=tl.float32)
    Z_inter = tl.zeros((C,), dtype=tl.float32)
    # Full d_v in one tile — head_v typically <= 64 fits comfortably and
    # avoids per-tile loop overhead.
    O_inter = tl.zeros((C, d_v), dtype=tl.float32)

    # Single M-loop: load Q_g/K_g once, do all per-group work in one pass.
    offs_dv_full = tl.arange(0, d_v)
    for m_id in tl.static_range(M):
        qk_g_off = (
            b * T * H * ME
            + t_idx[:, None] * H * ME
            + h * ME
            + m_id * E
            + offs_e[None, :]
        )
        Q_g = tl.load(Q_addr + qk_g_off).to(tl.float32)
        K_g = tl.load(K_addr + qk_g_off).to(tl.float32)

        A_g = tl.dot(Q_g, tl.trans(K_g), allow_tf32=False)
        A_total += A_g * A_g

        Q_outer_g = Q_g[:, :, None] * Q_g[:, None, :]
        Q_flat_g = tl.reshape(Q_outer_g, (C, Q_PER_G))

        Zp_g_off = (
            b * H * NC * Q_TOTAL
            + h * NC * Q_TOTAL
            + pid_nc * Q_TOTAL
            + m_id * Q_PER_G
            + offs_q_pg
        )
        Zp_g = tl.load(Z_pref_addr + Zp_g_off).to(tl.float32)
        Z_inter += tl.sum(Q_flat_g * Zp_g[None, :], axis=1)

        Sp_g_off = (
            b * H * NC * Q_TOTAL * d_v
            + h * NC * Q_TOTAL * d_v
            + pid_nc * Q_TOTAL * d_v
            + (m_id * Q_PER_G + offs_q_pg)[:, None] * d_v
            + offs_dv_full[None, :]
        )
        Sp_g = tl.load(S_pref_addr + Sp_g_off).to(tl.float32)
        O_inter += tl.dot(Q_flat_g, Sp_g, allow_tf32=False)

    A_masked = tl.where(causal, A_total, 0.0)
    Z_intra = tl.sum(A_masked, axis=1)
    D = Z_inter + Z_intra + EPS

    v_off = b * T * H * d_v + t_idx[:, None] * H * d_v + h * d_v + offs_dv_full[None, :]
    V_tile = tl.load(V_addr + v_off).to(tl.float32)
    O_intra = tl.dot(A_masked, V_tile, allow_tf32=False)
    O = O_inter + O_intra
    if NORMALIZE:
        O = O / D[:, None]      # sum-score normalization (off for GDN-style scaling)

    tl.store(O_addr + v_off, O.to(O_addr.dtype.element_ty))

    d_off = b * T * H + t_idx * H + h
    tl.store(D_addr + d_off, D)


@triton.jit
def spd_final_state_kernel(
    S_local_addr,
    Z_local_addr,
    S_prefix_addr,
    Z_prefix_addr,
    S_final_addr,
    Z_final_addr,
    H,
    NC: tl.constexpr,
    Q_TOTAL: tl.constexpr,
    d_v: tl.constexpr,
):
    """Final state at end of sequence: prefix at chunk NC-1 + local NC-1."""
    pid_bh = tl.program_id(0)
    q_idx = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_dv = tl.arange(0, d_v)
    last = NC - 1

    z_off = b * H * NC * Q_TOTAL + h * NC * Q_TOTAL + last * Q_TOTAL + q_idx
    z_final = tl.load(Z_prefix_addr + z_off) + tl.load(Z_local_addr + z_off)
    tl.store(Z_final_addr + b * H * Q_TOTAL + h * Q_TOTAL + q_idx, z_final)

    s_off = (
        b * H * NC * Q_TOTAL * d_v
        + h * NC * Q_TOTAL * d_v
        + last * Q_TOTAL * d_v
        + q_idx * d_v
        + offs_dv
    )
    s_final = tl.load(S_prefix_addr + s_off) + tl.load(S_local_addr + s_off)
    tl.store(
        S_final_addr + b * H * Q_TOTAL * d_v + h * Q_TOTAL * d_v + q_idx * d_v + offs_dv,
        s_final,
    )


def chunk_kata_spd_fwd(
    q_hat: torch.Tensor,
    k_hat: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int = 32,
    num_groups: int = 1,
    eps: float = 1e-6,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    output_final_state: bool = False,
    scan_mode: str = "tree",
    normalize: bool = True,
):
    """SPD chunked linear-attention forward.

    Args:
        q_hat, k_hat: (B, T, H, ME) where ME = num_groups * E_per_group
        v:            (B, T, H, d_v)
        num_groups:   M >= 1; head_k must be divisible by M and E = head_k/M >= 16.
        dv_block:     d_v tile size; d_v must be divisible.
        initial_state: optional (S_init, Z_init) of shapes (B, H, M*E^2, d_v)
                       and (B, H, M*E^2).
        output_final_state: if True, also return (S_final, Z_final).
    Returns:
        o: (B, T, H, d_v); D, S_prefix, Z_prefix (saved for backward); final_state
    """
    q_hat = q_hat.contiguous()
    k_hat = k_hat.contiguous()
    v = v.contiguous()

    B, T, H, ME = q_hat.shape
    d_v = v.shape[-1]
    if ME % num_groups != 0:
        raise ValueError(f"head_k_dim ME={ME} not divisible by num_groups={num_groups}")
    E = ME // num_groups
    M = num_groups
    if E < 16:
        raise ValueError(f"E = head_k/M = {E} < 16; raise head_k or lower M")
    C = chunk_size
    if T % C != 0:
        raise ValueError(f"T={T} not divisible by chunk_size {C}")
    NC = T // C
    if NC & (NC - 1) != 0:
        raise ValueError(f"NC=T/C={NC} must be a power of 2 for tl.cumsum")

    Q_PER_G = E * E
    Q_TOTAL = M * Q_PER_G

    dev = q_hat.device

    S_local = torch.empty(B, H, NC, Q_TOTAL, d_v, device=dev, dtype=torch.float32)
    Z_local = torch.empty(B, H, NC, Q_TOTAL, device=dev, dtype=torch.float32)
    S_prefix = torch.empty_like(S_local)
    Z_prefix = torch.empty_like(Z_local)
    O = torch.empty_like(v)
    D = torch.empty(B, T, H, device=dev, dtype=torch.float32)

    spd_chunk_state_kernel[(B * H, NC)](
        k_hat,
        v,
        S_local,
        Z_local,
        H,
        T,
        NC,
        C=C,
        ME=ME,
        E=E,
        M=M,
        d_v=d_v,
        Q_PER_G=Q_PER_G,
    )

    if initial_state is not None:
        S_init, Z_init = initial_state
        S_init = S_init.contiguous().to(torch.float32)
        Z_init = Z_init.contiguous().to(torch.float32)
        has_init = True
    else:
        S_init = torch.empty(0, device=dev, dtype=torch.float32)
        Z_init = torch.empty(0, device=dev, dtype=torch.float32)
        has_init = False

    if scan_mode == "tree":
        scan_kernel = spd_chunk_scan_kernel
    else:
        scan_kernel = spd_chunk_scan_linear_kernel

    scan_kernel[(B * H, Q_TOTAL)](
        S_local,
        Z_local,
        S_prefix,
        Z_prefix,
        S_init,
        Z_init,
        H,
        HAS_INIT=has_init,
        NC=NC,
        Q_TOTAL=Q_TOTAL,
        d_v=d_v,
    )

    spd_chunk_output_kernel[(B * H, NC)](
        q_hat,
        k_hat,
        v,
        S_prefix,
        Z_prefix,
        O,
        D,
        H,
        T,
        NC,
        C=C,
        ME=ME,
        E=E,
        M=M,
        d_v=d_v,
        Q_PER_G=Q_PER_G,
        EPS=eps,
        NORMALIZE=normalize,
    )

    final_state = None
    if output_final_state:
        S_final = torch.empty(B, H, Q_TOTAL, d_v, device=dev, dtype=torch.float32)
        Z_final = torch.empty(B, H, Q_TOTAL, device=dev, dtype=torch.float32)
        spd_final_state_kernel[(B * H, Q_TOTAL)](
            S_local,
            Z_local,
            S_prefix,
            Z_prefix,
            S_final,
            Z_final,
            H,
            NC,  # type: ignore
            Q_TOTAL,  # type: ignore
            d_v,  # type: ignore
        )
        final_state = (S_final, Z_final)

    return O, D, S_prefix, Z_prefix, final_state


# =====================================================================
# BACKWARD KERNELS
# =====================================================================


@triton.autotune(
    configs=_SPD_CONFIGS_d_v,
    key=_SPD_KEY,
    prune_configs_by={"early_config_prune": _spd_dv_prune},
)
@triton.jit
def spd_bwd_chunk_state_kernel(
    Q_addr,
    dN_addr,
    dD_addr,
    RS_addr,
    RZ_addr,
    H,
    T,
    NC,
    C: tl.constexpr,
    ME: tl.constexpr,
    E: tl.constexpr,
    M: tl.constexpr,
    d_v: tl.constexpr,
    Q_PER_G: tl.constexpr,
    d_v_BLK: tl.constexpr,
):
    """Reverse-state local: RS_g = Q_outer_g^T @ dN, RZ_g = Q_outer_g^T @ dD.
    Same loop structure as fwd state with (q, dN, dD) instead of (k, v, 1).
    Grid: (B*H, NC).
    """
    pid_bh = tl.program_id(0)
    pid_nc = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_c = tl.arange(0, C)
    offs_e = tl.arange(0, E)
    offs_q_pg = tl.arange(0, Q_PER_G)
    t_idx = pid_nc * C + offs_c
    Q_TOTAL: tl.constexpr = M * Q_PER_G

    dd_off = b * T * H + t_idx * H + h
    dD = tl.load(dD_addr + dd_off).to(tl.float32)  # (C,)

    for m_id in tl.static_range(M):
        Q_off = (
            b * T * H * ME
            + t_idx[:, None] * H * ME
            + h * ME
            + m_id * E
            + offs_e[None, :]
        )
        Q_g = tl.load(Q_addr + Q_off).to(tl.float32)  # (C, E)

        Q_outer = Q_g[:, :, None] * Q_g[:, None, :]
        Q_flat = tl.reshape(Q_outer, (C, Q_PER_G))  # (C, Q_PER_G)

        # RZ
        RZ_g = tl.sum(Q_flat * dD[:, None], axis=0)
        rz_off = (
            b * H * NC * Q_TOTAL
            + h * NC * Q_TOTAL
            + pid_nc * Q_TOTAL
            + m_id * Q_PER_G
            + offs_q_pg
        )
        tl.store(RZ_addr + rz_off, RZ_g)

        # RS d_v-tiled
        for dv_id in tl.static_range(d_v // d_v_BLK):
            offs_dv = dv_id * d_v_BLK + tl.arange(0, d_v_BLK)
            dn_off = (
                b * T * H * d_v + t_idx[:, None] * H * d_v + h * d_v + offs_dv[None, :]
            )
            dN_tile = tl.load(dN_addr + dn_off).to(tl.float32)  # (C, d_v_BLK)
            RS_tile = tl.dot(tl.trans(Q_flat), dN_tile, allow_tf32=False)

            rs_off = (
                b * H * NC * Q_TOTAL * d_v
                + h * NC * Q_TOTAL * d_v
                + pid_nc * Q_TOTAL * d_v
                + (m_id * Q_PER_G + offs_q_pg)[:, None] * d_v
                + offs_dv[None, :]
            )
            tl.store(RS_addr + rs_off, RS_tile)


@triton.autotune(configs=_SPD_CONFIGS_SCAN, key=_SPD_KEY_SCAN)
@triton.jit
def spd_bwd_chunk_scan_kernel(
    RS_local_addr,
    RZ_local_addr,
    RS_suffix_addr,
    RZ_suffix_addr,
    H,
    NC: tl.constexpr,
    Q_TOTAL: tl.constexpr,
    d_v: tl.constexpr,
):
    """Reverse exclusive cumsum along NC. Grid: (B*H, Q_TOTAL)."""
    pid_bh = tl.program_id(0)
    q_idx = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_nc = tl.arange(0, NC)
    rev_nc = NC - 1 - offs_nc
    offs_dv = tl.arange(0, d_v)

    z_off = b * H * NC * Q_TOTAL + h * NC * Q_TOTAL + rev_nc * Q_TOTAL + q_idx
    Z = tl.load(RZ_local_addr + z_off)
    Z_cum = tl.cumsum(Z, axis=0)
    tl.store(RZ_suffix_addr + z_off, Z_cum - Z)

    s_off = (
        b * H * NC * Q_TOTAL * d_v
        + h * NC * Q_TOTAL * d_v
        + rev_nc[:, None] * Q_TOTAL * d_v
        + q_idx * d_v
        + offs_dv[None, :]
    )
    S = tl.load(RS_local_addr + s_off)
    S_cum = tl.cumsum(S, axis=0)
    tl.store(RS_suffix_addr + s_off, S_cum - S)


@triton.autotune(
    configs=_SPD_CONFIGS_d_v,
    key=_SPD_KEY,
    prune_configs_by={"early_config_prune": _spd_dv_prune},
)
@triton.jit
def spd_bwd_chunk_output_dq_kernel(
    Q_addr,
    K_addr,
    V_addr,
    dN_addr,
    dD_addr,
    S_pref_addr,
    Z_pref_addr,
    dQ_addr,
    H,
    T,
    NC,
    C: tl.constexpr,
    ME: tl.constexpr,
    E: tl.constexpr,
    M: tl.constexpr,
    d_v: tl.constexpr,
    Q_PER_G: tl.constexpr,
    d_v_BLK: tl.constexpr,
):
    """dq for one chunk (M-aware). Inter via fwd states + intra causal.

    Per-group decomposition:
        dq_g[t, l] = 2 sum_m (q_g[t]. k_g[m]) (<dN[t], v[m]> + dD[t]) k_g[m, l]

    Inter: 2 sum_d dN[t,d] * (sum_i Qt_g[c,i] Sp_g[i,l,d]) + 2 dD * (Z_g @ q_g)[l]
    Intra: 2 (beta_c @ k_g)[t, l]  where beta = (q_g . k_g) (dN.v + dD), causal.
    """
    pid_bh = tl.program_id(0)
    pid_nc = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_c = tl.arange(0, C)
    offs_e = tl.arange(0, E)
    offs_q_pg = tl.arange(0, Q_PER_G)
    t_idx = pid_nc * C + offs_c
    Q_TOTAL: tl.constexpr = M * Q_PER_G

    causal = offs_c[:, None] >= offs_c[None, :]

    dd_off = b * T * H + t_idx * H + h
    dD = tl.load(dD_addr + dd_off).to(tl.float32)  # (C,)

    # dN_v: (C, C). Computed once per (chunk, group) — but actually dN_v
    # depends on V which is same for all groups. We compute once by tiling d_v.
    dN_v = tl.zeros((C, C), dtype=tl.float32)
    for dv_id in tl.static_range(d_v // d_v_BLK):
        offs_dv = dv_id * d_v_BLK + tl.arange(0, d_v_BLK)
        v_off = b * T * H * d_v + t_idx[:, None] * H * d_v + h * d_v + offs_dv[None, :]
        V_tile = tl.load(V_addr + v_off).to(tl.float32)
        dN_off = v_off
        dN_tile = tl.load(dN_addr + dN_off).to(tl.float32)
        dN_v += tl.dot(dN_tile, tl.trans(V_tile), allow_tf32=False)

    for m_id in tl.static_range(M):
        # Q_g, K_g
        qk_g_off = (
            b * T * H * ME
            + t_idx[:, None] * H * ME
            + h * ME
            + m_id * E
            + offs_e[None, :]
        )
        Q_g = tl.load(Q_addr + qk_g_off).to(tl.float32)
        K_g = tl.load(K_addr + qk_g_off).to(tl.float32)

        # ---- Intra dq_g ----
        A_g = tl.dot(Q_g, tl.trans(K_g), allow_tf32=False)  # (C, C)
        beta = A_g * (dN_v + dD[:, None])
        beta_c = tl.where(causal, beta, 0.0)
        dq_intra_g = 2.0 * tl.dot(beta_c, K_g, allow_tf32=False)  # (C, E)

        # ---- Inter dq_g ----
        # M_pre[c, ij] = sum_d dN[c, d] Sp_g[ij, d]; tile d_v
        M_pre = tl.zeros((C, Q_PER_G), dtype=tl.float32)
        for dv_id in tl.static_range(d_v // d_v_BLK):
            offs_dv = dv_id * d_v_BLK + tl.arange(0, d_v_BLK)
            dn_off = (
                b * T * H * d_v + t_idx[:, None] * H * d_v + h * d_v + offs_dv[None, :]
            )
            dN_tile = tl.load(dN_addr + dn_off).to(tl.float32)  # (C, d_v_BLK)
            sp_g_off = (
                b * H * NC * Q_TOTAL * d_v
                + h * NC * Q_TOTAL * d_v
                + pid_nc * Q_TOTAL * d_v
                + (m_id * Q_PER_G + offs_q_pg)[:, None] * d_v
                + offs_dv[None, :]
            )
            Sp_g_tile = tl.load(S_pref_addr + sp_g_off).to(
                tl.float32
            )  # (Q_PER_G, d_v_BLK)
            M_pre += tl.dot(dN_tile, tl.trans(Sp_g_tile), allow_tf32=False)

        M_3d = tl.reshape(M_pre, (C, E, E))
        dq_inter_N_g = 2.0 * tl.sum(Q_g[:, :, None] * M_3d, axis=1)  # (C, E)

        # dq_inter_D_g: 2 dD[c] (Z_g @ q_g)[l]  with Z_g viewed as (E, E)
        zp_g_off = (
            b * H * NC * Q_TOTAL
            + h * NC * Q_TOTAL
            + pid_nc * Q_TOTAL
            + m_id * Q_PER_G
            + offs_q_pg
        )
        Zp_g = tl.load(Z_pref_addr + zp_g_off).to(tl.float32)
        Z_3d = tl.reshape(Zp_g, (E, E))
        QZ_g = tl.dot(Q_g, Z_3d, allow_tf32=False)
        dq_inter_D_g = 2.0 * dD[:, None] * QZ_g

        dq_g = dq_intra_g + dq_inter_N_g + dq_inter_D_g

        dq_g_off = (
            b * T * H * ME
            + t_idx[:, None] * H * ME
            + h * ME
            + m_id * E
            + offs_e[None, :]
        )
        tl.store(dQ_addr + dq_g_off, dq_g.to(dQ_addr.dtype.element_ty))


@triton.autotune(
    configs=_SPD_CONFIGS_d_v,
    key=_SPD_KEY,
    prune_configs_by={"early_config_prune": _spd_dv_prune},
)
@triton.jit
def spd_bwd_chunk_output_dkdv_kernel(
    q_addr,
    k_addr,
    v_addr,
    dN_addr,
    dD_addr,
    RS_suff_addr,
    RZ_suff_addr,
    dK_addr,
    dV_addr,
    H,
    T,
    num_chunks,
    chunk_size: tl.constexpr,
    ME: tl.constexpr,
    E: tl.constexpr,
    num_groups: tl.constexpr,
    d_v: tl.constexpr,
    Q_PER_G: tl.constexpr,
    d_v_BLK: tl.constexpr,
):
    """dk and dv for one chunk (M-aware). Inter via reverse states + intra anti-causal.

    Per-group dk: same beta as dq but transposed reduction (sum over t).
    dv: cross-group accumulated A_total^2 in masked attention pattern, plus
    inter via RS_g per group.
    """
    pid_bh = tl.program_id(0)
    pid_nc = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_c = tl.arange(0, chunk_size)
    offs_e = tl.arange(0, E)
    offs_q_pg = tl.arange(0, Q_PER_G)
    t_idx = pid_nc * chunk_size + offs_c
    Q_TOTAL: tl.constexpr = num_groups * Q_PER_G

    causal = offs_c[:, None] >= offs_c[None, :]

    dd_off = b * T * H + t_idx * H + h
    dD = tl.load(dD_addr + dd_off).to(tl.float32)

    # dN_v (C, C) — same as in dq kernel
    dN_v = tl.zeros((chunk_size, chunk_size), dtype=tl.float32)
    for dv_id in tl.static_range(d_v // d_v_BLK):
        offs_dv = dv_id * d_v_BLK + tl.arange(0, d_v_BLK)
        v_off = b * T * H * d_v + t_idx[:, None] * H * d_v + h * d_v + offs_dv[None, :]
        V_tile = tl.load(v_addr + v_off).to(tl.float32)
        dN_tile = tl.load(dN_addr + v_off).to(tl.float32)
        dN_v += tl.dot(dN_tile, tl.trans(V_tile), allow_tf32=False)

    # A_total = sum_g (q_g . k_g)^2 (for dv intra)
    A_total = tl.zeros((chunk_size, chunk_size), dtype=tl.float32)
    for m_id in tl.static_range(num_groups):
        qk_g_off = (
            b * T * H * ME
            + t_idx[:, None] * H * ME
            + h * ME
            + m_id * E
            + offs_e[None, :]
        )
        Q_g = tl.load(q_addr + qk_g_off).to(tl.float32)
        K_g = tl.load(k_addr + qk_g_off).to(tl.float32)
        A_g = tl.dot(Q_g, tl.trans(K_g), allow_tf32=False)
        A_total += A_g * A_g
    A_total_c = tl.where(causal, A_total, 0.0)  # (C, C)

    # ----- dv = inter + intra (d_v tiled) -----
    for dv_id in tl.static_range(d_v // d_v_BLK):
        offs_dv = dv_id * d_v_BLK + tl.arange(0, d_v_BLK)
        v_off = b * T * H * d_v + t_idx[:, None] * H * d_v + h * d_v + offs_dv[None, :]
        dN_tile = tl.load(dN_addr + v_off).to(tl.float32)  # (C, d_v_BLK)

        # dv_intra_tile = A_total_c^T @ dN_tile  (C, d_v_BLK)
        dv_intra_tile = tl.dot(tl.trans(A_total_c), dN_tile, allow_tf32=False)

        # dv_inter_tile: per-group
        dv_inter_tile = tl.zeros((chunk_size, d_v_BLK), dtype=tl.float32)
        for m_id in tl.static_range(num_groups):
            k_g_off = (
                b * T * H * ME
                + t_idx[:, None] * H * ME
                + h * ME
                + m_id * E
                + offs_e[None, :]
            )
            K_g = tl.load(k_addr + k_g_off).to(tl.float32)
            K_outer = K_g[:, :, None] * K_g[:, None, :]
            K_flat = tl.reshape(K_outer, (chunk_size, Q_PER_G))
            rs_g_off = (
                b * H * num_chunks * Q_TOTAL * d_v
                + h * num_chunks * Q_TOTAL * d_v
                + pid_nc * Q_TOTAL * d_v
                + (m_id * Q_PER_G + offs_q_pg)[:, None] * d_v
                + offs_dv[None, :]
            )
            RS_g_tile = tl.load(RS_suff_addr + rs_g_off).to(tl.float32)
            dv_inter_tile += tl.dot(K_flat, RS_g_tile, allow_tf32=False)

        dv_tile = dv_inter_tile + dv_intra_tile
        tl.store(dV_addr + v_off, dv_tile.to(dV_addr.dtype.element_ty))

    # ----- dk per group -----
    for m_id in tl.static_range(num_groups):
        qk_g_off = (
            b * T * H * ME
            + t_idx[:, None] * H * ME
            + h * ME
            + m_id * E
            + offs_e[None, :]
        )
        Q_g = tl.load(q_addr + qk_g_off).to(tl.float32)
        K_g = tl.load(k_addr + qk_g_off).to(tl.float32)

        # ---- Intra dk_g ----
        A_g = tl.dot(Q_g, tl.trans(K_g), allow_tf32=False)
        beta = A_g * (dN_v + dD[:, None])
        beta_c = tl.where(causal, beta, 0.0)
        dk_intra_g = 2.0 * tl.dot(tl.trans(beta_c), Q_g, allow_tf32=False)  # (C, E)

        # ---- Inter dk_g ----
        # dk_inter_N_g: 2 sum_i K_g[m,i] sum_d v[m,d] RS_g[l, i, d]
        # Compute U_g[m, q] = sum_d V_tile[m, d] RS_g[q, d]; tile d_v
        U_g = tl.zeros((chunk_size, Q_PER_G), dtype=tl.float32)
        for dv_id in tl.static_range(d_v // d_v_BLK):
            offs_dv = dv_id * d_v_BLK + tl.arange(0, d_v_BLK)
            v_off = (
                b * T * H * d_v + t_idx[:, None] * H * d_v + h * d_v + offs_dv[None, :]
            )
            V_tile = tl.load(v_addr + v_off).to(tl.float32)
            rs_g_off = (
                b * H * num_chunks * Q_TOTAL * d_v
                + h * num_chunks * Q_TOTAL * d_v
                + pid_nc * Q_TOTAL * d_v
                + (m_id * Q_PER_G + offs_q_pg)[:, None] * d_v
                + offs_dv[None, :]
            )
            RS_g_tile = tl.load(RS_suff_addr + rs_g_off).to(tl.float32)
            U_g += tl.dot(V_tile, tl.trans(RS_g_tile), allow_tf32=False)
        U_3d = tl.reshape(U_g, (chunk_size, E, E))
        dk_inter_N_g = 2.0 * tl.sum(K_g[:, None, :] * U_3d, axis=2)  # (C, E)

        # dk_inter_D_g: 2 sum_i K_g[m,i] RZ_g[i, l]
        rz_g_off = (
            b * H * num_chunks * Q_TOTAL
            + h * num_chunks * Q_TOTAL
            + pid_nc * Q_TOTAL
            + m_id * Q_PER_G
            + offs_q_pg
        )
        RZ_g = tl.load(RZ_suff_addr + rz_g_off).to(tl.float32)
        RZ_3d = tl.reshape(RZ_g, (E, E))
        dk_inter_D_g = 2.0 * tl.dot(K_g, RZ_3d, allow_tf32=False)

        dk_g = dk_intra_g + dk_inter_N_g + dk_inter_D_g
        dk_g_off = qk_g_off
        tl.store(dK_addr + dk_g_off, dk_g.to(dK_addr.dtype.element_ty))


def chunk_kata_spd_bwd(
    q_hat,
    k_hat,
    v,
    do,
    o,
    D,
    S_prefix,
    Z_prefix,
    chunk_size: int = 32,
    num_groups: int = 1,
):
    B, T, H, ME = q_hat.shape
    d_v = v.shape[-1]
    M = num_groups
    E = ME // M
    Q_PER_G = E * E
    Q_TOTAL = M * Q_PER_G
    C = chunk_size
    NC = T // C
    dev = q_hat.device

    Df = D.float().clamp_min(1e-30)
    dN = (do.float() / Df.unsqueeze(-1)).contiguous()
    dD = (-(do.float() * o.float()).sum(dim=-1) / Df).contiguous()

    RS_local = torch.empty(B, H, NC, Q_TOTAL, d_v, device=dev, dtype=torch.float32)
    RZ_local = torch.empty(B, H, NC, Q_TOTAL, device=dev, dtype=torch.float32)
    RS_suffix = torch.empty_like(RS_local)
    RZ_suffix = torch.empty_like(RZ_local)

    dq = torch.empty_like(q_hat)
    dk = torch.empty_like(k_hat)
    dv = torch.empty_like(v)

    spd_bwd_chunk_state_kernel[(B * H, NC)](
        q_hat,
        dN,
        dD,
        RS_local,
        RZ_local,
        H,
        T,
        NC,
        C=C,
        ME=ME,
        E=E,
        M=M,
        d_v=d_v,
        Q_PER_G=Q_PER_G,
    )

    spd_bwd_chunk_scan_kernel[(B * H, Q_TOTAL)](
        RS_local,
        RZ_local,
        RS_suffix,
        RZ_suffix,
        H,
        NC=NC,
        Q_TOTAL=Q_TOTAL,
        d_v=d_v,
    )

    spd_bwd_chunk_output_dq_kernel[(B * H, NC)](
        q_hat,
        k_hat,
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
        ME=ME,
        E=E,
        M=M,
        d_v=d_v,
        Q_PER_G=Q_PER_G,
    )

    spd_bwd_chunk_output_dkdv_kernel[(B * H, NC)](
        q_hat,
        k_hat,
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
        ME=ME,
        E=E,
        M=M,
        d_v=d_v,
        Q_PER_G=Q_PER_G,
    )

    return dq, dk, dv


# =====================================================================
# autograd.Function wrapper
# =====================================================================


class ChunkKataSPDFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_hat, k_hat, v, chunk_size, num_groups, eps):
        O, D, S_prefix, Z_prefix, _ = chunk_kata_spd_fwd(
            q_hat,
            k_hat,
            v,
            chunk_size=chunk_size,
            num_groups=num_groups,
            eps=eps,
        )
        ctx.save_for_backward(q_hat, k_hat, v, O, D, S_prefix, Z_prefix)
        ctx.chunk_size = chunk_size
        ctx.num_groups = num_groups
        ctx.eps = eps
        return O

    @staticmethod
    def backward(ctx, do):
        q_hat, k_hat, v, O, D, S_prefix, Z_prefix = ctx.saved_tensors
        dq, dk, dv = chunk_kata_spd_bwd(
            q_hat,
            k_hat,
            v,
            do,
            O,
            D,
            S_prefix,
            Z_prefix,
            chunk_size=ctx.chunk_size,
            num_groups=ctx.num_groups,
        )
        return dq, dk, dv, None, None, None


def chunk_kata_spd(
    q_hat,
    k_hat,
    v,
    chunk_size: int = 32,
    num_groups: int = 1,
    eps: float = 1e-6,
    initial_state=None,
    output_final_state: bool = False,
):
    """Chunked SPD linear attention. Block sizes (d_v_BLK, num_warps,
    num_stages) are picked by triton.autotune at first call per shape."""
    if initial_state is not None or output_final_state:
        O, _, _, _, final = chunk_kata_spd_fwd(
            q_hat,
            k_hat,
            v,
            chunk_size=chunk_size,
            num_groups=num_groups,
            eps=eps,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )
        return O, final
    return ChunkKataSPDFunction.apply(
        q_hat,
        k_hat,
        v,
        chunk_size,
        num_groups,
        eps,
    )


def kata_spd_recurrent_step(
    q_new, k_new, v_new, state_S, state_Z, num_groups=1, eps=1e-6
):
    """Single-token decoding step. Pure pytorch."""
    B, H, ME = k_new.shape
    M = num_groups
    E = ME // M
    Q_PER_G = E * E
    Q_TOTAL = M * Q_PER_G

    qf = q_new.float().reshape(B, H, M, E)
    kf = k_new.float().reshape(B, H, M, E)
    vf = v_new.float()

    kk = (kf.unsqueeze(-1) * kf.unsqueeze(-2)).reshape(B, H, Q_TOTAL)  # (B, H, M*E^2)
    qq = (qf.unsqueeze(-1) * qf.unsqueeze(-2)).reshape(B, H, Q_TOTAL)

    state_S = state_S + kk.unsqueeze(-1) * vf.unsqueeze(-2)  # (B, H, Q_TOTAL, d_v)
    state_Z = state_Z + kk

    N = (qq.unsqueeze(-1) * state_S).sum(dim=-2)
    D = (qq * state_Z).sum(dim=-1) + eps
    o_new = (N / D.unsqueeze(-1)).to(q_new.dtype)
    return o_new, state_S, state_Z
