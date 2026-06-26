import triton
import triton.language as tl


@triton.jit
def spd_chunk_state_kernel(
    k_addr,
    v_addr,
    s_addr,
    z_addr,
    H,
    T,
    NC,
    C: tl.constexpr,
    ME: tl.constexpr,
    E: tl.constexpr,
    M: tl.constexpr,
    DV: tl.constexpr,
    Q_PER_G: tl.constexpr,
    DV_BLK: tl.constexpr,
):
    """Per-chunk per-group state local statistics:
        S_local[c, m, q, d]  +=  K_outer_g[i, j]  *  V[c, d]
        Z_local[c, m, q]     +=  K_outer_g[i, j]
    where K_outer_g = K_g outer K_g flattened to E^2 = Q_PER_G.

    Loops: M groups (outer) -> DV tiles (inner). Each program does one
    (b, h, nc) chunk.
    Grid: (B*H, NC).

    State layout: S[B, H, NC, M*Q_PER_G, DV] flat-concat over M groups.
                  Z[B, H, NC, M*Q_PER_G] same.
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

    for m_id in tl.static_range(M):
        # K_g[c, e] = K[b, t_idx, h, m_id*E + e]
        k_off = (
            b * T * H * ME
            + t_idx[:, None] * H * ME
            + h * ME
            + m_id * E
            + offs_e[None, :]
        )
        K_g = tl.load(k_addr + k_off).to(tl.float32)  # (C, E)

        K_outer = K_g[:, :, None] * K_g[:, None, :]  # (C, E, E)
        K_flat = tl.reshape(K_outer, (C, Q_PER_G))  # (C, Q_PER_G)

        # Z_local for this group (E^2,)
        Z_g = tl.sum(K_flat, axis=0)
        z_off = (
            b * H * NC * Q_TOTAL
            + h * NC * Q_TOTAL
            + pid_nc * Q_TOTAL
            + m_id * Q_PER_G
            + offs_q_pg
        )
        tl.store(z_addr + z_off, Z_g)

        # DV-tiled S_local: S_g_tile = K_flat^T @ V_tile, store per tile.
        for dv_id in tl.static_range(DV // DV_BLK):
            offs_dv = dv_id * DV_BLK + tl.arange(0, DV_BLK)
            v_off = b * T * H * DV + t_idx[:, None] * H * DV + h * DV + offs_dv[None, :]
            V_tile = tl.load(v_addr + v_off).to(tl.float32)  # (C, DV_BLK)

            S_tile = tl.dot(
                tl.trans(K_flat), V_tile, allow_tf32=False
            )  # (Q_PER_G, DV_BLK)

            s_off = (
                b * H * NC * Q_TOTAL * DV
                + h * NC * Q_TOTAL * DV
                + pid_nc * Q_TOTAL * DV
                + (m_id * Q_PER_G + offs_q_pg)[:, None] * DV
                + offs_dv[None, :]
            )
            tl.store(s_addr + s_off, S_tile)
