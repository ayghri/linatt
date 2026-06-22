"""Pure-pytorch SPD-attention references. Ground truth for kernel parity.

We use the FULL outer-product representation: psi_full(g) = vec(g g^T) of dim E^2.
This is equivalent to the lower-tri packed form with sqrt(2) on off-diagonals
in terms of inner products: <psi_full(g), psi_full(h)> = sum_{i,j} g_i g_j h_i h_j
= (g . h)^2. Both forms yield identical attention scores.

Causal convention: INCLUSIVE (q_t attends to k_{<=t}), matching FLA's
chunk_linear_attn so the bespoke kernel and the phase-1 spd_concat path agree.

Two references:
- spd_recurrent_ref: token-by-token loop. Slowest, no chunking, but trivially
  correct. Used as ground truth.
- spd_chunked_ref: chunked decomposition mirroring the triton kernel structure.
  No psi materialized in HBM (we still allocate (B,H,NC,E,E,DV) and
  (B,H,NC,E,E) chunk-boundary states, just like the kernel will).

Single group only (M=1). Multi-group is an outer concat over groups; trivial
to extend once single-group is verified.
"""

import torch


def spd_recurrent_ref(
    q_hat: torch.Tensor,
    k_hat: torch.Tensor,
    v: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Token-by-token SPD linear attention.

    Args:
        q_hat: (B, T, H, E)
        k_hat: (B, T, H, E)
        v:     (B, T, H, DV)
    Returns:
        o:     (B, T, H, DV)

    Forward (inclusive causal, matches FLA convention):
        S_t[i,j,d] = sum_{m<=t} k_hat[m,i] k_hat[m,j] v[m,d]    (E,E,DV)
        Z_t[i,j]   = sum_{m<=t} k_hat[m,i] k_hat[m,j]            (E,E)
        N_t[d]     = sum_{i,j} q_hat[t,i] q_hat[t,j] S_t[i,j,d]
        D_t        = sum_{i,j} q_hat[t,i] q_hat[t,j] Z_t[i,j] + eps
        o_t[d]     = N_t[d] / D_t
    """
    B, T, H, E = q_hat.shape
    DV = v.shape[-1]
    dtype = q_hat.dtype
    dev = q_hat.device

    # accumulate state in fp32 for numerical stability
    state_S = torch.zeros(B, H, E, E, DV, device=dev, dtype=torch.float32)
    state_Z = torch.zeros(B, H, E, E, device=dev, dtype=torch.float32)
    out = torch.empty(B, T, H, DV, device=dev, dtype=dtype)

    for t in range(T):
        kt = k_hat[:, t].float()  # (B, H, E)
        vt = v[:, t].float()  # (B, H, DV)
        qt = q_hat[:, t].float()  # (B, H, E)

        # update state
        kk = kt.unsqueeze(-1) * kt.unsqueeze(-2)  # (B, H, E, E)
        state_Z = state_Z + kk
        state_S = state_S + kk.unsqueeze(-1) * vt.unsqueeze(-2).unsqueeze(
            -2
        )  # (B,H,E,E,DV)

        # contract with q outer q
        qq = qt.unsqueeze(-1) * qt.unsqueeze(-2)  # (B, H, E, E)
        N = (qq.unsqueeze(-1) * state_S).sum(dim=(-2, -3))  # (B, H, DV)
        D = (qq * state_Z).sum(dim=(-1, -2)) + eps  # (B, H)
        out[:, t] = (N / D.unsqueeze(-1)).to(dtype)

    return out


def spd_chunked_ref(
    q_hat: torch.Tensor,
    k_hat: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int = 64,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Chunked SPD linear attention. Decomposition mirrors the triton kernel.

    Args:
        q_hat: (B, T, H, E)
        k_hat: (B, T, H, E)
        v:     (B, T, H, DV)
        chunk_size: C. T must be divisible by C.
    Returns:
        o:     (B, T, H, DV)
    """
    B, T, H, E = q_hat.shape
    DV = v.shape[-1]
    dtype = q_hat.dtype
    dev = q_hat.device
    C = chunk_size
    if T % C != 0:
        raise ValueError(f"T={T} must be divisible by chunk_size {C}")
    NC = T // C

    qf = q_hat.float()
    kf = k_hat.float()
    vf = v.float()

    # ---- Phase 1: per-chunk local statistics ----
    # k_chunked: (B, NC, C, H, E); k outer k -> (B, NC, C, H, E, E)
    k_ch = kf.view(B, NC, C, H, E)
    v_ch = vf.view(B, NC, C, H, DV)

    kk_ch = k_ch.unsqueeze(-1) * k_ch.unsqueeze(-2)  # (B,NC,C,H,E,E)
    Z_local = kk_ch.sum(dim=2)  # (B,NC,H,E,E)
    # S_local[b,nc,h,i,j,d] = sum_c kk_ch[b,nc,c,h,i,j] * v_ch[b,nc,c,h,d]
    S_local = torch.einsum(
        "bnchij,bncha->bnhija", kk_ch, v_ch
    )  # (B,NC,H,E,E,DV)

    # ---- Phase 2: exclusive prefix sum over NC ----
    # cumsum, then subtract local for exclusive
    Z_prefix = Z_local.cumsum(dim=1) - Z_local  # (B,NC,H,E,E)
    S_prefix = S_local.cumsum(dim=1) - S_local  # (B,NC,H,E,E,DV)

    # ---- Phase 3: per-chunk outputs ----
    q_ch = qf.view(B, NC, C, H, E)

    # Inter-chunk: sum_{i,j} q[c,i] q[c,j] S_prefix[i,j,d]
    qq_ch = q_ch.unsqueeze(-1) * q_ch.unsqueeze(-2)  # (B,NC,C,H,E,E)
    N_inter = torch.einsum(
        "bnchij,bnhija->bncha", qq_ch, S_prefix
    )  # (B,NC,C,H,DV)
    D_inter = torch.einsum("bnchij,bnhij->bnch", qq_ch, Z_prefix)  # (B,NC,C,H)

    # Intra-chunk: causal lower-tri (inclusive), via SPD identity
    #   <psi(q_a), psi(k_b)> = (q_a . k_b)^2
    qk = torch.einsum("bnchk,bndhk->bnhcd", q_ch, k_ch)  # (B,NC,H,C,C)
    A = qk * qk  # (B,NC,H,C,C)
    causal = torch.tril(
        torch.ones(C, C, device=dev, dtype=torch.bool)
    )  # inclusive
    A = A.masked_fill(~causal, 0.0)
    N_intra = torch.einsum("bnhcd,bndha->bncha", A, v_ch)  # (B,NC,C,H,DV)
    D_intra = A.sum(dim=-1).transpose(
        -1, -2
    )  # (B,NC,C,H)? sum over d -> (B,NC,H,C)
    # einsum mistake above; fix:
    D_intra = torch.einsum("bnhcd->bnch", A)  # (B,NC,C,H)

    N = N_inter + N_intra  # (B,NC,C,H,DV)
    D = D_inter + D_intra + eps  # (B,NC,C,H)

    o = N / D.unsqueeze(-1)
    return o.reshape(B, T, H, DV).to(dtype)
