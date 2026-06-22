"""Chunked manual backward for SPD attention. Pure pytorch, no autograd graph.

Math (using SPD identity <psi(g), psi(h)> = (g.h)^2 throughout):

    Forward
        A[t, m] = q_t . k_m
        N[t, d] = sum_{m<=t} A[t,m]^2 v[m, d]
        D[t]    = sum_{m<=t} A[t,m]^2 + eps
        o[t, d] = N[t, d] / D[t]

    Local backward
        dN[t, d] = do[t, d] / D[t]
        dD[t]    = -<do[t], o[t]> / D[t]

    Token-level gradients (closed form)
        dq[t, l] = 2 * sum_{m<=t} A[t,m] * (<dN[t], v[m]> + dD[t]) * k[m, l]
        dk[m, l] = 2 * sum_{t>=m} A[t,m] * (<dN[t], v[m]> + dD[t]) * q[t, l]
        dv[m, d] =     sum_{t>=m} A[t,m]^2 * dN[t, d]

    Chunked decomposition: split into inter-chunk (uses cumulative states) +
    intra-chunk (causal C x C attention pattern).

    Inter-chunk dq uses forward states S_prefix, Z_prefix (already computed).
    Inter-chunk dk, dv use REVERSE states (suffix sums) RS, RZ:
        RS[c, i, j, d] = sum_{t in chunks > c} q[t, i] q[t, j] dN[t, d]
        RZ[c, i, j]    = sum_{t in chunks > c} q[t, i] q[t, j] dD[t]
    These are exclusive suffix sums over NC.

This file is the running reference for the triton bwd kernels. Validates
against autograd-through-recurrent.
"""
import torch


def spd_chunked_bwd(
    q_hat: torch.Tensor,
    k_hat: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    o: torch.Tensor,
    D: torch.Tensor,
    chunk_size: int = 64,
):
    """Chunked manual backward.

    Args:
        q_hat:  (B, T, H, E)
        k_hat:  (B, T, H, E)
        v:      (B, T, H, DV)
        do:     (B, T, H, DV)  upstream gradient on o
        o:      (B, T, H, DV)  forward output (saved by fwd)
        D:      (B, T, H)      forward denominator (saved by fwd)
        chunk_size: must divide T; T/C must be a power of 2.
    Returns:
        dq, dk, dv  matching shapes of q_hat, k_hat, v
    """
    B, T, H, E = q_hat.shape
    DV = v.shape[-1]
    C = chunk_size
    if T % C != 0:
        raise ValueError(f"T={T} not divisible by C={C}")
    NC = T // C

    qf = q_hat.float()
    kf = k_hat.float()
    vf = v.float()
    dof = do.float()
    of = o.float()
    Df = D.float().clamp_min(1e-30)

    # Local grads
    dN = dof / Df.unsqueeze(-1)  # (B, T, H, DV)
    dD = -(dof * of).sum(dim=-1) / Df  # (B, T, H)

    # Reshape to chunks
    q_ch = qf.view(B, NC, C, H, E)
    k_ch = kf.view(B, NC, C, H, E)
    v_ch = vf.view(B, NC, C, H, DV)
    dN_ch = dN.view(B, NC, C, H, DV)
    dD_ch = dD.view(B, NC, C, H)

    # ---- Forward states needed for dq inter-chunk (recompute here for
    # simplicity; the kernel will reuse what fwd already saved) ----
    kk_ch = k_ch.unsqueeze(-1) * k_ch.unsqueeze(-2)  # (B,NC,C,H,E,E)
    Z_local = kk_ch.sum(dim=2)  # (B,NC,H,E,E)
    S_local = torch.einsum(
        "bnchij,bncha->bnhija", kk_ch, v_ch
    )  # (B,NC,H,E,E,DV)
    Z_prefix = Z_local.cumsum(dim=1) - Z_local  # exclusive
    S_prefix = S_local.cumsum(dim=1) - S_local

    # ---- Reverse states for dk, dv inter-chunk ----
    qq_ch = q_ch.unsqueeze(-1) * q_ch.unsqueeze(-2)  # (B,NC,C,H,E,E)
    # RS_local[b, nc, h, i, j, d] = sum_c qq_ch[c,i,j] * dN_ch[c,d]
    RS_local = torch.einsum(
        "bnchij,bncha->bnhija", qq_ch, dN_ch
    )  # (B,NC,H,E,E,DV)
    # RZ_local[b, nc, h, i, j] = sum_c qq_ch[c,i,j] * dD_ch[c]
    RZ_local = torch.einsum("bnchij,bnch->bnhij", qq_ch, dD_ch)  # (B,NC,H,E,E)

    # Exclusive SUFFIX sum (reverse cumulative: chunk c gets sum over c' > c)
    # Equivalent to flipping NC, doing exclusive prefix, flipping back.
    def reverse_excl_cumsum(x, dim):
        flipped = torch.flip(x, dims=(dim,))
        prefix = flipped.cumsum(dim=dim) - flipped
        return torch.flip(prefix, dims=(dim,))

    RS_suffix = reverse_excl_cumsum(RS_local, dim=1)  # (B,NC,H,E,E,DV)
    RZ_suffix = reverse_excl_cumsum(RZ_local, dim=1)  # (B,NC,H,E,E)

    # ============================================================
    # dq: forward-causal. dq[t] uses inter (S_prefix, Z_prefix at chunk c)
    # + intra (causal pattern within chunk).
    # ============================================================
    # Inter dq from N: dq_inter_N[t, l] = 2 sum_i q[t,i] (sum_d dN[t,d] S_prev[i,l,d])
    # = 2 sum_d dN[t,d] sum_i q[t,i] S_prev[i,l,d]
    # = 2 (q[t] @ S_prev[:,l,:]).dN[t]   ... let's just einsum:
    # dq_inter_N[b,n,c,h,l] = 2 * sum_{i,d} q_ch[c,i] dN_ch[c,d] S_prefix[i,l,d]
    dq_inter_N = 2.0 * torch.einsum(
        "bnchi,bncha,bnhila->bnchl",
        q_ch,
        dN_ch,
        S_prefix,
    )
    # Inter dq from D: 2 dD[t] (Z_prev @ q[t])[l] = 2 dD[t] sum_i Z_prev[i,l] q[t,i]
    dq_inter_D = 2.0 * torch.einsum(
        "bnch,bnchi,bnhil->bnchl",
        dD_ch,
        q_ch,
        Z_prefix,
    )
    dq_inter = dq_inter_N + dq_inter_D  # (B,NC,C,H,E)

    # Intra dq:
    #   beta[t, m] = A_in[t, m] * (<dN[t], v[m]> + dD[t])  with causal mask t>=m (incl)
    #   dq_intra[t, l] = 2 sum_m beta[t, m] k[m, l]
    A_in = torch.einsum("bnchk,bnshk->bnhcs", q_ch, k_ch)  # (B,NC,H,C,C)
    dN_v = torch.einsum("bncha,bnsha->bnhcs", dN_ch, v_ch)  # (B,NC,H,C,C)
    beta = A_in * (dN_v + dD_ch.transpose(-1, -2).unsqueeze(-1))  # (B,NC,H,C,C)

    causal = torch.tril(torch.ones(C, C, device=q_hat.device, dtype=torch.bool))
    beta_causal = beta.masked_fill(~causal, 0.0)  # (B,NC,H,C,C)
    dq_intra = 2.0 * torch.einsum(
        "bnhcs,bnshl->bnchl", beta_causal, k_ch
    )  # (B,NC,C,H,E)

    dq = (dq_inter + dq_intra).reshape(B, T, H, E).to(q_hat.dtype)

    # ============================================================
    # dk, dv: reverse-causal. dk[m] / dv[m] use inter (RS_suffix, RZ_suffix
    # at chunk c+1) + intra (anti-causal pattern within chunk, t>=m).
    # ============================================================

    # Inter dk:
    # dk_inter_N[m, l] = 2 sum_i k[m,i] sum_d v[m,d] RS_next[l, i, d]
    # dk_inter_D[m, l] = 2 sum_i k[m,i] RZ_next[i, l]
    # NOTE: RS, RZ are symmetric in (i,j) since q outer q is symmetric,
    # but we keep the (l, i) / (i, l) indexing explicit for clarity.
    dk_inter_N = 2.0 * torch.einsum(
        "bnchi,bncha,bnhlia->bnchl",
        k_ch,
        v_ch,
        RS_suffix,
    )
    dk_inter_D = 2.0 * torch.einsum(
        "bnchi,bnhil->bnchl",
        k_ch,
        RZ_suffix,
    )
    dk_inter = dk_inter_N + dk_inter_D

    # Intra dk: same beta pattern but transposed reduction (m fixed, sum over t)
    # mask: t >= m  (causal lower tri inclusive — same beta_causal, but
    # transposed for dk reduction)
    dk_intra = 2.0 * torch.einsum("bnhts,bnthl->bnshl", beta_causal, q_ch)
    # dk_intra[b,n,m,h,l] = 2 sum_t beta_causal[t, m] q[t, l]   ✓

    dk = (dk_inter + dk_intra).reshape(B, T, H, E).to(k_hat.dtype)

    # Inter dv:
    # dv_inter[m, d] = sum_{i,j} k[m,i] k[m,j] RS_suffix[i,j,d]
    dv_inter = torch.einsum(
        "bnchi,bnchj,bnhija->bncha",
        k_ch,
        k_ch,
        RS_suffix,
    )

    # Intra dv: dv_intra[m, d] = sum_{t>=m} A_in[t,m]^2 dN[t, d]
    A2_causal = (A_in * A_in).masked_fill(~causal, 0.0)  # (B,NC,H,C,C)
    dv_intra = torch.einsum("bnhts,bntha->bnsha", A2_causal, dN_ch)
    # dv_intra[b,n,m,h,d] = sum_t A2_causal[t, m] dN[t, d]              ✓

    dv = (dv_inter + dv_intra).reshape(B, T, H, DV).to(v.dtype)

    return dq, dk, dv
