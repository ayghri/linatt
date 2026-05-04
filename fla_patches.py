"""
Runtime patches over installed fla. Keep all upstream-fla deltas here so
`fla` itself stays untouched.

Apply with:
    import fla            # registers HF Auto* for fla model_types
    import fla_patches    # patches over fla, idempotent
    fla_patches.apply()

Patches:
- Mamba2ForCausalLM._tied_weights_keys: upstream is [] but the head weight
  IS tied via tie_word_embeddings=True; transformers' safetensors saver then
  refuses to save because it sees an undeclared shared tensor. Set the key
  to the same value GatedDeltaNet/DeltaNet use, so save/load roundtrips work.
"""

from __future__ import annotations

_APPLIED = False


def apply() -> None:
    global _APPLIED
    if _APPLIED:
        return

    from fla.models.mamba2.modeling_mamba2 import Mamba2ForCausalLM
    if list(getattr(Mamba2ForCausalLM, '_tied_weights_keys', [])) != ['lm_head.weight']:
        Mamba2ForCausalLM._tied_weights_keys = ['lm_head.weight']

    _patch_fla_attn_to_sdpa_flash()

    _APPLIED = True


def _patch_fla_attn_to_sdpa_flash() -> None:
    """Replace fla's flash_attn_func dependency with torch SDPA, FLASH backend forced.

    fla.layers.attn imports `flash_attn_func` from the flash-attn package and
    raises ImportError if missing. The flash-attn package has no torch 2.10
    prebuilt wheel today. Instead, drop in a shim that calls
    torch.nn.functional.scaled_dot_product_attention under
    `sdpa_kernel(SDPBackend.FLASH_ATTENTION)`, which uses PyTorch's bundled
    Flash-Attention 2 kernel on Ampere/Hopper bf16 — equivalent perf, no
    extra dependency.
    """
    import fla.layers.attn as _attn
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def _sdpa_flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None,
                              causal=False, window_size=(-1, -1), **kwargs):
        # flash_attn convention: (B, T, H, D); SDPA wants (B, H, T, D).
        q_ = q.transpose(1, 2)
        k_ = k.transpose(1, 2)
        v_ = v.transpose(1, 2)
        if window_size != (-1, -1):
            # Sliding-window not implemented in SDPA flash backend; fall back
            # to mem-efficient with a manual mask. The full-attn baseline
            # doesn't use windows so this branch is unused here.
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                o = F.scaled_dot_product_attention(
                    q_, k_, v_, is_causal=causal, dropout_p=dropout_p,
                    scale=softmax_scale,
                )
        else:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                o = F.scaled_dot_product_attention(
                    q_, k_, v_, is_causal=causal, dropout_p=dropout_p,
                    scale=softmax_scale,
                )
        return o.transpose(1, 2).contiguous()

    def _sdpa_flash_attn_varlen_func(*args, **kwargs):
        raise NotImplementedError(
            "varlen flash-attn is not supported by the SDPA shim; "
            "set cu_seqlens=None or install flash-attn upstream."
        )

    _attn.flash_attn_func = _sdpa_flash_attn_func
    _attn.flash_attn_varlen_func = _sdpa_flash_attn_varlen_func


# Auto-apply on import for ergonomic `import fla_patches` usage.
apply()
