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

    _APPLIED = True


# Auto-apply on import for ergonomic `import fla_patches` usage.
apply()
