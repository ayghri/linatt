"""KATA: Kernelized Linear Attention with cone-aware feature maps.

Importing this module registers the KATA model with HF Auto* so that
    AutoConfig.for_model(model_type='kata', ...)
and
    AutoModelForCausalLM.from_config(...)
work out of the box.
"""

from __future__ import annotations

# Lightweight, no-transformers imports always available
from kata.references.feature_maps import FeatureMap, positive, lorentz, spd_concat

# Heavy imports (transformers/fla.models) loaded only when actually accessed,
# so kernel-only consumers (kata.spd_kernels, benchmarks) don't pay the cost.
_LAZY = {
    "KataConfig": ("kata.configuration", "KataConfig"),
    "KataModel": ("kata.modeling", "KataModel"),
    "KataForCausalLM": ("kata.modeling", "KataForCausalLM"),
    "KataPreTrainedModel": ("kata.modeling", "KataPreTrainedModel"),
    "KataAttention": ("kata.layer", "KataAttention"),
}


def __getattr__(name: str):
    if name in _LAZY:
        from importlib import import_module

        mod_name, attr = _LAZY[name]
        mod = import_module(mod_name)
        val = getattr(mod, attr)
        globals()[name] = val
        # On first access of any HF-integrated symbol, register Auto* classes.
        _maybe_register()
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_REGISTERED = False


def _maybe_register():
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
    from kata.configuration import KataConfig
    from kata.modeling import KataForCausalLM, KataModel

    AutoConfig.register(KataConfig.model_type, KataConfig, exist_ok=True)
    AutoModel.register(KataConfig, KataModel, exist_ok=True)
    AutoModelForCausalLM.register(KataConfig, KataForCausalLM, exist_ok=True)


__all__ = [
    "KataConfig",
    "KataForCausalLM",
    "KataModel",
    "KataPreTrainedModel",
    "KataAttention",
    "FeatureMap",
    "positive",
    "lorentz",
    "spd_concat",
]
