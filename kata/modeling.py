"""KataForCausalLM. Subclasses fla.LinearAttention modeling, swaps the
attention layer for KataAttention.

We don't reimplement the model body, MLP, embedding tying, generation, or
loss — those are identical to the FLA LinearAttention LM. The only change is
that the attention block instantiates KataAttention instead of LinearAttention,
which feeds psi(q), psi(k) (the KATA feature map) into the FLA chunk kernel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn

from fla.layers.attn import Attention
from fla.models.linear_attn.modeling_linear_attn import (
    LinearAttentionForCausalLM,
    LinearAttentionModel,
    LinearAttentionPreTrainedModel,
    LinearAttentionBlock,
)
from fla.models.modeling_layers import GradientCheckpointingLayer
from fla.modules import RMSNorm
from fla.modules import GatedMLP

from kata.configuration import KataConfig
from kata.layer import KataAttention

if TYPE_CHECKING:
    pass


class KataBlock(GradientCheckpointingLayer):
    """Mirror of fla LinearAttentionBlock, with KataAttention as the attn op."""

    def __init__(self, config: KataConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        norm_cls = RMSNorm if config.fuse_norm else nn.RMSNorm
        self.attn_norm = norm_cls(config.hidden_size, eps=config.norm_eps)

        if config.attn is not None and layer_idx in config.attn["layers"]:
            self.attn = Attention(
                hidden_size=config.hidden_size,
                num_heads=config.attn["num_heads"],
                num_kv_heads=config.attn["num_kv_heads"],
                qkv_bias=config.attn["qkv_bias"],
                window_size=config.attn["window_size"],
                rope_theta=config.attn["rope_theta"],
                max_position_embeddings=config.max_position_embeddings,
                layer_idx=layer_idx,
            )
        else:
            self.attn = KataAttention(
                mode=config.attn_mode,
                hidden_size=config.hidden_size,
                expand_k=config.expand_k,
                expand_v=config.expand_v,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                feature_map=config.feature_map,
                spd_num_groups=config.spd_num_groups,
                spd_use_kernel=config.spd_use_kernel,
                spd_chunk_size=config.spd_chunk_size,
                use_delta=config.use_delta,
                delta_scale=config.delta_scale,
                delta_normalize=config.delta_normalize,
                delta_state=config.delta_state,
                use_offset_gate=config.use_offset_gate,
                use_decay=config.use_decay,
                feature_map_eps=config.feature_map_eps,
                qk_norm=config.qk_norm,
                norm_q=config.norm_q,
                norm_k=config.norm_k,
                use_short_conv=config.use_short_conv,
                conv_size=config.conv_size,
                conv_bias=config.conv_bias,
                use_rope=config.use_rope,
                rope_theta=config.rope_theta,
                rope_group=config.rope_group,
                max_position_embeddings=config.max_position_embeddings,
                elementwise_affine=config.elementwise_affine,
                norm_eps=config.norm_eps,
                layer_idx=layer_idx,
            )

        self.mlp_norm = norm_cls(config.hidden_size, eps=config.norm_eps)
        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        output_attentions=False,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )
        if self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(
                hidden_states, residual, True
            )
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states
        return (hidden_states, attentions, past_key_values)


class KataPreTrainedModel(LinearAttentionPreTrainedModel):
    config_class = KataConfig
    base_model_prefix = "model"


class KataModel(LinearAttentionModel):
    """Same as LinearAttentionModel, but with KataBlock for each layer.

    We override config_class and the layer factory; the rest of the body
    (embeddings, final norm, forward) is reused.
    """

    config_class = KataConfig

    def __init__(self, config: KataConfig):
        # Call PreTrainedModel.__init__ via the parent's parent to avoid
        # LinearAttentionModel's __init__ creating LinearAttentionBlocks.
        LinearAttentionPreTrainedModel.__init__(self, config)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                KataBlock(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        norm_cls = RMSNorm if config.fuse_norm else nn.RMSNorm
        self.norm = norm_cls(config.hidden_size, eps=config.norm_eps)

        self.gradient_checkpointing = False
        self.post_init()


class KataForCausalLM(LinearAttentionForCausalLM):
    """LinearAttentionForCausalLM with KataModel as the body."""

    config_class = KataConfig

    def __init__(self, config: KataConfig):
        # Same trick: skip the parent's body construction so we can plug our
        # KataModel in without instantiating LinearAttention layers first.
        LinearAttentionPreTrainedModel.__init__(self, config)

        self.model = KataModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
        self.criterion = None
        self.post_init()
