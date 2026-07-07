"""KataConfig: HF PretrainedConfig for KATA models.

Mirrors fla.LinearAttentionConfig with additional KATA-specific fields:
- feature_map: 'positive' | 'lorentz' | 'spd_concat'
- spd_num_groups: M (only used when feature_map='spd_concat')
- feature_map_eps: epsilon for the 'positive' map
"""


from transformers.configuration_utils import PretrainedConfig


class KataConfig(PretrainedConfig):

    model_type = "kata"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        attn_mode: str = "chunk",
        hidden_size: int = 768,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        hidden_ratio: int | None = 4,
        intermediate_size: int | None = None,
        num_hidden_layers: int = 24,
        num_heads: int = 12,
        num_kv_heads: int | None = None,
        feature_map: str = "positive",
        spd_num_groups: int = 1,
        spd_use_kernel: bool = False,
        spd_chunk_size: int = 32,
        use_delta: bool = False,
        use_offset_gate: bool = False,
        use_decay: bool = False,
        feature_map_eps: float = 1e-6,
        qk_norm: bool = False,
        norm_q: str = "rmsnorm",
        norm_k: str = "rmsnorm",
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
        rope_group: bool = False,
        hidden_act: str = "swish",
        max_position_embeddings: int = 2048,
        elementwise_affine: bool | None = True,
        norm_eps: float = 1e-6,
        attn: dict | None = None,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        initializer_range: float = 0.02,
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        fuse_linear_cross_entropy: bool = False,
        use_l2warp: bool = False,
        vocab_size: int = 32000,
        **kwargs,
    ):
        self.attn_mode = attn_mode
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.feature_map = feature_map
        self.spd_num_groups = spd_num_groups
        self.spd_use_kernel = spd_use_kernel
        self.spd_chunk_size = spd_chunk_size
        self.use_delta = use_delta
        self.use_offset_gate = use_offset_gate
        self.use_decay = use_decay
        self.feature_map_eps = feature_map_eps
        self.qk_norm = qk_norm
        self.norm_q = norm_q
        self.norm_k = norm_k
        self.use_short_conv = use_short_conv
        self.use_rope = use_rope
        self.rope_theta = rope_theta
        self.rope_group = rope_group
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.elementwise_affine = elementwise_affine
        self.norm_eps = norm_eps
        self.attn = attn
        self.use_cache = use_cache
        self.initializer_range = initializer_range

        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy
        self.use_l2warp = use_l2warp
        self.vocab_size = vocab_size

        if fuse_cross_entropy and fuse_linear_cross_entropy:
            raise ValueError(
                "`fuse_cross_entropy` and `fuse_linear_cross_entropy` cannot both be True",
            )

        if attn is not None:
            if not isinstance(attn, dict):
                raise ValueError("attn must be a dictionary")
            if "layers" not in attn:
                raise ValueError('attn["layers"] required for hybrid attention')
            if "num_heads" not in attn:
                raise ValueError(
                    'attn["num_heads"] required for hybrid attention'
                )
            attn["num_kv_heads"] = attn.get("num_kv_heads", attn["num_heads"])
            attn["qkv_bias"] = attn.get("qkv_bias", False)
            attn["window_size"] = attn.get("window_size", None)
            attn["rope_theta"] = attn.get("rope_theta", 10000.0)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
