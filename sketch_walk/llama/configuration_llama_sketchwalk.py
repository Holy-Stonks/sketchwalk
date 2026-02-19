"""
SketchWalk LLaMA Model Configuration

Modified from transformers and SeerAttention Llama configuration.
"""

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation


class SketchWalkLlamaConfig(PretrainedConfig):
    """
    Configuration class for SketchWalk-enabled LLaMA models.

    This extends the standard LLaMA configuration with SketchWalk-specific
    parameters for training-free sparse attention.

    Args:
        # Standard LLaMA parameters
        vocab_size (`int`, defaults to 32000):
            Vocabulary size of the LLaMA model.
        hidden_size (`int`, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, defaults to 32):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, optional):
            Number of key_value heads for Grouped Query Attention.
        max_position_embeddings (`int`, defaults to 2048):
            Maximum sequence length that this model might be used with.
        rope_theta (`float`, defaults to 10000.0):
            The base period of the RoPE embeddings.

        # SketchWalk-specific parameters
        sketchwalk_enabled (`bool`, defaults to True):
            Whether to enable SketchWalk sparse attention.
        sketchwalk_block_size (`int`, defaults to 64):
            Block size B for token-space sketching.
        sketchwalk_sketch_dim (`int`, defaults to 64):
            Sketch dimension k for feature-space sketching.
        sketchwalk_top_k_blocks (`int`, defaults to 16):
            Number of key blocks τ selected per query block.
        sketchwalk_sparsity_exponent (`int`, defaults to 8):
            Sparsity exponent s for sharpening attention distribution.
        sketchwalk_skip_first_n_layers (`int`, defaults to 2):
            Number of initial layers to skip (use dense attention).
        sketchwalk_hadamard_seed (`int`, defaults to 42):
            Random seed for reproducible Hadamard transform.
    """

    model_type = "llama_sketchwalk"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        base_model=None,
        # Standard LLaMA parameters
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        # SketchWalk-specific parameters
        sketchwalk_enabled=True,
        sketchwalk_block_size=64,
        sketchwalk_sketch_dim=64,
        sketchwalk_top_k_blocks=16,
        sketchwalk_sparsity_exponent=8,
        sketchwalk_skip_first_n_layers=2,
        sketchwalk_hadamard_seed=42,
        **kwargs,
    ):
        # Standard LLaMA config
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # GQA support
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        # Head dimension
        self.head_dim = hidden_size // num_attention_heads

        # SketchWalk config
        self.sketchwalk_enabled = sketchwalk_enabled
        self.sketchwalk_block_size = sketchwalk_block_size
        self.sketchwalk_sketch_dim = sketchwalk_sketch_dim
        self.sketchwalk_top_k_blocks = sketchwalk_top_k_blocks
        self.sketchwalk_sparsity_exponent = sketchwalk_sparsity_exponent
        self.sketchwalk_skip_first_n_layers = sketchwalk_skip_first_n_layers
        self.sketchwalk_hadamard_seed = sketchwalk_hadamard_seed

        # Validate rope_scaling
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        # Base model for loading weights
        self.base_model = base_model

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Load a SketchWalkLlamaConfig from a pretrained LLaMA model.

        This method loads the base LLaMA config and converts it to a
        SketchWalkLlamaConfig with SketchWalk-specific parameters.

        Args:
            pretrained_model_name_or_path: Path or hub ID of the base LLaMA model
            **kwargs: Additional arguments for SketchWalk configuration

        Returns:
            SketchWalkLlamaConfig with base_model set to the input path
        """
        from transformers import AutoConfig

        # Load base config
        base_config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Create kwargs from base config
        config_kwargs = {
            "vocab_size": base_config.vocab_size,
            "hidden_size": base_config.hidden_size,
            "intermediate_size": base_config.intermediate_size,
            "num_hidden_layers": base_config.num_hidden_layers,
            "num_attention_heads": base_config.num_attention_heads,
            "num_key_value_heads": getattr(base_config, "num_key_value_heads", None),
            "max_position_embeddings": base_config.max_position_embeddings,
            "rope_theta": getattr(base_config, "rope_theta", 10000.0),
            "rope_scaling": getattr(base_config, "rope_scaling", None),
            "hidden_act": getattr(base_config, "hidden_act", "silu"),
            "initializer_range": getattr(base_config, "initializer_range", 0.02),
            "rms_norm_eps": getattr(base_config, "rms_norm_eps", 1e-6),
            "use_cache": getattr(base_config, "use_cache", True),
            "pad_token_id": getattr(base_config, "pad_token_id", None),
            "bos_token_id": getattr(base_config, "bos_token_id", 1),
            "eos_token_id": getattr(base_config, "eos_token_id", 2),
            "tie_word_embeddings": getattr(base_config, "tie_word_embeddings", False),
            "attention_bias": getattr(base_config, "attention_bias", False),
            "attention_dropout": getattr(base_config, "attention_dropout", 0.0),
        }

        # Add any user-provided kwargs (may include SketchWalk-specific params)
        # Extract kwargs that are not in the base config
        sketchwalk_params = {}
        for key in list(kwargs.keys()):
            if key.startswith("sketchwalk_"):
                sketchwalk_params[key] = kwargs.pop(key)

        config_kwargs.update(sketchwalk_params)
        config_kwargs["base_model"] = pretrained_model_name_or_path

        return cls(**config_kwargs)
