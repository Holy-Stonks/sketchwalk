"""
SketchWalk LLaMA Model Implementation

This module implements LLaMA models with SketchWalk sparse attention.
SketchWalk is a training-free sparse attention method that combines:
1. Small-World Sketching (SWS): Efficient block-level attention estimation
2. Sketch-Determined Walk: Cross-layer attention accumulation

Reference: "Scout Before You Attend: Sketch-and-Walk Sparse Attention"
"""

import math
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import logging

from .configuration_llama_sketchwalk import SketchWalkLlamaConfig
from sketch_walk.common.core import (
    DecodeCache,
    SketchWalkConfig,
    SketchWalkAttention,
    create_sketch_walk_config,
)

logger = logging.get_logger(__name__)


# Register SketchWalkLlamaForCausalLM as a causal LM model for .generate() compatibility
# This must be done after the class is defined, but we declare it here
_MODEL_MAPPING = None


# ============================================================================
# RMS Normalization
# ============================================================================

class LlamaRMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


# ============================================================================
# Rotary Position Embedding
# ============================================================================

class LlamaRotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for LLaMA."""

    def __init__(self, config: SketchWalkLlamaConfig, device=None):
        super().__init__()
        self.rope_type = "default"
        if config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", "default")

        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config

        # Get RoPE initialization function
        self.rope_init_fn = ROPE_INIT_FUNCTIONS.get(self.rope_type, ROPE_INIT_FUNCTIONS["default"])

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq.clone()

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute rotary embeddings."""
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dimensions of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key."""
    cos_embed = cos.repeat_interleave(q.shape[-1] // cos.shape[-1], dim=-1)
    sin_embed = sin.repeat_interleave(q.shape[-1] // sin.shape[-1], dim=-1)

    q_embed = (q * cos_embed) + (rotate_half(q) * sin_embed)
    k_embed = (k * cos_embed) + (rotate_half(k) * sin_embed)

    return q_embed, k_embed


# ============================================================================
# Repeat KV for GQA
# ============================================================================

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads for grouped query attention."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# ============================================================================
# SketchWalk Attention
# ============================================================================

class SketchWalkLlamaAttention(nn.Module):
    """
    Multi-head attention with SketchWalk sparse attention.

    This replaces the standard attention mechanism with SketchWalk for
    efficient sparse attention computation during inference.
    """

    def __init__(self, config: SketchWalkLlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size

        # QKV projections
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * config.head_dim, config.hidden_size, bias=config.attention_bias)

        # Rotary embedding
        self.rotary_emb = LlamaRotaryEmbedding(config)

        # Initialize SketchWalk if enabled
        if config.sketchwalk_enabled and layer_idx >= config.sketchwalk_skip_first_n_layers:
            sw_config = create_sketch_walk_config(
                block_size=config.sketchwalk_block_size,
                sketch_dim=config.sketchwalk_sketch_dim,
                top_k_blocks=config.sketchwalk_top_k_blocks,
                sparsity_exponent=config.sketchwalk_sparsity_exponent,
                skip_first_n_layers=0,  # Already handled at layer level
                hadamard_seed=config.sketchwalk_hadamard_seed,
            )
            self.sketch_walk = SketchWalkAttention(sw_config, config.head_dim)
        else:
            self.sketch_walk = None

        # Decode cache for incremental updates
        self.decode_cache: Optional[DecodeCache] = None
        self.last_prefill_A_hat: Optional[torch.Tensor] = None
        self.last_prefill_q_len: int = 0

    def reset_decode_cache(self):
        """Reset decode cache (e.g., when starting a new sequence)."""
        self.decode_cache = None
        self.last_prefill_A_hat = None
        self.last_prefill_q_len = 0
        # Also reset walk state in SketchWalk
        if self.sketch_walk is not None:
            self.sketch_walk.reset_state()

    def _get_layer_cache(self, past_key_values: Optional[Tuple], layer_idx: int) -> Optional[Tuple]:
        """
        Extract the cache for a specific layer from the full past_key_values.

        Args:
            past_key_values: Full past_key_values from model (tuple of (key, value) or list of layer caches)
            layer_idx: Index of the current layer

        Returns:
            Cache for this specific layer as tuple (key, value), or None
        """
        if past_key_values is None:
            return None

        # If it's a tuple (key, value) - single cache for all layers (not typical)
        if isinstance(past_key_values, tuple) and len(past_key_values) == 2:
            return past_key_values

        # If it's a list/tuple of layer-specific caches (transformers format)
        if isinstance(past_key_values, (list, tuple)) and layer_idx < len(past_key_values):
            layer_cache = past_key_values[layer_idx]
            return layer_cache if layer_cache is not None else None

        return None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for SketchWalk attention.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Attention mask
            position_ids: Position IDs for RoPE
            past_key_value: Cached key/values from previous forward passes
            output_attentions: Whether to return attention weights
            use_cache: Whether to use KV cache

        Returns:
            Tuple of (attention_output, attention_weights, past_key_value)
        """
        batch_size, q_len, _ = hidden_states.shape

        # Compute Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to (batch, num_heads, seq_len, head_dim)
        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Detect decode mode
        # Decode mode: single token (q_len == 1) with past_key_value
        # Prefill mode: multiple tokens (q_len > 1) or no past_key_value
        is_decode_mode = (q_len == 1 and past_key_value is not None)

        # Reset decode cache if this is a new prefill (sequence changed)
        if not is_decode_mode and self.last_prefill_q_len > q_len:
            self.reset_decode_cache()

        # Handle KV cache
        # past_key_value is the cache for THIS layer (already extracted by the decoder layer loop)
        # Format: tuple of (key_states, value_states) or None
        if past_key_value is not None:
            # past_key_value is a tuple (key, value)
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        # Create new cache entry for this layer
        past_key_value = (key_states, value_states)

        # Compute attention
        if self.sketch_walk is not None and self.layer_idx >= self.config.sketchwalk_skip_first_n_layers:
            if is_decode_mode and self.decode_cache is not None:
                # Decode mode: use optimized decode path
                # Reset walk state before starting decode (it accumulated during prefill)
                self.sketch_walk.walk.reset_state()

                # K_cache and V_cache have shape (batch, num_kv_heads, cache_len, head_dim)
                K_cache = key_states
                V_cache = value_states

                # Q_t has shape (batch, num_heads, 1, head_dim)
                Q_t = query_states

                attn_output, self.decode_cache = self.sketch_walk.decode(
                    Q_t=Q_t,
                    K_cache=K_cache,
                    V_cache=V_cache,
                    layer_idx=self.layer_idx,
                    cache=self.decode_cache,
                    causal=True,
                )
            else:
                # Prefill mode: use standard SketchWalk forward
                # Repeat K/V heads for GQA (needed for prefill)
                K_prefill = repeat_kv(key_states, self.num_heads // self.num_key_value_heads)
                V_prefill = repeat_kv(value_states, self.num_heads // self.num_key_value_heads)

                attn_output, A_hat = self.sketch_walk(
                    Q=query_states,
                    K=K_prefill,
                    V=V_prefill,
                    layer_idx=self.layer_idx,
                    attention_mask=attention_mask,
                    causal=True,
                )

                # Initialize decode cache for future decode steps
                # Store the prefill outputs for decode cache initialization
                self.last_prefill_A_hat = A_hat
                self.last_prefill_q_len = q_len

                # Initialize decode cache after prefill
                # Use the full Q, K, V (before GQA expansion) for cache init
                Q_prefill = query_states  # (batch, num_heads, q_len, head_dim)
                K_prefill_full = key_states  # (batch, num_kv_heads, q_len, head_dim)
                V_prefill_full = value_states  # (batch, num_kv_heads, q_len, head_dim)

                self.decode_cache = self.sketch_walk.init_decode_cache(
                    Q_prefill=Q_prefill,
                    K_prefill=K_prefill_full,
                    V_prefill=V_prefill_full,
                    A_hat_prefill=A_hat,
                    num_tokens=q_len,
                )
        else:
            # Use standard dense attention
            # Repeat K/V heads for GQA
            key_states = repeat_kv(key_states, self.num_heads // self.num_key_value_heads)
            value_states = repeat_kv(value_states, self.num_heads // self.num_key_value_heads)

            attn_output = self._dense_attention(
                query_states, key_states, value_states, attention_mask
            )

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    def _dense_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute standard dense attention."""
        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            # Ensure attention_mask has the right shape for broadcasting
            # attn_scores shape: (batch, num_heads, q_len, kv_len)
            # attention_mask can be: (batch, kv_len), (batch, 1, kv_len), or (batch, 1, q_len, kv_len)
            if attention_mask.dim() == 2:
                # (batch, kv_len) -> (batch, 1, 1, kv_len)
                attention_mask = attention_mask[:, None, None, :]
            elif attention_mask.dim() == 3:
                # (batch, q_len, kv_len) or (batch, 1, kv_len) -> (batch, 1, q_len, kv_len)
                if attention_mask.size(1) == 1:
                    attention_mask = attention_mask.unsqueeze(1)  # (batch, 1, 1, kv_len)
                else:
                    attention_mask = attention_mask.unsqueeze(1)  # (batch, 1, q_len, kv_len)

            # Only apply if shapes match or attention_mask can be broadcast
            if attention_mask.size(-1) == attn_scores.size(-1):
                attn_scores = attn_scores + attention_mask

        attn_weights = F.softmax(attn_scores, dim=-1)
        # Cast attn_weights to the same dtype as V to avoid dtype mismatch
        attn_output = torch.matmul(attn_weights.to(V.dtype), V)

        return attn_output


# ============================================================================
# MLP Block
# ============================================================================

class LlamaMLP(nn.Module):
    """MLP block for LLaMA."""

    def __init__(self, config: SketchWalkLlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ============================================================================
# Transformer Layer
# ============================================================================

class SketchWalkLlamaDecoderLayer(nn.Module):
    """Transformer decoder layer with SketchWalk attention."""

    def __init__(self, config: SketchWalkLlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = SketchWalkLlamaAttention(config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[Cache]]:
        """
        Forward pass for decoder layer.

        Args:
            hidden_states: Input tensor
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_value: Cached KV from previous layers (can be list or Cache)
            output_attentions: Whether to output attention weights
            use_cache: Whether to use KV cache

        Returns:
            Tuple of (hidden_states, self_attn_weights, present_key_value)
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Extract layer-specific cache if past_key_value is a list/tuple of all layers
        layer_cache = self.self_attn._get_layer_cache(past_key_value, self.layer_idx)

        # Self-attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=layer_cache,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


# ============================================================================
# Full Model
# ============================================================================

class SketchWalkLlamaModel(PreTrainedModel):
    """
    Base LLaMA model with SketchWalk attention.

    This is a drop-in replacement for standard LLaMA models that uses
    SketchWalk sparse attention for efficient inference.
    """

    config_class = SketchWalkLlamaConfig
    base_model_prefix = "model"
    _no_split_modules = ["embed_tokens"]

    def __init__(self, config: SketchWalkLlamaConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # Layers
        self.layers = nn.ModuleList([
            SketchWalkLlamaDecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])

        # Final norm
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Initialize weights
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        self.embed_tokens = value

    def reset_decode_cache(self):
        """Reset decode cache for all layers (call when starting a new sequence)."""
        for layer in self.layers:
            if hasattr(layer.self_attn, 'reset_decode_cache'):
                layer.self_attn.reset_decode_cache()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, dict]:
        """
        Forward pass for the model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_values: Past key/value states for generation
            inputs_embeds: Input embeddings (alternative to input_ids)
            use_cache: Whether to return KV cache
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return a dictionary

        Returns:
            Model outputs (last_hidden_state, hidden_states, attentions, past_key_values)
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        # Handle embedded inputs
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Generate position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)

        # Handle attention mask
        if attention_mask is not None:
            # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=inputs_embeds.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(inputs_embeds.dtype).min

        # Reset hidden states
        hidden_states = inputs_embeds

        # Decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                past_key_values = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # Final normalization
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Return outputs
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attns, past_key_values] if v is not None)

        return {
            "last_hidden_state": hidden_states,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attns,
            "past_key_values": past_key_values,
        }


# ============================================================================
# Causal LM Model
# ============================================================================

class SketchWalkLlamaForCausalLM(PreTrainedModel, GenerationMixin):
    """
    LLaMA model for causal language modeling with SketchWalk attention.

    This model can be used for text generation with efficient sparse attention.
    """

    config_class = SketchWalkLlamaConfig
    base_model_prefix = "model"
    _tied_weights_keys = ["lm_head.weight"]
    _no_split_modules = ["model.embed_tokens", "lm_head"]

    def __init__(self, config: SketchWalkLlamaConfig):
        super().__init__(config)
        self.model = SketchWalkLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def _rehash_cache(self, key, value, **kwargs):
        """
        Rehash cache for SketchWalk attention.

        This is called during generation when the cache grows beyond the initial size.
        For SketchWalk, we need to reset the decode cache when this happens.
        """
        # Reset decode cache for all layers when cache is rehashed
        if hasattr(self.model, 'layers'):
            for layer in self.model.layers:
                if hasattr(layer.self_attn, 'reset_decode_cache'):
                    layer.self_attn.reset_decode_cache()
        return key, value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, dict]:
        """
        Forward pass for causal language modeling.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_values: Past key/value states for generation
            inputs_embeds: Input embeddings (alternative to input_ids)
            labels: Labels for language modeling loss
            use_cache: Whether to return KV cache
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return a dictionary

        Returns:
            Model outputs (loss, logits, hidden_states, attentions, past_key_values)
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Decoder outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Handle both tuple and dict return formats
        if return_dict:
            hidden_states = outputs["last_hidden_state"]
        else:
            hidden_states = outputs[0]

        # Compute logits
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits_view = shift_logits.view(-1, self.vocab_size)
            shift_labels_view = shift_labels.view(-1)
            loss = loss_fct(shift_logits_view, shift_labels_view)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": outputs.get("hidden_states"),
            "attentions": outputs.get("attentions"),
            "past_key_values": outputs.get("past_key_values"),
        }

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        **kwargs
    ):
        """
        Load a SketchWalk-enabled LLaMA model from a pretrained LLaMA checkpoint.

        This method loads the base LLaMA model weights and applies SketchWalk
        sparse attention mechanism to them.

        Args:
            pretrained_model_name_or_path: Path or hub ID of the base LLaMA model
            *model_args: Additional positional arguments
            **kwargs: Additional keyword arguments, including:
                - sketchwalk_block_size: Block size for SketchWalk
                - sketchwalk_sketch_dim: Sketch dimension
                - sketchwalk_top_k_blocks: Number of top-k blocks
                - sketchwalk_sparsity_exponent: Sparsity exponent

        Returns:
            SketchWalkLlamaForCausalLM with loaded weights
        """
        # Load or create SketchWalk config
        config = kwargs.pop("config", None)
        if config is None:
            config = SketchWalkLlamaConfig.from_pretrained(
                pretrained_model_name_or_path, **kwargs
            )
        else:
            # Ensure config has base_model set
            if not hasattr(config, "base_model") or config.base_model is None:
                config.base_model = pretrained_model_name_or_path

        # Update kwargs with any remaining sketchwalk parameters
        for key in list(kwargs.keys()):
            if hasattr(config, key) and key.startswith("sketchwalk_"):
                setattr(config, key, kwargs.pop(key))

        # Load model from the base model path with our custom config
        model = super().from_pretrained(
            config.base_model,
            config=config,
            *model_args,
            **kwargs
        )
        return model

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs
    ):
        """
        Prepare inputs for generation.

        This method is required by transformers' `generate()` API.
        """
        # If past_key_values is provided, we only need the last input_ids
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # Build model inputs
        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
        }

        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask

        # Position ids are needed for RoPE
        if hasattr(self, "_prepare_position_ids"):
            position_ids = self._prepare_position_ids(input_ids, past_key_values)
            model_inputs["position_ids"] = position_ids

        return model_inputs


# Register SketchWalkLlamaForCausalLM in the MODEL_FOR_CAUSAL_LM_MAPPING
# This allows .generate() to work with our model
MODEL_FOR_CAUSAL_LM_MAPPING.register(SketchWalkLlamaConfig, SketchWalkLlamaForCausalLM)


# ============================================================================
# Factory function for easy loading
# ============================================================================

def create_sketch_walk_llama(
    base_model_name: str = "meta-llama/Llama-3.1-8B",
    block_size: int = 64,
    sketch_dim: int = 64,
    top_k_blocks: int = 16,
    sparsity_exponent: int = 8,
    **kwargs
) -> SketchWalkLlamaForCausalLM:
    """
    Factory function to create a SketchWalk-enabled LLaMA model.

    Args:
        base_model_name: HuggingFace model name to base configuration on
        block_size: Block size for SketchWalk
        sketch_dim: Sketch dimension
        top_k_blocks: Number of blocks to select
        sparsity_exponent: Sparsity exponent
        **kwargs: Additional configuration parameters

    Returns:
        SketchWalkLlamaForCausalLM model
    """
    from transformers import AutoConfig

    # Load base config
    base_config = AutoConfig.from_pretrained(base_model_name)

    # Create SketchWalk config
    config = SketchWalkLlamaConfig(
        # Copy from base config
        vocab_size=base_config.vocab_size,
        hidden_size=base_config.hidden_size,
        intermediate_size=base_config.intermediate_size,
        num_hidden_layers=base_config.num_hidden_layers,
        num_attention_heads=base_config.num_attention_heads,
        num_key_value_heads=getattr(base_config, 'num_key_value_heads', base_config.num_attention_heads),
        max_position_embeddings=base_config.max_position_embeddings,
        rope_theta=getattr(base_config, 'rope_theta', 10000.0),
        # SketchWalk settings
        sketchwalk_enabled=True,
        sketchwalk_block_size=block_size,
        sketchwalk_sketch_dim=sketch_dim,
        sketchwalk_top_k_blocks=top_k_blocks,
        sketchwalk_sparsity_exponent=sparsity_exponent,
        **kwargs,
    )

    # Create model
    model = SketchWalkLlamaForCausalLM(config)

    return model
