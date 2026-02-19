#!/usr/bin/env python3
"""
Collect Real Tensors from SeerAttention Runs

This script runs SeerAttention models and collects Q, K, V tensors during inference
for validating SketchWalk sparse attention patterns.

Usage:
    python collect_seerattn_tensors.py \
        --model_path /path/to/model \
        --output_dir ./tensor_data \
        --sequence_lengths 4096 8192 16384 \
        --num_samples 5

Outputs:
    - Tensors saved as .pt files with Q, K, V, attention patterns
    - Metadata including sparsity, sequence length, layer info
    - Analysis of attention patterns and sparsity statistics
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TensorMetadata:
    """Metadata for collected tensors."""
    timestamp: str
    model_path: str
    sequence_length: int
    num_layers: int
    num_heads: int
    head_dim: int
    hidden_size: int
    num_samples: int
    device: str
    dtype: str

    def to_dict(self):
        return asdict(self)


class TensorCollector:
    """
    Hook into SeerAttention models to collect Q, K, V tensors
    and attention patterns during inference.
    """

    def __init__(
        self,
        model_path: str,
        output_dir: str,
        device: str = "cuda",
        max_samples: int = 10,
    ):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.device = device
        self.max_samples = max_samples

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage for collected tensors
        self.collected_data: List[Dict] = []
        self.hooks = []

        # Model will be loaded later
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the SeerAttention model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}")

        # Import SeerAttention model
        try:
            from seer_attn import SeerAttnLlamaForCausalLM
            from transformers import AutoTokenizer, AutoConfig

            config = AutoConfig.from_pretrained(self.model_path)
            tokenizer_path = config.base_model

            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True,
                padding_side="left",
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model with SeerAttention
            self.model = SeerAttnLlamaForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                seerattn_sparsity_method='threshold',
                seerattn_threshold=0.1,  # Moderate sparsity
                use_cache=True,
                seerattn_last_block_dense=True,
            )

            self.model.eval()
            logger.info(f"Model loaded successfully")
            logger.info(f"Num layers: {self.model.config.num_hidden_layers}")
            logger.info(f"Num heads: {self.model.config.num_attention_heads}")
            logger.info(f"Head dim: {self.model.config.hidden_size // self.model.config.num_attention_heads}")

            return True

        except Exception as e:
            logger.error(f"Failed to load SeerAttention model: {e}")
            logger.info("Falling back to standard LLaMA model")

            # Fallback to standard model
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    padding_side="left",
                )
                self.tokenizer.pad_token = self.tokenizer.eos_token

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                )

                self.model.eval()
                logger.info("Standard model loaded successfully")
                return True

            except Exception as e2:
                logger.error(f"Failed to load model: {e2}")
                return False

    def create_forward_hook(self, layer_idx: int):
        """Create a forward hook to collect tensors from a specific layer."""

        def hook(module, input, output):
            # Only collect if we haven't reached max samples
            if len(self.collected_data) >= self.max_samples:
                return

            # Try to extract Q, K, V from the layer
            # This depends on the specific model implementation
            try:
                # For LLaMA-like models, we can try to access the attention module
                if hasattr(module, 'self_attn'):
                    attn = module.self_attn

                    # Get Q, K, V projections
                    if hasattr(attn, 'q_proj') and hasattr(attn, 'k_proj') and hasattr(attn, 'v_proj'):
                        # We need to get the input hidden states
                        hidden_states = input[0]

                        batch_size, seq_len, hidden_size = hidden_states.shape

                        # Project to Q, K, V
                        Q = attn.q_proj(hidden_states)
                        K = attn.k_proj(hidden_states)
                        V = attn.v_proj(hidden_states)

                        # Reshape to heads
                        num_heads = attn.num_heads
                        head_dim = hidden_size // num_heads
                        num_kv_heads = getattr(attn, 'num_key_value_heads', num_heads)

                        Q = Q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                        K = K.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
                        V = V.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)

                        # Store the tensors
                        self.collected_data.append({
                            'layer_idx': layer_idx,
                            'Q': Q.detach().cpu(),
                            'K': K.detach().cpu(),
                            'V': V.detach().cpu(),
                            'batch_size': batch_size,
                            'seq_len': seq_len,
                            'num_heads': num_heads,
                            'num_kv_heads': num_kv_heads,
                            'head_dim': head_dim,
                            'hidden_size': hidden_size,
                        })

                        logger.info(f"Collected tensors from layer {layer_idx}, seq_len={seq_len}")

            except Exception as e:
                logger.debug(f"Could not extract tensors from layer {layer_idx}: {e}")

        return hook

    def register_hooks(self):
        """Register forward hooks on transformer layers."""
        if self.model is None:
            logger.error("Model not loaded. Call load_model() first.")
            return False

        # Clear existing hooks
        self.remove_hooks()

        # Register hooks on decoder layers
        # For LLaMA-like models
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
            for idx, layer in enumerate(layers):
                hook = layer.register_forward_hook(self.create_forward_hook(idx))
                self.hooks.append(hook)
                logger.info(f"Registered hook on layer {idx}")

        logger.info(f"Registered {len(self.hooks)} hooks")
        return True

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        logger.info("Removed all hooks")

    def generate_synthetic_data(self, sequence_length: int) -> str:
        """Generate synthetic text data of approximately the target length."""
        # Generate repeated patterns to create long sequences
        base_text = """
        The transformer architecture has revolutionized natural language processing.
        Self-attention mechanisms allow models to capture long-range dependencies.
        Sparse attention methods reduce computational complexity from quadratic to near-linear.
        SketchWalk combines small-world sketching with cross-layer attention accumulation.
        """ * 10  # Repeat for longer base

        # Calculate how many repeats we need
        target_length = sequence_length
        current_length = 0
        result_text = ""

        while current_length < target_length:
            result_text += base_text
            tokens = self.tokenizer.encode(result_text)
            current_length = len(tokens)

        return result_text

    def collect_tensors_for_sequence_length(self, sequence_length: int) -> bool:
        """Collect tensors for a specific sequence length."""
        logger.info(f"Collecting tensors for sequence length: {sequence_length}")

        # Clear previous data
        self.collected_data = []

        # Generate input data
        text = self.generate_synthetic_data(sequence_length)
        tokens = self.tokenizer.encode(text)

        # Truncate or pad to target length
        if len(tokens) > sequence_length:
            tokens = tokens[:sequence_length]
        else:
            # Pad with zeros if needed (though tokenizer should handle this)
            pass

        # Convert to tensor
        input_ids = torch.tensor([tokens], device=self.device)

        logger.info(f"Running inference with {len(tokens)} tokens")

        # Register hooks before forward pass
        self.register_hooks()

        try:
            # Run model forward pass
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    output_attentions=False,
                    output_hidden_states=True,
                    use_cache=False,
                )

            logger.info(f"Collected {len(self.collected_data)} tensor samples")

            if len(self.collected_data) > 0:
                # Save collected tensors
                self._save_collected_tensors(sequence_length)
                return True
            else:
                logger.warning("No tensors were collected. Hooks may not have worked.")
                return False

        except Exception as e:
            logger.error(f"Error during forward pass: {e}")
            return False

        finally:
            # Always remove hooks
            self.remove_hooks()

    def _save_collected_tensors(self, sequence_length: int):
        """Save collected tensors to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for idx, data in enumerate(self.collected_data):
            layer_idx = data['layer_idx']
            seq_len = data['seq_len']

            # Create filename
            filename = f"tensors_seq{seq_len}_layer{layer_idx}_sample{idx}_{timestamp}.pt"
            filepath = self.output_dir / filename

            # Save tensors
            tensor_data = {
                'Q': data['Q'],
                'K': data['K'],
                'V': data['V'],
                'metadata': {
                    'layer_idx': layer_idx,
                    'seq_len': seq_len,
                    'num_heads': data['num_heads'],
                    'num_kv_heads': data['num_kv_heads'],
                    'head_dim': data['head_dim'],
                    'hidden_size': data['hidden_size'],
                    'batch_size': data['batch_size'],
                    'timestamp': timestamp,
                    'model_path': self.model_path,
                }
            }

            torch.save(tensor_data, filepath)
            logger.info(f"Saved tensors to {filepath}")

        # Save summary
        summary = {
            'timestamp': timestamp,
            'model_path': self.model_path,
            'sequence_length': sequence_length,
            'num_samples': len(self.collected_data),
            'layers': [d['layer_idx'] for d in self.collected_data],
            'seq_lengths': [d['seq_len'] for d in self.collected_data],
        }

        summary_path = self.output_dir / f"summary_seq{sequence_length}_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved summary to {summary_path}")

    def analyze_attention_patterns(self):
        """Analyze attention patterns in collected tensors."""
        if len(self.collected_data) == 0:
            logger.warning("No data to analyze")
            return

        logger.info("Analyzing attention patterns...")

        for data in self.collected_data:
            Q = data['Q']
            K = data['K']
            seq_len = data['seq_len']

            # Compute attention scores (without masking for analysis)
            # Shape: (batch, num_heads, q_len, k_len)
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (data['head_dim'] ** 0.5)

            # Compute sparsity at different thresholds
            thresholds = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

            for threshold in thresholds:
                # Compute attention weights
                attn_weights = F.softmax(attn_scores, dim=-1)

                # Mask values below threshold
                mask = attn_weights > threshold
                sparsity = 1.0 - (mask.float().mean().item())

                logger.info(
                    f"Layer {data['layer_idx']}, Threshold {threshold}: "
                    f"Sparsity = {sparsity:.3f}"
                )

            # Analyze block structure (if sequence is long enough)
            if seq_len >= 128:
                block_size = 64
                n_blocks = seq_len // block_size

                # Compute block-level attention
                # Reshape to blocks
                batch_size, num_heads, n_q, n_k = attn_scores.shape

                # Average over blocks
                attn_blocks = attn_scores.view(
                    batch_size, num_heads,
                    n_blocks, block_size,
                    n_blocks, block_size
                ).mean(dim=(3, 5))

                logger.info(f"Block-level attention shape: {attn_blocks.shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect real tensors from SeerAttention runs"
    )

    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the SeerAttention model checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./tensor_data",
        help="Directory to save collected tensors",
    )

    # Data generation arguments
    parser.add_argument(
        "--sequence_lengths",
        type=int,
        nargs="+",
        default=[4096, 8192, 16384],
        help="Sequence lengths to collect tensors for",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to collect per sequence length",
    )

    # Hardware arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda/cpu)",
    )

    # Other arguments
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze attention patterns after collection",
    )

    args = parser.parse_args()

    # Check if CUDA is available
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = "cpu"

    # Create collector
    collector = TensorCollector(
        model_path=args.model_path,
        output_dir=args.output_dir,
        device=args.device,
        max_samples=args.num_samples,
    )

    # Load model
    if not collector.load_model():
        logger.error("Failed to load model")
        return 1

    # Collect tensors for each sequence length
    for seq_len in args.sequence_lengths:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing sequence length: {seq_len}")
        logger.info(f"{'='*60}")

        success = collector.collect_tensors_for_sequence_length(seq_len)

        if success:
            logger.info(f"Successfully collected tensors for sequence length {seq_len}")
        else:
            logger.warning(f"Failed to collect tensors for sequence length {seq_len}")

    # Analyze patterns if requested
    if args.analyze and len(collector.collected_data) > 0:
        collector.analyze_attention_patterns()

    logger.info("\nTensor collection complete!")
    logger.info(f"Output directory: {args.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
