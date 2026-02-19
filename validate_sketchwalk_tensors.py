#!/usr/bin/env python3
"""
Validate SketchWalk on Collected Tensors

This script loads Q, K, V tensors collected from real SeerAttention runs
and validates that SketchWalk produces correct sparse attention patterns.

Usage:
    python validate_sketchwalk_tensors.py \
        --tensor_dir ./tensor_data \
        --output_dir ./validation_results \
        --block_size 64 \
        --sketch_dim 64 \
        --top_k_blocks 16

Validation:
    - Compares dense vs sparse attention outputs
    - Measures sparsity levels and block selection
    - Validates numerical accuracy of SketchWalk
    - Generates performance metrics
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Add parent directory to path to import sketch_walk
sys.path.insert(0, str(Path(__file__).parent))

try:
    from sketch_walk.common.core import (
        SketchWalkConfig,
        SketchWalkAttention,
        create_sketch_walk_config,
    )
except ImportError:
    logger.error("Could not import SketchWalk. Make sure sketch_walk module is available.")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Results from validating SketchWalk on a single tensor sample."""
    sample_id: str
    layer_idx: int
    seq_len: int
    num_heads: int

    # Sparsity metrics
    actual_sparsity: float
    target_sparsity: float
    block_sparsity: float

    # Accuracy metrics
    cosine_similarity: float
    mse: float
    max_error: float

    # Block selection metrics
    num_blocks_selected: int
    total_blocks: int
    block_selection_ratio: float

    # Timing metrics
    dense_time_ms: float
    sparse_time_ms: float
    speedup: float

    def to_dict(self):
        return asdict(self)


class SketchWalkValidator:
    """
    Validate SketchWalk sparse attention on collected real tensors.
    """

    def __init__(
        self,
        tensor_dir: str,
        output_dir: str,
        sketchwalk_config: SketchWalkConfig,
        device: str = "cuda",
    ):
        self.tensor_dir = Path(tensor_dir)
        self.output_dir = Path(output_dir)
        self.config = sketchwalk_config
        self.device = device

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize SketchWalk attention
        self.sketch_walk = SketchWalkAttention(
            config=self.config,
            num_heads=1,  # Will be set per sample
        )

        # Load all tensor files
        self.tensor_files = self._find_tensor_files()

        logger.info(f"Found {len(self.tensor_files)} tensor files")
        logger.info(f"SketchWalk config: {self.config}")

    def _find_tensor_files(self) -> List[Path]:
        """Find all tensor files in the tensor directory."""
        return list(self.tensor_dir.glob("tensors_*.pt"))

    def load_tensor_file(self, filepath: Path) -> Optional[Dict]:
        """Load a tensor file and extract Q, K, V tensors."""
        try:
            data = torch.load(filepath, map_location=self.device)
            return data
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            return None

    def compute_dense_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        """Compute dense attention output."""

        batch_size, num_heads, n_q, head_dim = Q.shape
        n_k = K.shape[2]

        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)

        # Apply causal mask if needed
        if causal and n_q == n_k:
            # Create causal mask
            mask = torch.triu(torch.ones(n_q, n_k, device=Q.device), diagonal=1).bool()
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Compute output
        output = torch.matmul(attn_weights, V)

        return output

    def compute_sketchwalk_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        layer_idx: int,
        causal: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute SketchWalk sparse attention output."""

        batch_size, num_heads, n_q, head_dim = Q.shape
        n_k = K.shape[2]

        # Run SketchWalk
        output, selected_blocks = self.sketch_walk(
            Q=Q,
            K=K,
            V=V,
            layer_idx=layer_idx,
            attention_mask=None,
            causal=causal,
        )

        return output, selected_blocks

    def compute_sparsity(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        selected_blocks: Optional[torch.Tensor],
    ) -> Tuple[float, float]:
        """Compute actual and block-level sparsity."""

        seq_len = Q.shape[2]
        block_size = self.config.block_size
        total_blocks = (seq_len + block_size - 1) // block_size

        if selected_blocks is not None:
            # Block-level sparsity
            num_blocks_selected = selected_blocks.shape[-1]
            block_sparsity = 1.0 - (num_blocks_selected / total_blocks)

            # Estimate token-level sparsity
            attended_tokens = num_blocks_selected * block_size
            token_sparsity = 1.0 - min(attended_tokens / seq_len, 1.0)
        else:
            # Fallback to target sparsity
            block_sparsity = self.config.sparsity_level(seq_len)
            token_sparsity = block_sparsity

        return token_sparsity, block_sparsity

    def compare_outputs(
        self,
        dense_output: torch.Tensor,
        sparse_output: torch.Tensor,
    ) -> Tuple[float, float, float]:
        """Compare dense and sparse outputs."""

        # Flatten for comparison
        dense_flat = dense_output.flatten()
        sparse_flat = sparse_output.flatten()

        # Cosine similarity
        cosine_sim = F.cosine_similarity(
            dense_flat.unsqueeze(0),
            sparse_flat.unsqueeze(0),
        ).item()

        # Mean squared error
        mse = F.mse_loss(dense_output, sparse_output).item()

        # Max absolute error
        max_error = torch.max(torch.abs(dense_output - sparse_output)).item()

        return cosine_sim, mse, max_error

    def validate_single_sample(
        self,
        data: Dict,
        sample_id: str,
    ) -> Optional[ValidationResult]:
        """Validate SketchWalk on a single tensor sample."""

        try:
            # Extract tensors and metadata
            Q = data['Q'].to(self.device)
            K = data['K'].to(self.device)
            V = data['V'].to(self.device)
            metadata = data['metadata']

            layer_idx = metadata['layer_idx']
            seq_len = metadata['seq_len']
            num_heads = metadata['num_heads']

            logger.info(f"Validating sample {sample_id}: layer={layer_idx}, seq_len={seq_len}")

            # Compute dense attention
            import time
            start_time = time.time()

            dense_output = self.compute_dense_attention(Q, K, V, causal=True)

            dense_time = (time.time() - start_time) * 1000  # Convert to ms

            # Compute SketchWalk attention
            start_time = time.time()

            sparse_output, selected_blocks = self.compute_sketchwalk_attention(
                Q, K, V, layer_idx, causal=True
            )

            sparse_time = (time.time() - start_time) * 1000  # Convert to ms

            # Compute sparsity
            token_sparsity, block_sparsity = self.compute_sparsity(
                Q, K, selected_blocks
            )

            # Compare outputs
            cosine_sim, mse, max_error = self.compare_outputs(
                dense_output, sparse_output
            )

            # Compute block selection metrics
            block_size = self.config.block_size
            total_blocks = (seq_len + block_size - 1) // block_size

            if selected_blocks is not None:
                num_blocks_selected = selected_blocks.shape[-1]
            else:
                num_blocks_selected = int((1 - block_sparsity) * total_blocks)

            block_selection_ratio = num_blocks_selected / total_blocks if total_blocks > 0 else 0

            # Target sparsity
            target_sparsity = self.config.sparsity_level(seq_len)

            # Compute speedup
            speedup = dense_time / sparse_time if sparse_time > 0 else 1.0

            # Create result
            result = ValidationResult(
                sample_id=sample_id,
                layer_idx=layer_idx,
                seq_len=seq_len,
                num_heads=num_heads,
                actual_sparsity=token_sparsity,
                target_sparsity=target_sparsity,
                block_sparsity=block_sparsity,
                cosine_similarity=cosine_sim,
                mse=mse,
                max_error=max_error,
                num_blocks_selected=num_blocks_selected,
                total_blocks=total_blocks,
                block_selection_ratio=block_selection_ratio,
                dense_time_ms=dense_time,
                sparse_time_ms=sparse_time,
                speedup=speedup,
            )

            logger.info(
                f"  Sparsity: {token_sparsity:.3f} (target: {target_sparsity:.3f}), "
                f"Cosine Sim: {cosine_sim:.4f}, "
                f"Speedup: {speedup:.2f}x"
            )

            return result

        except Exception as e:
            logger.error(f"Error validating sample {sample_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_validation(self) -> List[ValidationResult]:
        """Run validation on all collected tensor samples."""

        results = []

        for filepath in tqdm(self.tensor_files, desc="Validating samples"):
            data = self.load_tensor_file(filepath)

            if data is None:
                continue

            sample_id = filepath.stem

            result = self.validate_single_sample(data, sample_id)

            if result is not None:
                results.append(result)

        return results

    def save_results(self, results: List[ValidationResult]):
        """Save validation results to disk."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save individual results
        results_data = [r.to_dict() for r in results]

        results_file = self.output_dir / f"validation_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"Saved results to {results_file}")

        # Save summary statistics
        if len(results) > 0:
            summary = self._compute_summary(results)

            summary_file = self.output_dir / f"validation_summary_{timestamp}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

            logger.info(f"Saved summary to {summary_file}")

            # Print summary
            self._print_summary(summary)

    def _compute_summary(self, results: List[ValidationResult]) -> Dict:
        """Compute summary statistics from validation results."""

        metrics = {
            'num_samples': len(results),
            'sequence_lengths': sorted(set(r.seq_len for r in results)),
            'layers': sorted(set(r.layer_idx for r in results)),
        }

        # Compute statistics for each metric
        for field in [
            'actual_sparsity', 'target_sparsity', 'block_sparsity',
            'cosine_similarity', 'mse', 'max_error',
            'num_blocks_selected', 'total_blocks', 'block_selection_ratio',
            'dense_time_ms', 'sparse_time_ms', 'speedup',
        ]:
            values = [getattr(r, field) for r in results]

            metrics[f'{field}_mean'] = float(np.mean(values))
            metrics[f'{field}_std'] = float(np.std(values))
            metrics[f'{field}_min'] = float(np.min(values))
            metrics[f'{field}_max'] = float(np.max(values))

        # Group by sequence length
        seq_lens = sorted(set(r.seq_len for r in results))
        metrics['by_sequence_length'] = {}

        for seq_len in seq_lens:
            seq_results = [r for r in results if r.seq_len == seq_len]

            seq_summary = {
                'num_samples': len(seq_results),
                'mean_cosine_similarity': float(np.mean([r.cosine_similarity for r in seq_results])),
                'mean_sparsity': float(np.mean([r.actual_sparsity for r in seq_results])),
                'mean_speedup': float(np.mean([r.speedup for r in seq_results])),
            }

            metrics['by_sequence_length'][str(seq_len)] = seq_summary

        return metrics

    def _print_summary(self, summary: Dict):
        """Print summary statistics."""

        logger.info("\n" + "="*60)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*60)

        logger.info(f"Total samples: {summary['num_samples']}")

        if 'cosine_similarity_mean' in summary:
            logger.info(f"\nAccuracy Metrics:")
            logger.info(f"  Cosine Similarity: {summary['cosine_similarity_mean']:.4f} ± {summary['cosine_similarity_std']:.4f}")
            logger.info(f"  MSE: {summary['mse_mean']:.6f} ± {summary['mse_std']:.6f}")

        if 'actual_sparsity_mean' in summary:
            logger.info(f"\nSparsity Metrics:")
            logger.info(f"  Actual Sparsity: {summary['actual_sparsity_mean']:.3f} ± {summary['actual_sparsity_std']:.3f}")
            logger.info(f"  Target Sparsity: {summary['target_sparsity_mean']:.3f} ± {summary['target_sparsity_std']:.3f}")

        if 'speedup_mean' in summary:
            logger.info(f"\nPerformance Metrics:")
            logger.info(f"  Dense Time: {summary['dense_time_ms_mean']:.2f} ± {summary['dense_time_ms_std']:.2f} ms")
            logger.info(f"  Sparse Time: {summary['sparse_time_ms_mean']:.2f} ± {summary['sparse_time_ms_std']:.2f} ms")
            logger.info(f"  Speedup: {summary['speedup_mean']:.2f}x ± {summary['speedup_std']:.2f}x")

        if 'by_sequence_length' in summary:
            logger.info(f"\nBy Sequence Length:")
            for seq_len, seq_summary in sorted(summary['by_sequence_length'].items(), key=lambda x: int(x[0])):
                logger.info(f"  {seq_len}: CosSim={seq_summary['mean_cosine_similarity']:.4f}, "
                           f"Sparsity={seq_summary['mean_sparsity']:.3f}, "
                           f"Speedup={seq_summary['mean_speedup']:.2f}x")

        logger.info("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Validate SketchWalk on collected tensors"
    )

    # Data arguments
    parser.add_argument(
        "--tensor_dir",
        type=str,
        required=True,
        help="Directory containing collected tensor files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./validation_results",
        help="Directory to save validation results",
    )

    # SketchWalk configuration
    parser.add_argument(
        "--block_size",
        type=int,
        default=64,
        help="Block size for SketchWalk",
    )
    parser.add_argument(
        "--sketch_dim",
        type=int,
        default=64,
        help="Sketch dimension for SketchWalk",
    )
    parser.add_argument(
        "--top_k_blocks",
        type=int,
        default=16,
        help="Number of top blocks to select",
    )
    parser.add_argument(
        "--sparsity_exponent",
        type=int,
        default=8,
        help="Sparsity exponent for attention sharpening",
    )

    # Hardware arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda/cpu)",
    )

    args = parser.parse_args()

    # Check if CUDA is available
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = "cpu"

    # Create SketchWalk config
    config = SketchWalkConfig(
        block_size=args.block_size,
        sketch_dim=args.sketch_dim,
        top_k_blocks=args.top_k_blocks,
        sparsity_exponent=args.sparsity_exponent,
        device=torch.device(args.device),
    )

    # Create validator
    validator = SketchWalkValidator(
        tensor_dir=args.tensor_dir,
        output_dir=args.output_dir,
        sketchwalk_config=config,
        device=args.device,
    )

    # Run validation
    logger.info("Starting validation...")
    results = validator.run_validation()

    # Save results
    if len(results) > 0:
        logger.info(f"\nValidation complete! Processed {len(results)} samples")
        validator.save_results(results)
    else:
        logger.warning("No validation results to save")

    return 0 if len(results) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
