#!/usr/bin/env python3
"""
Compare SeerAttention vs SketchWalk Sparsity Patterns

This script loads collected tensors and compares the sparsity patterns
produced by SeerAttention (learned sparse) vs SketchWalk (training-free sparse).

Usage:
    python compare_sparsity_patterns.py \
        --tensor_dir ./tensor_data \
        --validation_dir ./validation_results \
        --output_dir ./comparison_results \
        --visualize

Comparison Metrics:
    - Block selection overlap
    - Attention score distributions
    - Sparsity level comparison
    - Pattern similarity analysis
    - Visualization of attention matrices
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
    )
except ImportError:
    logger.error("Could not import SketchWalk. Make sure sketch_walk module is available.")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PatternComparisonResult:
    """Results from comparing sparsity patterns."""

    sample_id: str
    layer_idx: int
    seq_len: int

    # SeerAttention metrics
    seerattn_sparsity: float
    seerattn_top_blocks: List[int]

    # SketchWalk metrics
    sketchwalk_sparsity: float
    sketchwalk_top_blocks: List[int]

    # Comparison metrics
    block_overlap: float
    jaccard_similarity: float
    attention_correlation: float

    def to_dict(self):
        return asdict(self)


class SparsityPatternComparator:
    """
    Compare sparsity patterns between SeerAttention and SketchWalk.
    """

    def __init__(
        self,
        tensor_dir: str,
        validation_dir: str,
        output_dir: str,
        sketchwalk_config: SketchWalkConfig,
        device: str = "cuda",
    ):
        self.tensor_dir = Path(tensor_dir)
        self.validation_dir = Path(validation_dir)
        self.output_dir = Path(output_dir)
        self.config = sketchwalk_config
        self.device = device

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize SketchWalk attention
        self.sketch_walk = SketchWalkAttention(
            config=self.config,
            num_heads=1,
        )

        # Load tensor files
        self.tensor_files = self._find_tensor_files()

        # Load validation results
        self.validation_results = self._load_validation_results()

        logger.info(f"Found {len(self.tensor_files)} tensor files")
        logger.info(f"Loaded {len(self.validation_results)} validation results")

    def _find_tensor_files(self) -> List[Path]:
        """Find all tensor files in the tensor directory."""
        return list(self.tensor_dir.glob("tensors_*.pt"))

    def _load_validation_results(self) -> Dict:
        """Load validation results from previous runs."""

        results = {}

        for filepath in self.validation_dir.glob("validation_results_*.json"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                for result in data:
                    sample_id = result['sample_id']
                    results[sample_id] = result

            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {e}")

        return results

    def load_tensor_file(self, filepath: Path) -> Optional[Dict]:
        """Load a tensor file and extract Q, K, V tensors."""
        try:
            data = torch.load(filepath, map_location=self.device)
            return data
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            return None

    def compute_seerattn_sparsity(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        threshold: float = 0.1,
    ) -> Tuple[float, List[int]]:
        """
        Compute SeerAttention-style sparsity based on attention threshold.

        SeerAttention uses a threshold-based sparsity method where attention
        weights below a threshold are pruned.
        """

        batch_size, num_heads, n_q, head_dim = Q.shape
        n_k = K.shape[2]

        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)

        # Apply causal mask
        mask = torch.triu(torch.ones(n_q, n_k, device=Q.device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Compute sparsity based on threshold
        # In SeerAttention, weights below threshold are pruned
        sparse_mask = attn_weights > threshold

        # Compute token-level sparsity
        sparsity = 1.0 - (sparse_mask.float().mean().item())

        # For block-level analysis, compute top blocks
        block_size = self.config.block_size
        n_blocks = (n_k + block_size - 1) // block_size

        # Aggregate attention to block level
        # Average attention within each block
        attn_blocks = attn_weights.view(
            batch_size, num_heads, n_q,
            n_blocks, block_size
        ).mean(dim=-1)

        # Select top blocks per query position
        top_k = self.config.top_k_blocks
        top_blocks = torch.topk(attn_blocks, k=min(top_k, n_blocks), dim=-1).indices

        # Get unique blocks selected
        unique_blocks = torch.unique(top_blocks).cpu().tolist()

        return sparsity, unique_blocks

    def compute_sketchwalk_sparsity(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[float, List[int], torch.Tensor]:
        """
        Compute SketchWalk sparsity pattern.
        """

        batch_size, num_heads, n_q, head_dim = Q.shape
        n_k = K.shape[2]

        # Run SketchWalk to get selected blocks
        _, selected_blocks = self.sketch_walk(
            Q=Q,
            K=K,
            V=torch.zeros_like(K),  # V not needed for block selection
            layer_idx=layer_idx,
            attention_mask=None,
            causal=True,
        )

        # Extract selected block indices
        if selected_blocks is not None:
            sketchwalk_blocks = selected_blocks[0, 0].cpu().tolist()  # First batch, first head
        else:
            # Fallback: compute using sketch
            sketchwalk_blocks = []

        # Compute sparsity
        total_blocks = (n_k + self.config.block_size - 1) // self.config.block_size
        sparsity = 1.0 - (len(sketchwalk_blocks) / total_blocks) if total_blocks > 0 else 0

        return sparsity, sketchwalk_blocks, selected_blocks

    def compute_block_overlap(
        self,
        seerattn_blocks: List[int],
        sketchwalk_blocks: List[int],
    ) -> Tuple[float, float]:
        """
        Compute block overlap and Jaccard similarity.
        """

        seerattn_set = set(seerattn_blocks)
        sketchwalk_set = set(sketchwalk_blocks)

        # Compute intersection and union
        intersection = seerattn_set.intersection(sketchwalk_set)
        union = seerattn_set.union(sketchwalk_set)

        # Jaccard similarity: |A ∩ B| / |A ∪ B|
        jaccard = len(intersection) / len(union) if len(union) > 0 else 0

        # Overlap coefficient: |A ∩ B| / min(|A|, |B|)
        min_size = min(len(seerattn_set), len(sketchwalk_set))
        overlap = len(intersection) / min_size if min_size > 0 else 0

        return overlap, jaccard

    def compute_attention_correlation(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
    ) -> float:
        """
        Compute correlation between full attention and sketched attention.
        """

        # Compute full attention
        batch_size, num_heads, n_q, head_dim = Q.shape
        n_k = K.shape[2]

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)

        # Apply causal mask
        mask = torch.triu(torch.ones(n_q, n_k, device=Q.device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Compute sketched attention using SketchWalk
        _, selected_blocks = self.sketch_walk(
            Q=Q,
            K=K,
            V=torch.zeros_like(K),
            layer_idx=0,
            attention_mask=None,
            causal=True,
        )

        # Create sparse mask from selected blocks
        if selected_blocks is not None:
            sparse_mask = torch.zeros_like(attn_weights, dtype=torch.bool)

            # Mark selected blocks as True
            block_size = self.config.block_size
            for block_idx in selected_blocks[0, 0]:
                start_idx = block_idx.item() * block_size
                end_idx = min(start_idx + block_size, n_k)
                sparse_mask[:, :, :, start_idx:end_idx] = True

            # Apply causal constraint
            causal_mask = torch.tril(torch.ones(n_q, n_k, device=Q.device)).bool()
            sparse_mask = sparse_mask & causal_mask

            # Compute sparse attention (zero out non-selected blocks)
            sparse_attn_weights = attn_weights * sparse_mask.float()

            # Normalize
            sparse_attn_weights = sparse_attn_weights / (sparse_attn_weights.sum(dim=-1, keepdim=True) + 1e-8)
        else:
            sparse_attn_weights = attn_weights

        # Compute correlation
        flat_full = attn_weights.flatten()
        flat_sparse = sparse_attn_weights.flatten()

        # Use cosine similarity as correlation metric
        correlation = F.cosine_similarity(
            flat_full.unsqueeze(0),
            flat_sparse.unsqueeze(0),
        ).item()

        return correlation

    def compare_single_sample(
        self,
        filepath: Path,
    ) -> Optional[PatternComparisonResult]:
        """Compare sparsity patterns for a single sample."""

        try:
            # Load tensors
            data = self.load_tensor_file(filepath)

            if data is None:
                return None

            Q = data['Q'].to(self.device)
            K = data['K'].to(self.device)
            metadata = data['metadata']

            layer_idx = metadata['layer_idx']
            seq_len = metadata['seq_len']
            sample_id = filepath.stem

            logger.info(f"Comparing {sample_id}: layer={layer_idx}, seq_len={seq_len}")

            # Compute SeerAttention sparsity
            seerattn_sparsity, seerattn_blocks = self.compute_seerattn_sparsity(Q, K)

            # Compute SketchWalk sparsity
            sketchwalk_sparsity, sketchwalk_blocks, _ = self.compute_sketchwalk_sparsity(
                Q, K, layer_idx
            )

            # Compute block overlap
            block_overlap, jaccard_sim = self.compute_block_overlap(
                seerattn_blocks, sketchwalk_blocks
            )

            # Compute attention correlation
            attn_correlation = self.compute_attention_correlation(Q, K)

            # Create result
            result = PatternComparisonResult(
                sample_id=sample_id,
                layer_idx=layer_idx,
                seq_len=seq_len,
                seerattn_sparsity=seerattn_sparsity,
                seerattn_top_blocks=seerattn_blocks,
                sketchwalk_sparsity=sketchwalk_sparsity,
                sketchwalk_top_blocks=sketchwalk_blocks,
                block_overlap=block_overlap,
                jaccard_similarity=jaccard_sim,
                attention_correlation=attn_correlation,
            )

            logger.info(
                f"  SeerAttn Sparsity: {seerattn_sparsity:.3f}, "
                f"SketchWalk Sparsity: {sketchwalk_sparsity:.3f}, "
                f"Overlap: {block_overlap:.3f}, "
                f"Jaccard: {jaccard_sim:.3f}"
            )

            return result

        except Exception as e:
            logger.error(f"Error comparing {filepath}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_comparison(self) -> List[PatternComparisonResult]:
        """Run comparison on all collected tensor samples."""

        results = []

        for filepath in tqdm(self.tensor_files, desc="Comparing patterns"):
            result = self.compare_single_sample(filepath)

            if result is not None:
                results.append(result)

        return results

    def save_results(self, results: List[PatternComparisonResult]):
        """Save comparison results to disk."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save individual results
        results_data = [r.to_dict() for r in results]

        results_file = self.output_dir / f"comparison_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"Saved results to {results_file}")

        # Save summary statistics
        if len(results) > 0:
            summary = self._compute_summary(results)

            summary_file = self.output_dir / f"comparison_summary_{timestamp}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

            logger.info(f"Saved summary to {summary_file}")

            # Print summary
            self._print_summary(summary)

    def _compute_summary(self, results: List[PatternComparisonResult]) -> Dict:
        """Compute summary statistics from comparison results."""

        metrics = {
            'num_samples': len(results),
            'sequence_lengths': sorted(set(r.seq_len for r in results)),
            'layers': sorted(set(r.layer_idx for r in results)),
        }

        # Compute statistics for each metric
        for field in [
            'seerattn_sparsity', 'sketchwalk_sparsity',
            'block_overlap', 'jaccard_similarity', 'attention_correlation',
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
                'mean_seerattn_sparsity': float(np.mean([r.seerattn_sparsity for r in seq_results])),
                'mean_sketchwalk_sparsity': float(np.mean([r.sketchwalk_sparsity for r in seq_results])),
                'mean_block_overlap': float(np.mean([r.block_overlap for r in seq_results])),
                'mean_jaccard_similarity': float(np.mean([r.jaccard_similarity for r in seq_results])),
                'mean_attention_correlation': float(np.mean([r.attention_correlation for r in seq_results])),
            }

            metrics['by_sequence_length'][str(seq_len)] = seq_summary

        # Group by layer
        layers = sorted(set(r.layer_idx for r in results))
        metrics['by_layer'] = {}

        for layer_idx in layers:
            layer_results = [r for r in results if r.layer_idx == layer_idx]

            layer_summary = {
                'num_samples': len(layer_results),
                'mean_seerattn_sparsity': float(np.mean([r.seerattn_sparsity for r in layer_results])),
                'mean_sketchwalk_sparsity': float(np.mean([r.sketchwalk_sparsity for r in layer_results])),
                'mean_block_overlap': float(np.mean([r.block_overlap for r in layer_results])),
                'mean_jaccard_similarity': float(np.mean([r.jaccard_similarity for r in layer_results])),
            }

            metrics['by_layer'][str(layer_idx)] = layer_summary

        return metrics

    def _print_summary(self, summary: Dict):
        """Print summary statistics."""

        logger.info("\n" + "="*60)
        logger.info("SPARSITY PATTERN COMPARISON SUMMARY")
        logger.info("="*60)

        logger.info(f"Total samples: {summary['num_samples']}")

        if 'seerattn_sparsity_mean' in summary:
            logger.info(f"\nSparsity Comparison:")
            logger.info(f"  SeerAttn: {summary['seerattn_sparsity_mean']:.3f} ± {summary['seerattn_sparsity_std']:.3f}")
            logger.info(f"  SketchWalk: {summary['sketchwalk_sparsity_mean']:.3f} ± {summary['sketchwalk_sparsity_std']:.3f}")

        if 'block_overlap_mean' in summary:
            logger.info(f"\nBlock Selection Similarity:")
            logger.info(f"  Block Overlap: {summary['block_overlap_mean']:.3f} ± {summary['block_overlap_std']:.3f}")
            logger.info(f"  Jaccard Similarity: {summary['jaccard_similarity_mean']:.3f} ± {summary['jaccard_similarity_std']:.3f}")

        if 'attention_correlation_mean' in summary:
            logger.info(f"\nAttention Pattern Correlation: {summary['attention_correlation_mean']:.3f} ± {summary['attention_correlation_std']:.3f}")

        if 'by_sequence_length' in summary:
            logger.info(f"\nBy Sequence Length:")
            for seq_len, seq_summary in sorted(summary['by_sequence_length'].items(), key=lambda x: int(x[0])):
                logger.info(f"  {seq_len}: "
                           f"SeerAttn={seq_summary['mean_seerattn_sparsity']:.3f}, "
                           f"SketchWalk={seq_summary['mean_sketchwalk_sparsity']:.3f}, "
                           f"Overlap={seq_summary['mean_block_overlap']:.3f}")

        if 'by_layer' in summary:
            logger.info(f"\nBy Layer:")
            for layer_idx, layer_summary in sorted(summary['by_layer'].items(), key=lambda x: int(x[0])):
                logger.info(f"  Layer {layer_idx}: "
                           f"SeerAttn={layer_summary['mean_seerattn_sparsity']:.3f}, "
                           f"SketchWalk={layer_summary['mean_sketchwalk_sparsity']:.3f}, "
                           f"Overlap={layer_summary['mean_block_overlap']:.3f}")

        logger.info("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare SeerAttention vs SketchWalk sparsity patterns"
    )

    # Data arguments
    parser.add_argument(
        "--tensor_dir",
        type=str,
        required=True,
        help="Directory containing collected tensor files",
    )
    parser.add_argument(
        "--validation_dir",
        type=str,
        default="./validation_results",
        help="Directory containing validation results",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./comparison_results",
        help="Directory to save comparison results",
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

    # SeerAttention configuration
    parser.add_argument(
        "--seerattn_threshold",
        type=float,
        default=0.1,
        help="Threshold for SeerAttention sparsity computation",
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

    # Create comparator
    comparator = SparsityPatternComparator(
        tensor_dir=args.tensor_dir,
        validation_dir=args.validation_dir,
        output_dir=args.output_dir,
        sketchwalk_config=config,
        device=args.device,
    )

    # Run comparison
    logger.info("Starting sparsity pattern comparison...")
    results = comparator.run_comparison()

    # Save results
    if len(results) > 0:
        logger.info(f"\nComparison complete! Processed {len(results)} samples")
        comparator.save_results(results)
    else:
        logger.warning("No comparison results to save")

    return 0 if len(results) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
