#!/usr/bin/env python3
"""
Test Script for Tensor Collection and Validation

This script tests the tensor collection and validation pipeline with
synthetic data to ensure everything works correctly.

Usage:
    python test_tensor_collection.py --device cuda
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_synthetic_tensors(
    batch_size: int = 1,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    seq_len: int = 4096,
    head_dim: int = 128,
    device: str = "cuda",
) -> tuple:
    """Create synthetic Q, K, V tensors for testing."""

    logger.info(f"Creating synthetic tensors: batch={batch_size}, heads={num_heads}, seq_len={seq_len}")

    Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
    K = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
    V = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)

    # Normalize for numerical stability
    Q = F.normalize(Q.float(), dim=-1).bfloat16()
    K = F.normalize(K.float(), dim=-1).bfloat16()
    V = F.normalize(V.float(), dim=-1).bfloat16()

    return Q, K, V


def test_sketchwalk_import():
    """Test that SketchWalk can be imported."""

    logger.info("Testing SketchWalk import...")

    try:
        from sketch_walk.common.core import (
            SketchWalkConfig,
            SketchWalkAttention,
            create_sketch_walk_config,
        )

        logger.info("✓ SketchWalk imported successfully")

        # Test config creation
        config = SketchWalkConfig(
            block_size=64,
            sketch_dim=64,
            top_k_blocks=16,
        )

        logger.info(f"✓ Created config: {config}")
        logger.info(f"✓ Target sparsity for seq_len=4096: {config.sparsity_level(4096):.3f}")

        return True

    except ImportError as e:
        logger.error(f"✗ Failed to import SketchWalk: {e}")
        return False


def test_sketchwalk_forward(device: str = "cuda"):
    """Test SketchWalk forward pass with synthetic data."""

    logger.info("Testing SketchWalk forward pass...")

    try:
        from sketch_walk.common.core import (
            SketchWalkConfig,
            SketchWalkAttention,
        )

        # Create synthetic data
        batch_size = 1
        num_heads = 32
        num_kv_heads = 8
        seq_len = 2048
        head_dim = 128

        Q, K, V = create_synthetic_tensors(
            batch_size=batch_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            device=device,
        )

        # Create SketchWalk attention
        config = SketchWalkConfig(
            block_size=64,
            sketch_dim=64,
            top_k_blocks=16,
            device=torch.device(device),
        )

        sketch_walk = SketchWalkAttention(
            config=config,
            head_dim=head_dim,
        )

        logger.info(f"✓ Created SketchWalk attention")

        # For GQA, repeat K/V to match Q's head count
        if num_kv_heads < num_heads:
            repeat_factor = num_heads // num_kv_heads
            K = K.repeat_interleave(repeat_factor, dim=1)
            V = V.repeat_interleave(repeat_factor, dim=1)
            logger.info(f"Repeated K/V for GQA: {num_kv_heads} -> {num_heads}")

        # Run forward pass
        logger.info("Running forward pass...")

        output, selected_blocks = sketch_walk(
            Q=Q,
            K=K,
            V=V,
            layer_idx=0,
            attention_mask=None,
            causal=True,
        )

        logger.info(f"✓ Forward pass completed")
        logger.info(f"  Output shape: {output.shape}")
        logger.info(f"  Selected blocks shape: {selected_blocks.shape if selected_blocks is not None else 'None'}")

        # Validate output
        expected_shape = (batch_size, num_heads, seq_len, head_dim)
        assert output.shape == expected_shape, f"Output shape mismatch: {output.shape} vs {expected_shape}"

        logger.info("✓ Output shape validated")

        # Compute sparsity
        if selected_blocks is not None:
            num_blocks = selected_blocks.shape[-1]
            total_blocks = (seq_len + config.block_size - 1) // config.block_size
            sparsity = 1.0 - (num_blocks / total_blocks)

            logger.info(f"✓ Sparsity: {sparsity:.3f} (selected {num_blocks}/{total_blocks} blocks)")

        return True

    except Exception as e:
        logger.error(f"✗ SketchWalk forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dense_vs_sparse_comparison(device: str = "cuda"):
    """Test comparison between dense and sparse attention."""

    logger.info("Testing dense vs sparse comparison...")

    try:
        from sketch_walk.common.core import (
            SketchWalkConfig,
            SketchWalkAttention,
        )

        # Create synthetic data
        batch_size = 1
        num_heads = 32
        num_kv_heads = 8
        seq_len = 2048
        head_dim = 128

        Q, K, V = create_synthetic_tensors(
            batch_size=batch_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            device=device,
        )

        # Compute dense attention
        logger.info("Computing dense attention...")

        # Repeat K/V for GQA
        repeat_factor = num_heads // num_kv_heads
        K_expanded = K.repeat_interleave(repeat_factor, dim=1)
        V_expanded = V.repeat_interleave(repeat_factor, dim=1)

        # Compute attention scores
        # Ensure consistent dtype for matmul
        Q_compute = Q.float() if Q.dtype != torch.float32 else Q
        K_compute = K_expanded.float() if K_expanded.dtype != torch.float32 else K_expanded
        attn_scores = torch.matmul(Q_compute, K_compute.transpose(-2, -1)) / (head_dim ** 0.5)

        # Apply causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Compute output
        V_compute = V_expanded.float() if V_expanded.dtype != torch.float32 else V_expanded
        dense_output = torch.matmul(attn_weights, V_compute)

        logger.info(f"✓ Dense attention computed: shape={dense_output.shape}")

        # Compute sparse attention with SketchWalk
        logger.info("Computing sparse attention...")

        config = SketchWalkConfig(
            block_size=64,
            sketch_dim=64,
            top_k_blocks=16,
            device=torch.device(device),
        )

        sketch_walk = SketchWalkAttention(
            config=config,
            head_dim=head_dim,
        )

        sparse_output, selected_blocks = sketch_walk(
            Q=Q,
            K=K_expanded,
            V=V_expanded,
            layer_idx=0,
            attention_mask=None,
            causal=True,
        )

        logger.info(f"✓ Sparse attention computed: shape={sparse_output.shape}")

        # Compare outputs
        logger.info("Comparing outputs...")

        # Flatten for comparison
        dense_flat = dense_output.flatten()
        sparse_flat = sparse_output.flatten()

        # Cosine similarity
        cosine_sim = F.cosine_similarity(
            dense_flat.unsqueeze(0),
            sparse_flat.unsqueeze(0),
        ).item()

        # MSE
        mse = F.mse_loss(dense_output, sparse_output).item()

        # Max error
        max_error = torch.max(torch.abs(dense_output - sparse_output)).item()

        logger.info(f"✓ Comparison metrics:")
        logger.info(f"  Cosine Similarity: {cosine_sim:.4f}")
        logger.info(f"  MSE: {mse:.6f}")
        logger.info(f"  Max Error: {max_error:.6f}")

        # Check if metrics are reasonable
        if cosine_sim < 0.8:
            logger.warning(f"⚠ Low cosine similarity: {cosine_sim:.4f}")

        if mse > 0.1:
            logger.warning(f"⚠ High MSE: {mse:.6f}")

        return True

    except Exception as e:
        logger.error(f"✗ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tensor_saving(output_dir: str = "./test_tensor_data", device: str = "cuda"):
    """Test saving and loading tensors."""

    logger.info("Testing tensor saving and loading...")

    try:
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create synthetic data
        batch_size = 1
        num_heads = 32
        num_kv_heads = 8
        seq_len = 2048
        head_dim = 128

        Q, K, V = create_synthetic_tensors(
            batch_size=batch_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            device=device,
        )

        # Create metadata
        metadata = {
            'layer_idx': 0,
            'seq_len': seq_len,
            'num_heads': num_heads,
            'num_kv_heads': num_kv_heads,
            'head_dim': head_dim,
            'hidden_size': num_heads * head_dim,
            'batch_size': batch_size,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'model_path': 'test_model',
        }

        # Save tensors
        filepath = output_path / "test_tensors.pt"

        tensor_data = {
            'Q': Q.cpu(),
            'K': K.cpu(),
            'V': V.cpu(),
            'metadata': metadata,
        }

        torch.save(tensor_data, filepath)

        logger.info(f"✓ Saved tensors to {filepath}")

        # Load tensors
        loaded_data = torch.load(filepath)

        logger.info(f"✓ Loaded tensors from {filepath}")

        # Validate loaded data
        assert loaded_data['Q'].shape == Q.shape
        assert loaded_data['K'].shape == K.shape
        assert loaded_data['V'].shape == V.shape
        assert loaded_data['metadata'] == metadata

        logger.info("✓ Loaded tensors validated")

        return True

    except Exception as e:
        logger.error(f"✗ Tensor saving/loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test tensor collection and validation")

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda/cpu)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_tensor_data",
        help="Directory for test outputs",
    )

    args = parser.parse_args()

    # Check if CUDA is available
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = "cpu"

    logger.info("="*60)
    logger.info("Testing Tensor Collection and Validation Pipeline")
    logger.info("="*60)
    logger.info(f"Device: {args.device}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("="*60)

    # Run tests
    tests = [
        ("SketchWalk Import", lambda: test_sketchwalk_import()),
        ("SketchWalk Forward Pass", lambda: test_sketchwalk_forward(args.device)),
        ("Dense vs Sparse Comparison", lambda: test_dense_vs_sparse_comparison(args.device)),
        ("Tensor Saving/Loading", lambda: test_tensor_saving(args.output_dir, args.device)),
    ]

    results = []

    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Test: {test_name}")
        logger.info(f"{'='*60}")

        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("Test Summary")
    logger.info(f"{'='*60}")

    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{status}: {test_name}")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    logger.info(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        logger.info("\n✓ All tests passed! The pipeline is ready to use.")
        return 0
    else:
        logger.warning(f"\n✗ {total - passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
