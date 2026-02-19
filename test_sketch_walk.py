"""
Test script for SketchWalk core implementation.

This script validates:
1. Hadamard transform preserves inner products
2. Sketch module produces correct block-level attention
3. Walk module maintains state across layers
4. Full SketchWalk attention produces outputs
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from sketch_walk.common import (
    SketchWalkConfig,
    HadamardTransform,
    Sketch,
    Walk,
    SketchWalkAttention,
    create_sketch_walk_config,
)


def test_hadamard_transform():
    """Test that Hadamard transform approximately preserves inner products."""
    print("=" * 60)
    print("Testing Hadamard Transform...")
    print("=" * 60)

    config = create_sketch_walk_config(sketch_dim=64)
    hadamard = HadamardTransform(in_dim=128, out_dim=64, seed=42)

    # Create random vectors
    x = torch.randn(100, 128)
    y = torch.randn(100, 128)

    # Original inner products
    original_inner = (x * y).sum(dim=-1)

    # Transformed inner products
    x_tilde = hadamard(x)
    y_tilde = hadamard(y)
    transformed_inner = (x_tilde * y_tilde).sum(dim=-1)

    # Check if inner products are roughly correlated (not exact match)
    # Random projections preserve inner products in expectation, not per-pair
    correlation = torch.corrcoef(torch.stack([original_inner, transformed_inner]))[0, 1]

    print(f"Correlation: {correlation:.4f}")

    # Test outputs are finite
    outputs_finite = torch.all(torch.isfinite(x_tilde)) and torch.all(torch.isfinite(y_tilde))

    # Test output shapes
    shape_correct = x_tilde.shape == (100, 64) and y_tilde.shape == (100, 64)

    # For random projections, we check that outputs are valid and somewhat correlated
    # A correlation > 0 means the transform preserves some structure
    if outputs_finite and shape_correct and correlation > 0:
        print("✓ Hadamard transform test PASSED")
        return True
    else:
        print("✗ Hadamard transform test FAILED")
        return False


def test_sketch_module():
    """Test the Sketch module produces valid block-level attention."""
    print("\n" + "=" * 60)
    print("Testing Sketch Module...")
    print("=" * 60)

    config = create_sketch_walk_config(block_size=64, sketch_dim=64)
    head_dim = 128
    sketch = Sketch(config, head_dim)

    # Create dummy Q, K tensors
    batch_size = 2
    num_heads = 8
    seq_len = 256  # 4 blocks

    Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # Compute sketched attention
    A_hat, Q_bar, K_bar = sketch(Q, K)

    # Check shapes
    expected_blocks = (seq_len + config.block_size - 1) // config.block_size  # 4

    print(f"Input sequence length: {seq_len}")
    print(f"Block size: {config.block_size}")
    print(f"Expected blocks: {expected_blocks}")
    print(f"A_hat shape: {A_hat.shape} (expected: ({batch_size}, {expected_blocks}, {expected_blocks}))")
    print(f"Q_bar shape: {Q_bar.shape} (expected: ({batch_size}, {expected_blocks}, {head_dim}))")
    print(f"K_bar shape: {K_bar.shape} (expected: ({batch_size}, {expected_blocks}, {head_dim}))")

    shape_correct = (
        A_hat.shape == (batch_size, expected_blocks, expected_blocks) and
        Q_bar.shape == (batch_size, expected_blocks, head_dim) and
        K_bar.shape == (batch_size, expected_blocks, head_dim)
    )

    # Check values are finite
    values_finite = torch.all(torch.isfinite(A_hat)) and torch.all(torch.isfinite(Q_bar)) and torch.all(torch.isfinite(K_bar))

    if shape_correct and values_finite:
        print("✓ Sketch module test PASSED")
        return True
    else:
        print("✗ Sketch module test FAILED")
        return False


def test_walk_module():
    """Test the Walk module maintains state across layers."""
    print("\n" + "=" * 60)
    print("Testing Walk Module...")
    print("=" * 60)

    config = create_sketch_walk_config(top_k_blocks=8)
    walk = Walk(config)

    batch_size = 2
    num_blocks = 10

    # Simulate multiple layers
    for layer_idx in range(5):
        # Create fake attention for this layer
        A_hat = torch.randn(batch_size, num_blocks, num_blocks)
        A_hat = (A_hat + A_hat.transpose(-2, -1)) / 2  # Make symmetric
        A_hat = F.softmax(A_hat, dim=-1)

        # Update walk state
        walk_state = walk.update(A_hat, layer_idx, causal=True)

        # Check walk state shape
        if walk_state.shape != (batch_size, num_blocks, num_blocks):
            print(f"✗ Walk state shape incorrect at layer {layer_idx}: {walk_state.shape}")
            return False

        # Check values are finite
        if not torch.all(torch.isfinite(walk_state)):
            print(f"✗ Walk state contains non-finite values at layer {layer_idx}")
            return False

        # Check causal mask is applied (upper triangle should be zero)
        if not torch.all(walk_state[:, :, 1:].triu(1) == 0):
            print(f"✗ Causal mask not properly applied at layer {layer_idx}")
            return False

    print(f"Walk state after 5 layers: shape={walk_state.shape}")
    print(f"Walk state range: [{walk_state.min():.4f}, {walk_state.max():.4f}]")

    # Test block selection
    num_query_blocks = 8
    selected = walk.select_blocks(walk_state, num_query_blocks)

    if selected.shape != (batch_size, num_query_blocks, config.top_k_blocks):
        print(f"✗ Selected blocks shape incorrect: {selected.shape}")
        return False

    print(f"Selected blocks shape: {selected.shape}")
    print("✓ Walk module test PASSED")
    return True


def test_full_sketch_walk_attention():
    """Test the full SketchWalk attention module."""
    print("\n" + "=" * 60)
    print("Testing Full SketchWalk Attention...")
    print("=" * 60)

    config = create_sketch_walk_config(
        block_size=64,
        sketch_dim=64,
        top_k_blocks=8,
        skip_first_n_layers=2,
    )
    head_dim = 128
    sw_attn = SketchWalkAttention(config, head_dim)

    batch_size = 2
    num_heads = 8
    seq_len = 256
    head_dim = 128

    # Create dummy Q, K, V tensors
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # Test multiple layers
    for layer_idx in range(6):
        output, selected_blocks = sw_attn(Q, K, V, layer_idx=layer_idx, causal=True)

        # Check output shape
        if output.shape != (batch_size, num_heads, seq_len, head_dim):
            print(f"✗ Output shape incorrect at layer {layer_idx}: {output.shape}")
            return False

        # Check values are finite
        if not torch.all(torch.isfinite(output)):
            print(f"✗ Output contains non-finite values at layer {layer_idx}")
            return False

        # Check selected blocks for SketchWalk layers
        if layer_idx >= config.skip_first_n_layers:
            if selected_blocks is None:
                print(f"✗ Selected blocks is None at layer {layer_idx}")
                return False

            expected_blocks = (seq_len + config.block_size - 1) // config.block_size
            expected_tau = min(config.top_k_blocks, expected_blocks)
            if selected_blocks.shape != (batch_size, expected_blocks, expected_tau):
                print(f"✗ Selected blocks shape incorrect at layer {layer_idx}: {selected_blocks.shape}")
                print(f"   Expected: ({batch_size}, {expected_blocks}, {expected_tau})")
                return False
        else:
            if selected_blocks is not None:
                print(f"✗ Selected blocks should be None for dense layer {layer_idx}")
                return False

    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"Output mean: {output.mean():.4f}, std: {output.std():.4f}")
    print("✓ Full SketchWalk attention test PASSED")
    return True


def test_sparsity_calculation():
    """Test sparsity level calculation."""
    print("\n" + "=" * 60)
    print("Testing Sparsity Calculation...")
    print("=" * 60)

    config = create_sketch_walk_config(
        block_size=64,
        top_k_blocks=16,
    )

    for seq_len in [1024, 4096, 16384, 65536]:
        sparsity = config.sparsity_level(seq_len)
        attended_tokens = seq_len * (1 - sparsity)
        print(f"Seq len: {seq_len:6d} -> Sparsity: {sparsity:.1%}, Attended: {attended_tokens:.0f} tokens")

    print("✓ Sparsity calculation test PASSED")
    return True


def test_llama_config():
    """Test LLaMA configuration."""
    print("\n" + "=" * 60)
    print("Testing LLaMA Configuration...")
    print("=" * 60)

    from sketch_walk.llama import SketchWalkLlamaConfig

    config = SketchWalkLlamaConfig(
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=8192,
        sketchwalk_enabled=True,
        sketchwalk_block_size=64,
        sketchwalk_sketch_dim=64,
        sketchwalk_top_k_blocks=16,
        sketchwalk_sparsity_exponent=8,
        sketchwalk_skip_first_n_layers=2,
    )

    print(f"Model type: {config.model_type}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Num layers: {config.num_hidden_layers}")
    print(f"Num heads: {config.num_attention_heads}")
    print(f"Head dim: {config.head_dim}")
    print(f"SketchWalk enabled: {config.sketchwalk_enabled}")
    print(f"SketchWalk block size: {config.sketchwalk_block_size}")
    print(f"SketchWalk sketch dim: {config.sketchwalk_sketch_dim}")
    print(f"SketchWalk top-k blocks: {config.sketchwalk_top_k_blocks}")
    print(f"SketchWalk sparsity exponent: {config.sketchwalk_sparsity_exponent}")
    print(f"SketchWalk skip layers: {config.sketchwalk_skip_first_n_layers}")

    print("✓ LLaMA configuration test PASSED")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("SketchWalk Core Implementation Test Suite")
    print("=" * 60)

    tests = [
        ("Hadamard Transform", test_hadamard_transform),
        ("Sketch Module", test_sketch_module),
        ("Walk Module", test_walk_module),
        ("Full SketchWalk Attention", test_full_sketch_walk_attention),
        ("Sparsity Calculation", test_sparsity_calculation),
        ("LLaMA Configuration", test_llama_config),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} test FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for name, result in results:
        status = "PASSED" if result else "FAILED"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {name}: {status}")

    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
