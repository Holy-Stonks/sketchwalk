#!/usr/bin/env python3
"""
Test SketchWalk decode implementation (Algorithm 2).

This script tests the decode step after a prefill phase.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from sketch_walk.common import SketchWalkAttention, create_sketch_walk_config

def test_decode_single_step():
    """Test a single decode step after prefill."""
    print("=" * 60)
    print("Test 1: Single Decode Step After Prefill")
    print("=" * 60)

    # Prefill phase: process 128 tokens (2 blocks)
    batch_size = 2
    num_heads = 8
    num_kv_heads = 8
    seq_len = 128
    head_dim = 64

    # Create config
    config = create_sketch_walk_config(
        block_size=64,
        sketch_dim=64,
        top_k_blocks=16,
        sparsity_exponent=8,
    )

    # Create attention module
    attn = SketchWalkAttention(config, head_dim)

    Q_prefill = torch.randn(batch_size, num_heads, seq_len, head_dim)
    K_prefill = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
    V_prefill = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)

    # Run prefill
    print(f"Running prefill with {seq_len} tokens...")
    output_prefill, A_hat_prefill = attn(
        Q_prefill, K_prefill, V_prefill, layer_idx=2
    )
    print(f"Prefill output shape: {output_prefill.shape}")
    print(f"Block attention shape: {A_hat_prefill.shape}")

    # Initialize decode cache
    cache = attn.init_decode_cache(
        Q_prefill, K_prefill, V_prefill, A_hat_prefill, seq_len
    )
    print(f"Cache initialized. Position: {cache.current_position}")
    print(f"Cached key blocks shape: {cache.cached_key_blocks.shape}")
    print(f"Cached block attn shape: {cache.cached_block_attn.shape}")

    # Decode step: process 1 new token
    Q_t = torch.randn(batch_size, num_heads, 1, head_dim)
    K_t = torch.randn(batch_size, num_kv_heads, 1, head_dim)
    V_t = torch.randn(batch_size, num_kv_heads, 1, head_dim)

    # Extend cache with new token
    K_cache = torch.cat([K_prefill, K_t], dim=2)
    V_cache = torch.cat([V_prefill, V_t], dim=2)

    print(f"\nRunning decode step...")
    output_decode, cache_updated = attn.decode(
        Q_t, K_cache, V_cache, layer_idx=2, cache=cache
    )

    print(f"Decode output shape: {output_decode.shape}")
    print(f"Updated cache position: {cache_updated.current_position}")

    assert output_decode.shape == (batch_size, num_heads, 1, head_dim), \
        f"Expected shape {(batch_size, num_heads, 1, head_dim)}, got {output_decode.shape}"
    # After processing 1 decode token, position should be prefill_len + 1
    assert cache_updated.current_position == seq_len + 1, \
        f"Expected position {seq_len + 1}, got {cache_updated.current_position}"

    print("✓ Test 1 passed!")
    return True


def test_decode_multiple_steps():
    """Test multiple decode steps sequentially."""
    print("\n" + "=" * 60)
    print("Test 2: Multiple Decode Steps")
    print("=" * 60)

    # Prefill phase
    batch_size = 2
    num_heads = 8
    num_kv_heads = 8
    prefill_len = 64
    decode_steps = 10
    head_dim = 64

    # Create config
    config = create_sketch_walk_config(
        block_size=64,
        sketch_dim=64,
        top_k_blocks=16,
        sparsity_exponent=8,
    )

    # Create attention module
    attn = SketchWalkAttention(config, head_dim)

    Q_prefill = torch.randn(batch_size, num_heads, prefill_len, head_dim)
    K_prefill = torch.randn(batch_size, num_kv_heads, prefill_len, head_dim)
    V_prefill = torch.randn(batch_size, num_kv_heads, prefill_len, head_dim)

    print(f"Running prefill with {prefill_len} tokens...")
    output_prefill, A_hat_prefill = attn(
        Q_prefill, K_prefill, V_prefill, layer_idx=2
    )

    # Initialize cache
    cache = attn.init_decode_cache(
        Q_prefill, K_prefill, V_prefill, A_hat_prefill, prefill_len
    )

    # Run multiple decode steps
    K_cache = K_prefill.clone()
    V_cache = V_prefill.clone()

    print(f"\nRunning {decode_steps} decode steps...")
    for step in range(decode_steps):
        Q_t = torch.randn(batch_size, num_heads, 1, head_dim)
        K_t = torch.randn(batch_size, num_kv_heads, 1, head_dim)
        V_t = torch.randn(batch_size, num_kv_heads, 1, head_dim)

        K_cache = torch.cat([K_cache, K_t], dim=2)
        V_cache = torch.cat([V_cache, V_t], dim=2)

        output, cache = attn.decode(
            Q_t, K_cache, V_cache, layer_idx=2, cache=cache
        )

        expected_pos = prefill_len + step + 1
        assert cache.current_position == expected_pos, \
            f"Step {step}: Expected position {expected_pos}, got {cache.current_position}"
        assert output.shape == (batch_size, num_heads, 1, head_dim), \
            f"Step {step}: Unexpected output shape {output.shape}"

        print(f"  Step {step + 1}: position={cache.current_position}, output_shape={output.shape}")

    print(f"\nFinal cache position: {cache.current_position}")
    print(f"Final cache shape: {cache.cached_key_blocks.shape}")

    print("✓ Test 2 passed!")
    return True


def test_decode_across_block_boundary():
    """Test decode steps that cross block boundaries."""
    print("\n" + "=" * 60)
    print("Test 3: Decode Across Block Boundary")
    print("=" * 60)

    # Prefill phase: 1 block (8 tokens)
    batch_size = 2
    num_heads = 4
    num_kv_heads = 4
    prefill_len = 8
    head_dim = 32

    # Create config with small block size for testing
    config = create_sketch_walk_config(
        block_size=8,
        sketch_dim=16,
        top_k_blocks=4,
        sparsity_exponent=4,
    )

    # Create attention module
    attn = SketchWalkAttention(config, head_dim)

    Q_prefill = torch.randn(batch_size, num_heads, prefill_len, head_dim)
    K_prefill = torch.randn(batch_size, num_kv_heads, prefill_len, head_dim)
    V_prefill = torch.randn(batch_size, num_kv_heads, prefill_len, head_dim)

    print(f"Prefill: {prefill_len} tokens (1 block of size {config.block_size})")
    output_prefill, A_hat_prefill = attn(
        Q_prefill, K_prefill, V_prefill, layer_idx=2
    )

    cache = attn.init_decode_cache(
        Q_prefill, K_prefill, V_prefill, A_hat_prefill, prefill_len
    )

    K_cache = K_prefill.clone()
    V_cache = V_prefill.clone()

    # Run decode steps to cross into second block
    # Position 8 is start of block 1 (0-indexed)
    decode_steps = 12  # This will take us from position 8 to 19 (blocks 1 and 2)

    print(f"\nRunning {decode_steps} decode steps (crossing block boundaries)...")
    for step in range(decode_steps):
        Q_t = torch.randn(batch_size, num_heads, 1, head_dim)
        K_t = torch.randn(batch_size, num_kv_heads, 1, head_dim)
        V_t = torch.randn(batch_size, num_kv_heads, 1, head_dim)

        K_cache = torch.cat([K_cache, K_t], dim=2)
        V_cache = torch.cat([V_cache, V_t], dim=2)

        output, cache = attn.decode(
            Q_t, K_cache, V_cache, layer_idx=2, cache=cache
        )

        expected_pos = prefill_len + step + 1
        expected_block = (expected_pos - 1) // config.block_size

        assert cache.current_position == expected_pos, \
            f"Step {step}: Expected position {expected_pos}, got {cache.current_position}"

        print(f"  Step {step + 1}: pos={cache.current_position}, block={expected_block}, " +
              f"cached_blocks={cache.cached_key_blocks.shape[1]}")

    print("\n✓ Test 3 passed!")
    return True


def test_decode_skip_first_layers():
    """Test that decode respects skip_first_n_layers setting."""
    print("\n" + "=" * 60)
    print("Test 4: Decode Skip First Layers")
    print("=" * 60)

    batch_size = 2
    num_heads = 8
    num_kv_heads = 8
    seq_len = 128
    head_dim = 64

    # Create config that skips first 2 layers
    config = create_sketch_walk_config(
        block_size=64,
        sketch_dim=64,
        top_k_blocks=16,
        sparsity_exponent=8,
        skip_first_n_layers=2,
    )

    attn = SketchWalkAttention(config, head_dim)

    Q_prefill = torch.randn(batch_size, num_heads, seq_len, head_dim)
    K_prefill = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
    V_prefill = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)

    output_prefill, A_hat_prefill = attn(
        Q_prefill, K_prefill, V_prefill, layer_idx=2
    )

    cache = attn.init_decode_cache(
        Q_prefill, K_prefill, V_prefill, A_hat_prefill, seq_len
    )

    K_cache = torch.cat([K_prefill, torch.randn(batch_size, num_kv_heads, 1, head_dim)], dim=2)
    V_cache = torch.cat([V_prefill, torch.randn(batch_size, num_kv_heads, 1, head_dim)], dim=2)
    Q_t = torch.randn(batch_size, num_heads, 1, head_dim)

    # Layer 0 should use dense attention (skip)
    print("Testing layer 0 (should skip SketchWalk)...")
    output_layer0, cache_layer0 = attn.decode(
        Q_t, K_cache, V_cache, layer_idx=0, cache=cache
    )

    # Layer 2 should use SketchWalk
    print("Testing layer 2 (should use SketchWalk)...")
    output_layer2, cache_layer2 = attn.decode(
        Q_t, K_cache, V_cache, layer_idx=2, cache=cache
    )

    assert output_layer0.shape == (batch_size, num_heads, 1, head_dim)
    assert output_layer2.shape == (batch_size, num_heads, 1, head_dim)

    print("✓ Test 4 passed!")
    return True


def main():
    """Run all tests."""
    tests = [
        test_decode_single_step,
        test_decode_multiple_steps,
        test_decode_across_block_boundary,
        test_decode_skip_first_layers,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"\n✗ {test.__name__} failed:")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{len(tests)} tests passed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
