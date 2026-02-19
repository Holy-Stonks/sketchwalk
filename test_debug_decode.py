#!/usr/bin/env python3
"""Debug decode issue."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from sketch_walk.common import SketchWalkAttention, create_sketch_walk_config

# Create config
config = create_sketch_walk_config(
    block_size=64,
    sketch_dim=64,
    top_k_blocks=16,
    sparsity_exponent=8,
)

batch_size = 2
num_heads = 8
num_kv_heads = 8
prefill_len = 64
head_dim = 64

attn = SketchWalkAttention(config, head_dim)

# Prefill phase
Q_prefill = torch.randn(batch_size, num_heads, prefill_len, head_dim)
K_prefill = torch.randn(batch_size, num_kv_heads, prefill_len, head_dim)
V_prefill = torch.randn(batch_size, num_kv_heads, prefill_len, head_dim)

print(f"Prefill: {prefill_len} tokens")
output_prefill, A_hat_prefill = attn(Q_prefill, K_prefill, V_prefill, layer_idx=2)

# Initialize cache
cache = attn.init_decode_cache(Q_prefill, K_prefill, V_prefill, A_hat_prefill, prefill_len)
print(f"After prefill: position={cache.current_position}, cached_blocks shape={cache.cached_key_blocks.shape}")

# Decode step 1
Q_t = torch.randn(batch_size, num_heads, 1, head_dim)
K_t = torch.randn(batch_size, num_kv_heads, 1, head_dim)
V_t = torch.randn(batch_size, num_kv_heads, 1, head_dim)

K_cache = torch.cat([K_prefill, K_t], dim=2)
V_cache = torch.cat([V_prefill, V_t], dim=2)

print(f"\nDecode step 1:")
print(f"  t (new position) = {cache.current_position + 1}")
t = cache.current_position + 1
B = config.block_size
b_curr = (t + B - 1) // B - 1
print(f"  b_curr = {b_curr}")
print(f"  t % B = {t % B}")

output, cache = attn.decode(Q_t, K_cache, V_cache, layer_idx=2, cache=cache)
print(f"  Success! position={cache.current_position}")

# Decode step 2
Q_t = torch.randn(batch_size, num_heads, 1, head_dim)
K_t = torch.randn(batch_size, num_kv_heads, 1, head_dim)
V_t = torch.randn(batch_size, num_kv_heads, 1, head_dim)

K_cache = torch.cat([K_cache, K_t], dim=2)
V_cache = torch.cat([V_cache, V_t], dim=2)

print(f"\nDecode step 2:")
print(f"  t (new position) = {cache.current_position + 1}")
t = cache.current_position + 1
b_curr = (t + B - 1) // B - 1
print(f"  b_curr = {b_curr}")
print(f"  t % B = {t % B}")

# Check shapes before decode
print(f"  cached_key_blocks shape: {cache.cached_key_blocks.shape}")
print(f"  key_block_counts shape: {cache.key_block_counts.shape}")
print(f"  key_block_counts[:, {b_curr}:{b_curr+1}] shape would be: {cache.key_block_counts[:, b_curr:b_curr+1].shape}")
print(f"  cached_key_blocks[:, {b_curr}:{b_curr+1}] shape would be: {cache.cached_key_blocks[:, b_curr:b_curr+1].shape}")

# Add monkey patch to debug
original_decode = attn.sketch.decode

def debug_decode(Q_t, K_t, cache, block_size):
    batch_size, num_heads, n_q, head_dim = Q_t.shape
    B = block_size
    t = cache.current_position + 1
    b_curr = (t + B - 1) // B - 1

    print(f"  Sketch.decode: t={t}, b_curr={b_curr}, t%B={t%B}")
    print(f"    cached_key_blocks shape before: {cache.cached_key_blocks.shape}")
    print(f"    key_block_counts shape before: {cache.key_block_counts.shape}")

    if t % B != 1:  # Only debug the else branch
        K_avg = K_t.mean(dim=1).squeeze(1)
        print(f"    K_avg shape: {K_avg.shape}")
        print(f"    K_avg.unsqueeze(1) shape: {K_avg.unsqueeze(1).shape}")

        c = cache.key_block_counts[:, b_curr:b_curr+1].float()
        print(f"    c (key_block_counts[:, {b_curr}:{b_curr+1}]) shape: {c.shape}")
        old_k_bar = cache.cached_key_blocks[:, b_curr:b_curr+1].contiguous()
        print(f"    old_k_bar (cached_key_blocks[:, {b_curr}:{b_curr+1}].contiguous()) shape: {old_k_bar.shape}")

        # Check each component
        term1 = c * old_k_bar
        print(f"    c * old_k_bar shape: {term1.shape}")
        term2 = K_avg.unsqueeze(1)
        print(f"    K_avg.unsqueeze(1) shape: {term2.shape}")
        sum_term = term1 + term2
        print(f"    (c * old_k_bar + K_avg.unsqueeze(1)) shape: {sum_term.shape}")
        divisor = (c + 1)
        print(f"    (c + 1) shape: {divisor.shape}")

        new_k_bar = sum_term / divisor
        print(f"    new_k_bar shape: {new_k_bar.shape}")
        print(f"    Expected shape: ({batch_size}, 1, {head_dim})")

        # Check stride
        print(f"    old_k_bar stride: {old_k_bar.stride()}")
        print(f"    K_avg.unsqueeze(1) stride: {K_avg.unsqueeze(1).stride()}")

    return original_decode(Q_t, K_t, cache, block_size)

attn.sketch.decode = debug_decode

output, cache = attn.decode(Q_t, K_cache, V_cache, layer_idx=2, cache=cache)
print(f"  Success! position={cache.current_position}")

print("\nAll decode steps succeeded!")
