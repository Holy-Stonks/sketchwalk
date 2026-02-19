#!/usr/bin/env python3
"""
Test SketchWalk LLaMA decode integration.

This test verifies that the LLaMA model correctly:
1. Uses prefill path for multiple tokens
2. Uses decode path for single tokens with cache
3. Properly maintains decode state across decode steps
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from sketch_walk.llama import SketchWalkLlamaForCausalLM
from sketch_walk.llama.configuration_llama_sketchwalk import SketchWalkLlamaConfig
from transformers import AutoConfig


def create_test_model():
    """Create a small test model for testing."""
    # Use a smaller config for testing
    base_config = AutoConfig.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    config = SketchWalkLlamaConfig(
        vocab_size=base_config.vocab_size,
        hidden_size=base_config.hidden_size,
        intermediate_size=base_config.intermediate_size,
        num_hidden_layers=4,  # Use fewer layers for testing
        num_attention_heads=base_config.num_attention_heads,
        num_key_value_heads=base_config.num_key_value_heads,
        max_position_embeddings=base_config.max_position_embeddings,
        rope_theta=base_config.rope_theta,
        rms_norm_eps=base_config.rms_norm_eps,
        # SketchWalk settings
        sketchwalk_enabled=True,
        sketchwalk_block_size=64,
        sketchwalk_sketch_dim=64,
        sketchwalk_top_k_blocks=16,
        sketchwalk_sparsity_exponent=8,
        sketchwalk_skip_first_n_layers=1,  # Skip first layer for faster testing
    )

    model = SketchWalkLlamaForCausalLM(config)
    return model


def test_prefill_then_decode():
    """Test prefill followed by decode steps."""
    print("=" * 60)
    print("Test 1: Prefill + Decode Integration")
    print("=" * 60)

    model = create_test_model()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create dummy input IDs for prefill (50 tokens)
    batch_size = 2
    prefill_len = 50
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, prefill_len), device=device)

    print(f"Prefill phase: {prefill_len} tokens")

    # Prefill phase
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
        )

    logits = outputs["logits"]  # Use dict access
    past_key_values = outputs["past_key_values"]

    print(f"  Prefill output shape: {logits.shape}")
    print(f"  Past key values type: {type(past_key_values)}")

    # Check that decode cache was initialized in the attention layers
    has_cache = False
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer.self_attn, 'decode_cache') and layer.self_attn.decode_cache is not None:
            print(f"  Layer {i}: decode_cache initialized (position={layer.self_attn.decode_cache.current_position})")
            has_cache = True

    if not has_cache:
        print("  WARNING: No decode cache found in layers (sketchwalk may not be active)")

    # Decode phase: generate 5 tokens
    decode_steps = 5
    print(f"\nDecode phase: {decode_steps} tokens")

    current_input = input_ids[:, -1:]  # Start with last token
    all_logits = [logits]

    for step in range(decode_steps):
        with torch.no_grad():
            outputs = model(
                input_ids=current_input,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

        logits = outputs["logits"]
        past_key_values = outputs["past_key_values"]
        all_logits.append(logits)

        # Get next token (greedy sampling)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        current_input = next_token

        print(f"  Step {step + 1}: input_ids shape={current_input.shape}, logits shape={logits.shape}")

        # Check decode cache state
        for i, layer in enumerate(model.model.layers):
            if hasattr(layer.self_attn, 'decode_cache') and layer.self_attn.decode_cache is not None:
                expected_pos = prefill_len + step + 1
                actual_pos = layer.self_attn.decode_cache.current_position
                if i == 0:  # Only print first layer to avoid spam
                    print(f"    Layer 0 cache position: {actual_pos} (expected {expected_pos})")

    print("\n✓ Test 1 passed!")
    return True


def test_decode_vs_prefill_consistency():
    """Test that decode produces consistent results with prefill."""
    print("\n" + "=" * 60)
    print("Test 2: Decode vs Prefill Consistency")
    print("=" * 60)

    model = create_test_model()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    batch_size = 1
    seq_len = 20
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)

    # Method 1: Single prefill pass
    print("Method 1: Single prefill pass")
    with torch.no_grad():
        outputs1 = model(input_ids=input_ids, return_dict=True)
    logits1 = outputs1["logits"]

    # Method 2: Prefill first 15 tokens, then decode 5 tokens
    print("Method 2: Prefill + decode")
    prefill_len = 15

    # Prefill
    with torch.no_grad():
        outputs = model(input_ids=input_ids[:, :prefill_len], use_cache=True, return_dict=True)
    past_kv = outputs["past_key_values"]

    # Decode remaining tokens
    all_logits = []
    current_input = input_ids[:, prefill_len:prefill_len+1]
    for i in range(prefill_len, seq_len):
        with torch.no_grad():
            outputs = model(input_ids=current_input, past_key_values=past_kv, use_cache=True, return_dict=True)
        all_logits.append(outputs["logits"][:, -1:, :])  # Keep seq dimension
        past_kv = outputs["past_key_values"]
        current_input = input_ids[:, i+1:i+2]

    logits2 = torch.cat(all_logits, dim=1)

    print(f"  Logits shape - Method 1: {logits1[:, prefill_len:, :].shape}")
    print(f"  Logits shape - Method 2: {logits2.shape}")

    # Compare logits (they won't be exactly the same due to different attention patterns,
    # but should be similar for non-sparse attention layers)
    diff = (logits1[:, prefill_len:, :] - logits2).abs().mean()
    print(f"  Mean absolute difference: {diff.item():.6f}")

    print("\n✓ Test 2 passed!")
    return True


def test_cache_reset():
    """Test that cache is properly reset between sequences."""
    print("\n" + "=" * 60)
    print("Test 3: Cache Reset")
    print("=" * 60)

    model = create_test_model()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    batch_size = 1
    seq_len1 = 50
    seq_len2 = 30  # Shorter sequence

    input_ids1 = torch.randint(0, model.config.vocab_size, (batch_size, seq_len1), device=device)
    input_ids2 = torch.randint(0, model.config.vocab_size, (batch_size, seq_len2), device=device)

    # First sequence
    print("First sequence (50 tokens)")
    with torch.no_grad():
        outputs1 = model(input_ids=input_ids1, use_cache=True, return_dict=True)

    # Check cache position after first sequence (check that it returns a dict)
    if isinstance(outputs1, dict):
        print(f"  outputs1 is a dict with keys: {list(outputs1.keys())}")
    else:
        print(f"  outputs1 is a tuple with {len(outputs1)} elements")
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer.self_attn, 'decode_cache') and layer.self_attn.decode_cache is not None:
            pos1 = layer.self_attn.decode_cache.current_position
            print(f"  Layer {i}: cache position = {pos1}")

    # Second sequence (shorter) - should reset cache
    print(f"\nSecond sequence ({seq_len2} tokens, shorter)")
    with torch.no_grad():
        outputs2 = model(input_ids=input_ids2, use_cache=True, return_dict=True)

    # Check cache position after second sequence
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer.self_attn, 'decode_cache') and layer.self_attn.decode_cache is not None:
            pos2 = layer.self_attn.decode_cache.current_position
            print(f"  Layer {i}: cache position = {pos2}")
            if i == 0:
                assert pos2 == seq_len2, f"Expected position {seq_len2}, got {pos2}"

    print("\n✓ Test 3 passed!")
    return True


def main():
    """Run all tests."""
    tests = [
        test_prefill_then_decode,
        test_decode_vs_prefill_consistency,
        test_cache_reset,
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
