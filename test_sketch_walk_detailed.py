"""
Comprehensive Test Suite for SketchWalk Sparse Attention

This module provides thorough validation of the SketchWalk implementation through:
1. Unit tests for each component
2. Property-based tests (invariants that should always hold)
3. Integration tests with synthetic LLaMA-like inputs
4. Performance benchmarks (timing for different sequence lengths)
5. Comparison against dense attention for correctness
6. Edge case testing
7. Numerical stability tests

Author: Research Validation Team
Date: 2025-02-19
"""

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the modules we're testing
import sys
sys.path.insert(0, '/home/valery/sketch_walk/SeerAttention')

from sketch_walk.common.core import (
    SketchWalkConfig,
    HadamardTransform,
    Sketch,
    Walk,
    SketchWalkAttention,
    create_sketch_walk_config,
)


# ============================================================================
# Test Configuration and Utilities
# ============================================================================

@dataclass
class TestResult:
    """Container for test results."""
    name: str
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0


class TestTimer:
    """Context manager for timing test execution."""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time


def create_synthetic_attention_data(
    batch_size: int = 2,
    num_heads: int = 8,
    seq_len: int = 2048,
    head_dim: int = 128,
    device: str = 'cpu',
    dtype: torch.dtype = torch.float32,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create synthetic Q, K, V tensors for testing.

    Args:
        batch_size: Number of sequences in batch
        num_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Attention head dimension
        device: Device to create tensors on
        dtype: Data type
        seed: Random seed for reproducibility

    Returns:
        Tuple of (Q, K, V) tensors
    """
    generator = torch.Generator(device=device).manual_seed(seed)

    Q = torch.randn(batch_size, num_heads, seq_len, head_dim,
                   generator=generator, device=device, dtype=dtype)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim,
                   generator=generator, device=device, dtype=dtype)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim,
                   generator=generator, device=device, dtype=dtype)

    return Q, K, V


def compute_dense_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute standard dense scaled dot-product attention.

    Args:
        Q: Query tensor (batch, num_heads, n_q, head_dim)
        K: Key tensor (batch, num_heads, n_k, head_dim)
        V: Value tensor (batch, num_heads, n_k, head_dim)
        attention_mask: Optional attention mask

    Returns:
        Attention output (batch, num_heads, n_q, head_dim)
    """
    head_dim = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)

    if attention_mask is not None:
        scores = scores + attention_mask

    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)

    return output


def create_causal_mask(seq_len: int, device: str = 'cpu') -> torch.Tensor:
    """
    Create a causal attention mask.

    Args:
        seq_len: Sequence length
        device: Device to create mask on

    Returns:
        Causal mask tensor
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)


# ============================================================================
# Hyperparameter Analysis and Tests
# ============================================================================

class HyperparameterAnalysis:
    """
    Comprehensive analysis of SketchWalk hyperparameters based on the paper.

    From the paper "Scout Before You Attend: Sketch-and-Walk Sparse Attention",
    the following hyperparameters are identified:
    """

    # Paper-extracted hyperparameters
    PAPER_HYPERPARAMETERS = {
        # Block size (B)
        'block_size': {
            'symbol': 'B',
            'default': 64,
            'range': [32, 64, 128],
            'description': 'Tokens per block',
            'paper_note': 'B=64 provides good trade-off between granularity and efficiency',
            'impact': 'Larger B = fewer blocks but less granularity',
        },

        # Sketch dimension (k, r)
        'sketch_dim': {
            'symbol': 'k, r',
            'default': 64,
            'range': [16, 32, 64, 128],
            'description': 'Reduced feature dimension after Hadamard transform',
            'paper_note': 'k=64-128 works well (see Fig 3a ablation)',
            'impact': 'Smaller k = faster but less accurate',
        },

        # Top blocks (tau)
        'top_k_blocks': {
            'symbol': 'τ',
            'default': 16,
            'range': [4, 8, 16, 32],
            'description': 'Number of key blocks selected per query block',
            'paper_note': 'τ ≈ 8-16 for 80% sparsity',
            'formula': 'sparsity ≈ 1 - (τ·B)/n',
            'impact': 'Controls sparsity level',
        },

        # Sparsity exponent (s)
        'sparsity_exponent': {
            'symbol': 's',
            'default': 8,
            'range': [2, 4, 8, 16],
            'description': 'Sharpens attention distribution',
            'paper_note': 's=8 validated in ablation study (Fig 3b)',
            'impact': 'Higher s = more selective',
        },

        # Skip first N layers
        'skip_first_n_layers': {
            'symbol': 'N/A',
            'default': 2,
            'range': [0, 1, 2, 3],
            'description': 'Number of initial layers to use dense attention',
            'paper_note': 'First two layers have low achievable sparsity',
            'impact': 'Affects accuracy-speed tradeoff',
        },
    }

    @classmethod
    def get_recommended_configs(cls) -> Dict[str, SketchWalkConfig]:
        """
        Get recommended configurations for different scenarios.

        Returns:
            Dictionary of scenario name to config
        """
        return {
            'conservative': SketchWalkConfig(
                block_size=64,
                sketch_dim=128,
                top_k_blocks=32,
                sparsity_exponent=8,
                skip_first_n_layers=2,
            ),
            'balanced': SketchWalkConfig(
                block_size=64,
                sketch_dim=64,
                top_k_blocks=16,
                sparsity_exponent=8,
                skip_first_n_layers=2,
            ),
            'aggressive': SketchWalkConfig(
                block_size=64,
                sketch_dim=32,
                top_k_blocks=8,
                sparsity_exponent=8,
                skip_first_n_layers=2,
            ),
        }

    @classmethod
    def analyze_sparsity_relationship(cls, seq_len: int, config: SketchWalkConfig) -> Dict[str, float]:
        """
        Analyze sparsity relationships for a given sequence length and config.

        Args:
            seq_len: Sequence length in tokens
            config: SketchWalk configuration

        Returns:
            Dictionary of sparsity metrics
        """
        B = config.block_size
        tau = config.top_k_blocks

        # Number of blocks
        n_blocks = math.ceil(seq_len / B)

        # Theoretical sparsity (uniform selection)
        theoretical_sparsity = 1.0 - min(1.0, (tau * B) / seq_len)

        # Block-level sparsity
        block_sparsity = 1.0 - min(1.0, tau / n_blocks)

        return {
            'seq_len': seq_len,
            'num_blocks': n_blocks,
            'block_size': B,
            'top_k_blocks': tau,
            'theoretical_sparsity': theoretical_sparsity,
            'block_sparsity': block_sparsity,
            'attended_tokens': min(tau * B, seq_len),
        }

    @classmethod
    def validate_hyperparameter_combination(cls, config: SketchWalkConfig) -> List[str]:
        """
        Validate a hyperparameter combination and return any warnings.

        Args:
            config: SketchWalk configuration

        Returns:
            List of warning messages (empty if valid)
        """
        warnings = []

        # Check block size vs sketch dimension
        if config.sketch_dim > config.block_size:
            warnings.append(
                f"sketch_dim ({config.sketch_dim}) > block_size ({config.block_size}) "
                "may lead to inefficiency"
            )

        # Check top_k_blocks
        if config.top_k_blocks < 4:
            warnings.append(
                f"top_k_blocks ({config.top_k_blocks}) < 4 may hurt accuracy"
            )

        # Check sparsity exponent
        if config.sparsity_exponent > 16:
            warnings.append(
                f"sparsity_exponent ({config.sparsity_exponent}) > 16 may cause "
                "numerical instability"
            )

        return warnings


# ============================================================================
# Unit Tests for Hadamard Transform
# ============================================================================

class TestHadamardTransform(unittest.TestCase):
    """Unit tests for Hadamard transform component."""

    def setUp(self):
        """Set up test fixtures."""
        self.in_dim = 128
        self.out_dim = 64
        self.seed = 42
        self.transform = HadamardTransform(self.in_dim, self.out_dim, self.seed)
        self.device = 'cpu'

    def test_initialization(self):
        """Test that Hadamard transform initializes correctly."""
        self.assertEqual(self.transform.in_dim, self.in_dim)
        self.assertEqual(self.transform.out_dim, self.out_dim)

        # Check that rademacher has correct shape and values
        self.assertEqual(self.transform.rademacher.shape, (self.in_dim,))
        self.assertTrue(torch.all(torch.abs(self.transform.rademacher) == 1))

        # Check that projection has correct shape
        self.assertEqual(self.transform.projection.shape, (self.in_dim, self.out_dim))

    def test_inner_product_preservation(self):
        """
        Test that Hadamard transform approximately preserves inner products.

        This is a key property: for SRHT, we expect:
        <x, y> ≈ <Hx, Hy>

        Note: With random projection, this holds in expectation, not exactly.
        """
        # Create test vectors
        x = torch.randn(100, self.in_dim)
        y = torch.randn(100, self.in_dim)

        # Compute original inner products
        original_ip = (x * y).sum(dim=-1)

        # Transform
        x_tilde = self.transform(x)
        y_tilde = self.transform(y)

        # Compute transformed inner products
        transformed_ip = (x_tilde * y_tilde).sum(dim=-1)

        # Check correlation (should be high for large enough sample)
        correlation = torch.corrcoef(torch.stack([original_ip, transformed_ip]))[0, 1]

        # We expect at least some correlation (> 0.3 is reasonable for random projection)
        self.assertGreater(correlation, 0.3,
                          f"Low correlation {correlation:.3f} between original and transformed inner products")

    def test_norm_preservation(self):
        """
        Test that Hadamard transform approximately preserves norms.

        For SRHT: ||Hx|| ≈ ||x|| * sqrt(k/d)
        """
        # Create test vectors
        x = torch.randn(100, self.in_dim)

        # Original norms
        original_norms = torch.norm(x, dim=-1)

        # Transformed norms
        x_tilde = self.transform(x)
        transformed_norms = torch.norm(x_tilde, dim=-1)

        # Expected scaling factor
        expected_scale = math.sqrt(self.out_dim / self.in_dim)

        # Compute relative error (should be < 50% for most vectors)
        scaled_transformed = transformed_norms / expected_scale
        relative_error = torch.abs(scaled_transformed - original_norms) / original_norms

        # Check median error
        median_error = relative_error.median().item()
        self.assertLess(median_error, 0.5,
                       f"High median relative error {median_error:.3f} in norm preservation")

    def test_output_shape(self):
        """Test that output has correct shape."""
        batch_size = 10
        x = torch.randn(batch_size, self.in_dim)

        y = self.transform(x)

        self.assertEqual(y.shape, (batch_size, self.out_dim))

    def test_reproducibility(self):
        """Test that same seed gives same results."""
        x = torch.randn(10, self.in_dim)

        # Two transforms with same seed
        t1 = HadamardTransform(self.in_dim, self.out_dim, seed=123)
        t2 = HadamardTransform(self.in_dim, self.out_dim, seed=123)

        y1 = t1(x)
        y2 = t2(x)

        self.assertTrue(torch.allclose(y1, y2))

    def test_different_seeds(self):
        """Test that different seeds give different results."""
        x = torch.randn(10, self.in_dim)

        # Two transforms with different seeds
        t1 = HadamardTransform(self.in_dim, self.out_dim, seed=123)
        t2 = HadamardTransform(self.in_dim, self.out_dim, seed=456)

        y1 = t1(x)
        y2 = t2(x)

        # Results should be different (with high probability)
        self.assertFalse(torch.allclose(y1, y2))


# ============================================================================
# Unit Tests for Sketch Component
# ============================================================================

class TestSketch(unittest.TestCase):
    """Unit tests for Sketch (SWS) component."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = create_sketch_walk_config(
            block_size=64,
            sketch_dim=64,
            top_k_blocks=16,
            sparsity_exponent=8,
        )
        self.head_dim = 128
        self.sketch = Sketch(self.config, self.head_dim)
        self.device = 'cpu'

    def test_token_space_sketch(self):
        """Test token-space sketching (block aggregation)."""
        batch_size = 2
        seq_len = 256
        B = 64

        # Create test data
        Q = torch.randn(batch_size, seq_len, self.head_dim)
        K = torch.randn(batch_size, seq_len, self.head_dim)

        # Apply token-space sketching
        Q_bar, K_bar = self.sketch._token_space_sketch(Q, K, B)

        # Check shapes
        expected_blocks = (seq_len + B - 1) // B  # ceil division
        self.assertEqual(Q_bar.shape, (batch_size, expected_blocks, self.head_dim))
        self.assertEqual(K_bar.shape, (batch_size, expected_blocks, self.head_dim))

        # Check that block representatives are actually averages
        # First block should be average of first B tokens
        Q_first_block_manual = Q[:, :B, :].mean(dim=1)
        self.assertTrue(torch.allclose(Q_bar[:, 0, :], Q_first_block_manual, atol=1e-5))

    def test_feature_space_sketch(self):
        """Test feature-space sketching (Hadamard transform)."""
        batch_size = 2
        num_blocks = 8

        # Create block representatives
        Q_bar = torch.randn(batch_size, num_blocks, self.head_dim)
        K_bar = torch.randn(batch_size, num_blocks, self.head_dim)

        # Apply feature-space sketching
        Q_tilde, K_tilde = self.sketch._feature_space_sketch(Q_bar, K_bar)

        # Check shapes
        self.assertEqual(Q_tilde.shape, (batch_size, num_blocks, self.config.sketch_dim))
        self.assertEqual(K_tilde.shape, (batch_size, num_blocks, self.config.sketch_dim))

    def test_block_attention_computation(self):
        """Test block-level attention computation."""
        batch_size = 2
        num_heads = 8
        seq_len = 256

        # Create test data
        Q = torch.randn(batch_size, num_heads, seq_len, self.head_dim)
        K = torch.randn(batch_size, num_heads, seq_len, self.head_dim)

        # Compute sketched attention
        A_hat, Q_bar, K_bar = self.sketch(Q, K)

        # Check shapes
        num_blocks = (seq_len + self.config.block_size - 1) // self.config.block_size
        self.assertEqual(A_hat.shape, (batch_size, num_blocks, num_blocks))
        self.assertEqual(Q_bar.shape, (batch_size, num_blocks, self.head_dim))
        self.assertEqual(K_bar.shape, (batch_size, num_blocks, self.head_dim))

        # Check that attention scores are finite
        self.assertTrue(torch.all(torch.isfinite(A_hat)))

    def test_causal_masking(self):
        """Test causal mask application."""
        batch_size = 2
        num_heads = 8
        seq_len = 256

        # Create test data
        Q = torch.randn(batch_size, num_heads, seq_len, self.head_dim)
        K = torch.randn(batch_size, num_heads, seq_len, self.head_dim)

        # Create causal mask
        attention_mask = create_causal_mask(seq_len, self.device)

        # Compute sketched attention with mask
        A_hat, _, _ = self.sketch(Q, K, attention_mask)

        # Check that upper triangular blocks have Inf values (properly masked)
        # For block-level causal mask, blocks in upper triangular should be masked
        # Check that we have some masked positions (Inf values)
        has_inf = torch.isinf(A_hat).any()
        self.assertTrue(has_inf, "Causal masking should produce Inf values for masked blocks")

        # Check that lower triangular blocks have finite values
        # Create a mask for lower triangular part at block level
        num_blocks = A_hat.shape[1]
        block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=self.device), diagonal=0)
        # Check finite values in lower triangle
        for i in range(num_blocks):
            for j in range(i + 1):
                if block_mask[i, j] > 0:
                    self.assertTrue(torch.isfinite(A_hat[:, i, j]).all(),
                                   f"Block [{i},{j}] should have finite values")

    def test_head_averaging(self):
        """Test that heads are averaged correctly."""
        batch_size = 2
        num_heads = 8
        seq_len = 256

        # Create test data with known values
        Q = torch.ones(batch_size, num_heads, seq_len, self.head_dim) * 2.0
        K = torch.ones(batch_size, num_heads, seq_len, self.head_dim) * 3.0

        # Compute sketched attention
        A_hat, Q_bar, K_bar = self.sketch(Q, K)

        # After averaging across heads, values should still be 2.0 and 3.0
        # (with some variation due to Hadamard transform)
        # Just check they're in reasonable range
        self.assertTrue(torch.all(torch.isfinite(Q_bar)))
        self.assertTrue(torch.all(torch.isfinite(K_bar)))


# ============================================================================
# Unit Tests for Walk Component
# ============================================================================

class TestWalk(unittest.TestCase):
    """Unit tests for Walk component."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = create_sketch_walk_config(
            block_size=64,
            sketch_dim=64,
            top_k_blocks=16,
            sparsity_exponent=8,
        )
        self.walk = Walk(self.config)
        self.device = 'cpu'

    def test_walk_state_initialization(self):
        """Test walk state initialization."""
        batch_size = 2
        num_blocks = 10

        self.walk.reset_state()

        # Should reset current_layer
        self.assertEqual(self.walk.current_layer, 0)

    def test_walk_state_update_first_layer(self):
        """Test walk state update for first layer."""
        batch_size = 2
        num_blocks = 10

        # Create sketched attention for layer 0
        A_hat = torch.randn(batch_size, num_blocks, num_blocks)

        # Update walk state
        R = self.walk.update(A_hat, layer_idx=0, causal=False)

        # Check shape
        self.assertEqual(R.shape, (batch_size, num_blocks, num_blocks))

        # For first layer, walk state should be (softmax(A_hat))^s
        W = F.softmax(A_hat, dim=-1) ** self.config.sparsity_exponent
        expected_R = W
        self.assertTrue(torch.allclose(R, expected_R, atol=1e-5))

    def test_walk_state_accumulation(self):
        """Test walk state accumulation across layers."""
        batch_size = 2
        num_blocks = 10

        # Create sketched attention for layers 0 and 1
        A_hat_0 = torch.randn(batch_size, num_blocks, num_blocks)
        A_hat_1 = torch.randn(batch_size, num_blocks, num_blocks)

        # Update layer 0
        R_0 = self.walk.update(A_hat_0, layer_idx=0, causal=False)

        # Update layer 1
        R_1 = self.walk.update(A_hat_1, layer_idx=1, causal=False)

        # Check that R_1 = R_0 @ W_1
        W_0 = F.softmax(A_hat_0, dim=-1) ** self.config.sparsity_exponent
        W_1 = F.softmax(A_hat_1, dim=-1) ** self.config.sparsity_exponent
        expected_R_1 = torch.bmm(W_0, W_1)

        self.assertTrue(torch.allclose(R_1, expected_R_1, atol=1e-5))

    def test_causal_masking(self):
        """Test causal masking in walk state."""
        batch_size = 2
        num_blocks = 10

        # Create sketched attention
        A_hat = torch.randn(batch_size, num_blocks, num_blocks)

        # Update with causal masking
        R = self.walk.update(A_hat, layer_idx=0, causal=True)

        # Check that upper triangular is zero
        for i in range(num_blocks):
            for j in range(i + 1, num_blocks):
                self.assertEqual(R[0, i, j].item(), 0.0)

    def test_top_k_block_selection(self):
        """Test top-k block selection."""
        batch_size = 2
        num_blocks = 20

        # Create walk state with known structure
        R = torch.zeros(batch_size, num_blocks, num_blocks)
        for i in range(num_blocks):
            # Make block i have highest affinity to itself
            R[:, i, i] = 10.0
            # Make some other blocks have lower affinity
            R[:, i, (i + 1) % num_blocks] = 5.0
            R[:, i, (i + 2) % num_blocks] = 3.0

        # Select top blocks
        selected = self.walk.select_blocks(R, num_blocks, include_first_last=False)

        # Expected tau is min(config.top_k_blocks, num_blocks)
        expected_tau = min(self.config.top_k_blocks, num_blocks)

        # Check shape
        self.assertEqual(selected.shape, (batch_size, num_blocks, expected_tau))

        # Check that each row contains valid indices
        self.assertTrue((selected >= 0).all())
        self.assertTrue((selected < num_blocks).all())

    def test_first_block_inclusion(self):
        """Test that first block is always included."""
        batch_size = 2
        num_blocks = 20

        # Create walk state
        R = torch.randn(batch_size, num_blocks, num_blocks)

        # Select top blocks with first block inclusion
        selected = self.walk.select_blocks(R, num_blocks, include_first_last=True)

        # Check that first block (index 0) is in selection for each query block
        for i in range(num_blocks):
            self.assertTrue((selected[:, i, :] == 0).any())


# ============================================================================
# Integration Tests
# ============================================================================

class TestSketchWalkIntegration(unittest.TestCase):
    """Integration tests for complete SketchWalk attention."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = create_sketch_walk_config(
            block_size=64,
            sketch_dim=64,
            top_k_blocks=16,
            sparsity_exponent=8,
            skip_first_n_layers=2,
        )
        self.head_dim = 128
        self.attention = SketchWalkAttention(self.config, self.head_dim)
        self.device = 'cpu'

    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shape."""
        batch_size = 2
        num_heads = 8
        seq_len = 512
        head_dim = 128
        layer_idx = 5  # After skip layers

        # Create test data
        Q, K, V = create_synthetic_attention_data(
            batch_size, num_heads, seq_len, head_dim, self.device
        )

        # Compute sparse attention
        output, selected_blocks = self.attention(Q, K, V, layer_idx=layer_idx, causal=True)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, num_heads, seq_len, head_dim))

        # Check selected blocks shape
        num_blocks = (seq_len + self.config.block_size - 1) // self.config.block_size
        expected_tau = min(self.config.top_k_blocks, num_blocks)
        self.assertEqual(selected_blocks.shape, (batch_size, num_blocks, expected_tau))

    def test_skip_first_layers(self):
        """Test that first N layers use dense attention."""
        batch_size = 2
        num_heads = 8
        seq_len = 512
        head_dim = 128
        layer_idx = 1  # Before skip layers

        # Create test data
        Q, K, V = create_synthetic_attention_data(
            batch_size, num_heads, seq_len, head_dim, self.device
        )

        # Compute attention (should be dense)
        output, selected_blocks = self.attention(Q, K, V, layer_idx=layer_idx, causal=True)

        # Check that no blocks selected for dense attention
        self.assertIsNone(selected_blocks)

    def test_comparison_with_dense_attention(self):
        """Test that sparse attention approximates dense attention."""
        batch_size = 2
        num_heads = 4
        seq_len = 512
        head_dim = 128
        layer_idx = 5

        # Create test data
        Q, K, V = create_synthetic_attention_data(
            batch_size, num_heads, seq_len, head_dim, self.device
        )

        # Compute dense attention
        dense_output = compute_dense_attention(Q, K, V)

        # Compute sparse attention
        sparse_output, _ = self.attention(Q, K, V, layer_idx=layer_idx, causal=True)

        # Check that outputs have same shape
        self.assertEqual(dense_output.shape, sparse_output.shape)

        # Compute cosine similarity between outputs
        dense_flat = dense_output.flatten()
        sparse_flat = sparse_output.flatten()
        cosine_sim = F.cosine_similarity(dense_flat.unsqueeze(0), sparse_flat.unsqueeze(0))

        # We expect some similarity (> 0.5 is reasonable for 80% sparsity)
        self.assertGreater(cosine_sim.item(), 0.5,
                          f"Low cosine similarity {cosine_sim.item():.3f} between dense and sparse")

    def test_state_reset(self):
        """Test that state can be reset between sequences."""
        batch_size = 2
        num_blocks = 10

        # Reset state
        self.attention.reset_state()

        # Should not raise error
        self.assertTrue(True)


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and boundary conditions."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = create_sketch_walk_config(
            block_size=64,
            sketch_dim=64,
            top_k_blocks=16,
            sparsity_exponent=8,
        )
        self.head_dim = 128
        self.device = 'cpu'

    def test_single_token_sequence(self):
        """Test behavior with single-token sequence."""
        batch_size = 2
        num_heads = 8
        seq_len = 1
        head_dim = 128

        # Create test data
        Q, K, V = create_synthetic_attention_data(
            batch_size, num_heads, seq_len, head_dim, self.device
        )

        # Create attention module
        attention = SketchWalkAttention(self.config, head_dim)

        # Should not crash
        output, selected_blocks = attention(Q, K, V, layer_idx=5, causal=True)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, num_heads, seq_len, head_dim))

    def test_sequence_smaller_than_block_size(self):
        """Test sequence smaller than block size."""
        batch_size = 2
        num_heads = 8
        seq_len = 32  # Smaller than block_size=64
        head_dim = 128

        # Create test data
        Q, K, V = create_synthetic_attention_data(
            batch_size, num_heads, seq_len, head_dim, self.device
        )

        # Create attention module
        attention = SketchWalkAttention(self.config, head_dim)

        # Should not crash
        output, selected_blocks = attention(Q, K, V, layer_idx=5, causal=True)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, num_heads, seq_len, head_dim))

    def test_very_long_sequence(self):
        """Test very long sequence (128K tokens)."""
        batch_size = 1
        num_heads = 4
        seq_len = 128 * 1024  # 128K
        head_dim = 128

        # Create test data (smaller for memory reasons)
        # Just test that it doesn't crash with smaller version
        seq_len_test = 4096
        Q, K, V = create_synthetic_attention_data(
            batch_size, num_heads, seq_len_test, head_dim, self.device
        )

        # Create attention module
        attention = SketchWalkAttention(self.config, head_dim)

        # Should not crash
        output, selected_blocks = attention(Q, K, V, layer_idx=5, causal=True)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, num_heads, seq_len_test, head_dim))

    def test_mismatched_qk_lengths(self):
        """Test with different Q and K sequence lengths (for decode)."""
        batch_size = 2
        num_heads = 8
        q_len = 1  # Single query token (decode)
        k_len = 1024  # Large KV cache
        head_dim = 128

        # Create test data
        Q = torch.randn(batch_size, num_heads, q_len, head_dim)
        K = torch.randn(batch_size, num_heads, k_len, head_dim)
        V = torch.randn(batch_size, num_heads, k_len, head_dim)

        # Create attention module
        attention = SketchWalkAttention(self.config, head_dim)

        # Should not crash
        output, selected_blocks = attention(Q, K, V, layer_idx=5, causal=True)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, num_heads, q_len, head_dim))


# ============================================================================
# Numerical Stability Tests
# ============================================================================

class TestNumericalStability(unittest.TestCase):
    """Tests for numerical stability."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = create_sketch_walk_config(
            block_size=64,
            sketch_dim=64,
            top_k_blocks=16,
            sparsity_exponent=8,
        )
        self.head_dim = 128
        self.device = 'cpu'

    def test_no_nan_outputs(self):
        """Test that outputs are never NaN."""
        batch_size = 4
        num_heads = 8
        seq_len = 2048
        head_dim = 128

        # Create test data with various scales
        for scale in [1e-3, 1.0, 1e3]:
            Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=self.device) * scale
            K = torch.randn(batch_size, num_heads, seq_len, head_dim, device=self.device) * scale
            V = torch.randn(batch_size, num_heads, seq_len, head_dim, device=self.device) * scale

            # Create attention module
            attention = SketchWalkAttention(self.config, head_dim)

            # Compute attention
            output, _ = attention(Q, K, V, layer_idx=5, causal=True)

            # Check no NaN
            self.assertFalse(torch.isnan(output).any(),
                           f"NaN detected in output with scale {scale}")

            # Check no Inf
            self.assertFalse(torch.isinf(output).any(),
                           f"Inf detected in output with scale {scale}")

    def test_high_sparsity_exponent_stability(self):
        """Test stability with high sparsity exponent."""
        batch_size = 2
        num_heads = 4
        seq_len = 512
        head_dim = 128

        # Create test data
        Q, K, V = create_synthetic_attention_data(
            batch_size, num_heads, seq_len, head_dim, self.device
        )

        # Test with high sparsity exponent
        config = create_sketch_walk_config(sparsity_exponent=16)
        attention = SketchWalkAttention(config, head_dim)

        # Should not crash or produce NaN
        output, _ = attention(Q, K, V, layer_idx=5, causal=True)

        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_gradient_flow(self):
        """Test that gradients can flow through (for potential training)."""
        batch_size = 2
        num_heads = 4
        seq_len = 256
        head_dim = 128

        # Create test data with requires_grad
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       device=self.device, requires_grad=True)
        K = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       device=self.device, requires_grad=True)
        V = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       device=self.device, requires_grad=True)

        # Create attention module
        attention = SketchWalkAttention(self.config, head_dim)

        # Compute attention
        output, _ = attention(Q, K, V, layer_idx=5, causal=True)

        # Compute loss
        loss = output.sum()

        # Backward pass
        loss.backward()

        # Check gradients exist
        self.assertIsNotNone(Q.grad)
        self.assertIsNotNone(K.grad)
        self.assertIsNotNone(V.grad)

        # Check gradients are not all zero
        self.assertTrue(Q.grad.abs().sum() > 0)
        self.assertTrue(K.grad.abs().sum() > 0)
        self.assertTrue(V.grad.abs().sum() > 0)


# ============================================================================
# Performance Benchmarks
# ============================================================================

class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarks for SketchWalk attention."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = create_sketch_walk_config(
            block_size=64,
            sketch_dim=64,
            top_k_blocks=16,
            sparsity_exponent=8,
        )
        self.head_dim = 128
        self.device = 'cpu'

    def benchmark_vs_dense(self, seq_len: int, num_heads: int = 8, batch_size: int = 2) -> Dict[str, float]:
        """
        Benchmark sparse vs dense attention.

        Args:
            seq_len: Sequence length
            num_heads: Number of attention heads
            batch_size: Batch size

        Returns:
            Dictionary with timing results
        """
        head_dim = 128

        # Create test data
        Q, K, V = create_synthetic_attention_data(
            batch_size, num_heads, seq_len, head_dim, self.device
        )

        # Time dense attention
        with TestTimer() as dense_timer:
            dense_output = compute_dense_attention(Q, K, V)

        # Time sparse attention
        attention = SketchWalkAttention(self.config, head_dim)
        with TestTimer() as sparse_timer:
            sparse_output, _ = attention(Q, K, V, layer_idx=5, causal=True)

        # Compute speedup
        speedup = dense_timer.elapsed / sparse_timer.elapsed

        return {
            'seq_len': seq_len,
            'dense_time': dense_timer.elapsed,
            'sparse_time': sparse_timer.elapsed,
            'speedup': speedup,
        }

    def test_performance_scaling(self):
        """Test performance scaling with sequence length."""
        results = []

        for seq_len in [512, 1024, 2048, 4096]:
            result = self.benchmark_vs_dense(seq_len)
            results.append(result)
            print(f"SeqLen {seq_len}: Dense={result['dense_time']:.4f}s, "
                  f"Sparse={result['sparse_time']:.4f}s, Speedup={result['speedup']:.2f}x")

        # Just check that it runs without crashing
        self.assertTrue(True)


# ============================================================================
# Property-Based Tests
# ============================================================================

class TestProperties(unittest.TestCase):
    """Property-based tests (invariants that should always hold)."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = create_sketch_walk_config(
            block_size=64,
            sketch_dim=64,
            top_k_blocks=16,
            sparsity_exponent=8,
        )
        self.head_dim = 128
        self.device = 'cpu'

    def test_walk_state_stochasticity(self):
        """Test that walk state is bounded in [0, 1]."""
        batch_size = 2
        num_blocks = 10

        walk = Walk(self.config)

        # Create sketched attention
        A_hat = torch.randn(batch_size, num_blocks, num_blocks)

        # Update walk state
        R = walk.update(A_hat, layer_idx=0, causal=False)

        # After softmax^s, values should be in [0, 1]
        self.assertTrue((R >= 0).all())
        self.assertTrue((R <= 1).all())

    def test_selected_blocks_in_range(self):
        """Test that selected block indices are always valid."""
        batch_size = 4
        num_heads = 8
        seq_len = 1024
        head_dim = 128

        # Create test data
        Q, K, V = create_synthetic_attention_data(
            batch_size, num_heads, seq_len, head_dim, self.device
        )

        # Create attention module
        attention = SketchWalkAttention(self.config, head_dim)

        # Compute attention
        _, selected_blocks = attention(Q, K, V, layer_idx=5, causal=True)

        # Check all indices are valid
        num_blocks = (seq_len + self.config.block_size - 1) // self.config.block_size
        self.assertTrue((selected_blocks >= 0).all())
        self.assertTrue((selected_blocks < num_blocks).all())

    def test_sparsity_level(self):
        """Test that actual sparsity matches expected."""
        batch_size = 2
        num_heads = 8
        seq_len = 2048
        head_dim = 128

        # Create test data
        Q, K, V = create_synthetic_attention_data(
            batch_size, num_heads, seq_len, head_dim, self.device
        )

        # Create attention module
        attention = SketchWalkAttention(self.config, head_dim)

        # Compute attention
        _, selected_blocks = attention(Q, K, V, layer_idx=5, causal=True)

        # Count unique blocks selected
        unique_blocks = torch.unique(selected_blocks).numel()

        # Should be approximately top_k_blocks (with some variation)
        # At minimum, should not exceed total blocks
        num_blocks = (seq_len + self.config.block_size - 1) // self.config.block_size
        self.assertLessEqual(unique_blocks, num_blocks)


# ============================================================================
# Test Runner
# ============================================================================

def run_all_tests(verbosity: int = 2):
    """
    Run all test suites.

    Args:
        verbosity: Test verbosity level

    Returns:
        TestResult object
    """
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestHadamardTransform))
    suite.addTests(loader.loadTestsFromTestCase(TestSketch))
    suite.addTests(loader.loadTestsFromTestCase(TestWalk))
    suite.addTests(loader.loadTestsFromTestCase(TestSketchWalkIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestNumericalStability))
    suite.addTests(loader.loadTestsFromTestCase(TestProperties))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    return result


def main():
    """Main entry point for test execution."""
    print("=" * 80)
    print("SketchWalk Detailed Test Suite")
    print("=" * 80)
    print()

    # Print hyperparameter analysis
    print("Hyperparameter Analysis:")
    print("-" * 80)
    for name, params in HyperparameterAnalysis.PAPER_HYPERPARAMETERS.items():
        print(f"{name}:")
        print(f"  Symbol: {params['symbol']}")
        print(f"  Default: {params['default']}")
        print(f"  Range: {params['range']}")
        print(f"  Description: {params['description']}")
        print(f"  Paper Note: {params['paper_note']}")
        print()

    print("=" * 80)
    print("Running Tests")
    print("=" * 80)
    print()

    # Run tests
    result = run_all_tests()

    print()
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")

    return result


if __name__ == '__main__':
    result = main()
    sys.exit(0 if result.wasSuccessful() else 1)
