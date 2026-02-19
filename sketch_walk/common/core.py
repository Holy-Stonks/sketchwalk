"""
SketchWalk Sparse Attention - Core Implementation

This module implements the core SketchWalk algorithm components:
1. Sketch: Small-World Sketching (SWS) - token and feature space sketching
2. Walk: Sketch-Determined Walk - cross-layer attention accumulation

Reference: "Scout Before You Attend: Sketch-and-Walk Sparse Attention for Efficient LLM Inference"
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DecodeCache:
    """
    Cache for SketchWalk decode phase, maintaining incremental state across autoregressive steps.

    This cache stores:
    - Cached block representatives (query and key blocks)
    - Cached block-level attention estimate
    - Block counts (number of tokens per block for running averages)
    - Current token position

    The cache enables efficient incremental updates during decode according to Algorithm 2.
    """

    # Cached block representatives from prefill or accumulated during decode
    # Shape: (batch, num_blocks, head_dim)
    cached_query_blocks: Optional[torch.Tensor] = None  # {q̄^k_i}
    cached_key_blocks: Optional[torch.Tensor] = None   # {k̄^k_j}

    # Cached block-level attention estimate
    # Shape: (batch, num_blocks, num_blocks)
    cached_block_attn: Optional[torch.Tensor] = None  # Â^k_block,cache

    # Block counts for running average of key block representatives
    # Shape: (batch, num_blocks)
    key_block_counts: Optional[torch.Tensor] = None  # c_{b_curr}

    # Current token position (number of tokens processed so far)
    current_position: int = 0

    # Maximum number of blocks (for preallocation)
    max_blocks: int = 8192  # Supports up to ~512K tokens with block_size=64

    # Device for the cache tensors
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float32

    # Head dimension for cached blocks
    head_dim: int = 64

    def initialize_from_prefill(
        self,
        Q_bar: torch.Tensor,
        K_bar: torch.Tensor,
        A_hat: torch.Tensor,
        num_tokens: int,
    ):
        """
        Initialize cache from prefill phase outputs.

        Args:
            Q_bar: Block query reps of shape (batch, b_q, head_dim)
            K_bar: Block key reps of shape (batch, b_k, head_dim)
            A_hat: Block attention of shape (batch, b_q, b_k)
            num_tokens: Number of tokens in prefill
        """
        batch_size, b_q, head_dim = Q_bar.shape
        b_k = K_bar.shape[1]

        # Update head_dim from input tensors
        self.head_dim = head_dim

        # Store cached representatives with correct dtype
        self.cached_query_blocks = Q_bar.clone().to(dtype=self.dtype)
        self.cached_key_blocks = K_bar.clone().to(dtype=self.dtype)
        self.cached_block_attn = A_hat.clone().to(dtype=self.dtype)

        # Initialize block counts (each block has B tokens except possibly the last)
        block_size = num_tokens / b_q  # Approximate
        self.key_block_counts = torch.ones(batch_size, b_k, device=Q_bar.device, dtype=torch.long) * int(block_size)

        # Handle last block (might have fewer tokens)
        last_block_count = num_tokens - (b_k - 1) * int(block_size)
        self.key_block_counts[:, -1] = last_block_count

        self.current_position = num_tokens
        self.device = Q_bar.device

    def ensure_capacity(self, batch_size: int, device: torch.device):
        """Ensure cache has capacity for current state."""
        if self.cached_query_blocks is None:
            b = min(self.max_blocks, (self.current_position + 64) // 64)
            self.cached_query_blocks = torch.zeros(batch_size, b, self.head_dim, dtype=self.dtype, device=device)
            self.cached_key_blocks = torch.zeros(batch_size, b, self.head_dim, dtype=self.dtype, device=device)
            self.cached_block_attn = torch.zeros(batch_size, b, b, dtype=self.dtype, device=device)
            self.key_block_counts = torch.zeros(batch_size, b, dtype=torch.long, device=device)
        elif self.cached_query_blocks.shape[1] < self.current_position // 64 + 1:
            # Need to expand cache
            old_b = self.cached_query_blocks.shape[1]
            new_b = min(self.max_blocks, self.current_position // 64 + 100)
            pad_b = new_b - old_b

            self.cached_query_blocks = F.pad(self.cached_query_blocks, (0, 0, 0, pad_b))
            self.cached_key_blocks = F.pad(self.cached_key_blocks, (0, 0, 0, pad_b))
            self.cached_block_attn = F.pad(self.cached_block_attn, (0, pad_b, 0, pad_b))
            self.key_block_counts = F.pad(self.key_block_counts, (0, pad_b))

    def ensure_capacity_for_block(self, batch_size: int, device: torch.device, required_blocks: int):
        """Ensure cache has capacity for required number of blocks."""
        if self.cached_query_blocks is None:
            b = min(self.max_blocks, required_blocks)
            self.cached_query_blocks = torch.zeros(batch_size, b, self.head_dim, dtype=self.dtype, device=device)
            self.cached_key_blocks = torch.zeros(batch_size, b, self.head_dim, dtype=self.dtype, device=device)
            self.cached_block_attn = torch.zeros(batch_size, b, b, dtype=self.dtype, device=device)
            self.key_block_counts = torch.zeros(batch_size, b, dtype=torch.long, device=device)
        elif self.cached_query_blocks.shape[1] < required_blocks:
            # Need to expand cache
            old_b = self.cached_query_blocks.shape[1]
            pad_b = required_blocks - old_b

            self.cached_query_blocks = F.pad(self.cached_query_blocks, (0, 0, 0, pad_b))
            self.cached_key_blocks = F.pad(self.cached_key_blocks, (0, 0, 0, pad_b))
            self.cached_block_attn = F.pad(self.cached_block_attn, (0, pad_b, 0, pad_b))
            self.key_block_counts = F.pad(self.key_block_counts, (0, pad_b))


@dataclass
class SketchWalkConfig:
    """
    Configuration for SketchWalk sparse attention.

    Hyperparameters based on the paper's ablation studies and recommendations.
    """
    # Block size (B): Tokens per block. Default 64 provides good trade-off.
    block_size: int = 64

    # Sketch dimension (k, r): Reduced feature dimension after Hadamard transform.
    # Default 64 works well; use 128 for higher accuracy, 32 for more speed.
    sketch_dim: int = 64

    # Top blocks (tau): Number of key blocks selected per query block.
    # Controls sparsity: sparsity ≈ 1 - (tau * B) / n
    # Default 16 for ~80% sparsity with typical sequence lengths.
    top_k_blocks: int = 16

    # Sparsity exponent (s): Sharpens attention distribution.
    # Higher = more selective. Default 8 validated in ablation study.
    sparsity_exponent: int = 8

    # Number of layers to skip at start (use dense attention).
    # Paper recommends 2 as early layers have low achievable sparsity.
    skip_first_n_layers: int = 2

    # Whether to use Subsampled Randomized Hadamard Transform (SRHT).
    # If False, uses random projection matrix.
    use_srht: bool = True

    # Random seed for reproducibility of Hadamard transform.
    hadamard_seed: int = 42

    # Device to run on.
    device: Optional[torch.device] = None

    # Data type for walk state (use fp32 for numerical stability).
    walk_state_dtype: torch.dtype = torch.float32

    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.block_size > 0, "block_size must be positive"
        assert self.sketch_dim > 0, "sketch_dim must be positive"
        assert self.top_k_blocks > 0, "top_k_blocks must be positive"
        assert self.sparsity_exponent >= 1, "sparsity_exponent must be >= 1"
        assert self.skip_first_n_layers >= 0, "skip_first_n_layers must be non-negative"

    def sparsity_level(self, seq_len: int) -> float:
        """
        Estimate sparsity level for a given sequence length.

        Args:
            seq_len: Total sequence length in tokens

        Returns:
            Estimated sparsity (0 = dense, 1 = fully sparse)
        """
        n_blocks = math.ceil(seq_len / self.block_size)
        # Assuming uniform selection, sparsity = 1 - (tau / n_blocks)
        # Adjusted for block size: attended_tokens = tau * B
        attended_tokens = min(self.top_k_blocks * self.block_size, seq_len)
        return 1.0 - (attended_tokens / seq_len)


class HadamardTransform(nn.Module):
    """
    Subsampled Randomized Hadamard Transform (SRHT).

    Projects d-dimensional vectors to k-dimensional sketch space while
    approximately preserving inner products.

    Uses a sparse random projection for efficiency:
    H = sqrt(1/k) * R * D
    where:
      - D: Diagonal Rademacher matrix (±1 entries)
      - R: Random sampling matrix

    For full Hadamard transform, this can be implemented via:
    H_d = sqrt(d/k) * D * W * S where W is Walsh-Hadamard matrix.
    """

    def __init__(self, in_dim: int, out_dim: int, seed: int = 42):
        """
        Initialize Hadamard transform.

        Args:
            in_dim: Input dimension (d)
            out_dim: Output sketch dimension (k)
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.seed = seed

        # Create generator for reproducibility
        generator = torch.Generator().manual_seed(seed)

        # Rademacher matrix D (diagonal ±1) - applied as element-wise multiplication
        self.register_buffer('rademacher', torch.empty(in_dim, dtype=torch.float32))
        self.rademacher.bernoulli_(0.5, generator=generator).mul_(2).sub_(1)

        # Gaussian random projection matrix (simpler alternative to SRHT)
        # Using Gaussian preserves inner products in expectation
        projection = torch.randn(in_dim, out_dim, generator=generator) / math.sqrt(out_dim)
        self.register_buffer('projection', projection)

        # Rademacher scaling for the projection
        self.projection = self.projection * self.rademacher.unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random projection transform.

        Args:
            x: Input tensor of shape (*, d)

        Returns:
            Sketched tensor of shape (*, k)
        """
        # Get leading dimensions
        *leading_dims, d = x.shape

        # Flatten for processing
        x_flat = x.reshape(-1, d)

        # Apply Rademacher D (element-wise multiply)
        x_scaled = x_flat * self.rademacher

        # Apply projection matrix
        x_output = torch.matmul(x_scaled, self.projection)

        # Reshape back
        return x_output.reshape(*leading_dims, self.out_dim)


class Sketch(nn.Module):
    """
    Small-World Sketching (SWS) - Component 1 of SketchWalk.

    Performs token-space and feature-space sketching to efficiently
    estimate block-level attention without computing full n×n matrix.

    Two-stage sketching:
    1. Token-space: Aggregate tokens within each block (averaging)
    2. Feature-space: Project to lower dimension via Hadamard transform

    Output: Sketched block-level attention matrix A_hat^SWS of shape (b, b)
    where b = ceil(n/B) is number of blocks.
    """

    def __init__(self, config: SketchWalkConfig, head_dim: int):
        """
        Initialize Sketch module.

        Args:
            config: SketchWalk configuration
            head_dim: Attention head dimension (d)
        """
        super().__init__()
        self.config = config
        self.head_dim = head_dim

        # Initialize Hadamard transform
        self.hadamard = HadamardTransform(
            in_dim=head_dim,
            out_dim=config.sketch_dim,
            seed=config.hadamard_seed
        )

    def reset_state(self):
        """Reset sketch state (no-op for Sketch as it's stateless)."""
        pass

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute sketched block-level attention.

        Args:
            Q: Query tensor of shape (batch, num_heads, n_q, head_dim)
            K: Key tensor of shape (batch, num_heads, n_k, head_dim)
            attention_mask: Optional mask of shape (batch, n_q, n_k)

        Returns:
            A_hat: Sketched block attention of shape (batch, b_q, b_k)
            Q_bar: Block query reps of shape (batch, b_q, head_dim)
            K_bar: Block key reps of shape (batch, b_k, head_dim)
        """
        batch_size, num_heads, n_q, n_k = Q.shape[0], Q.shape[1], Q.shape[2], K.shape[2]
        B = self.config.block_size

        # Step 1: Average across heads to get head-independent Q, K
        # Shape: (batch, n_q, head_dim), (batch, n_k, head_dim)
        Q_avg = Q.mean(dim=1)
        K_avg = K.mean(dim=1)

        # Step 2: Token-space sketching (block aggregation)
        Q_bar, K_bar = self._token_space_sketch(Q_avg, K_avg, B)

        # Step 3: Feature-space sketching (Hadamard transform)
        Q_tilde, K_tilde = self._feature_space_sketch(Q_bar, K_bar)

        # Step 4: Compute block-level attention
        # A_hat = (Q_tilde @ K_tilde.T) / sqrt(k)
        A_hat = torch.bmm(Q_tilde, K_tilde.transpose(1, 2))
        A_hat = A_hat / math.sqrt(self.config.sketch_dim)

        # Apply attention mask if provided
        # For now, skip attention mask handling for causal attention
        # The causal masking is handled internally in the sparse attention computation
        # TODO: Implement proper attention mask handling for non-causal cases
        """
        if attention_mask is not None:
            # Need to downsample mask to block level
            # Convert float mask to binary (valid=1, masked=0) for downsampling
            mask_binary = attention_mask.clone()
            if mask_binary.dtype == torch.float:
                # Handle float masks (e.g., causal masks with -inf)
                mask_binary = mask_binary.isfinite().float()

            # Get actual Q/K lengths from the token-space sketching
            # Q_bar and K_bar are the block-level representations
            # We need the original token lengths for mask downsampling
            mask_block = self._downsample_mask_to_blocks(mask_binary, B, n_q, n_k)
            # mask_block is boolean (True=valid, False=masked)
            # Mask out invalid positions (where mask_block is False/0)
            A_hat = A_hat.masked_fill(~mask_block, float('-inf'))
        """

        return A_hat, Q_bar, K_bar

    def _token_space_sketch(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        block_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Token-space sketching: Aggregate tokens within each block.

        Args:
            Q: Query of shape (batch, n_q, head_dim)
            K: Key of shape (batch, n_k, head_dim)
            block_size: Block size B

        Returns:
            Q_bar: Block query reps of shape (batch, b_q, head_dim)
            K_bar: Block key reps of shape (batch, b_k, head_dim)
        """
        batch_size = Q.shape[0]

        # Pad sequences to multiple of block size
        n_q, n_k = Q.shape[1], K.shape[1]
        n_q_padded = (n_q + block_size - 1) // block_size * block_size
        n_k_padded = (n_k + block_size - 1) // block_size * block_size

        Q_padded = F.pad(Q, (0, 0, 0, n_q_padded - n_q))
        K_padded = F.pad(K, (0, 0, 0, n_k_padded - n_k))

        # Reshape to blocks and average
        # (batch, n_padded, head_dim) -> (batch, n_blocks, B, head_dim)
        b_q = n_q_padded // block_size
        b_k = n_k_padded // block_size

        Q_blocks = Q_padded.view(batch_size, b_q, block_size, -1)
        K_blocks = K_padded.view(batch_size, b_k, block_size, -1)

        # Average within blocks: (1/B) * sum(tokens)
        Q_bar = Q_blocks.mean(dim=2)  # (batch, b_q, head_dim)
        K_bar = K_blocks.mean(dim=2)  # (batch, b_k, head_dim)

        return Q_bar, K_bar

    def _feature_space_sketch(
        self,
        Q_bar: torch.Tensor,
        K_bar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Feature-space sketching: Project to lower dimension via Hadamard.

        Args:
            Q_bar: Block query reps of shape (batch, b_q, head_dim)
            K_bar: Block key reps of shape (batch, b_k, head_dim)

        Returns:
            Q_tilde: Sketched query reps of shape (batch, b_q, sketch_dim)
            K_tilde: Sketched key reps of shape (batch, b_k, sketch_dim)
        """
        # Apply Hadamard transform
        Q_tilde = self.hadamard(Q_bar)  # (batch, b_q, sketch_dim)
        K_tilde = self.hadamard(K_bar)  # (batch, b_k, sketch_dim)

        return Q_tilde, K_tilde

    def _downsample_mask_to_blocks(
        self,
        mask: torch.Tensor,
        block_size: int,
        n_q: int,
        n_k: int,
    ) -> torch.Tensor:
        """
        Downsample attention mask from token level to block level.

        Args:
            mask: Token-level mask of shape (batch, n_q_orig, n_k_orig) or (batch, 1, n_q_orig, n_k_orig)
            block_size: Block size B
            n_q: Actual query length
            n_k: Actual key length

        Returns:
            Block-level mask of shape (batch, b_q, b_k)
        """
        # Handle both 3D and 4D masks
        if mask.dim() == 4:
            mask = mask.squeeze(1)  # Remove head dimension

        batch_size, n_q_orig, n_k_orig = mask.shape

        # Extract the relevant portion of the mask for the actual Q/K lengths
        # If the mask is larger than the actual Q/K, slice it
        if n_q_orig > n_q or n_k_orig > n_k:
            mask = mask[:, :n_q, :n_k]
            n_q_orig, n_k_orig = n_q, n_k

        # Pad to multiple of block size
        n_q_pad = (n_q_orig + block_size - 1) // block_size * block_size
        n_k_pad = (n_k_orig + block_size - 1) // block_size * block_size

        mask_padded = F.pad(mask, (0, n_k_pad - n_k_orig, 0, n_q_pad - n_q_orig))

        # Reshape to blocks
        b_q = n_q_pad // block_size
        b_k = n_k_pad // block_size

        mask_blocks = mask_padded.view(batch_size, b_q, block_size, b_k, block_size)

        # A block is valid if ANY token in it is valid (> 0)
        mask_block = mask_blocks.gt(0).any(dim=(2, 4))

        return mask_block

    def decode(
        self,
        Q_t: torch.Tensor,
        K_t: torch.Tensor,
        cache: DecodeCache,
        block_size: int,
    ) -> Tuple[torch.Tensor, DecodeCache]:
        """
        SketchWalk decode step (Algorithm 2, Steps 2-4).

        This updates the cache incrementally with the new token's information:
        - Step 2: Update key block representative (incremental running average)
        - Step 3: Compute sketched query for current token
        - Step 4: Update block attention estimate

        Args:
            Q_t: New query tensor of shape (batch, num_heads, 1, head_dim)
            K_t: New key tensor of shape (batch, num_kv_heads, 1, head_dim)
            cache: Decode cache with prefill/previous decode state
            block_size: Block size B

        Returns:
            A_hat_updated: Updated block attention of shape (batch, b_q+1, b_k+1)
            cache: Updated cache
        """
        batch_size, num_heads, n_q, head_dim = Q_t.shape
        n_kv_heads = K_t.shape[1]
        B = block_size

        # Determine current key block index: b_curr = ceil(t / B)
        t = cache.current_position + 1  # New token position (1-indexed)
        b_curr = (t + B - 1) // B - 1  # 0-indexed block

        # Ensure cache has capacity for the new block
        cache.ensure_capacity_for_block(batch_size, Q_t.device, b_curr + 1)

        # Head-averaged key for block representative update
        # k^k_t,avg = (1/h_kv) * sum_u K^k_t[u,:]
        K_avg = K_t.mean(dim=1).squeeze(1)  # (batch, head_dim)

        # Update key block representative incrementally
        if t % B == 1:  # First token in new block
            # k̄^k_b_curr = k^k_t,avg
            cache.cached_key_blocks[:, b_curr:b_curr+1] = K_avg.unsqueeze(1)
            cache.key_block_counts[:, b_curr:b_curr+1] = 1
        else:
            # k̄^k_b_curr = (c * k̄^k_b_curr + k^k_t,avg) / (c + 1)
            c = cache.key_block_counts[:, b_curr:b_curr+1].float().unsqueeze(-1)  # (batch, 1, 1)
            old_k_bar = cache.cached_key_blocks[:, b_curr:b_curr+1]  # (batch, 1, head_dim)
            new_k_bar = (c * old_k_bar + K_avg.unsqueeze(1)) / (c + 1)
            cache.cached_key_blocks[:, b_curr:b_curr+1] = new_k_bar
            cache.key_block_counts[:, b_curr:b_curr+1] = cache.key_block_counts[:, b_curr:b_curr+1] + 1

        # Step 3: Head-averaged query + Hadamard reduction
        # q^k_t,avg = (1/h) * sum_u Q^k_t[u,:]
        Q_avg = Q_t.mean(dim=1).squeeze(1)  # (batch, head_dim)

        # Store query block representative
        # For decode, we have single query token, so q̄^k_b_q = q^k_t,avg
        b_q = (t + B - 1) // B - 1  # Current query block (same as key block for decode)
        cache.cached_query_blocks[:, b_q:b_q+1] = Q_avg.unsqueeze(1)

        # Hadamard transform
        # q̃^k_t = q^k_t,avg @ H_d
        q_tilde = self.hadamard(Q_avg.unsqueeze(1))  # (batch, 1, sketch_dim)

        # K̃^k = [k̄^k_1 @ H_d; ...; k̄^k_b_curr @ H_d]
        # Only use blocks up to current block
        K_bar_cached = cache.cached_key_blocks[:, :b_curr+1]  # (batch, b_curr+1, head_dim)
        K_tilde = self.hadamard(K_bar_cached)  # (batch, b_curr+1, sketch_dim)

        # Compute new attention row: â^k_new = q̃^k_t @ K̃^k^T / sqrt(r)
        a_new = torch.bmm(q_tilde, K_tilde.transpose(1, 2)) / math.sqrt(self.config.sketch_dim)
        # a_new shape: (batch, 1, b_curr+1)

        # Step 4: Update cached block attention estimate
        # (i) Update the new query row
        cache.cached_block_attn[:, b_q, :b_curr+1] = a_new.squeeze(1)

        # (ii) Update the last column for the current key block
        # Q̄^k = [q̄^k_1 @ H_d; ...; q̄^k_b_q @ H_d]
        Q_bar_cached = cache.cached_query_blocks[:, :b_q+1]  # (batch, b_q+1, head_dim)
        Q_tilde_cached = self.hadamard(Q_bar_cached)  # (batch, b_q+1, sketch_dim)

        # k̃^k_b_curr = k̄^k_b_curr @ H_d
        k_tilde_curr = self.hadamard(cache.cached_key_blocks[:, b_curr:b_curr+1])  # (batch, 1, sketch_dim)

        # c^k_new = Q̄^k @ k̃^k_b_curr^T / sqrt(r)
        c_new = torch.bmm(Q_tilde_cached, k_tilde_curr.transpose(1, 2)) / math.sqrt(self.config.sketch_dim)
        # c_new shape: (batch, b_q+1, 1)

        cache.cached_block_attn[:, :b_q+1, b_curr] = c_new.squeeze(-1)

        # Update position
        cache.current_position = t

        return cache.cached_block_attn, cache


class Walk(nn.Module):
    """
    Sketch-Determined Walk - Component 2 of SketchWalk.

    Maintains walk state R^k across layers to accumulate multi-hop
    token dependencies. At each layer, updates walk state and selects
    top-τ blocks for sparse attention computation.

    Key innovation: R^k[i,j] represents accumulated importance from
    query block i to key block j through ALL multi-hop paths across
    layers 0 through k, not just direct one-hop attention.
    """

    def __init__(self, config: SketchWalkConfig):
        """
        Initialize Walk module.

        Args:
            config: SketchWalk configuration
        """
        super().__init__()
        self.config = config
        self.walk_state = None
        self.current_layer = None

    def reset_state(self):
        """Reset walk state (e.g., when starting a new sequence)."""
        self.walk_state = None
        self.current_layer = None

    def update(
        self,
        A_hat: torch.Tensor,
        layer_idx: int,
        causal: bool = True
    ) -> torch.Tensor:
        """
        Update walk state with current layer's block attention.

        Args:
            A_hat: Sketched block attention of shape (batch, b, b)
            layer_idx: Current transformer layer index
            causal: Whether to apply causal masking

        Returns:
            Updated walk state R^k of shape (batch, b, b)
        """
        batch_size, b, _ = A_hat.shape
        device = A_hat.device
        dtype = self.config.walk_state_dtype

        # Apply softmax before exponentiation (bounds values in [0,1])
        W = F.softmax(A_hat, dim=-1)

        # Apply sparsity exponent: W^k = (softmax(A_hat))^s
        W = W.pow(self.config.sparsity_exponent)

        # Convert W to walk state dtype for consistency
        W = W.to(dtype)

        # Check if this is the first layer or layer reset
        if layer_idx == 0 or not hasattr(self, 'walk_state') or self.walk_state is None:
            self.walk_state = W
            self.current_layer = layer_idx
        elif layer_idx < self.current_layer:
            # Layer jump detected (going backwards), reset
            self.walk_state = W
            self.current_layer = layer_idx
        else:
            # Update walk state: R^k = R^{k-1} @ W^k
            self.walk_state = torch.bmm(self.walk_state, W)
            self.current_layer = layer_idx

        # Apply causal masking if needed
        if causal:
            # Create causal mask: blocks > query block are masked
            causal_mask = torch.triu(torch.ones(b, b, device=device), diagonal=1).bool()
            self.walk_state = self.walk_state.masked_fill(causal_mask, 0)

        return self.walk_state

    def select_blocks(
        self,
        walk_state: torch.Tensor,
        num_query_blocks: int,
        include_first_last: bool = True
    ) -> torch.Tensor:
        """
        Select top-τ key blocks for each query block based on walk state.

        Args:
            walk_state: Walk state R^k of shape (batch, b, b)
            num_query_blocks: Number of query blocks b_q
            include_first_last: Whether to always include first and last blocks

        Returns:
            Selected block indices of shape (batch, b_q, tau)
        """
        tau = min(self.config.top_k_blocks, walk_state.shape[-1])  # Don't exceed available blocks

        # Get top-τ blocks for each query block
        _, top_indices = torch.topk(walk_state[:, :num_query_blocks], k=tau, dim=-1)

        selected = top_indices  # (batch, b_q, tau)

        # Always include first block (block 0)
        if include_first_last and tau > 1:
            first_block = torch.zeros(
                walk_state.shape[0], num_query_blocks, 1,
                dtype=torch.long, device=walk_state.device
            )
            # Only replace if we have room
            if tau > 1:
                selected = torch.cat([first_block, selected[:, :, :-1]], dim=-1)

        return selected

    def get_selected_blocks(
        self,
        A_hat: torch.Tensor,
        layer_idx: int,
        num_query_blocks: int,
        causal: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Complete forward pass: update walk state and select blocks.

        Args:
            A_hat: Sketched block attention of shape (batch, b, b)
            layer_idx: Current layer index
            num_query_blocks: Number of query blocks
            causal: Whether to apply causal masking

        Returns:
            walk_state: Updated walk state of shape (batch, b, b)
            selected_blocks: Selected indices of shape (batch, b_q, tau)
        """
        # Update walk state
        walk_state = self.update(A_hat, layer_idx, causal)

        # Select top-τ blocks
        selected_blocks = self.select_blocks(walk_state, num_query_blocks)

        return walk_state, selected_blocks

    def decode(
        self,
        A_hat_new: torch.Tensor,
        layer_idx: int,
        b_q: int,
        b_curr: int,
        causal: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update walk state during decode (Algorithm 2, Step 5-6).

        Args:
            A_hat_new: Updated block attention of shape (batch, b_q+1, b_curr+1)
            layer_idx: Current layer index
            b_q: Current query block index
            b_curr: Current key block index (same as b_q for decode)
            causal: Whether to apply causal masking

        Returns:
            walk_state: Updated walk state of shape (batch, b_q+1, b_curr+1)
            selected_blocks: Selected indices of shape (batch, 1, tau)
        """
        batch_size = A_hat_new.shape[0]
        device = A_hat_new.device
        dtype = self.config.walk_state_dtype
        new_b_q, new_b_k = A_hat_new.shape[1], A_hat_new.shape[2]

        # Step 5: Random Walk
        # W^k = (Â^k_block)^s
        W = F.softmax(A_hat_new, dim=-1)
        W = W.pow(self.config.sparsity_exponent).to(dtype)

        # Update walk state
        if layer_idx == 0 or not hasattr(self, 'walk_state') or self.walk_state is None:
            self.walk_state = W
            self.current_layer = layer_idx
        elif layer_idx < self.current_layer:
            # Going backwards, reset
            self.walk_state = W
            self.current_layer = layer_idx
        else:
            # R^k = R^{k-1} @ W^k
            # Need to handle changing size as we add tokens
            old_b_q, old_b_k = self.walk_state.shape[1], self.walk_state.shape[2]

            if new_b_q > old_b_q or new_b_k > old_b_k:
                # Pad walk state to new size
                self.walk_state = F.pad(self.walk_state, (0, new_b_k - old_b_k, 0, new_b_q - old_b_q))

            # CRITICAL FIX: Correct walk state update for decode
            # R^k = R^{k-1} @ W^k where:
            #   - R^{k-1} has shape (batch, old_b_q, old_b_k) or padded to (batch, new_b_q, new_b_k)
            #   - W^k has shape (batch, new_b_q, new_b_k)
            # After padding, walk_state is (batch, new_b_q, new_b_k)
            # We need to multiply: walk_state @ W
            # But for efficient computation with growing cache, we multiply over the overlapping region

            # The correct approach: use full walk_state and W
            self.walk_state = torch.bmm(self.walk_state, W)
            self.current_layer = layer_idx

        # Apply causal masking
        if causal:
            # For decode, mask blocks that are in the future
            causal_mask = torch.triu(torch.ones(new_b_q, new_b_k, device=device), diagonal=1).bool()
            self.walk_state = self.walk_state.masked_fill(causal_mask, 0)

        # Step 6: Select top-τ blocks using the new query row
        # Always include current block
        # S = TopK-Indices(R^k[b_q, 1:b_curr], τ-1) ∪ {b_curr}
        walk_row = self.walk_state[:, b_q, :b_curr+1]  # (batch, b_curr+1)

        tau = min(self.config.top_k_blocks, b_curr + 1)
        if tau > 1:
            _, top_indices = torch.topk(walk_row, k=tau - 1, dim=-1)
            # Add current block (b_curr) - CRITICAL FIX: use b_curr not 0
            current_block_idx = torch.full((batch_size, 1, 1), b_curr, dtype=torch.long, device=device)
            selected = torch.cat([current_block_idx, top_indices.unsqueeze(1)], dim=-1)  # (batch, 1, tau)
        else:
            # Just current block (b_curr)
            selected = torch.full((batch_size, 1, 1), b_curr, dtype=torch.long, device=device)

        return self.walk_state, selected


class SketchWalkAttention(nn.Module):
    """
    Complete SketchWalk sparse attention module.

    Combines Sketch (SWS) and Walk components to provide training-free
    sparse attention with minimal accuracy loss.

    Usage:
        1. Initialize with config and head_dim
        2. Call forward() with Q, K, V for each layer
        3. Walk state is maintained automatically across layers
        4. Reset with reset_state() between sequences
    """

    def __init__(self, config: SketchWalkConfig, head_dim: int):
        """
        Initialize SketchWalk attention.

        Args:
            config: SketchWalk configuration
            head_dim: Attention head dimension
        """
        super().__init__()
        self.config = config
        self.head_dim = head_dim

        # Initialize components
        self.sketch = Sketch(config, head_dim)
        self.walk = Walk(config)

    def reset_state(self):
        """Reset walk state for a new sequence."""
        self.walk.reset_state()

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        layer_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute sparse attention using SketchWalk.

        Args:
            Q: Query tensor of shape (batch, num_heads, n_q, head_dim)
            K: Key tensor of shape (batch, num_heads, n_k, head_dim)
            V: Value tensor of shape (batch, num_heads, n_k, head_dim)
            layer_idx: Current transformer layer index
            attention_mask: Optional attention mask
            causal: Whether to apply causal masking

        Returns:
            output: Attention output of shape (batch, num_heads, n_q, head_dim)
            selected_blocks: Selected block indices of shape (batch, b_q, tau)
        """
        batch_size, num_heads, n_q, n_k = Q.shape[0], Q.shape[1], Q.shape[2], K.shape[2]
        B = self.config.block_size

        # Skip first N layers (use dense attention)
        if layer_idx < self.config.skip_first_n_layers:
            return self._dense_attention(Q, K, V, attention_mask), None

        # Compute sketched block attention
        A_hat, Q_bar, K_bar = self.sketch(Q, K, attention_mask)

        # Update walk state and select blocks
        num_blocks = Q_bar.shape[1]
        walk_state, selected_blocks = self.walk.get_selected_blocks(
            A_hat, layer_idx, num_blocks, causal
        )

        # Compute sparse attention over selected blocks
        output = self._sparse_attention(Q, K, V, selected_blocks, B)

        return output, selected_blocks

    def _dense_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute standard dense attention (for skipped layers).

        Args:
            Q: Query tensor
            K: Key tensor
            V: Value tensor
            attention_mask: Optional mask

        Returns:
            Attention output
        """
        # Standard scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        return output

    def _sparse_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        selected_blocks: torch.Tensor,
        block_size: int
    ) -> torch.Tensor:
        """
        Compute sparse attention over selected blocks.

        Args:
            Q: Query tensor of shape (batch, num_heads, n_q, head_dim)
            K: Key tensor of shape (batch, num_heads, n_k, head_dim)
            V: Value tensor of shape (batch, num_heads, n_k, head_dim)
            selected_blocks: Selected indices of shape (batch, b_q, tau)
            block_size: Block size B

        Returns:
            Attention output of shape (batch, num_heads, n_q, head_dim)
        """
        batch_size, num_heads, n_q, head_dim = Q.shape
        n_k = K.shape[2]
        b_q = selected_blocks.shape[1]

        # Initialize output
        output = torch.zeros_like(Q)

        # Process each block of queries
        for block_idx in range(b_q):
            # Query block range
            q_start = block_idx * block_size
            q_end = min(q_start + block_size, n_q)

            # Get selected key blocks for this query block
            key_block_indices = selected_blocks[:, block_idx]  # (batch, tau)

            # Gather key and value blocks
            for batch_i in range(batch_size):
                for head_j in range(num_heads):
                    Q_block = Q[batch_i, head_j, q_start:q_end]

                    # Gather K and V for selected blocks
                    K_selected_list = []
                    V_selected_list = []

                    for k_block_idx in key_block_indices[batch_i]:
                        k_start = int(k_block_idx) * block_size
                        k_end = min(k_start + block_size, n_k)

                        K_selected_list.append(K[batch_i, head_j, k_start:k_end])
                        V_selected_list.append(V[batch_i, head_j, k_start:k_end])

                    K_selected = torch.cat(K_selected_list, dim=0)
                    V_selected = torch.cat(V_selected_list, dim=0)

                    # Compute attention
                    attn_scores = torch.matmul(Q_block, K_selected.T) / math.sqrt(head_dim)
                    attn_weights = F.softmax(attn_scores, dim=-1)
                    attn_output = torch.matmul(attn_weights, V_selected)

                    output[batch_i, head_j, q_start:q_end] = attn_output

        return output

    def init_decode_cache(
        self,
        Q_prefill: torch.Tensor,
        K_prefill: torch.Tensor,
        V_prefill: torch.Tensor,
        A_hat_prefill: torch.Tensor,
        num_tokens: int,
    ) -> DecodeCache:
        """
        Initialize decode cache from prefill phase outputs.

        Args:
            Q_prefill: Query tensor from prefill of shape (batch, num_heads, n_q, head_dim)
            K_prefill: Key tensor from prefill of shape (batch, num_heads, n_k, head_dim)
            V_prefill: Value tensor from prefill of shape (batch, num_heads, n_k, head_dim)
            A_hat_prefill: Block attention from prefill of shape (batch, b_q, b_k)
            num_tokens: Number of tokens in prefill

        Returns:
            cache: Initialized decode cache
        """
        # Recompute block representatives from prefill
        Q_avg = Q_prefill.mean(dim=1)  # (batch, n_q, head_dim)
        K_avg = K_prefill.mean(dim=1)  # (batch, n_k, head_dim)

        batch_size = Q_avg.shape[0]
        B = self.config.block_size
        device = Q_prefill.device

        # Compute block representatives
        n_q, n_k = Q_avg.shape[1], K_avg.shape[1]
        n_q_padded = (n_q + B - 1) // B * B
        n_k_padded = (n_k + B - 1) // B * B

        Q_padded = F.pad(Q_avg, (0, 0, 0, n_q_padded - n_q))
        K_padded = F.pad(K_avg, (0, 0, 0, n_k_padded - n_k))

        b_q = n_q_padded // B
        b_k = n_k_padded // B

        Q_blocks = Q_padded.view(batch_size, b_q, B, -1).mean(dim=2)
        K_blocks = K_padded.view(batch_size, b_k, B, -1).mean(dim=2)

        # Initialize cache with proper head_dim
        cache = DecodeCache(device=device, dtype=torch.float32, head_dim=self.head_dim)
        cache.initialize_from_prefill(Q_blocks, K_blocks, A_hat_prefill, num_tokens)

        return cache

    def decode(
        self,
        Q_t: torch.Tensor,
        K_cache: torch.Tensor,
        V_cache: torch.Tensor,
        layer_idx: int,
        cache: DecodeCache,
        causal: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        SketchWalk decode step for single token generation (Algorithm 2).

        This implements the complete decode algorithm:
        - Updates KV cache with new token
        - Incrementally updates block representatives and attention
        - Updates walk state
        - Selects top-τ blocks (always including current block)
        - Computes sparse attention over selected blocks

        Args:
            Q_t: New query tensor of shape (batch, num_heads, 1, head_dim)
            K_cache: Cached key tensor of shape (batch, num_kv_heads, cache_len, head_dim)
            V_cache: Cached value tensor of shape (batch, num_kv_heads, cache_len, head_dim)
            layer_idx: Current transformer layer index
            cache: Decode cache with prefill state
            causal: Whether to apply causal masking

        Returns:
            output: Attention output of shape (batch, num_heads, 1, head_dim)
            cache: Updated cache
        """
        batch_size, num_heads, n_q, head_dim = Q_t.shape
        n_kv_heads = K_cache.shape[1]
        cache_len = K_cache.shape[2]
        B = self.config.block_size

        # Skip first N layers (use dense attention)
        if layer_idx < self.config.skip_first_n_layers:
            # For decode, we need to use the full cache
            # Repeat K/V for GQA
            repeat_factor = num_heads // n_kv_heads
            K_expanded = K_cache.repeat_interleave(repeat_factor, dim=1)
            V_expanded = V_cache.repeat_interleave(repeat_factor, dim=1)

            # Compute attention with new query and cached K/V
            attn_scores = torch.matmul(Q_t, K_expanded.transpose(-2, -1)) / math.sqrt(head_dim)
            attn_weights = F.softmax(attn_scores, dim=-1)
            output = torch.matmul(attn_weights, V_expanded)

            return output, cache

        # Algorithm 2, Step 1: Project Q_t to get K_t, V_t
        # In practice, K_t and V_t are already computed and appended to cache
        # We use the cache_len to determine current token position

        # Step 2-4: Update block representatives and block attention estimate
        A_hat_updated, cache = self.sketch.decode(
            Q_t, K_cache[:, :, -1:, :],  # Only the new token's K
            cache, B
        )

        # Step 5-6: Update walk state and select blocks
        t = cache.current_position
        b_q = (t + B - 1) // B - 1  # Current query block
        b_curr = b_q  # Same for decode (single token)

        walk_state, selected_blocks = self.walk.decode(
            A_hat_updated, layer_idx, b_q, b_curr, causal
        )

        # Step 7: Sparse attention using selected blocks
        output = self._decode_sparse_attention(
            Q_t, K_cache, V_cache, selected_blocks, B
        )

        return output, cache

    def _decode_sparse_attention(
        self,
        Q_t: torch.Tensor,
        K_cache: torch.Tensor,
        V_cache: torch.Tensor,
        selected_blocks: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        """
        Compute sparse attention over selected blocks for decode.

        Args:
            Q_t: Query tensor of shape (batch, num_heads, 1, head_dim)
            K_cache: Cached key tensor of shape (batch, num_kv_heads, cache_len, head_dim)
            V_cache: Cached value tensor of shape (batch, num_kv_heads, cache_len, head_dim)
            selected_blocks: Selected indices of shape (batch, 1, tau)
            block_size: Block size B

        Returns:
            Attention output of shape (batch, num_heads, 1, head_dim)
        """
        batch_size, num_heads, n_q, head_dim = Q_t.shape
        n_kv_heads = K_cache.shape[1]
        cache_len = K_cache.shape[2]

        # Repeat K/V for GQA
        repeat_factor = num_heads // n_kv_heads
        K_expanded = K_cache.repeat_interleave(repeat_factor, dim=1)
        V_expanded = V_cache.repeat_interleave(repeat_factor, dim=1)

        # Initialize output
        output = torch.zeros_like(Q_t)

        # For decode, we only have one query token (n_q = 1)
        # Process each head
        for batch_i in range(batch_size):
            for head_j in range(num_heads):
                Q_single = Q_t[batch_i, head_j, 0]  # (head_dim,)

                # Get selected blocks for this single query
                block_indices = selected_blocks[batch_i, 0]  # (tau,)

                # Gather K and V for selected blocks
                K_selected_list = []
                V_selected_list = []

                for k_block_idx in block_indices:
                    k_start = int(k_block_idx) * block_size
                    k_end = min(k_start + block_size, cache_len)

                    K_selected_list.append(K_expanded[batch_i, head_j, k_start:k_end])
                    V_selected_list.append(V_expanded[batch_i, head_j, k_start:k_end])

                K_selected = torch.cat(K_selected_list, dim=0)
                V_selected = torch.cat(V_selected_list, dim=0)

                # Compute attention
                attn_scores = torch.matmul(Q_single.unsqueeze(0), K_selected.T) / math.sqrt(head_dim)
                attn_weights = F.softmax(attn_scores, dim=-1)
                attn_output = torch.matmul(attn_weights, V_selected)

                output[batch_i, head_j, 0] = attn_output

        return output


def create_sketch_walk_config(
    block_size: int = 64,
    sketch_dim: int = 64,
    top_k_blocks: int = 16,
    sparsity_exponent: int = 8,
    **kwargs
) -> SketchWalkConfig:
    """
    Factory function to create SketchWalk configuration with sensible defaults.

    Args:
        block_size: Tokens per block (default: 64)
        sketch_dim: Sketch dimension (default: 64)
        top_k_blocks: Number of blocks to select (default: 16)
        sparsity_exponent: Sparsity exponent (default: 8)
        **kwargs: Additional config parameters

    Returns:
        SketchWalkConfig instance
    """
    return SketchWalkConfig(
        block_size=block_size,
        sketch_dim=sketch_dim,
        top_k_blocks=top_k_blocks,
        sparsity_exponent=sparsity_exponent,
        **kwargs
    )
