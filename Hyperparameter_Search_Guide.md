# SketchWalk Hyperparameter Search Framework

**Version**: 1.0
**Date**: 2025-02-19
**Status**: Comprehensive Guide

## Table of Contents

1. [Hyperparameter Overview](#hyperparameter-overview)
2. [Theoretical Foundations](#theoretical-foundations)
3. [Search Strategies](#search-strategies)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Practical Guidelines](#practical-guidelines)
6. [Ablation Study Templates](#ablation-study-templates)
7. [Automated Search Framework](#automated-search-framework)
8. [Case Studies](#case-studies)

---

## 1. Hyperparameter Overview

### 1.1 Primary Hyperparameters

| Parameter | Symbol | Space | Default | Impact |
|-----------|--------|-------|---------|--------|
| Block Size | B | {32, 64, 128} | 64 | Granularity vs efficiency |
| Sketch Dim | k | {16, 32, 64, 128} | 64 | Accuracy vs speed |
| Top Blocks | τ | {4, 8, 16, 32} | 16 | Sparsity level |
| Sparsity Exp | s | {2, 4, 8, 16} | 8 | Selectivity |
| Skip Layers | N | {0, 1, 2, 3} | 2 | Early layer handling |

### 1.2 Secondary Hyperparameters

| Parameter | Space | Default | Purpose |
|-----------|-------|---------|---------|
| hadamard_seed | int | 42 | Reproducibility |
| walk_state_dtype | {fp32, fp16} | fp32 | Numerical stability |
| use_srht | bool | True | Projection method |

### 1.3 Derived Quantities

```python
# Number of blocks
b = ceil(n / B)

# Theoretical sparsity
sparsity = 1 - min(1, (τ * B) / n)

# Block-level sparsity
block_sparsity = 1 - min(1, τ / b)

# Complexity factor
complexity_factor = (τ * B / h) / n  # h = num_heads
```

---

## 2. Theoretical Foundations

### 2.1 Block Size Selection (B)

**Theoretical Trade-off**:
```
Memory: O(b²) = O((n/B)²)
Computation: O(n * τ * B * d)
Granularity: B tokens per block
```

**Guidelines**:
- **Small B (32)**: Higher granularity, more blocks, more memory
- **Medium B (64)**: Balanced (recommended)
- **Large B (128)**: Lower granularity, fewer blocks, less memory

**Sequence Length Adaptation**:
```python
def adaptive_block_size(seq_len: int) -> int:
    if seq_len < 2048:
        return 32
    elif seq_len < 32768:
        return 64
    else:
        return 128
```

### 2.2 Sketch Dimension Selection (k)

**Theoretical Foundation** (Johnson-Lindenstrauss):
For subspace embedding with error ε and failure probability δ:
```
k ≥ C * log(b/δ) / ε²
```

**Practical Implications**:
- Higher k → Better inner product preservation
- Lower k → Faster computation
- k ≥ log₂(b) recommended for stability

**Validation**:
```python
def validate_sketch_dim(k: int, num_blocks: int) -> bool:
    return k >= math.log2(num_blocks)
```

### 2.3 Top Blocks Selection (τ)

**Sparsity Control**:
```
target_sparsity = 0.80  # 80% sparse
τ = ceil((1 - target_sparsity) * n / B)
```

**Layer-wise Adaptation**:
```python
def adaptive_tau(layer_idx: int, num_layers: int, base_tau: int) -> int:
    """Increase τ for early layers, decrease for late layers"""
    progress = layer_idx / num_layers
    # Early layers: 2x τ, Late layers: 0.5x τ
    factor = 2.0 * (1 - progress) + 0.5 * progress
    return max(4, int(base_tau * factor))
```

### 2.4 Sparsity Exponent (s)

**Effect on Distribution**:
```
Before: p_i = exp(x_i) / Σ exp(x_j)
After:  p_i^s = [exp(x_i) / Σ exp(x_j)]^s
```

**Characteristics**:
- s=1: No sharpening (original softmax)
- s=2: Moderate sharpening
- s=8: Strong sharpening (paper default)
- s=16: Very strong sharpening (may overfit)

**Selection Rule**:
```python
def select_sparsity_exponent(target_sparsity: float) -> int:
    if target_sparsity < 0.5:
        return 2
    elif target_sparsity < 0.7:
        return 4
    elif target_sparsity < 0.9:
        return 8
    else:
        return 16
```

---

## 3. Search Strategies

### 3.1 Grid Search

**Best for**: Exhaustive evaluation, small search spaces

**Template**:
```python
param_grid = {
    'block_size': [32, 64, 128],
    'sketch_dim': [32, 64, 128],
    'top_k_blocks': [8, 16, 32],
    'sparsity_exponent': [4, 8, 16],
}

# Total: 3 × 3 × 3 × 3 = 81 combinations
```

**Implementation**:
```python
from itertools import product

def grid_search(param_grid, seq_len, head_dim):
    results = []
    keys = param_grid.keys()
    values = param_grid.values()

    for combination in product(*values):
        params = dict(zip(keys, combination))
        config = SketchWalkConfig(**params)

        # Evaluate
        metrics = evaluate_config(config, seq_len, head_dim)
        results.append({**params, **metrics})

    return results
```

### 3.2 Random Search

**Best for**: Large search spaces, non-uniform importance

**Implementation**:
```python
import random

def random_search(param_bounds, n_trials, seq_len, head_dim):
    results = []

    for _ in range(n_trials):
        params = {
            'block_size': random.choice([32, 64, 128]),
            'sketch_dim': random.randint(16, 128),
            'top_k_blocks': random.randint(4, 32),
            'sparsity_exponent': random.choice([2, 4, 8, 16]),
        }

        config = SketchWalkConfig(**params)
        metrics = evaluate_config(config, seq_len, head_dim)
        results.append({**params, **metrics})

    return results
```

### 3.3 Bayesian Optimization

**Best for**: Expensive evaluations, smooth objective

**Implementation** (using scikit-optimize):
```python
from skopt import gp_minimize

def objective(params):
    block_size, sketch_dim, top_k, sparsity_exp = params
    config = SketchWalkConfig(
        block_size=int(block_size),
        sketch_dim=int(sketch_dim),
        top_k_blocks=int(top_k),
        sparsity_exponent=int(sparsity_exp),
    )

    metrics = evaluate_config(config, seq_len, head_dim)
    return -metrics['accuracy']  # Minimize negative accuracy

# Define search space
space = [
    (32, 128),  # block_size
    (16, 128),  # sketch_dim
    (4, 32),    # top_k_blocks
    (2, 16),    # sparsity_exponent
]

result = gp_minimize(objective, space, n_calls=50)
```

### 3.4 Multi-Objective Optimization

**Pareto Front**:
```python
import numpy as np

def find_pareto_front(results):
    """Find Pareto-optimal configurations"""
    # Extract objectives
    accuracy = np.array([r['accuracy'] for r in results])
    speedup = np.array([r['speedup'] for r in results])

    # Find Pareto front
    pareto = []
    for i, (acc, spd) in enumerate(zip(accuracy, speedup)):
        dominated = False
        for j, (acc_j, spd_j) in enumerate(zip(accuracy, speedup)):
            if i != j and acc_j >= acc and spd_j >= spd:
                if acc_j > acc or spd_j > spd:
                    dominated = True
                    break
        if not dominated:
            pareto.append(results[i])

    return pareto
```

---

## 4. Evaluation Metrics

### 4.1 Accuracy Metrics

**Perplexity**:
```python
def compute_perplexity(model, dataloader, config):
    """Compute perplexity on validation set"""
    total_loss = 0
    total_tokens = 0

    for batch in dataloader:
        outputs = model(**batch)
        total_loss += outputs.loss * batch['input_ids'].numel()
        total_tokens += batch['input_ids'].numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity
```

**Task Performance**:
```python
def evaluate_task(model, task, dataset):
    """Evaluate on specific task (e.g., QA, summarization)"""
    if task == 'qa':
        return evaluate_qa(model, dataset)
    elif task == 'summarization':
        return evaluate_summarization(model, dataset)
    # ...
```

### 4.2 Efficiency Metrics

**Speedup**:
```python
def measure_speedup(config, seq_len, num_runs=10):
    """Measure speedup vs dense attention"""
    # Time dense attention
    dense_times = []
    for _ in range(num_runs):
        start = time.time()
        dense_attention(Q, K, V)
        dense_times.append(time.time() - start)

    # Time sparse attention
    sparse_times = []
    for _ in range(num_runs):
        start = time.time()
        sparse_attention(Q, K, V, config)
        sparse_times.append(time.time() - start)

    avg_dense = np.mean(dense_times)
    avg_sparse = np.mean(sparse_times)
    speedup = avg_dense / avg_sparse

    return speedup
```

**Memory Usage**:
```python
def measure_memory(config, seq_len):
    """Measure peak memory usage"""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Run forward pass
    output = model(input_ids)

    peak_memory = torch.cuda.max_memory_allocated()
    return peak_memory / (1024**3)  # GB
```

**Sparsity Level**:
```python
def compute_actual_sparsity(selected_blocks, total_blocks):
    """Compute actual sparsity achieved"""
    unique_blocks = len(torch.unique(selected_blocks))
    actual_sparsity = 1 - (unique_blocks / total_blocks)
    return actual_sparsity
```

### 4.3 Composite Metrics

**Efficiency-Accuracy Trade-off**:
```python
def composite_score(accuracy, speedup, memory_ratio,
                   w_acc=1.0, w_speed=1.0, w_mem=0.5):
    """
    Compute composite score.

    Args:
        accuracy: Normalized accuracy (0-1)
        speedup: Speedup factor vs dense
        memory_ratio: Memory ratio (sparse/dense)
        w_acc, w_speed, w_mem: Weights
    """
    score = (w_acc * accuracy +
             w_speed * math.log(speedup) -
             w_mem * memory_ratio)
    return score
```

---

## 5. Practical Guidelines

### 5.1 Sequence Length Categories

**Short Sequences (< 4K)**:
```python
config = SketchWalkConfig(
    block_size=32,
    sketch_dim=32,
    top_k_blocks=8,
    sparsity_exponent=4,
)
```

**Medium Sequences (4K - 32K)**:
```python
config = SketchWalkConfig(
    block_size=64,
    sketch_dim=64,
    top_k_blocks=16,
    sparsity_exponent=8,
)
```

**Long Sequences (> 32K)**:
```python
config = SketchWalkConfig(
    block_size=128,
    sketch_dim=128,
    top_k_blocks=32,
    sparsity_exponent=8,
)
```

### 5.2 Task-Specific Recommendations

**Question Answering** (High accuracy needed):
```python
config = SketchWalkConfig(
    block_size=64,
    sketch_dim=128,
    top_k_blocks=32,
    sparsity_exponent=4,
)
```

**Summarization** (Medium accuracy):
```python
config = SketchWalkConfig(
    block_size=64,
    sketch_dim=64,
    top_k_blocks=16,
    sparsity_exponent=8,
)
```

**Code Generation** (High accuracy):
```python
config = SketchWalkConfig(
    block_size=32,
    sketch_dim=128,
    top_k_blocks=32,
    sparsity_exponent=4,
)
```

### 5.3 Hardware-Specific Tuning

**GPU with High Memory**:
```python
config = SketchWalkConfig(
    block_size=32,
    sketch_dim=128,
    top_k_blocks=32,
    walk_state_dtype=torch.float32,
)
```

**GPU with Limited Memory**:
```python
config = SketchWalkConfig(
    block_size=128,
    sketch_dim=32,
    top_k_blocks=8,
    walk_state_dtype=torch.float16,
)
```

---

## 6. Ablation Study Templates

### 6.1 Block Size Ablation

```python
def ablate_block_size(seq_len=4096, head_dim=128):
    """Ablate block size while keeping other params fixed"""
    block_sizes = [16, 32, 64, 128, 256]

    results = []
    for B in block_sizes:
        config = SketchWalkConfig(
            block_size=B,
            sketch_dim=64,
            top_k_blocks=16,
            sparsity_exponent=8,
        )

        metrics = evaluate_config(config, seq_len, head_dim)
        results.append({
            'block_size': B,
            **metrics
        })

    return results
```

### 6.2 Sketch Dimension Ablation

```python
def ablate_sketch_dim(seq_len=4096, head_dim=128):
    """Ablate sketch dimension"""
    sketch_dims = [16, 32, 64, 128, 256]

    results = []
    for k in sketch_dims:
        config = SketchWalkConfig(
            block_size=64,
            sketch_dim=k,
            top_k_blocks=16,
            sparsity_exponent=8,
        )

        metrics = evaluate_config(config, seq_len, head_dim)
        results.append({
            'sketch_dim': k,
            **metrics
        })

    return results
```

### 6.3 Sparsity vs Accuracy Trade-off

```python
def ablate_sparsity(seq_len=4096, head_dim=128):
    """Ablate sparsity level via τ"""
    tau_values = [4, 8, 12, 16, 20, 24, 32]

    results = []
    for tau in tau_values:
        config = SketchWalkConfig(
            block_size=64,
            sketch_dim=64,
            top_k_blocks=tau,
            sparsity_exponent=8,
        )

        metrics = evaluate_config(config, seq_len, head_dim)
        theoretical_sparsity = 1 - (tau * 64) / seq_len

        results.append({
            'tau': tau,
            'theoretical_sparsity': theoretical_sparsity,
            **metrics
        })

    return results
```

### 6.4 Sparsity Exponent Ablation

```python
def ablate_sparsity_exponent(seq_len=4096, head_dim=128):
    """Ablate sparsity exponent"""
    exponents = [1, 2, 4, 8, 16, 32]

    results = []
    for s in exponents:
        config = SketchWalkConfig(
            block_size=64,
            sketch_dim=64,
            top_k_blocks=16,
            sparsity_exponent=s,
        )

        metrics = evaluate_config(config, seq_len, head_dim)
        results.append({
            'sparsity_exponent': s,
            **metrics
        })

    return results
```

---

## 7. Automated Search Framework

### 7.1 Complete Search Pipeline

```python
import json
import logging
from pathlib import Path
from datetime import datetime

class HyperparameterSearch:
    """Automated hyperparameter search framework."""

    def __init__(self, save_dir='./search_results'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger('HyperparamSearch')
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def search(self, strategy='grid', param_grid=None, n_trials=100,
               seq_len=4096, head_dim=128):
        """
        Run hyperparameter search.

        Args:
            strategy: 'grid', 'random', or 'bayesian'
            param_grid: Parameter grid for grid search
            n_trials: Number of trials for random/bayesian
            seq_len: Sequence length for evaluation
            head_dim: Attention head dimension
        """
        self.logger.info(f"Starting {strategy} search")

        if strategy == 'grid':
            results = self._grid_search(param_grid, seq_len, head_dim)
        elif strategy == 'random':
            results = self._random_search(n_trials, seq_len, head_dim)
        elif strategy == 'bayesian':
            results = self._bayesian_search(n_trials, seq_len, head_dim)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Save results
        self._save_results(results, strategy)

        # Find best configuration
        best = self._find_best(results)
        self.logger.info(f"Best config: {best}")

        return results, best

    def _grid_search(self, param_grid, seq_len, head_dim):
        from itertools import product

        results = []
        keys = param_grid.keys()
        values = param_grid.values()

        total = np.prod([len(v) for v in values])
        current = 0

        for combination in product(*values):
            current += 1
            params = dict(zip(keys, combination))

            self.logger.info(f"Trial {current}/{total}: {params}")

            try:
                metrics = self._evaluate_params(params, seq_len, head_dim)
                results.append({**params, **metrics})
            except Exception as e:
                self.logger.error(f"Error evaluating {params}: {e}")

        return results

    def _random_search(self, n_trials, seq_len, head_dim):
        import random

        results = []
        param_space = {
            'block_size': [32, 64, 128],
            'sketch_dim': list(range(16, 129, 16)),
            'top_k_blocks': list(range(4, 33, 4)),
            'sparsity_exponent': [2, 4, 8, 16],
        }

        for trial in range(n_trials):
            params = {
                k: random.choice(v) for k, v in param_space.items()
            }

            self.logger.info(f"Trial {trial + 1}/{n_trials}: {params}")

            try:
                metrics = self._evaluate_params(params, seq_len, head_dim)
                results.append({**params, **metrics})
            except Exception as e:
                self.logger.error(f"Error evaluating {params}: {e}")

        return results

    def _evaluate_params(self, params, seq_len, head_dim):
        """Evaluate a parameter configuration."""
        config = SketchWalkConfig(**params)

        # Create attention module
        attention = SketchWalkAttention(config, head_dim)

        # Create synthetic data
        Q, K, V = create_synthetic_attention_data(
            batch_size=2,
            num_heads=8,
            seq_len=seq_len,
            head_dim=head_dim,
        )

        # Measure speed
        import time
        start = time.time()
        output, selected_blocks = attention(Q, K, V, layer_idx=5, causal=True)
        elapsed = time.time() - start

        # Compute metrics
        num_blocks = (seq_len + config.block_size - 1) // config.block_size
        sparsity = 1 - (config.top_k_blocks / num_blocks)

        # Compare with dense
        dense_output = compute_dense_attention(Q, K, V)
        cosine_sim = F.cosine_similarity(
            output.flatten().unsqueeze(0),
            dense_output.flatten().unsqueeze(0)
        ).item()

        return {
            'time': elapsed,
            'sparsity': sparsity,
            'cosine_similarity': cosine_sim,
            'num_blocks': num_blocks,
        }

    def _save_results(self, results, strategy):
        """Save results to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.save_dir / f'{strategy}_search_{timestamp}.json'

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Results saved to {filename}")

    def _find_best(self, results):
        """Find best configuration based on composite score."""
        scored_results = []

        for r in results:
            score = (
                1.0 * r['cosine_similarity'] +
                0.5 * math.log(1 / r['time']) +
                0.3 * r['sparsity']
            )
            scored_results.append((score, r))

        scored_results.sort(key=lambda x: x[0], reverse=True)
        return scored_results[0][1]
```

### 7.2 Usage Example

```python
# Initialize search
search = HyperparameterSearch(save_dir='./search_results')

# Define parameter grid
param_grid = {
    'block_size': [32, 64, 128],
    'sketch_dim': [32, 64, 128],
    'top_k_blocks': [8, 16, 32],
    'sparsity_exponent': [4, 8, 16],
}

# Run grid search
results, best = search.search(
    strategy='grid',
    param_grid=param_grid,
    seq_len=4096,
    head_dim=128,
)

print(f"Best configuration: {best}")
```

---

## 8. Case Studies

### 8.1 Case Study 1: Optimizing for 64K Sequences

**Goal**: Find best config for 64K token sequences

**Constraints**:
- Memory: < 24GB
- Accuracy loss: < 5%
- Speedup: > 3x

**Search Process**:
```python
# Target sparsity for 3x speedup
target_speedup = 3.0
target_sparsity = 1 - (1 / target_speedup)  # ~67%

# Calculate required τ
seq_len = 65536
block_size = 64
required_tau = int((1 - target_sparsity) * seq_len / block_size)
# required_tau ≈ 340 (too high!)

# Solution: Increase block size
block_size = 256
required_tau = int((1 - target_sparsity) * seq_len / block_size)
# required_tau ≈ 85 (still high)

# Final config
config = SketchWalkConfig(
    block_size=256,
    sketch_dim=128,
    top_k_blocks=128,
    sparsity_exponent=8,
)
```

### 8.2 Case Study 2: Quality-Critical Application

**Goal**: Minimize accuracy loss

**Approach**:
```python
# Conservative configuration
config = SketchWalkConfig(
    block_size=32,        # High granularity
    sketch_dim=128,       # High accuracy
    top_k_blocks=64,      # Low sparsity (~50%)
    sparsity_exponent=2,  # Minimal sharpening
)

# Expected results
# Sparsity: ~50%
# Speedup: ~1.5-2x
# Accuracy loss: < 1%
```

### 8.3 Case Study 3: Speed-Critical Application

**Goal**: Maximize speedup

**Approach**:
```python
# Aggressive configuration
config = SketchWalkConfig(
    block_size=128,       # Low granularity
    sketch_dim=32,        # Fast projection
    top_k_blocks=8,       # High sparsity (~90%)
    sparsity_exponent=16, # Strong sharpening
)

# Expected results
# Sparsity: ~90%
# Speedup: ~4-6x
# Accuracy loss: 5-10%
```

---

## 9. Recommendations Summary

### 9.1 Quick Start Configurations

| Scenario | B | k | τ | s | Expected Sparsity | Expected Speedup |
|----------|---|---|---|---|-------------------|------------------|
| Default | 64 | 64 | 16 | 8 | ~80% | ~2-3x |
| High Quality | 32 | 128 | 32 | 4 | ~50% | ~1.5-2x |
| High Speed | 128 | 32 | 8 | 16 | ~90% | ~4-6x |
| Long Context | 128 | 128 | 32 | 8 | ~75% | ~3-4x |

### 9.2 Decision Tree

```
Is memory limited?
├─ Yes: Use B=128, k=32
└─ No: Is accuracy critical?
    ├─ Yes: Use B=32, k=128, τ=32, s=4
    └─ No: Is speed critical?
        ├─ Yes: Use B=128, k=32, τ=8, s=16
        └─ No: Use default (B=64, k=64, τ=16, s=8)
```

### 9.3 Validation Checklist

Before deploying a configuration:
- [ ] Validate block count: b = ceil(n/B)
- [ ] Check sparsity level: 1 - (τ·B)/n
- [ ] Verify sketch dimension: k ≥ log₂(b)
- [ ] Test numerical stability (no NaN/Inf)
- [ ] Benchmark speedup on target hardware
- [ ] Measure memory usage
- [ ] Validate accuracy on target task
- [ ] Test edge cases (min/max seq length)

---

## 10. Future Work

### 10.1 Adaptive Hyperparameters

**Dynamic τ Selection**:
```python
def adaptive_tau_selection(walk_state, target_sparsity):
    """Dynamically select τ to achieve target sparsity"""
    # Sort walk state values
    sorted_values = torch.sort(walk_state.flatten()).values

    # Find threshold
    threshold_idx = int(target_sparsity * len(sorted_values))
    threshold = sorted_values[threshold_idx]

    # Count blocks above threshold
    tau = (walk_state > threshold).sum(dim=-1)
    return tau
```

**Layer-wise Sparsity Schedule**:
```python
def sparsity_schedule(layer_idx, num_layers):
    """Increase sparsity with depth"""
    base_sparsity = 0.5
    max_sparsity = 0.9

    progress = layer_idx / num_layers
    sparsity = base_sparsity + (max_sparsity - base_sparsity) * progress
    return sparsity
```

### 10.2 Learned Hyperparameters

**Meta-Learning Approach**:
```python
class HyperparameterNetwork(nn.Module):
    """Learn hyperparameters from task features."""

    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),  # [B, k, τ, s]
        )

    def forward(self, task_features):
        """Predict optimal hyperparameters."""
        params = self.fc(task_features)
        B = 2 ** (5 + torch.sigmoid(params[0]) * 3)  # 32-256
        k = 16 + torch.sigmoid(params[1]) * 112      # 16-128
        tau = 4 + torch.sigmoid(params[2]) * 28      # 4-32
        s = 2 ** (1 + torch.sigmoid(params[3]) * 3)  # 2-16
        return B, k, tau, s
```

---

**End of Guide**
