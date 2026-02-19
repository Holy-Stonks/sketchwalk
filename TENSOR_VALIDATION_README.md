# Tensor Collection and Validation for SketchWalk

This directory contains scripts for collecting real tensors from SeerAttention runs and validating SketchWalk sparse attention patterns.

## Overview

The validation pipeline consists of three main scripts:

1. **collect_seerattn_tensors.py** - Collects Q, K, V tensors from SeerAttention model runs
2. **validate_sketchwalk_tensors.py** - Validates SketchWalk accuracy on collected tensors
3. **compare_sparsity_patterns.py** - Compares SeerAttention vs SketchWalk sparsity patterns

## Setup

### Prerequisites

```bash
# Install dependencies
pip install torch transformers tqdm numpy

# Ensure SeerAttention is available
cd /path/to/SeerAttention
pip install -e .
```

### Hardware Requirements

- NVIDIA GPU with at least 16GB VRAM recommended
- For 16K+ sequences, 24GB+ VRAM recommended
- CPU fallback available but will be slow

## Usage

### Step 1: Collect Tensors from SeerAttention

Run SeerAttention model and collect Q, K, V tensors during inference:

```bash
python collect_seerattn_tensors.py \
    --model_path /path/to/seerattn/model \
    --output_dir ./tensor_data \
    --sequence_lengths 4096 8192 16384 \
    --num_samples 5 \
    --device cuda
```

**Parameters:**
- `--model_path`: Path to SeerAttention model checkpoint
- `--output_dir`: Directory to save collected tensors
- `--sequence_lengths`: List of sequence lengths to test (default: 4096 8192 16384)
- `--num_samples`: Number of samples per sequence length (default: 5)
- `--device`: Device to run on (cuda/cpu)

**Output:**
- Tensor files: `tensors_seq{length}_layer{idx}_sample{id}_{timestamp}.pt`
- Summary files: `summary_seq{length}_{timestamp}.json`
- Analysis logs with sparsity statistics

**Example Output:**
```
tensor_data/
├── tensors_seq4096_layer0_sample0_20240219_120000.pt
├── tensors_seq4096_layer8_sample1_20240219_120000.pt
├── summary_seq4096_20240219_120000.json
└── summary_seq8192_20240219_120500.json
```

### Step 2: Validate SketchWalk on Collected Tensors

Run SketchWalk on the collected tensors and validate accuracy:

```bash
python validate_sketchwalk_tensors.py \
    --tensor_dir ./tensor_data \
    --output_dir ./validation_results \
    --block_size 64 \
    --sketch_dim 64 \
    --top_k_blocks 16 \
    --device cuda
```

**Parameters:**
- `--tensor_dir`: Directory containing collected tensor files
- `--output_dir`: Directory to save validation results
- `--block_size`: Block size for SketchWalk (default: 64)
- `--sketch_dim`: Sketch dimension (default: 64)
- `--top_k_blocks`: Number of top blocks to select (default: 16)
- `--sparsity_exponent`: Sparsity exponent for sharpening (default: 8)

**Output:**
- Validation results: `validation_results_{timestamp}.json`
- Summary statistics: `validation_summary_{timestamp}.json`
- Console output with metrics

**Example Output:**
```
validation_results/
├── validation_results_20240219_130000.json
└── validation_summary_20240219_130000.json
```

**Metrics:**
- **Accuracy:** Cosine similarity, MSE, max error between dense and sparse
- **Sparsity:** Actual vs target sparsity levels
- **Performance:** Dense vs sparse timing, speedup
- **Block Selection:** Number of blocks selected, selection ratio

### Step 3: Compare SeerAttention vs SketchWalk

Compare sparsity patterns between SeerAttention (learned) and SketchWalk (training-free):

```bash
python compare_sparsity_patterns.py \
    --tensor_dir ./tensor_data \
    --validation_dir ./validation_results \
    --output_dir ./comparison_results \
    --block_size 64 \
    --sketch_dim 64 \
    --top_k_blocks 16 \
    --seerattn_threshold 0.1 \
    --device cuda
```

**Parameters:**
- `--tensor_dir`: Directory containing collected tensor files
- `--validation_dir`: Directory containing validation results
- `--output_dir`: Directory to save comparison results
- `--seerattn_threshold`: Threshold for SeerAttention sparsity (default: 0.1)

**Output:**
- Comparison results: `comparison_results_{timestamp}.json`
- Summary statistics: `comparison_summary_{timestamp}.json`
- Detailed analysis by sequence length and layer

**Metrics:**
- **Sparsity Comparison:** SeerAttn vs SketchWalk sparsity levels
- **Block Overlap:** How many blocks both methods select
- **Jaccard Similarity:** Block selection similarity metric
- **Attention Correlation:** Correlation between attention patterns

## Expected Results

### For LLaMA 3.1-8B-Instruct with 4K-16K sequences:

**Sparsity Levels:**
- SeerAttention: ~0.7-0.9 sparsity (threshold-dependent)
- SketchWalk: ~0.8 sparsity (with default config)

**Accuracy Metrics:**
- Cosine Similarity: > 0.95
- MSE: < 0.01
- Max Error: < 0.1

**Block Selection:**
- Block Overlap: 0.3-0.5 (moderate overlap)
- Jaccard Similarity: 0.2-0.4

**Performance:**
- Speedup: 1.5-3x for 8K sequences
- Speedup: 2-5x for 16K sequences

## Troubleshooting

### Issue: No tensors collected

**Cause:** Forward hooks may not work with your model architecture.

**Solutions:**
1. Check if your model has the expected layer structure
2. Modify the `create_forward_hook` method in `collect_seerattn_tensors.py`
3. Add debug logging to verify hook registration

### Issue: Out of memory errors

**Cause:** Sequence length too large for GPU memory.

**Solutions:**
1. Reduce `--sequence_lengths` (e.g., try 2048 4096 instead of 8192 16384)
2. Reduce `--num_samples`
3. Use CPU with `--device cpu` (will be slower)
4. Clear GPU cache between runs

### Issue: Low accuracy results

**Cause:** SketchWalk configuration may not match the data.

**Solutions:**
1. Try different `--block_size` values (32, 64, 128)
2. Adjust `--top_k_blocks` for target sparsity
3. Increase `--sketch_dim` for better accuracy (e.g., 128)
4. Check that causal masking is applied correctly

### Issue: SketchWalk import errors

**Cause:** SketchWalk module not in Python path.

**Solutions:**
1. Ensure you're running from SeerAttention root directory
2. Install SketchWalk: `pip install -e .`
3. Check that `sketch_walk/common/core.py` exists

## Advanced Usage

### Custom Data Collection

To collect tensors from your own data:

```python
from collect_seerattn_tensors import TensorCollector

collector = TensorCollector(
    model_path="/path/to/model",
    output_dir="./my_tensors",
    device="cuda"
)

collector.load_model()

# Your custom forward pass
input_ids = tokenizer("Your text here", return_tensors="pt").input_ids
with torch.no_grad():
    outputs = collector.model(input_ids)
```

### Batch Processing

Process multiple models or configurations:

```bash
# Collect from multiple models
for model in model1 model2 model3; do
    python collect_seerattn_tensors.py \
        --model_path $model \
        --output_dir ./tensor_data_$model
done

# Validate with different configurations
for bs in 32 64 128; do
    python validate_sketchwalk_tensors.py \
        --tensor_dir ./tensor_data \
        --output_dir ./results_bs$bs \
        --block_size $bs
done
```

### Integration with RULER Evaluation

To use with RULER evaluation scripts:

```bash
# First collect tensors during RULER eval
cd eval/ruler
python pred/call_api.py \
    --server_type SeerAttn \
    --model_name_or_path /path/to/model \
    --task niah \
    --data_dir ./data/niah \
    --save_dir ./predictions

# Then validate SketchWalk on collected tensors
cd ../../
python validate_sketchwalk_tensors.py \
    --tensor_dir ./tensor_data \
    --output_dir ./validation_results
```

## File Format

### Tensor Files (.pt)

```python
{
    'Q': torch.Tensor,  # Shape: (batch, num_heads, seq_len, head_dim)
    'K': torch.Tensor,  # Shape: (batch, num_kv_heads, seq_len, head_dim)
    'V': torch.Tensor,  # Shape: (batch, num_kv_heads, seq_len, head_dim)
    'metadata': {
        'layer_idx': int,
        'seq_len': int,
        'num_heads': int,
        'num_kv_heads': int,
        'head_dim': int,
        'hidden_size': int,
        'batch_size': int,
        'timestamp': str,
        'model_path': str,
    }
}
```

### Validation Results (.json)

```json
{
    "sample_id": "tensors_seq4096_layer0_sample0_20240219_120000",
    "layer_idx": 0,
    "seq_len": 4096,
    "num_heads": 32,
    "actual_sparsity": 0.8,
    "target_sparsity": 0.8,
    "block_sparsity": 0.75,
    "cosine_similarity": 0.98,
    "mse": 0.001,
    "max_error": 0.05,
    "num_blocks_selected": 16,
    "total_blocks": 64,
    "block_selection_ratio": 0.25,
    "dense_time_ms": 45.2,
    "sparse_time_ms": 15.8,
    "speedup": 2.86
}
```

## Citation

If you use these validation scripts, please cite:

```bibtex
@article{sketchwalk2024,
    title={Scout Before You Attend: Sketch-and-Walk Sparse Attention},
    author={...},
    year={2024}
}
```

## License

MIT License - See LICENSE file for details.
