# Quick Start Guide - Tensor Validation Pipeline

## One-Line Setup Test

```bash
python test_tensor_collection.py --device cuda
```

## Complete Pipeline (3 Steps)

### 1. Collect Tensors
```bash
python collect_seerattn_tensors.py \
    --model_path /path/to/model \
    --output_dir ./tensor_data \
    --sequence_lengths 4096 8192
```

### 2. Validate SketchWalk
```bash
python validate_sketchwalk_tensors.py \
    --tensor_dir ./tensor_data
```

### 3. Compare Patterns
```bash
python compare_sparsity_patterns.py \
    --tensor_dir ./tensor_data \
    --validation_dir ./validation_results
```

## Or Use Automated Script

```bash
MODEL_PATH=/path/to/model ./run_validation_pipeline.sh
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--block_size` | 64 | Tokens per block |
| `--sketch_dim` | 64 | Sketch dimension |
| `--top_k_blocks` | 16 | Blocks to select |
| `--sequence_lengths` | 4096 8192 16384 | Seq lengths to test |

## Expected Results

- Cosine Similarity: > 0.95
- Sparsity: ~0.8
- Speedup: 1.5-5x (depends on seq length)

## Troubleshooting

| Problem | Solution |
|---------|----------|
| OOM | Reduce `--sequence_lengths` or use CPU |
| Low accuracy | Try `--block_size 32` or `--sketch_dim 128` |
| Import error | Run from SeerAttention root directory |

## Documentation

- `TENSOR_VALIDATION_README.md` - Full documentation
- `VALIDATION_SUMMARY.md` - Implementation details
- `test_tensor_collection.py` - Test suite
