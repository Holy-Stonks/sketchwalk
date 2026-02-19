# SketchWalk RULER Evaluation

This guide explains how to run RULER evaluation with SketchWalk sparse attention on the niah_single_1 (Needle In A Haystack) task.

## Overview

The evaluation consists of three main steps:

1. **Generate test data**: Create niah_single_1 test samples (3-5 cases)
2. **Run evaluation**: Evaluate SketchWalk model on the test data
3. **Compare results**: Compare with SeerAttention baseline

## Prerequisites

- Python 3.8+
- PyTorch with CUDA support
- Transformers library
- SketchWalk model checkpoint
- Required packages: `nltk`, `wonderwords`, `tqdm`

Install dependencies:
```bash
pip install transformers nltk wonderwords tqdm torch nemo-toolkit
```

## Step 1: Generate Test Data

Generate niah_single_1 test data with 5 samples:

```bash
cd /home/valery/sketch_walk/SeerAttention

python eval/ruler/data/synthetic/generate_niah_single_1.py \
    --save_dir ./eval/ruler/data \
    --num_samples 5 \
    --max_seq_length 4096 \
    --tokens_to_generate 128 \
    --tokenizer_path meta-llama/Llama-3.1-8B-Instruct \
    --tokenizer_type hf
```

This creates:
- `eval/ruler/data/niah_single_1/validation.jsonl` - Test data in RULER format

### Data Configuration

The niah_single_1 task tests "Needle In A Haystack" - finding a small piece of text (needle) hidden in a long context (haystack).

- **Haystack type**: `repeat` - repeated sentences for controlled testing
- **Needle key type**: `words` - random word identifiers
- **Needle value type**: `numbers` - random number values to find
- **Needle depths**: 10%, 30%, 50%, 70%, 90% (needle placed at different positions)

## Step 2: Run SketchWalk Evaluation

### Option A: Using the standalone evaluation script (Recommended)

```bash
cd /home/valery/sketch_walk/SeerAttention

python eval/ruler/pred/eval_sketchwalk.py \
    --model_path ./checkpoints/sketchwalk_llama \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --data_dir ./eval/ruler/data \
    --num_samples 5 \
    --block_size 64 \
    --sketch_dim 64 \
    --top_k_blocks 16 \
    --sparsity_exponent 8 \
    --max_new_tokens 128 \
    --output_dir ./results/sketchwalk
```

### Option B: Using the RULER call_api.py script

```bash
cd /home/valery/sketch_walk/SeerAttention

python eval/ruler/pred/call_api.py \
    --server_type SketchWalk \
    --model_name_or_path ./checkpoints/sketchwalk_llama \
    --data_dir ./eval/ruler/data \
    --save_dir ./results/sketchwalk \
    --task niah_single_1 \
    --subset validation \
    --benchmark synthetic \
    --block_size 64 \
    --sketch_dim 64 \
    --top_k_blocks 16 \
    --sparsity_exponent 8 \
    --max_new_tokens 128 \
    --temperature 0.0
```

### SketchWalk Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--block_size` | Block size B for token-space sketching | 64 |
| `--sketch_dim` | Sketch dimension k for feature-space sketching | 64 |
| `--top_k_blocks` | Number of key blocks τ selected per query block | 16 |
| `--sparsity_exponent` | Sparsity exponent s for sharpening attention | 8 |
| `--skip_decode` | Skip sketch_walk during decode stage | False |

### Output Files

The evaluation generates:

- `results/sketchwalk/niah_single_1_results.json` - Detailed results with timings
- `results/sketchwalk/niah_single_1.jsonl` - RULER-compatible format
- `results/sketchwalk/summary.csv` - Summary for RULER evaluation framework

## Step 3: Run SeerAttention Baseline (Optional)

To compare with SeerAttention, first run the baseline:

```bash
cd /home/valery/sketch_walk/SeerAttention

python eval/ruler/pred/call_api.py \
    --server_type SeerAttn \
    --model_name_or_path ./checkpoints/seerattn_llama \
    --data_dir ./eval/ruler/data \
    --save_dir ./results/seerattn \
    --task niah_single_1 \
    --subset validation \
    --benchmark synthetic \
    --threshold 0.1 \
    --max_new_tokens 128 \
    --temperature 0.0
```

## Step 4: Compare Results

Compare SeerAttention and SketchWalk results:

```bash
cd /home/valery/sketch_walk/SeerAttention

python eval/ruler/pred/compare_results.py \
    --seerattn_dir ./results/seerattn \
    --sketchwalk_dir ./results/sketchwalk \
    --output_dir ./results/comparison
```

This generates:
- `results/comparison/comparison_report.md` - Human-readable comparison
- `results/comparison/comparison.json` - Machine-readable comparison

## Understanding the Results

### Accuracy Metrics

Two accuracy metrics are reported:

1. **Accuracy (all needles)**: All reference values must be found in prediction (stricter)
2. **Accuracy (part needles)**: At least one reference value must be found (lenient)

For niah_single_1 with a single needle, both metrics are equivalent.

### Performance Metrics

- **Total time**: Wall-clock time for all samples
- **Average time per sample**: Average inference time
- **Speedup**: Performance improvement over baseline

### Expected Results

With proper SketchWalk implementation, you should see:

- Accuracy: Close to baseline (90-100% for niah_single_1)
- Speed: Comparable or faster than SeerAttention
- Memory: Lower memory usage due to sparse attention

## Troubleshooting

### Issue: "Data file not found"

**Solution**: Make sure you generated the test data first (Step 1).

### Issue: "Model checkpoint not found"

**Solution**: Update the `--model_path` to point to your SketchWalk checkpoint.

### Issue: Low accuracy (< 80%)

**Possible causes**:
1. SketchWalk parameters too aggressive (reduce `top_k_blocks` or `sparsity_exponent`)
2. Model checkpoint not properly loaded
3. Context length exceeds model's max position embeddings

### Issue: Out of memory

**Solutions**:
1. Reduce `--max_seq_length` in data generation
2. Reduce batch size (currently hardcoded to 1)
3. Use a smaller model variant

## Testing Different Configurations

### Test with prefill-only (skip decode)

```bash
python eval/ruler/pred/eval_sketchwalk.py \
    --model_path ./checkpoints/sketchwalk_llama \
    --data_dir ./eval/ruler/data \
    --skip_decode \
    --output_dir ./results/sketchwalk_prefill_only
```

### Test with different sparsity levels

```bash
# Lower sparsity (more accurate, slower)
python eval/ruler/pred/eval_sketchwalk.py \
    --model_path ./checkpoints/sketchwalk_llama \
    --data_dir ./eval/ruler/data \
    --top_k_blocks 32 \
    --sparsity_exponent 4 \
    --output_dir ./results/sketchwalk_low_sparsity

# Higher sparsity (faster, may be less accurate)
python eval/ruler/pred/eval_sketchwalk.py \
    --model_path ./checkpoints/sketchwalk_llama \
    --data_dir ./eval/ruler/data \
    --top_k_blocks 8 \
    --sparsity_exponent 16 \
    --output_dir ./results/sketchwalk_high_sparsity
```

## Advanced: Running Full RULER Evaluation

To run the full RULER evaluation suite with multiple tasks:

```bash
cd /home/valery/sketch_walk/SeerAttention

# Generate data for all tasks (requires more setup)
python eval/ruler/data/prepare.py --config_file eval/ruler/synthetic.yaml

# Run evaluation for all tasks
python eval/ruler/pred/call_api.py \
    --server_type SketchWalk \
    --model_name_or_path ./checkpoints/sketchwalk_llama \
    --data_dir ./eval/ruler/data \
    --save_dir ./results/sketchwalk_full \
    --task niah_single_1 \
    --benchmark synthetic

# Evaluate results
python eval/ruler/eval/evaluate.py \
    --data_dir ./results/sketchwalk_full \
    --benchmark synthetic
```

## File Structure

```
eval/ruler/
├── data/
│   ├── synthetic/
│   │   └── generate_niah_single_1.py     # Data generation script
│   └── niah_single_1/
│       └── validation.jsonl              # Generated test data
├── pred/
│   ├── model_wrappers.py                 # Model wrappers (includes SketchWalkModel)
│   ├── call_api.py                       # RULER API script (supports SketchWalk)
│   ├── eval_sketchwalk.py                # Standalone evaluation script
│   └── compare_results.py                # Comparison script
└── README_SKETCHWALK.md                  # This file
```

## References

- RULER paper: [RULER: What's the Real Context Size of Your Long-Context LLMs?](https://arxiv.org/abs/2406.18563)
- SeerAttention: Training-free sparse attention for LLMs
- SketchWalk: Scout Before You Attend - Sketch-and-Walk Sparse Attention

## Contact

For issues or questions about SketchWalk RULER integration, please refer to the main SeerAttention repository.
