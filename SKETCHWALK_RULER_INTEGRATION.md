# SketchWalk RULER Integration - Summary

## Overview

SketchWalk has been successfully integrated into the RULER evaluation framework for testing with real data. The implementation follows the same pattern as SeerAttention's prefill_sparse approach for LLaMA models.

## Created Files

### 1. Model Wrapper
**File**: `/home/valery/sketch_walk/SeerAttention/eval/ruler/pred/model_wrappers.py`

Added `SketchWalkModel` class that:
- Loads SketchWalk-enabled LLaMA models
- Handles prefill with sparse attention
- Supports the RULER API interface
- Configurable block_size, sketch_dim, top_k_blocks, sparsity_exponent
- Option to skip sketch_walk during decode (skip_decode)

### 2. RULER API Integration
**File**: `/home/valery/sketch_walk/SeerAttention/eval/ruler/pred/call_api.py`

Modified to:
- Add "SketchWalk" as a server type option
- Add command-line arguments for SketchWalk parameters
- Integrate with get_llm() function

### 3. Data Generation Script
**File**: `/home/valery/sketch_walk/SeerAttention/eval/ruler/data/synthetic/generate_niah_single_1.py`

Generates niah_single_1 test data:
- Creates 3-5 test samples (configurable)
- Uses repeat pattern for haystack
- Places needles at different depths (10%, 30%, 50%, 70%, 90%)
- Outputs RULER-compatible JSONL format

### 4. Evaluation Script
**File**: `/home/valery/sketch_walk/SeerAttention/eval/ruler/pred/eval_sketchwalk.py`

Standalone evaluation script that:
- Loads SketchWalk model
- Runs niah_single_1 evaluation
- Measures accuracy (all/part match)
- Records timing information
- Outputs results in multiple formats (JSON, JSONL, CSV)

### 5. Comparison Script
**File**: `/home/valery/sketch_walk/SeerAttention/eval/ruler/pred/compare_results.py`

Compares SeerAttention vs SketchWalk:
- Shows accuracy comparison
- Shows performance comparison (speed, timing)
- Generates summary report (Markdown and JSON)

### 6. Documentation
**File**: `/home/valery/sketch_walk/SeerAttention/eval/ruler/README_SKETCHWALK.md`

Complete guide with:
- Step-by-step instructions
- Parameter explanations
- Troubleshooting guide
- Example commands

## Quick Start

### Step 1: Generate Test Data

```bash
cd /home/valery/sketch_walk/SeerAttention

python eval/ruler/data/synthetic/generate_niah_single_1.py \
    --save_dir ./eval/ruler/data \
    --num_samples 5 \
    --max_seq_length 4096
```

### Step 2: Run Evaluation

```bash
python eval/ruler/pred/eval_sketchwalk.py \
    --model_path ./checkpoints/sketchwalk_llama \
    --data_dir ./eval/ruler/data \
    --num_samples 5 \
    --output_dir ./results/sketchwalk
```

### Step 3: Compare Results

```bash
python eval/ruler/pred/compare_results.py \
    --seerattn_dir ./results/seerattn \
    --sketchwalk_dir ./results/sketchwalk \
    --output_dir ./results/comparison
```

## Key Features

### 1. Prefill-Only Testing
The evaluation focuses on prefill (processing the input context) rather than decode/generation. This tests SketchWalk's sparse attention mechanism on long contexts.

### 2. Real RULER Data
Uses the same data format and evaluation metrics as RULER:
- niah_single_1 task (Needle In A Haystack)
- String match accuracy metric
- JSONL format for compatibility

### 3. Configurable Sparsity
SketchWalk parameters can be adjusted:
- `block_size`: 64 (default)
- `sketch_dim`: 64 (default)
- `top_k_blocks`: 16 (default)
- `sparsity_exponent`: 8 (default)

### 4. Skip Decode Option
The `skip_decode` flag allows testing prefill-only performance:
```bash
python eval/ruler/pred/eval_sketchwalk.py \
    --model_path ./checkpoints/sketchwalk_llama \
    --skip_decode \
    --data_dir ./eval/ruler/data
```

## Expected Results

With proper implementation:
- **Accuracy**: 90-100% on niah_single_1 (comparable to dense attention)
- **Performance**: Faster than dense attention for long sequences
- **Memory**: Lower memory usage due to sparse attention

## Integration Points

The implementation integrates with existing SeerAttention components:

1. **SketchWalk Model**: Uses `/home/valery/sketch_walk/SeerAttention/sketch_walk/llama/modeling_llama_sketchwalk.py`
2. **Core Module**: Uses `/home/valery/sketch_walk/SeerAttention/sketch_walk/common/core.py`
3. **RULER Framework**: Follows the pattern in `seer_attn/prefill_sparse/llama/`

## File Locations

All files are in `/home/valery/sketch_walk/SeerAttention/`:

```
eval/ruler/
├── data/synthetic/
│   └── generate_niah_single_1.py          # Data generation
├── pred/
│   ├── model_wrappers.py                  # SketchWalkModel (added)
│   ├── call_api.py                        # RULER API (modified)
│   ├── eval_sketchwalk.py                 # Evaluation script (new)
│   └── compare_results.py                 # Comparison script (new)
└── README_SKETCHWALK.md                   # Documentation (new)
```

## Next Steps

1. **Generate test data**: Run the data generation script
2. **Prepare model checkpoint**: Ensure SketchWalk model checkpoint exists
3. **Run evaluation**: Execute the evaluation script
4. **Analyze results**: Use the comparison script to compare with baseline

## Troubleshooting

If the model doesn't load:
- Check that the checkpoint path is correct
- Verify that SketchWalk model files exist
- Ensure dependencies are installed

If accuracy is low:
- Try less aggressive sparsity (increase top_k_blocks)
- Check that the model checkpoint is properly initialized
- Verify the data format is correct

For more details, see `eval/ruler/README_SKETCHWALK.md`.
