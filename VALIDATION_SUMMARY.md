# Tensor Collection and Validation System - Implementation Summary

## Overview

I've created a comprehensive tensor collection and validation system for SketchWalk that allows you to:

1. **Collect real Q, K, V tensors** from SeerAttention model runs during inference
2. **Validate SketchWalk accuracy** on those real tensors
3. **Compare sparsity patterns** between SeerAttention (learned sparse) and SketchWalk (training-free sparse)

## Created Files

### Core Scripts

#### 1. `collect_seerattn_tensors.py`
**Purpose:** Collect Q, K, V tensors from SeerAttention model runs

**Key Features:**
- Hooks into SeerAttention model layers to capture intermediate tensors
- Supports multiple sequence lengths (4K-16K+)
- Handles both SeerAttention and standard LLaMA models
- Saves tensors with metadata for later analysis
- Includes basic sparsity analysis

**Usage:**
```bash
python collect_seerattn_tensors.py \
    --model_path /path/to/model \
    --output_dir ./tensor_data \
    --sequence_lengths 4096 8192 16384 \
    --num_samples 5 \
    --device cuda
```

#### 2. `validate_sketchwalk_tensors.py`
**Purpose:** Validate SketchWalk accuracy on collected tensors

**Key Features:**
- Loads collected Q, K, V tensors
- Computes both dense and sparse attention
- Measures accuracy metrics (cosine similarity, MSE, max error)
- Computes actual vs target sparsity
- Measures performance (timing, speedup)
- Generates detailed validation reports

**Usage:**
```bash
python validate_sketchwalk_tensors.py \
    --tensor_dir ./tensor_data \
    --output_dir ./validation_results \
    --block_size 64 \
    --sketch_dim 64 \
    --top_k_blocks 16 \
    --device cuda
```

#### 3. `compare_sparsity_patterns.py`
**Purpose:** Compare SeerAttention vs SketchWalk sparsity patterns

**Key Features:**
- Computes block selection for both methods
- Measures block overlap and Jaccard similarity
- Analyzes attention score correlations
- Groups results by sequence length and layer
- Generates comparison summaries

**Usage:**
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

### Supporting Files

#### 4. `test_tensor_collection.py`
**Purpose:** Test the validation pipeline setup

**Key Features:**
- Tests SketchWalk imports
- Tests forward pass with synthetic data
- Tests dense vs sparse comparison
- Tests tensor saving/loading
- Comprehensive test suite

**Usage:**
```bash
python test_tensor_collection.py --device cuda
```

#### 5. `run_validation_pipeline.sh`
**Purpose:** Automated end-to-end validation pipeline

**Key Features:**
- Runs all validation steps in sequence
- Configurable via command-line args
- Skips steps if requested
- Creates organized output structure
- Provides progress feedback

**Usage:**
```bash
MODEL_PATH=/path/to/model ./run_validation_pipeline.sh --device cuda
```

#### 6. `TENSOR_VALIDATION_README.md`
**Purpose:** Comprehensive documentation

**Contents:**
- Detailed usage instructions
- Parameter explanations
- Expected results
- Troubleshooting guide
- File format specifications
- Advanced usage examples

## How It Works

### Data Flow

```
SeerAttention Model
    ↓ (Forward Hook)
Q, K, V Tensors
    ↓ (Save to Disk)
Tensor Files (.pt)
    ↓ (Load)
SketchWalk Validation
    ↓ (Compare)
Validation Results (.json)
    ↓ (Analyze)
Comparison Results (.json)
```

### Tensor Collection Process

1. **Model Loading:** Loads SeerAttention or standard LLaMA model
2. **Hook Registration:** Registers forward hooks on transformer layers
3. **Inference:** Runs model on synthetic or real data
4. **Tensor Capture:** Hooks capture Q, K, V projections during forward pass
5. **Saving:** Saves tensors with metadata to disk

### Validation Process

1. **Loading:** Loads collected Q, K, V tensors
2. **Dense Computation:** Computes full attention (baseline)
3. **Sparse Computation:** Computes SketchWalk sparse attention
4. **Comparison:** Measures accuracy, sparsity, performance
5. **Reporting:** Saves detailed validation results

### Comparison Process

1. **Pattern Extraction:** Extracts block selections from both methods
2. **Similarity Metrics:** Computes overlap, Jaccard, correlation
3. **Grouping Analysis:** Groups by sequence length and layer
4. **Summary:** Generates comparison summaries

## Key Metrics

### Accuracy Metrics
- **Cosine Similarity:** Measures output vector similarity (>0.95 is good)
- **MSE:** Mean squared error (<0.01 is good)
- **Max Error:** Maximum absolute error (<0.1 is good)

### Sparsity Metrics
- **Actual Sparsity:** Measured sparsity level
- **Target Sparsity:** Theoretical sparsity from config
- **Block Sparsity:** Block-level sparsity

### Performance Metrics
- **Dense Time:** Time for full attention (ms)
- **Sparse Time:** Time for sparse attention (ms)
- **Speedup:** Dense time / Sparse time

### Pattern Comparison Metrics
- **Block Overlap:** Fraction of blocks both methods select
- **Jaccard Similarity:** |A ∩ B| / |A ∪ B|
- **Attention Correlation:** Correlation between attention patterns

## Expected Results

For LLaMA 3.1-8B-Instruct with 4K-16K sequences:

| Metric | Expected Range |
|--------|---------------|
| Cosine Similarity | 0.95 - 0.99 |
| MSE | 0.001 - 0.01 |
| Actual Sparsity | 0.75 - 0.85 |
| Speedup (8K) | 1.5 - 3x |
| Speedup (16K) | 2 - 5x |
| Block Overlap | 0.3 - 0.5 |

## Quick Start

### Step 1: Test Setup
```bash
python test_tensor_collection.py --device cuda
```

### Step 2: Collect Tensors
```bash
python collect_seerattn_tensors.py \
    --model_path /path/to/model \
    --output_dir ./tensor_data \
    --sequence_lengths 4096 8192 \
    --num_samples 3
```

### Step 3: Validate SketchWalk
```bash
python validate_sketchwalk_tensors.py \
    --tensor_dir ./tensor_data \
    --output_dir ./validation_results
```

### Step 4: Compare Patterns
```bash
python compare_sparsity_patterns.py \
    --tensor_dir ./tensor_data \
    --validation_dir ./validation_results \
    --output_dir ./comparison_results
```

### Or Use the Automated Pipeline
```bash
MODEL_PATH=/path/to/model ./run_validation_pipeline.sh
```

## Troubleshooting

### No Tensors Collected
- Check if model has the expected layer structure
- Verify hooks are registered correctly
- Add debug logging to trace hook execution

### Out of Memory
- Reduce sequence lengths
- Reduce number of samples
- Use CPU instead of GPU
- Clear GPU cache between runs

### Low Accuracy
- Adjust block_size (try 32, 64, 128)
- Adjust top_k_blocks for target sparsity
- Increase sketch_dim (try 128)
- Verify causal masking is applied

### Import Errors
- Ensure running from SeerAttention root
- Install SketchWalk: `pip install -e .`
- Check that `sketch_walk/common/core.py` exists

## File Locations

All scripts are in `/home/valery/sketch_walk/SeerAttention/`:

- `collect_seerattn_tensors.py` - Main collection script
- `validate_sketchwalk_tensors.py` - Validation script
- `compare_sparsity_patterns.py` - Comparison script
- `test_tensor_collection.py` - Test suite
- `run_validation_pipeline.sh` - Automated pipeline
- `TENSOR_VALIDATION_README.md` - Documentation
- `VALIDATION_SUMMARY.md` - This file

## Integration with RULER

To use with RULER evaluation:

```bash
# First run RULER evaluation
cd eval/ruler
python pred/call_api.py \
    --server_type SeerAttn \
    --model_name_or_path /path/to/model \
    --task niah

# Then validate on collected tensors
cd ../../
python validate_sketchwalk_tensors.py \
    --tensor_dir ./tensor_data \
    --output_dir ./validation_results
```

## Next Steps

1. **Test with Real Model:** Run the pipeline with your actual model
2. **Analyze Results:** Review validation metrics and adjust parameters
3. **Extended Testing:** Try longer sequences (16K, 32K)
4. **Integration:** Integrate with your evaluation workflow

## Contact

For issues or questions about the validation system, please refer to:
- `TENSOR_VALIDATION_README.md` for detailed documentation
- Test output in `test_tensor_collection.py` for debugging
- Validation result files for detailed metrics
