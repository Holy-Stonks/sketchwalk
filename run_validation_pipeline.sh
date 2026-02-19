#!/bin/bash
# Quick-Start Script for Tensor Validation Pipeline
#
# This script runs the complete validation pipeline:
# 1. Test the setup
# 2. Collect tensors (optional)
# 3. Validate SketchWalk
# 4. Compare sparsity patterns

set -e  # Exit on error

# Configuration
MODEL_PATH="${MODEL_PATH:-/path/to/your/model}"
OUTPUT_DIR="${OUTPUT_DIR:-./validation_data}"
DEVICE="${DEVICE:-cuda}"

# Parse command line arguments
SKIP_COLLECTION=false
SKIP_VALIDATION=false
SKIP_COMPARISON=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --skip-collection)
            SKIP_COLLECTION=true
            shift
            ;;
        --skip-validation)
            SKIP_VALIDATION=true
            shift
            ;;
        --skip-comparison)
            SKIP_COMPARISON=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model_path PATH       Path to SeerAttention model"
            echo "  --output_dir PATH       Output directory for results"
            echo "  --device DEVICE         Device to use (cuda/cpu)"
            echo "  --skip-collection       Skip tensor collection"
            echo "  --skip-validation       Skip SketchWalk validation"
            echo "  --skip-comparison       Skip sparsity pattern comparison"
            echo "  --help                  Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  MODEL_PATH              Path to SeerAttention model"
            echo "  OUTPUT_DIR              Output directory for results"
            echo "  DEVICE                  Device to use (cuda/cpu)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Tensor Validation Pipeline"
echo "=============================================="
echo "Model path: $MODEL_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "=============================================="
echo ""

# Step 1: Test setup
echo "Step 1: Testing setup..."
echo "=============================================="

if ! python test_tensor_collection.py --device "$DEVICE"; then
    echo "ERROR: Setup test failed. Please fix the issues before proceeding."
    exit 1
fi

echo ""
echo "✓ Setup test passed!"
echo ""

# Step 2: Collect tensors
if [ "$SKIP_COLLECTION" = false ]; then
    echo "Step 2: Collecting tensors from SeerAttention..."
    echo "=============================================="

    TENSOR_DIR="$OUTPUT_DIR/tensor_data"

    python collect_seerattn_tensors.py \
        --model_path "$MODEL_PATH" \
        --output_dir "$TENSOR_DIR" \
        --sequence_lengths 4096 8192 \
        --num_samples 3 \
        --device "$DEVICE"

    echo ""
    echo "✓ Tensor collection complete!"
    echo ""
else
    echo "Skipping tensor collection (--skip-collection flag set)"
    TENSOR_DIR="$OUTPUT_DIR/tensor_data"
    echo ""
fi

# Step 3: Validate SketchWalk
if [ "$SKIP_VALIDATION" = false ]; then
    echo "Step 3: Validating SketchWalk on collected tensors..."
    echo "=============================================="

    VALIDATION_DIR="$OUTPUT_DIR/validation_results"

    python validate_sketchwalk_tensors.py \
        --tensor_dir "$TENSOR_DIR" \
        --output_dir "$VALIDATION_DIR" \
        --block_size 64 \
        --sketch_dim 64 \
        --top_k_blocks 16 \
        --device "$DEVICE"

    echo ""
    echo "✓ SketchWalk validation complete!"
    echo ""
else
    echo "Skipping SketchWalk validation (--skip-validation flag set)"
    VALIDATION_DIR="$OUTPUT_DIR/validation_results"
    echo ""
fi

# Step 4: Compare sparsity patterns
if [ "$SKIP_COMPARISON" = false ]; then
    echo "Step 4: Comparing SeerAttention vs SketchWalk sparsity patterns..."
    echo "=============================================="

    COMPARISON_DIR="$OUTPUT_DIR/comparison_results"

    python compare_sparsity_patterns.py \
        --tensor_dir "$TENSOR_DIR" \
        --validation_dir "$VALIDATION_DIR" \
        --output_dir "$COMPARISON_DIR" \
        --block_size 64 \
        --sketch_dim 64 \
        --top_k_blocks 16 \
        --seerattn_threshold 0.1 \
        --device "$DEVICE"

    echo ""
    echo "✓ Sparsity pattern comparison complete!"
    echo ""
else
    echo "Skipping sparsity pattern comparison (--skip-comparison flag set)"
    echo ""
fi

# Final summary
echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Results:"
echo "  - Tensor data: $TENSOR_DIR"
echo "  - Validation results: $VALIDATION_DIR"
echo "  - Comparison results: $COMPARISON_DIR"
echo ""
echo "Next steps:"
echo "  1. Review the validation results"
echo "  2. Check the summary JSON files"
echo "  3. Adjust parameters if needed"
echo "  4. Run with longer sequences if desired"
echo "=============================================="
