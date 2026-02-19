#!/usr/bin/env python3
"""
Evaluate SketchWalk model on RULER niah_single_1 task.

This script runs the niah_single_1 evaluation with SketchWalk enabled,
measuring accuracy, timing, and sparsity information.

Usage:
    python eval_sketchwalk.py --model_path ./checkpoints/sketchwalk_llama --data_dir ./data --num_samples 5
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

# Import model_wrappers from the same directory
import importlib.util
spec = importlib.util.spec_from_file_location(
    "model_wrappers",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_wrappers.py")
)
model_wrappers = importlib.util.module_from_spec(spec)
sys.modules["model_wrappers"] = model_wrappers
spec.loader.exec_module(model_wrappers)

SketchWalkModel = model_wrappers.SketchWalkModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SketchWalk on RULER niah_single_1")

    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to SketchWalk model checkpoint"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Base model name for tokenizer loading"
    )

    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to data directory containing niah_single_1/validation.jsonl"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to evaluate (default: 5)"
    )

    # SketchWalk arguments
    parser.add_argument("--block_size", type=int, default=64, help="SketchWalk block size")
    parser.add_argument("--sketch_dim", type=int, default=64, help="SketchWalk sketch dimension")
    parser.add_argument("--top_k_blocks", type=int, default=16, help="SketchWalk top-k blocks")
    parser.add_argument("--sparsity_exponent", type=int, default=8, help="SketchWalk sparsity exponent")
    parser.add_argument("--skip_decode", action="store_true", help="Skip sketch_walk during decode")

    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--top_k", type=int, default=1, help="Top-k for sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p for sampling")

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/sketchwalk",
        help="Output directory for results"
    )

    return parser.parse_args()


def load_data(data_dir: str, num_samples: int = None) -> List[Dict]:
    """Load niah_single_1 test data."""
    data_file = Path(data_dir) / "niah_single_1" / "validation.jsonl"

    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    data = []
    with open(data_file, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
            if num_samples and len(data) >= num_samples:
                break

    print(f"Loaded {len(data)} samples from {data_file}")
    return data


def evaluate_accuracy(predictions: List[str], references: List[List[str]]) -> float:
    """
    Evaluate accuracy using string match (all needles must be found).

    Args:
        predictions: List of predicted strings
        references: List of reference answer lists

    Returns:
        Accuracy score (0-100)
    """
    correct = 0
    for pred, ref in zip(predictions, references):
        # Check if all reference strings are in the prediction
        all_found = all(r.lower() in pred.lower() for r in ref)
        if all_found:
            correct += 1

    accuracy = (correct / len(predictions)) * 100 if predictions else 0.0
    return accuracy


def evaluate_accuracy_part(predictions: List[str], references: List[List[str]]) -> float:
    """
    Evaluate accuracy using string match (at least one needle must be found).

    Args:
        predictions: List of predicted strings
        references: List of reference answer lists

    Returns:
        Accuracy score (0-100)
    """
    correct = 0
    for pred, ref in zip(predictions, references):
        # Check if at least one reference string is in the prediction
        any_found = any(r.lower() in pred.lower() for r in ref)
        if any_found:
            correct += 1

    accuracy = (correct / len(predictions)) * 100 if predictions else 0.0
    return accuracy


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SketchWalk RULER Evaluation")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_dir}")
    print(f"Samples: {args.num_samples}")
    print(f"SketchWalk Config:")
    print(f"  Block Size: {args.block_size}")
    print(f"  Sketch Dim: {args.sketch_dim}")
    print(f"  Top-K Blocks: {args.top_k_blocks}")
    print(f"  Sparsity Exponent: {args.sparsity_exponent}")
    print(f"  Skip Decode: {args.skip_decode}")
    print("=" * 60)

    # Load data
    data = load_data(args.data_dir, args.num_samples)

    # Load model
    print("\nLoading SketchWalk model...")
    model_start = time.time()
    model = SketchWalkModel(
        name_or_path=args.model_path,
        block_size=args.block_size,
        sketch_dim=args.sketch_dim,
        top_k_blocks=args.top_k_blocks,
        sparsity_exponent=args.sparsity_exponent,
        skip_decode=args.skip_decode,
        do_sample=args.temperature > 0,
        repetition_penalty=1,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        stop=[],
        max_new_tokens=args.max_new_tokens,
    )
    model_load_time = time.time() - model_start
    print(f"Model loaded in {model_load_time:.2f} seconds")

    # Run evaluation
    print("\nRunning evaluation...")
    predictions = []
    references = []
    timings = []

    results = []

    for sample in tqdm(data, desc="Evaluating"):
        input_text = sample["input"]
        reference = sample["outputs"]

        # Time the prediction
        start_time = time.time()
        output = model(prompt=input_text)
        end_time = time.time()

        prediction = output["text"][0]
        pred_time = end_time - start_time

        predictions.append(prediction)
        references.append(reference)
        timings.append(pred_time)

        # Store result
        result = {
            "index": sample["index"],
            "input_length": len(input_text),
            "prediction": prediction,
            "reference": reference,
            "time": pred_time,
            "depth": sample.get("depth", "N/A"),
        }
        results.append(result)

        print(f"\nSample {sample['index']} (Depth: {sample.get('depth', 'N/A')}%):")
        print(f"  Input length: {len(input_text)} chars")
        print(f"  Prediction time: {pred_time:.2f}s")
        print(f"  Reference: {reference}")
        print(f"  Prediction: {prediction[:100]}..." if len(prediction) > 100 else f"  Prediction: {prediction}")

    # Calculate metrics
    accuracy_all = evaluate_accuracy(predictions, references)
    accuracy_part = evaluate_accuracy_part(predictions, references)
    avg_time = sum(timings) / len(timings)
    total_time = sum(timings)

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Accuracy (all needles must match): {accuracy_all:.1f}%")
    print(f"Accuracy (at least one needle): {accuracy_part:.1f}%")
    print(f"Total samples: {len(data)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per sample: {avg_time:.2f}s")
    print(f"Model load time: {model_load_time:.2f}s")
    print("=" * 60)

    # Save results
    summary = {
        "model_path": args.model_path,
        "block_size": args.block_size,
        "sketch_dim": args.sketch_dim,
        "top_k_blocks": args.top_k_blocks,
        "sparsity_exponent": args.sparsity_exponent,
        "skip_decode": args.skip_decode,
        "num_samples": len(data),
        "accuracy_all": accuracy_all,
        "accuracy_part": accuracy_part,
        "total_time": total_time,
        "avg_time": avg_time,
        "model_load_time": model_load_time,
        "results": results,
    }

    output_file = output_dir / "niah_single_1_results.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_file}")

    # Also save in RULER-compatible format (jsonl)
    ruler_file = output_dir / "niah_single_1.jsonl"
    with open(ruler_file, 'w') as f:
        for i, sample in enumerate(data):
            ruler_entry = {
                "index": sample["index"],
                "input": sample["input"],
                "outputs": sample["outputs"],
                "pred": predictions[i],
            }
            f.write(json.dumps(ruler_entry) + "\n")

    print(f"RULER-compatible results saved to {ruler_file}")

    # Save summary CSV
    csv_file = output_dir / "summary.csv"
    with open(csv_file, 'w') as f:
        f.write("Task,Score,Nulls\n")
        f.write(f"niah_single_1,{accuracy_all:.2f},0\n")

    print(f"Summary CSV saved to {csv_file}")


if __name__ == "__main__":
    main()
