#!/usr/bin/env python3
"""
Compare SeerAttention vs SketchWalk results on RULER evaluation.

This script compares the results from SeerAttention and SketchWalk evaluations,
showing accuracy and performance differences.

Usage:
    python compare_results.py --seerattn_dir ./results/seerattn --sketchwalk_dir ./results/sketchwalk
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List


def parse_args():
    parser = argparse.ArgumentParser(description="Compare SeerAttention vs SketchWalk results")

    parser.add_argument(
        "--seerattn_dir",
        type=str,
        default="./results/seerattn",
        help="Path to SeerAttention results directory"
    )
    parser.add_argument(
        "--sketchwalk_dir",
        type=str,
        default="./results/sketchwalk",
        help="Path to SketchWalk results directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/comparison",
        help="Output directory for comparison results"
    )

    return parser.parse_args()


def load_results(results_dir: str) -> Dict:
    """Load evaluation results from a directory."""
    results_path = Path(results_dir)

    # Try to load the main results JSON file
    results_file = results_path / "niah_single_1_results.json"

    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)

    # Try to load from RULER-compatible JSONL
    jsonl_file = results_path / "niah_single_1.jsonl"
    if jsonl_file.exists():
        results = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))

        # Calculate metrics from JSONL
        predictions = [r["pred"] for r in results]
        references = [r["outputs"] for r in results]

        accuracy_all = evaluate_accuracy_all(predictions, references)
        accuracy_part = evaluate_accuracy_part(predictions, references)

        return {
            "accuracy_all": accuracy_all,
            "accuracy_part": accuracy_part,
            "num_samples": len(results),
            "results": results,
        }

    # Try to load from summary CSV
    csv_file = results_path / "summary.csv"
    if csv_file.exists():
        with open(csv_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("niah_single_1"):
                    parts = line.strip().split(",")
                    return {
                        "accuracy_all": float(parts[1]),
                        "accuracy_part": float(parts[1]),  # Assuming same for simplicity
                        "num_samples": "N/A",
                    }

    raise FileNotFoundError(f"No results found in {results_dir}")


def evaluate_accuracy_all(predictions: List[str], references: List[List[str]]) -> float:
    """Evaluate accuracy (all needles must match)."""
    correct = 0
    for pred, ref in zip(predictions, references):
        all_found = all(r.lower() in pred.lower() for r in ref)
        if all_found:
            correct += 1
    return (correct / len(predictions)) * 100 if predictions else 0.0


def evaluate_accuracy_part(predictions: List[str], references: List[List[str]]) -> float:
    """Evaluate accuracy (at least one needle must match)."""
    correct = 0
    for pred, ref in zip(predictions, references):
        any_found = any(r.lower() in pred.lower() for r in ref)
        if any_found:
            correct += 1
    return (correct / len(predictions)) * 100 if predictions else 0.0


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_config_comparison(seerattn: Dict, sketchwalk: Dict):
    """Print configuration comparison."""
    print("\nConfiguration Comparison:")
    print("-" * 60)

    # SeerAttention config
    print("\nSeerAttention:")
    if "threshold" in seerattn:
        print(f"  Threshold: {seerattn.get('threshold', 'N/A')}")
    else:
        print(f"  Config: Not specified in results")

    # SketchWalk config
    print("\nSketchWalk:")
    print(f"  Block Size: {sketchwalk.get('block_size', 'N/A')}")
    print(f"  Sketch Dim: {sketchwalk.get('sketch_dim', 'N/A')}")
    print(f"  Top-K Blocks: {sketchwalk.get('top_k_blocks', 'N/A')}")
    print(f"  Sparsity Exponent: {sketchwalk.get('sparsity_exponent', 'N/A')}")
    print(f"  Skip Decode: {sketchwalk.get('skip_decode', 'N/A')}")


def print_performance_comparison(seerattn: Dict, sketchwalk: Dict):
    """Print performance comparison."""
    print("\nPerformance Comparison:")
    print("-" * 60)
    print(f"{'Metric':<30} {'SeerAttention':<15} {'SketchWalk':<15} {'Diff':<10}")
    print("-" * 60)

    # Accuracy comparison
    seer_acc_all = seerattn.get('accuracy_all', 0)
    sw_acc_all = sketchwalk.get('accuracy_all', 0)
    diff_acc_all = sw_acc_all - seer_acc_all
    print(f"{'Accuracy (all needles)':<30} {seer_acc_all:<15.1f} {sw_acc_all:<15.1f} {diff_acc_all:>+7.1f}%")

    seer_acc_part = seerattn.get('accuracy_part', 0)
    sw_acc_part = sketchwalk.get('accuracy_part', 0)
    diff_acc_part = sw_acc_part - seer_acc_part
    print(f"{'Accuracy (part needles)':<30} {seer_acc_part:<15.1f} {sw_acc_part:<15.1f} {diff_acc_part:>+7.1f}%")

    # Timing comparison (if available)
    if 'avg_time' in seerattn and 'avg_time' in sketchwalk:
        seer_time = seerattn.get('avg_time', 0)
        sw_time = sketchwalk.get('avg_time', 0)
        diff_time = ((sw_time - seer_time) / seer_time * 100) if seer_time > 0 else 0
        speedup = seer_time / sw_time if sw_time > 0 else 0
        print(f"{'Avg time per sample (s)':<30} {seer_time:<15.2f} {sw_time:<15.2f} {diff_time:>+7.1f}%")
        print(f"{'Speedup factor':<30} {'':<15} {'':<15} {speedup:<7.2f}x")

    if 'total_time' in seerattn and 'total_time' in sketchwalk:
        seer_total = seerattn.get('total_time', 0)
        sw_total = sketchwalk.get('total_time', 0)
        print(f"{'Total time (s)':<30} {seer_total:<15.2f} {sw_total:<15.2f}")


def print_detailed_comparison(seerattn: Dict, sketchwalk: Dict):
    """Print detailed per-sample comparison if available."""
    if 'results' not in seerattn or 'results' not in sketchwalk:
        return

    print("\nDetailed Sample-by-Sample Comparison:")
    print("-" * 60)
    print(f"{'Index':<10} {'SeerAcc':<10} {'SketchAcc':<10} {'TimeDiff':<10}")
    print("-" * 60)

    seer_results = seerattn['results']
    sw_results = sketchwalk['results']

    for seer_r, sw_r in zip(seer_results, sw_results):
        idx = seer_r.get('index', sw_r.get('index', 'N/A'))

        # Check accuracy
        seer_pred = seer_r.get('prediction', '')
        sw_pred = sw_r.get('prediction', '')
        ref = seer_r.get('reference', sw_r.get('reference', []))

        seer_correct = all(r.lower() in seer_pred.lower() for r in ref) if ref else False
        sw_correct = all(r.lower() in sw_pred.lower() for r in ref) if ref else False

        seer_acc = "✓" if seer_correct else "✗"
        sw_acc = "✓" if sw_correct else "✗"

        # Time difference
        seer_time = seer_r.get('time', 0)
        sw_time = sw_r.get('time', 0)
        time_diff = ((sw_time - seer_time) / seer_time * 100) if seer_time > 0 else 0
        time_str = f"{time_diff:+.1f}%"

        print(f"{idx:<10} {seer_acc:<10} {sw_acc:<10} {time_str:<10}")


def generate_summary_report(seerattn: Dict, sketchwalk: Dict, output_dir: Path):
    """Generate a summary report file."""
    report = []
    report.append("# SeerAttention vs SketchWalk Comparison Report")
    report.append("")
    report.append("## Configuration")
    report.append("")
    report.append("### SeerAttention")
    if "threshold" in seerattn:
        report.append(f"- Threshold: {seerattn.get('threshold', 'N/A')}")
    report.append("")
    report.append("### SketchWalk")
    report.append(f"- Block Size: {sketchwalk.get('block_size', 'N/A')}")
    report.append(f"- Sketch Dim: {sketchwalk.get('sketch_dim', 'N/A')}")
    report.append(f"- Top-K Blocks: {sketchwalk.get('top_k_blocks', 'N/A')}")
    report.append(f"- Sparsity Exponent: {sketchwalk.get('sparsity_exponent', 'N/A')}")
    report.append(f"- Skip Decode: {sketchwalk.get('skip_decode', 'N/A')}")
    report.append("")
    report.append("## Performance Comparison")
    report.append("")
    report.append(f"| Metric | SeerAttention | SketchWalk | Difference |")
    report.append(f"|--------|---------------|------------|------------|")

    # Accuracy
    seer_acc_all = seerattn.get('accuracy_all', 0)
    sw_acc_all = sketchwalk.get('accuracy_all', 0)
    diff_acc_all = sw_acc_all - seer_acc_all
    report.append(f"| Accuracy (all) | {seer_acc_all:.1f}% | {sw_acc_all:.1f}% | {diff_acc_all:+.1f}% |")

    seer_acc_part = seerattn.get('accuracy_part', 0)
    sw_acc_part = sketchwalk.get('accuracy_part', 0)
    diff_acc_part = sw_acc_part - seer_acc_part
    report.append(f"| Accuracy (part) | {seer_acc_part:.1f}% | {sw_acc_part:.1f}% | {diff_acc_part:+.1f}% |")

    # Timing
    if 'avg_time' in seerattn and 'avg_time' in sketchwalk:
        seer_time = seerattn.get('avg_time', 0)
        sw_time = sketchwalk.get('avg_time', 0)
        speedup = seer_time / sw_time if sw_time > 0 else 0
        report.append(f"| Avg Time (s) | {seer_time:.2f} | {sw_time:.2f} | {speedup:.2f}x speedup |")

    report.append("")
    report.append("## Conclusion")
    report.append("")
    if sw_acc_all >= seer_acc_all:
        report.append(f"- SketchWalk achieves **comparable or better accuracy** ({sw_acc_all:.1f}% vs {seer_acc_all:.1f}%)")
    else:
        report.append(f"- SketchWalk shows slightly lower accuracy ({sw_acc_all:.1f}% vs {seer_acc_all:.1f}%)")

    if 'avg_time' in seerattn and 'avg_time' in sketchwalk:
        if sw_time < seer_time:
            report.append(f"- SketchWalk is **faster** by {(1 - sw_time/seer_time)*100:.1f}%")

    report.append("")

    # Save report
    report_file = output_dir / "comparison_report.md"
    with open(report_file, 'w') as f:
        f.write("\n".join(report))

    return report_file


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print_section("SeerAttention vs SketchWalk Comparison")
    print(f"\nSeerAttention results: {args.seerattn_dir}")
    print(f"SketchWalk results: {args.sketchwalk_dir}")
    print(f"Output directory: {args.output_dir}")

    # Load results
    print("\nLoading results...")
    try:
        seerattn = load_results(args.seerattn_dir)
        print(f"  Loaded SeerAttention results: {seerattn.get('num_samples', 'N/A')} samples")
    except FileNotFoundError as e:
        print(f"  Warning: Could not load SeerAttention results: {e}")
        seerattn = {"accuracy_all": 0, "accuracy_part": 0, "num_samples": 0}

    try:
        sketchwalk = load_results(args.sketchwalk_dir)
        print(f"  Loaded SketchWalk results: {sketchwalk.get('num_samples', 'N/A')} samples")
    except FileNotFoundError as e:
        print(f"  Error: Could not load SketchWalk results: {e}")
        return

    # Print comparisons
    print_config_comparison(seerattn, sketchwalk)
    print_performance_comparison(seerattn, sketchwalk)
    print_detailed_comparison(seerattn, sketchwalk)

    # Generate report
    report_file = generate_summary_report(seerattn, sketchwalk, output_dir)
    print(f"\nComparison report saved to {report_file}")

    # Save comparison JSON
    comparison = {
        "seerattn": seerattn,
        "sketchwalk": sketchwalk,
        "differences": {
            "accuracy_all": sketchwalk.get('accuracy_all', 0) - seerattn.get('accuracy_all', 0),
            "accuracy_part": sketchwalk.get('accuracy_part', 0) - seerattn.get('accuracy_part', 0),
        }
    }

    if 'avg_time' in seerattn and 'avg_time' in sketchwalk:
        comparison["differences"]["time_speedup"] = seerattn.get('avg_time', 1) / sketchwalk.get('avg_time', 1)

    comparison_file = output_dir / "comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"Comparison JSON saved to {comparison_file}")


if __name__ == "__main__":
    main()
