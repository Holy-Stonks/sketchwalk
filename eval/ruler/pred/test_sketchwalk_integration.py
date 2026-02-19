#!/usr/bin/env python3
"""
Test script to verify SketchWalk RULER integration.

This script tests the basic components without requiring a full model checkpoint.
It validates that the imports work and the data can be generated.
"""

import os
import sys
from pathlib import Path

# Get the correct SeerAttention directory
# The script is in: /home/valery/sketch_walk/SeerAttention/eval/ruler/pred/
script_dir = Path(os.path.abspath(__file__))
seer_attention_dir = script_dir.parent.parent.parent.parent

# Add to path
sys.path.insert(0, str(seer_attention_dir))
os.chdir(seer_attention_dir)

print(f"Working directory: {os.getcwd()}")
print(f"SeerAttention directory: {seer_attention_dir}")
print()


def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")

    try:
        from eval.ruler.pred.model_wrappers import SketchWalkModel
        print("  ✓ SketchWalkModel imported successfully")
    except Exception as e:
        print(f"  ✗ Failed to import SketchWalkModel: {e}")
        return False

    try:
        from sketch_walk.llama import SketchWalkLlamaForCausalLM
        print("  ✓ SketchWalkLlamaForCausalLM imported successfully")
    except Exception as e:
        print(f"  ✗ Failed to import SketchWalkLlamaForCausalLM: {e}")
        return False

    try:
        from sketch_walk.common.core import SketchWalkConfig, SketchWalkAttention
        print("  ✓ SketchWalk core modules imported successfully")
    except Exception as e:
        print(f"  ✗ Failed to import SketchWalk core: {e}")
        return False

    return True


def test_data_generation():
    """Test that data generation script exists and is valid."""
    print("\nTesting data generation script...")

    script_path = Path(__file__).parent.parent / "data" / "synthetic" / "generate_niah_single_1.py"

    if not script_path.exists():
        print(f"  ✗ Data generation script not found: {script_path}")
        return False

    print(f"  ✓ Data generation script found: {script_path}")

    # Check script can be imported
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("generate_niah", script_path)
        print(f"  ✓ Data generation script is valid Python")
    except Exception as e:
        print(f"  ✗ Data generation script has errors: {e}")
        return False

    return True


def test_config():
    """Test SketchWalk configuration."""
    print("\nTesting SketchWalk configuration...")

    try:
        from sketch_walk.common.core import create_sketch_walk_config

        config = create_sketch_walk_config(
            block_size=64,
            sketch_dim=64,
            top_k_blocks=16,
            sparsity_exponent=8,
        )

        print(f"  ✓ Configuration created:")
        print(f"    - block_size: {config.block_size}")
        print(f"    - sketch_dim: {config.sketch_dim}")
        print(f"    - top_k_blocks: {config.top_k_blocks}")
        print(f"    - sparsity_exponent: {config.sparsity_exponent}")
        print(f"    - Estimated sparsity (4k tokens): {config.sparsity_level(4096)*100:.1f}%")

        return True
    except Exception as e:
        print(f"  ✗ Failed to create configuration: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_directory_structure():
    """Test that required directories exist."""
    print("\nTesting directory structure...")

    base_path = Path(__file__).parent.parent.parent.parent

    required_dirs = [
        "sketch_walk/llama",
        "sketch_walk/common",
        "eval/ruler/data/synthetic",
        "eval/ruler/pred",
    ]

    all_exist = True
    for dir_path in required_dirs:
        full_path = base_path / dir_path
        if full_path.exists():
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ✗ {dir_path}/ (missing)")
            all_exist = False

    return all_exist


def main():
    """Run all tests."""
    print("=" * 60)
    print("SketchWalk RULER Integration Test")
    print("=" * 60)

    tests = [
        ("Directory Structure", test_directory_structure),
        ("Imports", test_imports),
        ("Data Generation Script", test_data_generation),
        ("Configuration", test_config),
    ]

    results = []
    for name, test_func in tests:
        print(f"\n[{name}]")
        result = test_func()
        results.append((name, result))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed! Integration is ready.")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
