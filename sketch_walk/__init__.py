"""
SketchWalk Sparse Attention for LLMs

A training-free sparse attention method combining:
1. Small-World Sketching (SWS): Efficient block-level attention estimation
2. Sketch-Determined Walk: Cross-layer attention accumulation

Reference: "Scout Before You Attend: Sketch-and-Walk Sparse Attention for Efficient LLM Inference"
"""

from .common import (
    SketchWalkConfig,
    HadamardTransform,
    Sketch,
    Walk,
    SketchWalkAttention,
    create_sketch_walk_config,
)

from .llama import (
    SketchWalkLlamaConfig,
    SketchWalkLlamaModel,
    SketchWalkLlamaForCausalLM,
    create_sketch_walk_llama,
)

__version__ = "0.1.0"
__all__ = [
    # Core components
    "SketchWalkConfig",
    "HadamardTransform",
    "Sketch",
    "Walk",
    "SketchWalkAttention",
    "create_sketch_walk_config",
    # LLaMA integration
    "SketchWalkLlamaConfig",
    "SketchWalkLlamaModel",
    "SketchWalkLlamaForCausalLM",
    "create_sketch_walk_llama",
]
