# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modefied from MInference 
import json
import logging
from typing import Dict, List, Optional

import requests
import torch


class HuggingFaceModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        self.tokenizer = AutoTokenizer.from_pretrained(
            name_or_path, trust_remote_code=True
        )

        if "Yarn-Llama" in name_or_path:
            model_kwargs = None
        else:
            model_kwargs = {"attn_implementation": "flash_attention_2"}

        try:

            if "llama-3" in name_or_path.lower():
                model = AutoModelForCausalLM.from_pretrained(
                    name_or_path,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                )
                self.pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=self.tokenizer,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                )
            else:
                self.pipeline = pipeline(
                    "text-generation",
                    model=name_or_path,
                    tokenizer=self.tokenizer,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    model_kwargs=model_kwargs,
                )
        except:
            self.pipeline = None
            self.model = AutoModelForCausalLM.from_pretrained(
                name_or_path,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )

        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop("stop")

    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        if self.pipeline is None:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            output = self.model.generate(**inputs, **self.generation_kwargs)
            generated_text = self.tokenizer.decode(
                output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
            )
        else:
            output = self.pipeline(
                text_inputs=prompt,
                **self.generation_kwargs,
            )
            assert len(output) == 1
            generated_text = output[0]["generated_text"]

        # remove the input form the generated text
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt) :]

        if self.stop is not None:
            for s in self.stop:
                generated_text = generated_text.split(s)[0]
        return {"text": [generated_text]}


class SeerAttnModel:
    def __init__(self, name_or_path: str, threshold, **generation_kwargs) -> None:
        from seer_attn import SeerAttnLlamaForCausalLM

        from transformers import AutoTokenizer, pipeline, AutoConfig

        config = AutoConfig.from_pretrained(name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.base_model,
            trust_remote_code=True,
            padding_side="left",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("Using threshold: ", threshold)

        model = SeerAttnLlamaForCausalLM.from_pretrained(
            name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            seerattn_sparsity_method='threshold',
            seerattn_threshold = threshold,
            use_cache=True,
            seerattn_last_block_dense=True,
        )

        self.pipeline =None
        self.model = model


        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop("stop")

    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        if self.pipeline is None:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            output = self.model.generate(**inputs, **self.generation_kwargs)
            generated_text = self.tokenizer.decode(
                output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
            )
        else:
            output = self.pipeline(
                text_inputs=prompt,
                **self.generation_kwargs,
            )
            assert len(output) == 1
            generated_text = output[0]["generated_text"]


        # torch.cuda.empty_cache()

        # remove the input form the generated text
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt) :]

        if self.stop is not None:
            for s in self.stop:
                generated_text = generated_text.split(s)[0]
        return {"text": [generated_text]}


class SketchWalkModel:
    """
    SketchWalk model wrapper for RULER evaluation.

    This wrapper loads a SketchWalk-enabled LLaMA model and provides
    the same interface as other RULER model wrappers.
    """

    def __init__(
        self,
        name_or_path: str,
        block_size: int = 64,
        sketch_dim: int = 64,
        top_k_blocks: int = 16,
        sparsity_exponent: int = 8,
        **generation_kwargs
    ) -> None:
        """
        Initialize SketchWalk model.

        Args:
            name_or_path: Path to the SketchWalk model checkpoint
            block_size: Block size for SketchWalk (default: 64)
            sketch_dim: Sketch dimension (default: 64)
            top_k_blocks: Number of top-k blocks (default: 16)
            sparsity_exponent: Sparsity exponent (default: 8)
            **generation_kwargs: Additional generation arguments
        """
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

        from sketch_walk.llama import SketchWalkLlamaForCausalLM
        from transformers import AutoTokenizer, AutoConfig

        # Load tokenizer
        config = AutoConfig.from_pretrained(name_or_path)
        base_model = getattr(config, 'base_model', name_or_path)

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True,
            padding_side="left",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Print SketchWalk configuration
        print("=" * 60)
        print("SketchWalk Model Configuration:")
        print(f"  Block Size: {block_size}")
        print(f"  Sketch Dim: {sketch_dim}")
        print(f"  Top-K Blocks: {top_k_blocks}")
        print(f"  Sparsity Exponent: {sparsity_exponent}")
        print("=" * 60)

        # Load model with SketchWalk configuration
        model = SketchWalkLlamaForCausalLM.from_pretrained(
            name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            sketchwalk_block_size=block_size,
            sketchwalk_sketch_dim=sketch_dim,
            sketchwalk_top_k_blocks=top_k_blocks,
            sketchwalk_sparsity_exponent=sparsity_exponent,
        )

        self.pipeline = None
        self.model = model
        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop("stop", None)

    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        """
        Generate text using SketchWalk model.

        Args:
            prompt: Input prompt text
            **kwargs: Additional arguments

        Returns:
            Dictionary with generated text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, **self.generation_kwargs)
        generated_text = self.tokenizer.decode(
            output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )

        # Remove the input from the generated text
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):]

        # Apply stop words
        if self.stop is not None:
            for s in self.stop:
                generated_text = generated_text.split(s)[0]

        return {"text": [generated_text]}
