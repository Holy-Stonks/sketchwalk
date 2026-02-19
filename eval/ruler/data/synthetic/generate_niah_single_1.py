#!/usr/bin/env python3
"""
Generate niah_single_1 test data for SketchWalk evaluation.

This script generates a small set (3-5 samples) of needle-in-a-haystack
test cases for evaluating SketchWalk with the RULER framework.

Usage:
    python generate_niah_single_1.py --save_dir ./data --num_samples 5
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from data.tokenizer import select_tokenizer
from nltk.tokenize import sent_tokenize

parser = argparse.ArgumentParser()
# Basic Configurations
parser.add_argument(
    "--save_dir", type=Path, required=True, help="dataset folder to save dataset"
)
parser.add_argument(
    "--subset", type=str, default="validation", help="Options: validation or test"
)
parser.add_argument(
    "--tokenizer_path", type=str, default="meta-llama/Llama-3.1-8B",
    help="path to the tokenizer model (will use HF tokenizer)"
)
parser.add_argument(
    "--tokenizer_type", type=str, default="hf", help="[Options] nemo, hf, openai."
)
parser.add_argument(
    "--max_seq_length",
    type=int,
    default=4096,
    help="max sequence length including all input tokens and generated tokens.",
)
parser.add_argument(
    "--tokens_to_generate",
    type=int,
    default=128,
    help="expected generated token amount.",
)
parser.add_argument(
    "--num_samples", type=int, default=5, help="number of samples to generate"
)
parser.add_argument("--random_seed", type=int, default=42)

# Complexity Configurations (niah_single_1 settings)
parser.add_argument("--num_needle_k", type=int, default=1)
parser.add_argument("--num_needle_v", type=int, default=1)
parser.add_argument("--num_needle_q", type=int, default=1)
parser.add_argument(
    "--type_haystack",
    type=str,
    default="repeat",
    help="[Options] repeat, essay, needle.",
)
parser.add_argument(
    "--type_needle_k",
    type=str,
    default="words",
    help="[Options] numbers, words, uuids.",
)
parser.add_argument(
    "--type_needle_v",
    type=str,
    default="numbers",
    help="[Options] numbers, words, uuids.",
)

args = parser.parse_args()
random.seed(args.random_seed)

# Load Tokenizer (use HF tokenizer by default)
TOKENIZER = select_tokenizer(args.tokenizer_type, args.tokenizer_path)

# Define Needle/Haystack Format
needle = "One of the special magic {type_needle_v} for {key} is: {value}."
if args.type_haystack == "essay":
    essay = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "json/PaulGrahamEssays.json"
    )
    if os.path.exists(essay):
        essay = json.load(open(essay))["text"]
        haystack = re.sub(r"\s+", " ", essay).split(" ")
    else:
        # Fallback to repeat pattern if essay file not found
        print("Warning: PaulGrahamEssays.json not found, using repeat pattern")
        args.type_haystack = "repeat"
        haystack = None
elif args.type_haystack == "repeat":
    haystack = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
elif args.type_haystack == "needle":
    haystack = needle
else:
    raise NotImplementedError(f"{args.type_haystack} is not implemented.")

# Load wonderwords for random word generation
try:
    import wonderwords
    nouns = wonderwords.random_word._get_words_from_text_file("nounlist.txt")
    adjs = wonderwords.random_word._get_words_from_text_file("adjectivelist.txt")
    words = [f"{adj}-{noun}" for adj in adjs for noun in nouns]
    words = sorted(list(set(words)))
except ImportError:
    # Fallback simple word list if wonderwords not available
    words = [
        "apple-red", "banana-yellow", "cherry-red", "grape-purple", "lemon-yellow",
        "orange-orange", "peach-pink", "pear-green", "plum-purple", "strawberry-red",
        "watermelon-pink", "blueberry-blue", "blackberry-black", "raspberry-red",
        "mango-orange", "pineapple-yellow", "kiwi-green", "lime-green", "coconut-white",
        "avocado-green", "pomegranate-red", "fig-purple", "date-brown", "nut-brown",
    ]

# Positions for needle placement
DEPTHS = [10, 30, 50, 70, 90]  # Percentages for needle placement


def generate_random_number(num_digits=7):
    lower_bound = 10 ** (num_digits - 1)
    upper_bound = 10**num_digits - 1
    return str(random.randint(lower_bound, upper_bound))


def generate_random_word():
    word = random.choice(words)
    return word


def generate_random_uuid():
    import uuid
    return str(uuid.UUID(int=random.getrandbits(128), version=4))


def generate_random(type_needle: str):
    if type_needle == "numbers":
        return generate_random_number()
    elif type_needle == "words":
        return generate_random_word()
    elif type_needle == "uuids":
        return generate_random_uuid()
    else:
        raise NotImplementedError(f"{args.type_needle} is not implemented.")


def generate_input_output(num_haystack, depth_percent):
    keys, values, needles = [], [], []
    for _ in range(args.num_needle_k):
        keys.append(generate_random(args.type_needle_k))
        value = []
        for _ in range(args.num_needle_v):
            value.append(generate_random(args.type_needle_v))
            needles.append(
                needle.format(
                    type_needle_v=args.type_needle_v,
                    key=keys[-1],
                    value=value[-1],
                )
            )
        values.append(value)

    # Randomize needle order
    random.Random(args.random_seed + depth_percent).shuffle(needles)

    # Context
    if args.type_haystack == "essay" and haystack is not None:
        text = " ".join(haystack[:num_haystack])
        document_sents = sent_tokenize(text.strip())
        insertion_position = int(len(document_sents) * (depth_percent / 100))

        document_sents_list = document_sents[:insertion_position]
        document_sents_list.append(needles[0])
        document_sents_list.extend(document_sents[insertion_position:])
        context = " ".join(document_sents_list)

    else:
        if args.type_haystack == "repeat":
            sentences = [haystack] * num_haystack
        elif args.type_haystack == "needle":
            sentences = [
                haystack.format(
                    type_needle_v=args.type_needle_v,
                    key=generate_random(args.type_needle_k),
                    value=generate_random(args.type_needle_v),
                )
                for _ in range(num_haystack)
            ]

        # Insert needle at specified depth
        insertion_position = int(num_haystack * (depth_percent / 100))
        sentences.insert(insertion_position, needles[0])
        context = "\n".join(sentences)

    # Query and Answer
    query = keys[0]
    answers = values[0]

    # Template for niah_single_1
    template = (
        "A special magic {type_needle_v} is hidden within the following text. "
        "Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n"
        "{context}\n"
        "What is the special magic {type_needle_v} for {query} mentioned in the provided text? "
        "The special magic {type_needle_v} for {query} mentioned in the provided text is"
    )

    type_needle_v = args.type_needle_v if args.num_needle_v > 1 else args.type_needle_v[:-1]

    input_text = template.format(
        type_needle_v=type_needle_v,
        context=context,
        query=query,
    )

    return input_text, answers


def generate_samples(
    num_samples: int, max_seq_length: int, save_dir: str
):
    write_jsons = []
    tokens_to_generate = args.tokens_to_generate

    # Determine incremental step based on haystack type
    if args.type_haystack == "essay":
        incremental = 500
    elif args.type_haystack == "repeat":
        incremental = 25
    elif args.type_haystack == "needle":
        incremental = 25
    else:
        incremental = 25

    # Calculate appropriate haystack size
    num_haystack = incremental
    total_tokens = 0

    while total_tokens + tokens_to_generate < max_seq_length:
        input_text, answer = generate_input_output(num_haystack, 50)  # Use middle depth for sizing
        total_tokens = len(TOKENIZER.text_to_tokens(input_text + " ".join(answer)))
        print(
            f"Max length {max_seq_length} | Current length {total_tokens + tokens_to_generate} | Haystack: {num_haystack}"
        )
        if total_tokens + tokens_to_generate > max_seq_length:
            num_haystack -= incremental
            break

        if args.type_haystack == "essay" and haystack is not None and num_haystack > len(haystack):
            num_haystack = len(haystack)
            break

        num_haystack += incremental

    print("Final haystack size:", num_haystack)

    # Generate samples with different depths
    depths = DEPTHS[:num_samples] if num_samples <= len(DEPTHS) else DEPTHS

    for index, depth in enumerate(depths):
        used_haystack = num_haystack
        while True:
            try:
                input_text, answer = generate_input_output(used_haystack, depth)
                length = len(TOKENIZER.text_to_tokens(input_text)) + tokens_to_generate
                assert length <= max_seq_length, f"{length} exceeds max_seq_length."
                print(f"Sample {index}: Depth={depth}%, Length={length}, Haystack={used_haystack}")
                break
            except:
                if used_haystack > incremental:
                    used_haystack -= incremental
                else:
                    raise

        formatted_output = {
            "index": index,
            "input": input_text,
            "outputs": answer,
            "length": length,
            "depth": depth,
        }
        write_jsons.append(formatted_output)

    return write_jsons


def main():
    from nemo.collections.asr.parts.utils.manifest_utils import write_manifest

    save_file = args.save_dir / f"niah_single_1" / f"{args.subset}.jsonl"
    save_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.num_samples} niah_single_1 samples...")
    print(f"Configuration:")
    print(f"  Type haystack: {args.type_haystack}")
    print(f"  Type needle key: {args.type_needle_k}")
    print(f"  Type needle value: {args.type_needle_v}")
    print(f"  Max sequence length: {args.max_seq_length}")
    print(f"  Tokens to generate: {args.tokens_to_generate}")

    write_jsons = generate_samples(
        num_samples=args.num_samples,
        max_seq_length=args.max_seq_length,
        save_dir=args.save_dir,
    )

    write_manifest(save_file, write_jsons)
    print(f"\nSaved {len(write_jsons)} samples to {save_file}")


if __name__ == "__main__":
    main()
