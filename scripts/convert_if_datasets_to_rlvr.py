#!/usr/bin/env python3
"""
Convert google/IFEval and allenai/IFBench_test to RLVR format (messages, ground_truth, dataset)
and upload to HuggingFace.

Usage:
    python scripts/convert_if_datasets_to_rlvr.py [--upload]
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset, load_dataset


def convert_if_example(example: dict, dataset_name: str) -> dict:
    """Convert a single IFEval/IFBench example to RLVR format."""
    prompt = example["prompt"]
    instruction_id_list = example["instruction_id_list"]
    kwargs = example["kwargs"]
    kwargs = [{k: v for k, v in entry.items() if v is not None} or None for entry in kwargs]

    # RLVR format: messages, ground_truth, dataset
    messages = [
        {"role": "user", "content": prompt},
    ]

    # IFEvalVerifier expects ground_truth as JSON string:
    # [{"instruction_id": [...], "kwargs": [...]}]
    ground_truth = json.dumps([{"instruction_id": instruction_id_list, "kwargs": kwargs}]).replace("null", "None")

    return {
        "messages": messages,
        "ground_truth": ground_truth,
        "dataset": dataset_name,
    }


def convert_dataset(hf_path: str, split: str, dataset_name: str) -> Dataset:
    """Load and convert a dataset to RLVR format."""
    ds = load_dataset(hf_path, split=split)
    rows = [convert_if_example(ds[i], dataset_name) for i in range(len(ds))]
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser(description="Convert IF datasets to RLVR format")
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload converted datasets to HuggingFace",
    )
    parser.add_argument(
        "--hf-user",
        default="mohdelgaar",
        help="HuggingFace username for upload",
    )
    args = parser.parse_args()

    # Convert IFEval
    print("Converting google/IFEval...")
    ifeval_rlvr = convert_dataset("google/IFEval", "train", "ifeval")
    print(f"  -> {len(ifeval_rlvr)} examples")

    # Convert IFBench_test
    print("Converting allenai/IFBench_test...")
    ifbench_rlvr = convert_dataset("allenai/IFBench_test", "train", "ifbench")
    print(f"  -> {len(ifbench_rlvr)} examples")

    if args.upload:
        print("\nUploading to HuggingFace...")
        ifeval_rlvr.push_to_hub(f"{args.hf_user}/ifeval_rlvr", private=False)
        print(f"  -> {args.hf_user}/ifeval_rlvr")
        ifbench_rlvr.push_to_hub(f"{args.hf_user}/ifbench_rlvr", private=False)
        print(f"  -> {args.hf_user}/ifbench_rlvr")
        print("Done.")
    else:
        # Save locally for verification
        out_dir = Path(__file__).parent.parent
        ifeval_rlvr.to_json(out_dir / "ifeval_rlvr.jsonl")
        ifbench_rlvr.to_json(out_dir / "ifbench_rlvr.jsonl")
        print(f"\nSaved locally: {out_dir}/ifeval_rlvr.jsonl, {out_dir}/ifbench_rlvr.jsonl")
        print("Run with --upload to push to HuggingFace.")


if __name__ == "__main__":
    main()
