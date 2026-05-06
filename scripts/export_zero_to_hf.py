#!/usr/bin/env python

import argparse
import os
import subprocess
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a DeepSpeed ZeRO checkpoint to a HF-style directory.\n\n"
            "This will:\n"
            "  1) Run scripts/zero_to_fp32.py on the given checkpoint_dir and tag,\n"
            "     writing FP32 safetensors weights into <checkpoint_dir>/<tag>_hf\n"
            "  2) Download config and tokenizer for the given model_name from the hub\n"
            "     and save them into the same <checkpoint_dir>/<tag>_hf directory."
        )
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Hugging Face hub model id to use for config/tokenizer (e.g. Qwen/Qwen3-1.7B).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help=(
            "Path to the checkpoint directory that contains the checkpoint (e.g. /path/to/checkpoint-12)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Optional explicit output directory. Defaults to "
            "<checkpoint-dir>/<tag>_hf."
        ),
    )
    parser.add_argument(
        "--safe-serialization",
        default=True,
        type=argparse.BooleanOptionalAction,
        help="Disable safetensors and write pytorch_model.bin instead.",
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise SystemExit(f"checkpoint_dir does not exist or is not a directory: {checkpoint_dir}")

    tag = checkpoint_dir.name
    checkpoint_dir = checkpoint_dir.parent

    if args.output_dir is None:
        output_dir = checkpoint_dir / f"hf_{tag}"
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()

    zero_script = checkpoint_dir / "zero_to_fp32.py"
    if not zero_script.is_file():
        raise SystemExit(f"Could not find zero_to_fp32.py next to this script at {zero_script}")

    cmd = [
        "python",
        str(zero_script),
        str(checkpoint_dir),
        str(output_dir),
        "--max_shard_size",
        "5GB",
        "--tag",
        tag,
    ]
    if args.safe_serialization:
        cmd.append("--safe_serialization")

    print("Running zero_to_fp32.py to export FP32 weights:")
    print(" ", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"zero_to_fp32.py failed with return code {result.returncode}")

    # Save config and tokenizer from the hub into the same directory.
    print(f"Saving config and tokenizer for {args.model_name} into {output_dir}")
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    config.save_pretrained(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    print(f"Done. HF-style checkpoint is in: {output_dir}")


if __name__ == "__main__":
    main()
