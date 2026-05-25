#!/usr/bin/env python
"""Fill missing Gemma 4 HF weight keys from the base hub model for vLLM loading."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

PATCH_MARKER = ".vllm_gemma4_weights_patched"


def is_gemma4_hf_dir(hf_dir: Path) -> bool:
    config_path = hf_dir / "config.json"
    if not config_path.is_file():
        return False
    config = json.loads(config_path.read_text(encoding="utf-8"))
    architectures = config.get("architectures") or []
    return any("Gemma4" in arch for arch in architectures)


def patch_shard(hf_dir: Path, base_model_name: str, shard_name: str = "model.safetensors") -> int:
    shard_path = hf_dir / shard_name
    if not shard_path.is_file():
        raise FileNotFoundError(f"Missing {shard_path}")

    base_shard = Path(hf_hub_download(base_model_name, shard_name))
    ft_tensors = dict(load_file(str(shard_path)))
    base_tensors = load_file(str(base_shard))

    merged = 0
    for key, tensor in base_tensors.items():
        if key in ft_tensors:
            continue
        if not key.startswith("model.language_model."):
            continue
        ft_tensors[key] = tensor
        merged += 1

    if merged:
        save_file(ft_tensors, str(shard_path))
    return merged


def patch_hf_dir(hf_dir: Path, base_model_name: str, force: bool = False) -> int:
    hf_dir = hf_dir.expanduser().resolve()
    if not is_gemma4_hf_dir(hf_dir):
        return 0

    marker = hf_dir / PATCH_MARKER
    if marker.is_file() and not force:
        return 0

    merged = patch_shard(hf_dir, base_model_name)
    if merged:
        marker.write_text(f"base_model={base_model_name}\nmerged_tensors={merged}\n", encoding="utf-8")
        print(f"Patched {hf_dir}: merged {merged} tensor(s) from {base_model_name}", flush=True)
    else:
        marker.touch()
        print(f"Patched {hf_dir}: no missing language_model tensors", flush=True)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", type=str, required=True)
    parser.add_argument("--base-model-name", type=str, default="google/gemma-4-E2B-it")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    patch_hf_dir(Path(args.hf_dir), args.base_model_name, force=args.force)


if __name__ == "__main__":
    main()
