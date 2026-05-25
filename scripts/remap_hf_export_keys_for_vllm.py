#!/usr/bin/env python
"""Remap HF export weight keys to match the Qwen3.5 hub layout expected by vLLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def load_fp32_weight(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, torch.Tensor):
        return payload
    if isinstance(payload, dict):
        if "param" in payload and isinstance(payload["param"], torch.Tensor):
            return payload["param"]
        for value in payload.values():
            if isinstance(value, torch.Tensor):
                return value
    raise TypeError(f"Unsupported weight payload in {path}: {type(payload)}")


def remap_key(key: str) -> str:
    if key.startswith("model.language_model."):
        return key
    if key.startswith("language_model."):
        return f"model.{key}"
    return key


def remap_hf_dir(hf_dir: Path, universal_zero_dir: Path | None, dtype: torch.dtype) -> None:
    index_path = hf_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing {index_path}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]
    shard_names = sorted(set(weight_map.values()))

    new_weight_map: dict[str, str] = {}
    for shard_name in shard_names:
        shard_path = hf_dir / shard_name
        tensors = load_file(str(shard_path))
        remapped = {remap_key(key): tensor for key, tensor in tensors.items()}
        save_file(remapped, str(shard_path))
        for key in remapped:
            new_weight_map[key] = shard_name
        print(f"Remapped {len(remapped)} tensors in {shard_name}", flush=True)

    if universal_zero_dir is not None:
        lm_head_dir = universal_zero_dir / "lm_head.weight"
        lm_head_file = lm_head_dir / "fp32.pt"
        if "lm_head.weight" not in new_weight_map and lm_head_file.is_file():
            tensor = load_fp32_weight(lm_head_file)
            if tensor.dtype != dtype:
                tensor = tensor.to(dtype=dtype)
            target_shard = shard_names[0]
            shard_path = hf_dir / target_shard
            tensors = load_file(str(shard_path))
            tensors["lm_head.weight"] = tensor
            save_file(tensors, str(shard_path))
            new_weight_map["lm_head.weight"] = target_shard
            print(f"Added lm_head.weight to {target_shard}", flush=True)

    index["weight_map"] = dict(sorted(new_weight_map.items()))
    index["metadata"] = index.get("metadata") or {}
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Updated {index_path} ({len(new_weight_map)} entries)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", type=str, required=True)
    parser.add_argument("--universal-zero-dir", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    hf_dir = Path(args.hf_dir).expanduser().resolve()
    universal_zero_dir = (
        Path(args.universal_zero_dir).expanduser().resolve() if args.universal_zero_dir else None
    )
    dtype = getattr(torch, args.dtype)
    remap_hf_dir(hf_dir, universal_zero_dir, dtype)


if __name__ == "__main__":
    main()
