#!/usr/bin/env python
"""Export a DeepSpeed universal checkpoint (ds_universal_global_step*) to HuggingFace format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer


def is_universal_checkpoint_dir(checkpoint_dir: Path) -> bool:
    return (checkpoint_dir / "zero").is_dir() and (checkpoint_dir / "mp_rank_00_model_states.pt").is_file()


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


def universal_param_candidates(param_name: str) -> list[str]:
    """Map DeepSpeed universal shard names onto HF Qwen3.5 parameter names."""
    stripped = param_name.removeprefix("model.") if param_name.startswith("model.") else param_name
    candidates = [
        param_name,
        stripped,
        f"model.{stripped}",
        f"language_model.{stripped}",
        f"language_model.model.{stripped}",
    ]
    if stripped == "lm_head.weight" or param_name == "lm_head.weight":
        candidates.extend(["lm_head.weight", "language_model.lm_head.weight"])
    return list(dict.fromkeys(candidates))


def load_universal_weights_into_module(
    module: torch.nn.Module, zero_dir: Path, dtype: torch.dtype
) -> tuple[int, int]:
    model_params = dict(module.named_parameters())
    loaded = 0
    missing = 0

    param_dirs = sorted(p for p in zero_dir.iterdir() if p.is_dir())
    total = len(param_dirs)

    for idx, param_dir in enumerate(param_dirs, start=1):
        shard_name = param_dir.name
        fp32_path = param_dir / "fp32.pt"
        if not fp32_path.is_file():
            continue

        target_name = None
        for candidate in universal_param_candidates(shard_name):
            if candidate in model_params:
                target_name = candidate
                break

        if target_name is None:
            missing += 1
            continue

        tensor = load_fp32_weight(fp32_path)
        if tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        model_params[target_name].data.copy_(tensor)
        loaded += 1

        if idx % 50 == 0 or idx == total:
            print(f"Loaded {loaded} tensors ({idx}/{total} dirs scanned) from {zero_dir}", flush=True)

    return loaded, missing


def remap_state_dict_for_hub(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Match Qwen/Qwen3.5-9B hub key layout (model.language_model.*, lm_head.*)."""
    remapped: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if key.startswith("language_model."):
            remapped[f"model.{key}"] = tensor
        else:
            remapped[key] = tensor
    return remapped


def write_hub_config(output_dir: Path, model_name: str) -> None:
    hub_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config_dict = hub_config.to_dict()
    config_dict["architectures"] = ["Qwen3_5ForConditionalGeneration"]
    (output_dir / "config.json").write_text(json.dumps(config_dict, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a DeepSpeed universal checkpoint to a HuggingFace model directory."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Hugging Face hub model id for config/tokenizer/architecture (e.g. Qwen/Qwen3.5-9B).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Path to ds_universal_global_step<N> (or similar universal checkpoint directory).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output HF directory. Defaults to <run_dir>/hf_<checkpoint_tag>.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Dtype to store weights in the exported HF checkpoint.",
    )
    parser.add_argument(
        "--max-shard-size",
        type=str,
        default="5GB",
        help="Maximum shard size passed to model.save_pretrained.",
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise SystemExit(f"checkpoint_dir does not exist or is not a directory: {checkpoint_dir}")

    if not is_universal_checkpoint_dir(checkpoint_dir):
        raise SystemExit(
            f"Not a DeepSpeed universal checkpoint (expected zero/ and mp_rank_00_model_states.pt): "
            f"{checkpoint_dir}"
        )

    run_dir = checkpoint_dir.parent
    tag = checkpoint_dir.name
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else run_dir / f"hf_{tag}"
    )

    dtype = getattr(torch, args.dtype)
    zero_dir = checkpoint_dir / "zero"

    print(f"Loading hub skeleton from {args.model_name} (dtype={args.dtype})", flush=True)
    model = AutoModel.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    print(f"Loading RL weights from {zero_dir}", flush=True)
    loaded, missing = load_universal_weights_into_module(model, zero_dir, dtype=dtype)
    print(f"Loaded {loaded} tensors; skipped {missing} dirs without a matching model parameter", flush=True)

    if loaded == 0:
        raise RuntimeError(f"No weights were loaded from {zero_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving HF checkpoint to {output_dir}", flush=True)
    state_dict = remap_state_dict_for_hub(dict(model.state_dict()))
    lm_head_path = zero_dir / "lm_head.weight" / "fp32.pt"
    if "lm_head.weight" not in state_dict and lm_head_path.is_file():
        state_dict["lm_head.weight"] = load_fp32_weight(lm_head_path).to(dtype=dtype)

    model.save_pretrained(
        output_dir,
        state_dict=state_dict,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    write_hub_config(output_dir, args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    processor.save_pretrained(output_dir)

    print(f"Done. HF-style checkpoint is in: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
