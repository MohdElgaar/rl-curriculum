#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any


DEFAULT_EVAL_LIST = [
    "mohdelgaar/ifeval_rlvr",
    "32",
    "mohdelgaar/ifbench_rlvr",
    "64",
    "allenai/aime2024-25-rlvr",
    "32",
    "allenai/aime2024-25-rlvr",
    "32",
    "allenai/RLVR-MATH",
    "32",
    "allenai/RLVR-GSM",
    "32",
    "allenai/rlvr-code-data-python-r1-format-filtered",
    "32",
]
DEFAULT_EVAL_SPLITS = ["train", "train", "test_2024", "test_2025", "train", "train", "train"]
INTERRUPTED_FAILURE_REASONS = {"allocation_lost"}


@dataclass
class RunSpec:
    stage: str
    name: str
    num_engines: int
    learner_layout: list[int]
    per_device_batch_size: int
    learning_rate: float
    num_training_steps: int
    local_eval_every: int
    enable_eval: bool


class AllocationInterrupted(RuntimeError):
    pass


def env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def env_int(name: str, default: int) -> int:
    value = env(name)
    return int(value) if value is not None else default


def env_float(name: str, default: float) -> float:
    value = env(name)
    return float(value) if value is not None else default


def env_bool(name: str, default: bool) -> bool:
    value = env(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def parse_csv_ints(raw: str) -> list[int]:
    return sorted({int(part.strip()) for part in raw.split(",") if part.strip()})


def parse_csv_floats(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def parse_eval_list(var_name: str, default: list[str]) -> list[str]:
    raw = env(var_name)
    return shlex.split(raw) if raw else list(default)


def append_jsonl(path: pathlib.Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        json.dump(record, f, sort_keys=True)
        f.write("\n")


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def spec_name_from_result(result: dict[str, Any]) -> str | None:
    spec = result.get("spec")
    if not isinstance(spec, dict):
        return None
    name = spec.get("name")
    return name if isinstance(name, str) and name else None


def is_reusable_result(result: dict[str, Any]) -> bool:
    status = result.get("status")
    failure_reason = result.get("failure_reason")
    if status == "success":
        return True
    return status == "failure" and failure_reason not in INTERRUPTED_FAILURE_REASONS


def build_result_indexes(
    records: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    reusable_by_name: dict[str, dict[str, Any]] = {}
    latest_by_name: dict[str, dict[str, Any]] = {}
    for record in records:
        name = spec_name_from_result(record)
        if name is None:
            continue
        latest_by_name[name] = record
        if is_reusable_result(record):
            reusable_by_name[name] = record
    return reusable_by_name, latest_by_name


def parse_gpu_count(token: str) -> int | None:
    match = re.search(r"(\d+)(?!.*\d)", token)
    return int(match.group(1)) if match else None


def discover_gpus_per_node() -> list[int]:
    job_id = env("SLURM_JOB_ID")
    if job_id:
        proc = subprocess.run(
            ["scontrol", "-d", "show", "job", job_id],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            gpus_per_node: list[int] = []
            for line in proc.stdout.splitlines():
                if "Nodes=" not in line or "GRES=" not in line:
                    continue
                match = re.search(r"GRES=[^:\s]+:[^:\s]+:(\d+)", line)
                if match:
                    gpus_per_node.append(int(match.group(1)))
            if gpus_per_node:
                return gpus_per_node

    slurm_gpus_on_node = env("SLURM_GPUS_ON_NODE")
    if slurm_gpus_on_node:
        parsed = parse_gpu_count(slurm_gpus_on_node)
        if parsed is not None:
            return [parsed]

    num_gpus = env("NUM_GPUS")
    if num_gpus:
        return [int(num_gpus)]

    raise RuntimeError("Unable to determine GPU allocation from SLURM or NUM_GPUS")


def auto_engine_candidates(max_engines: int) -> list[int]:
    if max_engines <= 8:
        return list(range(1, max_engines + 1))

    candidates = {1, max_engines, max(1, max_engines // 2)}
    value = 2
    while value < max_engines:
        candidates.add(value)
        value *= 2
    return sorted(c for c in candidates if 1 <= c <= max_engines)


def derive_learner_layout(gpus_per_node: list[int], engine_gpus: int) -> list[int] | None:
    learner_per_node = list(gpus_per_node)
    remaining = engine_gpus
    for index in range(len(learner_per_node) - 1, -1, -1):
        take = min(learner_per_node[index], remaining)
        learner_per_node[index] -= take
        remaining -= take
    if remaining > 0:
        return None
    return [count for count in learner_per_node if count > 0]


def build_candidate_name(spec: RunSpec) -> str:
    learners = sum(spec.learner_layout)
    return (
        f"{spec.stage}_eng{spec.num_engines}_learn{learners}_"
        f"bs{spec.per_device_batch_size}_lr{spec.learning_rate:g}"
    )


def tail_mean(values: list[float], window: int) -> float | None:
    if not values:
        return None
    tail = values[-min(window, len(values)) :]
    return sum(tail) / len(tail)


def summarize_metrics(records: list[dict[str, Any]], metric_window: int) -> dict[str, Any]:
    step_records = [record for record in records if record.get("kind") == "step"]
    eval_records = [record for record in records if record.get("kind") == "eval"]

    def metric_values(key: str) -> list[float]:
        values: list[float] = []
        for record in step_records:
            metrics = record.get("metrics", {})
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                values.append(float(value))
        return values

    def ratio_values(numerator_key: str, denominator_key: str = "time/total") -> list[float]:
        values: list[float] = []
        for record in step_records:
            metrics = record.get("metrics", {})
            numerator = metrics.get(numerator_key)
            denominator = metrics.get(denominator_key)
            if isinstance(numerator, (int, float)) and isinstance(denominator, (int, float)) and denominator > 0:
                values.append(float(numerator) / float(denominator))
        return values

    final_eval_score = None
    if eval_records:
        last_eval_metrics = eval_records[-1].get("metrics", {})
        value = last_eval_metrics.get("eval/scores")
        if isinstance(value, (int, float)):
            final_eval_score = float(value)

    return {
        "num_step_records": len(step_records),
        "num_eval_records": len(eval_records),
        "tail_mean_learner_tokens_per_second_step": tail_mean(
            metric_values("learner_tokens_per_second_step"), metric_window
        ),
        "tail_mean_scores": tail_mean(metric_values("scores"), metric_window),
        "tail_mean_trainer_idle_ratio": tail_mean(
            ratio_values("time/trainer_idle_waiting_for_inference"), metric_window
        ),
        "tail_mean_generation_idle_ratio": tail_mean(
            ratio_values("time/generation_idle_waiting_for_trainer"), metric_window
        ),
        "tail_mean_weight_sync_ratio": tail_mean(ratio_values("time/weight_sync"), metric_window),
        "tail_mean_actor_mfu": tail_mean(metric_values("actor_mfu"), metric_window),
        "tail_mean_learner_mfu": tail_mean(metric_values("learner_mfu"), metric_window),
        "final_eval_score": final_eval_score,
        "last_training_step": step_records[-1]["training_step"] if step_records else None,
    }


def meets_balance_constraints(summary: dict[str, Any]) -> bool:
    trainer_idle = summary.get("tail_mean_trainer_idle_ratio")
    generation_idle = summary.get("tail_mean_generation_idle_ratio")
    if trainer_idle is None or generation_idle is None:
        return False
    max_trainer_idle = env_float("AUTOTUNE_MAX_TRAINER_IDLE_RATIO", 0.35)
    max_generation_idle = env_float("AUTOTUNE_MAX_GENERATION_IDLE_RATIO", 0.35)
    return trainer_idle <= max_trainer_idle and generation_idle <= max_generation_idle


def hardware_score(summary: dict[str, Any]) -> float | None:
    throughput = summary.get("tail_mean_learner_tokens_per_second_step")
    if throughput is None or throughput <= 0:
        return None

    trainer_idle = summary.get("tail_mean_trainer_idle_ratio") or 0.0
    generation_idle = summary.get("tail_mean_generation_idle_ratio") or 0.0
    weight_sync = summary.get("tail_mean_weight_sync_ratio") or 0.0
    max_trainer_idle = env_float("AUTOTUNE_MAX_TRAINER_IDLE_RATIO", 0.35)
    max_generation_idle = env_float("AUTOTUNE_MAX_GENERATION_IDLE_RATIO", 0.35)
    trainer_over = max(0.0, trainer_idle - max_trainer_idle)
    generation_over = max(0.0, generation_idle - max_generation_idle)
    penalty = 1.0 + (4.0 * trainer_idle) + (4.0 * generation_idle) + (1.0 * weight_sync)
    penalty *= 1.0 + (10.0 * trainer_over) + (10.0 * generation_over)
    return float(throughput) / penalty


def lr_score(summary: dict[str, Any]) -> float | None:
    final_eval = summary.get("final_eval_score")
    if final_eval is not None:
        return float(final_eval)
    tail_scores = summary.get("tail_mean_scores")
    return float(tail_scores) if tail_scores is not None else None


def classify_failure(log_path: pathlib.Path, timed_out: bool) -> str:
    if timed_out:
        return "timeout"
    if not log_path.exists():
        return "missing_log"
    text = log_path.read_text(encoding="utf-8", errors="ignore").lower()[-50000:]
    if (
        "unable to confirm allocation for job" in text
        or "expired or invalid job" in text
        or "force terminated stepid" in text
        or "due to signal terminated" in text
        or "preempted" in text
    ):
        return "allocation_lost"
    if "out of memory" in text or "cuda oom" in text:
        return "oom"
    if "no learner gpus after vllm reservation" in text:
        return "invalid_layout"
    if "vllm needs" in text and "allocation is smaller" in text:
        return "invalid_vllm_reservation"
    return "runtime_error"


def shell_join(parts: list[str]) -> str:
    return shlex.join(parts)


def summarize_existing_results(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"success": 0, "failure": 0, "interrupted": 0}
    reusable_by_name, latest_by_name = build_result_indexes(records)
    for name, result in latest_by_name.items():
        if name in reusable_by_name:
            status = reusable_by_name[name].get("status")
        else:
            status = result.get("status")
        if status in counts:
            counts[status] += 1
    return counts


def build_train_command(spec: RunSpec, run_dir: pathlib.Path, project_root: pathlib.Path) -> list[str]:
    num_unique_prompts = env_int("NUM_UNIQUE_PROMPTS", 48)
    num_samples_per_prompt = env_int("NUM_SAMPLES_PER_PROMPT", 16)
    total_episodes = spec.num_training_steps * num_unique_prompts * num_samples_per_prompt
    output_dir = run_dir / "output"
    metrics_path = run_dir / "metrics.jsonl"

    cmd = [
        "srun",
        "-n",
        env("SLURM_NNODES", "1") or "1",
        "--ntasks-per-node=1",
        str(project_root / "scripts/slurm/if_rlvr_ray_node.sh"),
        "python",
        "-m",
        "open_instruct.grpo_fast",
        "--exp_name",
        spec.name,
        "--beta",
        env("BETA", "0.01") or "0.01",
        "--num_unique_prompts_rollout",
        str(num_unique_prompts),
        "--num_samples_per_prompt_rollout",
        str(num_samples_per_prompt),
        "--kl_estimator",
        env("KL_ESTIMATOR", "2") or "2",
        "--learning_rate",
        f"{spec.learning_rate:g}",
        "--dataset_local_cache_dir",
        env("DATASET_LOCAL_CACHE_DIR", "local_dataset_cache") or "local_dataset_cache",
        "--dataset_mixer_list",
        env("TRAIN_DATASET", "allenai/IF_multi_constraints_upto5") or "allenai/IF_multi_constraints_upto5",
        env("TRAIN_DATASET_FRACTION", "1.0") or "1.0",
        "--dataset_mixer_list_splits",
        env("TRAIN_SPLIT", "train") or "train",
    ]

    if spec.enable_eval:
        cmd.extend(
            [
                "--dataset_mixer_eval_list",
                *parse_eval_list("DATASET_MIXER_EVAL_LIST", DEFAULT_EVAL_LIST),
                "--dataset_mixer_eval_list_splits",
                *parse_eval_list("DATASET_MIXER_EVAL_LIST_SPLITS", DEFAULT_EVAL_SPLITS),
            ]
        )

    cmd.extend(
        [
            "--max_prompt_token_length",
            str(env_int("MAX_PROMPT_TOKEN_LENGTH", 2048)),
            "--response_length",
            str(env_int("RESPONSE_LENGTH", 2048)),
            "--pack_length",
            str(env_int("PACK_LENGTH", 4096)),
            "--model_name_or_path",
            env("MODEL_NAME", "allenai/Llama-3.1-Tulu-3-8B-DPO") or "allenai/Llama-3.1-Tulu-3-8B-DPO",
            "--apply_verifiable_reward",
            env("APPLY_VERIFIABLE_REWARD", "True") or "True",
            "--non_stop_penalty",
            env("NON_STOP_PENALTY", "True") or "True",
            "--non_stop_penalty_value",
            env("NON_STOP_PENALTY_VALUE", "0.0") or "0.0",
            "--temperature",
            env("TEMPERATURE", "1.0") or "1.0",
            "--total_episodes",
            str(total_episodes),
            "--num_training_steps",
            str(spec.num_training_steps),
            "--ifeval_reward_shaping",
            env("IFEVAL_REWARD_SHAPING", "False") or "False",
            "--ifeval_reward_shaping_curriculum",
            env("IFEVAL_REWARD_SHAPING_CURRICULUM", "False") or "False",
            "--ifeval_competence_c0",
            env("IFEVAL_COMPETENCE_C0", "0.1") or "0.1",
            "--ifeval_competence_alpha",
            env("IFEVAL_COMPETENCE_ALPHA", "1.0") or "1.0",
            "--ifeval_num_curriculum_steps",
            env("IFEVAL_NUM_CURRICULUM_STEPS", "-1") or "-1",
            "--math_reward_shaping",
            env("MATH_REWARD_SHAPING", "False") or "False",
            "--math_reward_shaping_curriculum",
            env("MATH_REWARD_SHAPING_CURRICULUM", "False") or "False",
            "--math_competence_c0",
            env("MATH_COMPETENCE_C0", "0.1") or "0.1",
            "--math_competence_alpha",
            env("MATH_COMPETENCE_ALPHA", "1.0") or "1.0",
            "--math_num_curriculum_steps",
            env("MATH_NUM_CURRICULUM_STEPS", "-1") or "-1",
            "--gsm_reward_shaping",
            env("GSM_REWARD_SHAPING", "False") or "False",
            "--gsm_reward_shaping_curriculum",
            env("GSM_REWARD_SHAPING_CURRICULUM", "False") or "False",
            "--gsm_competence_c0",
            env("GSM_COMPETENCE_C0", "0.1") or "0.1",
            "--gsm_competence_alpha",
            env("GSM_COMPETENCE_ALPHA", "1.0") or "1.0",
            "--gsm_num_curriculum_steps",
            env("GSM_NUM_CURRICULUM_STEPS", "-1") or "-1",
            "--deepspeed_stage",
            str(env_int("DEEPSPEED_STAGE", 2)),
            "--per_device_train_batch_size",
            str(spec.per_device_batch_size),
            "--num_mini_batches",
            str(env_int("NUM_MINI_BATCHES", 2)),
            "--num_learners_per_node",
            *[str(value) for value in spec.learner_layout],
            "--num_epochs",
            str(env_int("NUM_EPOCHS", 1)),
            "--vllm_tensor_parallel_size",
            str(env_int("VLLM_TP", 1)),
            "--vllm_num_engines",
            str(spec.num_engines),
            "--lr_scheduler_type",
            env("LR_SCHEDULER_TYPE", "constant") or "constant",
            "--async_steps",
            str(env_int("ASYNC_STEPS", 1)),
            "--seed",
            str(env_int("SEED", 1)),
            "--local_eval_every",
            str(spec.local_eval_every),
            "--eval_on_step_0",
            "False",
            "--save_freq",
            "-1",
            "--keep_last_n_checkpoints",
            "-1",
            "--checkpoint_state_freq",
            "-1",
            "--output_dir",
            str(output_dir),
            "--with_tracking",
            "False",
            "--push_to_hub",
            "False",
            "--try_auto_save_to_beaker",
            "False",
            "--enable_queue_dashboard",
            "False",
            "--save_final_model",
            "False",
            "--metrics_jsonl_path",
            str(metrics_path),
            "--inflight_updates",
            env("INFLIGHT_UPDATES", "True") or "True",
            "--code_pass_rate_reward_threshold",
            env("CODE_PASS_RATE_REWARD_THRESHOLD", "0.99") or "0.99",
            "--code_api_url",
            env(
                "CODE_API_URL",
                "https://p9f1719l7f.execute-api.us-west-2.amazonaws.com/prod/test_program",
            )
            or "https://p9f1719l7f.execute-api.us-west-2.amazonaws.com/prod/test_program",
        ]
    )

    if env_bool("GRADIENT_CHECKPOINTING", True):
        cmd.append("--gradient_checkpointing")

    return cmd


def run_candidate(
    spec: RunSpec,
    project_root: pathlib.Path,
    result_root: pathlib.Path,
    metric_window: int,
    candidate_index: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    run_dir = result_root / "runs" / spec.name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    metrics_path = run_dir / "metrics.jsonl"
    command = build_train_command(spec, run_dir, project_root)

    env_vars = os.environ.copy()
    env_vars["PROJECT_ROOT"] = str(project_root)
    env_vars["PYTHONUNBUFFERED"] = "1"

    job_id = env("SLURM_JOB_ID")
    port_seed = int(job_id) if job_id and job_id.isdigit() else os.getpid()
    env_vars["RAY_HEAD_PORT"] = str(30000 + ((port_seed + candidate_index * 17) % 20000))
    env_vars["RAY_DASHBOARD_PORT"] = str(50000 + ((port_seed + candidate_index * 17) % 10000))

    timed_out = False
    start_time = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {shell_join(command)}\n\n")
        log_file.flush()
        try:
            proc = subprocess.run(
                command,
                cwd=project_root / "open-instruct",
                env=env_vars,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=timeout_seconds if timeout_seconds > 0 else None,
            )
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            exit_code = 124
            log_file.write("\n[autotune] Candidate timed out\n")

    records = load_jsonl(metrics_path)
    summary = summarize_metrics(records, metric_window) if records else {}
    status = "success" if exit_code == 0 and summary.get("num_step_records", 0) > 0 else "failure"
    failure_reason = None if status == "success" else classify_failure(log_path, timed_out)
    if failure_reason in INTERRUPTED_FAILURE_REASONS:
        status = "interrupted"

    result = {
        "stage": spec.stage,
        "status": status,
        "failure_reason": failure_reason,
        "exit_code": exit_code,
        "duration_seconds": time.time() - start_time,
        "command": command,
        "command_shell": shell_join(command),
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "metrics_path": str(metrics_path),
        "spec": asdict(spec),
        "summary": summary,
    }
    if spec.stage == "hardware" and summary:
        result["passes_balance_constraints"] = meets_balance_constraints(summary)
    if status == "success":
        result["score"] = hardware_score(summary) if spec.stage == "hardware" else lr_score(summary)
    else:
        result["score"] = None
    return result


def choose_best(results: list[dict[str, Any]], *, stage: str) -> dict[str, Any]:
    successful = [result for result in results if result.get("status") == "success" and result.get("score") is not None]
    if not successful:
        raise RuntimeError("No successful candidates were found")

    def sort_key(result: dict[str, Any]) -> tuple[float, float, float]:
        balance_ok = 0.0
        if stage == "hardware":
            balance_ok = 1.0 if result.get("passes_balance_constraints") else 0.0
        throughput = result["summary"].get("tail_mean_learner_tokens_per_second_step") or 0.0
        return (balance_ok, float(result["score"]), float(throughput))

    return max(successful, key=sort_key)


def write_best_overrides(
    path: pathlib.Path,
    hardware_best: dict[str, Any],
    lr_best: dict[str, Any],
) -> None:
    hardware_spec = hardware_best["spec"]
    lr_spec = lr_best["spec"]
    learners_per_node = " ".join(str(value) for value in hardware_spec["learner_layout"])
    payload = "\n".join(
        [
            "#!/bin/bash",
            "# Generated by scripts/tune/if_rlvr_autotune.py",
            f"# Hardware score: {hardware_best['score']}",
            f"# LR score: {lr_best['score']}",
            f"# Learners per node for the tuned allocation: {learners_per_node}",
            f"export VLLM_NUM_ENGINES={hardware_spec['num_engines']}",
            f"export VLLM_TP={env_int('VLLM_TP', 1)}",
            f"export PER_DEVICE_BATCH_SIZE={hardware_spec['per_device_batch_size']}",
            f"export LEARNING_RATE={lr_spec['learning_rate']}",
            f"export AUTOTUNE_SELECTED_TOTAL_LEARNERS={sum(hardware_spec['learner_layout'])}",
            f'export AUTOTUNE_SELECTED_LEARNERS_PER_NODE="{learners_per_node}"',
            "",
        ]
    )
    path.write_text(payload, encoding="utf-8")
    path.chmod(0o755)


def write_submit_script(
    path: pathlib.Path,
    project_root: pathlib.Path,
    gpus_per_node: list[int],
    recommended_time: str,
    resolved_configs: list[str],
    best_override_path: pathlib.Path,
) -> None:
    total_gpus = sum(gpus_per_node)
    command = [
        "sbatch",
        "-G",
        str(total_gpus),
        "--time",
        recommended_time,
        str(project_root / "scripts/slurm/if_rlvr.sbatch"),
        *resolved_configs,
        str(best_override_path),
    ]
    payload = "\n".join(
        [
            "#!/bin/bash",
            "set -euo pipefail",
            "",
            shell_join(command),
            "",
        ]
    )
    path.write_text(payload, encoding="utf-8")
    path.chmod(0o755)


def maybe_reuse_result(
    spec: RunSpec,
    reusable_results: dict[str, dict[str, Any]],
    latest_results: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    reusable = reusable_results.get(spec.name)
    if reusable is not None:
        return reusable
    latest = latest_results.get(spec.name)
    if latest is not None and latest.get("status") == "interrupted":
        return None
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autotune IF-RLVR hardware and learning-rate settings")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--pilot-steps", type=int, default=env_int("AUTOTUNE_PILOT_STEPS", 2))
    parser.add_argument("--lr-steps", type=int, default=env_int("AUTOTUNE_LR_STEPS", 64))
    parser.add_argument("--metric-window", type=int, default=env_int("AUTOTUNE_METRIC_WINDOW", 5))
    parser.add_argument("--batch-sizes", default=env("AUTOTUNE_BATCH_SIZES", "1,2,4,8,16,32"))
    parser.add_argument("--learning-rates", default=env("AUTOTUNE_LEARNING_RATES", "1e-7,5e-7,1e-6"))
    parser.add_argument("--fixed-lr", type=float, default=env_float("AUTOTUNE_FIXED_LR", 5e-7))
    parser.add_argument("--engine-candidates", default=env("AUTOTUNE_ENGINE_CANDIDATES", ""))
    parser.add_argument("--run-timeout-seconds", type=int, default=env_int("AUTOTUNE_RUN_TIMEOUT_SECONDS", 0))
    parser.add_argument(
        "--inter-run-sleep-seconds",
        type=int,
        default=env_int("AUTOTUNE_INTER_RUN_SLEEP_SECONDS", 5),
    )
    parser.add_argument("--recommended-time", default=env("AUTOTUNE_FINAL_TIME", env("SLURM_TIMELIMIT", "48:00:00")))
    parser.add_argument("--resolved-config", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = pathlib.Path(args.project_root).resolve()
    result_root = pathlib.Path(args.result_root).resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    results_jsonl = result_root / "results.jsonl"
    existing_records = load_jsonl(results_jsonl)
    reusable_results, latest_results = build_result_indexes(existing_records)
    if existing_records:
        summary_counts = summarize_existing_results(existing_records)
        print(
            "[autotune] Resuming from existing results: "
            f"{summary_counts['success']} reusable successes, "
            f"{summary_counts['failure']} reusable failures, "
            f"{summary_counts['interrupted']} interrupted candidates"
        )

    gpus_per_node = discover_gpus_per_node()
    total_gpus = sum(gpus_per_node)
    vllm_tp = env_int("VLLM_TP", 1)
    max_engines = (total_gpus - 1) // vllm_tp
    if max_engines < 1:
        raise RuntimeError(
            f"Need at least one learner GPU after reserving vLLM engines, but allocation only has {total_gpus} GPUs"
        )

    engine_candidates = (
        [candidate for candidate in parse_csv_ints(args.engine_candidates) if 1 <= candidate <= max_engines]
        if args.engine_candidates
        else auto_engine_candidates(max_engines)
    )
    if not engine_candidates:
        raise RuntimeError("No feasible engine candidates remain after filtering")

    batch_sizes = [batch_size for batch_size in parse_csv_ints(args.batch_sizes) if batch_size > 0]
    learning_rates = parse_csv_floats(args.learning_rates)

    print(
        f"[autotune] Allocation gpus_per_node={gpus_per_node}, engine_candidates={engine_candidates}, "
        f"batch_sizes={batch_sizes}, learning_rates={learning_rates}"
    )

    candidate_index = 0
    hardware_results: list[dict[str, Any]] = []
    for num_engines in engine_candidates:
        learner_layout = derive_learner_layout(gpus_per_node, num_engines * vllm_tp)
        if not learner_layout:
            continue

        print(
            f"[autotune] Exploring hardware candidates with num_engines={num_engines}, "
            f"learner_layout={learner_layout}"
        )
        oom_encountered = False
        for per_device_batch_size in batch_sizes:
            if oom_encountered:
                break

            spec = RunSpec(
                stage="hardware",
                name=build_candidate_name(
                    RunSpec(
                        stage="hardware",
                        name="",
                        num_engines=num_engines,
                        learner_layout=learner_layout,
                        per_device_batch_size=per_device_batch_size,
                        learning_rate=args.fixed_lr,
                        num_training_steps=args.pilot_steps,
                        local_eval_every=-1,
                        enable_eval=False,
                    )
                ),
                num_engines=num_engines,
                learner_layout=learner_layout,
                per_device_batch_size=per_device_batch_size,
                learning_rate=args.fixed_lr,
                num_training_steps=args.pilot_steps,
                local_eval_every=-1,
                enable_eval=False,
            )
            cached_result = maybe_reuse_result(spec, reusable_results, latest_results)
            if cached_result is not None:
                hardware_results.append(cached_result)
                print(
                    f"[autotune] {spec.name}: reusing cached {cached_result['status']} "
                    f"score={cached_result.get('score')} failure={cached_result.get('failure_reason')}"
                )
                if cached_result.get("failure_reason") == "oom":
                    oom_encountered = True
                continue
            result = run_candidate(
                spec,
                project_root,
                result_root,
                args.metric_window,
                candidate_index,
                args.run_timeout_seconds,
            )
            candidate_index += 1
            append_jsonl(results_jsonl, result)
            latest_results[spec.name] = result
            if is_reusable_result(result):
                reusable_results[spec.name] = result
            hardware_results.append(result)
            print(
                f"[autotune] {spec.name}: status={result['status']} "
                f"score={result['score']} failure={result['failure_reason']}"
            )
            if result["status"] == "interrupted":
                raise AllocationInterrupted(
                    f"Allocation lost while running {spec.name}; stop now and let SLURM retry later"
                )
            if result["failure_reason"] == "oom":
                oom_encountered = True
            time.sleep(args.inter_run_sleep_seconds)

    hardware_best = choose_best(hardware_results, stage="hardware")
    print(f"[autotune] Best hardware candidate: {hardware_best['spec']}")
    if not hardware_best.get("passes_balance_constraints"):
        print(
            "[autotune] WARNING: best hardware candidate does not satisfy the default balance constraints; "
            "picked the least-bad option by penalized throughput"
        )

    lr_results: list[dict[str, Any]] = []
    hardware_spec = hardware_best["spec"]
    for learning_rate in learning_rates:
        spec = RunSpec(
            stage="lr",
            name=build_candidate_name(
                RunSpec(
                    stage="lr",
                    name="",
                    num_engines=hardware_spec["num_engines"],
                    learner_layout=hardware_spec["learner_layout"],
                    per_device_batch_size=hardware_spec["per_device_batch_size"],
                    learning_rate=learning_rate,
                    num_training_steps=args.lr_steps,
                    local_eval_every=args.lr_steps,
                    enable_eval=True,
                )
            ),
            num_engines=hardware_spec["num_engines"],
            learner_layout=hardware_spec["learner_layout"],
            per_device_batch_size=hardware_spec["per_device_batch_size"],
            learning_rate=learning_rate,
            num_training_steps=args.lr_steps,
            local_eval_every=args.lr_steps,
            enable_eval=True,
        )
        cached_result = maybe_reuse_result(spec, reusable_results, latest_results)
        if cached_result is not None:
            lr_results.append(cached_result)
            print(
                f"[autotune] {spec.name}: reusing cached {cached_result['status']} "
                f"score={cached_result.get('score')} failure={cached_result.get('failure_reason')}"
            )
            continue
        result = run_candidate(
            spec,
            project_root,
            result_root,
            args.metric_window,
            candidate_index,
            args.run_timeout_seconds,
        )
        candidate_index += 1
        append_jsonl(results_jsonl, result)
        latest_results[spec.name] = result
        if is_reusable_result(result):
            reusable_results[spec.name] = result
        lr_results.append(result)
        print(f"[autotune] {spec.name}: status={result['status']} score={result['score']}")
        if result["status"] == "interrupted":
            raise AllocationInterrupted(
                f"Allocation lost while running {spec.name}; stop now and let SLURM retry later"
            )
        time.sleep(args.inter_run_sleep_seconds)

    lr_best = choose_best(lr_results, stage="lr")
    print(f"[autotune] Best LR candidate: {lr_best['spec']}")

    best_overrides_path = result_root / "best_overrides.sh"
    write_best_overrides(best_overrides_path, hardware_best, lr_best)

    submit_script_path = result_root / "submit_best.sh"
    if args.resolved_config:
        write_submit_script(
            submit_script_path,
            project_root,
            gpus_per_node,
            args.recommended_time,
            args.resolved_config,
            best_overrides_path,
        )

    summary = {
        "project_root": str(project_root),
        "result_root": str(result_root),
        "gpus_per_node": gpus_per_node,
        "batch_sizes": batch_sizes,
        "learning_rates": learning_rates,
        "engine_candidates": engine_candidates,
        "hardware_best": hardware_best,
        "lr_best": lr_best,
        "best_overrides_path": str(best_overrides_path),
        "submit_script_path": str(submit_script_path) if args.resolved_config else None,
    }
    write_json(result_root / "summary.json", summary)

    print(f"[autotune] Wrote summary to {result_root / 'summary.json'}")
    print(f"[autotune] Wrote best overrides to {best_overrides_path}")
    if args.resolved_config:
        print(f"[autotune] Wrote submit helper to {submit_script_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AllocationInterrupted as exc:
        print(f"[autotune] INTERRUPTED: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except Exception as exc:
        print(f"[autotune] ERROR: {exc}", file=sys.stderr)
        raise
