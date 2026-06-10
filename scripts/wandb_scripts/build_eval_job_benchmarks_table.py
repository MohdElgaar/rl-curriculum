#!/usr/bin/env python3
"""Export lm-eval-harness benchmark results from W&B eval jobs to CSV.

Uses the same run selection as ``plot_eval_job_benchmarks.py``: training jobs
filtered by ``jobType == eval``, dataset mixer, shaping arms, and optional base
model evals. Each row is one (model × training dataset × approach) cell with
mean scores across seeds and lm-eval bootstrap stderr (also averaged across
seeds when multiple eval runs exist).

Benchmark columns use the short names from ``METRIC_SHORT_LABELS`` in
``plot_eval_job_benchmarks.py``.

Example (from repo root):

  uv run python scripts/wandb_scripts/build_eval_job_benchmarks_table.py \\
      --out scripts/figures/eval_job_benchmarks_table.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path
from typing import Any

import numpy as np
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _runsets import (  # noqa: E402
    APPROACH_ORDER,
    APPROACH_PLAIN_LABELS,
    DATASETS,
    MODEL_DISPLAY,
    MODEL_NAME_OR_PATH,
    DatasetSpec,
    kind_to_group_key,
    model_sort_key,
    resolve_dataset,
)
from plot_eval_job_benchmarks import (  # noqa: E402
    BASE_MODEL_KIND,
    BASE_MODEL_LABEL,
    DEFAULT_METRICS,
    METRIC_SHORT_LABELS,
    collect_base_model_rows,
    collect_rows_for_dataset,
    ordered_model_keys,
)

RowKey = tuple[str, str, str]


def metric_stderr_key(metric: str) -> str:
    """W&B summary key for lm-eval bootstrap stderr (see ``wandb_logger.py``)."""
    if "/" not in metric:
        return f"{metric}_stderr"
    task, rest = metric.split("/", 1)
    if "," in rest:
        name, suffix = rest.split(",", 1)
        return f"{task}/{name}_stderr,{suffix}"
    return f"{task}/{rest}_stderr"


def _mean_or_none(vals: list[float]) -> float | None:
    if not vals:
        return None
    return float(np.asarray(vals, dtype=float).mean())


def _settings_from_kind(kind: str) -> dict[str, Any]:
    if kind == BASE_MODEL_KIND:
        return {
            "approach": BASE_MODEL_LABEL,
            "reward_shaping": "",
            "reward_shaping_curriculum": "",
            "competence_alpha": "",
            "random_zero_reward": "",
            "is_base_model": True,
        }
    rs, curr, alpha, rz = kind_to_group_key(kind)
    return {
        "approach": APPROACH_PLAIN_LABELS.get(kind, kind),
        "reward_shaping": rs,
        "reward_shaping_curriculum": curr,
        "competence_alpha": "" if alpha is None else alpha,
        "random_zero_reward": rz,
        "is_base_model": False,
    }


def _aggregate(
    rows: list[dict[str, Any]],
    metrics: list[str],
) -> dict[RowKey, dict[str, dict[str, float | None]]]:
    """Mean score and mean lm-eval stderr across seeds for each cell."""
    groups: dict[RowKey, list[dict[str, Any]]] = {}
    for r in rows:
        key: RowKey = (str(r["model_key"]), str(r["dataset_key"]), str(r["kind"]))
        groups.setdefault(key, []).append(r)

    out: dict[RowKey, dict[str, dict[str, float | None]]] = {}
    for key, group_rows in groups.items():
        cell: dict[str, dict[str, float | None]] = {}
        for m in metrics:
            score_vals: list[float] = []
            stderr_vals: list[float] = []
            stderr_key = metric_stderr_key(m)
            for r in group_rows:
                v = r.get(m)
                if v is not None:
                    try:
                        score_vals.append(float(v))
                    except (TypeError, ValueError):
                        pass
                se = r.get(stderr_key)
                if se is not None:
                    try:
                        stderr_vals.append(float(se))
                    except (TypeError, ValueError):
                        pass
            cell[m] = {
                "mean": _mean_or_none(score_vals),
                "stderr": _mean_or_none(stderr_vals),
            }
        out[key] = cell
    return out


def _sort_row_keys(keys: list[RowKey]) -> list[RowKey]:
    dataset_order = {d.key: i for i, d in enumerate(DATASETS)}
    kind_order = {k: i for i, k in enumerate((*APPROACH_ORDER, BASE_MODEL_KIND))}

    def sort_key(item: RowKey) -> tuple[Any, ...]:
        model_key, dataset_key, kind = item
        ds_idx = dataset_order.get(dataset_key, len(dataset_order))
        if kind == BASE_MODEL_KIND:
            ds_idx = -1
        return (model_sort_key(model_key), ds_idx, kind_order.get(kind, 999), model_key, dataset_key, kind)

    return sorted(keys, key=sort_key)


def render_csv(
    agg: dict[RowKey, dict[str, dict[str, float | None]]],
    metrics: list[str],
) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)

    setting_cols = [
        "model",
        "training_dataset",
        "approach",
        "reward_shaping",
        "reward_shaping_curriculum",
        "competence_alpha",
        "random_zero_reward",
        "is_base_model",
    ]
    metric_cols: list[str] = []
    for m in metrics:
        short = METRIC_SHORT_LABELS.get(m, m)
        metric_cols.extend([f"{short}_mean", f"{short}_stderr"])
    writer.writerow([*setting_cols, *metric_cols])

    for model_key, dataset_key, kind in _sort_row_keys(list(agg.keys())):
        settings = _settings_from_kind(kind)
        if kind == BASE_MODEL_KIND:
            training_dataset = ""
        else:
            training_dataset = dataset_key
        row: list[Any] = [
            MODEL_DISPLAY.get(model_key, model_key),
            training_dataset,
            settings["approach"],
            settings["reward_shaping"],
            settings["reward_shaping_curriculum"],
            settings["competence_alpha"],
            settings["random_zero_reward"],
            settings["is_base_model"],
        ]
        cells = agg[(model_key, dataset_key, kind)]
        for m in metrics:
            cell = cells[m]
            row.extend([
                "" if cell["mean"] is None else f"{cell['mean']:.6f}",
                "" if cell["stderr"] is None else f"{cell['stderr']:.6f}",
            ])
        writer.writerow(row)
    return buf.getvalue()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entity", default="mohdelgaar")
    p.add_argument("--project", default="open_instruct_internal")
    p.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Training dataset alias (repeat for several). Default: all datasets.",
    )
    p.add_argument("--job-type", default="eval")
    p.add_argument(
        "--kinds",
        nargs="+",
        default=list(APPROACH_ORDER),
        help=f"Approach arms to include (default: {list(APPROACH_ORDER)}).",
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="Full W&B metric keys (defaults to lm-eval harness list).",
    )
    p.add_argument(
        "--qwen-only",
        action="store_true",
        help="Restrict to MODEL_NAME_OR_PATH checkpoints (default: include all models with eval runs).",
    )
    p.add_argument(
        "--no-base-models",
        action="store_true",
        help="Omit base-model eval runs.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: ../figures/eval_job_benchmarks_table.csv).",
    )
    return p.parse_args()


def resolve_datasets(cli_datasets: list[str] | None) -> list[DatasetSpec]:
    if not cli_datasets:
        return list(DATASETS)
    out: list[DatasetSpec] = []
    seen: set[str] = set()
    for name in cli_datasets:
        ds = resolve_dataset(name)
        if ds.key not in seen:
            seen.add(ds.key)
            out.append(ds)
    return out


def main() -> None:
    args = parse_args()
    datasets = resolve_datasets(args.dataset)

    include_base_models = not args.no_base_models
    allowed_kinds = set(APPROACH_ORDER)
    kinds = list(args.kinds)
    if include_base_models:
        allowed_kinds.add(BASE_MODEL_KIND)
        if BASE_MODEL_KIND not in kinds:
            kinds = [BASE_MODEL_KIND, *kinds]
    else:
        kinds = [k for k in kinds if k != BASE_MODEL_KIND]

    unknown = [k for k in kinds if k not in allowed_kinds]
    if unknown:
        raise SystemExit(f"Unknown --kinds {unknown}; choose from {sorted(allowed_kinds)}")

    kinds_allow = frozenset(kinds)
    metrics = list(args.metrics)
    stderr_keys = {metric_stderr_key(m) for m in metrics}
    metrics_set = set(metrics) | stderr_keys
    restrict_models = args.qwen_only

    out_path = args.out or (
        Path(__file__).resolve().parents[1] / "figures" / "eval_job_benchmarks_table.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    api = wandb.Api(timeout=180)
    rows: list[dict[str, Any]] = []

    if include_base_models:
        rows.extend(
            collect_base_model_rows(
                api,
                args.entity,
                args.project,
                metrics_set,
                job_type=args.job_type,
                restrict_models=restrict_models,
            )
        )

    for ds in datasets:
        rows.extend(
            collect_rows_for_dataset(
                api,
                args.entity,
                args.project,
                ds,
                metrics_set,
                job_type=args.job_type,
                restrict_models=restrict_models,
                kinds_allow=kinds_allow,
            )
        )

    if not rows:
        raise SystemExit("No eval runs matched filters; check WANDB_API_KEY/project.")

    metrics_present = [m for m in metrics if any(r.get(m) is not None for r in rows)]
    if not metrics_present:
        raise SystemExit("None of the requested metrics were logged on any run.")

    agg = _aggregate(rows, metrics_present)
    out_path.write_text(render_csv(agg, metrics_present), encoding="utf-8")
    print(f"Wrote CSV: {out_path} ({len(agg)} rows, {len(rows)} runs)")

    model_keys = ordered_model_keys({str(r["model_key"]) for r in rows})
    print(f"Models: {', '.join(MODEL_DISPLAY.get(m, m) for m in model_keys)}")
    print(f"Metrics: {', '.join(METRIC_SHORT_LABELS.get(m, m) for m in metrics_present)}")
    kinds_seen = sorted({str(r["kind"]) for r in rows})
    print(f"Approaches: {', '.join(kinds_seen)}")


if __name__ == "__main__":
    main()
