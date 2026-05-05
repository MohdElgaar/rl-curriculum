#!/usr/bin/env python3
"""Grouped bar chart of final eval/objective/*_correct_rate for paper conditions.

Data now comes from W&B directly (not from a stale JSON file) so the bars
share their **runsets** with ``build_wandb_comparison_report.py`` and
``plot_eval_objective_rewards_curriculum.py``.

Selection, per ``_runsets.py``:
  - learning_rate == 1e-6
  - model_name_or_path ∈ {Qwen3-0.6B, Qwen3-1.7B}
  - dataset_mixer_list == <training dataset>
  - shaping arms ∈ {baseline, shaping, curriculum α=1, curriculum α=10}
  - tags ∉ {broken-async, broken-ifeval-eval-log}

Metric: by default we use the per-benchmark ``*_correct_rate`` (bounded
in [0, 1], matches the bar chart's original semantics). Pass ``--reward`` to
switch to the raw ``*_reward`` metrics the report uses, or override with
``--metrics eval/objective/foo_bar``.

The default "paper" conditions are baseline, baseline with random-zero reward,
shaping-only, and curriculum α=10. Per (model, kind,
eval metric), seeds are aggregated into a mean plus SEM error bar.

Usage (from repo root):
  uv run python scripts/wandb_scripts/plot_multi_condition_bar.py
  uv run python scripts/wandb_scripts/plot_multi_condition_bar.py --dataset GSM
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _runsets import (  # noqa: E402  (path hack must run first)
    DATASET_ALIASES,
    MODEL_DISPLAY,
    MODEL_NAME_OR_PATH,
    PAPER_APPROACH_KINDS,
    DatasetSpec,
    classify_run_kind,
    dataset_filter_mongo,
    resolve_dataset,
)

BAR_COLORS: tuple[str, ...] = ("#4C72B0", "#9467BD", "#DD8452", "#55A868")

KIND_LABELS: dict[str, str] = {
    "baseline": "Baseline",
    "baseline_rz": "Baseline (random-zero)",
    "shaping": "Shaping only",
    "curr_a10": r"Curriculum ($\alpha{=}10$)",
}

DEFAULT_BENCHMARKS: tuple[str, ...] = (
    "ifbench",
    "ifeval",
    "gsm8k",
    "math",
    "code",
    "verifiable",
)

BENCHMARK_LABELS_BARE: dict[str, str] = {
    "ifbench": "IFBench",
    "ifeval": "IFEval",
    "gsm8k": "GSM8K",
    "math": "MATH",
    "code": "Code",
    "verifiable": "Verifiable",
}


def metric_label(metric: str) -> str:
    """Pretty label for an ``eval/objective/<name>_<suffix>`` metric."""
    name = metric.split("/")[-1]
    for suffix in ("_correct_rate", "_reward"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return BENCHMARK_LABELS_BARE.get(name, name)


def build_metric_list(benchmarks: list[str], suffix: str) -> list[str]:
    return [f"eval/objective/{b}{suffix}" for b in benchmarks]

CACHE_VERSION = 2  # v2: baseline vs baseline_rz (random_zero_reward)


def _flat_config(cfg: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in (cfg or {}).items():
        if isinstance(v, dict) and "value" in v:
            out[k] = v["value"]
        else:
            out[k] = v
    return out


def _final_metric_values(run: Any, metrics: set[str]) -> dict[str, float]:
    """Return the last non-null value per metric across the run history.

    ``scan_history`` returns exact rows (no sampling). We iterate once and keep
    the most recent value seen for each metric of interest.
    """
    keep: dict[str, float] = {}
    keys = ["_step", "global_step", *sorted(metrics)]
    for row in run.scan_history(keys=keys):
        for m in metrics:
            v = row.get(m)
            if v is None:
                continue
            try:
                keep[m] = float(v)
            except (TypeError, ValueError):
                continue
    return keep


def _collect_rows(
    api: Any,
    entity: str,
    project: str,
    dataset: DatasetSpec,
    metrics: set[str],
) -> list[dict[str, Any]]:
    """One row per W&B run with its final value for each tracked metric."""
    filters = dataset_filter_mongo(dataset)
    runs = api.runs(f"{entity}/{project}", filters=filters, per_page=400)
    rows: list[dict[str, Any]] = []
    for run in runs:
        cfg = _flat_config(run.config or {})
        kind = classify_run_kind(cfg, dataset.prefix)
        if kind is None:
            continue
        seed = cfg.get("seed", None)
        model_path = cfg.get("model_name_or_path")
        model_key = next(
            (k for k, v in MODEL_NAME_OR_PATH.items() if v == model_path),
            None,
        )
        if model_key is None:
            continue
        final_vals = _final_metric_values(run, metrics)
        if not final_vals:
            continue
        rows.append(
            {
                "run_id": run.id,
                "run_name": run.name,
                "model_key": model_key,
                "model_display": MODEL_DISPLAY[model_key],
                "kind": kind,
                "seed": seed,
                **{m: final_vals.get(m) for m in metrics},
            }
        )
    return rows


def _summarize(
    rows: list[dict[str, Any]],
    models: list[str],
    kinds: list[str],
    metrics: list[str],
) -> dict[str, Any]:
    """Per (model, kind, metric): mean and SEM across seeds (ignoring None values)."""
    summary: dict[str, Any] = {}
    for model_key in models:
        summary[model_key] = {}
        for kind in kinds:
            summary[model_key][kind] = {}
            for m in metrics:
                values: list[float] = []
                for row in rows:
                    if row["model_key"] != model_key or row["kind"] != kind:
                        continue
                    v = row.get(m)
                    if v is None:
                        continue
                    try:
                        values.append(float(v))
                    except (TypeError, ValueError):
                        continue
                if not values:
                    summary[model_key][kind][m] = {"mean": None, "sem": None, "n": 0}
                    continue
                arr = np.asarray(values, dtype=float)
                mean = float(arr.mean())
                if len(arr) >= 2:
                    sem = float(arr.std(ddof=1) / np.sqrt(len(arr)))
                else:
                    sem = 0.0
                summary[model_key][kind][m] = {"mean": mean, "sem": sem, "n": int(len(arr))}
    return summary


def _write_cache(
    path: Path,
    entity: str,
    project: str,
    dataset: DatasetSpec,
    rows: list[dict[str, Any]],
    metrics: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": CACHE_VERSION,
                "entity": entity,
                "project": project,
                "dataset": dataset.key,
                "mixer_list": dataset.mixer_list,
                "metrics": metrics,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _read_cache(
    path: Path,
    entity: str,
    project: str,
    dataset: DatasetSpec,
    metrics: list[str],
) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if raw.get("version") != CACHE_VERSION:
        return None
    if raw.get("entity") != entity or raw.get("project") != project:
        return None
    if raw.get("dataset") != dataset.key:
        return None
    if set(raw.get("metrics") or []) != set(metrics):
        return None
    rows = raw.get("rows")
    if not isinstance(rows, list):
        return None
    return rows


def _plot(
    summary: dict[str, Any],
    models: list[str],
    kinds: list[str],
    metrics: list[str],
    *,
    show_errorbar: bool,
    out_pdf: Path,
    title_suffix: str,
    y_label: str,
    bounded_unit_axis: bool,
) -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")

    plt.rcParams.update(
        {
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    width = max(0.14, 0.72 / max(1, len(kinds)))
    x = np.arange(len(metrics))
    fig, axes = plt.subplots(1, len(models), figsize=(5.1 * len(models), 3.8), sharey=True)
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, model_key in zip(axes_flat, models):
        for i, kind in enumerate(kinds):
            means = [summary[model_key][kind][m]["mean"] for m in metrics]
            sems = [summary[model_key][kind][m]["sem"] for m in metrics]
            means_arr = np.asarray([0.0 if v is None else v for v in means], dtype=float)
            sems_arr = np.asarray([0.0 if v is None else v for v in sems], dtype=float)
            offset = (i - (len(kinds) - 1) / 2.0) * width
            ax.bar(
                x + offset,
                means_arr,
                width,
                label=KIND_LABELS.get(kind, kind),
                color=BAR_COLORS[i % len(BAR_COLORS)],
                edgecolor="0.25",
                linewidth=0.4,
                yerr=sems_arr if show_errorbar else None,
                capsize=2.5 if show_errorbar else 0,
                ecolor="0.25",
                error_kw={"elinewidth": 0.8},
            )
        ax.set_xticks(x)
        ax.set_xticklabels([metric_label(m) for m in metrics], rotation=22, ha="right")
        if ax is axes_flat[0]:
            ax.set_ylabel(y_label)
        ax.set_title(MODEL_DISPLAY[model_key])
        if bounded_unit_axis:
            ax.set_ylim(0, 1.02)
            ax.set_yticks(np.linspace(0, 1.0, 6))
        ax.grid(True, axis="y", alpha=0.35, linestyle=":")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(kinds),
        frameon=True,
        fancybox=False,
        edgecolor="0.82",
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(title_suffix, y=1.08, fontsize=10.5)
    fig.tight_layout(rect=[0.02, 0, 1, 0.88])
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Wrote {out_pdf}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entity", default="mohdelgaar")
    p.add_argument("--project", default="open_instruct_internal")
    p.add_argument(
        "--dataset",
        default="IF-RLVR",
        choices=sorted(set(DATASET_ALIASES)),
        help="Training dataset (default: IF-RLVR).",
    )
    p.add_argument(
        "--kinds",
        nargs="+",
        default=list(PAPER_APPROACH_KINDS),
        help=f"Approach kinds to show (default: {list(PAPER_APPROACH_KINDS)}).",
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help=(
            "Eval metrics to show as bars (full names, e.g. "
            "'eval/objective/ifeval_correct_rate'). Default: derive from "
            "--benchmarks + --reward/--correct-rate."
        ),
    )
    p.add_argument(
        "--benchmarks",
        nargs="+",
        default=list(DEFAULT_BENCHMARKS),
        help=f"Eval benchmark stems to plot (default: {list(DEFAULT_BENCHMARKS)}).",
    )
    p.add_argument(
        "--reward",
        action="store_true",
        help="Plot *_reward instead of *_correct_rate.",
    )
    p.add_argument("--out-dir", type=Path, default=None, help="Default: ../figures")
    p.add_argument("--out-basename", type=str, default=None)
    p.add_argument("--cache-path", type=Path, default=None)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--refresh", action="store_true")
    p.add_argument(
        "--no-stderr",
        action="store_true",
        help="Hide ±SEM error bars.",
    )
    return p.parse_args()


def default_cache_path(dataset_key: str, suffix: str) -> Path:
    slug = dataset_key.replace("-", "_").lower()
    metric_kind = "reward" if suffix == "_reward" else "correct_rate"
    return (
        Path(__file__).resolve().parent
        / "cache"
        / f".wandb_multi_condition_bar_{slug}_{metric_kind}.json"
    )


def main() -> None:
    args = parse_args()
    dataset: DatasetSpec = resolve_dataset(args.dataset)

    unknown_kinds = [k for k in args.kinds if k not in KIND_LABELS]
    if unknown_kinds:
        raise SystemExit(
            f"Unknown approach kind(s): {unknown_kinds}; "
            f"valid options are {sorted(KIND_LABELS)}."
        )

    suffix = "_reward" if args.reward else "_correct_rate"
    metric_kind_slug = "reward" if args.reward else "correct_rate"

    out_dir = args.out_dir or Path(__file__).resolve().parents[1] / "figures"
    out_basename = args.out_basename or (
        f"multi_condition_{metric_kind_slug}_bar_"
        + dataset.key.replace("-", "_").lower()
    )
    out_pdf = out_dir / f"{out_basename}.pdf"

    metrics = list(args.metrics) if args.metrics else build_metric_list(args.benchmarks, suffix)

    cache_path = args.cache_path or default_cache_path(dataset.key, suffix)
    use_cache = not args.no_cache

    rows: list[dict[str, Any]] | None = None
    if use_cache and not args.refresh:
        rows = _read_cache(cache_path, args.entity, args.project, dataset, metrics)
        if rows is not None:
            print(f"Loaded bar data from cache: {cache_path}")

    if rows is None:
        api = wandb.Api(timeout=180)
        rows = _collect_rows(api, args.entity, args.project, dataset, set(metrics))
        if not rows:
            raise SystemExit(
                f"No runs found for dataset {dataset.key} with the shared filters. "
                "Check entity/project/auth."
            )
        if use_cache:
            _write_cache(cache_path, args.entity, args.project, dataset, rows, metrics)
            print(f"Wrote bar data cache: {cache_path}")

    metrics_present = [
        m for m in metrics if any((r.get(m) is not None) for r in rows)
    ]
    if not metrics_present:
        raise SystemExit("None of the requested metrics were logged by any run.")

    models = sorted(MODEL_NAME_OR_PATH.keys(), key=lambda k: float(k.rstrip("B")))
    summary = _summarize(rows, models, list(args.kinds), metrics_present)

    print(f"Summary ({dataset.key}, suffix={suffix}):")
    for model_key in models:
        for kind in args.kinds:
            row_bits = []
            for m in metrics_present:
                cell = summary[model_key][kind][m]
                if cell["mean"] is None:
                    row_bits.append(f"{metric_label(m)}=NA")
                else:
                    row_bits.append(
                        f"{metric_label(m)}={cell['mean']:.3f}±{cell['sem']:.3f}(n={cell['n']})"
                    )
            print(f"  {MODEL_DISPLAY[model_key]} · {KIND_LABELS.get(kind, kind)}: " + " ".join(row_bits))

    y_label = "Final eval reward" if args.reward else "Final correct rate"
    bounded_unit_axis = suffix == "_correct_rate"
    _plot(
        summary,
        models,
        list(args.kinds),
        metrics_present,
        show_errorbar=not args.no_stderr,
        out_pdf=out_pdf,
        title_suffix=f"{y_label} · {dataset.display}",
        y_label=y_label,
        bounded_unit_axis=bounded_unit_axis,
    )


if __name__ == "__main__":
    main()
