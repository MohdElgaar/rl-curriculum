#!/usr/bin/env python3
"""Grouped bar charts of lm-eval-harness metrics from W&B runs with job type ``eval``.

Runs are selected to match the paper run sets in ``_runsets.py`` except that we
**do not** filter on ``learning_rate`` (evaluation jobs often omit it).

- **Job type**: Mongo filter ``jobType == eval`` (CLI: ``--job-type``), with a
  client-side sanity check on ``run.job_type``.
- **Train data**: ``config.dataset_mixer_list`` must match a known
  :class:`DatasetSpec` (same list as ``DATASETS`` in ``_runsets.py``).
- **Approach**: ``classify_run_kind`` using the dataset prefix — reward shaping,
  curriculum, and competence α (plus baseline random-zero).
- **Models**: default to the Qwen3 paths in ``MODEL_NAME_OR_PATH``; pass
  ``--all-models`` to include any ``model_name_or_path`` (unknown paths get a
  slug key).
- **Base model evals**: Included by default. Runs qualify if ``config.eval_is_base_model``
  is truthy or ``config.eval_approach_kind`` matches base-style labels (see
  ``is_base_model_eval_cfg``). Use ``--no-base-models`` to omit that bar series.

One PDF is written **per training dataset** (filename encodes the dataset key).
Each figure has one column per model and grouped bars per approach (mean ± SEM
across ``config.seed``).

Default metrics (flat W&B summary / history keys as logged by ``WandbLogger``):

  mbpp_plus_instruct/pass_at_1,extract_code
  mbpp_instruct/pass_at_1,extract_code
  ifeval/prompt_level_loose_acc
  ifbench/prompt_level_loose_acc,pass_at_1_repeats
  humaneval_plus_instruct/pass@1,create_test
  humaneval_instruct/pass@1,create_test
  hendrycks_math500/exact_match
  aime26/exact_match,keep_repeats
  aime25/exact_match,keep_repeats

Usage (from repo root):

  uv run python scripts/wandb_scripts/plot_eval_job_benchmarks.py
  uv run python scripts/wandb_scripts/plot_eval_job_benchmarks.py --no-base-models \\
      --dataset IF-RLVR
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
from _runsets import (  # noqa: E402
    APPROACH_ORDER,
    APPROACH_TEX_LABELS,
    DATASETS,
    MODEL_DISPLAY,
    MODEL_NAME_OR_PATH,
    DatasetSpec,
    model_sort_key,
    classify_run_kind,
    excluded_run_tags_mongo,
    shaping_arms_mongo,
    resolve_dataset,
)

DEFAULT_METRICS: tuple[str, ...] = (
    "mbpp_plus_instruct/pass_at_1,extract_code",
    "mbpp_instruct/pass_at_1,extract_code",
    "ifeval/prompt_level_loose_acc",
    "ifbench/prompt_level_loose_acc,pass_at_1_repeats",
    "humaneval_plus_instruct/pass@1,create_test",
    "humaneval_instruct/pass@1,create_test",
    "hendrycks_math500/exact_match",
    "aime26/exact_match,keep_repeats",
    "aime25/exact_match,keep_repeats",
)

METRIC_SHORT_LABELS: dict[str, str] = {
    "mbpp_plus_instruct/pass_at_1,extract_code": "MBPP+ instruct",
    "mbpp_instruct/pass_at_1,extract_code": "MBPP instruct",
    "ifeval/prompt_level_loose_acc": "IFEval (loose)",
    "ifbench/prompt_level_loose_acc,pass_at_1_repeats": "IFBench (loose)",
    "humaneval_plus_instruct/pass@1,create_test": "HE+ instruct",
    "humaneval_instruct/pass@1,create_test": "HumanEval instruct",
    "hendrycks_math500/exact_match": "MATH-500",
    "aime26/exact_match,keep_repeats": "AIME26",
    "aime25/exact_match,keep_repeats": "AIME25",
}

BAR_COLORS: tuple[str, ...] = (
    "#4C72B0",
    "#55A868",
    "#C44E52",
    "#8172B3",
    "#CCB974",
    "#64B5CD",
)

CACHE_VERSION = 3
BASE_MODEL_KIND = "base_model"
BASE_MODEL_LABEL = "Base model"

# eval_approach_kind values (after lower + strip + space/dash → _) treated as base-model evals.
_BASE_APPROACH_KIND_SLUGS: frozenset[str] = frozenset(
    {
        "base",
        "base_model",
        "basemodel",
        "foundation",
        "pretrained",
        "raw",
        "raw_model",
    }
)


def _flat_config(cfg: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in (cfg or {}).items():
        if isinstance(v, dict) and "value" in v:
            out[k] = v["value"]
        else:
            out[k] = v
    return out


def _coerce_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return x != 0
    if isinstance(x, str):
        return x.strip().lower() in ("1", "true", "yes", "on")
    return False


def _normalize_eval_approach_kind(v: Any) -> str | None:
    if v is None or not isinstance(v, str):
        return None
    s = v.strip().lower().replace("-", "_")
    for ch in (" ", "\t"):
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s or None


def is_base_model_eval_cfg(cfg: dict[str, Any]) -> bool:
    """True if this run is a base / foundation checkpoint eval (not an RL arm)."""
    if _coerce_bool(cfg.get("eval_is_base_model")):
        return True
    slug = _normalize_eval_approach_kind(cfg.get("eval_approach_kind"))
    if slug is None:
        return False
    if slug in _BASE_APPROACH_KIND_SLUGS:
        return True
    if slug.startswith("base_") or slug.endswith("_base"):
        return True
    return False


def ordered_model_keys(seen: set[str]) -> list[str]:
    ranked = sorted(
        (k for k in seen if k in MODEL_NAME_OR_PATH),
        key=model_sort_key,
    )
    rest = sorted(k for k in seen if k not in MODEL_NAME_OR_PATH)
    return ranked + rest


def kind_label(kind: str) -> str:
    if kind == BASE_MODEL_KIND:
        return BASE_MODEL_LABEL
    return APPROACH_TEX_LABELS.get(kind, kind)


def metric_axis_label(metric: str) -> str:
    return METRIC_SHORT_LABELS.get(metric, metric.split("/")[-1].replace(",", " · "))


def _coerce_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _metric_final_value(run: Any, key: str) -> float | None:
    """Prefer summary_metrics / summary; fall back to last non-null in history."""
    sm = getattr(run, "summary_metrics", None)
    if isinstance(sm, dict) and key in sm:
        v = _coerce_float(sm[key])
        if v is not None:
            return v
    su = getattr(run, "summary", None)
    if su is not None:
        raw_v: Any
        try:
            raw_v = su[key] if key in su else None
        except (TypeError, KeyError):
            raw_v = None
        if raw_v is None and hasattr(su, "get"):
            raw_v = su.get(key)
        v = _coerce_float(raw_v)
        if v is not None:
            return v

    keys = ["_step", "global_step", key]
    last: float | None = None
    for row in run.scan_history(keys=keys):
        v = _coerce_float(row.get(key))
        if v is not None:
            last = v
    return last


def dataset_for_mixer(cfg: dict[str, Any]) -> DatasetSpec | None:
    mixer = cfg.get("dataset_mixer_list")
    if mixer is None:
        return None
    for ds in DATASETS:
        if mixer == ds.mixer_list:
            return ds
    return None


def model_key_and_display(
    cfg: dict[str, Any], *, restrict_models: bool
) -> tuple[str | None, str]:
    """Return (model_key, display_name); model_key None if filtered out."""
    path = cfg.get("model_name_or_path")
    if not path:
        return None, "unknown model"
    for k, v in MODEL_NAME_OR_PATH.items():
        if v == path:
            return k, MODEL_DISPLAY[k]
    if restrict_models:
        return None, str(path)
    slug = path.rsplit("/", 1)[-1].replace(".", "p").replace("-", "_")
    return slug, str(path)


def base_model_eval_filter_mongo(job_type: str, model_paths: list[str] | None) -> dict[str, Any]:
    """Broad W&B filter; always re-check with ``is_base_model_eval_cfg`` client-side."""
    kind_list = sorted(
        _BASE_APPROACH_KIND_SLUGS
        | {"base model", "base model eval"}
        | {s.replace("_", " ") for s in _BASE_APPROACH_KIND_SLUGS}
    )
    ors: list[dict[str, Any]] = [
        {"config.eval_is_base_model": {"$in": [True, 1, "true", "True", "1", "yes", "Yes"]}},
        {"config.eval_approach_kind": {"$in": kind_list}},
    ]
    clauses: list[dict[str, Any]] = [
        {"jobType": job_type},
        excluded_run_tags_mongo(),
        {"$or": ors},
    ]
    if model_paths is not None:
        clauses.append({"config.model_name_or_path": {"$in": model_paths}})
    return {"$and": clauses}


def eval_runs_filter_mongo(
    dataset: DatasetSpec,
    *,
    job_type: str,
    model_paths: list[str] | None,
) -> dict[str, Any]:
    clauses: list[dict[str, Any]] = [
        {"jobType": job_type},
        dataset.mongo_clause,
        shaping_arms_mongo(dataset.prefix),
        excluded_run_tags_mongo(),
    ]
    if model_paths is not None:
        clauses.append({"config.model_name_or_path": {"$in": model_paths}})
    return {"$and": clauses}


def collect_rows_for_dataset(
    api: Any,
    entity: str,
    project: str,
    dataset: DatasetSpec,
    metrics: set[str],
    *,
    job_type: str,
    restrict_models: bool,
    kinds_allow: frozenset[str] | None,
) -> list[dict[str, Any]]:
    paths = None if not restrict_models else list(MODEL_NAME_OR_PATH.values())
    filt = eval_runs_filter_mongo(dataset, job_type=job_type, model_paths=paths)
    rows: list[dict[str, Any]] = []
    runs = api.runs(f"{entity}/{project}", filters=filt, per_page=400)
    for run in runs:
        if getattr(run, "job_type", None) != job_type:
            continue
        cfg = _flat_config(run.config or {})
        if is_base_model_eval_cfg(cfg):
            continue
        kind = classify_run_kind(cfg, dataset.prefix)
        if kind is None:
            continue
        if kinds_allow is not None and kind not in kinds_allow:
            continue

        mixer_ds = dataset_for_mixer(cfg)
        if mixer_ds is None or mixer_ds.key != dataset.key:
            continue

        model_key, model_display = model_key_and_display(cfg, restrict_models=restrict_models)
        if model_key is None:
            continue

        finals: dict[str, float] = {}
        for m in metrics:
            v = _metric_final_value(run, m)
            if v is not None:
                finals[m] = v
        if not finals:
            continue

        rows.append(
            {
                "run_id": run.id,
                "run_name": run.name,
                "dataset_key": dataset.key,
                "model_key": model_key,
                "model_display": model_display,
                "kind": kind,
                "seed": cfg.get("seed"),
                **{m: finals.get(m) for m in metrics},
            }
        )
    return rows


def collect_base_model_rows(
    api: Any,
    entity: str,
    project: str,
    metrics: set[str],
    *,
    job_type: str,
    restrict_models: bool,
) -> list[dict[str, Any]]:
    paths = None if not restrict_models else list(MODEL_NAME_OR_PATH.values())
    filt = base_model_eval_filter_mongo(job_type, paths)
    rows: list[dict[str, Any]] = []
    runs = api.runs(f"{entity}/{project}", filters=filt, per_page=400)
    for run in runs:
        if getattr(run, "job_type", None) != job_type:
            continue
        cfg = _flat_config(run.config or {})
        if not is_base_model_eval_cfg(cfg):
            continue
        model_key, model_display = model_key_and_display(cfg, restrict_models=restrict_models)
        if model_key is None:
            continue

        finals: dict[str, float] = {}
        for m in metrics:
            v = _metric_final_value(run, m)
            if v is not None:
                finals[m] = v
        if not finals:
            continue

        rows.append(
            {
                "run_id": run.id,
                "run_name": run.name,
                "dataset_key": BASE_MODEL_KIND,
                "model_key": model_key,
                "model_display": model_display,
                "kind": BASE_MODEL_KIND,
                "seed": cfg.get("seed"),
                **{m: finals.get(m) for m in metrics},
            }
        )
    return rows


def summarize(
    rows: list[dict[str, Any]],
    model_keys: list[str],
    kinds: list[str],
    metrics: list[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for mk in model_keys:
        out[mk] = {}
        for kind in kinds:
            out[mk][kind] = {}
            for m in metrics:
                vals: list[float] = []
                for r in rows:
                    if r["model_key"] != mk or r["kind"] != kind:
                        continue
                    v = r.get(m)
                    if v is None:
                        continue
                    fv = _coerce_float(v)
                    if fv is not None:
                        vals.append(fv)
                if not vals:
                    out[mk][kind][m] = {"mean": None, "sem": None, "n": 0}
                    continue
                arr = np.asarray(vals, dtype=float)
                mean = float(arr.mean())
                sem = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) >= 2 else 0.0
                out[mk][kind][m] = {"mean": mean, "sem": sem, "n": int(len(arr))}
    return out


def plot_dataset(
    summary: dict[str, Any],
    model_keys: list[str],
    model_titles: dict[str, str],
    kinds: list[str],
    metrics: list[str],
    dataset: DatasetSpec,
    *,
    job_type: str,
    show_errorbar: bool,
    out_pdf: Path,
    y_label: str,
) -> None:
    if not model_keys:
        raise ValueError("no models to plot")
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")

    plt.rcParams.update(
        {
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
        }
    )

    n_models = len(model_keys)
    n_kinds = len(kinds)
    width = max(0.12, 0.72 / max(1, n_kinds))
    x = np.arange(len(metrics))
    fig_h = max(4.2, len(metrics) * 0.24)
    fig, axes = plt.subplots(1, n_models, figsize=(5.0 * n_models, fig_h), sharey=False)
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, mk in zip(axes_flat, model_keys):
        present_values: list[float] = []

        for i, kind in enumerate(kinds):
            means = [summary[mk][kind][mm]["mean"] for mm in metrics]
            sems = [summary[mk][kind][mm]["sem"] for mm in metrics]
            heights = [0.0 if v is None else float(v) for v in means]

            offset = (i - (n_kinds - 1) / 2.0) * width
            if show_errorbar:
                yerr_disp = [
                    0.0 if (mval is None or sd is None) else float(sd)
                    for mval, sd in zip(means, sems, strict=True)
                ]
            else:
                yerr_disp = None

            for hv, mval in zip(heights, means, strict=True):
                if mval is not None:
                    present_values.append(float(hv))
            means_arr = np.asarray(heights, dtype=float)

            if show_errorbar and yerr_disp is not None:
                ax.bar(
                    x + offset,
                    means_arr,
                    width,
                    label=kind_label(kind),
                    color=BAR_COLORS[i % len(BAR_COLORS)],
                    edgecolor="0.25",
                    linewidth=0.35,
                    yerr=yerr_disp,
                    capsize=2.2,
                    ecolor="0.25",
                    error_kw={"elinewidth": 0.7},
                )
            else:
                ax.bar(
                    x + offset,
                    means_arr,
                    width,
                    label=kind_label(kind),
                    color=BAR_COLORS[i % len(BAR_COLORS)],
                    edgecolor="0.25",
                    linewidth=0.35,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([metric_axis_label(mm) for mm in metrics], rotation=28, ha="right")
        ax.set_title(model_titles.get(mk, MODEL_DISPLAY.get(mk, mk)))
        ax.grid(True, axis="y", alpha=0.35, linestyle=":")

        if not present_values:
            ax.set_ylim(0.0, 1.02)
            continue
        vmin, vmax = min(present_values), max(present_values)
        if vmin >= -1e-6 and vmax <= 1.001:
            ymax = max(vmax * 1.08 + 0.02, 0.1)
            ax.set_ylim(0.0, min(1.02, ymax))
        elif vmax > vmin:
            pad = (vmax - vmin) * 0.08
            ax.set_ylim(vmin - pad, vmax + pad)

    axes_flat[0].set_ylabel(y_label)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    ncol = max(2, min(len(kinds), 6))
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=ncol,
        frameon=True,
        fancybox=False,
        edgecolor="0.82",
        fontsize=8,
        bbox_to_anchor=(0.5, 1.025),
    )
    fig.suptitle(f"{dataset.display} · job_type={job_type}", fontsize=10.5, y=1.06)
    fig.tight_layout(rect=[0.02, 0, 1, 0.86])
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Wrote {out_pdf}")


def cache_path_default(dataset_key: str, job_type: str, restrict_models: bool) -> Path:
    slug_ds = dataset_key.replace("-", "_").lower()
    slug_jt = job_type.replace("/", "_").lower()
    sfx = "qwen_only" if restrict_models else "all_models"
    return (
        Path(__file__).resolve().parent
        / "cache"
        / f".wandb_plot_eval_benchmarks_{slug_ds}_{slug_jt}_{sfx}.json"
    )


def write_cache(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_cache(path: Path, *, expect: dict[str, Any]) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if raw.get("version") != expect["version"]:
        return None
    for fk, fv in expect.items():
        if fk == "version":
            continue
        if raw.get(fk) != fv:
            return None
    rows = raw.get("rows")
    if not isinstance(rows, list):
        return None
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entity", default="mohdelgaar")
    p.add_argument("--project", default="open_instruct_internal")
    p.add_argument(
        "--dataset",
        action="append",
        default=None,
        help=(
            "Training dataset alias (repeat for several). Defaults to all datasets "
            f"in _runsets: {', '.join(d.key for d in DATASETS)}."
        ),
    )
    p.add_argument(
        "--job-type",
        default="eval",
        help='W&B run job_type / Mongo jobType filter (default: "eval").',
    )
    p.add_argument(
        "--kinds",
        nargs="+",
        default=list(APPROACH_ORDER),
        help="Approach arms to plot (default: all five in APPROACH_ORDER).",
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="Full W&B metric keys (defaults to lm-eval harness list in module docstring).",
    )
    p.add_argument(
        "--all-models",
        action="store_true",
        help="Do not restrict to MODEL_NAME_OR_PATH Qwen checkpoints.",
    )
    p.add_argument(
        "--no-base-models",
        action="store_true",
        help="Omit eval runs identified by eval_is_base_model / eval_approach_kind (base checkpoint).",
    )
    p.add_argument("--out-dir", type=Path, default=None, help="Default: scripts/figures")
    p.add_argument(
        "--cache-path",
        type=Path,
        default=None,
        help=(
            "JSON cache path. When omitted, a per-dataset file under wandb_scripts/cache/ "
            "is used. If you pass --dataset more than once, the dataset slug is appended "
            "to this path's stem."
        ),
    )
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--no-stderr", action="store_true", help="Hide ±SEM error caps.")
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
    if include_base_models:
        allowed_kinds.add(BASE_MODEL_KIND)
        if BASE_MODEL_KIND not in args.kinds:
            args.kinds.insert(0, BASE_MODEL_KIND)
    else:
        args.kinds = [k for k in args.kinds if k != BASE_MODEL_KIND]

    unknown = [k for k in args.kinds if k not in allowed_kinds]
    if unknown:
        raise SystemExit(f"Unknown --kinds {unknown}; choose from {sorted(allowed_kinds)}")

    kinds_allow = frozenset(args.kinds)
    metrics_requested = list(args.metrics)
    metrics_set = set(metrics_requested)

    out_dir = args.out_dir or Path(__file__).resolve().parents[1] / "figures"
    restrict_models = not args.all_models
    api = wandb.Api(timeout=180)
    base_rows_all: list[dict[str, Any]] = []
    if include_base_models:
        base_rows_all = collect_base_model_rows(
            api,
            args.entity,
            args.project,
            metrics_set,
            job_type=args.job_type,
            restrict_models=restrict_models,
        )

    for ds in datasets:
        cache_expect = {
            "version": CACHE_VERSION,
            "entity": args.entity,
            "project": args.project,
            "dataset": ds.key,
            "metrics": sorted(metrics_requested),
            "job_type": args.job_type,
            "kinds": sorted(kinds_allow),
            "restrict_models": restrict_models,
            "include_base_models": include_base_models,
        }
        if args.cache_path is None:
            cp = cache_path_default(ds.key, args.job_type, restrict_models)
        elif len(datasets) > 1:
            slug_ds = ds.key.replace("-", "_").lower()
            cp = args.cache_path.parent / f"{args.cache_path.stem}_{slug_ds}{args.cache_path.suffix}"
        else:
            cp = args.cache_path
        rows: list[dict[str, Any]] | None = None
        if not args.no_cache and not args.refresh:
            rows = read_cache(cp, expect=cache_expect)
            if rows is not None:
                print(f"Loaded eval rows from cache: {cp}")

        if rows is None:
            rows = collect_rows_for_dataset(
                api,
                args.entity,
                args.project,
                ds,
                metrics_set,
                job_type=args.job_type,
                restrict_models=restrict_models,
                kinds_allow=kinds_allow,
            )
            if include_base_models:
                rows.extend({**r, "dataset_key": ds.key} for r in base_rows_all)
            if not args.no_cache:
                write_cache(
                    cp,
                    {
                        **cache_expect,
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                        "rows": rows,
                    },
                )
                print(f"Wrote cache: {cp}")

        metrics_present = [m for m in metrics_requested if any(r.get(m) is not None for r in rows or [])]

        if not rows:
            print(f"Skipping {ds.key}: no eval runs matched filters.")
            continue
        if not metrics_present:
            print(f"Skipping {ds.key}: none of the requested metrics logged on any run.")
            continue

        model_keys_ordered = ordered_model_keys({str(r["model_key"]) for r in rows})
        model_titles = {str(r["model_key"]): str(r["model_display"]) for r in rows}

        summary = summarize(rows, model_keys_ordered, list(args.kinds), metrics_present)

        print(f"Summary · {ds.key} · metrics={len(metrics_present)} · runs={len(rows)}")
        for mk in model_keys_ordered:
            for kind in args.kinds:
                bits = []
                for m in metrics_present:
                    cell = summary[mk][kind][m]
                    if cell["mean"] is None:
                        bits.append(f"{metric_axis_label(m)}=NA")
                    else:
                        bits.append(
                            f"{metric_axis_label(m)}={cell['mean']:.3f}±{cell['sem']:.3f}(n={cell['n']})"
                        )
                print(
                    f"  {model_titles.get(mk, mk)} · {kind_label(kind)}: "
                    + " ".join(bits)
                )

        basename = f"eval_job_benchmarks_{ds.key.replace('-', '_').lower()}_{args.job_type}"
        out_pdf = out_dir / f"{basename}.pdf"

        plot_dataset(
            summary,
            model_keys_ordered,
            model_titles,
            list(args.kinds),
            metrics_present,
            ds,
            job_type=args.job_type,
            show_errorbar=not args.no_stderr,
            out_pdf=out_pdf,
            y_label="Benchmark score",
        )


if __name__ == "__main__":
    main()
