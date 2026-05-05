#!/usr/bin/env python3
"""
Write a LaTeX table of final GSM8K and MATH eval performance for every
(training dataset × approach × model) cell.

The table shares runsets with ``build_wandb_comparison_report.py`` and the two
plot scripts: ``_runsets.py`` selects runs by model, learning rate, training
dataset, allowed shaping arms, and drops runs tagged as broken.

Each cell is mean ± SEM of the **last non-null value** logged for
``eval/objective/<bench>_correct_rate`` (``--reward`` switches to
``_reward``), averaged across seeds for a given (model, training dataset,
approach) combo.

Example:

  uv run python scripts/wandb_scripts/build_final_performance_table.py \\
      --out scripts/figures/final_performance_table.tex

The companion ``.csv`` is written alongside the ``.tex`` for pasting into
spreadsheets.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _runsets import (  # noqa: E402  (path hack must run first)
    APPROACH_ORDER,
    APPROACH_TEX_LABELS,
    DATASETS,
    MODEL_DISPLAY,
    MODEL_NAME_OR_PATH,
    DatasetSpec,
    classify_run_kind,
    dataset_filter_mongo,
)

# Eval benchmarks to report on. "For GSM and Math" (from the task) → these two
# columns; override on the CLI if you want e.g. IFEval / IFBench / Code added.
DEFAULT_EVAL_BENCHMARKS: tuple[str, ...] = ("gsm8k", "math")

BENCHMARK_DISPLAY: dict[str, str] = {
    "gsm8k": "GSM8K",
    "math": "MATH",
    "ifeval": "IFEval",
    "ifbench": "IFBench",
    "code": "Code",
    "verifiable": "Verifiable",
}


def _flat_config(cfg: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in (cfg or {}).items():
        if isinstance(v, dict) and "value" in v:
            out[k] = v["value"]
        else:
            out[k] = v
    return out


def _final_metric_values(run: Any, metric_keys: list[str]) -> dict[str, float]:
    """Final non-null value seen in ``scan_history`` for each metric."""
    keep: dict[str, float] = {}
    keys = ["_step", "global_step", *metric_keys]
    for row in run.scan_history(keys=keys):
        for m in metric_keys:
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
    datasets: list[DatasetSpec],
    metric_keys: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        filt = dataset_filter_mongo(dataset)
        runs = list(api.runs(f"{entity}/{project}", filters=filt, per_page=400))
        for run in runs:
            cfg = _flat_config(run.config or {})
            kind = classify_run_kind(cfg, dataset.prefix)
            if kind is None:
                continue
            model_path = cfg.get("model_name_or_path")
            model_key = next(
                (k for k, v in MODEL_NAME_OR_PATH.items() if v == model_path),
                None,
            )
            if model_key is None:
                continue
            final_vals = _final_metric_values(run, metric_keys)
            if not final_vals:
                continue
            rows.append(
                {
                    "dataset_key": dataset.key,
                    "dataset_display": dataset.display,
                    "model_key": model_key,
                    "model_display": MODEL_DISPLAY[model_key],
                    "kind": kind,
                    "seed": cfg.get("seed"),
                    "run_id": run.id,
                    "run_name": run.name,
                    **{m: final_vals.get(m) for m in metric_keys},
                }
            )
    return rows


def _aggregate(
    rows: list[dict[str, Any]],
    models: list[str],
    datasets: list[DatasetSpec],
    kinds: list[str],
    metric_keys: list[str],
) -> dict[tuple[str, str, str], dict[str, dict[str, float | int | None]]]:
    """Mean ± SEM across seeds for each (model, dataset, kind, metric) cell."""
    out: dict[tuple[str, str, str], dict[str, dict[str, float | int | None]]] = {}
    for model_key in models:
        for ds in datasets:
            for kind in kinds:
                cell: dict[str, dict[str, float | int | None]] = {}
                for m in metric_keys:
                    vals: list[float] = []
                    for r in rows:
                        if r["model_key"] != model_key:
                            continue
                        if r["dataset_key"] != ds.key:
                            continue
                        if r["kind"] != kind:
                            continue
                        v = r.get(m)
                        if v is None:
                            continue
                        try:
                            vals.append(float(v))
                        except (TypeError, ValueError):
                            continue
                    if not vals:
                        cell[m] = {"mean": None, "sem": None, "n": 0}
                        continue
                    arr = np.asarray(vals, dtype=float)
                    mean = float(arr.mean())
                    sem = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) >= 2 else 0.0
                    cell[m] = {"mean": mean, "sem": sem, "n": int(len(arr))}
                out[(model_key, ds.key, kind)] = cell
    return out


def _fmt_cell(
    cell: dict[str, float | int | None], *, include_n: bool, bold: bool
) -> str:
    mean = cell.get("mean")
    sem = cell.get("sem")
    n = cell.get("n") or 0
    if mean is None:
        return r"\textemdash"
    mean_tex = rf"\mathbf{{{mean:.3f}}}" if bold else f"{mean:.3f}"
    if n <= 1 or sem is None or sem == 0:
        # pad to match "0.000 \pm 0.000" width so columns align
        body = mean_tex + r"\phantom{\,\pm\,0.000}"
    else:
        sem_tex = rf"\mathbf{{{sem:.3f}}}" if bold else f"{sem:.3f}"
        body = rf"{mean_tex}\,\pm\,{sem_tex}"
    out = f"${body}$"
    if include_n:
        out = rf"{out}\,{{\scriptsize $n{{=}}{n}$}}"
    return out


def _bold_best_per_group(
    agg: dict[tuple[str, str, str], dict[str, dict[str, float | int | None]]],
    models: list[str],
    datasets: list[DatasetSpec],
    kinds: list[str],
    metric_keys: list[str],
) -> dict[tuple[str, str, str], dict[str, bool]]:
    """For each (model, dataset, metric), mark the approach with the max mean."""
    bold: dict[tuple[str, str, str], dict[str, bool]] = {}
    for model_key in models:
        for ds in datasets:
            for kind in kinds:
                bold.setdefault((model_key, ds.key, kind), {m: False for m in metric_keys})
    for model_key in models:
        for ds in datasets:
            for m in metric_keys:
                best_kind = None
                best_mean: float | None = None
                for kind in kinds:
                    mean = agg[(model_key, ds.key, kind)][m]["mean"]
                    if mean is None:
                        continue
                    if best_mean is None or float(mean) > best_mean:
                        best_mean = float(mean)
                        best_kind = kind
                if best_kind is not None:
                    bold[(model_key, ds.key, best_kind)][m] = True
    return bold


def render_latex(
    agg: dict[tuple[str, str, str], dict[str, dict[str, float | int | None]]],
    models: list[str],
    datasets: list[DatasetSpec],
    kinds: list[str],
    metric_keys: list[str],
    *,
    caption: str,
    label: str,
    include_n: bool,
    bold_best: bool,
) -> str:
    bold = _bold_best_per_group(agg, models, datasets, kinds, metric_keys) if bold_best else {}

    bench_cols = "".join("c" for _ in metric_keys)
    n_benches = len(metric_keys)
    col_spec = f"lll{bench_cols}"
    bench_headers = " & ".join(BENCHMARK_DISPLAY.get(b.split("/")[-1].replace("_correct_rate", "").replace("_reward", ""), b) for b in metric_keys)

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(rf"  \caption{{{caption}}}")
    lines.append(rf"  \label{{{label}}}")
    lines.append(rf"  \begin{{tabular}}{{{col_spec}}}")
    lines.append(r"    \toprule")
    lines.append(rf"    Model & Training & Approach & {bench_headers} \\")
    lines.append(r"    \midrule")

    total_kinds = len(kinds)
    for mi, model_key in enumerate(models):
        model_block = rf"\multirow{{{len(datasets) * total_kinds}}}{{*}}{{{MODEL_DISPLAY[model_key]}}}"
        for di, ds in enumerate(datasets):
            ds_block = rf"\multirow{{{total_kinds}}}{{*}}{{{ds.display}}}"
            for ki, kind in enumerate(kinds):
                cells = agg[(model_key, ds.key, kind)]
                vals = []
                for m in metric_keys:
                    is_bold = bold_best and bold[(model_key, ds.key, kind)].get(m, False)
                    vals.append(_fmt_cell(cells[m], include_n=include_n, bold=is_bold))
                approach_label = APPROACH_TEX_LABELS.get(kind, kind)
                first_cell = model_block if (di == 0 and ki == 0) else ""
                second_cell = ds_block if ki == 0 else ""
                lines.append(
                    "    "
                    + " & ".join([first_cell, second_cell, approach_label, *vals])
                    + r" \\"
                )
            if di < len(datasets) - 1:
                lines.append(rf"    \cmidrule(lr){{2-{3 + n_benches}}}")
        if mi < len(models) - 1:
            lines.append(r"    \midrule")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def render_csv(
    agg: dict[tuple[str, str, str], dict[str, dict[str, float | int | None]]],
    models: list[str],
    datasets: list[DatasetSpec],
    kinds: list[str],
    metric_keys: list[str],
) -> str:
    import io

    buf = io.StringIO()
    writer = csv.writer(buf)
    header = ["model", "training_dataset", "approach"]
    for m in metric_keys:
        stem = m.split("/")[-1].replace("_correct_rate", "").replace("_reward", "")
        header.extend([f"{stem}_mean", f"{stem}_sem", f"{stem}_n"])
    writer.writerow(header)
    for model_key in models:
        for ds in datasets:
            for kind in kinds:
                row: list[Any] = [MODEL_DISPLAY[model_key], ds.display, kind]
                cells = agg[(model_key, ds.key, kind)]
                for m in metric_keys:
                    cell = cells[m]
                    row.extend([
                        "" if cell["mean"] is None else f"{cell['mean']:.6f}",
                        "" if cell["sem"] is None else f"{cell['sem']:.6f}",
                        cell["n"],
                    ])
                writer.writerow(row)
    return buf.getvalue()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entity", default="mohdelgaar")
    p.add_argument("--project", default="open_instruct_internal")
    p.add_argument(
        "--benchmarks",
        nargs="+",
        default=list(DEFAULT_EVAL_BENCHMARKS),
        help=f"Eval benchmarks to include as columns (default: {list(DEFAULT_EVAL_BENCHMARKS)}).",
    )
    p.add_argument(
        "--reward",
        action="store_true",
        help="Use *_reward values instead of *_correct_rate.",
    )
    p.add_argument(
        "--kinds",
        nargs="+",
        default=list(APPROACH_ORDER),
        help=(
            f"Approach kinds to include as rows (default: {list(APPROACH_ORDER)}). "
            "Valid kinds: baseline, baseline_rz, shaping, curr_a1, curr_a10."
        ),
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=sorted(MODEL_NAME_OR_PATH.keys(), key=lambda k: float(k.rstrip("B"))),
        help=f"Model keys to include (default: {sorted(MODEL_NAME_OR_PATH)}).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .tex path (default: ../figures/final_performance_table.tex).",
    )
    p.add_argument(
        "--caption",
        type=str,
        default=(
            "Final performance on GSM8K and MATH eval benchmarks (mean $\\pm$ SEM "
            "across seeds). Training datasets: IF-RLVR, RLVR-GSM, RLVR-MATH. "
            "Runs share all selection criteria with the comparison report."
        ),
    )
    p.add_argument(
        "--label",
        type=str,
        default="tab:final_perf_gsm_math",
    )
    p.add_argument(
        "--with-n",
        action="store_true",
        help="Append seed count (n=K) to each cell.",
    )
    p.add_argument(
        "--no-bold",
        action="store_true",
        help="Do not bold the best approach per (model, training, metric) group.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    unknown_kinds = [k for k in args.kinds if k not in APPROACH_ORDER]
    if unknown_kinds:
        raise SystemExit(
            f"Unknown approach kind(s): {unknown_kinds}. "
            f"Valid: {list(APPROACH_ORDER)}."
        )
    unknown_models = [m for m in args.models if m not in MODEL_NAME_OR_PATH]
    if unknown_models:
        raise SystemExit(
            f"Unknown model key(s): {unknown_models}. "
            f"Valid: {sorted(MODEL_NAME_OR_PATH)}."
        )

    suffix = "_reward" if args.reward else "_correct_rate"
    metric_keys = [f"eval/objective/{b}{suffix}" for b in args.benchmarks]

    out_path = args.out or (
        Path(__file__).resolve().parents[1] / "figures" / "final_performance_table.tex"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    api = wandb.Api(timeout=180)
    datasets = list(DATASETS)
    rows = _collect_rows(api, args.entity, args.project, datasets, metric_keys)
    if not rows:
        raise SystemExit("No runs found with the shared filters; check WANDB_API_KEY/project.")

    agg = _aggregate(rows, args.models, datasets, args.kinds, metric_keys)

    tex = render_latex(
        agg,
        args.models,
        datasets,
        args.kinds,
        metric_keys,
        caption=args.caption,
        label=args.label,
        include_n=args.with_n,
        bold_best=not args.no_bold,
    )
    out_path.write_text(tex, encoding="utf-8")
    print(f"Wrote LaTeX: {out_path}")

    csv_path = out_path.with_suffix(".csv")
    csv_path.write_text(
        render_csv(agg, args.models, datasets, args.kinds, metric_keys),
        encoding="utf-8",
    )
    print(f"Wrote CSV:   {csv_path}")

    # Echo a plain-text view on stdout to preview the numbers.
    print()
    for model_key in args.models:
        print(f"=== {MODEL_DISPLAY[model_key]} ===")
        for ds in datasets:
            for kind in args.kinds:
                cells = agg[(model_key, ds.key, kind)]
                bits = []
                for m in metric_keys:
                    cell = cells[m]
                    stem = m.split("/")[-1].replace(suffix, "")
                    label = BENCHMARK_DISPLAY.get(stem, stem)
                    if cell["mean"] is None:
                        bits.append(f"{label}=NA")
                    else:
                        bits.append(f"{label}={cell['mean']:.3f}±{cell['sem']:.3f}(n={cell['n']})")
                print(f"  {ds.display:8s} · {kind:10s}: " + "  ".join(bits))


if __name__ == "__main__":
    main()
