#!/usr/bin/env python3
"""
Plot training-step vs eval/objective/*_reward for Qwen3 @ 1e-6 lr (default 1.7B).

Run filters, dataset definitions, and shaping-arm taxonomy are imported from
``_runsets.py`` so this script selects the same runs as
``build_wandb_comparison_report.py`` and ``plot_multi_condition_bar.py``.

Runs are filtered by model + lr + dataset + approved shaping arms; runs tagged
as known-broken are excluded. They are grouped by
(``{prefix}_reward_shaping``, ``{prefix}_reward_shaping_curriculum``,
``{prefix}_competence_alpha``, ``{prefix}_random_zero_reward``) where ``prefix`` depends on the training dataset
(``ifeval`` for IF-RLVR, ``gsm`` for RLVR-GSM, ``math`` for RLVR-MATH). Seed is
not part of the group key. Same-seed runs are merged (concat / sort / dedupe by
step) for crash/resume. Before merging across W&B jobs, each job's eval steps
are snapped to a shared grid (default: nearest multiple of ``eval_step_period``,
with the training-step-1 bucket kept as 1) so async eval jitter and overlapping
sweeps dedupe to one x per eval round. Across seeds, steps are aligned on a
rounded step axis; seaborn draws the mean line and a 95% confidence band from
the seed-level values at each step (when multiple seeds exist).

Pipeline:
  W&B runs → long DataFrame (run × step × metric)
  → normalize step per run_id, dedupe within run → merge segments per (group, metric, seed)
  → round step for alignment, optional ffill/bfill across seeds on the union of steps
  → optional rolling smooth on each seed curve
  → seaborn lineplot (hue=condition, errorbar=("ci", 95)) on a unified DataFrame

The on-disk cache stores the full aligned plot DataFrame (per-seed rows), so offline
plots match live W&B plots (including seaborn CI bands). Use --refresh to refetch.

Requires WANDB_API_KEY to fetch (unless using a warm cache). From repo:
  cd scripts && uv run python plot_eval_objective_rewards_curriculum.py
  cd scripts && uv run python plot_eval_objective_rewards_curriculum.py --model 0.6B
  cd scripts && uv run python plot_eval_objective_rewards_curriculum.py --dataset GSM

Cache (default under scripts/cache/, per model × dataset): use --refresh to
refetch, --no-cache to skip. Cache invalidates automatically when the filter or
eval_step_period change.

Use --max-steps STEP to plot only steps <= STEP (full series remain in cache).
Use --eval-step-period 0 to disable step snapping (legacy merge behavior).
Use --no-fill-missing to skip forward/backward filling when a seed lacks a step.
Use --paper for a single-panel IFBench figure (baseline, baseline random-zero, shaping-only, curriculum α=10);
    requires --dataset IF-RLVR because the paper panel targets IFBench reward.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from wandb.errors.errors import CommError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _runsets import (  # noqa: E402  (path hack must run first)
    DATASET_ALIASES,
    DATASETS,
    MODEL_DISPLAY,
    MODEL_NAME_OR_PATH,
    DatasetSpec,
    classify_run_kind,
    dataset_filter_mongo,
    group_key_from_json,
    group_key_json,
    group_key_tuple,
    kind_to_group_key,
    resolve_dataset,
)

REWARD_SUFFIX = re.compile(r"^eval/objective/.+_reward$")

CACHE_VERSION = 9  # v9: group key includes random_zero_reward for baseline splits

# Paper figure: baseline, baseline random-zero, shaping-only, curriculum α=10 (drop α=1 ablations).
PAPER_GROUP_KEYS: frozenset[str] = frozenset(
    {
        "[false,false,null,false]",
        "[false,false,null,true]",
        "[true,false,null,false]",
        "[true,true,10.0,false]",
    }
)

PAPER_LEGEND_LABELS: dict[str, str] = {
    "[false,false,null,false]": "Baseline",
    "[false,false,null,true]": "Baseline (random-zero)",
    "[true,false,null,false]": "Shaping only",
    "[true,true,10.0,false]": r"Curriculum ($\alpha{=}10$)",
}


# --- Config helpers -------------------------------------------------------


def flat_config(cfg: dict | None) -> dict:
    out: dict = {}
    for k, v in (cfg or {}).items():
        if isinstance(v, dict) and "value" in v:
            out[k] = v["value"]
        else:
            out[k] = v
    return out


def sort_key_gk(gk: tuple[bool, bool, float | None, bool]) -> tuple:
    rs, curr, alpha, rz = gk
    return (rs, curr, float("inf") if alpha is None else alpha, rz)


def _format_alpha(alpha: float) -> str:
    if float(alpha).is_integer():
        return str(int(round(alpha)))
    return f"{alpha:g}"


def group_legend_label(gk: tuple[bool, bool, float | None, bool]) -> str:
    rs, curr, alpha, rz = gk
    if not rs and not curr:
        if rz:
            return "Baseline (random-zero reward)"
        return "Baseline (no shaping, no curriculum)"
    if rs and not curr:
        return "Reward shaping only"
    if not rs and curr:
        return "Curriculum only (no shaping)"
    if alpha is None:
        return "Reward shaping + curriculum"
    return f"Reward shaping + curriculum (competence α = {_format_alpha(alpha)})"


def group_label_debug(gk: tuple[bool, bool, float | None, bool]) -> str:
    rs, curr, alpha, rz = gk
    a = "∅" if alpha is None else str(alpha)
    return f"RS={rs}, curr={curr}, α={a}, rz={rz}"


# --- History → long DataFrame ---------------------------------------------


def _run_history_df(
    run: Any,
    *,
    samples: int,
    keys: list[str] | None,
    max_tries: int = 4,
) -> pd.DataFrame:
    delay = 2.0
    last_err: CommError | None = None
    for _ in range(max_tries):
        try:
            return run.history(samples=samples, keys=keys, pandas=True)
        except CommError as e:
            last_err = e
            time.sleep(delay)
            delay = min(delay * 2.0, 45.0)
    assert last_err is not None
    raise last_err


def resolve_step_col(hist: pd.DataFrame) -> str | None:
    if "_step" in hist.columns:
        return "_step"
    if "global_step" in hist.columns:
        return "global_step"
    return None


def discover_reward_metrics(runs: list[Any], sample_cap: int) -> list[str]:
    seen: set[str] = set()
    cap = min(2000, sample_cap)
    for run in runs:
        hk = getattr(run, "history_keys", None)
        if isinstance(hk, dict):
            keys = hk.get("keys") or []
            from_hk = {k for k in keys if isinstance(k, str) and REWARD_SUFFIX.match(k)}
            if from_hk:
                seen |= from_hk
                continue
        hist = _run_history_df(run, samples=cap, keys=None)
        if hist.empty:
            continue
        seen.update(c for c in hist.columns if REWARD_SUFFIX.match(c))
    return sorted(seen)


def runs_history_long(
    runs: list[Any],
    metrics: list[str],
    samples: int,
    prefix: str,
) -> pd.DataFrame:
    """
    One row per (run, step, metric) where the metric is logged.
    Columns: run_id, run_name, seed, group_key_json, condition, metric, step, value.
    Grouping uses the shaping prefix matching the training dataset.
    """
    cols = [
        "run_id",
        "run_name",
        "seed",
        "group_key_json",
        "condition",
        "metric",
        "step",
        "value",
    ]
    chunks: list[pd.DataFrame] = []
    step_metric_keys = ["_step", "global_step"]
    for run in runs:
        cfg = flat_config(run.config or {})
        gk = group_key_tuple(cfg, prefix)
        gkj = group_key_json(gk)
        cond = group_legend_label(gk)
        seed = cfg.get("seed", None)
        hk = getattr(run, "history_keys", None)
        if isinstance(hk, dict) and hk.get("keys"):
            known = set(hk["keys"])
            use_metrics = [m for m in metrics if m in known]
        else:
            use_metrics = metrics
        for m in use_metrics:
            hist = _run_history_df(run, samples=samples, keys=[*step_metric_keys, m])
            if hist.empty:
                continue
            sc = resolve_step_col(hist)
            if sc is None or m not in hist.columns:
                continue
            sub = (
                hist[[sc, m]]
                .melt(id_vars=[sc], var_name="metric", value_name="value")
                .dropna(subset=["value"])
            )
            sub = sub.rename(columns={sc: "step"})
            sub["step"] = sub["step"].astype(float)
            sub["value"] = sub["value"].astype(float)
            sub["run_id"] = run.id
            sub["run_name"] = run.name
            sub["seed"] = seed
            sub["group_key_json"] = gkj
            sub["condition"] = cond
            chunks.append(sub)
    if not chunks:
        return pd.DataFrame(columns=cols)
    return pd.concat(chunks, ignore_index=True)


# --- Merge resume segments & plot-ready frame -----------------------------


STEP_ROUND_DECIMALS = 6


def normalize_training_steps(steps: np.ndarray, period: float) -> np.ndarray:
    """
    Map raw training steps to a canonical eval grid so async jitter and overlapping
    jobs share one x per eval round.

    Uses ``round(step / period) * period``, except the bucket that rounds to 0 is stored
    as 1.0 (initial eval before the first multiple of ``period``).
    """
    if period <= 0:
        return np.asarray(steps, dtype=float)
    s = np.asarray(steps, dtype=float)
    out = np.rint(s / period) * period
    # rint(1/10)*10 == 0 — keep early eval at training step 1 as x=1
    out = np.where(np.abs(out) < 1e-9, 1.0, out)
    return out.astype(float)


def merge_resume_per_seed(long_df: pd.DataFrame, *, eval_step_period: float) -> pd.DataFrame:
    """
    For each (group, metric, seed), merge all run_ids: within each W&B run, snap steps
    to the eval grid (when ``eval_step_period`` > 0), dedupe, then concat segments,
    sort by step, dedupe step (keep last).
    """
    if long_df.empty:
        return long_df.iloc[0:0].drop(columns=["run_id"], errors="ignore")

    pieces: list[pd.DataFrame] = []
    group_cols = ["group_key_json", "condition", "metric", "seed"]
    for _, g in long_df.groupby(group_cols, sort=False):
        merged_parts: list[pd.DataFrame] = []
        for _, gr in g.groupby("run_id", sort=False):
            sub = gr[["step", "value"]].dropna().sort_values("step")
            if sub.empty:
                continue
            if eval_step_period > 0:
                sub = sub.copy()
                sub["step"] = normalize_training_steps(sub["step"].to_numpy(), eval_step_period)
                sub = sub.drop_duplicates(subset="step", keep="last")
            merged_parts.append(sub)
        if not merged_parts:
            continue
        one = pd.concat(merged_parts, ignore_index=True).sort_values("step")
        one = one.drop_duplicates(subset="step", keep="last")
        meta = g.iloc[0]
        one["group_key_json"] = meta["group_key_json"]
        one["condition"] = meta["condition"]
        one["metric"] = meta["metric"]
        one["seed"] = meta["seed"]
        pieces.append(one)
    if not pieces:
        return long_df.iloc[0:0]
    return pd.concat(pieces, ignore_index=True)


def align_steps(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Round step so seeds share an x-axis; mean values that collide on the same bin."""
    if merged_df.empty:
        return merged_df
    d = merged_df.copy()
    d["step"] = np.round(d["step"].to_numpy(dtype=float), decimals=STEP_ROUND_DECIMALS)
    return (
        d.groupby(["group_key_json", "condition", "metric", "seed", "step"], sort=False, as_index=False)[
            "value"
        ]
        .mean()
        .sort_values(["metric", "group_key_json", "seed", "step"])
        .reset_index(drop=True)
    )


def smooth_plot_data(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling mean on `value` within each trajectory (includes per-seed or cached mean curve)."""
    if window <= 1 or df.empty:
        return df
    w = max(1, int(window))
    keys = ["group_key_json", "metric"]
    if "seed" in df.columns and df["seed"].notna().any():
        keys = keys + ["seed"]
    # Use transform, not groupby.apply: apply() may omit grouping columns from the
    # passed frame (pandas >= 2.2), which would drop `metric` / `group_key_json`.
    out = df.sort_values(keys + ["step"], kind="mergesort").copy()
    out["value"] = out.groupby(keys, sort=False)["value"].transform(
        lambda s: s.rolling(window=w, center=True, min_periods=1).mean()
    )
    return out


def truncate_steps(df: pd.DataFrame, max_step: float | None) -> pd.DataFrame:
    if max_step is None or df.empty:
        return df
    cap = float(max_step)
    return df.loc[df["step"] <= cap].copy()


def fill_missing_steps_across_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (group_key_json, condition, metric), build the union of ``step`` across
    all seeds, then reindex each seed to that union and fill gaps with forward-fill
    then backward-fill so every seed has a value wherever any seed has data.
    """
    if df.empty or "seed" not in df.columns:
        return df
    out_parts: list[pd.DataFrame] = []
    group_cols = ["group_key_json", "condition", "metric"]
    for _, g in df.groupby(group_cols, sort=False):
        seeds = g["seed"].dropna().unique()
        if len(seeds) <= 1:
            out_parts.append(g)
            continue
        union_steps = np.sort(np.unique(g["step"].to_numpy(dtype=float)))
        if len(union_steps) == 0:
            continue
        meta = g.iloc[0]
        for seed in seeds:
            sub = g.loc[g["seed"] == seed, ["step", "value"]].copy()
            if sub.empty:
                continue
            sub = sub.drop_duplicates(subset="step", keep="last").sort_values("step")
            ser = sub.set_index("step")["value"].sort_index()
            ser = ser.reindex(union_steps)
            ser = ser.ffill().bfill()
            piece = pd.DataFrame(
                {
                    "step": union_steps,
                    "value": ser.to_numpy(),
                    "seed": seed,
                    "group_key_json": meta["group_key_json"],
                    "condition": meta["condition"],
                    "metric": meta["metric"],
                }
            )
            out_parts.append(piece)
    if not out_parts:
        return df
    return (
        pd.concat(out_parts, ignore_index=True)
        .sort_values(["metric", "group_key_json", "seed", "step"], kind="mergesort")
        .reset_index(drop=True)
    )


# --- Cache ----------------------------------------------------------------


def default_cache_path(model_key: str, dataset_key: str) -> Path:
    model_slug = model_key.replace(".", "p").lower()
    dataset_slug = dataset_key.replace("-", "_").lower()
    return (
        Path(__file__).resolve().parent
        / "cache"
        / f".wandb_eval_objective_{dataset_slug}_{model_slug}.json"
    )


def output_basename(model_key: str, dataset_key: str) -> str:
    model_slug = model_key.replace(".", "p").lower()
    dataset_slug = dataset_key.replace("-", "_").lower()
    return f"eval_objective_rewards_vs_step_by_shaping_curriculum_{dataset_slug}_{model_slug}"


def _plot_df_to_json_obj(plot_df: pd.DataFrame) -> dict[str, Any]:
    """Split-orient table for JSON (compact, stable column order)."""
    cols = [
        c
        for c in (
            "group_key_json",
            "condition",
            "metric",
            "seed",
            "step",
            "value",
        )
        if c in plot_df.columns
    ]
    d = plot_df[cols].copy()
    blob = d.to_json(orient="split", double_precision=15)
    return json.loads(blob)


def _plot_df_from_json_obj(obj: Any) -> pd.DataFrame | None:
    if not isinstance(obj, dict):
        return None
    if "columns" not in obj or "data" not in obj:
        return None
    try:
        return pd.read_json(StringIO(json.dumps(obj, separators=(",", ":"))), orient="split")
    except (ValueError, TypeError):
        return None


def load_plot_cache(
    path: Path,
    entity: str,
    project: str,
    filters: dict,
    samples: int,
    eval_step_period: float,
) -> tuple[pd.DataFrame, dict[str, list[str]]] | None:
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
    if raw.get("samples") != samples:
        return None
    if raw.get("filters") != filters:
        return None
    cached_period = raw.get("eval_step_period")
    try:
        cached_period_f = float(cached_period) if cached_period is not None else None
    except (TypeError, ValueError):
        cached_period_f = None
    if cached_period_f != float(eval_step_period):
        return None

    plot_df = _plot_df_from_json_obj(raw.get("plot_df"))
    if plot_df is None or plot_df.empty:
        return None

    names_raw = raw.get("run_names_by_group")
    names_by_gk: dict[str, list[str]] = {}
    if isinstance(names_raw, dict):
        for gk_str, names in names_raw.items():
            if isinstance(names, list):
                names_by_gk[str(gk_str)] = [str(x) for x in names]

    return plot_df.reset_index(drop=True), names_by_gk


def save_plot_cache(
    path: Path,
    entity: str,
    project: str,
    filters: dict,
    samples: int,
    eval_step_period: float,
    plot_df: pd.DataFrame,
    run_names_by_group: dict[tuple[bool, bool, float | None, bool], list[str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    run_names_by_gkj = {group_key_json(gk): names for gk, names in run_names_by_group.items()}
    payload = {
        "version": CACHE_VERSION,
        "entity": entity,
        "project": project,
        "filters": filters,
        "samples": samples,
        "eval_step_period": float(eval_step_period),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "plot_df": _plot_df_to_json_obj(plot_df),
        "run_names_by_group": run_names_by_gkj,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# --- Plot -----------------------------------------------------------------


def _collapsed_to_single_step(sub: pd.DataFrame, *, step_col: str = "step") -> bool:
    return sub[step_col].nunique() < 2


def _lineplot_markers_kw(sub: pd.DataFrame, *, step_col: str = "step") -> dict[str, Any]:
    if not _collapsed_to_single_step(sub, step_col=step_col):
        return {}
    return {
        "marker": "o",
        "markersize": 7.5,
        "markeredgewidth": 0.65,
        "markeredgecolor": "0.2",
    }


def _widen_xlim_if_single_step(
    ax: plt.Axes,
    sub: pd.DataFrame,
    *,
    step_col: str = "step",
    left_zero: bool = False,
) -> None:
    if not _collapsed_to_single_step(sub, step_col=step_col):
        return
    xmid = float(sub[step_col].median())
    half = max(1.0, abs(xmid) * 0.05) if xmid else 1.0
    if left_zero:
        ax.set_xlim(0.0, max(xmid + half, 2.0))
    else:
        ax.set_xlim(max(0.0, xmid - half), xmid + half)


def plot_curves(
    plot_df: pd.DataFrame,
    metrics_order: list[str],
    model_display: str,
    dataset_display: str,
    wandb_path: str,
    smooth_window: int,
    out_dir: Path,
    base_name: str,
    *,
    errorbar: str | tuple[str, float] | None,
) -> None:
    df = smooth_plot_data(plot_df.copy(), smooth_window)

    metrics = [m for m in metrics_order if m in set(df["metric"])]
    if not metrics:
        raise SystemExit("No metrics left to plot after filtering.")

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")

    plt.rcParams.update(
        {
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
        }
    )

    gk_order = sorted(
        df["group_key_json"].unique(),
        key=lambda j: sort_key_gk(group_key_from_json(j)),
    )
    cond_order = [group_legend_label(group_key_from_json(j)) for j in gk_order]
    palette = dict(zip(cond_order, sns.color_palette("tab10", n_colors=max(1, len(cond_order)))))

    n_metrics = len(metrics)
    ncols = min(3, n_metrics)
    nrows = (n_metrics + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.1 * ncols, 3.15 * nrows), sharex=False)
    axes_flat = [axes] if n_metrics == 1 else np.atleast_1d(axes).ravel()

    for j, m in enumerate(metrics):
        ax = axes_flat[j]
        sub = df.loc[df["metric"] == m].copy()
        if sub.empty:
            ax.set_visible(False)
            continue

        sns.lineplot(
            data=sub,
            x="step",
            y="value",
            hue="condition",
            hue_order=cond_order,
            palette=palette,
            ax=ax,
            legend=False,
            linewidth=2.35,
            errorbar=errorbar,
            **_lineplot_markers_kw(sub),
        )
        _widen_xlim_if_single_step(ax, sub)

        short = m.replace("eval/objective/", "").replace("_", " ")
        ax.set_title(short.title(), fontweight="semibold")
        ax.set_ylabel("Reward")
        ax.set_xlabel("")
        ax.grid(True, alpha=0.42, linestyle=":")

    for k in range(n_metrics, len(axes_flat)):
        axes_flat[k].set_visible(False)

    handles = [Line2D([0], [0], color=palette[c], lw=2.35, label=c) for c in cond_order]
    labels = cond_order
    seen: set[str] = set()
    dedup_h, dedup_l = [], []
    for h, lab in zip(handles, labels):
        if lab in seen:
            continue
        seen.add(lab)
        dedup_h.append(h)
        dedup_l.append(lab)
    if dedup_h:
        fig.legend(
            dedup_h,
            dedup_l,
            loc="upper center",
            ncol=min(3, max(1, len(dedup_l))),
            fontsize=8.5,
            frameon=True,
            fancybox=False,
            edgecolor="0.82",
            bbox_to_anchor=(0.5, 0.01),
            title="Condition" if len(dedup_l) > 1 else None,
        )

    fig.supxlabel("Training step (logged)", fontsize=10, y=0.05)
    fig.suptitle(
        f"{wandb_path}\n{model_display} · {dataset_display} · learning rate $10^{{-6}}$",
        fontsize=10.5,
        y=1.015,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.97])
    out_pdf = out_dir / f"{base_name}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Wrote {out_pdf}")


def filter_paper_groups(plot_df: pd.DataFrame) -> pd.DataFrame:
    d = plot_df.loc[plot_df["group_key_json"].isin(PAPER_GROUP_KEYS)].copy()
    d["condition"] = d["group_key_json"].map(PAPER_LEGEND_LABELS).fillna(d["condition"])
    return d


def compute_speedup_pct_ifbench(
    plot_df: pd.DataFrame,
    *,
    metric: str,
    final_step: float,
) -> float | None:
    """
    Percent fewer training steps for curriculum (α=10) to first reach the baseline
    group's mean IFBench reward at ``final_step`` (mean over seeds at each step).
    """
    sub = plot_df.loc[plot_df["metric"] == metric]
    if sub.empty:
        return None
    baseline_gkj = group_key_json(kind_to_group_key("baseline"))
    curr_gkj = group_key_json(kind_to_group_key("curr_a10"))
    baseline = sub.loc[sub["group_key_json"] == baseline_gkj]
    curr = sub.loc[sub["group_key_json"] == curr_gkj]
    if baseline.empty or curr.empty:
        return None
    b_series = baseline.groupby("step", sort=True)["value"].mean()
    c_series = curr.groupby("step", sort=True)["value"].mean().sort_index()
    if final_step not in b_series.index:
        return None
    target = float(b_series.loc[final_step])
    for step, val in c_series.items():
        if float(val) >= target:
            return float((final_step - float(step)) / final_step * 100.0)
    return None


def plot_paper_ifbench_panel(
    plot_df: pd.DataFrame,
    smooth_window: int,
    out_dir: Path,
    base_name: str,
    *,
    errorbar: str | tuple[str, float] | None,
    final_step: float,
) -> None:
    """Single-panel IFBench reward plot with presentation typography and speed-up callout."""
    metric = "eval/objective/ifbench_reward"
    df = filter_paper_groups(plot_df)
    df = df.loc[df["metric"] == metric].copy()
    if df.empty:
        raise SystemExit("No rows for paper IFBench panel after filtering.")

    df = smooth_plot_data(df, smooth_window)
    speedup = compute_speedup_pct_ifbench(plot_df, metric=metric, final_step=final_step)

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")

    plt.rcParams.update(
        {
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )

    gk_order = [
        "[false,false,null,false]",
        "[false,false,null,true]",
        "[true,false,null,false]",
        "[true,true,10.0,false]",
    ]
    cond_order = [PAPER_LEGEND_LABELS[k] for k in gk_order]
    palette = dict(zip(cond_order, ["#4C72B0", "#9467BD", "#DD8452", "#55A868"]))

    fig, ax = plt.subplots(figsize=(7.2, 4.35))
    sns.lineplot(
        data=df,
        x="step",
        y="value",
        hue="condition",
        hue_order=cond_order,
        palette=palette,
        ax=ax,
        legend=False,
        linewidth=2.8,
        errorbar=errorbar,
        **_lineplot_markers_kw(df),
    )
    ax.set_title("IFBench reward (aggregated verifier score)", fontweight="semibold", pad=10)
    ax.set_ylabel("Reward")
    ax.set_xlabel("Training step")
    ax.grid(True, alpha=0.45, linestyle=":")
    if _collapsed_to_single_step(df):
        _widen_xlim_if_single_step(ax, df, left_zero=True)
    else:
        ax.set_xlim(left=0)

    handles = [Line2D([0], [0], color=palette[c], lw=2.8, label=c) for c in cond_order]
    ax.legend(
        handles,
        cond_order,
        loc="lower right",
        frameon=True,
        fancybox=False,
        edgecolor="0.82",
    )

    if speedup is not None:
        ax.annotate(
            f"{speedup:.0f}% fewer steps\nto reach baseline\nfinal IFBench reward",
            xy=(0.03, 0.97),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.75", alpha=0.95),
        )

    fig.tight_layout()
    out_pdf = out_dir / f"{base_name}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Wrote {out_pdf}")
    if speedup is not None:
        print(f"Speed-up annotation: {speedup:.1f}% fewer steps (curriculum vs baseline final @ step {final_step:g})")


# --- main -----------------------------------------------------------------


def _filter_runs_by_kind(run_refs: list[Any], prefix: str) -> list[Any]:
    """Drop runs whose config doesn't classify into one of the four approved arms.

    The Mongo filter allows curriculum with alpha∈{1,10} but stray runs may
    still sneak through (e.g. alpha logged as a string). Double-check here.
    """
    kept: list[Any] = []
    for r in run_refs:
        cfg = flat_config(r.config or {})
        if classify_run_kind(cfg, prefix) is None:
            continue
        kept.append(r)
    return kept


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="mohdelgaar")
    parser.add_argument("--project", default="open_instruct_internal")
    parser.add_argument(
        "--model",
        default="1.7B",
        choices=sorted(MODEL_NAME_OR_PATH.keys()),
        help="Model size (maps to config.model_name_or_path).",
    )
    parser.add_argument(
        "--dataset",
        default="IF-RLVR",
        choices=sorted(set(DATASET_ALIASES)),
        help="Training dataset (aliases: IF, GSM, MATH; full: IF-RLVR, RLVR-GSM, RLVR-MATH).",
    )
    parser.add_argument("--samples", type=int, default=20000, help="Max history rows per run")
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=10,
        help="Rolling-mean window on each series (≤1 = off).",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="Default: ../figures")
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--max-steps",
        type=float,
        default=None,
        metavar="STEP",
        help="Plot only logged training step <= STEP.",
    )
    parser.add_argument(
        "--eval-step-period",
        type=float,
        default=10.0,
        metavar="P",
        help="Snap eval steps to this grid before merging W&B jobs (0 = disable).",
    )
    parser.add_argument(
        "--no-fill-missing",
        action="store_true",
        help="Do not ffill/bfill so each seed has every step in the group union (default: fill).",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Single-panel IFBench figure: baseline, shaping-only, curriculum α=10 only.",
    )
    parser.add_argument(
        "--paper-out-basename",
        type=str,
        default=None,
        help="Output base name for --paper (default: ifbench_curriculum_paper_<model slug>).",
    )
    parser.add_argument(
        "--paper-final-step",
        type=float,
        default=1000.0,
        metavar="STEP",
        help="Baseline final step for the speed-up annotation (default: 1000).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset: DatasetSpec = resolve_dataset(args.dataset)
    if args.paper and dataset.key != "IF-RLVR":
        raise SystemExit("--paper targets the IFBench metric; use --dataset IF-RLVR.")

    out_dir = args.out_dir or Path(__file__).resolve().parents[1] / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_NAME_OR_PATH[args.model]
    filters = dataset_filter_mongo(dataset, model_paths=[model_path])
    path = f"{args.entity}/{args.project}"
    cache_path = args.cache_path or default_cache_path(args.model, dataset.key)
    use_cache = not args.no_cache

    plot_df: pd.DataFrame
    groups_for_print: dict[tuple[bool, bool, float | None, bool], list[str]] = {}
    metric_cols: list[str]

    cached = None
    if use_cache and not args.refresh:
        cached = load_plot_cache(
            cache_path,
            args.entity,
            args.project,
            filters,
            args.samples,
            args.eval_step_period,
        )

    if cached is not None:
        plot_df, names_by_gk = cached
        print(f"Loaded plot data from cache: {cache_path}")
        for gk_str, names in names_by_gk.items():
            groups_for_print[group_key_from_json(gk_str)] = names
        metric_cols = sorted(plot_df["metric"].unique().tolist())
        if not metric_cols:
            raise SystemExit("Cache contained no metrics; use --refresh.")
    else:
        api = wandb.Api(timeout=300)
        run_refs = list(api.runs(path, filters=filters))
        if not run_refs:
            raise SystemExit(
                f"No runs matched filters for {dataset.key} on {args.model}. "
                "Check entity/project and WANDB_API_KEY."
            )
        run_refs = _filter_runs_by_kind(run_refs, dataset.prefix)
        if not run_refs:
            raise SystemExit(
                f"No runs left after approach-kind filter for {dataset.key} on {args.model}."
            )
        run_list = [api.run(f"{path}/{r.id}") for r in run_refs]

        metric_cols = discover_reward_metrics(run_list, args.samples)
        if not metric_cols:
            raise SystemExit("No eval/objective/*_reward columns found in run histories.")

        long_df = runs_history_long(run_list, metric_cols, args.samples, dataset.prefix)
        if long_df.empty:
            raise SystemExit("No history rows after fetch.")

        merged = merge_resume_per_seed(long_df, eval_step_period=args.eval_step_period)
        aligned = align_steps(merged)
        plot_df = aligned

        by_gk: dict[tuple[bool, bool, float | None, bool], list[Any]] = {}
        for run in run_list:
            cfg = flat_config(run.config or {})
            gk = group_key_tuple(cfg, dataset.prefix)
            by_gk.setdefault(gk, []).append(run)
        for gk, rs in by_gk.items():
            groups_for_print[gk] = [r.name for r in rs]

        if use_cache:
            save_plot_cache(
                cache_path,
                args.entity,
                args.project,
                filters,
                args.samples,
                args.eval_step_period,
                aligned,
                groups_for_print,
            )
            print(f"Wrote plot cache: {cache_path}")

    plot_df = truncate_steps(plot_df, args.max_steps)
    if plot_df.empty:
        raise SystemExit("No rows left after --max-steps.")

    if not args.no_fill_missing:
        plot_df = fill_missing_steps_across_seeds(plot_df)

    if plot_df["group_key_json"].nunique() == 0:
        raise SystemExit("No non-empty groups.")

    errorbar = "se"

    if args.paper:
        paper_base = args.paper_out_basename or f"ifbench_curriculum_paper_{args.model.replace('.', 'p').lower()}"
        plot_paper_ifbench_panel(
            plot_df,
            args.smooth_window,
            out_dir,
            paper_base,
            errorbar=errorbar,
            final_step=float(args.paper_final_step),
        )
        return

    plot_curves(
        plot_df,
        metric_cols,
        MODEL_DISPLAY[args.model],
        dataset.display,
        path,
        args.smooth_window,
        out_dir,
        output_basename(args.model, dataset.key),
        errorbar=errorbar,
    )

    # Uncomment for troubleshooting which runs were bucketed into each arm.
    # print("Groups (runs each):")
    # for gk in sorted(groups_for_print.keys(), key=sort_key_gk):
    #     names = groups_for_print.get(gk, [])
    #     print(f"  {group_label_debug(gk)}: {len(names)} run(s) — {names}")
    # print(f"Metrics: {metric_cols}")
    _ = DATASETS  # retain import for editor tooling


if __name__ == "__main__":
    main()
