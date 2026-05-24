#!/usr/bin/env python3
"""Compare curriculum competence α ∈ {0.1, 1, 10} on two model sizes (0.6B, 1.7B).

Fetches W&B runs with the same conventions as ``_runsets.py`` (learning rate,
dataset mixer, excluded tags) but restricts to **reward shaping + curriculum**
with ``{prefix}_competence_alpha`` in ``{0.1, 1, 10}``.

Produces **one figure** with two subplots (Qwen3-0.6B and Qwen3-1.7B) for a
chosen ``eval/objective/*_reward`` metric (default: IFBench for IF-RLVR).

Usage (from repo root):
  uv run python scripts/wandb_scripts/plot_curriculum_alpha_two_models.py
  uv run python scripts/wandb_scripts/plot_curriculum_alpha_two_models.py --dataset GSM --metric eval/objective/gsm8k_reward
  uv run python scripts/wandb_scripts/plot_curriculum_alpha_two_models.py --refresh

Requires ``WANDB_API_KEY`` unless the cache is warm.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _runsets import (  # noqa: E402
    DATASET_ALIASES,
    MODEL_DISPLAY,
    MODEL_NAME_OR_PATH,
    DatasetSpec,
    excluded_run_tags_mongo,
    model_lr_filter_mongo,
    resolve_dataset,
)
from plot_eval_objective_rewards_curriculum import (  # noqa: E402
    align_steps,
    discover_reward_metrics,
    fill_missing_steps_across_seeds,
    flat_config,
    group_key_from_json,
    group_legend_label,
    merge_resume_per_seed,
    runs_history_long,
    smooth_plot_data,
    sort_key_gk,
    truncate_steps,
    _lineplot_markers_kw,
    _plot_df_from_json_obj,
    _plot_df_to_json_obj,
    _widen_xlim_if_single_step,
)

# Competence schedule exponents compared in this figure.
DEFAULT_ALPHAS: tuple[float, ...] = (0.1, 1.0, 10.0)

CACHE_VERSION = 1


def _alpha_match(cfg_alpha: float | None, targets: tuple[float, ...]) -> bool:
    if cfg_alpha is None:
        return False
    a = float(cfg_alpha)
    for t in targets:
        if abs(a - float(t)) < 1e-6:
            return True
    return False


def is_curriculum_alpha_run(cfg: dict[str, Any], prefix: str, alphas: tuple[float, ...]) -> bool:
    cfg = flat_config(cfg)
    rs = bool(cfg.get(f"{prefix}_reward_shaping", False))
    curr = bool(cfg.get(f"{prefix}_reward_shaping_curriculum", False))
    if not (rs and curr):
        return False
    raw = cfg.get(f"{prefix}_competence_alpha")
    if raw is None:
        return False
    try:
        alpha = float(raw)
    except (TypeError, ValueError):
        return False
    return _alpha_match(alpha, alphas)


def curriculum_alpha_filter_mongo(
    dataset: DatasetSpec,
    model_paths: list[str],
    alphas: tuple[float, ...],
) -> dict[str, Any]:
    """Mongo filter: lr + model(s) + dataset + shaping+curriculum + α values + tag exclusions."""
    prefix = dataset.prefix
    rs = f"config.{prefix}_reward_shaping"
    curr = f"config.{prefix}_reward_shaping_curriculum"
    alpha_key = f"config.{prefix}_competence_alpha"
    alpha_in: list[float | int] = []
    for a in alphas:
        alpha_in.append(float(a))
        if float(a).is_integer():
            alpha_in.append(int(round(float(a))))
    return {
        "$and": [
            model_lr_filter_mongo(model_paths),
            dataset.mongo_clause,
            {"$and": [{rs: True}, {curr: True}, {alpha_key: {"$in": alpha_in}}]},
            excluded_run_tags_mongo(),
        ]
    }


def _cache_path(dataset_key: str, metric: str, alphas: tuple[float, ...]) -> Path:
    slug_ds = dataset_key.replace("-", "_").lower()
    slug_m = metric.replace("/", "_").replace("eval_objective_", "")
    a_slug = "_".join(str(a).replace(".", "p") for a in alphas)
    return (
        Path(__file__).resolve().parent
        / "cache"
        / f".wandb_curriculum_alpha_two_models_{slug_ds}_{slug_m}_a{a_slug}.json"
    )


def _load_cache(
    path: Path,
    entity: str,
    project: str,
    filters: dict[str, Any],
    samples: int,
    eval_step_period: float,
    metric: str,
    alphas: tuple[float, ...],
) -> dict[str, pd.DataFrame] | None:
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
    if raw.get("filters") != filters:
        return None
    if raw.get("samples") != samples:
        return None
    if float(raw.get("eval_step_period", -1)) != float(eval_step_period):
        return None
    if raw.get("metric") != metric:
        return None
    if tuple(raw.get("alphas", ())) != tuple(alphas):
        return None
    frames: dict[str, pd.DataFrame] = {}
    per_model = raw.get("per_model") or {}
    if not isinstance(per_model, dict):
        return None
    for mk, obj in per_model.items():
        df = _plot_df_from_json_obj(obj)
        if df is None or df.empty:
            return None
        frames[str(mk)] = df.reset_index(drop=True)
    if set(frames.keys()) != {"0.6B", "1.7B"}:
        return None
    return frames


def _save_cache(
    path: Path,
    entity: str,
    project: str,
    filters: dict[str, Any],
    samples: int,
    eval_step_period: float,
    metric: str,
    alphas: tuple[float, ...],
    per_model: dict[str, pd.DataFrame],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CACHE_VERSION,
        "entity": entity,
        "project": project,
        "filters": filters,
        "samples": samples,
        "eval_step_period": float(eval_step_period),
        "metric": metric,
        "alphas": list(alphas),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "per_model": {k: _plot_df_to_json_obj(v) for k, v in per_model.items()},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _filter_runs_curriculum_alpha(
    run_refs: list[Any], prefix: str, alphas: tuple[float, ...]
) -> list[Any]:
    kept: list[Any] = []
    for r in run_refs:
        cfg = flat_config(r.config or {})
        if is_curriculum_alpha_run(cfg, prefix, alphas):
            kept.append(r)
    return kept


def _build_one_model_frame(
    run_list: list[Any],
    metric: str,
    samples: int,
    prefix: str,
    eval_step_period: float,
) -> pd.DataFrame:
    long_df = runs_history_long(run_list, [metric], samples, prefix)
    if long_df.empty:
        return long_df
    merged = merge_resume_per_seed(long_df, eval_step_period=eval_step_period)
    return align_steps(merged)


def _gk_order_and_labels(plot_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    gk_order = sorted(
        plot_df["group_key_json"].unique(),
        key=lambda j: sort_key_gk(group_key_from_json(j)),
    )
    cond_order = [group_legend_label(group_key_from_json(j)) for j in gk_order]
    return gk_order, cond_order


def _format_alpha(a: float) -> str:
    if float(a).is_integer():
        return str(int(round(a)))
    return f"{a:g}"


def plot_two_subplots(
    per_model: dict[str, pd.DataFrame],
    metric: str,
    dataset_display: str,
    wandb_path: str,
    smooth_window: int,
    out_dir: Path,
    base_name: str,
    alphas: tuple[float, ...],
    *,
    errorbar: str | tuple[str, float] | None,
    max_steps: float | None,
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

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)

    # Shared hue order + palette from the first non-empty model.
    gk_order: list[str] | None = None
    cond_order: list[str] | None = None
    palette: dict[str, tuple[float, ...]] | None = None

    for ax, model_key in zip(axes, ("0.6B", "1.7B"), strict=True):
        df = per_model.get(model_key)
        if df is None or df.empty:
            ax.set_visible(False)
            continue
        sub = df.loc[df["metric"] == metric].copy()
        sub = truncate_steps(sub, max_steps)
        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(MODEL_DISPLAY[model_key])
            continue
        sub = smooth_plot_data(sub, smooth_window)
        if gk_order is None:
            gk_order, cond_order = _gk_order_and_labels(sub)
            palette = dict(
                zip(cond_order, sns.color_palette("tab10", n_colors=max(1, len(cond_order))))
            )
        assert cond_order is not None and palette is not None

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
        short = metric.replace("eval/objective/", "").replace("_", " ")
        ax.set_title(f"{MODEL_DISPLAY[model_key]}\n{short.title()}", fontweight="semibold")
        ax.set_ylabel("Reward")
        ax.set_xlabel("Training step")
        ax.grid(True, alpha=0.42, linestyle=":")

    if cond_order and palette:
        handles = [Line2D([0], [0], color=palette[c], lw=2.35, label=c) for c in cond_order]
        fig.legend(
            handles,
            cond_order,
            loc="upper center",
            ncol=min(3, len(cond_order)),
            fontsize=8.8,
            frameon=True,
            fancybox=False,
            edgecolor="0.82",
            bbox_to_anchor=(0.5, -0.02),
            title="Curriculum (competence α)",
        )

    fig.supxlabel("Training step (logged)", fontsize=10, y=-0.02)
    fig.suptitle(
        f"{wandb_path}\n{dataset_display} · curriculum α ∈ {{{', '.join(_format_alpha(a) for a in alphas)}}} · lr $10^{{-6}}$",
        fontsize=10.5,
        y=1.02,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    out_pdf = out_dir / f"{base_name}.pdf"
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
        help="Training dataset (IF-RLVR, RLVR-GSM, …).",
    )
    p.add_argument(
        "--metric",
        default=None,
        help="eval/objective/*_reward key (default: eval/objective/ifbench_reward for IF-RLVR, else first discovered reward).",
    )
    p.add_argument("--samples", type=int, default=20000)
    p.add_argument("--smooth-window", type=int, default=10)
    p.add_argument("--eval-step-period", type=float, default=10.0)
    p.add_argument("--max-steps", type=float, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--cache-path", type=Path, default=None)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--refresh", action="store_true")
    p.add_argument(
        "--alphas",
        type=str,
        default=None,
        help="Comma-separated competence alphas (default: 0.1,1,10).",
    )
    return p.parse_args()


def _parse_alphas(s: str | None) -> tuple[float, ...]:
    if not s:
        return DEFAULT_ALPHAS
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: list[float] = []
    for p in parts:
        out.append(float(p))
    return tuple(out)


def main() -> None:
    args = parse_args()
    dataset = resolve_dataset(args.dataset)
    alphas = _parse_alphas(args.alphas)

    out_dir = args.out_dir or Path(__file__).resolve().parents[1] / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    path = f"{args.entity}/{args.project}"
    model_paths = [MODEL_NAME_OR_PATH["0.6B"], MODEL_NAME_OR_PATH["1.7B"]]
    filters = curriculum_alpha_filter_mongo(dataset, model_paths, alphas)

    use_cache = not args.no_cache
    metric_used: str | None = args.metric
    cache_path = args.cache_path or (
        _cache_path(dataset.key, metric_used, alphas) if metric_used is not None else None
    )

    per_model: dict[str, pd.DataFrame] = {}

    cached = None
    if use_cache and not args.refresh and cache_path is not None:
        cached = _load_cache(
            cache_path,
            args.entity,
            args.project,
            filters,
            args.samples,
            args.eval_step_period,
            metric_used,
            alphas,
        )

    if cached is not None:
        per_model = cached
        print(f"Loaded cache: {cache_path}")
    else:
        api = wandb.Api(timeout=300)
        run_refs = list(api.runs(path, filters=filters, per_page=400))
        if not run_refs:
            raise SystemExit(
                f"No runs matched curriculum α filter for {dataset.key}. Check entity/project and WANDB_API_KEY."
            )
        run_refs = _filter_runs_curriculum_alpha(run_refs, dataset.prefix, alphas)
        if not run_refs:
            raise SystemExit("No runs left after config filter (expect shaping+curriculum and α in target set).")

        run_list = [api.run(f"{path}/{r.id}") for r in run_refs]

        if metric_used is None:
            discovered = discover_reward_metrics(run_list, args.samples)
            if not discovered:
                raise SystemExit("No eval/objective/*_reward metrics found.")
            if dataset.key == "IF-RLVR" and "eval/objective/ifbench_reward" in discovered:
                metric_used = "eval/objective/ifbench_reward"
            else:
                metric_used = discovered[0]
            print(f"Using metric: {metric_used}")

        assert metric_used is not None
        cache_path = args.cache_path or _cache_path(dataset.key, metric_used, alphas)

        cached2 = None
        if use_cache and not args.refresh:
            cached2 = _load_cache(
                cache_path,
                args.entity,
                args.project,
                filters,
                args.samples,
                args.eval_step_period,
                metric_used,
                alphas,
            )

        if cached2 is not None:
            per_model = cached2
            print(f"Loaded cache: {cache_path}")
        else:
            for model_key in ("0.6B", "1.7B"):
                mp = MODEL_NAME_OR_PATH[model_key]
                runs_m = [r for r in run_list if flat_config(r.config or {}).get("model_name_or_path") == mp]
                if not runs_m:
                    per_model[model_key] = pd.DataFrame()
                    print(f"Warning: no runs for {model_key}")
                    continue
                per_model[model_key] = _build_one_model_frame(
                    runs_m,
                    metric_used,
                    args.samples,
                    dataset.prefix,
                    args.eval_step_period,
                )

            for mk, df in per_model.items():
                if df.empty:
                    continue
                per_model[mk] = fill_missing_steps_across_seeds(df)

            if use_cache:
                _save_cache(
                    cache_path,
                    args.entity,
                    args.project,
                    filters,
                    args.samples,
                    args.eval_step_period,
                    metric_used,
                    alphas,
                    per_model,
                )
                print(f"Wrote cache: {cache_path}")

    assert metric_used is not None
    if per_model["0.6B"].empty and per_model["1.7B"].empty:
        raise SystemExit("No plot data for either model.")

    base = (
        f"curriculum_alpha_{'_'.join(_format_alpha(a) for a in alphas)}_"
        f"{dataset.key.replace('-', '_').lower()}_"
        f"{metric_used.replace('eval/objective/', '').replace('/', '_')}_two_models"
    )
    plot_two_subplots(
        per_model,
        metric_used,
        dataset.display,
        path,
        args.smooth_window,
        out_dir,
        base,
        alphas,
        errorbar="se",
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
