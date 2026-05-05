#!/usr/bin/env python3
"""
Build a W&B Report comparing reward-shaping regimes (baseline, shaping-only,
curriculum with α∈{1,10}) with seeds aggregated per regime (mean ± stderr),
plus a "Seeds" section with one line per seed (grouped by config.seed) per approach.

Filters and approach definitions are shared with ``plot_multi_condition_bar.py``
and ``plot_eval_objective_rewards_curriculum.py`` via ``_runsets.py`` so all
three scripts always select the same runs.

Run from repo root (needs wandb + wandb-workspaces):

  uv run --with wandb --with wandb-workspaces python \\
    scripts/wandb_scripts/build_wandb_comparison_report.py \\
    --project open_instruct_internal

Auth: ``wandb login`` or ``WANDB_API_KEY``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import wandb
import wandb_workspaces.reports.v2 as wr
from wandb_workspaces.expr import expr_to_filters

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _runsets import (  # noqa: E402  (path hack must run first)
    APPROACH_KINDS,
    DATASETS,
    LR_VALUE,
    MODEL_DISPLAY,
    MODEL_NAME_OR_PATH,
    DatasetSpec,
    dataset_filter_mongo,
    groupby_config_keys,
    runset_filter_expr,
)

REWARD_SUFFIX = re.compile(r"^eval/objective/.+_reward$")

# Line-plot defaults (applied to every chart; matches UI "Time weighted EMA" + coefficient).
GLOBAL_SMOOTHING_TYPE = "exponentialTimeWeighted"
GLOBAL_SMOOTHING_FACTOR = 0.95
GLOBAL_SMOOTHING_SHOW_ORIGINAL = False

_GLOBAL_LINE_SMOOTHING: dict[str, Any] = {
    "smoothing_type": GLOBAL_SMOOTHING_TYPE,
    "smoothing_factor": GLOBAL_SMOOTHING_FACTOR,
    "smoothing_show_original": GLOBAL_SMOOTHING_SHOW_ORIGINAL,
}

SECTION_ORDER: tuple[str, ...] = tuple(d.key for d in DATASETS)

IFBENCH_REWARD_METRIC = "eval/objective/ifbench_reward"
SEED_LEGEND_TEMPLATE = "seed = ${config:seed}"


def _validate_expr(expr: str) -> str:
    """Raise early if the expression is malformed; return it unchanged."""
    expr_to_filters(expr)
    return expr


def discover_reward_metrics(
    entity: str,
    project: str,
    filters: dict[str, Any],
    max_runs: int,
) -> list[str]:
    api = wandb.Api(timeout=180)
    runs = api.runs(f"{entity}/{project}", filters=filters, per_page=max_runs)
    seen: set[str] = set()
    for run in runs[:max_runs]:
        path = "/".join([run.entity, run.project, run.id])
        rr = api.run(path)
        hk = rr.history_keys["keys"]
        from_keys = {k for k in hk if REWARD_SUFFIX.match(k)}
        if from_keys:
            seen |= from_keys

    def custom_sort_key(metric_name: str) -> tuple[int, str]:
        if "ifbench" in metric_name:
            return (0, metric_name)
        if "ifeval" in metric_name:
            return (1, metric_name)
        return (2, metric_name)

    return sorted(seen, key=custom_sort_key)


def legend_template_for_prefix(prefix: str) -> str:
    # W&B line plots: `${config:key}` — see
    # https://docs.wandb.ai/guides/app/features/panels/line-plot/reference (Legend template).
    return (
        f"rs=${{config:{prefix}_reward_shaping}} "
        f"cur=${{config:{prefix}_reward_shaping_curriculum}} "
        f"a=${{config:{prefix}_competence_alpha}}"
    )


def mean_reward_expression(metrics: list[str]) -> str:
    terms = [f"${{{m}}}" for m in metrics]
    return "(" + " + ".join(terms) + f") / {len(metrics)}"


def panel_layout(index: int, ncol: int = 2, w: int = 8, h: int = 6) -> wr.Layout:
    col = index % ncol
    row = index // ncol
    return wr.Layout(x=col * w, y=row * h, w=w, h=h)


def build_panel_grid(
    entity: str,
    project: str,
    runset_name: str,
    filters: str,
    groupby: list[str],
    reward_metrics: list[str],
    legend_template: str,
) -> wr.PanelGrid:
    runset = wr.Runset(
        entity=entity,
        project=project,
        name=runset_name,
        filters=filters,
        groupby=groupby,
    )
    plot_style: dict[str, Any] = {
        **_GLOBAL_LINE_SMOOTHING,
        "legend_template": legend_template,
    }
    panels: list[Any] = []
    idx = 0
    if len(reward_metrics) > 1:
        panels.append(
            wr.LinePlot(
                title="Average eval reward",
                x="_step",
                y=[reward_metrics[0]],
                custom_expressions=[mean_reward_expression(reward_metrics)],
                groupby_aggfunc="mean",
                groupby_rangefunc="stderr",
                max_runs_to_show=1000,
                layout=panel_layout(idx),
                **plot_style,
            )
        )
        idx += 2

    for m in reward_metrics:
        panels.append(
            wr.LinePlot(
                title=m.split("/")[-1],
                x="_step",
                y=[m],
                groupby_aggfunc="mean",
                groupby_rangefunc="stderr",
                max_runs_to_show=1000,
                layout=panel_layout(idx),
                **plot_style,
            )
        )
        idx += 1
    return wr.PanelGrid(runsets=[runset], panels=panels, hide_run_sets=True)


def resolve_ifbench_metric(metrics: list[str]) -> str:
    """Prefer eval/objective/ifbench_reward; else any *ifbench* reward; else first metric."""
    if IFBENCH_REWARD_METRIC in metrics:
        return IFBENCH_REWARD_METRIC
    for m in metrics:
        if "ifbench" in m:
            return m
    return metrics[0]


def build_seeds_approach_grid(
    entity: str,
    project: str,
    runset_name: str,
    filters: str,
    reward_metric: str,
    approach_title: str,
) -> wr.PanelGrid:
    runset = wr.Runset(
        entity=entity,
        project=project,
        name=runset_name,
        filters=filters,
        groupby=["config.seed"],
    )
    plot_style: dict[str, Any] = {
        **_GLOBAL_LINE_SMOOTHING,
        "legend_template": SEED_LEGEND_TEMPLATE,
    }
    panels: list[Any] = [
        wr.LinePlot(
            title=approach_title,
            x="_step",
            y=[reward_metric],
            groupby_aggfunc="mean",
            groupby_rangefunc="none",
            max_runs_to_show=1000,
            layout=panel_layout(1),
            **plot_style,
        ),
    ]
    return wr.PanelGrid(runsets=[runset], panels=panels, hide_run_sets=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--entity",
        default="",
        help="W&B entity (default: wandb.Api().default_entity)",
    )
    p.add_argument("--project", required=True, help="W&B project name")
    p.add_argument(
        "--title",
        default="Reward shaping vs curriculum (alpha 1,10), all models and datasets",
        help="Report title (max 128 chars; ASCII keeps the share URL readable)",
    )
    p.add_argument(
        "--draft",
        action="store_true",
        help="Save as draft (not visible in project reports list until published)",
    )
    p.add_argument(
        "--discover-runs-cap",
        type=int,
        default=40,
        help="Max runs scanned for metric discovery per dataset section",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    api = wandb.Api(timeout=180)
    entity = args.entity or api.default_entity

    blocks: list[Any] = [wr.TableOfContents()]
    metrics_by_dataset: dict[str, list[str]] = {}

    blocks.append(wr.H1("Results"))

    for dataset in DATASETS:
        blocks.append(wr.H2(dataset.display))
        disc_f = dataset_filter_mongo(dataset)
        metrics = discover_reward_metrics(
            entity,
            args.project,
            disc_f,
            args.discover_runs_cap,
        )
        metrics_by_dataset[dataset.key] = metrics
        if not metrics:
            blocks.append(
                wr.P(
                    "No runs with `eval/objective/*_reward` history matched the discovery filter "
                    "(lr, dataset, model sizes, and shaping arms). Skipping this dataset."
                )
            )
            continue

        for model_key, model_path in MODEL_NAME_OR_PATH.items():
            label = MODEL_DISPLAY[model_key]
            blocks.append(wr.H3(label))
            filt = _validate_expr(runset_filter_expr(model_path, dataset))
            grid = build_panel_grid(
                entity=entity,
                project=args.project,
                runset_name=f"{dataset.display} · {label}",
                filters=filt,
                groupby=groupby_config_keys(dataset.prefix),
                reward_metrics=metrics,
                legend_template=legend_template_for_prefix(dataset.prefix),
            )
            blocks.append(grid)

    blocks.append(wr.H1("Seeds"))

    for dataset in DATASETS:
        metrics = metrics_by_dataset.get(dataset.key, [])
        if not metrics:
            blocks.append(wr.H2(dataset.display))
            blocks.append(
                wr.P(
                    "No metrics discovered for this dataset; skipping Seeds panels "
                    "(same discovery filter as Results)."
                )
            )
            continue

        seed_metric = resolve_ifbench_metric(metrics)

        blocks.append(wr.H2(dataset.display))

        for model_key, model_path in MODEL_NAME_OR_PATH.items():
            label = MODEL_DISPLAY[model_key]
            blocks.append(wr.H3(label))
            for approach_title, kind in APPROACH_KINDS:
                filt = _validate_expr(
                    runset_filter_expr(model_path, dataset, approach_kind=kind)
                )
                grid = build_seeds_approach_grid(
                    entity=entity,
                    project=args.project,
                    runset_name=f"{dataset.display} · {label} · {approach_title}",
                    filters=filt,
                    reward_metric=seed_metric,
                    approach_title=approach_title,
                )
                blocks.append(grid)

    report = wr.Report(
        entity=entity,
        project=args.project,
        title=args.title[:128],
        blocks=blocks,
        width="fixed",
    )
    report.save(draft=args.draft)
    print(report.url)


# Quiet the unused-import warning; kept to preserve the original public surface.
_ = (DatasetSpec, LR_VALUE, SECTION_ORDER)


if __name__ == "__main__":
    main()
