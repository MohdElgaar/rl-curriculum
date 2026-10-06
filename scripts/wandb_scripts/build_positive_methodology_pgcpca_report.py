#!/usr/bin/env python3
"""Build a W&B Report for the positive-methodology PG-CPCA experiment matrix.

Run from the repository root:

  uv run --with wandb --with wandb-workspaces python \
    scripts/wandb_scripts/build_positive_methodology_pgcpca_report.py \
    --project open_instruct_internal

By default the report is saved as a draft. Pass ``--publish`` to publish it.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from typing import Any

import wandb
import wandb_workspaces.reports.v2 as wr
from wandb_workspaces.expr import expr_to_filters


PROJECT_DEFAULT = "open_instruct_internal"
EXP_PREFIX = "pm_pgcpca"
SEEDS = (1, 2, 3)

GLOBAL_SMOOTHING: dict[str, Any] = {
    "smoothing_type": "exponentialTimeWeighted",
    "smoothing_factor": 0.95,
    "smoothing_show_original": False,
}

CONDITION_GROUPBY = [
    "config.ifeval_reward_shaping",
    "config.ifeval_reward_shaping_curriculum",
    "config.ifeval_pg_cpca",
    "config.ifeval_pg_cpca_calibration",
    "config.ifeval_pg_cpca_collapse_gating",
    "config.ifeval_pg_cpca_adaptive_annealing",
    "config.ifeval_pg_cpca_placebo",
    "config.filter_zero_std_samples",
    "config.advantage_normalization_type",
]

CONDITION_LEGEND = (
    "rs=${config:ifeval_reward_shaping} "
    "cur=${config:ifeval_reward_shaping_curriculum} "
    "pg=${config:ifeval_pg_cpca} "
    "cal=${config:ifeval_pg_cpca_calibration} "
    "gate=${config:ifeval_pg_cpca_collapse_gating} "
    "adapt=${config:ifeval_pg_cpca_adaptive_annealing} "
    "placebo=${config:ifeval_pg_cpca_placebo} "
    "filt0=${config:filter_zero_std_samples} "
    "norm=${config:advantage_normalization_type}"
)

SEED_LEGEND = "${config:exp_name}"


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    display: str
    model_name: str


@dataclass(frozen=True)
class ConditionSpec:
    slug: str
    display: str
    note: str


MODELS = (
    ModelSpec("gemma_4_e2b", "Gemma-4-E2B-it", "google/gemma-4-E2B-it"),
    ModelSpec("qwen3_1p7b", "Qwen3-1.7B", "Qwen/Qwen3-1.7B"),
)

CONDITIONS = (
    ConditionSpec("binary_grpo", "Binary GRPO", "Exact IF verifier only."),
    ConditionSpec("hard_only", "High-precision hard-only", "Exact verifier with zero-std group filtering."),
    ConditionSpec("stabilized_grpo", "Stabilized GRPO", "Exact verifier with standard advantage normalization."),
    ConditionSpec("constant_shaping", "Constant partial shaping", "Raw partial credit without curriculum."),
    ConditionSpec("fixed_alpha", "Fixed alpha curriculum", "Existing fixed alpha=10 partial-credit curriculum."),
    ConditionSpec("pg_cpca", "PG-CPCA", "Collapse-gated calibrated partial rescue."),
    ConditionSpec("pg_no_calibration", "PG-CPCA without calibration", "Ablates reliability gating."),
    ConditionSpec("pg_no_collapse_gating", "PG-CPCA without collapse gating", "Uses calibrated partial scores outside all-zero groups."),
    ConditionSpec("pg_fixed_anneal", "PG-CPCA with fixed annealing", "Uses calibrated rescue with fixed alpha schedule."),
    ConditionSpec("placebo_matched", "Distribution-matched placebo", "Shuffles partial scores within constraint family."),
)

EVAL_METRICS = [
    "eval/objective/ifbench_reward",
    "eval/objective/ifeval_reward",
    "eval/scores",
    "eval/sequence_lengths",
    "eval/stop_rate",
]

TRAINING_METRICS = [
    "objective/verifiable_reward",
    "objective/ifbench_reward",
    "objective/verifiable_correct_rate",
    "scores",
    "val/sequence_lengths",
    "val/stop_rate",
    "objective/kl2_avg",
    "loss/kl_avg",
    "time/total",
]

PG_CPCA_METRICS = [
    "objective/ifeval_pg_cpca_cacr0",
    "objective/ifeval_pg_cpca_cacr1",
    "objective/ifeval_pg_cpca_eligible",
    "objective/ifeval_pg_cpca_dispersed",
    "objective/ifeval_pg_cpca_exact_score",
    "objective/ifeval_pg_cpca_rescue_score",
    "objective/ifeval_pg_cpca_mean_rho",
]


class GroupedPanelGrid(wr.PanelGrid):
    """Make panels respect Runset.groupby in the saved report."""

    def _to_model(self):
        model = super()._to_model()
        settings = model.metadata.panel_bank_section_config.local_panel_settings
        settings.use_runs_table_grouping_in_panels = True
        return model


def exp_name(model: ModelSpec, condition: ConditionSpec, seed: int) -> str:
    return f"{EXP_PREFIX}_{model.slug}_{condition.slug}_s{seed}"


def all_exp_names() -> list[str]:
    return [exp_name(m, c, s) for m in MODELS for c in CONDITIONS for s in SEEDS]


def exp_names_for_model(model: ModelSpec) -> list[str]:
    return [exp_name(model, c, s) for c in CONDITIONS for s in SEEDS]


def expr_for_exp_names(names: list[str]) -> str:
    expr = " or ".join(f"Config('exp_name') == {name!r}" for name in names)
    wrapped = f"({expr})"
    expr_to_filters(wrapped)
    return wrapped


def panel_layout(index: int, ncol: int = 2, w: int = 8, h: int = 6) -> wr.Layout:
    col = index % ncol
    row = index // ncol
    return wr.Layout(x=col * w, y=row * h, w=w, h=h)


def line_panels(metrics: list[str], *, legend_template: str, aggregate: bool) -> list[Any]:
    panels = []
    for idx, metric in enumerate(metrics):
        kwargs: dict[str, Any] = {
            **GLOBAL_SMOOTHING,
            "title": metric,
            "x": "_step",
            "y": [metric],
            "legend_template": legend_template,
            "max_runs_to_show": 1000,
            "layout": panel_layout(idx),
        }
        if aggregate:
            kwargs["groupby_aggfunc"] = "mean"
            kwargs["groupby_rangefunc"] = "stderr"
        else:
            kwargs["groupby_aggfunc"] = "mean"
            kwargs["groupby_rangefunc"] = "none"
        panels.append(wr.LinePlot(**kwargs))
    return panels


def build_grid(
    *,
    entity: str,
    project: str,
    name: str,
    filters: str,
    groupby: list[str],
    metrics: list[str],
    legend_template: str,
    aggregate: bool,
) -> wr.PanelGrid:
    runset = wr.Runset(entity=entity, project=project, name=name, filters=filters, groupby=groupby)
    return GroupedPanelGrid(
        runsets=[runset],
        panels=line_panels(metrics, legend_template=legend_template, aggregate=aggregate),
        hide_run_sets=True,
    )


def markdown_block(text: str):
    block = getattr(wr, "MarkdownBlock", None)
    if block is None:
        return wr.P(text)
    try:
        return block(text=text)
    except TypeError:
        return block(text)


def condition_markdown() -> str:
    rows = ["| Condition | Experiment slug | Notes |", "|---|---|---|"]
    rows.extend(f"| {c.display} | `{c.slug}` | {c.note} |" for c in CONDITIONS)
    return "\n".join(rows)


def fetch_inventory(entity: str, project: str, limit: int) -> tuple[list[Any], str]:
    api = wandb.Api(timeout=180)
    runs = api.runs(
        f"{entity}/{project}",
        filters={"config.exp_name": {"$in": all_exp_names()}},
        order="-created_at",
        per_page=min(limit, 200),
    )
    found = runs[:limit]
    state_counts = Counter(getattr(run, "state", "unknown") for run in found)
    by_model = Counter()
    by_seed = Counter()
    by_condition = Counter()
    for run in found:
        cfg = run.config or {}
        name = cfg.get("exp_name", "")
        for model in MODELS:
            if f"_{model.slug}_" in name:
                by_model[model.display] += 1
                break
        for condition in CONDITIONS:
            if f"_{condition.slug}_s" in name:
                by_condition[condition.display] += 1
                break
        seed = cfg.get("seed")
        if seed is not None:
            by_seed[f"seed {seed}"] += 1

    lines = [
        f"Found **{len(found)}** W&B runs matching the `{EXP_PREFIX}_*` experiment matrix.",
        "",
        f"States: {dict(sorted(state_counts.items())) or '{}'}",
        f"By model: {dict(sorted(by_model.items())) or '{}'}",
        f"By seed: {dict(sorted(by_seed.items())) or '{}'}",
        f"By condition: {dict(sorted(by_condition.items())) or '{}'}",
    ]
    return found, "\n\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entity", default="", help="W&B entity (default: wandb.Api().default_entity)")
    p.add_argument("--project", default=PROJECT_DEFAULT, help=f"W&B project (default: {PROJECT_DEFAULT})")
    p.add_argument(
        "--title",
        default="Positive methodology PG-CPCA experiment matrix",
        help="Report title",
    )
    p.add_argument("--publish", action="store_true", help="Publish instead of saving as a draft")
    p.add_argument("--inventory-limit", type=int, default=120, help="Max runs to count in the inventory section")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    api = wandb.Api(timeout=180)
    entity = args.entity or api.default_entity
    _, inventory = fetch_inventory(entity, args.project, args.inventory_limit)

    blocks: list[Any] = [
        wr.TableOfContents(),
        wr.H1(args.title[:128]),
        wr.P(
            "Live W&B report for the PG-CPCA positive-methodology matrix. "
            "The runsets filter exact `config.exp_name` values generated under "
            "`configs/positive_methodology`, so panels will populate as queued jobs start."
        ),
        wr.H2("Run Inventory"),
        markdown_block(inventory),
        wr.H2("Condition Map"),
        markdown_block(condition_markdown()),
    ]

    for model in MODELS:
        model_filter = expr_for_exp_names(exp_names_for_model(model))
        blocks.append(wr.H1(model.display))
        blocks.append(
            wr.P(
                f"Model path: `{model.model_name}`. Condition-level charts aggregate matching seeds "
                "with mean and stderr; seed-level charts keep each `exp_name` separate."
            )
        )

        blocks.append(wr.H2("Condition Means: Evaluation"))
        blocks.append(
            build_grid(
                entity=entity,
                project=args.project,
                name=f"{model.display} condition means eval",
                filters=model_filter,
                groupby=CONDITION_GROUPBY,
                metrics=EVAL_METRICS,
                legend_template=CONDITION_LEGEND,
                aggregate=True,
            )
        )

        blocks.append(wr.H2("Condition Means: Training"))
        blocks.append(
            build_grid(
                entity=entity,
                project=args.project,
                name=f"{model.display} condition means train",
                filters=model_filter,
                groupby=CONDITION_GROUPBY,
                metrics=TRAINING_METRICS,
                legend_template=CONDITION_LEGEND,
                aggregate=True,
            )
        )

        blocks.append(wr.H2("PG-CPCA Mechanism Metrics"))
        blocks.append(
            build_grid(
                entity=entity,
                project=args.project,
                name=f"{model.display} PG-CPCA mechanism",
                filters=model_filter,
                groupby=CONDITION_GROUPBY,
                metrics=PG_CPCA_METRICS,
                legend_template=CONDITION_LEGEND,
                aggregate=True,
            )
        )

        blocks.append(wr.H2("Seed-Level Evaluation"))
        blocks.append(
            build_grid(
                entity=entity,
                project=args.project,
                name=f"{model.display} seed-level eval",
                filters=model_filter,
                groupby=["config.exp_name"],
                metrics=EVAL_METRICS,
                legend_template=SEED_LEGEND,
                aggregate=False,
            )
        )

    report = wr.Report(
        entity=entity,
        project=args.project,
        title=args.title[:128],
        blocks=blocks,
        width="fixed",
    )
    report.save(draft=not args.publish)
    print(report.url)


if __name__ == "__main__":
    main()
