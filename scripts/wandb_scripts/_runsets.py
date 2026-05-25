"""Shared run-set definitions for the W&B report and plot scripts.

The report (``build_wandb_comparison_report.py``) and the plot scripts
(``plot_eval_objective_rewards_curriculum.py`` and
``plot_multi_condition_bar.py``) must pick the same runs or they will show
different numbers for "the same experiment". This module centralises:

- model paths and display names
- learning-rate pin (default ``1e-6``; Qwen3.5-9B and Gemma-4-E2B IF-RLVR use ``5e-7`` — see ``learning_rate_for_model_path``)
- dataset identifiers (mongo, wandb_workspaces expression, and config prefix)
- shaping-arm definitions (baseline, baseline with random-zero reward, shaping only,
  curriculum α∈{1,10})
- excluded tags for known-broken runs

Two filter flavours are exposed because we use two W&B APIs:

- Mongo-style dicts for ``wandb.Api().runs(filters=...)`` (used by the plotting
  scripts and the report's metric-discovery step).
- Expression strings for ``wandb_workspaces.Runset(filters=...)`` (used by the
  report's panel grids).

If you change a filter, update both flavours together.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

LR_VALUE: float = 1e-6

MODEL_NAME_OR_PATH: dict[str, str] = {
    "1.7B": "Qwen/Qwen3-1.7B",
    "0.6B": "Qwen/Qwen3-0.6B",
    "9B": "Qwen/Qwen3.5-9B",
    "E2B": "google/gemma-4-E2B-it",
}

MODEL_DISPLAY: dict[str, str] = {
    "1.7B": "Qwen3-1.7B",
    "0.6B": "Qwen3-0.6B",
    "9B": "Qwen3.5-9B",
    "E2B": "Gemma-4-E2B-it",
}

# Monotonic size order for bar-chart columns (do not use str.rstrip("B") — "E2B" → "E2").
MODEL_SORT_KEY: dict[str, float] = {
    "0.6B": 0.6,
    "1.7B": 1.7,
    "E2B": 2.0,
    "9B": 9.0,
}


def model_sort_key(model_key: str) -> float:
    """Numeric sort key for ``MODEL_NAME_OR_PATH`` keys and unknown slugs."""
    if model_key in MODEL_SORT_KEY:
        return MODEL_SORT_KEY[model_key]
    if model_key.endswith("B"):
        try:
            return float(model_key[:-1])
        except ValueError:
            pass
    return float("inf")


def learning_rate_for_model_path(model_path: str) -> float:
    """Training LR pinned in W&B filters (9B / Gemma-4-E2B IF-RLVR @ 5e-7; smaller Qwen3 @ 1e-6)."""
    if model_path in ("Qwen/Qwen3.5-9B", "google/gemma-4-E2B-it"):
        return 5e-7
    return LR_VALUE


def model_lr_filter_mongo(model_paths: list[str]) -> dict[str, Any]:
    """``$and`` of learning_rate + model_name_or_path, or ``$or`` when paths use different LRs."""
    buckets: dict[float, list[str]] = {}
    for mp in model_paths:
        lr = learning_rate_for_model_path(mp)
        buckets.setdefault(lr, []).append(mp)
    if len(buckets) == 1:
        lr = next(iter(buckets))
        paths = buckets[lr]
        return {"$and": [{"config.learning_rate": lr}, {"config.model_name_or_path": {"$in": paths}}]}
    or_clauses: list[dict[str, Any]] = []
    for lr, paths in buckets.items():
        or_clauses.append({"$and": [{"config.learning_rate": lr}, {"config.model_name_or_path": {"$in": paths}}]})
    return {"$or": or_clauses}


@dataclass(frozen=True)
class DatasetSpec:
    """One training-dataset "column" used across report and plots.

    ``prefix`` selects the ``{prefix}_reward_shaping*`` config keys used by this
    training run family (e.g. ``ifeval`` for IF-RLVR, ``gsm`` for RLVR-GSM).
    """

    key: str
    display: str
    mixer_list: list[str]
    prefix: str

    @property
    def mongo_clause(self) -> dict[str, Any]:
        return {"config.dataset_mixer_list": self.mixer_list}

    @property
    def expr_clause(self) -> str:
        return f"Config('dataset_mixer_list') == {self.mixer_list!r}"


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        key="IF-RLVR",
        display="IF-RLVR",
        mixer_list=["allenai/IF_multi_constraints_upto5", "1.0"],
        prefix="ifeval",
    ),
    DatasetSpec(
        key="RLVR-GSM",
        display="RLVR-GSM",
        mixer_list=["allenai/RLVR-GSM", "1.0"],
        prefix="gsm",
    ),
    DatasetSpec(
        key="RLVR-MATH",
        display="RLVR-MATH",
        mixer_list=["allenai/RLVR-MATH", "1.0"],
        prefix="math",
    ),
)

DATASETS_BY_KEY: dict[str, DatasetSpec] = {d.key: d for d in DATASETS}

# Short aliases the user may pass on the CLI (e.g. "IF", "GSM", "MATH").
DATASET_ALIASES: dict[str, str] = {
    "IF": "IF-RLVR",
    "IF-RLVR": "IF-RLVR",
    "IFRLVR": "IF-RLVR",
    "ifeval": "IF-RLVR",
    "GSM": "RLVR-GSM",
    "RLVR-GSM": "RLVR-GSM",
    "gsm": "RLVR-GSM",
    "MATH": "RLVR-MATH",
    "RLVR-MATH": "RLVR-MATH",
    "math": "RLVR-MATH",
}


def resolve_dataset(name: str) -> DatasetSpec:
    """Return the DatasetSpec for any of its known aliases (case-sensitive keys)."""
    try:
        key = DATASET_ALIASES[name]
    except KeyError as exc:
        known = sorted(set(DATASET_ALIASES))
        raise ValueError(f"unknown dataset {name!r}; known aliases: {known}") from exc
    return DATASETS_BY_KEY[key]


# ----- Shaping-arm taxonomy ------------------------------------------------

# CLI / label / mnemonic for each approach arm the report compares.
APPROACH_KINDS: tuple[tuple[str, str], ...] = (
    ("Baseline", "baseline"),
    ("Baseline (random-zero)", "baseline_rz"),
    ("Shaping only", "shaping"),
    (r"Curriculum ($\alpha{=}1$)", "curr_a1"),
    (r"Curriculum ($\alpha{=}10$)", "curr_a10"),
)

# Order + plain-text (no TeX) labels; handy for CSV/LaTeX column headers.
APPROACH_ORDER: tuple[str, ...] = tuple(kind for _, kind in APPROACH_KINDS)

APPROACH_PLAIN_LABELS: dict[str, str] = {
    "baseline": "Baseline",
    "baseline_rz": "Baseline (random-zero)",
    "shaping": "Shaping only",
    "curr_a1": "Curriculum (alpha=1)",
    "curr_a10": "Curriculum (alpha=10)",
}

APPROACH_TEX_LABELS: dict[str, str] = {
    "baseline": "Baseline",
    "baseline_rz": "Baseline (random-zero)",
    "shaping": "Shaping only",
    "curr_a1": r"Curriculum ($\alpha{=}1$)",
    "curr_a10": r"Curriculum ($\alpha{=}10$)",
}

# Kinds shown in bar charts / slim plots (drops the alpha=1 ablation).
PAPER_APPROACH_KINDS: tuple[str, ...] = ("baseline", "baseline_rz", "shaping", "curr_a10")

# Runs tagged for known-broken behaviour; dropped from every analysis.
EXCLUDED_RUN_TAGS: tuple[str, ...] = ("broken-async", "broken-ifeval-eval-log")


# ----- Config-value coercion (for classifying a fetched run) ---------------


def _flat_config(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Unwrap ``{"value": x}`` wrappers used by the W&B public API."""
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


def _coerce_alpha(cfg: dict[str, Any], prefix: str) -> float | None:
    v = cfg.get(f"{prefix}_competence_alpha")
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def classify_run_kind(cfg: dict[str, Any], prefix: str) -> str | None:
    """Return the approach kind ("baseline"/"baseline_rz"/"shaping"/"curr_a1"/"curr_a10").

    ``baseline_rz`` is baseline training with ``{prefix}_random_zero_reward`` enabled.
    Returns ``None`` for runs that do not match any approved arm
    (e.g. shaping+curriculum with an alpha other than 1 or 10).
    """
    cfg = _flat_config(cfg)
    rs = _coerce_bool(cfg.get(f"{prefix}_reward_shaping", False))
    curr = _coerce_bool(cfg.get(f"{prefix}_reward_shaping_curriculum", False))
    alpha = _coerce_alpha(cfg, prefix)
    if not rs and not curr:
        if _coerce_bool(cfg.get(f"{prefix}_random_zero_reward", False)):
            return "baseline_rz"
        return "baseline"
    if rs and not curr:
        return "shaping"
    if rs and curr:
        if alpha is not None and abs(alpha - 1.0) < 1e-9:
            return "curr_a1"
        if alpha is not None and abs(alpha - 10.0) < 1e-9:
            return "curr_a10"
    return None


def group_key_tuple(cfg: dict[str, Any], prefix: str) -> tuple[bool, bool, float | None, bool]:
    """Grouping key (rs, curr, alpha, random_zero); last component only applies when rs and curr are false."""
    cfg = _flat_config(cfg)
    rs = _coerce_bool(cfg.get(f"{prefix}_reward_shaping", False))
    curr = _coerce_bool(cfg.get(f"{prefix}_reward_shaping_curriculum", False))
    alpha = _coerce_alpha(cfg, prefix)
    if not (rs and curr):
        alpha = None
    elif alpha is not None:
        alpha = round(float(alpha), 6)
    rz = False
    if not rs and not curr:
        rz = _coerce_bool(cfg.get(f"{prefix}_random_zero_reward", False))
    return (rs, curr, alpha, rz)


def group_key_json(gk: tuple[bool, bool, float | None, bool]) -> str:
    return json.dumps([gk[0], gk[1], gk[2], gk[3]], separators=(",", ":"))


def group_key_from_json(s: str) -> tuple[bool, bool, float | None, bool]:
    a = json.loads(s)
    if len(a) == 3:
        alpha = a[2]
        if alpha is not None:
            alpha = float(alpha)
        return (bool(a[0]), bool(a[1]), alpha, False)
    alpha = a[2]
    if alpha is not None:
        alpha = float(alpha)
    return (bool(a[0]), bool(a[1]), alpha, bool(a[3]))


def kind_to_group_key(kind: str) -> tuple[bool, bool, float | None, bool]:
    if kind == "baseline":
        return (False, False, None, False)
    if kind == "baseline_rz":
        return (False, False, None, True)
    if kind == "shaping":
        return (True, False, None, False)
    if kind == "curr_a1":
        return (True, True, 1.0, False)
    if kind == "curr_a10":
        return (True, True, 10.0, False)
    raise ValueError(f"unknown approach kind: {kind!r}")


# ----- Mongo-style filters (wandb.Api) -------------------------------------


def _config_eq_false_or_unset_mongo(key: str) -> dict[str, Any]:
    """Treat False, null, and "field missing" as equivalent (matches dataclass defaults)."""
    return {
        "$or": [
            {key: False},
            {key: None},
            {key: {"$exists": False}},
        ]
    }


def excluded_run_tags_mongo() -> dict[str, Any]:
    return {"$nor": [{"tags": t} for t in EXCLUDED_RUN_TAGS]}


def shaping_arms_mongo(prefix: str) -> dict[str, Any]:
    """Mongo "$or" across allowed shaping arms for a given prefix (incl. both baselines)."""
    rs = f"config.{prefix}_reward_shaping"
    curr = f"config.{prefix}_reward_shaping_curriculum"
    alpha = f"config.{prefix}_competence_alpha"
    f_rs = _config_eq_false_or_unset_mongo(rs)
    f_curr = _config_eq_false_or_unset_mongo(curr)
    return {
        "$or": [
            {"$and": [f_rs, f_curr]},
            {"$and": [{rs: True}, f_curr]},
            {"$and": [{rs: True}, {curr: True}, {alpha: {"$in": [1, 1.0, 10, 10.0]}}]},
        ]
    }


def approach_mongo(prefix: str, kind: str) -> dict[str, Any]:
    rs = f"config.{prefix}_reward_shaping"
    curr = f"config.{prefix}_reward_shaping_curriculum"
    alpha = f"config.{prefix}_competence_alpha"
    rz = f"config.{prefix}_random_zero_reward"
    f_rs = _config_eq_false_or_unset_mongo(rs)
    f_curr = _config_eq_false_or_unset_mongo(curr)
    if kind == "baseline":
        return {"$and": [f_rs, f_curr, _config_eq_false_or_unset_mongo(rz)]}
    if kind == "baseline_rz":
        return {"$and": [f_rs, f_curr, {rz: True}]}
    if kind == "shaping":
        return {"$and": [{rs: True}, f_curr]}
    if kind == "curr_a1":
        return {"$and": [{rs: True}, {curr: True}, {alpha: {"$in": [1, 1.0]}}]}
    if kind == "curr_a10":
        return {"$and": [{rs: True}, {curr: True}, {alpha: {"$in": [10, 10.0]}}]}
    raise ValueError(f"unknown approach kind: {kind!r}")


def dataset_filter_mongo(
    dataset: DatasetSpec,
    *,
    model_paths: list[str] | None = None,
    approach_kind: str | None = None,
) -> dict[str, Any]:
    """Full mongo filter: dataset + lr + model(s) + shaping arms + exclusion tags."""
    if model_paths is None:
        model_paths = list(MODEL_NAME_OR_PATH.values())
    arms = approach_mongo(dataset.prefix, approach_kind) if approach_kind else shaping_arms_mongo(dataset.prefix)
    return {
        "$and": [
            model_lr_filter_mongo(model_paths),
            dataset.mongo_clause,
            arms,
            excluded_run_tags_mongo(),
        ]
    }


# ----- Expression-string filters (wandb_workspaces.Runset) -----------------


def _config_eq_false_or_unset_expr(config_call: str) -> str:
    return f"(({config_call} == False) or ({config_call} == None))"


def excluded_run_tags_expr() -> str:
    items = ", ".join(repr(t) for t in EXCLUDED_RUN_TAGS)
    return f"Tags() not in [{items}]"


def shaping_arms_expr(prefix: str) -> str:
    rs = f"Config('{prefix}_reward_shaping')"
    curr = f"Config('{prefix}_reward_shaping_curriculum')"
    alpha = f"Config('{prefix}_competence_alpha')"
    f_rs = _config_eq_false_or_unset_expr(rs)
    f_curr = _config_eq_false_or_unset_expr(curr)
    return (
        f"(({f_rs}) and ({f_curr})) or "
        f"(({rs} == True) and ({f_curr})) or "
        f"(({rs} == True) and ({curr} == True) and (({alpha} == 1) or ({alpha} == 1.0))) or "
        f"(({rs} == True) and ({curr} == True) and (({alpha} == 10) or ({alpha} == 10.0)))"
    )


def approach_expr(prefix: str, kind: str) -> str:
    rs = f"Config('{prefix}_reward_shaping')"
    curr = f"Config('{prefix}_reward_shaping_curriculum')"
    alpha = f"Config('{prefix}_competence_alpha')"
    rz = f"Config('{prefix}_random_zero_reward')"
    f_rs = _config_eq_false_or_unset_expr(rs)
    f_curr = _config_eq_false_or_unset_expr(curr)
    f_rz = _config_eq_false_or_unset_expr(rz)
    if kind == "baseline":
        return f"(({f_rs}) and ({f_curr}) and ({f_rz}))"
    if kind == "baseline_rz":
        return f"(({f_rs}) and ({f_curr}) and ({rz} == True))"
    if kind == "shaping":
        return f"(({rs} == True) and ({f_curr}))"
    if kind == "curr_a1":
        return f"(({rs} == True) and ({curr} == True) and (({alpha} == 1) or ({alpha} == 1.0)))"
    if kind == "curr_a10":
        return f"(({rs} == True) and ({curr} == True) and (({alpha} == 10) or ({alpha} == 10.0)))"
    raise ValueError(f"unknown approach kind: {kind!r}")


def runset_filter_expr(
    model_path: str,
    dataset: DatasetSpec,
    *,
    approach_kind: str | None = None,
) -> str:
    lr = learning_rate_for_model_path(model_path)
    parts = [
        f"Config('model_name_or_path') == '{model_path}'",
        f"Config('learning_rate') == {lr}",
    ]
    parts.append(f"({dataset.expr_clause})")
    parts.append(approach_expr(dataset.prefix, approach_kind) if approach_kind else shaping_arms_expr(dataset.prefix))
    parts.append(excluded_run_tags_expr())
    return " and ".join(f"({p})" for p in parts)


def groupby_config_keys(prefix: str) -> list[str]:
    return [
        f"config.{prefix}_reward_shaping",
        f"config.{prefix}_reward_shaping_curriculum",
        f"config.{prefix}_competence_alpha",
        f"config.{prefix}_random_zero_reward",
    ]
