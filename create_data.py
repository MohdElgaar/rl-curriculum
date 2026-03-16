#!/usr/bin/env python
"""
Utility to generate a synthetic IFEval-style dataset.

The script:
- samples random prompts,
- samples random constraints from `open_instruct.IFEvalG.instructions_registry.FUNCTION_DICT`,
- upsamples the smaller of {num_prompts, num_constraints} to match the larger,
- and writes a JSONL dataset under `data/`.

Each JSONL row has the following fields:
- messages: [{"role": "user", "content": prompt_with_constraints}]
- ground_truth: stringified list with a single dict
    [{"instruction_id": [...], "kwargs": [...]}]
- dataset: "ifeval"
- constraint: human-readable constraint_description
"""

import argparse
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from open_instruct.IFEvalG import instructions_registry, instructions_util
import pandas as pd


# Instructions that need the actual prompt passed as `prompt_to_repeat`.
PROMPT_ARG_INSTRUCTION_IDS = {
    "combination:repeat_prompt",  # RepeatPromptThenAnswer
    "copy:copy",  # CopyChecker
    "new:copy_span_idx",  # CopySpanIdxChecker
    "copy:copying_simple",  # CopyingSimpleChecker
    "copy:copying_multiple",  # CopyingMultipleChecker
}

# Instructions to ignore (e.g., rephrase that need complex construction).
IGNORE_INSTRUCTION_IDS = {
    "detectable_format:rephrase",  # RephraseChecker
    "keywords:exclude_word_harder",  # ExcludeWordHarderChecker needs instruction arg
}


def sample_constraint_id_combo(
    rng: random.Random,
    instruction_ids: List[str],
    max_constraints_per_example: int = 1,
    fixed_constraints_per_example: Optional[int] = None,
) -> List[str]:
    """Sample a random combination of instruction ids (a constraint blueprint)."""
    # Use fixed number if provided, otherwise random number between 1 and max.
    if fixed_constraints_per_example is not None:
        k = fixed_constraints_per_example
    else:
        k = rng.randint(1, max(1, max_constraints_per_example))

    if len(instruction_ids) >= k:
        return rng.sample(instruction_ids, k)
    return [rng.choice(instruction_ids) for _ in range(k)]


def instantiate_constraints_for_prompt(
    prompt: str,
    instruction_ids: List[str],
) -> Tuple[str, Dict[str, Any]]:
    """Instantiate constraints for a given prompt and list of instruction ids.

    Returns:
      - description: human-readable text combining all individual constraint descriptions.
      - ground_truth_entry: {"instruction_id": [...], "kwargs": [...]}
    """
    instantiated_ids: List[str] = []
    kwargs_list: List[Any] = []
    descriptions: List[str] = []

    for instruction_id in instruction_ids:
        instruction_cls = instructions_registry.FUNCTION_DICT[instruction_id]
        inst = instruction_cls(instruction_id)

        # Special handling for constraints that need the prompt.
        if instruction_id in PROMPT_ARG_INSTRUCTION_IDS:
            desc = inst.build_description(prompt_to_repeat=prompt)
        else:
            desc = inst.build_description()

        instantiated_ids.append(instruction_id)
        kwargs_list.append(inst.get_instruction_args())
        descriptions.append(desc)

    constraint_description = " ".join(descriptions)
    ground_truth_entry: Dict[str, Any] = {
        "instruction_id": instantiated_ids,
        "kwargs": kwargs_list,
    }
    return constraint_description, ground_truth_entry


def generate_prompts(rng: random.Random, num_prompts: int) -> List[str]:
    """Generate simple generic prompts using the IFEval word list."""
    prompts: List[str] = []
    for _ in range(num_prompts):
        topic = rng.choice(instructions_util.WORD_LIST)
        prompts.append(f"Write a detailed and coherent paragraph about {topic}.")
    return prompts


def upsample_indices(rng: random.Random, n_small: int, n_large: int) -> List[int]:
    """Return indices of length n_large by randomly upsampling [0, n_small)."""
    if n_small <= 0:
        raise ValueError("Cannot upsample from an empty list.")
    return [rng.randrange(n_small) for _ in range(n_large)]


def build_examples(
    rng: random.Random,
    prompts: List[str],
    constraint_blueprints: List[List[str]],
) -> List[Dict[str, Any]]:
    """Pair prompts with constraints (with upsampling) and build dataset rows."""
    n_prompts = len(prompts)
    n_constraints = len(constraint_blueprints)
    if n_prompts == 0 or n_constraints == 0:
        raise ValueError("Both num_prompts and num_constraints must be > 0.")

    target_size = max(n_prompts, n_constraints)

    if n_prompts == n_constraints:
        prompt_indices = list(range(target_size))
        constraint_indices = list(range(target_size))
    elif n_prompts < n_constraints:
        # Upsample prompts.
        prompt_indices = upsample_indices(rng, n_prompts, n_constraints)
        constraint_indices = list(range(n_constraints))
    else:
        # Upsample constraints.
        prompt_indices = list(range(n_prompts))
        constraint_indices = upsample_indices(rng, n_constraints, n_prompts)

    examples: List[Dict[str, Any]] = []
    for p_idx, c_idx in zip(prompt_indices, constraint_indices):
        prompt = prompts[p_idx]
        blueprint_ids = constraint_blueprints[c_idx]
        constraint_desc, gt_entry = instantiate_constraints_for_prompt(prompt, blueprint_ids)
        prompt_with_constraints = " ".join([prompt, constraint_desc])

        # Ground truth is stored as a stringified list with a single dict, to
        # match what `IFEvalVerifier` expects (it uses `ast.literal_eval`).
        ground_truth_str = repr([gt_entry])

        example = {
            "messages": [{"role": "user", "content": prompt_with_constraints}],
            "ground_truth": ground_truth_str,
            "dataset": "ifeval",
            "constraint": constraint_desc,
        }
        examples.append(example)

    return examples


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_json(path, orient="records", lines=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic IFEval-style constraint dataset.")
    parser.add_argument("--num-prompts", type=int, required=True, help="Number of base prompts to generate.")
    parser.add_argument("--num-constraints", type=int, required=True, help="Number of random constraint sets to generate.")
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/ifeval_synthetic.jsonl",
        help="Output JSONL path (must be under data/ by convention).",
    )
    parser.add_argument(
        "--max-constraints-per-example",
        type=int,
        default=1,
        help="Maximum number of constraints to combine in a single example (used when --fixed-constraints-per-example is not set).",
    )
    parser.add_argument(
        "--fixed-constraints-per-example",
        type=int,
        default=None,
        help="Fixed number of constraints per prompt. If set, overrides --max-constraints-per-example.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_prompts <= 0 or args.num_constraints <= 0:
        raise ValueError("Both --num-prompts and --num-constraints must be positive.")
    if args.fixed_constraints_per_example is not None and args.fixed_constraints_per_example <= 0:
        raise ValueError("--fixed-constraints-per-example must be positive.")

    rng = random.Random(args.seed)

    # 1) Prepare instruction types and prompts.
    instruction_ids = [
        instruction_id
        for instruction_id in instructions_registry.FUNCTION_DICT.keys()
        if instruction_id not in IGNORE_INSTRUCTION_IDS
    ]
    if not instruction_ids:
        raise RuntimeError("No usable instructions found in FUNCTION_DICT.")

    prompts = generate_prompts(rng, args.num_prompts)

    # 2) Sample constraint blueprints (lists of instruction ids).
    constraint_blueprints: List[List[str]] = []
    for _ in range(args.num_constraints):
        combo = sample_constraint_id_combo(
            rng,
            instruction_ids,
            max_constraints_per_example=args.max_constraints_per_example,
            fixed_constraints_per_example=args.fixed_constraints_per_example,
        )
        constraint_blueprints.append(combo)

    # 3) Pair prompts and constraints (instantiated per prompt), with upsampling as needed.
    rows = build_examples(rng, prompts, constraint_blueprints)

    # 4) Write dataset.
    write_jsonl(args.output_path, rows)
    print(f"Wrote {len(rows)} examples to {args.output_path}")


if __name__ == "__main__":
    main()


