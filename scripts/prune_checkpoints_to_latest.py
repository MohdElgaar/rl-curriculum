#!/usr/bin/env python3
"""
Keep only the highest-step DeepSpeed checkpoint under each training run directory.

By default prints what would be removed (dry run). Use --apply to delete.

See plan: validated global_step<N> dirs must look like ZeRO checkpoints; paired
ds_universal_global_step<N> dirs are removed when N != max_step.

Use --delete-all-universal to remove every ds_universal_global_step* directory
under each run (and drop latest_universal). That is independent of ZeRO pruning.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

GLOBAL_STEP_DIR_RE = re.compile(r"^global_step(\d+)$")
DS_UNIVERSAL_DIR_RE = re.compile(r"^ds_universal_global_step(\d+)$")


def is_ds_zero_checkpoint_dir(d: Path) -> bool:
    if not d.is_dir():
        return False
    if (d / "mp_rank_00_model_states.pt").is_file() or (
        d / "zero_pp_rank_0_mp_rank_00_model_states.pt"
    ).is_file():
        pass
    else:
        return False
    return any(d.glob("*_optim_states.pt"))


def discover_run_roots(outputs_root: Path) -> list[Path]:
    roots: list[Path] = []
    if not outputs_root.is_dir():
        return roots
    for child in sorted(outputs_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name == "autotune":
            for job in sorted(child.iterdir()):
                if not job.is_dir():
                    continue
                runs = job / "runs"
                if not runs.is_dir():
                    continue
                for subrun in sorted(runs.iterdir()):
                    if subrun.is_dir():
                        roots.append(subrun)
        else:
            roots.append(child)
    return roots


def collect_validated_global_steps(run_root: Path) -> dict[int, Path]:
    """step -> path for subdirectory names global_step<digits> that pass ZeRO layout checks."""
    out: dict[int, Path] = {}
    try:
        for entry in run_root.iterdir():
            if not entry.is_dir():
                continue
            m = GLOBAL_STEP_DIR_RE.match(entry.name)
            if not m:
                continue
            step = int(m.group(1))
            if is_ds_zero_checkpoint_dir(entry):
                out[step] = entry
    except OSError:
        pass
    return out


def collect_universal_dirs(run_root: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    try:
        for entry in run_root.iterdir():
            if not entry.is_dir():
                continue
            m = DS_UNIVERSAL_DIR_RE.match(entry.name)
            if not m:
                continue
            step = int(m.group(1))
            out[step] = entry
    except OSError:
        pass
    return out


def paths_to_prune(
    run_root: Path, validated: dict[int, Path], universal: dict[int, Path]
) -> tuple[int | None, list[Path]]:
    if not validated:
        return None, []

    max_step = max(validated.keys())
    remove: list[Path] = []

    for step, path in validated.items():
        if step != max_step:
            remove.append(path)

    for step, path in universal.items():
        if step != max_step:
            remove.append(path)

    return max_step, remove


def update_pointer_files(run_root: Path, max_step: int, verbose: bool) -> None:
    latest_path = run_root / "latest"
    tag = f"global_step{max_step}"
    latest_path.write_text(tag, encoding="utf-8")
    if verbose:
        print(f"  wrote {latest_path}: {tag!r}")

    universal_dir = run_root / f"ds_universal_global_step{max_step}"
    universal_path = run_root / "latest_universal"
    universal_tag = f"ds_universal_{tag}"
    if universal_dir.is_dir():
        universal_path.write_text(universal_tag, encoding="utf-8")
        if verbose:
            print(f"  wrote {universal_path}: {universal_tag!r}")
    else:
        try:
            if universal_path.is_file():
                universal_path.unlink()
                if verbose:
                    print(f"  removed {universal_path} (no {universal_dir.name})")
        except OSError as e:
            print(f"  warning: could not remove {universal_path}: {e}", file=sys.stderr)


def prune_run(
    run_root: Path,
    apply: bool,
    verbose: bool,
) -> tuple[bool, int]:
    """
    Returns (did_process, num_paths_removed_or_would_remove).
    did_process is False if nothing to do (0 or 1 checkpoint).
    """
    validated = collect_validated_global_steps(run_root)
    universal = collect_universal_dirs(run_root)
    max_step, to_remove = paths_to_prune(run_root, validated, universal)

    if max_step is None:
        if verbose:
            print(f"{run_root}: skip (no validated global_step* DeepSpeed checkpoints)")
        return False, 0

    if not to_remove:
        if verbose:
            print(f"{run_root}: skip (only one validated checkpoint at step {max_step})")
        return False, 0

    print(f"{run_root}")
    print(f"  max_step={max_step}  remove {len(to_remove)} path(s)")
    for p in sorted(to_remove, key=lambda x: str(x)):
        print(f"    {'DELETE' if apply else 'would delete'} {p}")

    if apply:
        for p in sorted(to_remove, key=lambda x: str(x)):
            shutil.rmtree(p, ignore_errors=False)
        update_pointer_files(run_root, max_step, verbose)

    return True, len(to_remove)


def delete_all_universal_run(
    run_root: Path, apply: bool, verbose: bool
) -> tuple[bool, int]:
    """
    Remove every ds_universal_global_step* directory under run_root.
    When apply=True, also removes latest_universal if present.
    Returns (touched, n_dirs_removed_or_would_remove).
    """
    universal = collect_universal_dirs(run_root)
    paths = sorted(universal.values(), key=lambda p: str(p))
    universal_path = run_root / "latest_universal"
    has_pointer = universal_path.is_file()

    if not paths and not has_pointer:
        if verbose:
            print(f"{run_root}: no ds_universal dirs and no latest_universal")
        return False, 0

    if not paths and has_pointer:
        print(f"{run_root}")
        print("  (no ds_universal dirs; stale latest_universal)")
        print(f"    {'DELETE' if apply else 'would delete'} {universal_path}")
        if apply:
            try:
                universal_path.unlink()
            except OSError as e:
                print(f"  warning: could not remove {universal_path}: {e}", file=sys.stderr)
        return True, 0

    print(f"{run_root}")
    print(f"  remove all universal: {len(paths)} dir(s)")
    for p in paths:
        print(f"    {'DELETE' if apply else 'would delete'} {p}")
    if has_pointer:
        print(f"    {'DELETE' if apply else 'would delete'} {universal_path}")

    n = len(paths)
    if apply:
        for p in paths:
            shutil.rmtree(p, ignore_errors=False)
        try:
            if universal_path.is_file():
                universal_path.unlink()
                if verbose:
                    print(f"  removed {universal_path}")
        except OSError as e:
            print(f"  warning: could not remove {universal_path}: {e}", file=sys.stderr)

    return True, n


def main() -> None:
    default_root = os.environ.get(
        "PRUNE_OUTPUTS_ROOT",
        "/scratch4/workspace/mohamed_elgaar_student_uml_edu-rl-curriculum/outputs",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path(default_root),
        help=f"Training outputs directory (default: env PRUNE_OUTPUTS_ROOT or {default_root})",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete checkpoints and update latest / latest_universal. "
        "Without this flag, only a dry run is performed.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N run roots (for testing).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print skip reasons and pointer file updates.",
    )
    parser.add_argument(
        "--delete-all-universal",
        action="store_true",
        help="Delete every ds_universal_global_step* directory under each run root and "
        "remove latest_universal. Does not change global_step* ZeRO dirs (use "
        "--apply without this flag for that).",
    )
    args = parser.parse_args()

    roots = discover_run_roots(args.outputs_root)
    if args.limit is not None:
        roots = roots[: args.limit]

    total_removed = 0
    processed = 0

    if args.delete_all_universal:
        for run_root in roots:
            did, n = delete_all_universal_run(run_root, apply=args.apply, verbose=args.verbose)
            if did:
                processed += 1
                total_removed += n
        mode = "apply" if args.apply else "dry-run"
        print(
            f"\nDone universal purge ({mode}): {processed} run(s) touched, "
            f"{total_removed} ds_universal dir(s) {'removed' if args.apply else 'would remove'}."
        )
        if not args.apply:
            print(
                "Re-run with --apply --delete-all-universal to perform deletion.",
                file=sys.stderr,
            )
        return

    for run_root in roots:
        did, n = prune_run(run_root, apply=args.apply, verbose=args.verbose)
        if did:
            processed += 1
            total_removed += n

    mode = "apply" if args.apply else "dry-run"
    print(
        f"\nDone ({mode}): {processed} run(s) with prunes, "
        f"{total_removed} path(s) {'removed' if args.apply else 'would remove'}."
    )
    if not args.apply:
        print("Re-run with --apply to perform deletion.", file=sys.stderr)


if __name__ == "__main__":
    main()
