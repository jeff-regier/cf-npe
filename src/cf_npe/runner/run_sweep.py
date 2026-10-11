"""Launch an experiment sweep and aggregate its records into one results file.

Jobs run one at a time: a single job saturates the GPU because trainers vmap ensemble
members and ``seeds_per_job`` consecutive seeds together.

Usage::

    run-sweep --out out/runs/gauss_d200_std_npe_gaussian \\
        --results out/results/gauss_d200_std_npe_gaussian.json \\
        --experiment std_npe_test_ll \\
        --n-seeds 10 \\
        --method-from conf/recipes/gauss_d200_std_npe_gaussian \\
        --set train_params.posterior_type=gaussian --set problem.dim=200 \\
        --set seeds_per_job=5 problem=gaussian

Remaining arguments (``problem=gaussian`` above) are passed to every job as Hydra overrides.
Re-running with the same ``--out`` skips units that already have a record. The results
file is written only once every unit has a record.
"""

import argparse
import logging
import sys
from collections.abc import Container
from pathlib import Path

import yaml

from cf_npe import CONF, load_scale
from cf_npe.runner.aggregate import aggregate
from cf_npe.runner.launch import launch_sweep

logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def recipe_overlay(spec: str) -> dict:
    """The dotted Hydra overrides in the ``best_method.yaml`` of a recipe spec
    ``DIR[:PREFIX,...]``; with prefixes, keep only the keys under them (each must match)."""
    recipe_dir, _, prefixes = spec.partition(":")
    overlay = _load_yaml(Path(recipe_dir) / "best_method.yaml")
    if not prefixes:
        return overlay
    selected = {}
    for prefix in prefixes.split(","):
        keys = [k for k in overlay if k == prefix or k.startswith(prefix + ".")]
        if not keys:
            raise ValueError(f"recipe {recipe_dir} has no keys under {prefix!r}")
        selected.update({k: overlay[k] for k in keys})
    return selected


def merge_recipes(overlays: list[dict]) -> dict:
    """Union of recipe overlays; two recipes setting one key is an error."""
    merged: dict = {}
    for overlay in overlays:
        shared = merged.keys() & overlay.keys()
        if shared:
            raise ValueError(f"recipes both set {sorted(shared)}")
        merged.update(overlay)
    return merged


def smoke_sizes(train_params: Container[str]) -> dict:
    """Training sizes the smoke preset forces on an experiment with ``train_params``:
    n_train, and the epochs of each training stage's TrainingConfig (``training``, Stage 2's
    ``stage2_training``)."""
    smoke = load_scale(True)
    stages = [key for key in ("training", "stage2_training") if key in train_params]
    return {
        "n_train": smoke["n_train"],
        **{f"train_params.{key}.n_epochs": smoke["n_epochs"] for key in stages},
    }


def _seed_batches(seeds: list[int], seeds_per_job: int) -> list[tuple[int, int]]:
    """Group consecutive seeds into ``(base_seed, batch_size)`` batches."""
    return [
        (seeds[i], min(seeds_per_job, len(seeds) - i)) for i in range(0, len(seeds), seeds_per_job)
    ]


def _run_experiment(
    args: argparse.Namespace,
    seeds: list[int],
    recipe: dict,
    overrides: dict,
    stage1_from: str | None,
    hydra_overrides: list[str],
) -> int:
    """Launch the experiment as a content-addressed sweep with one grid point per seed
    batch; ``--set`` overrides win over the recipe."""
    exp = _load_yaml(CONF / "experiment" / f"{args.experiment}.yaml")
    overrides = dict(overrides)
    seeds_per_job = int(overrides.pop("seeds_per_job", exp["seeds_per_job"]))
    grid = [
        {
            **recipe,
            **overrides,
            **(smoke_sizes(exp["train_params"]) if args.smoke else {}),
            "seed": base_seed,
            "seeds_per_job": batch_size,
        }
        for base_seed, batch_size in _seed_batches(seeds, seeds_per_job)
    ]
    print(f"  {args.experiment}: {len(grid)} jobs", flush=True)
    return launch_sweep(
        args.experiment,
        "smoke" if args.smoke else "large",
        args.out / "multirun" / args.experiment,
        grid,
        out_dir=args.out,
        stage1_from=stage1_from,
        hydra_overrides=hydra_overrides,
    )


def parse_set_overrides(items: list[str]) -> dict:
    """Parse repeated ``--set KEY=VALUE``; values are parsed as YAML."""
    out = {}
    for item in items:
        key, sep, val = item.partition("=")
        if not sep:
            sys.exit(f"--set expects KEY=VALUE, got {item!r}")
        out[key] = yaml.safe_load(val)
    return out


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Launch an experiment sweep")
    parser.add_argument("--experiment", required=True, metavar="NAME")
    parser.add_argument(
        "--out", type=Path, required=True, help="Output dir (multirun/, cache/, models/)."
    )
    parser.add_argument(
        "--results", type=Path, required=True, help="JSON file to aggregate the records into."
    )
    parser.add_argument(
        "--method-from",
        action="append",
        default=[],
        metavar="RECIPE_DIR[:PREFIX,...]",
        help="Apply <RECIPE_DIR>/best_method.yaml as fixed overrides, only the keys under "
        "PREFIX if given (repeatable; recipes must not set the same key).",
    )
    parser.add_argument(
        "--stage1-from",
        type=Path,
        default=None,
        metavar="DIR",
        help="Output dir of a Stage-1 cache build; exposed to jobs as ``stage1_from``.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="set_overrides",
        help="Fixed Hydra override for every job (repeatable); wins over --method-from.",
    )
    parser.add_argument("--n-seeds", type=int, required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="scale=smoke: tiny sizes",
    )
    return parser.parse_known_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args, hydra_overrides = _parse_args()
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    recipe = merge_recipes([recipe_overlay(spec) for spec in args.method_from])
    overrides = parse_set_overrides(args.set_overrides)
    stage1_from = str(args.stage1_from.resolve()) if args.stage1_from else None

    if _run_experiment(args, seeds, recipe, overrides, stage1_from, hydra_overrides):
        sys.exit(f"Failed/incomplete: {args.experiment}; re-run to fill missing units.")
    aggregate(args.out / "multirun" / args.experiment, args.results)
    print(f"Results aggregated to {args.results}")


if __name__ == "__main__":
    main()
