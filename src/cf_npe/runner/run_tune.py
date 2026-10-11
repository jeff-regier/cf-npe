"""Run a Hydra-Optuna tuning sweep and write its best configuration as a recipe.

Runs ``run_job`` in multirun mode with ``conf/sweep/<name>.yaml`` as the config,
then turns ``optimization_results.yaml`` into ``<out>/best_method.yaml``, which
``run-sweep --method-from <out>`` applies.

Usage::

    run-tune --sweep tune_stage2_std_npe_gaussian --seed 1000 \\
        --out out/tune/recipes/gauss_d200_std_npe_gaussian
    run-tune --sweep sl_cf_npe_stage2_residual_nsf --seed 1000 \\
        --out out/tune/recipes/strong_lensing_cf_npe_nsf_residual \\
        --stage1-from out/tune/runs/sl10_cf_npe_m3_stage1
    run-tune --sweep tune_stage1_gaussian --seed 1000 \\
        --out out/tune/recipes/gauss_d200_stage1 problem=gaussian --set problem.dim=200

Remaining arguments are passed to Hydra as overrides.
"""

import argparse
import math
import subprocess
import sys
from pathlib import Path

import yaml
from hydra import compose, initialize_config_dir

from cf_npe import CONF
from cf_npe.runner.run_sweep import (
    merge_recipes,
    parse_set_overrides,
    recipe_overlay,
    smoke_sizes,
)


def _write_best_method(opt_results_path: Path, out_path: Path, sweep_name: str) -> None:
    """Translate ``optimization_results.yaml`` into a recipe file."""
    opt = yaml.safe_load(opt_results_path.read_text())
    if not math.isfinite(opt["best_value"]):
        raise SystemExit(
            f"Sweep {sweep_name}: no trial succeeded (best value {opt['best_value']}); "
            "see the trials' run_job.log files. No recipe written."
        )
    header = f"# Best configuration of sweep {sweep_name} (best_value: {opt['best_value']}).\n"
    out_path.write_text(header + yaml.safe_dump(opt["best_params"], sort_keys=False))


def main() -> None:
    p = argparse.ArgumentParser(description="Launch a Hydra-Optuna tuning sweep.")
    p.add_argument("--sweep", required=True, help="conf/sweep/<name>.yaml to run.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--seed", type=int, required=True, help="First seed of every trial (seeds_per_job on)."
    )
    p.add_argument(
        "--stage1-from",
        type=Path,
        default=None,
        help="Output dir of a Stage-1 cache build; exposed to jobs as ``stage1_from``.",
    )
    p.add_argument(
        "--method-from",
        action="append",
        default=[],
        metavar="RECIPE_DIR[:PREFIX,...]",
        help="Fixed overrides from a recipe, as for run-sweep (repeatable).",
    )
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="set_overrides",
        help="Fixed override, as for run-sweep (repeatable); wins over --method-from.",
    )
    p.add_argument(
        "--smoke", action="store_true", help="Two trials at smoke scale (an end-to-end check)."
    )
    args, extra = p.parse_known_args()
    if args.smoke:
        with initialize_config_dir(config_dir=str(CONF), version_base=None):
            sweep = compose(f"sweep/{args.sweep}")
        extra += ["scale=smoke", "hydra.sweeper.n_trials=2"]
        extra += [f"{key}={val}" for key, val in smoke_sizes(sweep.train_params).items()]

    sweep_dir = (args.out / "multirun").resolve()
    overrides = [
        f"hydra.sweep.dir={sweep_dir}",
        f"+out_dir={args.out.resolve()}",
        f"seed={args.seed}",
    ]
    if args.stage1_from is not None:
        overrides.append(f"+stage1_from={args.stage1_from.resolve()}")
    fixed = {
        **merge_recipes([recipe_overlay(spec) for spec in args.method_from]),
        **parse_set_overrides(args.set_overrides),
    }
    overrides += [f"++{key}={'null' if val is None else val}" for key, val in fixed.items()]
    cmd = [
        sys.executable,
        "-m",
        "cf_npe.runner.run_job",
        f"--config-name=sweep/{args.sweep}",
        "-m",
        *overrides,
        *extra,
    ]
    print("Launching: " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    out = args.out / "best_method.yaml"
    _write_best_method(sweep_dir / "optimization_results.yaml", out, args.sweep)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
