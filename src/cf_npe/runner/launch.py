"""Serial launcher for one experiment's grid.

Each grid point (a seed batch) is hashed into a ``config_id`` and launched as a
``run_job`` subprocess into ``<sweep>/<config_id>/``. The grid's
``(config_id, seed)`` units are written to ``plan.json``; a point re-runs only if
one of its units has no record, so resuming is the same pass as launching.
"""

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import cast

from cf_npe.runner.provenance import config_id, expand_units, missing_units, read_plan, write_plan

logger = logging.getLogger(__name__)


def stage1_config_ids(stage1_from: str | None) -> list[str]:
    """The config_ids of the Stage-1 cache build in ``stage1_from``, so rebuilding that cache
    with another config changes the identity of every Stage-2 unit that reads it."""
    if stage1_from is None:
        return []
    plans = Path(stage1_from).glob("multirun/*/plan.json")
    return sorted({u["config_id"] for plan in plans for u in read_plan(plan)})


def unit_config_id(
    experiment: str,
    point: dict,
    scale: str,
    stage1_from: str | None,
    stage1_ids: list[str],
    hydra_overrides: list[str],
) -> str:
    """Content hash of a grid point (including its seed batch), its experiment, scale, Hydra overrides,
    and Stage-1 source (its directory and ``stage1_ids``, the config_ids of the cache built
    there)."""
    ident = {
        **point,
        "_experiment": experiment,
        "_scale": scale,
        "_hydra_overrides": hydra_overrides,
        "_stage1_from": stage1_from,
        "_stage1_config_ids": stage1_ids,
    }
    return config_id(ident)


def _job_overrides(
    experiment: str,
    scale: str,
    job_dir: Path,
    point: dict[str, object],
    out_dir: Path,
    stage1_from: str | None,
    hydra_overrides: list[str],
) -> list[str]:
    overrides = [
        f"+experiment={experiment}",
        f"scale={scale}",
        f"hydra.run.dir={job_dir}",
        f"+out_dir={out_dir.resolve()}",
        *hydra_overrides,
    ]
    if stage1_from is not None:
        overrides.append(f"+stage1_from={stage1_from}")
    for key, val in point.items():
        if key == "config_id":
            # Quoted: a numeric-looking hash (all digits, or digit-e-digit) would be
            # type-inferred by Hydra and break the plan<->record unit match.
            overrides.append(f"++config_id='{val}'")
            continue
        prefix = "" if key == "seed" else "++"
        overrides.append(f"{prefix}{key}={'null' if val is None else val}")
    return overrides


def _run_job_subprocess(overrides: list[str]) -> int:
    cmd = [sys.executable, "-m", "cf_npe.runner.run_job", *overrides]
    result = subprocess.run(cmd, check=False, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")
        logger.error("Job failed (rc=%d): %s", result.returncode, stderr[-500:])
    return result.returncode


def _collect_records(sweep_dir: Path) -> list[dict]:
    return [json.loads(path.read_text()) for path in sweep_dir.rglob("result.json")]


def _log_progress(n_done: int, n_total: int, n_failed: int, elapsed_min: float) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    fail_part = f", {n_failed} failed" if n_failed else ""
    print(f"    [{now}] {n_done}/{n_total}{fail_part}, {elapsed_min:.1f} min", flush=True)


def _run_points(
    experiment: str,
    scale: str,
    sweep_dir: Path,
    pending: list[dict],
    out_dir: Path,
    stage1_from: str | None,
    hydra_overrides: list[str],
) -> int:
    """Run each pending point as a job subprocess; return the number that failed."""
    t0 = monotonic()
    n_failed = 0
    for n_done, point in enumerate(pending, start=1):
        job_dir = sweep_dir / cast(str, point["config_id"])
        overrides = _job_overrides(
            experiment, scale, job_dir, point, out_dir, stage1_from, hydra_overrides
        )
        n_failed += _run_job_subprocess(overrides) != 0
        _log_progress(n_done, len(pending), n_failed, (monotonic() - t0) / 60)
    return n_failed


def launch_sweep(
    experiment: str,
    scale: str,
    sweep_dir: Path,
    grid: list[dict],
    *,
    out_dir: Path,
    stage1_from: str | None,
    hydra_overrides: list[str],
) -> int:
    """Write the plan, run every point with a missing unit, return 1 if any unit is missing."""
    sweep_dir.mkdir(parents=True, exist_ok=True)
    stage1_ids = stage1_config_ids(stage1_from)
    for point in grid:
        point["config_id"] = unit_config_id(
            experiment, point, scale, stage1_from, stage1_ids, hydra_overrides
        )
    plan_path = sweep_dir / "plan.json"
    write_plan(plan_path, expand_units(grid))

    records = _collect_records(sweep_dir)
    pending = [point for point in grid if missing_units(expand_units([point]), records)]
    if len(grid) > len(pending):
        print(f"    {len(grid) - len(pending)}/{len(grid)} points already complete", flush=True)
    n_failed = _run_points(
        experiment, scale, sweep_dir, pending, out_dir, stage1_from, hydra_overrides
    )

    for point in pending:
        records += _collect_records(sweep_dir / cast(str, point["config_id"]))
    missing = missing_units(read_plan(plan_path), records)
    if missing:
        logger.error(
            "%s: %d declared units have no record (first: %s)",
            sweep_dir.name,
            len(missing),
            missing[0],
        )
    return 1 if (n_failed or missing) else 0
