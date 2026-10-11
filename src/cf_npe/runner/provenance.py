"""Content-addressed identity and completeness for sweep units.

A unit is one (config, seed) job. ``config_id`` hashes a grid point: a config together with
its seed batch (base seed and ``seeds_per_job``), because a seed's training shuffles and
dropout come from a random stream shared with the seeds batched with it. ``plan.json`` lists
every declared unit; ``missing_units`` compares the plan with the records on disk.
"""

import hashlib
import json
from pathlib import Path


def config_id(resolved: dict) -> str:
    """Content hash of a config (12 hex chars)."""
    blob = json.dumps(resolved, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def expand_units(grid: list[dict]) -> list[dict]:
    """One declared unit per seed of each seed-batched grid point."""
    units: list[dict] = []
    for point in grid:
        base = int(point["seed"])
        spec = {k: v for k, v in point.items() if k != "seeds_per_job"}
        units.extend({**spec, "seed": s} for s in range(base, base + int(point["seeds_per_job"])))
    return units


def write_plan(path: Path, units: list[dict]) -> None:
    path.write_text(json.dumps(units, indent=2, default=str))


def read_plan(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def missing_units(plan_units: list[dict], records: list[dict]) -> list[dict]:
    """Declared units with no matching record, keyed on ``(config_id, seed)``."""
    done = {(r["config_id"], r["seed"]) for r in records}
    return [u for u in plan_units if (u["config_id"], u["seed"]) not in done]
