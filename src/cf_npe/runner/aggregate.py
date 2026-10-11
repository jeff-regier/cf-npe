"""Aggregate a sweep's records into one JSON file.

A record whose ``(config_id, seed)`` the sweep's ``plan.json`` does not declare is
skipped with a warning.
"""

import json
import logging
from pathlib import Path

from cf_npe.runner.provenance import read_plan

logger = logging.getLogger(__name__)


def aggregate(sweep_dir: Path, dest: Path) -> None:
    """Write the plan-declared records under ``sweep_dir`` to ``dest``."""
    plan_path = sweep_dir / "plan.json"
    declared = {(u["config_id"], u["seed"]) for u in read_plan(plan_path)}
    results = []
    for result_path in sorted(sweep_dir.rglob("result.json")):
        result = json.loads(result_path.read_text())
        if (result["config_id"], result["seed"]) not in declared:
            logger.warning("Skipping stale record %s: not declared in %s", result_path, plan_path)
            continue
        results.append(result)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(results, indent=2))
    logger.info("Wrote %d results to %s", len(results), dest)
