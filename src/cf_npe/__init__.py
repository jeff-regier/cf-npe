"""Cross-Fitting Neural Posterior Estimation (CF-NPE)."""

from pathlib import Path

import yaml

CONF = Path(__file__).resolve().parents[2] / "conf"


def load_scale(smoke: bool) -> dict:
    """The smoke or large preset in ``conf/scale/``."""
    return yaml.safe_load((CONF / "scale" / f"{'smoke' if smoke else 'large'}.yaml").read_text())
