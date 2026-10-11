"""Process-wide bf16 autocast toggle, set once from ``cfg.use_amp``.

bf16's dynamic range avoids the loss scaling fp16 would need.
"""

from __future__ import annotations

import torch

ENABLED: bool = False


def set_enabled(enabled: bool) -> None:
    global ENABLED
    ENABLED = enabled


def autocast() -> torch.amp.autocast:
    return torch.amp.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=ENABLED,
    )
