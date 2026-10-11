"""Sample Figure 4's arrays from the strong-lensing CF-NPE-mixture run.

Writes ``<root>/caches/strong_lensing_recon.npz``.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate

from cf_npe.stages.cache import load_cache
from cf_npe.stages.stage2 import untrained_two_stage

# Seed of the strong-lensing run (model, Stage-1 cache, data file) shown in Figure 4.
_SL_RECON_RUN_SEED = 0
_SL_RECON_EXAMPLE_SEED = 0
_SL_RECON_DRAW_SEED = 1


def build_sl_recon_cache(cache_path, run_dir):
    """Build Figure 4's array cache from the CF-NPE-mixture run in ``run_dir``."""
    model_path = run_dir / "models" / f"seed{_SL_RECON_RUN_SEED}.pt"
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    problem = instantiate({**payload["problem"], "is_test": True})
    stage2_cfg = instantiate(payload["train_params"]["stage2_cfg"])
    stage1_cache_path = payload["train_params"]["stage1_cache_path"].replace(
        "__SEED__", str(_SL_RECON_RUN_SEED)
    )
    cache = load_cache(
        stage1_cache_path,
        problem=problem,
        expected_seed=_SL_RECON_RUN_SEED,
        expected_n_train=payload["n_train"],
        in_fold_residuals=False,
    )
    # One Stage-2 head per Stage-1 fold; strict loading checks the count.
    model = untrained_two_stage(
        problem,
        payload["train_params"]["posterior_type"],
        stage2_cfg,
        cache.ensemble,
        cache.n_folds,
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()

    torch.manual_seed(_SL_RECON_EXAMPLE_SEED)
    observed, truth = problem.sample_joint(1)
    torch.manual_seed(_SL_RECON_DRAW_SEED)
    with torch.inference_mode():
        samples = model.sample(observed, 3)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        observed=observed.reshape(64, 64).cpu().numpy(),
        truth=truth.reshape(16, 16).cpu().numpy(),
        samples=samples.reshape(3, 16, 16).cpu().numpy(),
    )
    print(f"  Saved {cache_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample Figure 4's strong-lensing arrays.")
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root
    build_sl_recon_cache(
        root / "caches" / "strong_lensing_recon.npz",
        root / "runs" / "sl10_cf_npe_mixture_residual",
    )
