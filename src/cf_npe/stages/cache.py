"""Save and reload Stage-1 outputs so Stage-2 runs reuse one K-fold mean ensemble.

Caches live in ``<out>/cache/`` of the ``run-sweep`` call that built them and are
consumed with ``run-sweep --stage1-from <out>``. One .pt file per seed, tensors on CPU:
    seed, n_train, n_folds: int
    problem: _problem_identity(problem), the problem's class name and dimensions
    x (n_train, dim_x); z, mu_hat_oof, mu_hat_infold (n_train, dim_z)
    fold_model_states: state dicts of the test-time ensemble (fold 0's M replicates)
    encoder_cfg: asdict(EncoderConfig)
    arch_fingerprint: Problem.arch_fingerprint(), None unless encoder_cfg misses the
        architecture (strong lensing)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from cf_npe.networks.models import EnsembleMeanNet, MeanRegressor
from cf_npe.problems.base import EncoderConfig, Problem
from cf_npe.stages.stage1 import Stage1Result, build_mean_regressor
from cf_npe.training.parallel_seed import SeedData

_logger = logging.getLogger(__name__)


def _problem_identity(problem: Problem) -> dict:
    return {"class": type(problem).__name__, "dim_x": problem.dim_x, "dim_z": problem.dim_z}


def save_cache(
    path: str | Path,
    problem: Problem,
    setup: SeedData,
    s1: Stage1Result,
    encoder_cfg: EncoderConfig,
) -> None:
    """Write one seed's Stage-1 cache file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": setup.seed,
        "n_train": setup.x.shape[0],
        "n_folds": len(setup.splits),
        "problem": _problem_identity(problem),
        "x": setup.x.detach().cpu(),
        "z": setup.z.detach().cpu(),
        "mu_hat_oof": s1.mu_hat_oof.detach().cpu(),
        "mu_hat_infold": s1.mu_hat_infold.detach().cpu(),
        "fold_model_states": [
            {k: v.detach().cpu() for k, v in m.state_dict().items()} for m in s1.ensemble.models
        ],
        "encoder_cfg": asdict(encoder_cfg),
        "arch_fingerprint": problem.arch_fingerprint(),
    }
    torch.save(payload, path)


def load_cache(
    path: str | Path,
    problem: Problem,
    expected_seed: int,
    expected_n_train: int,
    in_fold_residuals: bool,
) -> Stage1Cache:
    """Load a Stage-1 cache and rebuild its test-time ensemble on CUDA; raises
    ValueError if its problem, seed, n_train or architecture fingerprint differs."""
    payload = torch.load(Path(path), weights_only=False)
    if payload["problem"] != _problem_identity(problem):
        raise ValueError(f"cache problem {payload['problem']} != {_problem_identity(problem)}")
    if payload["seed"] != expected_seed:
        raise ValueError(f"cache seed {payload['seed']} != cfg.seed {expected_seed}")
    if payload["n_train"] != expected_n_train:
        raise ValueError(f"cache n_train {payload['n_train']} != cfg.n_train {expected_n_train}")
    expected_fp = problem.arch_fingerprint()
    cached_fp = payload["arch_fingerprint"]
    if cached_fp != expected_fp:
        raise ValueError(
            f"cache arch_fingerprint {cached_fp} != expected {expected_fp}; "
            "rebuild the Stage-1 cache with the current architecture"
        )

    encoder_cfg = EncoderConfig(**payload["encoder_cfg"])
    replicates: list[MeanRegressor] = []
    for state in payload["fold_model_states"]:
        m = build_mean_regressor(problem, encoder_cfg, "cuda")
        m.load_state_dict(state)
        m.eval()
        replicates.append(m)
    ensemble = EnsembleMeanNet(replicates)
    _logger.info(
        "Loaded Stage-1 cache %s: M=%d replicates/fold, n_folds=%d, seed=%d, n_train=%d",
        path,
        len(replicates),
        payload["n_folds"],
        payload["seed"],
        payload["n_train"],
    )

    return Stage1Cache(
        x=payload["x"].to("cuda"),
        z=payload["z"].to("cuda"),
        mu_hat=payload["mu_hat_infold" if in_fold_residuals else "mu_hat_oof"].to("cuda"),
        ensemble=ensemble,
        n_folds=payload["n_folds"],
    )


@dataclass
class Stage1Cache:
    """A Stage-1 cache on CUDA. ``mu_hat`` holds the out-of-fold means, or the in-fold
    means for the in-fold-residuals ablation."""

    x: Tensor
    z: Tensor
    mu_hat: Tensor
    ensemble: EnsembleMeanNet
    n_folds: int
