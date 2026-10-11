"""Stage-1: K-fold cross-validated mean-network training."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import cast

import torch
from torch import Tensor

from cf_npe.networks.models import EnsembleMeanNet, MeanRegressor
from cf_npe.problems.base import EncoderConfig, Problem
from cf_npe.training.loop import TrainingConfig, model_loss
from cf_npe.training.parallel_seed import SeedData, train_seeds


def build_mean_regressor(
    problem: Problem, encoder_cfg: EncoderConfig, device: torch.device | str
) -> MeanRegressor:
    """Build an untrained Stage-1 mean network (encoder + problem mean head)."""
    encoder = problem.create_encoder(encoder_cfg)
    head_dim = problem.encoder_head_dim(encoder_cfg)
    mean_head = problem.create_mean_head(head_dim, problem.dim_z)
    return MeanRegressor(encoder, mean_head).to(device)


def _scatter_means(
    mini_ensembles: list[EnsembleMeanNet],
    folds: list[tuple[Tensor, Tensor]],
    x: Tensor,
    z: Tensor,
) -> tuple[Tensor, Tensor]:
    """Fold k's validation points get their out-of-fold μ̂ from mini-ensemble k, trained
    without them, and their in-fold μ̂ from mini-ensemble (k+1) mod K, trained on them.
    A point in no validation fold stays NaN."""
    for mini in mini_ensembles:
        mini.eval()
    mu_hat_oof = torch.full_like(z, float("nan"))
    mu_hat_infold = torch.full_like(z, float("nan"))
    n_folds = len(folds)
    for k, (_, val_idx) in enumerate(folds):
        with torch.no_grad():
            mu_hat_oof[val_idx] = mini_ensembles[k].get_mean(x[val_idx])
            mu_hat_infold[val_idx] = mini_ensembles[(k + 1) % n_folds].get_mean(x[val_idx])
    return mu_hat_oof, mu_hat_infold


def _assemble_seed(
    setup: SeedData,
    nets: list[MeanRegressor],
    n_replicates: int,
    val_curves: list[list[float]],
) -> Stage1Result:
    """Group K*M nets, ordered (fold, replicate), into K mini-ensembles; fold 0's is
    the test-time ensemble."""
    minis = [
        EnsembleMeanNet(nets[k * n_replicates : (k + 1) * n_replicates])
        for k in range(len(setup.splits))
    ]
    mu_hat_oof, mu_hat_infold = _scatter_means(minis, setup.splits, setup.x, setup.z)
    return Stage1Result(mu_hat_oof, mu_hat_infold, minis[0], val_curves)


def run_stage1(
    problem: Problem,
    setups: list[SeedData],
    *,
    n_replicates: int,
    training: TrainingConfig,
    encoder_cfg: EncoderConfig,
    parallel: bool,
) -> list[Stage1Result]:
    """Train every seed's K folds x M replicates in one ``train_seeds`` call.

    Replicate m of fold k trains on fold k's complement, is early-stopped on fold k,
    and is initialized from ``combine_init_seed(seed, k * M + m)``.
    """
    device = setups[0].x.device
    results = train_seeds(
        [replace(s, splits=[f for f in s.splits for _ in range(n_replicates)]) for s in setups],
        lambda _s, _k: build_mean_regressor(problem, encoder_cfg, device),
        training=training,
        loss_fn=model_loss,
        parallel=parallel,
    )
    return [
        _assemble_seed(setup, cast(list[MeanRegressor], r.members), n_replicates, r.val_losses)
        for setup, r in zip(setups, results, strict=True)
    ]


@dataclass
class Stage1Result:
    """Stage-1 result for one seed.

    ``mu_hat_infold`` serves the in-fold-residuals ablation. ``ensemble`` is the
    test-time mean: fold 0's M replicates, never a cross-fold average. Validation
    loss curves are indexed [net][epoch].
    """

    mu_hat_oof: Tensor
    mu_hat_infold: Tensor
    ensemble: EnsembleMeanNet
    val_losses: list[list[float]]
