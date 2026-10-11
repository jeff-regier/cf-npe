"""Standard NPE (Std-NPE) and its mixture (Std-NPE-mixture)."""

from __future__ import annotations

from typing import cast

from torch import Tensor, nn

from cf_npe.networks.models import EnsemblePosterior
from cf_npe.networks.posterior import GaussianMixin, gaussian_nll_per_dim
from cf_npe.problems.base import EncoderConfig, NSFConfig, Problem
from cf_npe.training.loop import (
    BaseTrainResult,
    LossFn,
    TrainingConfig,
    _build_fold_indices,
    _random_subsample_folds,
    holdout_split,
    members_result,
    model_loss,
)
from cf_npe.training.parallel_seed import SeedData, simulate_seeds, train_seeds

_SUBSAMPLE_VAL_FRACTION = 0.1


def _beta_nll_loss(model: nn.Module, x: Tensor, z: Tensor) -> Tensor:
    """Beta-NLL loss (beta=1; Seitzer et al. 2022): each dimension's NLL is weighted by
    its detached predicted variance."""
    mean, log_std = cast(GaussianMixin, model).forward(x)
    weights = (2 * log_std).exp().detach()
    return (weights * gaussian_nll_per_dim(z, mean, log_std)).sum(dim=-1)


def _stop_gradient_nll_loss(model: nn.Module, x: Tensor, z: Tensor) -> Tensor:
    """Stop-gradient loss (Stirn et al. 2023): MSE trains the mean, and NLL with the mean
    detached trains log_std, so the variance cannot suppress the mean gradient."""
    mean, log_std = cast(GaussianMixin, model).forward(x)
    mse_per_dim = (z - mean) ** 2
    nll_per_dim = gaussian_nll_per_dim(z, mean.detach(), log_std)
    return (mse_per_dim + nll_per_dim).sum(dim=-1)


_LOSS_FNS: dict[str, LossFn] = {
    "standard": model_loss,
    "beta_nll": _beta_nll_loss,
    "stop_gradient": _stop_gradient_nll_loss,
}


def train_posteriors(
    problem: Problem,
    data: list[SeedData],
    *,
    training: TrainingConfig,
    encoder_cfg: EncoderConfig,
    posterior_type: str,
    flow_cfg: NSFConfig | None,
    loss: str,
    parallel: bool,
) -> list[BaseTrainResult]:
    """Train one posterior per split of each seed; several per seed form a uniform
    mixture."""
    device = data[0].x.device
    results = train_seeds(
        data,
        lambda _s, _k: cast(
            nn.Module, problem.create_posterior(encoder_cfg, posterior_type, flow_cfg)
        ).to(device),
        training=training,
        loss_fn=_LOSS_FNS[loss],
        parallel=parallel,
    )
    return [
        members_result(
            r.members[0] if len(r.members) == 1 else EnsemblePosterior(r.members), r.val_losses
        )
        for r in results
    ]


def train_std_npe(
    problem: Problem,
    n_train: int,
    seeds: list[int],
    training: TrainingConfig,
    encoder_cfg: EncoderConfig,
    posterior_type: str,
    flow_cfg: NSFConfig | None,
    loss: str,
    parallel_ensemble: bool,
) -> list[BaseTrainResult]:
    """Std-NPE with validation-based early stopping, one posterior per seed.

    ``loss`` is "standard" (NLL), "beta_nll" or "stop_gradient".
    """
    data = simulate_seeds(problem, n_train, seeds, holdout_split)
    return train_posteriors(
        problem,
        data,
        training=training,
        encoder_cfg=encoder_cfg,
        posterior_type=posterior_type,
        flow_cfg=flow_cfg,
        loss=loss,
        parallel=parallel_ensemble,
    )


def train_std_npe_mixture(
    problem: Problem,
    n_train: int,
    seeds: list[int],
    training: TrainingConfig,
    encoder_cfg: EncoderConfig,
    n_components: int,
    posterior_type: str,
    parallel_ensemble: bool,
    split_mode: str,
    flow_cfg: NSFConfig | None,
) -> list[BaseTrainResult]:
    """Std-NPE-mixture: a uniform mixture of K' independently trained Std-NPE posteriors.

    ``split_mode`` sets each component's training data: ``shared`` (one split for all;
    components differ only by initialization), ``partition`` (component k trains on the
    other folds and early-stops on fold k), or ``subsample`` (its own random holdout).
    """
    member_splits = {
        "shared": lambda n: holdout_split(n) * n_components,
        "partition": lambda n: _build_fold_indices(n, n_components),
        "subsample": lambda n: _random_subsample_folds(n, n_components, _SUBSAMPLE_VAL_FRACTION),
    }[split_mode]
    return train_posteriors(
        problem,
        simulate_seeds(problem, n_train, seeds, member_splits),
        training=training,
        encoder_cfg=encoder_cfg,
        posterior_type=posterior_type,
        flow_cfg=flow_cfg,
        loss="standard",
        parallel=parallel_ensemble,
    )
