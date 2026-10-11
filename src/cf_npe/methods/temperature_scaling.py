"""Temperature-scaled NPE baseline (Guo et al., 2017)."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from cf_npe.methods.std_npe import train_posteriors
from cf_npe.networks.posterior import GaussianMixin, GaussianPosterior, PosteriorModel
from cf_npe.problems.base import EncoderConfig, Problem
from cf_npe.training.loop import (
    BaseTrainResult,
    TrainingConfig,
    holdout_split,
)
from cf_npe.training.parallel_seed import simulate_seeds


def _fit_scalar_temperature(
    model: GaussianPosterior,
    x_val: Tensor,
    z_val: Tensor,
    batch_size: int,
) -> Tensor:
    """Return log T for the Gaussian-NLL-optimal temperature, T^2 = mean of
    (z - mu)^2 / sigma^2 over all (observation, dimension) pairs."""
    ratio_sum = torch.zeros((), dtype=torch.float64)
    ratio_count = 0
    with torch.no_grad():
        for x_c, z_c in zip(
            x_val.split(batch_size),
            z_val.split(batch_size),
            strict=True,
        ):
            mu, log_std = model(x_c)
            ratio = (z_c - mu).pow(2) / torch.exp(2 * log_std)
            ratio_sum = ratio_sum + ratio.sum().double().cpu()
            ratio_count += ratio.numel()
    t_squared = (ratio_sum / ratio_count).clamp(min=1e-12)
    return 0.5 * torch.log(t_squared).to(dtype=torch.float32)


def _with_temperature(
    result: BaseTrainResult, x_val: Tensor, z_val: Tensor, batch_size: int
) -> BaseTrainResult:
    """Fit a scalar temperature on the validation split and wrap the model."""
    model = cast(GaussianPosterior, result.model)
    log_temp = _fit_scalar_temperature(model, x_val, z_val, batch_size).to(x_val.device)
    scaled = TemperatureScaledModel(model, log_temp)
    return BaseTrainResult(cast(PosteriorModel, scaled), result.best_epoch, result.best_val_loss)


def train_temperature_scaled(
    problem: Problem,
    n_train: int,
    seeds: list[int],
    training: TrainingConfig,
    encoder_cfg: EncoderConfig,
    parallel_ensemble: bool,
) -> list[BaseTrainResult]:
    """Train Std-NPE, then fit one temperature per seed on its validation split."""
    data = simulate_seeds(problem, n_train, seeds, holdout_split)
    trained = train_posteriors(
        problem,
        data,
        training=training,
        encoder_cfg=encoder_cfg,
        posterior_type="gaussian",
        flow_cfg=None,
        loss="standard",
        parallel=parallel_ensemble,
    )
    scaled = []
    for result, d in zip(trained, data, strict=True):
        ((_, val_idx),) = d.splits
        scaled.append(_with_temperature(result, d.x[val_idx], d.z[val_idx], training.batch_size))
    return scaled


class TemperatureScaledModel(GaussianMixin, nn.Module):
    """A GaussianPosterior whose sigma is scaled by one temperature in every dimension."""

    log_temperature: Tensor

    def __init__(self, base_model: GaussianPosterior, log_temperature: Tensor):
        super().__init__()
        self.base_model = base_model
        self.register_buffer("log_temperature", log_temperature)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        mean, log_std = self.base_model.forward(x)
        return mean, log_std + self.log_temperature
