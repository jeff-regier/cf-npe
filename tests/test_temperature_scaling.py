"""Temperature-scaled NPE baseline (Guo et al., 2017): one scalar T rescales sigma,
fit on validation data to minimize Gaussian NLL."""

import math
from typing import cast

import torch
from conftest import training_config
from torch import Tensor, nn

from cf_npe.methods import temperature_scaling
from cf_npe.methods.temperature_scaling import TemperatureScaledModel, _fit_scalar_temperature
from cf_npe.networks.posterior import GaussianMixin, GaussianPosterior
from cf_npe.problems.base import EncoderConfig
from cf_npe.problems.simple import Gaussian
from cf_npe.training.loop import validation_split
from cf_npe.training.parallel_seed import simulate_seeds


class ConstantGaussian(GaussianMixin, nn.Module):
    """Constant-prediction Gaussian posterior: mean=mu, std=sigma per dim."""

    def __init__(self, mu: float, sigma: float, dim_z: int):
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        self.dim_z = dim_z
        self.encoder = nn.Identity()
        self._p = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        shape = (x.shape[0], self.dim_z)
        return (
            torch.full(shape, self.mu, device=x.device),
            torch.full(shape, math.log(self.sigma), device=x.device),
        )


def test_fit_scalar_temperature_recovers_sqrt_variance_ratio():
    """z ~ N(mu, 3^2), base sigma=2, d>1 -> T = sqrt(E[(z-mu)^2]/sigma^2) = 1.5."""
    torch.manual_seed(0)
    mu, base_sigma, true_sigma, dim_z = 1.0, 2.0, 3.0, 4
    n = 200_000
    model = cast(GaussianPosterior, ConstantGaussian(mu, base_sigma, dim_z).cuda())
    x_val = torch.zeros(n, 3).cuda()
    z_val = (mu + true_sigma * torch.randn(n, dim_z)).cuda()

    log_temp = _fit_scalar_temperature(model, x_val, z_val, batch_size=8192)
    temperature = torch.exp(log_temp)

    assert torch.allclose(temperature, torch.tensor(1.5).cuda(), atol=2e-3)

    scaled = TemperatureScaledModel(model, log_temp)
    std = scaled.get_std(x_val[:1])
    assert torch.allclose(std, torch.full_like(std, 3.0), atol=4e-3)


def test_log_prob_matches_normal_with_scaled_std():
    """log_prob(x,z) == Normal(mu, sigma*T).log_prob(z).sum(-1)."""
    torch.manual_seed(0)
    mu, base_sigma, dim_z = 1.0, 2.0, 5
    log_temp = torch.log(torch.tensor(1.5)).cuda()
    model = cast(GaussianPosterior, ConstantGaussian(mu, base_sigma, dim_z).cuda())
    scaled = TemperatureScaledModel(model, log_temp)

    x = torch.zeros(7, 3).cuda()
    z = (mu + base_sigma * torch.randn(7, dim_z)).cuda()

    lp = scaled.log_prob(x, z)

    scaled_std = base_sigma * 1.5
    ref = (
        torch.distributions.Normal(
            torch.full_like(z, mu),
            torch.full_like(z, scaled_std),
        )
        .log_prob(z)
        .sum(dim=-1)
    )
    assert torch.allclose(lp, ref, atol=1e-5)


def test_samples_spread_scales_by_temperature():
    """Samples have the base mean and the base std times T."""
    torch.manual_seed(0)
    base = ConstantGaussian(mu=0.5, sigma=2.0, dim_z=3)
    model = TemperatureScaledModel(
        cast(GaussianPosterior, base), torch.tensor(math.log(1.5), device="cuda")
    )
    samples = model.sample(torch.zeros(4, 1, device="cuda"), 20000)
    torch.testing.assert_close(
        samples.mean(dim=1), torch.full((4, 3), 0.5, device="cuda"), atol=0.05, rtol=0
    )
    torch.testing.assert_close(
        samples.std(dim=1), torch.full((4, 3), 3.0, device="cuda"), atol=0.05, rtol=0
    )


def test_temperature_is_fit_on_the_validation_split(monkeypatch):
    """T is fit on each seed's early-stopping validation split, not on fresh simulations."""
    calls = []

    class CountingGaussian(Gaussian):
        def sample_joint(self, n):
            calls.append(n)
            return super().sample_joint(n)

    fitted = []

    def record(model, x_val, z_val, batch_size):
        fitted.append(x_val)
        return torch.zeros((), device=x_val.device)

    monkeypatch.setattr(temperature_scaling, "_fit_scalar_temperature", record)
    problem = CountingGaussian(dim=2, noise_std=0.5)
    temperature_scaling.train_temperature_scaled(
        problem,
        50,
        [3, 4],
        training_config(n_epochs=1, batch_size=16, lr=1e-3),
        EncoderConfig(hidden_dim=8, n_layers=3, dropout=0.0),
        True,
    )
    assert calls == [50, 50]
    data = simulate_seeds(
        Gaussian(dim=2, noise_std=0.5), 50, [3, 4], lambda n: [validation_split(n, 0.2)]
    )
    for x_val, d in zip(fitted, data, strict=True):
        ((_, val_idx),) = d.splits
        torch.testing.assert_close(x_val, d.x[val_idx])
