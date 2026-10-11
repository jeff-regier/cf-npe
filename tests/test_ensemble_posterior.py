"""EnsemblePosterior mixture density and sampling; TwoStageModel's Gaussian log_prob."""

import math
from typing import cast

import torch
from torch import Tensor, nn
from torch.distributions import Normal

from cf_npe.networks.models import EnsembleMeanNet, EnsemblePosterior, GaussianHead, TwoStageModel


class _ConstGaussian(nn.Module):
    """Closed-form diagonal Gaussian posterior with constant mean/std."""

    _mean: Tensor
    _std: Tensor

    def __init__(self, mean: Tensor, std: Tensor):
        super().__init__()
        self.register_buffer("_mean", mean)
        self.register_buffer("_std", std)

    def _dist(self, x: Tensor) -> Normal:
        batch = x.shape[0]
        loc = self._mean.expand(batch, -1)
        scale = self._std.expand(batch, -1)
        return Normal(loc, scale)

    def log_prob(self, x: Tensor, z: Tensor) -> Tensor:
        return self._dist(x).log_prob(z).sum(dim=-1)

    def get_mean(self, x: Tensor) -> Tensor:
        return self._mean.expand(x.shape[0], -1)

    def rsample(self, x: Tensor, n_samples: int) -> Tensor:
        return self._dist(x).rsample((n_samples,)).transpose(0, 1)


class _ConstMeanNet(nn.Module):
    """Stand-in for EnsembleMeanNet: constant get_mean."""

    _mean: Tensor

    def __init__(self, mean: Tensor):
        super().__init__()
        self.register_buffer("_mean", mean)

    def get_mean(self, x: Tensor) -> Tensor:
        return self._mean.expand(x.shape[0], -1)


class _ConstOffset(nn.Module):
    """Offset net emitting a constant log_sigma, ignoring (x, mu)."""

    _log_sigma: Tensor

    def __init__(self, log_sigma: Tensor):
        super().__init__()
        self.register_buffer("_log_sigma", log_sigma)

    def forward(self, x: Tensor, mu: Tensor) -> Tensor:
        return self._log_sigma.expand(x.shape[0], -1)


def test_ensemble_log_prob_is_log_mixture_density():
    """log_prob == log((1/K) sum_k q_k(z|x)), independent of the code path."""
    dim_z = 3
    m1 = _ConstGaussian(torch.tensor([0.0, 1.0, -2.0]).cuda(), torch.tensor([1.0, 0.5, 2.0]).cuda())
    m2 = _ConstGaussian(torch.tensor([1.0, -1.0, 0.5]).cuda(), torch.tensor([0.7, 1.3, 0.9]).cuda())
    ens = EnsemblePosterior([m1, m2]).cuda()

    x = torch.randn(8, 4).cuda()
    z = torch.randn(8, dim_z).cuda()

    got = ens.log_prob(x, z)

    p1 = m1.log_prob(x, z).exp()
    p2 = m2.log_prob(x, z).exp()
    expected = (0.5 * p1 + 0.5 * p2).log()

    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)
    # A mean of log-densities would equal 0.5*(lp1+lp2).
    mean_of_logs = 0.5 * (m1.log_prob(x, z) + m2.log_prob(x, z))
    assert (got - mean_of_logs).abs().max() > 1e-3


def test_ensemble_sample_mixture_balanced_and_attributable():
    """With well-separated members, each component is sampled with frequency 1/K."""
    torch.manual_seed(0)
    k = 3
    sep = 100.0
    centers = [torch.tensor([sep * i, -sep * i]).cuda() for i in range(k)]
    members = [_ConstGaussian(c, torch.tensor([0.01, 0.01]).cuda()) for c in centers]
    ens = EnsemblePosterior(members).cuda()

    x = torch.randn(1, 4).cuda()
    n = 30000
    samples = ens.rsample(x, n).squeeze(0)  # (n, dim_z)

    centers_t = torch.stack(centers)  # (K, dim_z)
    dists = (samples[:, None, :] - centers_t[None, :, :]).pow(2).sum(-1)  # (n, K)
    assigned = dists.argmin(dim=1)
    nearest_dist = dists.gather(1, assigned[:, None]).squeeze(1)
    assert nearest_dist.max() < 1.0

    counts = torch.bincount(assigned, minlength=k).float()
    freq = counts / n
    torch.testing.assert_close(freq, torch.full((k,), 1.0 / k).cuda(), atol=0.02, rtol=0)


def test_two_stage_log_prob_is_normal_at_mu_hat():
    """TwoStageModel.log_prob == Normal(mu_hat, exp(log_sigma)).log_prob(z).sum(-1)."""
    dim_z = 4
    mu_const = torch.tensor([0.5, -1.0, 2.0, 0.0]).cuda()
    log_sigma_const = torch.tensor([0.0, -0.5, 1.0, 0.25]).cuda()
    mean_net = cast(EnsembleMeanNet, _ConstMeanNet(mu_const))
    offset = _ConstOffset(log_sigma_const)
    model = TwoStageModel(mean_net, [GaussianHead(offset)]).cuda()

    x = torch.randn(7, 5).cuda()
    z = torch.randn(7, dim_z).cuda()

    got = model.log_prob(x, z)

    mu_hat = mu_const.expand(7, -1)
    std = log_sigma_const.exp().expand(7, -1)
    expected = Normal(mu_hat, std).log_prob(z).sum(dim=-1)

    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)
    assert got.shape == (7,)
    assert math.isfinite(float(got.sum()))
