"""Neural-network building blocks for cf_npe two-stage models."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Protocol, cast

import torch
from torch import Tensor, nn

from cf_npe.networks.posterior import (
    PosteriorModel,
    SampleMixin,
    gaussian_nll_per_dim,
    gaussian_rsample,
    mlp_layers,
)

_LOG_SIGMA_MIN = -5.0
_LOG_SIGMA_MAX = 5.0


def _clamp_log_sigma(log_sigma: Tensor) -> Tensor:
    return log_sigma.clamp(_LOG_SIGMA_MIN, _LOG_SIGMA_MAX)


def pack_xmu(x: Tensor, mu: Tensor) -> Tensor:
    """Pack ``(x, mu)`` as ``cat([x, mu], -1)`` so Stage 2 fits the two-tensor trainer."""
    return torch.cat([x, mu], dim=-1)


def unpack_xmu(packed: Tensor, dim_x_raw: int) -> tuple[Tensor, Tensor]:
    """Inverse of ``pack_xmu``."""
    return packed[..., :dim_x_raw], packed[..., dim_x_raw:]


def mixture_log_prob(member_log_probs: Tensor) -> Tensor:
    """log of the uniform mixture density from (K, batch) member log densities."""
    return torch.logsumexp(member_log_probs, dim=0) - math.log(member_log_probs.shape[0])


def mixture_rsample(
    k: int, n_rows: int, n_samples: int, device: torch.device, draw: Callable[[int], Tensor]
) -> Tensor:
    """(batch, n_samples, dim_z) from a uniform mixture of ``k`` components.

    ``draw(n)`` returns (K, batch, n, dim_z), n draws per row from every component. Each slot
    draws its component first; every component then samples only as many draws per row as
    the busiest (row, component) pair needs, and slot (b, s) takes the next unused draw of its
    component in row b. A single component is sampled directly.
    """
    if k == 1:
        return draw(n_samples)[0]
    comp = torch.randint(k, (n_rows, n_samples), device=device)
    onehot = nn.functional.one_hot(comp, k)  # (batch, n_samples, K)
    rank = (onehot.cumsum(dim=1) - 1).gather(2, comp[..., None]).squeeze(2)
    n_draws = int(onehot.sum(dim=1).max())
    rows = torch.arange(n_rows, device=device)[:, None]
    return draw(n_draws)[comp, rows, rank]


class Stage2Head(Protocol):
    """q(z | x, mu) given the Stage-1 mean mu; ``rsample`` returns (batch, n_samples, dim_z)."""

    def log_prob(self, x: Tensor, mu: Tensor, z: Tensor) -> Tensor: ...

    def rsample(self, x: Tensor, mu: Tensor, n_samples: int) -> Tensor: ...


class MeanRegressor(nn.Module):
    """MSE-trained Stage-1 mean regressor; ``mean_head`` comes from ``Problem.create_mean_head``."""

    def __init__(self, encoder: nn.Module, mean_head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.mean_head = mean_head

    def get_mean(self, x: Tensor) -> Tensor:
        return self.mean_head(self.encoder(x))

    def training_loss(self, x: Tensor, z: Tensor) -> Tensor:
        return (self.get_mean(x) - z).pow(2).mean(dim=-1)


class OffsetNetwork(nn.Module):
    """SiLU MLP mapping raw ``cat(x, mu_hat)`` (width ``dim_in = dim_x + dim_mu``) to log_sigma."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.net = nn.Sequential(
            *mlp_layers(dim_in, hidden_dim, n_layers, nn.SiLU, dropout),
            nn.Linear(hidden_dim, dim_out),
        )

    def forward(self, x: Tensor, mu: Tensor) -> Tensor:
        return self.net(torch.cat([x, mu], dim=-1))


class GaussianHead(nn.Module):
    """Stage-2 Gaussian q(z | x, mu) = N(mu, sigma^2), log_sigma = ``offset(x, mu)``."""

    def __init__(self, offset: nn.Module):
        super().__init__()
        self.offset = offset

    def _log_sigma(self, x: Tensor, mu: Tensor) -> Tensor:
        return _clamp_log_sigma(self.offset(x, mu))

    def log_prob(self, x: Tensor, mu: Tensor, z: Tensor) -> Tensor:
        return -gaussian_nll_per_dim(z, mu, self._log_sigma(x, mu)).sum(dim=-1)

    def rsample(self, x: Tensor, mu: Tensor, n_samples: int) -> Tensor:
        return gaussian_rsample(mu, self._log_sigma(x, mu).exp(), n_samples)


class FlowHead(nn.Module):
    """Stage-2 conditional flow q(z | pack_xmu(x, mu))."""

    def __init__(self, flow: nn.Module):
        super().__init__()
        self.flow = flow

    def log_prob(self, x: Tensor, mu: Tensor, z: Tensor) -> Tensor:
        return cast(PosteriorModel, self.flow).log_prob(pack_xmu(x, mu), z)

    def rsample(self, x: Tensor, mu: Tensor, n_samples: int) -> Tensor:
        return cast(PosteriorModel, self.flow).rsample(pack_xmu(x, mu), n_samples)


class ResidualFlowHead(nn.Module):
    """Stage-2 flow on the scaled residual r = (z - mu) / residual_scale, conditioned on
    ``pack_xmu(x, mu)``; ``log_prob`` adds the change-of-variables term."""

    residual_scale: Tensor

    def __init__(self, flow: nn.Module, residual_scale: Tensor):
        super().__init__()
        self.flow = flow
        self.register_buffer("residual_scale", residual_scale.detach().clone())

    def log_prob(self, x: Tensor, mu: Tensor, z: Tensor) -> Tensor:
        r = (z - mu) / self.residual_scale
        log_q = cast(PosteriorModel, self.flow).log_prob(pack_xmu(x, mu), r)
        return log_q - self.residual_scale.log().sum()

    def rsample(self, x: Tensor, mu: Tensor, n_samples: int) -> Tensor:
        r = cast(PosteriorModel, self.flow).rsample(pack_xmu(x, mu), n_samples)
        return mu.unsqueeze(1) + self.residual_scale * r


class TwoStageModel(SampleMixin, nn.Module):
    """CF-NPE posterior: the Stage-1 mean mu_hat(x), computed once, and a uniform mixture of
    the K >= 1 Stage-2 heads conditioned on it."""

    def __init__(self, mean_net: EnsembleMeanNet, heads: list[nn.Module]):
        super().__init__()
        self.mean_net = mean_net
        self.heads = nn.ModuleList(heads)

    def get_mean(self, x: Tensor) -> Tensor:
        return self.mean_net.get_mean(x)

    def log_prob(self, x: Tensor, z: Tensor) -> Tensor:
        mu = self.get_mean(x)
        return mixture_log_prob(
            torch.stack([cast(Stage2Head, h).log_prob(x, mu, z) for h in self.heads])
        )

    def rsample(self, x: Tensor, n_samples: int) -> Tensor:
        mu = self.get_mean(x)

        def draw(n: int) -> Tensor:
            return torch.stack([cast(Stage2Head, h).rsample(x, mu, n) for h in self.heads])

        return mixture_rsample(len(self.heads), x.shape[0], n_samples, x.device, draw)


class EnsembleMeanNet(nn.Module):
    """Uniform average of mean networks' predictions.

    Stage 1 builds one per fold from that fold's M replicates; fold 0's is the test-time mean,
    never an average across folds.
    """

    def __init__(self, nets: list[MeanRegressor]):
        super().__init__()
        self.models = nn.ModuleList(nets)

    def get_mean(self, x: Tensor) -> Tensor:
        means = [cast(MeanRegressor, m).get_mean(x) for m in self.models]
        return torch.stack(means).mean(dim=0)


class EnsemblePosterior(SampleMixin, nn.Module):
    """Uniform mixture q(z|x) = (1/K) sum_k q_k(z|x) of K posterior networks."""

    def __init__(self, members: list[nn.Module]):
        super().__init__()
        self.members = nn.ModuleList(members)

    def log_prob(self, x: Tensor, z: Tensor) -> Tensor:
        return mixture_log_prob(
            torch.stack([cast(PosteriorModel, m).log_prob(x, z) for m in self.members])
        )

    def get_mean(self, x: Tensor) -> Tensor:
        means = [cast(PosteriorModel, m).get_mean(x) for m in self.members]
        return torch.stack(means).mean(dim=0)

    def rsample(self, x: Tensor, n_samples: int) -> Tensor:
        def draw(n: int) -> Tensor:
            return torch.stack([cast(PosteriorModel, m).rsample(x, n) for m in self.members])

        return mixture_rsample(len(self.members), x.shape[0], n_samples, x.device, draw)


class GaussianOffsetTrainingWrapper(nn.Module):
    """Trains a Gaussian offset on packed ``(x, mu)``: NLL of raw z under N(mu_hat, sigma^2)."""

    def __init__(self, inner: nn.Module, dim_x_raw: int):
        super().__init__()
        self.inner = inner
        self.dim_x_raw = dim_x_raw

    def training_loss(self, packed: Tensor, z: Tensor) -> Tensor:
        x, mu = unpack_xmu(packed, self.dim_x_raw)
        log_sigma = _clamp_log_sigma(self.inner(x, mu))
        return gaussian_nll_per_dim(z, mu, log_sigma).sum(dim=-1)


class ResidualTrainingWrapper(nn.Module):
    """Trains a conditional flow on the residual ``r = (z - mu) / residual_scale``.

    The loss omits the Jacobian term, which is constant in the weights.
    """

    residual_scale: Tensor

    def __init__(self, inner: nn.Module, dim_x_raw: int, residual_scale: Tensor):
        super().__init__()
        self.inner = inner
        self.dim_x_raw = dim_x_raw
        self.register_buffer("residual_scale", residual_scale.detach().clone())

    def training_loss(self, packed: Tensor, z: Tensor) -> Tensor:
        _, mu = unpack_xmu(packed, self.dim_x_raw)
        r = (z - mu) / self.residual_scale
        return cast(PosteriorModel, self.inner).training_loss(packed, r)
