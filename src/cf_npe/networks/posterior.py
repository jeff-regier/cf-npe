from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from typing import Any, Protocol, Self, cast

import torch
from torch import Tensor, nn
from torch.nn import Parameter
from zuko.transforms import MonotonicRQSTransform


def _spline_params_per_dim(n_bins: int) -> int:
    """3*K - 1 internal spline params per output dim (widths, heights, derivs)."""
    return 3 * n_bins - 1


def mlp_layers(
    dim_in: int,
    hidden_dim: int,
    n_layers: int,
    activation: type[nn.Module],
    dropout: float,
) -> list[nn.Module]:
    """``n_layers >= 1`` hidden layers R^dim_in -> R^hidden_dim, each Linear, activation, and
    Dropout when ``dropout > 0``."""
    layers: list[nn.Module] = []
    for dim_a in [dim_in] + [hidden_dim] * (n_layers - 1):
        layers += [nn.Linear(dim_a, hidden_dim), activation()]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
    return layers


def mlp_encoder(dim_in: int, hidden_dim: int, n_layers: int, dropout: float) -> nn.Sequential:
    """ReLU MLP encoder R^dim_in -> R^hidden_dim."""
    return nn.Sequential(*mlp_layers(dim_in, hidden_dim, n_layers, nn.ReLU, dropout))


def _gn(num_channels: int, num_groups: int) -> nn.GroupNorm:
    g = math.gcd(num_channels, num_groups)
    return nn.GroupNorm(num_groups=g, num_channels=num_channels)


def gaussian_nll_per_dim(z: Tensor, mean: Tensor, log_std: Tensor) -> Tensor:
    """-log N(z; mean, exp(log_std)^2), one value per coordinate."""
    return 0.5 * ((z - mean) ** 2 / torch.exp(2 * log_std) + 2 * log_std + math.log(2 * math.pi))


def gaussian_rsample(mean: Tensor, std: Tensor, n_samples: int) -> Tensor:
    """Reparameterized (batch, n_samples, dim_z) draws from N(mean, std^2)."""
    mean = mean.unsqueeze(1).expand(-1, n_samples, -1)
    std = std.unsqueeze(1).expand(-1, n_samples, -1)
    return mean + std * torch.randn_like(mean)


class PosteriorModel(Protocol):
    """An ``nn.Module`` that models q(z | x): ``log_prob(x, z)`` is summed over
    the dim_z coordinates; ``training_loss`` returns one loss per row; ``rsample`` returns (batch, n_samples, dim_z) with
    gradients, ``sample`` the same detached and squeezed when n_samples == 1."""

    def log_prob(self, x: Tensor, z: Tensor) -> Tensor: ...

    def training_loss(self, x: Tensor, z: Tensor) -> Tensor: ...

    def rsample(self, x: Tensor, n_samples: int) -> Tensor: ...

    def sample(self, x: Tensor, n_samples: int) -> Tensor: ...

    def get_mean(self, x: Tensor) -> Tensor: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> Any: ...

    def train(self, mode: bool = True) -> Self: ...

    def eval(self) -> Self: ...

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]: ...

    def to(self, device: str | torch.device) -> Self: ...


LOG_STD_MIN = -5


LOG_STD_MAX = 3


class RQSTransform(MonotonicRQSTransform):
    """zuko's RQS with a cancellation-free inverse.

    zuko's inverse solves a*t^2 + b*t + c = 0 with discriminant b^2 - 4ac, which cancels
    catastrophically when d0 / d1 is extreme and can round to zero or below, giving NaN.
    With u = (y - y0) / (y1 - y0) the discriminant is
    (y1 - y0)^2 * ((d0 (1 - u) - d1 u)^2 + 4 s^2 u (1 - u)), a sum of non-negative terms,
    and the (y1 - y0) factors cancel from the root t = 2c / (-b - sqrt(b^2 - 4ac)).
    """

    def _inverse(self, y: Tensor) -> Tensor:
        k = cast(torch.LongTensor, self.searchsorted(self.vertical, y) - 1)
        mask, x0, x1, y0, y1, d0, d1, s = self.bin(k)

        u = mask * (y - y0) / (y1 - y0)
        disc = (d0 * (1 - u) - d1 * u) ** 2 + 4 * s**2 * u * (1 - u)
        z = 2 * s * u / (d0 - u * (d0 + d1 - 2 * s) + disc.sqrt())

        x = x0 + z * (x1 - x0)

        return torch.where(mask, x, y)


class SampleMixin:
    """Derive sample() from rsample() with detach + squeeze."""

    def rsample(self, x: Tensor, n_samples: int) -> Tensor:
        raise NotImplementedError

    def sample(self, x: Tensor, n_samples: int) -> Tensor:
        samples = self.rsample(x, n_samples).detach()
        return samples.squeeze(1) if n_samples == 1 else samples


class GaussianMixin(SampleMixin):
    """Methods for posteriors whose forward() returns (mean, log_std)."""

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def log_prob(self, x: Tensor, z: Tensor) -> Tensor:
        mean, log_std = self.forward(x)
        return -gaussian_nll_per_dim(z, mean, log_std).sum(dim=-1)

    def rsample(self, x: Tensor, n_samples: int) -> Tensor:
        mean, log_std = self.forward(x)
        return gaussian_rsample(mean, torch.exp(log_std), n_samples)

    def get_mean(self, x: Tensor) -> Tensor:
        mean, _ = self.forward(x)
        return mean

    def get_std(self, x: Tensor) -> Tensor:
        _, log_std = self.forward(x)
        return torch.exp(log_std)

    def training_loss(self, x: Tensor, z: Tensor) -> Tensor:
        return -self.log_prob(x, z)


class SamplingMomentMixin(SampleMixin):
    """Derive get_mean as the average of 256 posterior samples."""

    def get_mean(self, x: Tensor) -> Tensor:
        return self.sample(x, 256).mean(dim=1)


class GaussianPosterior(GaussianMixin, nn.Module):
    """Diagonal Gaussian posterior q(z|x) = N(mu(x), diag(sigma^2(x)))."""

    def __init__(self, encoder: nn.Module, head_dim: int, dim_z: int):
        super().__init__()
        self.encoder = encoder
        self.mean_head = nn.Linear(head_dim, dim_z)
        self.log_std_head = nn.Linear(head_dim, dim_z)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.encoder(x)
        mean = self.mean_head(h)
        log_std = torch.clamp(self.log_std_head(h), min=LOG_STD_MIN, max=LOG_STD_MAX)
        return mean, log_std
