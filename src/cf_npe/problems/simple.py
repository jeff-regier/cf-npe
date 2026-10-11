"""Synthetic problems with closed-form simulators: Gaussian, two-mode Mixture, Funnel."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from cf_npe.problems.base import Problem


@dataclass
class Gaussian(Problem):
    """z ~ N(0, I), x|z ~ N(z, sigma^2 I)."""

    noise_std: float

    def sample_prior(self, n: int) -> Tensor:
        return torch.randn(n, self.dim, device="cuda")

    def sample_likelihood(self, z: Tensor) -> Tensor:
        return z + self.noise_std * torch.randn_like(z)

    def posterior_mean(self, x: Tensor) -> Tensor:
        shrinkage = 1 / (1 + self.noise_std**2)
        return x * shrinkage

    def posterior_std(self) -> float:
        return self.noise_std / math.sqrt(1 + self.noise_std**2)


@dataclass
class Mixture(Problem):
    """z ~ N(0, I), x | z ~ 1/2 N(Az + s, sigma^2 I) + 1/2 N(Az - s, sigma^2 I).

    d_z = dim and d_x = 2 dim. A is a fixed Gaussian matrix scaled by 1/sqrt(d_z); s is a
    fixed random direction with ||s|| = mode_sep. The posterior is bimodal.
    """

    noise_std: float
    mode_sep: float

    def __post_init__(self):
        device = "cuda"
        gen = torch.Generator(device=device).manual_seed(42)
        a_raw = torch.randn(self.dim_x, self.dim_z, device=device, generator=gen)
        self.A = a_raw / math.sqrt(self.dim_z)
        s_raw = torch.randn(self.dim_x, device=device, generator=gen)
        self.s = self.mode_sep * s_raw / s_raw.norm()
        self._log_lik_const = self.dim_x * math.log(self.noise_std * math.sqrt(2 * math.pi))
        self._log_prior_const = self.dim_z * 0.5 * math.log(2 * math.pi)
        self._inv_noise_var = 1.0 / (self.noise_std**2)
        self._log_2 = math.log(2)
        # Evidence covariance C = AAᵀ + σ²I in x-space, shared by both components.
        x_cov = self.A @ self.A.T + self.noise_std**2 * torch.eye(self.dim_x, device=device)
        self._x_cov_chol = torch.linalg.cholesky(x_cov)
        logdet = 2 * torch.log(torch.diagonal(self._x_cov_chol)).sum()
        self._x_evidence_const = 0.5 * (self.dim_x * math.log(2 * math.pi) + logdet)

    @property
    def dim_x(self) -> int:
        return 2 * self.dim

    def sample_prior(self, n: int) -> Tensor:
        return torch.randn(n, self.dim_z, device="cuda")

    def sample_likelihood(self, z: Tensor) -> Tensor:
        n = z.shape[0]
        device = z.device
        sign = (2 * torch.randint(0, 2, (n, 1), device=device) - 1).float()
        mean = z @ self.A.T + sign * self.s
        return mean + self.noise_std * torch.randn(n, self.dim_x, device=device)

    def log_likelihood(self, x: Tensor, z: Tensor) -> Tensor:
        Az = z @ self.A.T
        diff_pos = x - Az - self.s
        diff_neg = x - Az + self.s
        log_p_pos = -0.5 * (diff_pos**2).sum(dim=-1) * self._inv_noise_var
        log_p_neg = -0.5 * (diff_neg**2).sum(dim=-1) * self._inv_noise_var
        return torch.logaddexp(log_p_pos, log_p_neg) - self._log_2 - self._log_lik_const

    def log_prior(self, z: Tensor) -> Tensor:
        return -0.5 * (z**2).sum(dim=-1) - self._log_prior_const

    def _component_log_evidence(self, x: Tensor) -> Tensor:
        r"""$[\log N(x; s, C), \log N(x; -s, C)]$ stacked on the last dim."""

        def comp(mu: Tensor) -> Tensor:
            diff = x - mu
            sol = torch.cholesky_solve(diff.unsqueeze(-1), self._x_cov_chol).squeeze(-1)
            return -0.5 * (diff * sol).sum(dim=-1) - self._x_evidence_const

        return torch.stack([comp(self.s), comp(-self.s)], dim=-1)

    def marginal_log_prob(self, x: Tensor) -> Tensor:
        r"""$\log p(x)$ for $p(x) = \tfrac12 N(x; s, C) + \tfrac12 N(x; -s, C)$."""
        return torch.logsumexp(self._component_log_evidence(x), dim=-1) - self._log_2

    def posterior_log_prob(self, x: Tensor, z: Tensor) -> Tensor:
        r"""True posterior $\log p(z \mid x)$ via Bayes' rule."""
        return self.log_likelihood(x, z) + self.log_prior(z) - self.marginal_log_prob(x)


@dataclass
class Funnel(Problem):
    """Funnel: z_0 ~ N(0, scale^2), z_i ~ N(0, 1) for i >= 1, x_i = z_i exp(z_0 / 2) + noise for
    i >= 1, x_0 = z_0 + noise."""

    scale: float
    noise_std: float

    def sample_prior(self, n: int) -> Tensor:
        z = torch.randn(n, self.dim, device="cuda")
        z[:, 0] *= self.scale
        return z

    def sample_likelihood(self, z: Tensor) -> Tensor:
        log_scale = z[:, 0:1] / 2
        x = z.clone()
        x[:, 1:] = z[:, 1:] * torch.exp(log_scale)
        return x + self.noise_std * torch.randn_like(x)
