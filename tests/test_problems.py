import math
from types import SimpleNamespace
from typing import cast

import torch

from cf_npe.problems.simple import Funnel, Gaussian, Mixture
from cf_npe.problems.strong_lensing import StrongLensing

ATOL = 1e-5


def test_gaussian_posterior_mean():
    for sigma in (0.25, 0.5, 1.0, 2.0):
        problem = Gaussian(dim=3, noise_std=sigma)
        x = torch.randn(10, 3, device="cuda")
        mean = problem.posterior_mean(x)
        assert torch.allclose(mean, x / (1 + sigma**2), atol=ATOL)
        assert math.isclose(problem.posterior_std(), sigma / math.sqrt(1 + sigma**2))


def test_mixture_log_prob():
    problem = Mixture(dim=5, noise_std=0.3, mode_sep=3.0)
    x, z = problem.sample_joint(8)
    Az = z @ problem.A.T
    log_half = math.log(0.5)
    ref_pos = log_half + torch.distributions.Normal(Az + problem.s, 0.3).log_prob(x).sum(dim=-1)
    ref_neg = log_half + torch.distributions.Normal(Az - problem.s, 0.3).log_prob(x).sum(dim=-1)
    ref = torch.logaddexp(ref_pos, ref_neg)
    assert torch.allclose(problem.log_likelihood(x, z), ref, atol=ATOL)
    assert math.isclose(problem.s.norm().item(), 3.0, rel_tol=1e-5)


def test_mixture_forward_model_mean_matches_Az():
    problem = Mixture(dim=5, noise_std=0.3, mode_sep=3.0)
    z = problem.sample_prior(200_000)
    emp = problem.sample_likelihood(z).mean(dim=0)
    assert torch.allclose((z @ problem.A.T).mean(dim=0), emp, atol=2e-2)


def test_mixture_posterior_log_prob_matches_linear_gaussian_limit():
    """With ``mode_sep=0`` the Mixture is linear-Gaussian, with Bayes test LL
    ``-0.5 (d_z (1 + log 2pi) + log|Sigma|)``, ``Sigma = (I + A^T A/sigma^2)^{-1}``."""
    sigma, d_z = 0.3, 5
    problem = Mixture(dim=d_z, noise_std=sigma, mode_sep=0.0)
    eye = torch.eye(d_z, device=problem.A.device)
    sigma_post = torch.linalg.inv(eye + problem.A.T @ problem.A / sigma**2)
    logdet = torch.logdet(sigma_post)
    closed_form = -0.5 * (d_z * (1 + math.log(2 * math.pi)) + logdet.item())
    x, z = problem.sample_joint(400_000)
    mc = float(problem.posterior_log_prob(x, z).mean())
    assert math.isclose(mc, closed_form, abs_tol=0.05)


def test_funnel_sample_mean_matches_forward_model():
    problem = Funnel(dim=3, scale=1.0, noise_std=0.1)
    z = problem.sample_prior(4)
    z = z.repeat(100_000, 1)
    emp = problem.sample_likelihood(z).mean(dim=0)
    mean = z.clone()
    mean[:, 1:] = z[:, 1:] * torch.exp(z[:, 0:1] / 2)
    assert torch.allclose(mean.mean(dim=0), emp, atol=2e-2)


def test_mu_concat_offset_encoder_lifted_channels():
    """The lifted mu channels are appended to the image features on the grid."""
    from cf_npe.problems.strong_lensing import _ImageEncoder, _MuConcatOffsetEncoder

    class _StubEncoder(_ImageEncoder):
        def __init__(self, c: int, n: int):
            super().__init__()
            self.out_channels = c
            self.n = n

        def forward(self, x_raw):
            return torch.zeros(x_raw.shape[0], self.out_channels, self.n, self.n)

    n, dim_x, dim_z = 4, 64, 16
    cond = torch.randn(3, dim_x + dim_z)

    lifted = _MuConcatOffsetEncoder(_StubEncoder(6, n), dim_x, n, mu_proj_channels=8)
    assert lifted.out_channels == 14
    assert lifted(cond).shape == (3, 14, n, n)


def test_strong_lensing_train_and_test_pools_are_complementary():
    """With one split seed, the train and test pools are disjoint and cover the file."""
    n_total = 1000

    def pool(is_test):
        split = SimpleNamespace(split_seed=0, test_fraction=0.1, is_test=is_test)
        return StrongLensing._pool_indices(cast(StrongLensing, split), n_total)

    test_idx, train_idx = pool(True), pool(False)
    assert test_idx.numel() == 100
    assert torch.equal(torch.cat([test_idx, train_idx]).sort().values, torch.arange(n_total))
