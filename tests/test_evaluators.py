"""Each evaluator returns its documented metric key."""

from typing import cast

import pytest
import torch
from conftest import trained_model

from cf_npe.evaluators import (
    C2STEvaluator,
    CompositeEvaluator,
    LogLikelihoodEvaluator,
    SBCEvaluator,
    TARPEvaluator,
    VarianceCalibrationEvaluator,
)
from cf_npe.networks.posterior import PosteriorModel
from cf_npe.problems.simple import Gaussian


@pytest.fixture(scope="module")
def model_and_data():
    torch.manual_seed(42)
    problem = Gaussian(dim=2, noise_std=0.5)
    model = trained_model(problem)
    x_test, z_test = problem.sample_joint(50)
    return model, problem, x_test, z_test


class _ScaledPosteriorModel:
    """Samples N(posterior_mean + mean_shift, (scale * posterior_std)^2)."""

    def __init__(self, problem: Gaussian, scale: float = 1.0, mean_shift: float = 0.0):
        self.problem = problem
        self.scale = scale
        self.mean_shift = mean_shift

    def eval(self):
        return self

    def sample(self, x: torch.Tensor, n_samples: int) -> torch.Tensor:
        mean = self.problem.posterior_mean(x) + self.mean_shift
        std = self.scale * self.problem.posterior_std()
        eps = torch.randn(x.shape[0], n_samples, x.shape[1], device=x.device)
        return mean.unsqueeze(1) + std * eps


@pytest.fixture(scope="module")
def joint_draws():
    torch.manual_seed(7)
    problem = Gaussian(dim=3, noise_std=0.5)
    x_test, z_test = problem.sample_joint(500)
    return problem, x_test, z_test


def test_sbc_uniform_for_exact_posterior(joint_draws):
    problem, x_test, z_test = joint_draws
    torch.manual_seed(0)
    exact = _ScaledPosteriorModel(problem)
    metrics = CompositeEvaluator([], [SBCEvaluator(n_samples=128)]).compute(
        cast(PosteriorModel, exact), problem, x_test, z_test, batch_size=100
    )
    assert metrics["sbc_ks"] < 0.05


def test_sbc_detects_miscalibration(joint_draws):
    problem, x_test, z_test = joint_draws
    torch.manual_seed(0)
    collapsed = CompositeEvaluator([], [SBCEvaluator(n_samples=128)]).compute(
        cast(PosteriorModel, _ScaledPosteriorModel(problem, scale=0.5)),
        problem,
        x_test,
        z_test,
        batch_size=100,
    )
    inflated = CompositeEvaluator([], [SBCEvaluator(n_samples=128)]).compute(
        cast(PosteriorModel, _ScaledPosteriorModel(problem, scale=2.0)),
        problem,
        x_test,
        z_test,
        batch_size=100,
    )
    assert collapsed["sbc_ks"] > 0.1
    assert inflated["sbc_ks"] > 0.1


def test_tarp_uniform_for_exact_posterior(joint_draws):
    problem, x_test, z_test = joint_draws
    torch.manual_seed(0)
    exact = _ScaledPosteriorModel(problem)
    metrics = CompositeEvaluator([], [TARPEvaluator(n_samples=128)]).compute(
        cast(PosteriorModel, exact), problem, x_test, z_test, batch_size=100
    )
    assert metrics["tarp_ks"] < 0.05


def test_tarp_detects_collapse(joint_draws):
    problem, x_test, z_test = joint_draws
    torch.manual_seed(0)
    collapsed = _ScaledPosteriorModel(problem, scale=0.3)
    metrics = CompositeEvaluator([], [TARPEvaluator(n_samples=128)]).compute(
        cast(PosteriorModel, collapsed), problem, x_test, z_test, batch_size=100
    )
    assert metrics["tarp_ks"] > 0.1


def test_c2st_exact_posterior_indistinguishable(joint_draws):
    problem, x_test, z_test = joint_draws
    torch.manual_seed(0)
    exact = _ScaledPosteriorModel(problem)
    metrics = C2STEvaluator(n_pairs=4000, n_epochs=30).compute(
        cast(PosteriorModel, exact), problem, x_test, z_test, batch_size=100
    )
    assert metrics["c2st_accuracy"] == pytest.approx(0.5, abs=0.07)


def test_c2st_detects_mean_shift(joint_draws):
    problem, x_test, z_test = joint_draws
    torch.manual_seed(0)
    shifted = _ScaledPosteriorModel(problem, mean_shift=1.0)
    metrics = C2STEvaluator(n_pairs=4000, n_epochs=30).compute(
        cast(PosteriorModel, shifted), problem, x_test, z_test, batch_size=100
    )
    assert metrics["c2st_accuracy"] > 0.65


def test_composite_runs_calibration_suite(model_and_data):
    """The wired-in composite of LL + calibration evaluators unions all keys."""
    model, problem, x_test, z_test = model_and_data
    torch.manual_seed(0)
    evaluator = CompositeEvaluator(
        [LogLikelihoodEvaluator(), C2STEvaluator(n_pairs=1000, n_epochs=5)],
        [VarianceCalibrationEvaluator(n_samples=64), SBCEvaluator(n_samples=64)],
    )
    metrics = evaluator.compute(model, problem, x_test, z_test, batch_size=32)
    for key in ["log_likelihood", "pull_variance", "sbc_ks", "c2st_accuracy"]:
        assert key in metrics


def test_pull_variance_directions(joint_draws):
    problem, x_test, z_test = joint_draws
    torch.manual_seed(0)
    results = {}
    for name, scale in [("exact", 1.0), ("collapsed", 0.5), ("inflated", 2.0)]:
        metrics = CompositeEvaluator([], [VarianceCalibrationEvaluator(n_samples=256)]).compute(
            cast(PosteriorModel, _ScaledPosteriorModel(problem, scale=scale)),
            problem,
            x_test,
            z_test,
            batch_size=100,
        )
        results[name] = metrics["pull_variance"]
    assert results["exact"] == pytest.approx(1.0, abs=0.08)
    assert results["collapsed"] > 2.0
    assert results["inflated"] < 0.5


def test_pull_variance_unbiased_with_few_draws(joint_draws):
    """With S=16 draws the uncorrected ratio's mean for an exact posterior is 1.23."""
    problem, x_test, z_test = joint_draws
    torch.manual_seed(0)
    metrics = CompositeEvaluator([], [VarianceCalibrationEvaluator(n_samples=16)]).compute(
        cast(PosteriorModel, _ScaledPosteriorModel(problem)), problem, x_test, z_test, 100
    )
    assert metrics["pull_variance"] == pytest.approx(1.0, abs=0.1)


class _PrefixStableModel:
    """Draw j of test point i is a fixed function of (i, j); counts sample calls."""

    def __init__(self, dim: int):
        self.dim = dim
        self.calls: list[int] = []

    def eval(self):
        return self

    def sample(self, x: torch.Tensor, n_samples: int) -> torch.Tensor:
        self.calls.append(n_samples)
        j = torch.arange(n_samples, device=x.device, dtype=x.dtype)[None, :, None]
        d = torch.arange(self.dim, device=x.device, dtype=x.dtype)[None, None, :]
        return 2.0 * torch.sin(1.3 * j + d + x[:, :1, None])


def test_composite_shares_one_draw_per_chunk(joint_draws):
    """The composite samples once per chunk and matches each standalone evaluator."""
    problem, x_test, z_test = joint_draws
    evaluators = [
        VarianceCalibrationEvaluator(n_samples=64),
        SBCEvaluator(n_samples=32),
        TARPEvaluator(n_samples=16),
    ]
    shared = _PrefixStableModel(problem.dim)
    torch.manual_seed(0)
    composite = CompositeEvaluator([], list(evaluators)).compute(
        cast(PosteriorModel, shared), problem, x_test, z_test, batch_size=100
    )
    assert shared.calls == [64] * 5

    standalone = {}
    for evaluator in evaluators:
        torch.manual_seed(0)
        standalone.update(
            CompositeEvaluator([], [evaluator]).compute(
                cast(PosteriorModel, _PrefixStableModel(problem.dim)),
                problem,
                x_test,
                z_test,
                batch_size=100,
            )
        )
    assert composite == standalone
