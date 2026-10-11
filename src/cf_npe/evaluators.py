"""Evaluation metrics for posterior models."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Protocol

import torch
from torch import Tensor, nn

from cf_npe.networks.posterior import PosteriorModel
from cf_npe.problems.base import Problem
from cf_npe.training.loop import chunked_mean

_TARP_REFERENCE_DRAWS = 10
_C2ST_HIDDEN_DIM = 256
_C2ST_LR = 1e-3
_C2ST_BATCH_SIZE = 256


def _chunked_samples(
    model: PosteriorModel, x: Tensor, n_samples: int, batch_size: int
) -> Iterator[tuple[slice, Tensor]]:
    """Yield (slice, samples) pairs; samples has shape (chunk, n_samples, dim_z)."""
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            sl = slice(start, min(start + batch_size, x.shape[0]))
            yield sl, model.sample(x[sl], n_samples)


def _ks_uniform(u: Tensor) -> float:
    """Kolmogorov-Smirnov statistic of a 1-D sample against U(0, 1)."""
    u_sorted = u.flatten().sort().values
    n = u_sorted.shape[0]
    grid = torch.arange(1, n + 1, device=u.device, dtype=u_sorted.dtype) / n
    d_plus = (grid - u_sorted).max()
    d_minus = (u_sorted - grid + 1.0 / n).max()
    return torch.maximum(d_plus, d_minus).item()


def _randomized_pit(counts: Tensor, n_samples: int) -> Tensor:
    """Map counts in {0, ..., n_samples}, uniform under calibration, to values exactly
    U(0, 1) under calibration by spreading each count uniformly over its bin. The spread
    has its own fixed seed, so the metric is a function of the posterior draws alone."""
    gen = torch.Generator(counts.device).manual_seed(0)
    jitter = torch.rand(counts.shape, generator=gen, device=counts.device)
    return (counts.float() + jitter) / (n_samples + 1)


def _sample_metrics(
    evaluators: list[SampleEvaluator],
    model: PosteriorModel,
    problem: Problem,
    x_test: Tensor,
    z_test: Tensor,
    batch_size: int,
) -> dict[str, float]:
    """Draw max(n_samples) posterior samples per chunk once; each evaluator uses
    the first ``n_samples`` of them."""
    model.eval()
    n_samples = max(e.n_samples for e in evaluators)
    stats: list[list[Tensor]] = [[] for _ in evaluators]
    for sl, samples in _chunked_samples(model, x_test, n_samples, batch_size):
        for e, chunks in zip(evaluators, stats, strict=True):
            chunks.append(e.chunk_stats(problem, samples[:, : e.n_samples], z_test[sl]))
    metrics: dict[str, float] = {}
    for e, chunks in zip(evaluators, stats, strict=True):
        metrics.update(e.metrics(problem, x_test, z_test, torch.cat(chunks)))
    return metrics


class Evaluator(Protocol):
    def compute(
        self,
        model: PosteriorModel,
        problem: Problem,
        x_test: Tensor,
        z_test: Tensor,
        batch_size: int,
    ) -> dict[str, float]: ...


class MeanPredictionMSEEvaluator:
    """Point-prediction MSE between the model's ``get_mean`` and z_test."""

    def compute(
        self,
        model: PosteriorModel,
        problem: Problem,
        x_test: Tensor,
        z_test: Tensor,
        batch_size: int,
    ) -> dict[str, float]:
        del problem
        model.eval()
        mse = chunked_mean(
            lambda x_b, z_b: (model.get_mean(x_b) - z_b).pow(2).mean(dim=-1),
            x_test,
            z_test,
            batch_size=batch_size,
        )
        return {"mean_prediction_mse": mse}


class LogLikelihoodEvaluator:
    """Held-out log likelihood."""

    def compute(
        self,
        model: PosteriorModel,
        problem: Problem,
        x_test: Tensor,
        z_test: Tensor,
        batch_size: int,
    ) -> dict[str, float]:
        del problem
        model.eval()
        ll = chunked_mean(
            model.log_prob,
            x_test,
            z_test,
            batch_size=batch_size,
        )
        return {"log_likelihood": ll}


class SampleEvaluator(ABC):
    """A metric computed from ``n_samples`` posterior draws per test point.

    ``chunk_stats`` reduces one chunk of draws to per-test-point statistics (test
    points on dim 0); ``metrics`` turns their concatenation into the metric dict.
    """

    def __init__(self, n_samples: int):
        self.n_samples = n_samples

    @abstractmethod
    def chunk_stats(self, problem: Problem, samples: Tensor, z_b: Tensor) -> Tensor: ...

    @abstractmethod
    def metrics(
        self, problem: Problem, x_test: Tensor, z_test: Tensor, stats: Tensor
    ) -> dict[str, float]: ...


class CompositeEvaluator:
    """Combines multiple evaluators; the sample evaluators share one set of draws."""

    def __init__(self, evaluators: list[Evaluator], sample_evaluators: list[SampleEvaluator]):
        self.evaluators = evaluators
        self.sample_evaluators = sample_evaluators

    def compute(
        self,
        model: PosteriorModel,
        problem: Problem,
        x_test: Tensor,
        z_test: Tensor,
        batch_size: int,
    ) -> dict[str, float]:
        metrics = {}
        for evaluator in self.evaluators:
            metrics.update(evaluator.compute(model, problem, x_test, z_test, batch_size))
        if self.sample_evaluators:
            metrics.update(
                _sample_metrics(self.sample_evaluators, model, problem, x_test, z_test, batch_size)
            )
        return metrics


class VarianceCalibrationEvaluator(SampleEvaluator):
    """Predictive-variance calibration without an analytic posterior.

    For z from the true posterior, E[(z - mu_hat(x))^2 | x] = sigma*^2(x) +
    (mu*(x) - mu_hat(x))^2, so simulated pairs estimate it directly. mu_hat and
    sigma_hat^2 are the sample mean and unbiased variance of the S posterior draws.

    pull_variance is E[(z - mu_hat)^2 / sigma_hat^2] over (test point, coordinate)
    cells, divided by (1 + 1/S)(S - 1)/(S - 3): that ratio's mean for an exact Gaussian
    posterior, where it is (1 + 1/S) times a squared Student t with S - 1 degrees of
    freedom. Target 1, above 1 under-dispersed. residual_rms is sqrt(E[(z - mu_hat)^2]).
    """

    def chunk_stats(self, problem: Problem, samples: Tensor, z_b: Tensor) -> Tensor:
        sq = (z_b - samples.mean(dim=1)).pow(2)
        return torch.stack([samples.var(dim=1), sq], dim=1)

    def metrics(
        self, problem: Problem, x_test: Tensor, z_test: Tensor, stats: Tensor
    ) -> dict[str, float]:
        var, sq = stats[:, 0], stats[:, 1]
        s = self.n_samples
        finite_sample_mean = (1 + 1 / s) * (s - 1) / (s - 3)
        return {
            "pull_variance": (sq / var).mean().item() / finite_sample_mean,
            "residual_rms": math.sqrt(sq.mean().item()),
        }


class SBCEvaluator(SampleEvaluator):
    """Simulation-based calibration (Talts et al. 2018): sbc_ks is the KS statistic of
    the true z's randomized ranks among the posterior draws, pooled over (test point,
    coordinate), against U(0, 1). It tests marginal calibration only."""

    def chunk_stats(self, problem: Problem, samples: Tensor, z_b: Tensor) -> Tensor:
        return (samples < z_b.unsqueeze(1)).sum(dim=1)

    def metrics(
        self, problem: Problem, x_test: Tensor, z_test: Tensor, stats: Tensor
    ) -> dict[str, float]:
        return {"sbc_ks": _ks_uniform(_randomized_pit(stats, self.n_samples))}


class TARPEvaluator(SampleEvaluator):
    """Joint coverage test via TARP (Lemos et al. 2023).

    For a prior-drawn reference point, the count of posterior draws closer to it than
    the true z is uniform on {0, ..., S} iff the posterior is calibrated; tarp_ks is
    the KS statistic of the pooled randomized counts against U(0, 1).
    """

    def chunk_stats(self, problem: Problem, samples: Tensor, z_b: Tensor) -> Tensor:
        """Per test point, the count for each of the R reference draws: [chunk, R]."""
        ref = problem.sample_prior(z_b.shape[0] * _TARP_REFERENCE_DRAWS)
        ref = ref.view(z_b.shape[0], _TARP_REFERENCE_DRAWS, -1)
        d_true = (z_b.unsqueeze(1) - ref).norm(dim=-1)
        d_samples = torch.cdist(samples, ref, compute_mode="donot_use_mm_for_euclid_dist")
        return (d_samples < d_true.unsqueeze(1)).sum(dim=1)

    def metrics(
        self, problem: Problem, x_test: Tensor, z_test: Tensor, stats: Tensor
    ) -> dict[str, float]:
        return {"tarp_ks": _ks_uniform(_randomized_pit(stats, self.n_samples))}


class C2STEvaluator:
    """Classifier two-sample test between simulated (x, z) and (x, z_model) pairs.

    Draws its own ``n_pairs`` joint pairs, so the test's power is independent of
    ``n_test``. Held-out accuracy (c2st_accuracy) 0.5 means indistinguishable; no x
    appears in both the classifier's train and test halves.
    """

    def __init__(self, n_pairs: int, n_epochs: int):
        self.n_pairs = n_pairs
        self.n_epochs = n_epochs

    def compute(
        self,
        model: PosteriorModel,
        problem: Problem,
        x_test: Tensor,
        z_test: Tensor,
        batch_size: int,
    ) -> dict[str, float]:
        del x_test, z_test
        model.eval()
        with torch.no_grad():
            x, z_true = problem.sample_joint(self.n_pairs)
        z_model = torch.cat(
            [s.reshape(s.shape[0], -1) for _, s in _chunked_samples(model, x, 1, batch_size)]
        )
        feats_model = torch.cat([x, z_model], dim=-1)
        feats_true = torch.cat([x, z_true], dim=-1)

        n = self.n_pairs
        half = n // 2
        dim_feat = feats_model.shape[-1]
        train_f = torch.cat([feats_model[:half], feats_true[:half]])
        test_f = torch.cat([feats_model[half:], feats_true[half:]])
        train_y = torch.cat([torch.ones(half), torch.zeros(half)]).to(x.device)
        test_y = torch.cat([torch.ones(n - half), torch.zeros(n - half)]).to(x.device)

        mu = train_f.mean(dim=0)
        sd = train_f.std(dim=0).clamp(min=1e-6)
        train_f = (train_f - mu) / sd
        test_f = (test_f - mu) / sd

        clf = nn.Sequential(
            nn.Linear(dim_feat, _C2ST_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(_C2ST_HIDDEN_DIM, _C2ST_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(_C2ST_HIDDEN_DIM, 1),
        ).to(x.device)
        optimizer = torch.optim.Adam(clf.parameters(), lr=_C2ST_LR)
        loss_fn = nn.BCEWithLogitsLoss()
        n_rows = train_f.shape[0]
        for _ in range(self.n_epochs):
            perm = torch.randperm(n_rows, device=train_f.device)
            for start in range(0, n_rows, _C2ST_BATCH_SIZE):
                idx = perm[start : start + _C2ST_BATCH_SIZE]
                optimizer.zero_grad()
                loss = loss_fn(clf(train_f[idx]).squeeze(-1), train_y[idx])
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            accuracy = ((clf(test_f).squeeze(-1) > 0) == (test_y > 0.5)).float().mean().item()
        return {"c2st_accuracy": accuracy}
