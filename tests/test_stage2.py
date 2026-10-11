import math

import pytest
import torch
from conftest import training_config
from torch import Tensor, nn

from cf_npe.networks.models import (
    EnsembleMeanNet,
    FlowHead,
    ResidualFlowHead,
    TwoStageModel,
)
from cf_npe.problems.base import EncoderConfig
from cf_npe.problems.simple import Gaussian
from cf_npe.stages.stage1 import build_mean_regressor
from cf_npe.stages.stage2 import Stage2Config, train_stage2, untrained_two_stage
from cf_npe.training.loop import _build_fold_indices


def _train(posterior_type, cfg, folds, parallel, n=32):
    """Stage 2 for one seed on random (x, mu, z); returns the model and inputs."""
    torch.manual_seed(0)
    problem = Gaussian(dim=2, noise_std=0.5)
    mean_net = _gaussian_mean_net(problem)
    x = torch.randn(n, problem.dim_x, device="cuda")
    mu = torch.randn(n, problem.dim_z, device="cuda")
    z = torch.randn(n, problem.dim_z, device="cuda")
    (result,) = train_stage2(
        problem,
        posterior_type,
        cfg,
        training_config(n_epochs=2, batch_size=16, lr=1e-3),
        seeds=[0],
        xs=[x],
        mus=[mu],
        zs=[z],
        mean_nets=[mean_net],
        folds=[folds],
        parallel=parallel,
    )
    return result.model, x, mu, z


def _residual_cfg():
    return Stage2Config(hidden_dim=16, n_transforms=1, residual=True)


def test_train_stage2_rejects_unknown_posterior_type():
    """An unsupported posterior_type raises instead of training a Gaussian offset."""
    cfg = Stage2Config()
    with pytest.raises(ValueError, match="Unknown posterior_type"):
        _train("bogus", cfg, [(torch.arange(24), torch.arange(24, 32))], False)


class _StdNormalOffset(nn.Module):
    """Offset stub with iid standard-normal density over r."""

    def log_prob(self, cond: Tensor, r: Tensor) -> Tensor:
        return (-0.5 * (r**2 + math.log(2 * math.pi))).sum(dim=-1)

    def rsample(self, cond: Tensor, n_samples: int) -> Tensor:
        return torch.ones(cond.shape[0], n_samples, self._dim_z, device=cond.device)

    def training_loss(self, cond: Tensor, r: Tensor) -> Tensor:
        return -self.log_prob(cond, r)

    def __init__(self, dim_z: int):
        super().__init__()
        self._dim_z = dim_z


def _gaussian_mean_net(problem: Gaussian) -> EnsembleMeanNet:
    return EnsembleMeanNet(
        [
            build_mean_regressor(
                problem, EncoderConfig(hidden_dim=16, n_layers=3, dropout=0.0), "cuda"
            )
        ]
    )


def test_residual_log_prob_matches_analytic_gaussian():
    """The residual wrapper's log_prob equals the N(mu, residual_scale) density."""
    torch.manual_seed(0)
    problem = Gaussian(dim=3, noise_std=0.5)
    mean_net = _gaussian_mean_net(problem)
    scale = torch.tensor([0.5, 2.0, 1.3], device="cuda")
    model = TwoStageModel(mean_net, [ResidualFlowHead(_StdNormalOffset(3), scale)])

    x = torch.randn(8, problem.dim_x, device="cuda")
    z = torch.randn(8, problem.dim_z, device="cuda")
    mu = model.get_mean(x)

    expected = (-0.5 * (((z - mu) / scale) ** 2 + math.log(2 * math.pi)) - scale.log()).sum(dim=-1)
    torch.testing.assert_close(model.log_prob(x, z), expected)


def test_residual_rsample_maps_back_to_z_space():
    """Sampling adds mu and multiplies by residual_scale."""
    torch.manual_seed(0)
    problem = Gaussian(dim=3, noise_std=0.5)
    mean_net = _gaussian_mean_net(problem)
    scale = torch.tensor([0.5, 2.0, 1.3], device="cuda")
    model = TwoStageModel(mean_net, [ResidualFlowHead(_StdNormalOffset(3), scale)])

    x = torch.randn(4, problem.dim_x, device="cuda")
    mu = model.get_mean(x)

    # The stub's rsample returns all-ones r, so z = mu + scale exactly.
    samples = model.rsample(x, n_samples=2)
    torch.testing.assert_close(samples, (mu + scale).unsqueeze(1).expand(-1, 2, -1))


def test_flow_head_models_raw_z():
    """A non-residual flow head scores raw z conditioned on (x, mu_hat)."""
    torch.manual_seed(0)
    problem = Gaussian(dim=3, noise_std=0.5)
    mean_net = _gaussian_mean_net(problem)
    offset = _StdNormalOffset(3)
    model = TwoStageModel(mean_net, [FlowHead(offset)])

    x = torch.randn(8, problem.dim_x, device="cuda")
    z = torch.randn(8, problem.dim_z, device="cuda")
    cond = torch.cat([x, model.get_mean(x)], dim=-1)

    torch.testing.assert_close(model.log_prob(x, z), offset.log_prob(cond, z))


def test_train_stage2_residual_nsf_end_to_end():
    """Residual nsf stores the training residuals' per-dim scale; log_prob is finite."""
    train_idx, val_idx = torch.arange(24), torch.arange(24, 32)
    model, x, mu, z = _train("nsf", _residual_cfg(), [(train_idx, val_idx)], False)

    (head,) = model.heads
    expected_scale = (z - mu)[train_idx].std(dim=0).clamp_min(1e-6)
    torch.testing.assert_close(head.residual_scale, expected_scale)
    assert torch.isfinite(model.log_prob(x, z)).all()
    assert torch.isfinite(model.sample(x, 2)).all()


def test_train_stage2_residual_vmap_scales_each_member_by_its_own_split():
    """Under vmap, each mixture member keeps the residual std of its own training split."""
    torch.manual_seed(1)
    folds = _build_fold_indices(32, 2)
    model, x, mu, z = _train("nsf", _residual_cfg(), folds, True)

    for head, (train_idx, _) in zip(model.heads, folds, strict=True):
        expected = (z - mu)[train_idx].std(dim=0).clamp_min(1e-6)
        torch.testing.assert_close(head.residual_scale, expected)
    assert torch.isfinite(model.log_prob(x, z)).all()


def test_train_stage2_residual_rejects_gaussian_family():
    cfg = Stage2Config(residual=True)
    with pytest.raises(ValueError, match="mean-anchored"):
        _train("gaussian", cfg, [(torch.arange(24), torch.arange(24, 32))], False)


def test_untrained_two_stage_rejects_residual_gaussian_family():
    """Rebuilding a saved model rejects residual for the Gaussian family, as training does."""
    cfg = Stage2Config(residual=True)
    problem = Gaussian(dim=2, noise_std=0.5)
    with pytest.raises(ValueError, match="mean-anchored"):
        untrained_two_stage(problem, "gaussian", cfg, _gaussian_mean_net(problem), 1)
