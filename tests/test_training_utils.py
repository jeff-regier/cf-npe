import math

import pytest
import torch

from cf_npe.methods.std_npe import _beta_nll_loss, _stop_gradient_nll_loss
from cf_npe.problems.base import EncoderConfig
from cf_npe.problems.simple import Gaussian
from cf_npe.training.loop import (
    TrainingConfig,
    _random_subsample_folds,
    chunked_mean,
    evaluate_validation_loss,
    members_result,
    model_loss,
    run_training_loop_with_tracking,
    validation_split,
)


def _make_model(problem, hidden_dim=16):
    """Small Gaussian posterior."""
    return problem.create_posterior(
        EncoderConfig(hidden_dim=hidden_dim, n_layers=3, dropout=0.0), "gaussian", None
    )


# chunked_mean


def test_chunked_mean_matches_full():
    """chunked_mean equals fn(x).mean(0), including a short final chunk."""
    x = torch.randn(100, 3, device="cuda")
    z = torch.randn(100, 3, device="cuda")

    def fn(xb, zb):
        return (xb * zb).sum(dim=-1)

    full_result = (x * z).sum(dim=-1).mean().item()
    for batch_size in (32, 64):
        result = chunked_mean(fn, x, z, batch_size=batch_size)
        assert abs(result - full_result) < 1e-5


# run_training_loop_with_tracking best-checkpoint restore


def _overfit_loop():
    """Overfit a tiny model so val loss rises late; also return the re-evaluated val loss."""
    torch.manual_seed(0)
    problem = Gaussian(dim=2, noise_std=0.5)
    x, z = problem.sample_joint(400)
    train_idx, val_idx = validation_split(400, 0.2)
    x_train, x_val, z_train, z_val = x[train_idx], x[val_idx], z[train_idx], z[val_idx]

    model = _make_model(problem, hidden_dim=128)
    vl = run_training_loop_with_tracking(
        model,
        x_train[:40],
        z_train[:40],
        x_val,
        z_val,
        training=TrainingConfig(
            n_epochs=60, batch_size=16, lr=0.05, weight_decay=0.0, grad_clip=None
        ),
        loss_fn=model_loss,
    )
    final_val = evaluate_validation_loss(model, x_val, z_val, 16)
    return vl, final_val


def test_run_training_loop_restore_best_recovers_argmin():
    vl, final_val = _overfit_loop()

    assert min(vl) < vl[-1]  # non-vacuous: overfitting actually happened
    assert abs(final_val - min(vl)) < 1e-4
    assert final_val < vl[-1]


# _stop_gradient_nll_loss


def test_stop_gradient_mean_grad_is_pure_mse():
    """Stirn et al. 2023: the mean head's gradient is that of MSE alone; the variance head
    still gets a nonzero gradient."""
    torch.manual_seed(0)
    problem = Gaussian(dim=3, noise_std=0.5)
    model = _make_model(problem, hidden_dim=32)
    x, z = problem.sample_joint(64)

    model.zero_grad()
    _stop_gradient_nll_loss(model, x, z).mean().backward()
    sg_mean_grads = [p.grad.clone() for p in model.mean_head.parameters()]
    log_std_grads = [p.grad for p in model.log_std_head.parameters()]

    model.zero_grad()
    mean, _ = model.forward(x)
    mse = ((z - mean) ** 2).sum(dim=-1).mean()
    mse.backward()
    mse_mean_grads = [p.grad for p in model.mean_head.parameters()]

    for sg, ref in zip(sg_mean_grads, mse_mean_grads, strict=True):
        assert torch.equal(sg, ref)
    assert all(g is not None and g.abs().sum() > 0 for g in log_std_grads)


def test_stop_gradient_loss_value_matches_closed_form():
    """The loss equals MSE.sum(-1).mean() + the torch.distributions diagonal-Gaussian NLL."""
    torch.manual_seed(1)
    problem = Gaussian(dim=3, noise_std=0.5)
    model = _make_model(problem, hidden_dim=32).double()
    x, z = problem.sample_joint(64)
    x, z = x.double(), z.double()

    mean, log_std = model.forward(x)
    ref_dist = torch.distributions.Normal(mean.detach(), torch.exp(log_std))
    ref_nll = -ref_dist.log_prob(z).sum(dim=-1).mean()
    ref_mse = ((z - mean) ** 2).sum(dim=-1).mean()
    expected = (ref_mse + ref_nll).item()

    got = _stop_gradient_nll_loss(model, x, z).mean().item()
    assert abs(got - expected) < 1e-9
    assert math.isfinite(expected)


# _beta_nll_loss


def test_beta_nll_mean_gradient_is_free_of_each_dims_variance():
    """Seitzer et al. 2022, beta=1: d loss / d mean_j = (mean_j - z_j) / N for any variances."""

    class FixedGaussian(torch.nn.Module):
        def __init__(self, mean, log_std):
            super().__init__()
            self.mean = torch.nn.Parameter(mean)
            self.log_std = torch.nn.Parameter(log_std)

        def forward(self, x):
            return self.mean.expand(x.shape[0], -1), self.log_std.expand(x.shape[0], -1)

    torch.manual_seed(0)
    model = FixedGaussian(torch.randn(3), torch.tensor([-1.0, 0.0, 1.5]))
    x, z = torch.zeros(5, 1), torch.randn(5, 3)
    _beta_nll_loss(model, x, z).mean().backward()
    expected = (model.mean.detach() - z).mean(dim=0)
    torch.testing.assert_close(model.mean.grad, expected)


# members_result


def test_members_result_averages_each_networks_own_best():
    """Each network keeps its own best epoch, so the result averages those, not the argmin
    of the mean curve (epoch 2, mean loss 2.5)."""
    val_curves = [[0.1, 5.0, 5.0], [5.0, 5.0, 0.0]]
    result = members_result(torch.nn.Identity(), val_curves)
    assert result.best_epoch == 1.0
    assert result.best_val_loss == pytest.approx(0.05)
    assert result.extra_metrics() == {"best_epoch": 1.0, "best_val_loss": result.best_val_loss}


def test_random_subsample_folds_hold_out_independent_validation_sets():
    """Each Std-NPE-mixture component holds out its own validation set."""
    folds = _random_subsample_folds(200, 5, 0.1)
    for train_idx, val_idx in folds:
        assert val_idx.numel() == 20
        assert torch.equal(torch.cat([train_idx, val_idx]).sort().values, torch.arange(200))
    assert len({tuple(sorted(v.tolist())) for _, v in folds}) == 5
