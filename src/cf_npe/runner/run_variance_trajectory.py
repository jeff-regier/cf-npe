"""Trace mean accuracy and calibration along Standard NPE optimization (Figure 2).

Per-epoch held-out mean_mse_test (mu_hat - mu*)^2 and pull_variance_test (expected
reduced chi-squared) of a Gaussian Standard NPE on d=200, plus an MSE-only mean network.
"""

import argparse
import json
from pathlib import Path
from typing import cast

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import Tensor
from torch.optim import Adam

from cf_npe import CONF, load_scale
from cf_npe.networks.posterior import PosteriorModel
from cf_npe.problems.base import EncoderConfig
from cf_npe.problems.simple import Gaussian
from cf_npe.stages.stage1 import build_mean_regressor
from cf_npe.training.loop import (
    evaluate_validation_loss,
    model_loss,
    run_training_epoch,
    validation_split,
)

_DIM = 200
_SEED = 0
_ENCODER = EncoderConfig(hidden_dim=256, n_layers=3, dropout=0.0)
_LR = 1e-3
_BATCH_SIZE = 256
_MSE_N_EPOCHS = 300


def _simulate(
    problem: Gaussian, n_train: int, n_test: int
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Training draws split 80/20 into (x_train, z_train, x_val, z_val), then test x."""
    x, z = problem.sample_joint(n_train)
    train_idx, val_idx = validation_split(x.shape[0], 0.2)
    x_test, _ = problem.sample_joint(n_test)
    return x[train_idx], z[train_idx], x[val_idx], z[val_idx], x_test


def _exact_gaussian_metrics(model, problem, x_test):
    """Exact mean MSE and expected reduced chi-squared."""
    model.eval()
    with torch.no_grad():
        mean = model.get_mean(x_test)
        std = model.get_std(x_test)
        squared_mean_error = (mean - problem.posterior_mean(x_test)).pow(2)
        expected_squared_residual = problem.posterior_std() ** 2 + squared_mean_error
        return {
            "mean_mse_test": squared_mean_error.mean().item(),
            "pull_variance_test": (expected_squared_residual / std.pow(2)).mean().item(),
        }


def _std_npe_trajectory(problem, x_train, z_train, x_val, z_val, x_test, n_epochs):
    """Per-epoch test metrics of a Standard NPE fit, and its best-validation epoch."""
    model = problem.create_posterior(_ENCODER, "gaussian", None)
    optimizer = Adam(model.parameters(), lr=_LR)
    trajectory = []
    best_val = float("inf")
    best_epoch = 0
    for epoch in range(n_epochs):
        run_training_epoch(
            model,
            x_train,
            z_train,
            batch_size=_BATCH_SIZE,
            optimizer=optimizer,
            loss_fn=model_loss,
            grad_clip=None,
        )
        val_loss = evaluate_validation_loss(model, x_val, z_val, _BATCH_SIZE)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
        trajectory.append(_exact_gaussian_metrics(model, problem, x_test))
    return trajectory, best_epoch


def _mse_endpoint(problem, x_train, z_train, x_val, z_val, x_test):
    """Held-out mean error of a conditional-mean network early-stopped on validation MSE."""
    torch.manual_seed(_SEED)
    model = cast(PosteriorModel, build_mean_regressor(problem, _ENCODER, "cuda"))
    optimizer = Adam(model.parameters(), lr=_LR)
    best_val = float("inf")
    best_state = {key: value.clone() for key, value in model.state_dict().items()}
    for _ in range(_MSE_N_EPOCHS):
        run_training_epoch(
            model,
            x_train,
            z_train,
            batch_size=_BATCH_SIZE,
            optimizer=optimizer,
            loss_fn=model_loss,
            grad_clip=None,
        )
        val_loss = evaluate_validation_loss(model, x_val, z_val, _BATCH_SIZE)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return (model.get_mean(x_test) - problem.posterior_mean(x_test)).pow(2).mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    sizes = load_scale(args.smoke)["variance_trajectory"]

    torch.manual_seed(_SEED)
    problem = instantiate(OmegaConf.load(CONF / "problem" / "gaussian.yaml"), dim=_DIM)
    data = _simulate(problem, sizes["n_train"], sizes["n_test"])
    trajectory, best_epoch = _std_npe_trajectory(problem, *data, sizes["n_epochs"])
    record = {
        "best_epoch": best_epoch,
        "trajectory": trajectory,
        "mse_endpoint_test": _mse_endpoint(problem, *data),
    }
    out_path = args.root / "results" / "variance_trajectory.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([record]))
    last = trajectory[-1]
    print(
        f"Wrote {out_path}: best_epoch={best_epoch} "
        f"final_mse_test={last['mean_mse_test']:.4f} "
        f"final_chi2={last['pull_variance_test']:.3f} "
        f"mse_endpoint={record['mse_endpoint_test']:.4f}"
    )


if __name__ == "__main__":
    main()
