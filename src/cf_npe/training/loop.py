"""Single-network training loop and data-split helpers."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR

from cf_npe.networks.posterior import PosteriorModel
from cf_npe.training import amp

DEFAULT_VAL_FRACTION = 0.2


def epoch_prefix(step: int, n_total: int) -> str:
    pct = 100 * (step + 1) / n_total
    return f"Epoch {step + 1}/{n_total} ({pct:.0f}%)"


_logger = logging.getLogger(__name__)

# Returns one loss per row of the batch.
LossFn = Callable[[nn.Module, Tensor, Tensor], Tensor]


def model_loss(model: nn.Module, x: Tensor, z: Tensor) -> Tensor:
    return cast(PosteriorModel, model).training_loss(x, z)


def _chunks(tensors: tuple[Tensor, ...], batch_size: int) -> Iterator[tuple[Tensor, ...]]:
    """Contiguous chunks of ``tensors`` along dim 0."""
    n = tensors[0].shape[0]
    for start in range(0, n, batch_size):
        yield tuple(t[start : start + batch_size] for t in tensors)


def chunked_mean(
    fn: Callable[..., Tensor],
    *tensors: Tensor,
    batch_size: int,
) -> float:
    """Mean of the per-element values ``fn`` returns over contiguous chunks along dim 0."""
    with torch.no_grad():
        total = sum(fn(*b).sum().item() for b in _chunks(tensors, batch_size))
    return total / tensors[0].shape[0]


def run_training_epoch(
    model: PosteriorModel,
    x: Tensor,
    z: Tensor,
    *,
    batch_size: int,
    optimizer: Optimizer,
    loss_fn: LossFn,
    grad_clip: float | None,
) -> float:
    model.train()
    n = x.shape[0]
    perm = torch.randperm(n, device=x.device)
    total_loss = torch.zeros(1, device=x.device)
    module = cast(nn.Module, model)
    for start in range(0, n, batch_size):
        idx = perm[start : start + batch_size]
        optimizer.zero_grad()
        with amp.autocast():
            loss = loss_fn(module, x[idx], z[idx]).mean()
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(module.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.detach() * idx.shape[0]
    return (total_loss / n).item()


def evaluate_validation_loss(
    model: PosteriorModel,
    x_val: Tensor,
    z_val: Tensor,
    batch_size: int,
) -> float:
    """Validation ``training_loss`` in the training forward's precision."""
    model.eval()
    with amp.autocast():
        return chunked_mean(model.training_loss, x_val, z_val, batch_size=batch_size)


def members_result(model: nn.Module, val_curves: list[list[float]]) -> BaseTrainResult:
    """Result for networks with [network][epoch] validation curves ``val_curves``."""
    best_val, best_epoch = torch.tensor(val_curves).min(dim=1)
    return BaseTrainResult(
        cast(PosteriorModel, model), best_epoch.float().mean().item(), best_val.mean().item()
    )


def _build_fold_indices(n: int, n_folds: int) -> list[tuple[Tensor, Tensor]]:
    """Split a random permutation of [0, n) into n_folds (train_idx, val_idx) pairs."""
    perm = torch.randperm(n)
    fold_size = n // n_folds
    folds: list[tuple[Tensor, Tensor]] = []
    for k in range(n_folds):
        start = k * fold_size
        end = start + fold_size if k < n_folds - 1 else n
        val_idx = perm[start:end]
        train_idx = torch.cat([perm[:start], perm[end:]])
        folds.append((train_idx, val_idx))
    return folds


def _random_subsample_folds(
    n: int, n_components: int, val_fraction: float
) -> list[tuple[Tensor, Tensor]]:
    """n_components independent random (train_idx, val_idx) splits; validation sets overlap."""
    val_size = round(n * val_fraction)
    folds: list[tuple[Tensor, Tensor]] = []
    for _ in range(n_components):
        perm = torch.randperm(n)
        folds.append((perm[val_size:], perm[:val_size]))
    return folds


def validation_split(n: int, val_fraction: float) -> tuple[Tensor, Tensor]:
    """(train_idx, val_idx) holding out the first of round(1 / val_fraction) folds."""
    return _build_fold_indices(n, round(1.0 / val_fraction))[0]


def holdout_split(n: int) -> list[tuple[Tensor, Tensor]]:
    """The single train/validation split holding out DEFAULT_VAL_FRACTION."""
    return [validation_split(n, DEFAULT_VAL_FRACTION)]


def run_training_loop_with_tracking(
    model: PosteriorModel,
    x: Tensor,
    z: Tensor,
    x_val: Tensor,
    z_val: Tensor,
    *,
    training: TrainingConfig,
    loss_fn: LossFn,
) -> list[float]:
    """Train on ``loss_fn``, restore the best-validation weights, and return the
    per-epoch validation losses. Validation uses the model's own ``training_loss``."""
    val_losses: list[float] = []
    module = cast(nn.Module, model)
    optimizer = Adam(module.parameters(), lr=training.lr, weight_decay=training.weight_decay)
    best_val = math.inf
    best_state = {k: v.clone() for k, v in module.state_dict().items()}
    scheduler = CosineAnnealingLR(optimizer, T_max=training.n_epochs)

    for epoch in range(training.n_epochs):
        epoch_loss = run_training_epoch(
            model,
            x,
            z,
            batch_size=training.batch_size,
            optimizer=optimizer,
            loss_fn=loss_fn,
            grad_clip=training.grad_clip,
        )
        scheduler.step()
        val_loss = evaluate_validation_loss(model, x_val, z_val, training.batch_size)
        if not math.isfinite(val_loss):
            raise NonFiniteLossError(f"non-finite val loss {val_loss} at epoch {epoch}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in module.state_dict().items()}
        val_losses.append(val_loss)

        _logger.info(
            "%s: loss=%.4f, val=%.4f",
            epoch_prefix(epoch, training.n_epochs),
            epoch_loss,
            val_loss,
        )

    module.load_state_dict(best_state)
    return val_losses


@dataclass(frozen=True)
class TrainingConfig:
    """Adam with ``lr`` cosine-annealed to zero over ``n_epochs``; per-network gradient norm
    clipped to ``grad_clip`` unless None."""

    n_epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    grad_clip: float | None


class NonFiniteLossError(RuntimeError):
    """The validation loss became non-finite: training diverged."""


@dataclass
class BaseTrainResult:
    """A trained model; ``best_epoch`` and ``best_val_loss`` are the means over its networks
    of each network's own best-validation epoch (the weights it keeps) and loss there."""

    model: PosteriorModel
    best_epoch: float
    best_val_loss: float

    def extra_metrics(self) -> dict[str, float]:
        return {"best_epoch": self.best_epoch, "best_val_loss": self.best_val_loss}
