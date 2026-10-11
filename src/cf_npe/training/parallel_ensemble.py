"""Train several networks ("members"), each on its own split of one shared data pool.

The parallel backend stacks parameters along a leading member axis and runs
``functional_call`` under ``torch.vmap``. It requires one shared architecture, pads shorter
splits to the longest member's and weights the padding out of every loss, and does not support
state mutated in forward (e.g. BatchNorm stats).
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.func import functional_call, stack_module_state
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from cf_npe.networks.posterior import PosteriorModel
from cf_npe.training import amp
from cf_npe.training.loop import (
    LossFn,
    NonFiniteLossError,
    TrainingConfig,
    epoch_prefix,
    run_training_loop_with_tracking,
)

_logger = logging.getLogger(__name__)

_State = dict[str, Tensor]


def _build_base_and_stacked_state(
    modules: list[_LossDispatcher],
) -> tuple[_LossDispatcher, _State, _State]:
    """Stacked params and buffers, plus a copy of modules[0] whose own weights go unused."""
    params, buffers = stack_module_state(modules)
    params = {name: p.detach().clone().requires_grad_(True) for name, p in params.items()}
    buffers = {name: b.detach().clone() for name, b in buffers.items()}
    return copy.deepcopy(modules[0]), params, buffers


def _per_member_shape(t: Tensor, ndim: int) -> Tensor:
    """View a [M] tensor so it broadcasts against an [M, ...] tensor of ``ndim`` dims."""
    return t.view((t.shape[0],) + (1,) * (ndim - 1))


def _clip_per_member(params: _State, max_norm: float) -> None:
    """Scale each member's gradient to global norm <= max_norm, as clip_grad_norm_ does."""
    grads = [p.grad for p in params.values() if p.grad is not None]
    norms = torch.stack([g.flatten(1).pow(2).sum(dim=1) for g in grads]).sum(dim=0).sqrt()
    scale = (max_norm / (norms + 1e-6)).clamp(max=1.0)
    for g in grads:
        g.mul_(_per_member_shape(scale, g.dim()))


def _stack_indices(indices: Sequence[Tensor], device: torch.device) -> tuple[Tensor, Tensor]:
    """[M, n_max] index matrix, each row padded with index 0 past its own length, and the
    [M, n_max] float mask that is 1 on real entries and 0 on padding."""
    lengths = torch.tensor([idx.shape[0] for idx in indices])
    mask = torch.arange(int(lengths.max())) < lengths[:, None]
    return pad_sequence(list(indices), batch_first=True).to(device), mask.float().to(device)


def _masked_sums(per_sample: Tensor, mask: Tensor) -> Tensor:
    """Per-member sum of the [M, b] losses over real (unpadded) entries."""
    return (per_sample * mask).sum(dim=1)


def _run_training_epoch(
    vmap_loss: Callable[..., Tensor],
    params: _State,
    buffers: _State,
    x: Tensor,
    z: Tensor,
    train_idx: Tensor,
    train_mask: Tensor,
    optimizer: Adam,
    training: TrainingConfig,
) -> Tensor:
    """One epoch on independent per-member shuffles; returns the per-member mean loss [M].

    Each member's real entries are shuffled ahead of its padding, so its minibatches are full
    until its own split runs out, as in the sequential loop.
    """
    n_members, n = train_idx.shape
    keys = torch.rand(n_members, n, device=x.device).masked_fill(train_mask == 0, 2.0)
    rows = train_idx.gather(1, torch.argsort(keys, dim=1))
    accum = torch.zeros(n_members, device=x.device)
    for start in range(0, n, training.batch_size):
        batch = rows[:, start : start + training.batch_size]
        mask = train_mask[:, start : start + training.batch_size]
        optimizer.zero_grad()
        with amp.autocast():
            sums = _masked_sums(vmap_loss({**params, **buffers}, x[batch], z[batch]), mask)
        (sums / mask.sum(dim=1).clamp(min=1)).sum().backward()
        if training.grad_clip is not None:
            _clip_per_member(params, training.grad_clip)
        optimizer.step()
        accum = accum + sums.detach()
    return accum / train_mask.sum(dim=1)


def _run_validation(
    vmap_loss: Callable[..., Tensor],
    params: _State,
    buffers: _State,
    x: Tensor,
    z: Tensor,
    val_idx: Tensor,
    val_mask: Tensor,
    batch_size: int,
) -> Tensor:
    """Per-member mean validation loss [M] over each member's real entries."""
    n_members, n = val_idx.shape
    accum = torch.zeros(n_members, device=x.device)
    with torch.no_grad(), amp.autocast():
        for start in range(0, n, batch_size):
            batch = val_idx[:, start : start + batch_size]
            mask = val_mask[:, start : start + batch_size]
            accum = accum + _masked_sums(vmap_loss({**params, **buffers}, x[batch], z[batch]), mask)
    return accum / val_mask.sum(dim=1)


def _strip_inner_prefix(state: _State, k: int) -> _State:
    """Slice member k and drop the dispatcher's ``inner.`` prefix from state keys."""
    return {name[len("inner.") :]: t[k] for name, t in state.items() if name.startswith("inner.")}


def _train_parallel(
    modules: list[nn.Module],
    x: Tensor,
    z: Tensor,
    splits: Sequence[tuple[Tensor, Tensor]],
    *,
    training: TrainingConfig,
    loss_fn: LossFn,
) -> EnsembleResult:
    """Train all members under vmap and load each one's best-validation state into it."""
    n_members = len(modules)
    train_idx, train_mask = _stack_indices([tr for tr, _ in splits], x.device)
    val_idx, val_mask = _stack_indices([va for _, va in splits], x.device)
    base, params, buffers = _build_base_and_stacked_state(
        [_LossDispatcher(m, loss_fn) for m in modules]
    )
    optimizer = Adam(list(params.values()), lr=training.lr, weight_decay=training.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=training.n_epochs)

    def compute_loss(state: _State, xb: Tensor, zb: Tensor) -> Tensor:
        return functional_call(base, state, args=(xb, zb))

    # randomness='different' gives each member its own dropout mask; vmap otherwise errors on
    # random ops.
    vmap_loss = torch.vmap(compute_loss, in_dims=(0, 0, 0), randomness="different")

    best_val = torch.full((n_members,), float("inf"), device=x.device)
    best_state = {name: t.detach().clone() for name, t in {**params, **buffers}.items()}
    val_curves: list[list[float]] = [[] for _ in range(n_members)]
    _logger.info("Training %d networks in parallel", n_members)
    for epoch in range(training.n_epochs):
        base.train()
        train_loss = _run_training_epoch(
            vmap_loss, params, buffers, x, z, train_idx, train_mask, optimizer, training
        )
        scheduler.step()
        base.eval()
        val_loss = _run_validation(
            vmap_loss, params, buffers, x, z, val_idx, val_mask, training.batch_size
        )
        if not bool(torch.isfinite(val_loss).all()):
            raise NonFiniteLossError(f"non-finite val loss {val_loss.tolist()} at epoch {epoch}")
        improved = val_loss < best_val
        for name, t in {**params, **buffers}.items():
            mask = _per_member_shape(improved, t.dim())
            best_state[name] = torch.where(mask, t.detach(), best_state[name])
        best_val = torch.where(improved, val_loss, best_val)
        for k, vl in enumerate(val_loss.tolist()):
            val_curves[k].append(vl)
        _logger.info(
            "%s [M=%d]: loss=%.4f, val=%.4f",
            epoch_prefix(epoch, training.n_epochs),
            n_members,
            train_loss.mean().item(),
            val_loss.mean().item(),
        )

    for k, module in enumerate(modules):
        module.load_state_dict(_strip_inner_prefix(best_state, k), strict=True)
        module.eval()
    return EnsembleResult(modules, val_curves)


def _train_sequential(
    modules: list[nn.Module],
    x: Tensor,
    z: Tensor,
    splits: Sequence[tuple[Tensor, Tensor]],
    *,
    training: TrainingConfig,
    loss_fn: LossFn,
) -> EnsembleResult:
    result = EnsembleResult(modules, [])
    for k, (module, (tr, va)) in enumerate(zip(modules, splits, strict=True)):
        _logger.info("Training network %d/%d", k + 1, len(modules))
        vl = run_training_loop_with_tracking(
            cast(PosteriorModel, module),
            x[tr],
            z[tr],
            x[va],
            z[va],
            training=training,
            loss_fn=loss_fn,
        )
        module.eval()
        result.val_losses.append(vl)
    return result


def train_members(
    modules: list[nn.Module],
    x: Tensor,
    z: Tensor,
    splits: Sequence[tuple[Tensor, Tensor]],
    *,
    training: TrainingConfig,
    loss_fn: LossFn,
    parallel: bool,
) -> EnsembleResult:
    """Train ``modules[k]`` on ``(x, z)[splits[k][0]]``, early-stopped on ``splits[k][1]``.

    Validation uses each model's own ``training_loss``. ``parallel`` trains under vmap; the
    sequential path serves models vmap cannot trace.
    """
    train = _train_parallel if parallel and len(modules) > 1 else _train_sequential
    return train(modules, x, z, splits, training=training, loss_fn=loss_fn)


class _LossDispatcher(nn.Module):
    """``forward(x, z)`` returns ``loss_fn`` in train mode and ``training_loss`` in eval mode.

    ``functional_call`` only invokes ``forward``.
    """

    def __init__(self, inner: nn.Module, loss_fn: LossFn):
        super().__init__()
        self.inner = inner
        self.loss_fn = loss_fn

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        if self.training:
            return self.loss_fn(self.inner, x, z)
        return cast(PosteriorModel, self.inner).training_loss(x, z)


@dataclass
class EnsembleResult:
    """Trained members (best-validation weights, eval mode) and [member][epoch] validation
    loss curves."""

    members: list[nn.Module]
    val_losses: list[list[float]]
