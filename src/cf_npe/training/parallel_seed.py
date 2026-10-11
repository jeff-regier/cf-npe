"""Train every seed's networks of a job in one ``train_members`` call.

Members are flattened in row-major (seed, member) order over one pool of all seeds' data.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from cf_npe.problems.base import Problem
from cf_npe.training.loop import LossFn, TrainingConfig
from cf_npe.training.parallel_ensemble import EnsembleResult, train_members

Splits = list[tuple[Tensor, Tensor]]


def combine_init_seed(outer_seed: int, member_idx: int) -> int:
    """Init seed for member ``member_idx`` of outer seed ``outer_seed``, used on every path."""
    return (outer_seed * 1_000_003 + member_idx) & 0x7FFFFFFF


def build_seeded(
    make: Callable[[int, int], nn.Module], seeds: Sequence[int], members_per_seed: int
) -> list[nn.Module]:
    """``make(s, k)`` in row-major (seed, member) order, each after seeding the RNG with
    ``combine_init_seed(seeds[s], k)``."""
    modules = []
    for s, seed in enumerate(seeds):
        for k in range(members_per_seed):
            torch.manual_seed(combine_init_seed(seed, k))
            modules.append(make(s, k))
    return modules


def _pool_seeds(data: Sequence[SeedData]) -> tuple[Tensor, Tensor, Splits]:
    """Pooled ``(x, z)`` and every seed's member splits shifted to index into the pool."""
    offsets = torch.tensor([0] + [d.x.shape[0] for d in data]).cumsum(0).tolist()
    flat = [
        (train_idx + offset, val_idx + offset)
        for offset, d in zip(offsets[:-1], data, strict=True)
        for train_idx, val_idx in d.splits
    ]
    return torch.cat([d.x for d in data]), torch.cat([d.z for d in data]), flat


def simulate_seeds(
    problem: Problem, n_train: int, seeds: Sequence[int], member_splits: Callable[[int], Splits]
) -> list[SeedData]:
    """Per seed: reseed the RNG, simulate ``n_train`` pairs, and draw the member splits."""
    data = []
    for seed in seeds:
        torch.manual_seed(seed)
        x, z = problem.sample_joint(n_train)
        data.append(SeedData(seed, x, z, member_splits(n_train)))
    return data


def _unstack_by_seed[T](flat_list: Sequence[T], members_per_seed: int) -> list[list[T]]:
    n = len(flat_list)
    if n % members_per_seed != 0:
        raise ValueError(f"len(flat_list)={n} not divisible by members_per_seed={members_per_seed}")
    return [list(flat_list[s : s + members_per_seed]) for s in range(0, n, members_per_seed)]


def train_seeds(
    data: Sequence[SeedData],
    make: Callable[[int, int], nn.Module],
    *,
    training: TrainingConfig,
    loss_fn: LossFn,
    parallel: bool,
) -> list[EnsembleResult]:
    """Train member k of every seed s, ``make(s, k)``, on ``data[s].splits[k]``.

    Every seed must have the same number of members. Returns one EnsembleResult per seed.
    """
    x, z, splits = _pool_seeds(data)
    per_seed = len(data[0].splits)
    modules = build_seeded(make, [d.seed for d in data], per_seed)
    result = train_members(
        modules, x, z, splits, training=training, loss_fn=loss_fn, parallel=parallel
    )
    return [
        EnsembleResult(members, val_losses)
        for members, val_losses in zip(
            _unstack_by_seed(result.members, per_seed),
            _unstack_by_seed(result.val_losses, per_seed),
            strict=True,
        )
    ]


@dataclass
class SeedData:
    """One seed's training pairs and its members' (train_idx, val_idx) splits."""

    seed: int
    x: Tensor
    z: Tensor
    splits: Splits
