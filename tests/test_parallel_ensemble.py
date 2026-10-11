"""``train_members``: vmap/loop equivalence, no cross-member data leak, per-member clipping."""

from typing import cast

import torch
from torch import nn

from cf_npe.networks.models import MeanRegressor
from cf_npe.problems.base import EncoderConfig
from cf_npe.problems.simple import Gaussian
from cf_npe.stages.stage1 import build_mean_regressor
from cf_npe.training.loop import TrainingConfig, model_loss
from cf_npe.training.parallel_ensemble import (
    _clip_per_member,
    _run_training_epoch,
    _run_validation,
    _stack_indices,
    train_members,
)
from cf_npe.training.parallel_seed import build_seeded


def _members(problem, n_members: int, hidden_dim: int = 16) -> list[nn.Module]:
    enc_cfg = EncoderConfig(hidden_dim=hidden_dim, n_layers=2, dropout=0.0)
    return build_seeded(
        lambda _s, _k: build_mean_regressor(problem, enc_cfg, "cuda"), [0], n_members
    )


def test_parallel_matches_sequential_on_gaussian():
    """Per-member validation losses from vmap match the plain loop's for the same inits."""
    torch.manual_seed(0)
    problem = Gaussian(dim=3, noise_std=0.5)
    x, z = problem.sample_joint(200)
    split = (torch.arange(50, 200), torch.arange(50))
    n_members = 4

    def train(parallel: bool):
        return train_members(
            _members(problem, n_members),
            x,
            z,
            [split] * n_members,
            training=TrainingConfig(
                n_epochs=20, batch_size=32, lr=1e-3, weight_decay=0.0, grad_clip=None
            ),
            loss_fn=model_loss,
            parallel=parallel,
        )

    sequential, parallel = train(False), train(True)
    x_val, z_val = x[:50], z[:50]
    with torch.no_grad():
        for k in range(n_members):
            seq_loss = cast(MeanRegressor, sequential.members[k]).training_loss(x_val, z_val).mean()
            par_loss = cast(MeanRegressor, parallel.members[k]).training_loss(x_val, z_val).mean()
            assert abs(seq_loss - par_loss) / max(abs(seq_loss.item()), 1.0) < 0.05, (
                f"member {k}: sequential={seq_loss:.6f} parallel={par_loss:.6f}"
            )


def test_no_information_leak_between_members():
    """Member k, trained on its own block with constant target k + 1, predicts k + 1."""
    torch.manual_seed(0)
    problem = Gaussian(dim=3, noise_std=0.5)
    x_shared, _ = problem.sample_joint(200)
    n_members, n = 4, 200
    x = x_shared.repeat(n_members, 1)
    z = torch.cat(
        [torch.full((n, problem.dim_z), k + 1.0, device="cuda") for k in range(n_members)]
    )
    splits = [(torch.arange(n) + k * n, torch.arange(n) + k * n) for k in range(n_members)]

    result = train_members(
        _members(problem, n_members),
        x,
        z,
        splits,
        training=TrainingConfig(
            n_epochs=80, batch_size=64, lr=1e-2, weight_decay=0.0, grad_clip=None
        ),
        loss_fn=model_loss,
        parallel=True,
    )

    with torch.no_grad():
        for k in range(n_members):
            pred = cast(MeanRegressor, result.members[k]).get_mean(x_shared)
            assert abs(pred.mean().item() - (k + 1)) < 0.3, (
                f"member {k}: predicted mean={pred.mean().item():.3f}, expected {k + 1}"
            )
            own_mse = (pred - (k + 1)).pow(2).mean().item()
            for j in range(n_members):
                if j != k:
                    assert own_mse < (pred - (j + 1)).pow(2).mean().item()


def test_clip_per_member_matches_clip_grad_norm():
    """Each member's gradient is clipped as clip_grad_norm_ clips it alone."""
    torch.manual_seed(0)
    grads = {"w": torch.randn(3, 4, 5), "b": torch.randn(3, 5)}
    grads["w"][1] *= 1e-3
    grads["b"][1] *= 1e-3
    params = {name: torch.zeros_like(g, requires_grad=True) for name, g in grads.items()}
    for name, p in params.items():
        p.grad = grads[name].clone()
    _clip_per_member(params, 1.0)
    for k in range(3):
        member = [nn.Parameter(torch.zeros_like(g[k])) for g in grads.values()]
        for p, g in zip(member, grads.values(), strict=True):
            p.grad = g[k].clone()
        nn.utils.clip_grad_norm_(member, 1.0)
        for p, stacked in zip(member, params.values(), strict=True):
            torch.testing.assert_close(stacked.grad[k], p.grad)


def test_unequal_splits_average_each_members_own_entries():
    """Splits of different lengths are padded, not truncated: the per-member mean loss covers
    exactly each member's own indices, in training and in validation."""
    torch.manual_seed(0)
    z = torch.arange(20, dtype=torch.float32, device="cuda")[:, None]
    x = torch.zeros(20, 1, device="cuda")
    splits = [torch.arange(0, 3), torch.arange(3, 10), torch.arange(10, 20)]
    idx, mask = _stack_indices(splits, z.device)
    w = torch.zeros(len(splits), requires_grad=True, device="cuda")

    def vmap_loss(state, xb, zb):
        return zb[..., 0] + state["w"][:, None]

    expected = torch.stack([z[s].mean() for s in splits])
    val = _run_validation(vmap_loss, {"w": w}, {}, x, z, idx, mask, batch_size=4)
    torch.testing.assert_close(val, expected)
    train = _run_training_epoch(
        vmap_loss,
        {"w": w},
        {},
        x,
        z,
        idx,
        mask,
        torch.optim.Adam([w], lr=0.0),
        TrainingConfig(n_epochs=1, batch_size=4, lr=0.0, weight_decay=0.0, grad_clip=None),
    )
    torch.testing.assert_close(train, expected)
