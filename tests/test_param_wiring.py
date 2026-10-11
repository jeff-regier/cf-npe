"""Each tunable architecture parameter changes the trained model's parameter count."""

from dataclasses import replace

import pytest
import torch
from conftest import cf_npe_two_step, training_config

from cf_npe.problems.base import EncoderConfig
from cf_npe.problems.simple import Gaussian
from cf_npe.stages.stage2 import Stage2Config


def _num_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@pytest.fixture
def lp_problem():
    return Gaussian(dim=2, noise_std=0.5)


def _tiny_stage2_cfg() -> Stage2Config:
    return Stage2Config(n_layers=2, n_transforms=2, hidden_dim=16, dropout=0.0)


def _tiny_encoder_cfg(hidden_dim: int = 32) -> EncoderConfig:
    return EncoderConfig(
        hidden_dim=hidden_dim,
        n_layers=2,
        dropout=0.0,
    )


def _tiny_train_cfg() -> dict:
    """Minimal training config."""
    return {
        "n_train": 300,
        "training": training_config(n_epochs=3, batch_size=64, lr=1e-3),
        "encoder_cfg": _tiny_encoder_cfg(),
        "posterior_type": "nsf",
        "k_folds": 3,
        "n_replicates": 1,
        "stage2_cfg": _tiny_stage2_cfg(),
        "stage2_training": training_config(n_epochs=3, batch_size=64, lr=1e-3),
    }


def _count(problem, train_fn, **kwargs):
    return _num_params(train_fn(problem=problem, seeds=[0], **kwargs)[0].model)


def test_offset_hidden_dim_affects_nsf(lp_problem):
    base = _tiny_train_cfg()
    small_cfg = {**base, "stage2_cfg": replace(base["stage2_cfg"], hidden_dim=16)}
    large_cfg = {**base, "stage2_cfg": replace(base["stage2_cfg"], hidden_dim=64)}
    small = _count(lp_problem, cf_npe_two_step, **small_cfg)
    large = _count(lp_problem, cf_npe_two_step, **large_cfg)
    assert large > small, (
        f"stage2_cfg.hidden_dim doesn't affect NSF size: 16 → {small} params, 64 → {large} params"
    )


def test_offset_n_transforms_affects_nsf(lp_problem):
    base = _tiny_train_cfg()
    shallow_cfg = {**base, "stage2_cfg": replace(base["stage2_cfg"], n_transforms=2)}
    deep_cfg = {**base, "stage2_cfg": replace(base["stage2_cfg"], n_transforms=5)}
    shallow = _count(lp_problem, cf_npe_two_step, **shallow_cfg)
    deep = _count(lp_problem, cf_npe_two_step, **deep_cfg)
    assert deep > shallow, (
        f"stage2_cfg.n_transforms doesn't affect NSF depth: 2 → {shallow}, 5 → {deep}"
    )


def test_hidden_dim_affects_mean_network(lp_problem):
    small = _count(
        lp_problem,
        cf_npe_two_step,
        **{**_tiny_train_cfg(), "encoder_cfg": _tiny_encoder_cfg(hidden_dim=16)},
    )
    large = _count(
        lp_problem,
        cf_npe_two_step,
        **{**_tiny_train_cfg(), "encoder_cfg": _tiny_encoder_cfg(hidden_dim=64)},
    )
    assert large > small


def test_n_replicates_scales_mean_ensemble(lp_problem):
    """n_replicates multiplies the Stage-1 mean-net ensemble's parameters."""
    cfg = _tiny_train_cfg()
    cfg.pop("n_replicates")
    few = _count(
        lp_problem,
        cf_npe_two_step,
        **{**cfg, "n_replicates": 1},
    )
    many = _count(
        lp_problem,
        cf_npe_two_step,
        **{**cfg, "n_replicates": 3},
    )
    assert many > few
