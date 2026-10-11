"""Stage 2: networks for z given x and the Stage-1 mean, trained on out-of-fold means."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from cf_npe.networks.models import (
    EnsembleMeanNet,
    FlowHead,
    GaussianHead,
    GaussianOffsetTrainingWrapper,
    OffsetNetwork,
    ResidualFlowHead,
    ResidualTrainingWrapper,
    TwoStageModel,
    pack_xmu,
)
from cf_npe.problems.base import Problem
from cf_npe.training.loop import BaseTrainResult, TrainingConfig, members_result, model_loss
from cf_npe.training.parallel_seed import SeedData, train_seeds

_logger = logging.getLogger(__name__)


def _check_family(posterior_type: str, cfg: Stage2Config) -> None:
    if posterior_type not in ("gaussian", "nsf"):
        raise ValueError(f"Unknown posterior_type {posterior_type!r}; expected 'gaussian' or 'nsf'")
    if cfg.residual and posterior_type != "nsf":
        raise ValueError(
            "stage2_cfg.residual requires the nsf family; the Gaussian "
            "offset is already mean-anchored at mu_hat"
        )


def _make_offset(
    problem: Problem, posterior_type: str, cfg: Stage2Config, device: torch.device
) -> nn.Module:
    """Untrained Stage-2 network on packed ``(x, mu)``: a Gaussian log-sigma
    offset in its training wrapper, or a conditional flow."""
    _check_family(posterior_type, cfg)
    if posterior_type == "gaussian":
        offset = OffsetNetwork(
            problem.dim_x + problem.dim_z,
            problem.dim_z,
            cfg.hidden_dim,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
        )
        return GaussianOffsetTrainingWrapper(offset, dim_x_raw=problem.dim_x).to(device)
    return problem.create_offset_posterior(cfg).to(device)


def _head(member: nn.Module, posterior_type: str, residual: bool) -> nn.Module:
    """The Stage-2 head of a trained Stage-2 network, without its training wrapper."""
    if posterior_type == "gaussian":
        return GaussianHead(cast(GaussianOffsetTrainingWrapper, member).inner)
    if residual:
        wrapper = cast(ResidualTrainingWrapper, member)
        return ResidualFlowHead(wrapper.inner, wrapper.residual_scale)
    return FlowHead(member)


def untrained_two_stage(
    problem: Problem,
    posterior_type: str,
    cfg: Stage2Config,
    mean_net: EnsembleMeanNet,
    n_heads: int,
) -> TwoStageModel:
    """An untrained Stage-2 posterior with a trained one's structure, to load a saved state into."""
    device = next(mean_net.parameters()).device
    heads = []
    for _ in range(n_heads):
        offset = _make_offset(problem, posterior_type, cfg, device)
        if cfg.residual:
            scale = torch.ones(problem.dim_z, device=device)
            offset = ResidualTrainingWrapper(offset, problem.dim_x, scale)
        heads.append(_head(offset, posterior_type, cfg.residual))
    return TwoStageModel(mean_net, heads)


def train_stage2(
    problem: Problem,
    posterior_type: str,
    cfg: Stage2Config,
    training: TrainingConfig,
    *,
    seeds: list[int],
    xs: list[Tensor],
    mus: list[Tensor],
    zs: list[Tensor],
    mean_nets: list[EnsembleMeanNet],
    folds: list[list[tuple[Tensor, Tensor]]],
    parallel: bool,
) -> list[BaseTrainResult]:
    """Train every seed's Stage-2 networks, one per split in ``folds[s]``, in one
    ``train_seeds`` call; a seed with several networks gets their uniform mixture over one
    shared Stage-1 mean."""
    dim_x = xs[0].shape[-1]
    data = [
        SeedData(seed, pack_xmu(x, mu.detach()), z, f)
        for seed, x, mu, z, f in zip(seeds, xs, mus, zs, folds, strict=True)
    ]

    def make(s: int, k: int) -> nn.Module:
        d = data[s]
        offset = _make_offset(problem, posterior_type, cfg, d.x.device)
        if not cfg.residual:
            return offset
        train_idx = d.splits[k][0]
        scale = (d.z[train_idx] - d.x[train_idx, dim_x:]).std(dim=0).clamp_min(1e-6)
        return ResidualTrainingWrapper(offset, dim_x, scale)

    results = train_seeds(data, make, training=training, loss_fn=model_loss, parallel=parallel)
    out = []
    for mean_net, r in zip(mean_nets, results, strict=True):
        heads = [_head(m, posterior_type, cfg.residual) for m in r.members]
        model = TwoStageModel(mean_net, heads)
        out.append(members_result(model, r.val_losses))
    return out


@dataclass
class Stage2Config:
    """Architecture of the CF-NPE Stage-2 model.

    The Gaussian family reads ``hidden_dim``, ``n_layers`` and ``dropout``; the NSF
    family's ``Problem.create_offset_posterior`` also reads ``n_transforms``, ``n_bins``
    and ``tail_bound`` (strong lensing reads none of the six, taking its own architecture).

    ``residual`` (NSF only) makes the target ``z - mu_hat``, standardized per
    dimension on each network's training split.
    """

    n_layers: int = 2
    n_transforms: int = 2
    n_bins: int = 8
    tail_bound: float = 5.0
    hidden_dim: int = 256
    dropout: float = 0.0
    residual: bool = False
