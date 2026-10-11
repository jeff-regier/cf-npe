"""Standalone training of a CF-NPE Stage-1 mean regressor, to tune it in isolation."""

from cf_npe.problems.base import EncoderConfig, Problem
from cf_npe.stages.stage1 import build_mean_regressor
from cf_npe.training.loop import (
    BaseTrainResult,
    TrainingConfig,
    holdout_split,
    members_result,
    model_loss,
)
from cf_npe.training.parallel_seed import simulate_seeds, train_seeds


def train_mean_regressor(
    problem: Problem,
    n_train: int,
    seeds: list[int],
    training: TrainingConfig,
    encoder_cfg: EncoderConfig,
    parallel_ensemble: bool,
) -> list[BaseTrainResult]:
    """Train one mean regressor per seed with MSE loss on an 80/20 split."""
    data = simulate_seeds(problem, n_train, seeds, holdout_split)
    device = data[0].x.device
    results = train_seeds(
        data,
        lambda _s, _k: build_mean_regressor(problem, encoder_cfg, device),
        training=training,
        loss_fn=model_loss,
        parallel=parallel_ensemble,
    )
    return [members_result(r.members[0], r.val_losses) for r in results]
