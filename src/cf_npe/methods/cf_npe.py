"""CF-NPE trainers: Stage-1 cache build, Stage 2 from the cache, and the single-fit ablation."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import Tensor

from cf_npe.problems.base import EncoderConfig, Problem
from cf_npe.stages.cache import load_cache, save_cache
from cf_npe.stages.stage1 import run_stage1
from cf_npe.stages.stage2 import Stage2Config, train_stage2
from cf_npe.training.loop import (
    BaseTrainResult,
    TrainingConfig,
    _build_fold_indices,
    holdout_split,
    members_result,
)
from cf_npe.training.parallel_seed import simulate_seeds

_logger = logging.getLogger(__name__)


def _offset_folds(n: int, stage2_mixture: bool, n_folds: int) -> list[tuple[Tensor, Tensor]]:
    """Stage-2 splits: one per fold for the Stage-2 mixture, else one 80/20 split."""
    if stage2_mixture:
        return _build_fold_indices(n, n_folds)
    return holdout_split(n)


def train_cf_npe_stage1_only(
    problem: Problem,
    n_train: int,
    seeds: list[int],
    training: TrainingConfig,
    encoder_cfg: EncoderConfig,
    k_folds: int,
    n_replicates: int,
    parallel_ensemble: bool,
    stage1_cache_save: str | None,
) -> list[Stage1OnlyResult]:
    """Train CF-NPE Stage 1 (K folds x M replicates) per seed and write each seed's cache.

    ``__SEED__`` in ``stage1_cache_save`` is replaced by the seed.
    """
    if stage1_cache_save is not None and len(seeds) > 1 and "__SEED__" not in stage1_cache_save:
        raise ValueError(
            f"stage1_cache_save={stage1_cache_save!r} has no __SEED__ "
            f"placeholder: all {len(seeds)} seeds in the batch would write "
            "the same cache file, each overwriting the last."
        )
    setups = simulate_seeds(problem, n_train, seeds, lambda n: _build_fold_indices(n, k_folds))
    s1_results = run_stage1(
        problem,
        setups,
        n_replicates=n_replicates,
        training=training,
        encoder_cfg=encoder_cfg,
        parallel=parallel_ensemble,
    )
    results: list[Stage1OnlyResult] = []
    for setup, s1 in zip(setups, s1_results, strict=True):
        if stage1_cache_save is not None:
            path = stage1_cache_save.replace("__SEED__", str(setup.seed))
            save_cache(path, problem, setup, s1, encoder_cfg)
            _logger.info("Saved CF-NPE Stage-1 cache to %s", path)
        results.append(
            Stage1OnlyResult(
                **vars(members_result(s1.ensemble, s1.val_losses)),
                oof_mse=(s1.mu_hat_oof.detach() - setup.z).pow(2).mean().item(),
            )
        )
    return results


def train_cf_npe_single_fit(
    problem: Problem,
    n_train: int,
    seeds: list[int],
    training: TrainingConfig,
    encoder_cfg: EncoderConfig,
    stage2_cfg: Stage2Config,
    stage2_training: TrainingConfig,
    posterior_type: str,
    parallel_ensemble: bool,
) -> list[BaseTrainResult]:
    """Single-mean-fit ablation of CF-NPE: no folds, no replicates.

    Stage 1 trains one mean network on an 80/20 split and Stage 2 trains on the same
    split, so its training residuals are in-sample.
    """
    setups = simulate_seeds(problem, n_train, seeds, holdout_split)
    s1_results = run_stage1(
        problem,
        setups,
        n_replicates=1,
        training=training,
        encoder_cfg=encoder_cfg,
        parallel=parallel_ensemble,
    )
    ensembles = [s1.ensemble.eval() for s1 in s1_results]
    with torch.no_grad():
        mus = [e.get_mean(st.x) for e, st in zip(ensembles, setups, strict=True)]
    return train_stage2(
        problem,
        posterior_type,
        stage2_cfg,
        stage2_training,
        seeds=seeds,
        xs=[st.x for st in setups],
        mus=mus,
        zs=[st.z for st in setups],
        mean_nets=ensembles,
        folds=[st.splits for st in setups],
        parallel=parallel_ensemble,
    )


def train_stage2_only(
    problem: Problem,
    n_train: int,
    seeds: list[int],
    stage1_cache_path: str,
    stage2_cfg: Stage2Config,
    stage2_training: TrainingConfig,
    posterior_type: str,
    stage2_mixture: bool,
    parallel_ensemble: bool,
    in_fold_residuals: bool,
) -> list[BaseTrainResult]:
    """Train Stage 2 per seed on the Stage-1 cache at ``stage1_cache_path`` (``__SEED__``
    replaced by the seed), using in-fold means if ``in_fold_residuals``."""
    caches, folds = [], []
    for s in seeds:
        cache = load_cache(
            stage1_cache_path.replace("__SEED__", str(s)),
            problem=problem,
            expected_seed=s,
            expected_n_train=n_train,
            in_fold_residuals=in_fold_residuals,
        )
        caches.append(cache)
        # Seeded here, so a seed's Stage-2 split depends on the seed alone.
        torch.manual_seed(s)
        folds.append(_offset_folds(cache.x.shape[0], stage2_mixture, cache.n_folds))
    return train_stage2(
        problem,
        posterior_type,
        stage2_cfg,
        stage2_training,
        seeds=seeds,
        xs=[c.x for c in caches],
        mus=[c.mu_hat for c in caches],
        zs=[c.z for c in caches],
        mean_nets=[c.ensemble for c in caches],
        folds=folds,
        parallel=parallel_ensemble,
    )


@dataclass(kw_only=True)
class Stage1OnlyResult(BaseTrainResult):
    """Stage-1-only result; ``oof_mse`` is the out-of-fold μ̂ MSE."""

    oof_mse: float

    def extra_metrics(self) -> dict[str, float]:
        result = super().extra_metrics()
        result["oof_mse"] = self.oof_mse
        return result
