"""Hydra entry point for one (experiment, grid point, seed batch) job.

Trains and evaluates one model per seed. Launched by ``run-sweep``; can also be run
directly::

    python -m cf_npe.runner.run_job +experiment=gg_temperature scale=smoke +out_dir=out/runs/x seed=0
"""

import hashlib
import json
import logging
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from hydra.utils import get_method, instantiate
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from cf_npe import CONF
from cf_npe.problems.base import Problem
from cf_npe.training import amp
from cf_npe.training.loop import BaseTrainResult, NonFiniteLossError

logger = logging.getLogger(__name__)

TrainFn = Callable[..., list[BaseTrainResult]]


def _configure_cuda(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    amp.set_enabled(bool(cfg.use_amp))


def _warm_cuda_for_timing(seconds: float) -> None:
    """Warm CUDA clocks and libraries before a wall-clock benchmark."""
    if seconds <= 0.0:
        return
    logger.info("Warming CUDA for %.1f s before timed training", seconds)
    with torch.inference_mode():
        left = torch.ones((2048, 2048), device="cuda")
        right = torch.ones_like(left)
        product = torch.empty_like(left)
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            torch.mm(left, right, out=product)
            torch.cuda.synchronize()
    torch.cuda.synchronize()


def _to_dict(cfg: DictConfig) -> dict[str, Any]:
    return cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))


_OPTUNA_SWEEPER = "hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper"


def _is_tuning_trial() -> bool:
    """Whether this job is one trial of a Hydra-Optuna sweep (``run-tune``)."""
    hydra_cfg = HydraConfig.get()
    return hydra_cfg.mode == RunMode.MULTIRUN and hydra_cfg.sweeper._target_ == _OPTUNA_SWEEPER


def _extract_metadata(cfg: DictConfig) -> dict[str, Any]:
    """Config fields copied into every record, ``train_params`` and ``problem`` flattened
    to the top level."""
    problem = _to_dict(cfg.problem)
    del problem["_target_"]
    return {
        "config_id": cfg.config_id,
        "n_train": cfg.n_train,
        "seeds_per_job": int(cfg.seeds_per_job),
        "timing_warmup_seconds": float(cfg.timing_warmup_seconds),
        **_to_dict(cfg.train_params),
        **problem,
    }


def _test_seed(problem_cfg: DictConfig, seed: int) -> int:
    """Hash of the problem config (without ``data_path``) and run seed, so every method
    run on one problem and seed is scored on the same test pairs."""
    fields = _to_dict(problem_cfg)
    fields.pop("data_path", None)
    text = OmegaConf.to_yaml(OmegaConf.create(fields))
    digest = hashlib.sha256(f"{text}|{seed}".encode()).hexdigest()
    return int(digest[:8], 16) % (2**31)


def _test_data(
    problem: Problem, problem_cfg: DictConfig, n_test: int, seed: int
) -> tuple[Tensor, Tensor]:
    torch.manual_seed(_test_seed(problem_cfg, seed))
    x_test, z_test = problem.sample_joint(n_test)
    return x_test.cuda(), z_test.cuda()


def _prepare_train_params(cfg: DictConfig) -> dict[str, Any]:
    """Resolve train_params, instantiating any ``_target_`` entries."""
    return instantiate(cfg.train_params, _convert_="all")


def _maybe_save_model(cfg: DictConfig, seed: int, result: BaseTrainResult) -> None:
    """Save the model's state dict, resolved train_params and problem, and n_train when
    ``cfg.save_model``."""
    if not cfg.save_model:
        return
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in result.model.state_dict().items()},
        "train_params": _to_dict(cfg.train_params),
        "problem": _to_dict(cfg.problem),
        "n_train": int(cfg.n_train),
    }
    out = Path(str(cfg.out_dir)) / "models" / f"seed{seed}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    logger.info("Model saved to %s", out)


def _resolved_without_seed(cfg: DictConfig, seed: int) -> dict[str, Any]:
    """``cfg`` resolved with ``seed`` in place of ``cfg.seed``, minus the seed itself."""
    resolved = _to_dict(cast(DictConfig, OmegaConf.merge(cfg, {"seed": seed})))
    del resolved["seed"]
    return resolved


def _job_seeds(cfg: DictConfig) -> list[int]:
    """Seeds ``[cfg.seed, cfg.seed + seeds_per_job)``. A multi-seed batch cannot honor a
    field that interpolates the seed: it resolves once, from the base seed."""
    seeds = list(range(int(cfg.seed), int(cfg.seed) + int(cfg.seeds_per_job)))
    if len(seeds) > 1 and _resolved_without_seed(cfg, seeds[0]) != _resolved_without_seed(
        cfg, seeds[1]
    ):
        raise ValueError(
            "seeds_per_job>1 but the config interpolates the seed, which would resolve "
            "once from the base seed for the whole batch; set seeds_per_job=1"
        )
    return seeds


def _run_job(cfg: DictConfig) -> list[tuple[int, dict]]:
    """Train all the job's seeds in one call and evaluate each; return (seed, metrics).

    ``training_wall_time`` is that call's wall time divided evenly across the seeds, so
    it measures one seed's cost only when ``seeds_per_job == 1``.
    """
    seeds = _job_seeds(cfg)
    train_fn: TrainFn = get_method(cfg.train_fn._target_)
    problem = instantiate(cfg.problem)
    evaluator = instantiate(cfg.evaluator)
    if cfg.eval_problem is None:
        eval_cfg, eval_problem = cfg.problem, problem
    else:
        eval_cfg, eval_problem = cfg.eval_problem, instantiate(cfg.eval_problem)
    train_params = _prepare_train_params(cfg)
    test_data = [_test_data(eval_problem, eval_cfg, cfg.n_test, s) for s in seeds]

    _warm_cuda_for_timing(float(cfg.timing_warmup_seconds))
    torch.cuda.synchronize()
    t0 = time.monotonic()
    results = train_fn(problem=problem, n_train=cfg.n_train, seeds=seeds, **train_params)
    torch.cuda.synchronize()
    per_seed_wall = (time.monotonic() - t0) / len(seeds)

    out = []
    for seed, result, (x_test, z_test) in zip(seeds, results, test_data, strict=True):
        metrics = evaluator.compute(
            result.model, eval_problem, x_test, z_test, batch_size=cfg.eval_batch_size
        )
        metrics["training_wall_time"] = per_seed_wall
        metrics.update(result.extra_metrics())
        _maybe_save_model(cfg, seed, result)
        out.append((seed, metrics))
    return out


def _write_results(metadata: dict[str, Any], seed_results: list[tuple[int, dict]]) -> None:
    """Write one ``seed{s}/result.json`` per seed; a job dir holds every seed of its
    config_id, so the per-seed subdir keeps their records apart."""
    for seed, metrics in seed_results:
        out_path = Path(f"seed{seed}") / "result.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({**metadata, "seed": seed, **metrics}, f, indent=2)
        logger.info("Result written to %s", out_path)


def _run_tuning_trial(cfg: DictConfig, metadata: dict[str, Any]) -> float:
    """Run one Optuna trial and return the mean of ``cfg.optuna_objective`` over seeds.

    A diverged configuration (non-finite validation loss in training, or a non-finite
    objective) scores the worst value, signed infinity, so the study continues past it.
    """
    worst = -math.inf if HydraConfig.get().sweeper.direction == "maximize" else math.inf
    try:
        seed_results = _run_job(cfg)
    except NonFiniteLossError:
        logger.exception("Training diverged; scoring the trial %s", worst)
        return worst
    _write_results(metadata, seed_results)
    values = [metrics[cfg.optuna_objective] for _, metrics in seed_results]
    val = sum(values) / len(values)
    if not math.isfinite(val):
        logger.warning("Non-finite optuna_objective %s; scoring the trial %s", val, worst)
        return worst
    return val


@hydra.main(
    version_base=None,
    config_path=str(CONF),
    config_name="config",
)
def main(cfg: DictConfig) -> float | None:
    """Run one training + evaluation job; under a Hydra-Optuna sweep, return the trial's
    objective."""
    _configure_cuda(cfg)
    logger.info(
        "Job: seed=%d n_train=%d seeds_per_job=%d use_amp=%s",
        cfg.seed,
        cfg.n_train,
        int(cfg.seeds_per_job),
        amp.ENABLED,
    )

    metadata = _extract_metadata(cfg)
    if _is_tuning_trial():
        return _run_tuning_trial(cfg, metadata)
    _write_results(metadata, _run_job(cfg))
    return None


if __name__ == "__main__":
    main()
