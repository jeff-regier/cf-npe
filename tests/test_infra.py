"""Seed derivation, sweep batching, and recipe loading."""

import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import MissingMandatoryValue, OmegaConf

from cf_npe.problems.simple import Gaussian
from cf_npe.runner.run_job import _job_seeds, _test_data, _test_seed
from cf_npe.runner.run_sweep import (
    _seed_batches,
    merge_recipes,
    parse_set_overrides,
    recipe_overlay,
)
from cf_npe.runner.run_tune import _write_best_method

# run_job._test_seed


def _problem_cfg(dim: int, data_path):
    return OmegaConf.create({"name": "banana", "dim": dim, "data_path": data_path})


def test_test_seed_ignores_data_path():
    assert _test_seed(_problem_cfg(10, None), 0) == _test_seed(_problem_cfg(10, "/tmp/c.pt"), 0)


def test_test_seed_depends_on_seed_and_dim():
    base = _test_seed(_problem_cfg(10, None), 0)
    assert base != _test_seed(_problem_cfg(10, None), 1)
    assert base != _test_seed(_problem_cfg(20, None), 0)


# run_sweep._seed_batches


@pytest.mark.parametrize(
    ("base", "n_seeds", "seeds_per_job"),
    [(0, 10, 3), (0, 9, 3), (7, 11, 4), (0, 1, 5), (100, 5, 1)],
)
def test_seed_batches_partition(base, n_seeds, seeds_per_job):
    seeds = list(range(base, base + n_seeds))
    batches = _seed_batches(seeds, seeds_per_job)
    rebuilt = [s for b, size in batches for s in range(b, b + size)]
    assert rebuilt == seeds
    assert sum(size for _, size in batches) == n_seeds
    for _, size in batches[:-1]:
        assert size == seeds_per_job
    assert batches[-1][1] <= seeds_per_job
    bases = [b for b, _ in batches]
    assert bases == sorted(bases)
    for (b0, s0), (b1, _) in zip(batches, batches[1:], strict=False):
        assert b1 == b0 + s0


# run_tune._write_best_method / run_sweep._method_overlay


def test_best_method_roundtrips_into_overlay(tmp_path):
    opt = {
        "best_value": -1.234,
        "best_params": {
            "train_params.stage2_cfg.hidden_dim": 128,
            "train_params.stage2_training.lr": 0.001,
        },
    }
    opt_path = tmp_path / "optimization_results.yaml"
    opt_path.write_text(yaml.safe_dump(opt))
    _write_best_method(opt_path, tmp_path / "best_method.yaml", "tune_demo")

    text = (tmp_path / "best_method.yaml").read_text()
    assert "tune_demo" in text and "-1.234" in text
    assert recipe_overlay(str(tmp_path)) == opt["best_params"]


def testparse_set_overrides_splits_on_first_equals():
    assert parse_set_overrides([]) == {}
    assert parse_set_overrides(["train_params.loss=beta_nll"]) == {"train_params.loss": "beta_nll"}
    # value may itself contain '=' (e.g. an interpolation); only the first splits
    assert parse_set_overrides(["a.b=x=y"]) == {"a.b": "x=y"}


# run_job: seed batches and tuning-trial failures


def _job_cfg(seeds_per_job, split_seed):
    return OmegaConf.create(
        {"seed": 4, "seeds_per_job": seeds_per_job, "problem": {"split_seed": split_seed}}
    )


def test_seed_batch_rejects_seed_interpolation():
    """A seed interpolation, however spelled, would resolve once from the batch's base seed."""
    for interpolation in ("${seed}", "${..seed}", "${oc.select:seed,0}"):
        with pytest.raises(ValueError, match="interpolates"):
            _job_seeds(_job_cfg(2, interpolation))
    assert _job_seeds(_job_cfg(1, "${seed}")) == [4]
    assert _job_seeds(_job_cfg(2, 3)) == [4, 5]


_REPO = Path(__file__).resolve().parents[1]
# Fails in run_job before any training: the trainer rejects the unknown key.
_FAILING_OVERRIDE = "++train_params.not_a_trainer_arg=1"


def _run_job_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "cf_npe.runner.run_job", *args],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )


def test_failed_tuning_trial_fails_the_study(tmp_path):
    """Under the Optuna sweeper a trial that raises stops the study with its traceback."""
    proc = _run_job_cli(
        [
            "--config-name=sweep/tune_stage2_std_npe_gaussian",
            "-m",
            f"hydra.sweep.dir={tmp_path / 'multirun'}",
            "hydra.sweeper.n_trials=1",
            f"+out_dir={tmp_path}",
            "++problem.dim=2",
            _FAILING_OVERRIDE,
        ]
    )
    assert proc.returncode != 0
    assert "not_a_trainer_arg" in proc.stderr


def test_diverged_tuning_trial_scores_worst(tmp_path):
    """A trial whose validation loss goes non-finite scores +inf and the study completes."""
    proc = _run_job_cli(
        [
            "--config-name=sweep/tune_stage2_std_npe_gaussian",
            "-m",
            f"hydra.sweep.dir={tmp_path / 'multirun'}",
            "hydra.sweeper.n_trials=1",
            f"+out_dir={tmp_path}",
            "scale=smoke",
            "++problem.dim=2",
            "++problem.noise_std=nan",
        ]
    )
    assert proc.returncode == 0, proc.stderr
    results = yaml.safe_load((tmp_path / "multirun" / "optimization_results.yaml").read_text())
    assert results["best_value"] == math.inf


def test_run_tune_refuses_recipe_when_no_trial_succeeds(tmp_path):
    """A study whose every trial failed must not produce a recipe."""
    (tmp_path / "opt.yaml").write_text("name: optuna\nbest_params: {lr: 0.1}\nbest_value: .inf\n")
    with pytest.raises(SystemExit, match="no trial succeeded"):
        _write_best_method(tmp_path / "opt.yaml", tmp_path / "best_method.yaml", "s")
    assert not (tmp_path / "best_method.yaml").exists()


def test_failed_job_outside_tuning_raises(tmp_path):
    """Outside a tuning sweep the same failure propagates and the job exits nonzero."""
    proc = _run_job_cli(
        [
            "+experiment=tune_stage2_std_npe_gaussian",
            "scale=smoke",
            "seeds_per_job=1",
            "++problem.dim=2",
            "train_params.training.batch_size=16",
            f"hydra.run.dir={tmp_path / 'job'}",
            f"+out_dir={tmp_path}",
            _FAILING_OVERRIDE,
        ]
    )
    assert proc.returncode != 0
    assert "unexpected keyword argument 'not_a_trainer_arg'" in proc.stderr
    assert "Tuning trial failed" not in proc.stderr


def test_test_data_depends_on_seed_and_differs_from_training_data():
    """Test pairs are reproducible, seed-dependent, and distinct from the training pairs."""
    problem = Gaussian(dim=3, noise_std=0.5)
    problem_cfg = OmegaConf.create({"_target_": "cf_npe.problems.Gaussian", "dim": 3})
    x_a, z_a = _test_data(problem, problem_cfg, 50, 4)
    x_b, z_b = _test_data(problem, problem_cfg, 50, 4)
    assert torch.equal(x_a, x_b) and torch.equal(z_a, z_b)
    x_next, _ = _test_data(problem, problem_cfg, 50, 5)
    assert not torch.equal(x_a, x_next)
    torch.manual_seed(4)
    x_train, _ = problem.sample_joint(50)
    assert not torch.equal(x_a, x_train)


def test_merge_recipes_rejects_a_key_set_twice():
    """Recipes applied together (e.g. Stage 1 and Stage 2) must set disjoint keys."""
    stage1, stage2 = {"train_params.training.lr": 1e-3}, {"train_params.stage2_training.lr": 2e-3}
    assert merge_recipes([stage1, stage2]) == {**stage1, **stage2}
    with pytest.raises(ValueError, match="train_params.training.lr"):
        merge_recipes([stage1, {"train_params.training.lr": 2e-3}])


def test_recipe_overlay_selects_subtrees(tmp_path):
    """``DIR:PREFIX,...`` keeps only keys under the prefixes; an empty selection is an error."""
    (tmp_path / "best_method.yaml").write_text(
        "problem.encoder.width: 64\nproblem.head.width: 8\ntrain_params.training.lr: 0.1\n"
    )
    assert recipe_overlay(f"{tmp_path}:problem.encoder,train_params") == {
        "problem.encoder.width": 64,
        "train_params.training.lr": 0.1,
    }
    with pytest.raises(ValueError, match="no keys under 'problem.decoder'"):
        recipe_overlay(f"{tmp_path}:problem.decoder")


@pytest.mark.parametrize(
    ("experiment", "key"),
    [
        ("gg_ntrain_std_npe", "n_train"),
        ("cf_npe_build_cache", "train_params.training.batch_size"),
        ("cf_npe_build_cache", "problem.dim"),
        ("std_npe_test_ll_gaussian_dimsweep", "problem.dim"),
        ("std_npe_test_ll_mixture_dimsweep_nsf", "problem.dim"),
    ],
)
def test_required_value_has_no_default(experiment, key):
    """A ??? the launcher must fill is not satisfied by a conf/ default under it."""
    with initialize_config_dir(config_dir=str(_REPO / "conf"), version_base=None):
        cfg = compose("config", overrides=[f"+experiment={experiment}", "problem=gaussian"])
    with pytest.raises(MissingMandatoryValue):
        OmegaConf.select(cfg, key, throw_on_missing=True)
