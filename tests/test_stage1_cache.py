"""Stage-1 cache round-trip and mismatch guards."""

import pytest
import torch
from conftest import training_config

from cf_npe.problems.base import EncoderConfig
from cf_npe.problems.simple import Funnel, Gaussian
from cf_npe.stages.cache import load_cache, save_cache
from cf_npe.stages.stage1 import run_stage1
from cf_npe.training.loop import _build_fold_indices
from cf_npe.training.parallel_seed import simulate_seeds

DIM = 5
N_TRAIN = 200
N_EPOCHS = 8
BATCH_SIZE = 64
HIDDEN_DIM = 32
N_FOLDS = 4
N_REPLICATES = 3
LR = 1e-3


@pytest.fixture(scope="module")
def problem():
    return Gaussian(dim=DIM, noise_std=0.5)


@pytest.fixture(scope="module")
def encoder_cfg():
    return EncoderConfig(hidden_dim=HIDDEN_DIM, n_layers=3, dropout=0.0)


@pytest.fixture(scope="module")
def stage1_reps(problem, encoder_cfg):
    """K*M Stage-1 result."""
    (setup,) = simulate_seeds(problem, N_TRAIN, [0], lambda n: _build_fold_indices(n, N_FOLDS))
    (s1,) = run_stage1(
        problem,
        [setup],
        n_replicates=N_REPLICATES,
        training=training_config(n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, lr=LR),
        encoder_cfg=encoder_cfg,
        parallel=True,
    )
    return setup, s1


class TestStage1CacheRoundTrip:
    """save_cache then load_cache reproduces the test-time ensemble."""

    def test_reloaded_ensemble_matches_in_memory(self, tmp_path, problem, encoder_cfg, stage1_reps):
        setup, s1 = stage1_reps
        path = tmp_path / "stage1.pt"
        save_cache(path, problem, setup, s1, encoder_cfg)

        s1.ensemble.eval()
        with torch.no_grad():
            expected = s1.ensemble.get_mean(setup.x)

        loaded = load_cache(path, problem, setup.seed, N_TRAIN, False)
        loaded.ensemble.eval()
        with torch.no_grad():
            got = loaded.ensemble.get_mean(loaded.x)

        assert torch.allclose(got, expected, atol=1e-6), (
            "reloaded ensemble mean differs from in-memory s1.ensemble"
        )

    def test_tensors_round_trip_exactly(self, tmp_path, problem, encoder_cfg, stage1_reps):
        setup, s1 = stage1_reps
        path = tmp_path / "stage1.pt"
        save_cache(path, problem, setup, s1, encoder_cfg)

        loaded = load_cache(path, problem, setup.seed, N_TRAIN, False)
        assert torch.equal(loaded.x, setup.x)
        assert torch.equal(loaded.z, setup.z)
        assert torch.equal(loaded.mu_hat, s1.mu_hat_oof)
        assert loaded.n_folds == N_FOLDS
        in_fold = load_cache(path, problem, setup.seed, N_TRAIN, True)
        assert torch.equal(in_fold.mu_hat, s1.mu_hat_infold)


class TestLoadCacheGuards:
    """load_cache rejects a cache built for a different recipe."""

    @pytest.fixture(scope="class")
    def cache_path(self, tmp_path_factory, problem, encoder_cfg, stage1_reps):
        setup, s1 = stage1_reps
        path = tmp_path_factory.mktemp("guard") / "stage1.pt"
        save_cache(path, problem, setup, s1, encoder_cfg)
        return path

    def test_seed_mismatch_raises(self, cache_path, problem):
        with pytest.raises(ValueError, match="seed"):
            load_cache(cache_path, problem, 999, N_TRAIN, False)

    def test_n_train_mismatch_raises(self, cache_path, problem):
        with pytest.raises(ValueError, match="n_train"):
            load_cache(cache_path, problem, 0, N_TRAIN + 1, False)


def test_cache_rejects_another_problem(tmp_path, problem, encoder_cfg, stage1_reps):
    """A Gaussian cache does not load into a funnel Stage 2 of the same dimension."""
    setup, s1 = stage1_reps
    path = tmp_path / "stage1.pt"
    save_cache(path, problem, setup, s1, encoder_cfg)
    with pytest.raises(ValueError, match="cache problem"):
        load_cache(path, Funnel(dim=DIM, scale=1.0, noise_std=0.1), setup.seed, N_TRAIN, False)
