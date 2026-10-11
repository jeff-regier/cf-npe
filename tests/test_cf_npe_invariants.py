"""CF-NPE split consistency, OOF residuals, Stage-1 assembly, and the packed (x, mu) layout."""

from typing import cast

import pytest
import torch
from conftest import training_config

from cf_npe.methods import cf_npe as trainers
from cf_npe.networks.models import EnsembleMeanNet, MeanRegressor
from cf_npe.problems.base import EncoderConfig
from cf_npe.problems.simple import Gaussian
from cf_npe.stages.stage1 import (
    _assemble_seed,
    _scatter_means,
    build_mean_regressor,
    run_stage1,
)
from cf_npe.stages.stage2 import Stage2Config
from cf_npe.training.loop import _build_fold_indices
from cf_npe.training.parallel_seed import simulate_seeds

DIM = 5
N_TRAIN = 200
N_EPOCHS = 10
BATCH_SIZE = 64
HIDDEN_DIM = 32
N_FOLDS = 4


@pytest.fixture(scope="module")
def problem():
    return Gaussian(dim=DIM, noise_std=0.5)


@pytest.fixture(scope="module")
def joint_data(problem):
    torch.manual_seed(0)
    x, z = problem.sample_joint(N_TRAIN)
    return x, z


# Split consistency


def test_folds_partition_indices():
    """Validation folds are disjoint and cover every index; each fold's train set is
    the complement of its validation set."""
    n = 103
    folds = _build_fold_indices(n, 5)
    all_val = torch.cat([v for _, v in folds])
    assert torch.equal(all_val.sort().values, torch.arange(n))
    for train_idx, val_idx in folds:
        assert torch.equal(torch.cat([train_idx, val_idx]).sort().values, torch.arange(n))


# OOF residuals


class TestOOFResidualIntegrity:
    """CF-NPE's mu_hat_oof must be genuinely out-of-fold."""

    @pytest.fixture(scope="class")
    def stage1_result(self, problem):
        (setup,) = simulate_seeds(problem, N_TRAIN, [42], lambda n: _build_fold_indices(n, N_FOLDS))
        (s1,) = run_stage1(
            problem,
            [setup],
            n_replicates=1,
            training=training_config(n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, lr=1e-3),
            encoder_cfg=EncoderConfig(hidden_dim=HIDDEN_DIM, n_layers=3, dropout=0.0),
            parallel=True,
        )
        return s1, setup.x, setup.z

    def test_scatter_means_uses_held_out_and_neighbor_fold_predictors(self, problem, joint_data):
        """mu_hat_oof[val_k] is fold k's mini-ensemble prediction on val_k, and
        mu_hat_infold[val_k] is fold (k+1) mod K's, which trained on val_k."""
        x, z = joint_data
        torch.manual_seed(5)
        folds = _build_fold_indices(N_TRAIN, N_FOLDS)
        enc_cfg = EncoderConfig(hidden_dim=HIDDEN_DIM, n_layers=3, dropout=0.0)
        minis = [
            EnsembleMeanNet([build_mean_regressor(problem, enc_cfg, "cuda") for _ in range(2)])
            for _ in folds
        ]
        mu_hat_oof, mu_hat_infold = _scatter_means(minis, folds, x, z)
        with torch.no_grad():
            for k, (_, val_idx) in enumerate(folds):
                assert torch.equal(mu_hat_oof[val_idx], minis[k].get_mean(x[val_idx]))
                neighbor = minis[(k + 1) % N_FOLDS]
                assert torch.equal(mu_hat_infold[val_idx], neighbor.get_mean(x[val_idx]))

    def test_oof_variance_ge_insample(self, stage1_result):
        """OOF residual variance >= in-sample residual variance of the full ensemble."""
        s1, x, z = stage1_result
        mu_hat_oof = s1.mu_hat_oof.detach()
        r_oof = z - mu_hat_oof

        ensemble = s1.ensemble
        ensemble.eval()
        with torch.no_grad():
            mu_hat_insample = ensemble.get_mean(x)
        r_insample = z - mu_hat_insample

        oof_var = r_oof.var(dim=0).mean().item()
        insample_var = r_insample.var(dim=0).mean().item()
        assert oof_var >= insample_var * 0.95, (
            f"OOF variance ({oof_var:.6f}) < 0.95 * in-sample "
            f"variance ({insample_var:.6f}). OOF should be at "
            f"least as noisy as in-sample."
        )


# Ensemble mean


class TestEnsembleMeanNet:
    """EnsembleMeanNet averages its member networks' predictions."""

    def test_ensemble_averages_folds(self, problem, joint_data):
        x, z = joint_data
        torch.manual_seed(10)

        enc_cfg = EncoderConfig(hidden_dim=HIDDEN_DIM, n_layers=3, dropout=0.0)
        m1 = build_mean_regressor(problem, enc_cfg, "cuda")
        m2 = build_mean_regressor(problem, enc_cfg, "cuda")

        ensemble = EnsembleMeanNet([m1, m2])
        ensemble.eval()
        m1.eval()
        m2.eval()

        with torch.no_grad():
            mu_ens = ensemble.get_mean(x[:10])
            mu_1 = m1.get_mean(x[:10])
            mu_2 = m2.get_mean(x[:10])
            expected = (mu_1 + mu_2) / 2

        assert torch.allclose(mu_ens, expected, atol=1e-5), (
            "Ensemble mean doesn't match average of members"
        )


class TestPackUnpackXMu:
    """``pack_xmu`` / ``unpack_xmu`` round-trip."""

    def test_roundtrip_with_mu(self):
        from cf_npe.networks.models import pack_xmu, unpack_xmu

        x = torch.randn(8, 4)
        mu = torch.randn(8, 3)
        packed = pack_xmu(x, mu)
        x_back, mu_back = unpack_xmu(packed, dim_x_raw=4)
        assert torch.equal(x, x_back)
        assert torch.equal(mu, mu_back)


def test_single_fit_stage2_trains_on_the_stage1_split(deterministic_cuda, monkeypatch):
    """The single-fit ablation trains one Stage-1 network on an 80/20 split, and
    Stage 2 trains on that same split, so its training residuals are in-sample."""
    seen = {}
    stage1, stage2 = trainers.run_stage1, trainers.train_stage2

    def record_stage1(problem, setups, **kwargs):
        seen["stage1_folds"] = [st.splits for st in setups]
        return stage1(problem, setups, **kwargs)

    def record_stage2(problem, posterior_type, cfg, training, **kwargs):
        seen["stage2_folds"] = kwargs["folds"]
        seen["n_mean_nets"] = [len(e.models) for e in kwargs["mean_nets"]]
        return stage2(problem, posterior_type, cfg, training, **kwargs)

    monkeypatch.setattr(trainers, "run_stage1", record_stage1)
    monkeypatch.setattr(trainers, "train_stage2", record_stage2)
    trainers.train_cf_npe_single_fit(
        problem=Gaussian(dim=2, noise_std=0.5),
        n_train=200,
        training=training_config(n_epochs=2, batch_size=64, lr=1e-3),
        encoder_cfg=EncoderConfig(hidden_dim=8, n_layers=3, dropout=0.0),
        stage2_cfg=Stage2Config(hidden_dim=8, n_layers=1),
        stage2_training=training_config(n_epochs=2, batch_size=64, lr=1e-3),
        posterior_type="gaussian",
        parallel_ensemble=True,
        seeds=[7, 8],
    )
    assert seen["n_mean_nets"] == [1, 1]
    for s1_folds, s2_folds in zip(seen["stage1_folds"], seen["stage2_folds"], strict=True):
        assert len(s1_folds) == len(s2_folds) == 1
        (tr1, va1), (tr2, va2) = s1_folds[0], s2_folds[0]
        assert (tr1.numel(), va1.numel()) == (160, 40)
        assert torch.equal(tr1, tr2) and torch.equal(va1, va2)


def test_scatter_means_leaves_points_outside_every_validation_fold_nan(problem, joint_data):
    """With one 80/20 split (the single-fit ablation), the training points have no
    out-of-fold mean; they must read as NaN, not as a plausible mean of zero."""
    x, z = joint_data
    torch.manual_seed(3)
    ((train_idx, val_idx),) = _build_fold_indices(N_TRAIN, 5)[:1]
    enc_cfg = EncoderConfig(hidden_dim=HIDDEN_DIM, n_layers=3, dropout=0.0)
    mini = EnsembleMeanNet([build_mean_regressor(problem, enc_cfg, "cuda")])
    mu_hat_oof, mu_hat_infold = _scatter_means([mini], [(train_idx, val_idx)], x, z)
    for mu in (mu_hat_oof, mu_hat_infold):
        assert mu[train_idx].isnan().all()
        assert not mu[val_idx].isnan().any()


class _ConstantMeanNet(torch.nn.Module):
    """Stand-in Stage-1 member whose mean prediction is its own flat index."""

    def __init__(self, index: int, dim_z: int):
        super().__init__()
        self.index = index
        self.dim_z = dim_z

    def get_mean(self, x):
        return torch.full((x.shape[0], self.dim_z), float(self.index), device=x.device)


def test_reps_assembly_pairs_each_fold_with_its_own_replicates(problem):
    """With members ordered (fold, rep), fold k's OOF means average fold k's replicates."""
    n_folds, n_reps = 3, 2
    (setup,) = simulate_seeds(problem, 30, [5], lambda n: _build_fold_indices(n, n_folds))
    nets = [_ConstantMeanNet(i, DIM) for i in range(n_folds * n_reps)]
    s1 = _assemble_seed(setup, cast(list[MeanRegressor], nets), n_reps, [])

    def fold_mean(k):
        return k * n_reps + (n_reps - 1) / 2

    for k, (_, val_idx) in enumerate(setup.splits):
        assert torch.all(s1.mu_hat_oof[val_idx] == fold_mean(k))
        assert torch.all(s1.mu_hat_infold[val_idx] == fold_mean((k + 1) % n_folds))
    with torch.no_grad():
        assert torch.all(s1.ensemble.get_mean(setup.x) == fold_mean(0))
