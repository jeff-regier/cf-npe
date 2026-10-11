"""Training several seeds in one vmap call: equivalence to single-seed runs, and no
cross-seed leak of data or dropout randomness."""

from dataclasses import dataclass
from typing import Any

import torch
from conftest import cf_npe_two_step, training_config
from torch import Tensor

from cf_npe.methods.std_npe import train_std_npe, train_std_npe_mixture
from cf_npe.networks.models import EnsemblePosterior
from cf_npe.problems.base import EncoderConfig, Problem
from cf_npe.problems.simple import Gaussian
from cf_npe.stages.stage2 import Stage2Config
from cf_npe.training.parallel_seed import build_seeded, combine_init_seed


def _train_kwargs(problem: Problem) -> dict[str, Any]:
    """Small, fast hyperparams for the multi-seed test problems."""
    return {
        "problem": problem,
        "n_train": 256,
        "training": training_config(n_epochs=8, batch_size=64, lr=1e-3),
        "encoder_cfg": EncoderConfig(hidden_dim=16, n_layers=2, dropout=0.0),
        "posterior_type": "gaussian",
        "flow_cfg": None,
    }


def _mean_log_prob(model, x: Tensor, z: Tensor) -> float:
    """Mean per-sample log p(z|x) under the trained posterior."""
    model.eval()
    with torch.no_grad():
        return float(model.log_prob(x, z).mean().item())


def test_train_std_npe_multi_seed_matches_sequential(deterministic_cuda):
    """Each seed's test LL matches its single-seed run after convergence.

    Batch orders differ between the vmap and single-seed paths, so trajectories differ.
    """
    seeds = [11, 22, 33]
    problem = Gaussian(dim=2, noise_std=0.5)
    # Trained to convergence so the two paths' different batch orders barely matter.
    kwargs: dict[str, Any] = {
        "problem": problem,
        "n_train": 1024,
        "training": training_config(n_epochs=80, batch_size=128, lr=1e-3),
        "encoder_cfg": EncoderConfig(hidden_dim=32, n_layers=2, dropout=0.0),
        "posterior_type": "gaussian",
        "flow_cfg": None,
        "loss": "standard",
        "parallel_ensemble": True,
    }

    sequential = [train_std_npe(**kwargs, seeds=[s])[0] for s in seeds]
    parallel = train_std_npe(**kwargs, seeds=seeds)
    assert len(parallel) == len(seeds)

    torch.manual_seed(99)
    x_test, z_test = problem.sample_joint(512)

    for s, (seq, par) in enumerate(zip(sequential, parallel, strict=True)):
        ll_seq = _mean_log_prob(seq.model, x_test, z_test)
        ll_par = _mean_log_prob(par.model, x_test, z_test)
        denom = max(abs(ll_seq), 1.0)
        rel = abs(ll_par - ll_seq) / denom
        assert rel < 0.06, (
            f"seed={seeds[s]}: sequential test_ll={ll_seq:.4f} "
            f"parallel test_ll={ll_par:.4f} rel_err={rel:.4f}"
        )


@dataclass
class _PerSeedConstantZGaussian(Gaussian):
    """The i-th call to sample_joint returns z = targets[i] everywhere, x ~ N(z, sigma^2)."""

    targets: tuple[float, ...] = (0.0,)

    def __post_init__(self):
        self._counter = [0]

    def sample_joint(self, n: int) -> tuple[Tensor, Tensor]:
        idx = self._counter[0]
        if idx >= len(self.targets):
            raise RuntimeError(
                f"sample_joint called {idx + 1} times but only "
                f"{len(self.targets)} targets configured"
            )
        self._counter[0] = idx + 1
        target = self.targets[idx]
        z = torch.full((n, self.dim_z), float(target), device="cuda")
        x = self.sample_likelihood(z)
        return x, z


def test_train_std_npe_no_cross_seed_data_leak(deterministic_cuda):
    """Each seed's model predicts its own constant target, not the cross-seed mean."""
    targets = (0.0, 1.0, 2.0, 3.0)
    seeds = list(range(len(targets)))
    problem = _PerSeedConstantZGaussian(dim=1, noise_std=0.5, targets=targets)
    parallel = train_std_npe(
        problem=problem,
        n_train=256,
        training=training_config(n_epochs=80, batch_size=64, lr=1e-2),
        encoder_cfg=EncoderConfig(hidden_dim=16, n_layers=2, dropout=0.0),
        posterior_type="gaussian",
        flow_cfg=None,
        loss="standard",
        parallel_ensemble=True,
        seeds=seeds,
    )

    cross_mean = sum(targets) / len(targets)
    predicted = []
    for s, target in enumerate(targets):
        model = parallel[s].model
        model.eval()
        # x from this seed's own training distribution, x ~ N(target, noise_std^2).
        torch.manual_seed(7777 + s)
        x_eval = torch.randn(256, problem.dim_x, device="cuda") * problem.noise_std + target
        with torch.no_grad():
            mean = model.get_mean(x_eval).mean().item()
        predicted.append(mean)
        own_err = abs(mean - target)
        other_mean = (sum(targets) - target) / (len(targets) - 1)
        err_to_other_mean = abs(mean - other_mean)
        assert own_err < 0.2, (
            f"seed={s}, target={target}: predicted mean={mean:.3f}, "
            f"err={own_err:.3f}. cross-seed mean={cross_mean:.2f}; data "
            f"leak across the vmap seed axis suspected."
        )
        assert own_err < 0.5 * max(err_to_other_mean, 0.1), (
            f"seed={s}: own_err={own_err:.3f} not less than half of "
            f"err_to_other_mean={err_to_other_mean:.3f}; predictions look "
            f"like a mixture of seeds."
        )
    assert max(predicted) - min(predicted) > 0.5 * (max(targets) - min(targets)), (
        f"all seeds collapsed to similar predictions {predicted}; "
        f"expected span comparable to target range {targets}"
    )


def test_train_std_npe_dropout_independent_across_seeds(deterministic_cuda):
    """Two identical seeds with dropout > 0 diverge: each vmap row draws its own mask."""
    # Identical seeds give identical data and init, leaving dropout as the only difference.
    problem = Gaussian(dim=2, noise_std=0.5)
    kwargs = _train_kwargs(problem)
    kwargs["training"] = training_config(n_epochs=20, batch_size=64, lr=1e-3)
    kwargs["encoder_cfg"] = EncoderConfig(hidden_dim=16, n_layers=2, dropout=0.3)

    parallel = train_std_npe(**kwargs, loss="standard", parallel_ensemble=True, seeds=[42, 42])
    assert len(parallel) == 2

    p0 = torch.cat([p.detach().flatten() for p in parallel[0].model.parameters()])
    p1 = torch.cat([p.detach().flatten() for p in parallel[1].model.parameters()])
    diff = (p0 - p1).abs().max().item()
    assert diff > 1e-4, (
        f"two seeds with identical data and identical init produced "
        f"bit-identical models (max abs param diff={diff:.2e}); dropout "
        f"masks are sharing across the seed axis"
    )


# CF-NPE


def _cf_npe_kwargs() -> dict[str, Any]:
    """Small, fast hyperparams shared by the cf_npe multi-seed tests."""
    return {
        "n_train": 256,
        "training": training_config(n_epochs=8, batch_size=64, lr=1e-3),
        "encoder_cfg": EncoderConfig(hidden_dim=16, n_layers=2, dropout=0.0),
        "k_folds": 3,
        "n_replicates": 1,
        "stage2_cfg": Stage2Config(hidden_dim=32),
        "stage2_training": training_config(n_epochs=30, batch_size=64, lr=1e-3),
    }


def test_cf_npe_multi_seed_matches_sequential(deterministic_cuda):
    """Each seed's CF-NPE test LL matches its single-seed run after convergence."""
    seeds = [11, 22, 33]
    problem = Gaussian(dim=2, noise_std=0.5)
    kwargs = _cf_npe_kwargs()
    kwargs["training"] = training_config(n_epochs=20, batch_size=64, lr=5e-3)
    kwargs["stage2_training"] = training_config(n_epochs=80, batch_size=64, lr=1e-3)

    sequential = [cf_npe_two_step(problem=problem, seeds=[s], **kwargs)[0] for s in seeds]
    parallel = cf_npe_two_step(problem=problem, seeds=seeds, **kwargs)
    assert len(parallel) == len(seeds)

    torch.manual_seed(99)
    x_test, z_test = problem.sample_joint(512)
    for s, (seq, par) in enumerate(zip(sequential, parallel, strict=True)):
        ll_seq = _mean_log_prob(seq.model, x_test, z_test)
        ll_par = _mean_log_prob(par.model, x_test, z_test)
        denom = max(abs(ll_seq), 1.0)
        rel = abs(ll_par - ll_seq) / denom
        # Looser than train_std_npe: both stages shuffle differently from single-seed runs.
        assert rel < 0.15, (
            f"seed={seeds[s]}: sequential test_ll={ll_seq:.4f} "
            f"parallel test_ll={ll_par:.4f} rel_err={rel:.4f}"
        )


def test_cf_npe_stage2_uses_matching_seed_stage1(deterministic_cuda):
    """Each seed's Stage 2 pairs with its own Stage 1: the model predicts its own target."""
    targets = (0.0, 2.0, 4.0)
    seeds = list(range(len(targets)))
    problem = _PerSeedConstantZGaussian(dim=1, noise_std=0.5, targets=targets)
    parallel = cf_npe_two_step(
        problem=problem,
        n_train=256,
        training=training_config(n_epochs=40, batch_size=64, lr=1e-2),
        encoder_cfg=EncoderConfig(hidden_dim=16, n_layers=2, dropout=0.0),
        k_folds=3,
        n_replicates=1,
        stage2_cfg=Stage2Config(hidden_dim=32),
        stage2_training=training_config(n_epochs=60, batch_size=64, lr=1e-3),
        seeds=seeds,
    )

    for s, target in enumerate(targets):
        model = parallel[s].model
        model.eval()
        torch.manual_seed(7777 + s)
        x_eval = torch.randn(256, problem.dim_x, device="cuda") * problem.noise_std + target
        with torch.no_grad():
            mean = model.get_mean(x_eval).mean().item()
        own_err = abs(mean - target)
        other_targets = [t for t in targets if t != target]
        other_mean = sum(other_targets) / len(other_targets)
        err_to_other_mean = abs(mean - other_mean)
        assert own_err < 0.4, (
            f"seed={s}, target={target}: CF-NPE predicted mean={mean:.3f}, "
            f"err={own_err:.3f}. Suggests Stage 1 -> Stage 2 wiring crosses "
            f"seeds (mean_net paired with wrong-seed offset)."
        )
        assert own_err < 0.5 * max(err_to_other_mean, 0.1), (
            f"seed={s}: own_err={own_err:.3f} not less than half of "
            f"err_to_other_mean={err_to_other_mean:.3f}; predictions "
            f"look like a mixture of seeds."
        )


def test_train_std_npe_mixture_multi_seed_matches_sequential(deterministic_cuda):
    """Each seed's Std-NPE-mixture test LL matches its single-seed run."""
    seeds = [11, 22, 33]
    problem = Gaussian(dim=2, noise_std=0.5)
    kwargs: dict[str, Any] = {
        "n_train": 1024,
        "training": training_config(n_epochs=40, batch_size=128, lr=5e-3),
        "encoder_cfg": EncoderConfig(hidden_dim=32, n_layers=2, dropout=0.0),
        "n_components": 3,
        "posterior_type": "gaussian",
        "split_mode": "shared",
        "flow_cfg": None,
    }

    sequential = [
        train_std_npe_mixture(problem=problem, seeds=[s], parallel_ensemble=False, **kwargs)[0]
        for s in seeds
    ]
    parallel = train_std_npe_mixture(problem=problem, seeds=seeds, parallel_ensemble=True, **kwargs)
    assert len(parallel) == len(seeds)

    torch.manual_seed(99)
    x_test, z_test = problem.sample_joint(1024)
    for s, (seq, par) in enumerate(zip(sequential, parallel, strict=True)):
        ll_seq = _mean_log_prob(seq.model, x_test, z_test)
        ll_par = _mean_log_prob(par.model, x_test, z_test)
        denom = max(abs(ll_seq), 1.0)
        rel = abs(ll_par - ll_seq) / denom
        assert rel < 0.10, (
            f"seed={seeds[s]}: sequential test_ll={ll_seq:.4f} "
            f"parallel test_ll={ll_par:.4f} rel_err={rel:.4f}"
        )


def test_member_inits_distinct_across_seeds():
    """Component k of two different seeds starts from different weights."""
    problem = Gaussian(dim=2, noise_std=0.5)
    enc_cfg = EncoderConfig(hidden_dim=8, n_layers=2, dropout=0.0)
    n_components = 3
    modules = build_seeded(
        lambda _s, _k: problem.create_posterior(enc_cfg, "gaussian", None), [11, 22], n_components
    )
    for k in range(n_components):
        p0 = next(modules[k].parameters()).detach().flatten()
        p1 = next(modules[n_components + k].parameters()).detach().flatten()
        assert not torch.equal(p0, p1), f"component k={k}: seeds 11 and 22 share initial weights"


def test_train_std_npe_mixture_partition_multi_seed_matches_single_seed(deterministic_cuda):
    """Partition mode trains the same mixture for a seed alone or batched with others."""
    seeds = [3, 4]
    problem = Gaussian(dim=2, noise_std=0.5)
    kwargs: dict[str, Any] = {
        **_train_kwargs(problem),
        "n_components": 4,
        "split_mode": "partition",
        "parallel_ensemble": True,
    }
    parallel = train_std_npe_mixture(seeds=seeds, **kwargs)
    torch.manual_seed(99)
    x_test, z_test = problem.sample_joint(1024)
    for s, par in zip(seeds, parallel, strict=True):
        (alone,) = train_std_npe_mixture(seeds=[s], **kwargs)
        assert isinstance(par.model, EnsemblePosterior)
        assert len(par.model.members) == 4
        ll_alone = _mean_log_prob(alone.model, x_test, z_test)
        ll_par = _mean_log_prob(par.model, x_test, z_test)
        assert abs(ll_par - ll_alone) / max(abs(ll_alone), 1.0) < 0.10


def test_combine_init_seed_distinct_and_identity():
    seen = {}
    for outer in range(0, 1000):
        for member in range(0, 64):
            v = combine_init_seed(outer, member)
            assert v not in seen, f"collision {(outer, member)} vs {seen.get(v)}"
            seen[v] = (outer, member)
    for k in range(0, 64):
        assert combine_init_seed(0, k) == k
    assert all(0 <= v < 2**31 for v in seen)
    assert len(seen) == 1000 * 64
    assert combine_init_seed(3, 7) == (3 * 1_000_003 + 7) & 0x7FFFFFFF
