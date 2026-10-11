import os
import tempfile

import pytest
import torch

from cf_npe.methods.cf_npe import train_cf_npe_stage1_only, train_stage2_only
from cf_npe.problems.base import EncoderConfig
from cf_npe.training.loop import TrainingConfig

# Must be set before CUDA initializes.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@pytest.fixture
def deterministic_cuda():
    """Bit-stable cuBLAS reductions for one test; skips without CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_cudnn_det = torch.backends.cudnn.deterministic
    prev_cudnn_bench = torch.backends.cudnn.benchmark
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    yield
    torch.use_deterministic_algorithms(prev_det)
    torch.backends.cudnn.deterministic = prev_cudnn_det
    torch.backends.cudnn.benchmark = prev_cudnn_bench


def trained_model(problem, n_train=200, n_epochs=15):
    """Train a model enough for meaningful metrics."""
    model = problem.create_posterior(
        EncoderConfig(hidden_dim=32, n_layers=3, dropout=0.0), "gaussian", None
    )
    x, z = problem.sample_joint(n_train)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(n_epochs):
        model.train()
        loss = model.training_loss(x, z).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def training_config(n_epochs: int, batch_size: int, lr: float) -> TrainingConfig:
    """Training settings for a test fit: no weight decay, no gradient clipping."""
    return TrainingConfig(n_epochs, batch_size, lr, weight_decay=0.0, grad_clip=None)


def cf_npe_two_step(
    problem,
    n_train,
    training,
    encoder_cfg,
    stage2_cfg,
    stage2_training,
    k_folds,
    n_replicates,
    seeds,
    posterior_type="gaussian",
):
    """CF-NPE as the Makefile runs it: build the Stage-1 cache, then Stage 2 from it."""
    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/stage1_seed__SEED__.pt"
        train_cf_npe_stage1_only(
            problem=problem,
            n_train=n_train,
            training=training,
            encoder_cfg=encoder_cfg,
            k_folds=k_folds,
            n_replicates=n_replicates,
            parallel_ensemble=True,
            stage1_cache_save=path,
            seeds=seeds,
        )
        return train_stage2_only(
            problem=problem,
            n_train=n_train,
            stage1_cache_path=path,
            stage2_cfg=stage2_cfg,
            stage2_training=stage2_training,
            posterior_type=posterior_type,
            stage2_mixture=False,
            parallel_ensemble=True,
            in_fold_residuals=False,
            seeds=seeds,
        )
