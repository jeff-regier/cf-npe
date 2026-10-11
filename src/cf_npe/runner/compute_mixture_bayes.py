"""Bayes-optimal test LL and posterior-mean MSE for the Mixture dimension sweep.

Monte-Carlo averages of the exact posterior density and mean over joint draws (d_x = 2 d_z).
"""

import argparse
import json
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from cf_npe import CONF
from cf_npe.runner.pipeline import NSF_DIMS

_N_SAMPLES = 200_000


def _posterior_mean(problem, x):
    """Analytic mean of the two-component Gaussian posterior."""
    log_weights = problem._component_log_evidence(x)
    weights = torch.softmax(log_weights, dim=-1)
    centered = torch.stack((x - problem.s, x + problem.s), dim=1)
    flat = centered.reshape(-1, problem.dim_x)
    solved = torch.cholesky_solve(flat.unsqueeze(-1), problem._x_cov_chol).squeeze(-1)
    component_means = (solved @ problem.A).reshape(-1, 2, problem.dim_z)
    return (weights.unsqueeze(-1) * component_means).sum(dim=1)


def _bayes_metrics(dz):
    torch.manual_seed(0)
    problem = instantiate(OmegaConf.load(CONF / "problem" / "mixture.yaml"), dim=dz)
    x, z = problem.sample_joint(_N_SAMPLES)
    return {
        "log_likelihood": float(problem.posterior_log_prob(x, z).mean()),
        "posterior_mean_mse": float((z - _posterior_mean(problem, x)).pow(2).mean()),
    }


def run(root: Path) -> None:
    metrics = {dz: _bayes_metrics(dz) for dz in NSF_DIMS}
    path = root / "results" / "mixture_dimsweep_bayes.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
    for dz, m in metrics.items():
        print(
            f"d={dz}: Bayes-optimal test LL = {m['log_likelihood']:.4f}, "
            f"posterior-mean MSE = {m['posterior_mean_mse']:.4f}"
        )
    print(f"Wrote {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exact Bayes reference values for the Mixture.")
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root)
