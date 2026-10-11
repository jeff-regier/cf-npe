# Cross-Fitting Neural Posterior Estimation (CF-NPE)

Code for the paper *Cross-Fitting for Neural Posterior Estimation*.

## Setup

Requires Python 3.12 and a CUDA GPU with 32 GB of memory.

```bash
uv sync
```

## Usage

```bash
make smoke                # every step, including tuning, at tiny size (about 6 minutes)
make                      # every experiment with the committed recipes
make tune-synthetic       # re-run the synthetic Optuna studies first, then reproduce
make tune-strong-lensing  # re-run the strong-lensing Optuna studies first, then reproduce
make synthetic            # the synthetic experiments
make strong-lensing       # the strong-lensing experiments
make gauss_d200           # one part of the synthetic experiments; also funnel_d200,
                          # mixture_d20, dimsweeps, gaussian-scaling
make -n                   # list the commands without running them
make test lint
```

`src/cf_npe/runner/pipeline.py` lists which configuration produced each results file;
the `Makefile` builds its rules from it. Every sweep is a target whose output is its
results file, written only once the sweep is complete, so an interrupted or failed run
resumes where it stopped. Full runs go one job at a time. 
With an NVIDIA GeForce RTX 5090 GPU, the synthetic experiments take about 15–20 GPU-hours 
and the strong-lensing experiments about 130–140 GPU-hours.

Everything is written under `out/` (`out/smoke/` for `make smoke`; set `ROOT=<dir>` to
write elsewhere):

- `data/`: strong-lensing datasets
- `runs/<sweep>/`: one directory per sweep
- `tune/`: tuned recipes and their Stage-1 caches (tuning targets only)
- `results/`: one results file per sweep, plus the mixture problem's exact log-likelihoods
  and the variance-trajectory experiment
- `caches/`: the arrays shown in Figure 4

## Layout

- `src/cf_npe/`: `methods/` (trainers), `stages/` (Stage 1, Stage 2, Stage-1 cache),
  `networks/`, `training/`, `problems/` (generative models), `evaluators.py`,
  `runner/` (`run-sweep`, `run-tune`, and the scripts behind `results/` files that are
  not sweeps and behind `caches/`).
- `conf/`: Hydra configuration: `problem/`, `experiment/`, `recipes/` (tuned
  hyperparameters), `sweep/` (the Optuna studies behind the recipes), `scale/`.

## License

MIT; see `LICENSE`.

## Citation

```bibtex
@inproceedings{regier2026crossfitting,
  title     = {Cross-Fitting for Neural Posterior Estimation},
  author    = {Regier, Jeffrey},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
