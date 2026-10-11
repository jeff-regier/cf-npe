"""Write the Make rules for every sweep and Optuna study; the Makefile includes them.

A sweep runs ``run-sweep`` in ``<base>/runs/<name>/`` and its target is
``<base>/results/<name>.json``, where the base is ``$(TUNE)`` for the Stage-1 caches that
studies tune on and ``$(ROOT)`` otherwise. A study runs ``run-tune`` and its target is
``$(TUNE)/recipes/<name>/best_method.yaml``. Sweeps read the committed recipes in
``conf/recipes/``, or with ``TUNE_SYN``/``TUNE_SL`` set, the tuned ones. Whatever depends on
how make is invoked (output root, smoke or full, committed or tuned recipes) stays a Make
variable.

Usage::

    python -m cf_npe.runner.pipeline > sweeps.mk
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from itertools import pairwise
from typing import NamedTuple

# Synthetic and strong-lensing recipe roots.
R = "$(R)"
RSL = "$(RSL)"

# Dedicated one-seed timing.
RUNTIME = "--n-seeds 1 --seed-start 1 --set seeds_per_job=1"
NO_EVAL = "--set 'evaluator.evaluators=[]' --set 'evaluator.sample_evaluators=[]'"
# The three tuning seeds that the CF-NPE Stage-2 studies batch (their seeds_per_job).
TUNE_SEEDS = "--seed-start $(TUNE_SEED) --n-seeds 3"


def problem_args(family: str, dim: int) -> tuple[str, ...]:
    return (f"problem={family}", f"--set problem.dim={dim}")


def main_args(problem: str) -> tuple[str, ...]:
    """Problem arguments of a main problem's Stage-1 caches and tuning studies."""
    return problem_args(PROBLEMS[problem].family, PROBLEMS[problem].dim)


def cell_args(problem: str) -> tuple[str, ...]:
    """Problem arguments of a main problem's evaluation sweeps."""
    return (*main_args(problem), f"--set seeds_per_job={PROBLEMS[problem].seeds_per_job}")


def replicates(m: int) -> str:
    return f"--set train_params.n_replicates={m}"


def stage1_cache(section: str, name: str, main: str, args: tuple[str, ...]) -> Sweep:
    """A Stage-1 cache built with the ``main`` problem's Stage-1 recipe."""
    recipe = (f"{R}/{main}_stage1",)
    return Sweep(section, (), name, "cf_npe_build_cache", recipe, None, args)


def synthetic_tuning(problem: str) -> list[Job]:
    """The Stage-1 mean network, then standard NPE and CF-NPE Stage 2 per family."""
    family, _, m, _ = PROBLEMS[problem]
    study = partial(Study, ())
    args = main_args(problem)
    cache_args = (*args, replicates(m), TUNE_SEEDS)
    cache = stage1_cache("tune-caches", f"{problem}_stage1_m{m}", problem, cache_args)
    jobs: list[Job] = [study(f"{problem}_stage1", f"tune_stage1_{family}", (), None, args), cache]
    for f in FAMILIES:
        jobs += [
            study(f"{problem}_std_npe_{f}", f"tune_stage2_std_npe_{f}", (), None, args),
            study(f"{problem}_cf_npe_{f}", f"tune_stage2_cf_npe_{f}", (), cache, args),
        ]
    return jobs


def case_study(problem: str) -> list[Job]:
    """Stage-1 caches with M = 1, 3, 10 replicates, then every main-table cell per family;
    each cache and cell has a runtime twin."""
    sweep = partial(Sweep, problem, ())
    args = main_args(problem)
    caches = {
        m: stage1_cache(problem, f"{problem}_stage1_m{m}", problem, (*args, replicates(m)))
        for m in (1, 3, 10)
    }
    jobs: list[Job] = []
    for m, cache in caches.items():
        name = f"runtime1_stage1_{problem}_m{m}"
        jobs += [cache, stage1_cache(problem, name, problem, (*args, replicates(m), RUNTIME))]
    c30 = ("--set train_params.n_components=30",)
    stage2_mixture = ("--set train_params.stage2_mixture=true",)
    for f in FAMILIES:
        std_npe = (f"{R}/{problem}_std_npe_{f}",)
        cf_npe = (f"{R}/{problem}_cf_npe_{f}",)
        cells = [
            ("std_npe", "std_npe_test_ll", std_npe, None, ()),
            ("std_npe_mixture_c10", "std_npe_mixture_test_ll", std_npe, None, ()),
            ("std_npe_mixture_c30", "std_npe_mixture_test_ll", std_npe, None, c30),
            *((f"cf_npe_m{m}", "cf_npe_stage2_test_ll", cf_npe, c, ()) for m, c in caches.items()),
            ("cf_npe_mixture", "cf_npe_stage2_test_ll", cf_npe, caches[3], stage2_mixture),
        ]
        posterior = f"--set train_params.posterior_type={f}"
        for row, exp, recipes, stage1, extra in cells:
            full = (posterior, *cell_args(problem), *extra)
            timed = (posterior, *args, RUNTIME, NO_EVAL, *extra)
            jobs += [
                sweep(f"{problem}_{row}_{f}", exp, recipes, stage1, full),
                sweep(f"runtime1_{problem}_{row}_{f}", exp, recipes, stage1, timed),
            ]
    return jobs


def gaussian_ablations() -> list[Job]:
    """Gaussian d=200 ablations (appendix) on the M=3 cache: oversized Stage-2 flows;
    CF-NPE with in-fold residuals; a single Stage-1 fit; and Std-NPE-mixture (K'=10) with
    each component validated on its own fold."""
    sweep = partial(Sweep, "gauss_d200", ())
    gauss = cell_args("gauss_d200")
    cache_args = (*main_args("gauss_d200"), replicates(3))
    cache = stage1_cache("gauss_d200", "gauss_d200_stage1_m3", "gauss_d200", cache_args)
    oversized = (
        "--set train_params.posterior_type=nsf --set train_params.stage2_training.lr=1e-3 "
        "--set train_params.stage2_cfg.n_transforms=8 --set train_params.stage2_cfg.n_bins=32 "
        "--set train_params.stage2_cfg.hidden_dim=1024 --set eval_batch_size=4"
    )
    deeper = "--set train_params.stage2_cfg.n_layers=5"
    no_reg = "--set train_params.stage2_cfg.dropout=0.0 "
    no_reg += "--set train_params.stage2_training.weight_decay=0.0"
    flows = {"width": (), "depth": (deeper,), "noreg": (deeper, no_reg)}
    recipe = (f"{R}/gauss_d200_cf_npe_nsf",)
    jobs: list[Job] = [
        sweep(
            f"gauss_d200_flow_{name}_nsf",
            "cf_npe_stage2_test_ll",
            recipe,
            cache,
            (*gauss, oversized, *args),
        )
        for name, args in flows.items()
    ]
    for f in FAMILIES:
        posterior = f"--set train_params.posterior_type={f}"
        cf_npe = (f"{R}/gauss_d200_cf_npe_{f}",)
        jobs += [
            sweep(
                f"gauss_d200_cf_npe_m3_in_fold_{f}",
                "cf_npe_stage2_test_ll",
                cf_npe,
                cache,
                (posterior, *gauss, "--set train_params.in_fold_residuals=true"),
            ),
            sweep(
                f"gauss_d200_cf_npe_single_fit_{f}",
                "cf_npe_single_fit_test_ll_gauss_d200",
                (f"{R}/gauss_d200_stage1", *cf_npe),
                None,
                (posterior,),
            ),
            sweep(
                f"gauss_d200_std_npe_mixture_c10_partition_{f}",
                "std_npe_mixture_test_ll",
                (f"{R}/gauss_d200_std_npe_{f}",),
                None,
                (posterior, *gauss, "--set train_params.split_mode=partition"),
            ),
        ]
    return jobs


def dimsweep(family: str, dim: int, tag: str, posterior: str, main: str) -> list[Job]:
    """One dimension of a dimension sweep (appendix): a Stage-1 cache with M = 3 replicates
    (from the main problem's Stage-1 recipe) and standard-NPE and CF-NPE recipes tuned at
    that dimension. At the main problem's dimension these are the main problem's own M = 3
    cache and studies, except a CF-NPE study that the main problem tunes on another M."""
    sweep = partial(Sweep, "dimsweeps", ())
    study = partial(Study, ())
    set_dim = f"--set problem.dim={dim}"
    args = problem_args(family, dim)
    name = f"{family}_dimsweep{tag}"
    std_npe, cf_npe = f"{name}_std_npe_d{dim}", f"{name}_cf_npe_d{dim}"
    if dim == PROBLEMS[main].dim:
        cache_section, cache_name, cache_args = main, f"{main}_stage1_m3", (*args, replicates(3))
        std_npe_recipe = f"{main}_std_npe_{posterior}"
        cf_npe_recipe = f"{main}_cf_npe_{posterior}" if PROBLEMS[main].tune_m == 3 else cf_npe
    else:
        cache_section, cache_name = "dimsweeps", f"{family}_dimsweep_stage1_d{dim}"
        cache_args = (*args, "--set seeds_per_job=1")
        std_npe_recipe, cf_npe_recipe = std_npe, cf_npe
    cache = stage1_cache(cache_section, cache_name, main, cache_args)
    tune_cache = stage1_cache("tune-caches", cache_name, main, (*cache_args, TUNE_SEEDS))
    return [
        cache,
        tune_cache,
        study(std_npe_recipe, f"tune_stage2_std_npe_{posterior}", (), None, args),
        study(cf_npe_recipe, f"tune_stage2_cf_npe_{posterior}", (), tune_cache, args),
        sweep(std_npe, f"std_npe_test_ll_{name}", (f"{R}/{std_npe_recipe}",), None, (set_dim,)),
        sweep(cf_npe, f"cf_npe_test_ll_{name}", (f"{R}/{cf_npe_recipe}",), cache, (set_dim,)),
    ]


def gaussian_scaling() -> list[Job]:
    """Gaussian d=200 baselines and scaling (appendix): temperature scaling, gradient losses,
    and the training-set size N and fold count K. Standard NPE is tuned separately at each
    N; CF-NPE (M=3, Gaussian family) uses the main-table recipes. N=10000 and K=10 are the
    main-table cells, which the tables read directly."""
    sweep = partial(Sweep, "gaussian-scaling", ())
    study = partial(Study, ())
    std_npe = (f"{R}/gauss_d200_std_npe_gaussian",)
    jobs: list[Job] = [sweep("gg_temperature_metrics", "gg_temperature", std_npe, None, ())]
    for loss in ("beta_nll", "stop_gradient"):
        args = (f"--set train_params.loss={loss}",)
        jobs.append(sweep(f"gg_gradflow_{loss}_metrics", "gg_gradflow", std_npe, None, args))
    for n in N_TRAINS:
        recipe = f"gauss_d200_std_npe_gaussian_N{n}"
        n_train = f"--set n_train={n}"
        tune_args = (*main_args("gauss_d200"), n_train)
        jobs += [
            study(recipe, "tune_stage2_std_npe_gaussian", (), None, tune_args),
            sweep(f"gg_ntrain_N{n}", "gg_ntrain_std_npe", (f"{R}/{recipe}",), None, (n_train,)),
        ]
    cells = [(f"kfold_K{k}", k, 10000) for k in (2, 4, 8, 16)]
    cells += [(f"ntrain_cf_npe_N{n}", 10, n) for n in N_TRAINS]
    for name, k, n in cells:
        n_train = f"--set n_train={n}"
        k_folds = f"--set train_params.k_folds={k}"
        cache_args = (*main_args("gauss_d200"), replicates(3), k_folds, n_train)
        cache = stage1_cache("gaussian-scaling", f"gg_stage1_{name}", "gauss_d200", cache_args)
        jobs += [
            cache,
            sweep(
                f"gg_{name}",
                "cf_npe_stage2_test_ll",
                (f"{R}/gauss_d200_cf_npe_gaussian",),
                cache,
                ("--set train_params.posterior_type=gaussian", *cell_args("gauss_d200"), n_train),
            ),
        ]
    return jobs


def strong_lensing() -> list[Job]:
    """Strong-lensing tuning and experiments, which need the simulated datasets.

    CF-NPE Stage 2 takes the Stage-1 recipe's encoder and mean head (the cached Stage-1
    networks must match them, and the Stage-2 offset encoder copies the encoder), the
    standard-NPE flow head, and its own recipe. problem.encoder.dropout=0.0 sets only the
    offset encoder's dropout: the Stage-1 networks run in eval mode."""
    sweep = partial(Sweep, "strong-lensing", ("$(SL_DATA)",))
    study = partial(Study, ("$(SL_TUNE_DATA)",))
    stage1 = (f"{RSL}/strong_lensing_stage1",)
    std_npe = (f"{RSL}/strong_lensing_std_npe_nsf",)
    stage2 = (
        f"{RSL}/strong_lensing_stage1:problem.encoder,problem.mean_head",
        f"{RSL}/strong_lensing_std_npe_nsf:problem.velocity_head",
    )
    cf_npe = (*stage2, f"{RSL}/strong_lensing_cf_npe_nsf_residual")
    no_dropout = ("--set problem.encoder.dropout=0.0",)
    cache_exp = "cf_npe_build_cache_strong_lensing"
    tune_cache = Sweep(
        "tune-caches",
        ("$(SL_TUNE_DATA)",),
        "sl10_cf_npe_m3_stage1",
        cache_exp,
        stage1,
        None,
        (replicates(3), "--seed-start $(TUNE_SEED) --n-seeds 1"),
    )
    m1 = sweep("sl10_cf_npe_m1_stage1", cache_exp, stage1, None, (replicates(1),))
    m3 = sweep("sl10_cf_npe_m3_stage1", cache_exp, stage1, None, (replicates(3),))
    stage2_exp = "strong_lensing_cf_npe_stage2_eval_nsf"
    return [
        study("strong_lensing_stage1", "sl_cf_npe_mean", (), None, ()),
        study("strong_lensing_std_npe_nsf", "sl_std_npe_nsf", (), None, ()),
        tune_cache,
        study(
            "strong_lensing_cf_npe_nsf_residual",
            "sl_cf_npe_stage2_residual_nsf",
            stage2,
            tune_cache,
            no_dropout,
        ),
        m1,
        m3,
        sweep("sl10_std_npe", "strong_lensing_std_npe_eval_nsf", std_npe, None, ()),
        sweep(
            "sl10_std_npe_mixture_c10", "strong_lensing_std_npe_mixture_eval_nsf", std_npe, None, ()
        ),
        sweep(
            "sl10_std_npe_mixture_c50",
            "strong_lensing_std_npe_mixture_subsample_nsf",
            std_npe,
            None,
            (),
        ),
        sweep("sl10_cf_npe_m1_residual", stage2_exp, cf_npe, m1, no_dropout),
        sweep("sl10_cf_npe_m3_residual", stage2_exp, cf_npe, m3, no_dropout),
        sweep(
            "sl10_cf_npe_mixture_residual",
            "strong_lensing_cf_npe_mixture_eval_nsf",
            cf_npe,
            m3,
            no_dropout,
        ),
    ]


def all_jobs() -> list[Job]:
    jobs: list[Job] = []
    for problem in PROBLEMS:
        jobs += synthetic_tuning(problem) + case_study(problem)
    jobs += gaussian_ablations()
    jobs += [
        job
        for dim in GAUSSIAN_DIMS
        for job in dimsweep("gaussian", dim, "", "gaussian", "gauss_d200")
    ]
    jobs += [
        job
        for main in ("funnel_d200", "mixture_d20")
        for dim in NSF_DIMS
        for job in dimsweep(PROBLEMS[main].family, dim, "_nsf", "nsf", main)
    ]
    # Sections share some jobs (e.g. the main problems' M = 3 caches); one rule each.
    return list(dict.fromkeys(jobs + gaussian_scaling() + strong_lensing()))


def rule(job: Job) -> str:
    recipe_files = [f"{spec.partition(':')[0]}/best_method.yaml" for spec in job.recipes]
    args = [f"--method-from {spec}" for spec in job.recipes]
    stage1 = []
    if job.stage1:
        stage1 = [job.stage1.target]
        args.append(f"--stage1-from {job.stage1.run_dir}")
    prereqs = " ".join([*recipe_files, *stage1, *job.needs])
    return f"{job.target}: {prereqs}\n\t{' '.join([job.command, *args, *job.args])}\n"


def makefile(jobs: list[Job]) -> str:
    """The rules, a phony target per section, and the smoke-run chains.

    Smoke runs build concurrently, but chain each section's targets so they run one after
    another, as order-only prerequisites: jobs from one section can be large on the same
    problem, and together exhaust the GPU's memory."""
    sections: defaultdict[str, list[str]] = defaultdict(list)
    for job in jobs:
        sections[job.section].append(job.target)
    lines = [f".PHONY: {' '.join(sections)}", ""]
    lines += [rule(job) for job in jobs]
    lines += [f"{section}: {' '.join(targets)}" for section, targets in sections.items()]
    lines += ["", "ifdef SMOKE"]
    lines += [f"{b}: | {a}" for targets in sections.values() for a, b in pairwise(targets)]
    lines += ["endif"]
    return "\n".join(lines) + "\n"


class Problem(NamedTuple):
    family: str
    dim: int
    # Stage-1 replicates in the cache that CF-NPE Stage 2 is tuned on.
    tune_m: int
    # Seeds trained together in one job of a main-table cell.
    seeds_per_job: int


PROBLEMS = {
    "gauss_d200": Problem("gaussian", 200, 3, 5),
    "funnel_d200": Problem("funnel", 200, 1, 1),
    "mixture_d20": Problem("mixture", 20, 1, 10),
}


FAMILIES = ("gaussian", "nsf")


GAUSSIAN_DIMS = (2, 10, 50, 100, 200)


NSF_DIMS = (2, 10, 20, 50)


N_TRAINS = (1000, 3000, 30000, 100000)


@dataclass(frozen=True)
class Sweep:
    section: str
    needs: tuple[str, ...]
    name: str
    experiment: str
    recipes: tuple[str, ...]
    stage1: Sweep | None
    args: tuple[str, ...]

    @property
    def base(self) -> str:
        return "$(TUNE)" if self.section == "tune-caches" else "$(ROOT)"

    @property
    def run_dir(self) -> str:
        return f"{self.base}/runs/{self.name}"

    @property
    def target(self) -> str:
        return f"{self.base}/results/{self.name}.json"

    @property
    def command(self) -> str:
        return f"$(SWEEP) --out {self.run_dir} --results $@ --experiment {self.experiment}"


@dataclass(frozen=True)
class Study:
    needs: tuple[str, ...]
    name: str
    sweep: str
    recipes: tuple[str, ...]
    stage1: Sweep | None
    args: tuple[str, ...]
    section = "studies"

    @property
    def target(self) -> str:
        return f"$(TUNE)/recipes/{self.name}/best_method.yaml"

    @property
    def command(self) -> str:
        return f"$(STUDY) --sweep {self.sweep} --out $(@D)"


Job = Sweep | Study


if __name__ == "__main__":
    print(makefile(all_jobs()), end="")
