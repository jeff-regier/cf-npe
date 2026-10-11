# Reproduce the paper's results under $(ROOT); README.md lists the targets.

.DELETE_ON_ERROR:
.DEFAULT_GOAL := reproduce
.PHONY: reproduce smoke tune-synthetic tune-strong-lensing lint test synthetic strong-lensing

ROOT ?= $(if $(SMOKE),out/smoke,out)
SEEDS := $(if $(SMOKE),0 1,0 1 2 3 4 5 6 7 8 9)
SCALE := $(if $(SMOKE),--smoke)
N_SEEDS := $(words $(SEEDS))
# First Optuna seed; evaluation runs use lower seeds.
TUNE_SEED := 1000

TUNE := $(ROOT)/tune
DATA := $(ROOT)/data
# Synthetic and strong-lensing recipe roots.
R := $(if $(TUNE_SYN),$(TUNE)/recipes,conf/recipes)
RSL := $(if $(TUNE_SL),$(TUNE)/recipes,conf/recipes)

FIXED := $(SCALE) --set data_dir=$(abspath $(DATA))
SWEEP := uv run run-sweep $(FIXED) --n-seeds $(N_SEEDS)
STUDY := uv run run-tune $(FIXED) --seed $(TUNE_SEED)
PY := uv run python -u -m

sl_data = $(DATA)/strong_lensing/source16_varlens_seed$(1).pt
SL_DATA := $(foreach s,$(SEEDS),$(call sl_data,$(s)))
SL_TUNE_DATA := $(call sl_data,$(TUNE_SEED))

# Every sweep and Optuna study, from cf_npe.runner.pipeline.
include sweeps.mk
sweeps.mk: src/cf_npe/runner/pipeline.py
	$(PY) cf_npe.runner.pipeline > $@

# ── Synthetic results beyond the sweeps ─────────────────────────────────────

BAYES := $(ROOT)/results/mixture_dimsweep_bayes.json
$(BAYES): conf/problem/mixture.yaml
	$(PY) cf_npe.runner.compute_mixture_bayes --root $(ROOT)

VARIANCE := $(ROOT)/results/variance_trajectory.json
$(VARIANCE):
	$(PY) cf_npe.runner.run_variance_trajectory --root $(ROOT) $(SCALE)

synthetic: gauss_d200 funnel_d200 mixture_d20 dimsweeps gaussian-scaling $(BAYES) $(VARIANCE)

# ── Strong-lensing datasets and Figure 4 samples ───────────────────────────

$(DATA)/strong_lensing/source16_varlens_seed%.pt:
	$(PY) cf_npe.problems.prepare_strong_lensing $(SCALE) --source-res 16 --gp-lengthscale 1.0 \
		--seed $* --out $@

RECON := $(ROOT)/caches/strong_lensing_recon.npz
$(RECON): $(ROOT)/results/sl10_cf_npe_mixture_residual.json $(call sl_data,0)
	$(PY) cf_npe.runner.build_sl_recon_cache --root $(ROOT)

strong-lensing: $(RECON)

# ── Entry points ────────────────────────────────────────────────────────────

reproduce: synthetic strong-lensing

# At most one job per section at a time (see cf_npe.runner.pipeline).
smoke:
	$(MAKE) -j4 reproduce SMOKE=1 TUNE_SYN=1 TUNE_SL=1

tune-synthetic:
	$(MAKE) reproduce TUNE_SYN=1

tune-strong-lensing:
	$(MAKE) reproduce TUNE_SL=1

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run ty check .

test:
	uv run pytest
