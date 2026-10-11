"""Content-addressed sweep identity."""

from cf_npe.runner.provenance import config_id, expand_units, missing_units, read_plan, write_plan


def test_config_id_is_key_order_independent():
    assert config_id({"a": 1, "b": 2, "train_params": {"lr": 0.1, "k": 4}}) == config_id(
        {"train_params": {"k": 4, "lr": 0.1}, "b": 2, "a": 1}
    )


def test_config_id_distinguishes_seed_batches():
    """A seed's random stream depends on its batch, so batching is part of identity."""
    base = {"problem": "gaussian", "n_train": 1000}
    batch = config_id({**base, "seed": 0, "seeds_per_job": 3})
    assert batch != config_id({**base, "seed": 0, "seeds_per_job": 1})
    assert batch != config_id({**base, "seed": 1, "seeds_per_job": 3})


def test_config_id_distinguishes_real_config_differences():
    assert config_id({"problem": "gaussian", "n_train": 1000}) != config_id(
        {"problem": "gaussian", "n_train": 3000}
    )
    assert config_id({"train_params": {"k": 4}}) != config_id({"train_params": {"k": 10}})


def test_expand_units_expands_seed_batches():
    grid = [
        {"n_train": 1000, "seed": 0, "seeds_per_job": 3},
        {"n_train": 1000, "seed": 7, "seeds_per_job": 1},
    ]
    units = expand_units(grid)
    assert [u["seed"] for u in units] == [0, 1, 2, 7]
    # seeds_per_job is dropped; the rest of the spec is preserved per unit.
    assert all("seeds_per_job" not in u for u in units)
    assert all(u["n_train"] == 1000 for u in units)


def test_missing_units_is_plan_driven():
    plan = [
        {"config_id": "aaa", "seed": 0},
        {"config_id": "aaa", "seed": 1},
        {"config_id": "aaa", "seed": 2},
    ]
    # Seeds 0,1 done, seed 2 missing; a record with another config_id must not mask the miss.
    records = [
        {"config_id": "aaa", "seed": 0},
        {"config_id": "aaa", "seed": 1},
        {"config_id": "zzz", "seed": 5},
    ]
    miss = missing_units(plan, records)
    assert miss == [{"config_id": "aaa", "seed": 2}]


def test_missing_units_empty_when_all_present():
    plan = [{"config_id": "aaa", "seed": 0}, {"config_id": "bbb", "seed": 0}]
    records = [
        {"config_id": "bbb", "seed": 0},
        {"config_id": "aaa", "seed": 0},
    ]
    assert missing_units(plan, records) == []


def test_plan_roundtrip(tmp_path):
    units = [{"config_id": "aaa", "seed": 0, "n_train": 1000}]
    path = tmp_path / "plan.json"
    write_plan(path, units)
    assert read_plan(path) == units


def test_job_overrides_quote_config_id():
    """The ++config_id override is quoted so Hydra can't parse e.g. 77243688e679 as a float."""
    from pathlib import Path

    from cf_npe.runner.launch import _job_overrides

    overrides = _job_overrides(
        experiment="exp",
        scale="large",
        job_dir=Path("/tmp/x"),
        point={"seed": 0, "config_id": "77243688e679"},
        out_dir=Path("/tmp"),
        stage1_from=None,
        hydra_overrides=[],
    )
    assert "++config_id='77243688e679'" in overrides


def test_stage2_config_id_tracks_stage1_cache_config(tmp_path):
    """Rebuilding a Stage-1 cache with another config changes the Stage-2 units' identity."""
    from cf_npe.runner.launch import stage1_config_ids, unit_config_id

    def stage2_id():
        ids = stage1_config_ids(str(tmp_path))
        return unit_config_id("exp", {"seed": 0}, "large", str(tmp_path), ids, [])

    plan = tmp_path / "multirun" / "build_cache" / "plan.json"
    plan.parent.mkdir(parents=True)
    write_plan(plan, [{"config_id": "aaa", "seed": 0}])
    before = stage2_id()
    write_plan(plan, [{"config_id": "bbb", "seed": 0}])
    assert stage2_id() != before
