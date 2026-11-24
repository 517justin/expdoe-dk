"""Fixed, predeclared acceptance oracle and production-workflow gates."""
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def fixture():
    return importlib.import_module("experiments.multiobjective_validation.fixtures")


def runner():
    return importlib.import_module("experiments.multiobjective_validation.run")


def test_approved_protocol_is_explicit_and_locked():
    f = fixture()
    assert f.METHODS == ("random", "plain_gp", "knowledge_gp", "knowledge_gp_v2")
    assert f.PROTOCOL == "fixed-catalog-monotone"
    assert f.PROTOCOL_VERSION == "2.0"
    assert f.SEEDS == (2, 5, 11, 23, 47)
    assert (f.INITIAL, f.ROUNDS, f.Q) == (6, 8, 2)
    assert f.config().noise_mode == "noiseless"
    assert dict(f.config().reference_point) == {"gain": 0., "retention": 0.}
    spec, = f.knowledge("knowledge_gp_v2").to_envelope()["specs"]
    assert spec["pattern"] == "monotone" and spec["version"] == "2.0"
    assert spec["confidence"] == .8
    assert spec["scope"] == {"factors": ["x"], "objectives": ["gain"],
                             "region": None, "conditions": None}
    assert spec["parameters"] == {"direction": "increasing", "n_pairs_per_dim": 3,
                                  "epsilon": .05, "softness": 1.}
    old, = f.knowledge("knowledge_gp").to_envelope()["specs"]
    assert old["version"] == "1.0" and old["confidence"] == .8
    assert old["parameters"] == {"direction": "increasing", "epsilon": "auto",
                                 "delta_norm": .5, "n_pairs_per_dim": 3}
    assert not f.knowledge("random").specs and not f.knowledge("plain_gp").specs
    with pytest.raises(ValueError, match="method"):
        f.knowledge("knowledge_gp_latest")


def test_oracle_exact_feasible_catalog_hypervolume():
    from expdoe_dk.bo.multiobjective_metrics import hypervolume
    f = fixture()
    X = f.catalog()
    assert len(X) == 101
    np.testing.assert_array_equal(X.x, np.arange(101) / 100)
    feasible = X.loc[X.x <= .8]
    assert hypervolume(f.measure(feasible).to_numpy(), [0., 0.]) == pytest.approx(.476, abs=1e-12)


@pytest.mark.parametrize("seed", [2, 5, 11, 23, 47])
def test_initial_design_uses_only_seed_and_hard_feasible_catalog(seed):
    f = fixture()
    expected = np.random.default_rng(seed).choice(81, 6, replace=False)
    pd.testing.assert_frame_equal(f.initial(seed), f.catalog().iloc[expected].reset_index(drop=True))
    assert f.initial(seed).x.is_unique


def test_geometry_and_oracle_share_units_without_truth_fitted_scale():
    f = fixture()
    geometry = f.geometry(2)
    values = f.measure(f.initial(2)).to_numpy()
    np.testing.assert_allclose(geometry.location, values.min(axis=0))
    np.testing.assert_allclose(geometry.scale, np.ptp(values, axis=0))
    from expdoe_dk.bo.multiobjective_metrics import hypervolume
    oracle = f.measure(f.catalog().query("x <= .8")).to_numpy()
    normalized = (oracle - geometry.location) / geometry.scale
    assert hypervolume(normalized, geometry.report_reference) == pytest.approx(.476 / np.prod(geometry.scale))


@pytest.mark.parametrize("m,noise,pending,route", [
    (2, "noiseless", False, "q_log_ehvi"),
    (3, "known", False, "q_log_nehvi"),
    (4, "noiseless", True, "q_log_nehvi"),
    (5, "inferred", False, "q_log_nparego"),
    (6, "noiseless", False, "q_log_nparego"),
])
def test_real_catalog_route_smoke(m, noise, pending, route):
    from expdoe_dk import Campaign, MultiObjectiveConfig, Objective, ObservationBatch, PendingBatch, Space
    f = fixture()
    base = f.space()
    space = Space(base.params, objectives=[Objective(f"o{i}", "maximize") for i in range(m)],
                  constraints=base.constraints)
    X = f.initial(2)
    Y = pd.DataFrame({name: (X.x if i % 2 == 0 else 1 - X.x)
                      for i, name in enumerate(space.objectives)})
    campaign = Campaign(space, seed=2, validate=False, multiobjective=MultiObjectiveConfig(noise_mode=noise))
    variance = Y * 0 + .001 if noise == "known" else None
    campaign.tell_batch(ObservationBatch(X, Y[space.objectives[::-1]], Yvar=variance))
    if pending:
        campaign.set_pending(PendingBatch(("external",), pd.DataFrame({"x": [.79]})))
    catalog = f.catalog()
    ids = tuple(f"c{i:03}" for i in range(len(catalog)))
    result = campaign.ask_with_diagnostics(q=2, candidates=catalog, candidate_ids=ids)
    assert result.diagnostics["acquisition_kind"] == route
    assert result.diagnostics["objective_names"] == space.objectives
    assert result.predictions.mean.shape == (2, m)
    assert len(result.frame) == 2 and result.frame.x.is_unique
    assert not set(result.frame.x) & (set(X.x) | ({.79} if pending else set()))
    assert (result.frame.x <= .8).all()
    pd.testing.assert_frame_equal(result.frame, catalog.iloc[[ids.index(i) for i in result.candidate_ids]].reset_index(drop=True))


def test_runner_contract_and_quality_gate_rejects_bad_results():
    r = runner()
    rows = [{"method": m, "hv_regret": v} for m, v in
            [("random", .1), ("plain_gp", .12), ("knowledge_gp", .9), ("knowledge_gp_v2", .09)]]
    assert not r.quality_summary(rows)["passed"]
    rows[1]["hv_regret"] = .1
    summary = r.quality_summary(rows)
    assert summary["passed"]
    assert not summary["legacy_v1"]["passed"]
    assert set(summary["gates"]) == {"plain_gp_vs_random", "knowledge_gp_v2_vs_random", "knowledge_gp_v2_vs_plain"}
    rows[-1]["hv_regret"] = .111
    assert not r.quality_summary(rows)["passed"]
    rows[1]["hv_regret"], rows[-1]["hv_regret"] = .05, .06
    assert not r.quality_summary(rows)["passed"]  # v2 vs plain alone fails


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -.1])
def test_quality_gate_rejects_invalid_regret(bad):
    rows = [{"method": m, "hv_regret": .1} for m in fixture().METHODS]
    rows[-1]["hv_regret"] = bad
    with pytest.raises(ValueError, match="regret"):
        runner().quality_summary(rows)


def test_quality_gate_requires_all_methods():
    with pytest.raises(ValueError, match="method"):
        runner().quality_summary([{"method": "random", "hv_regret": .1}])


@pytest.mark.slow
def test_full_fixed_seed_matrix(tmp_path):
    r = runner()
    rows = r.run_matrix(seeds=(2, 5, 11, 23, 47), output_dir=tmp_path)
    assert len(rows) == 20
    assert len((tmp_path / "matrix.jsonl").read_text().splitlines()) == 20
    for row in rows:
        assert row["protocol"] == fixture().PROTOCOL
        assert row["protocol_version"] == fixture().PROTOCOL_VERSION
        assert row["knowledge"] == fixture().knowledge(row["method"]).to_envelope()
        assert row["measurement_count"] == 22
        assert row["parameter_feasible_rate"] == 1.
        assert row["duplicate_rate"] == 0.
        assert row["batch_sizes"] == [2] * 8
        assert row["reproducible"] and row["restart_consistent"]
        assert row["source_commit"] and isinstance(row["dirty"], bool)
        assert row["oracle_hv_raw"] == pytest.approx(.476, abs=1e-12)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["passed"], summary
