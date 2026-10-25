"""Audit regressions: strict sensitivity, frozen inputs, real finite evidence."""
import importlib.util
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


def _audit():
    path = Path(__file__).resolve().parents[3] / "experiments/four_pattern_v2_sensitivity.py"
    assert path.is_file(), "F6 must ship the executable sensitivity audit"
    spec = importlib.util.spec_from_file_location("four_pattern_audit", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_sensitivity_is_strict_max_pairwise_grid_difference():
    audit = _audit()
    def sensitivity_flag(delta, training_scale):
        return audit.is_prior_sensitive(np.array([[0., 0.], [0., delta]]), training_scale)
    assert sensitivity_flag(1.01, 1.) is True
    assert sensitivity_flag(1., 1.) is False
    # Non-adjacent configurations, not difference from a selected reference.
    assert audit.is_prior_sensitive(np.array([[-.6], [0.], [.6]]), 1.) is True
    for values in (np.empty((0, 2)), np.ones((1, 2)), np.array([[0.], [np.nan]])):
        with pytest.raises(ValueError):
            audit.is_prior_sensitive(values, 1.)
    with pytest.raises(ValueError):
        audit.is_prior_sensitive(np.ones((2, 2)), 0.)
    for means in ([], [[1., 2.]]):
        assert audit.summarize_sensitivity(means, 1.)["status"] == "not_evaluable"


def test_fixture_matrix_covers_directions_transitions_and_frozen_folds():
    audit = _audit()
    fixtures = audit.build_fixtures()
    names = {f.name for f in fixtures}
    assert {"saturation_increasing", "saturation_decreasing", "additive_only", "zero_effect",
            "weak_noisy", "changed_units", "missing_transition", "composed",
            "composed_two_shapes"} <= names
    assert {f"threshold_{a}_{b}" for a, b in itertools.permutations(
        ("increasing", "decreasing", "flat"), 2)} <= names
    assert {f"{kind}_{direction}" for kind in ("synergy", "antagonism")
            for direction in ("higher", "lower")} <= names
    again = {f.name: f.describe() for f in audit.build_fixtures()}
    for fixture in fixtures:
        description = fixture.describe()
        assert description == again[fixture.name]
        folds = description["folds"]
        assert sorted(i for fold in folds for i in fold) == list(range(len(fixture.observations.X)))
        assert fixture.correct.to_envelope() != fixture.wrong.to_envelope()
        if fixture.default_noiseless:
            for fold in folds:
                from expdoe_dk.bo.knowledge_runtime import compile_runtime
                from expdoe_dk.bo.evaluation import _subset
                training = [i for i in range(len(fixture.observations.X)) if i not in fold]
                compile_runtime(fixture.space, fixture.correct, _subset(fixture.observations, training))
    policy = audit.policy_grid()
    assert len(policy) == 27
    assert len({tuple(p.values()) for p in policy}) == 27


def test_composed_generator_metadata_matches_its_analytic_truth():
    audit = _audit()
    fixture = next(f for f in audit.build_fixtures() if f.name == "composed")
    assert fixture.generator["linear_x"] == 0.
    assert fixture.generator["saturation_amplitude"] == 1.5
    # A separate anchored shape replaces the nuisance x slope, not adds to it.
    x = fixture.observations.X.x.to_numpy()
    z = fixture.observations.X.z.to_numpy()
    np.testing.assert_allclose(fixture.truth, 4. + 1.5 * x / (.5 + x) - .2 * z + 1.2 * x * z)


def test_real_small_audit_writes_metrics_typed_coverage_and_repeats(tmp_path):
    audit = _audit()
    fixtures = [f for f in audit.build_fixtures() if f.name in
                {"saturation_increasing", "missing_transition"}]
    kwargs = dict(fixtures=fixtures, seeds=[17], scale_multipliers=[1.], width_centers=[.05])
    report = audit.run_audit(**kwargs)
    repeat = audit.run_audit(**kwargs)
    assert report["default_noiseless_failures"] == 0
    assert report["default_noiseless_gate"] == "passed"
    assert report["policy"]["seeds"] == [17]
    for left, right in zip(report["fixtures"], repeat["fixtures"], strict=True):
        assert left["definition"] == right["definition"]
        assert left["runs"] == right["runs"]
        assert left["max_prediction_change"] is None
        for run in left["runs"]:
            assert set(run["training_ids"]).isdisjoint(run["validation_ids"])
            assert run["sensitivity"]["status"] == "not_evaluable"
            if run["status"] == "failed":
                assert run["failure"]["code"] == "INSUFFICIENT_DATA"
                assert run["failure"]["details"]["coverage_status"] == "deficient"
            else:
                assert np.isfinite(run["grid_mean"]).all()
                assert np.isfinite(run["validation_mean"]).all()
                assert run["invariants"]["passed"]
                assert run["uncertainty"] == "conditional_on_MAP"
        for pooled in left["pooled"]:
            if pooled["complete"]:
                assert np.isfinite([pooled["rmse"], pooled["mae"]]).all()
            else:
                assert pooled["rmse"] is pooled["mae"] is None
        for reference in left["references"]:
            if reference["track"] != "M0":
                assert reference["uncertainty"] == "unavailable"
                assert reference["coverage_95"] is None
    audit.write_report(report, tmp_path)
    assert json.loads((tmp_path / "audit.json").read_text()) == report
    assert (tmp_path / "REPORT.md").is_file()
    json.dumps(report, allow_nan=False)


def test_main_preserves_default_policy_metadata_and_cli_paths(tmp_path):
    audit = _audit()
    # A small real CLI run, selecting one declared fixture without replacing fits.
    assert audit.main(["--output-dir", str(tmp_path), "--fixtures", "saturation_increasing",
                       "--seeds", "17", "--scale-multipliers", "1", "--width-centers", ".05"]) == 0
    result = json.loads((tmp_path / "audit.json").read_text())
    assert result["policy"]["seeds"] == [17]
    assert result["policy"]["magnitude_scale_multipliers"] == [1.]
    assert result["policy"]["width_prior_centers"] == [.05]
    defaults = audit.default_policy()
    assert defaults["seeds"] == [17, 29, 43]
    assert defaults["magnitude_scale_multipliers"] == [.5, 1., 2.]
    assert defaults["width_prior_centers"] == [.025, .05, .10]
