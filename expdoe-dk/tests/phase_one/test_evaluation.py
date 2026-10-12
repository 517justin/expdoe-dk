"""Public fitting and leakage-free evaluation contracts."""
import builtins

import numpy as np
import pandas as pd
import pytest
import torch

from expdoe_dk import (EngineError, ErrorCode, Knowledge, Objective, ObservationBatch,
                       Parameter, Space, evaluate_models, fit_surrogate, make_folds,
                       KnowledgeScope, make_pattern_spec)


@pytest.fixture(autouse=True)
def single_threaded_small_matrix_fits():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def linear_batch():
    space = Space([Parameter("x", bounds=(0., 11.))], objectives="y")
    return space, ObservationBatch(pd.DataFrame({"x": np.arange(12.)}),
        pd.DataFrame({"y": 100. + 3 * np.arange(12.)}), ids=tuple(f"r{i}" for i in range(12)))


def test_evaluation_uses_identical_oof_rows(linear_batch):
    space, observations = linear_batch
    result = evaluate_models(space, observations, tracks={"M0": None}, seed=17)
    assert {r["track"] for r in result.rows} == {"M0", "ridge", "random_forest"}
    assert all(len(r["oof_predictions"]) == 12 for r in result.rows)
    assert result.folds == make_folds(12, seed=17)
    assert sorted(i for fold in result.folds for i in fold) == list(range(12))
    for row in result.rows:
        residual = observations.Y.y.to_numpy() - np.array(row["oof_predictions"])
        assert row["rmse"] == pytest.approx(np.sqrt(np.mean(residual**2)))
        assert row["mae"] == pytest.approx(np.mean(abs(residual)))
        assert row["r2"] == pytest.approx(1 - np.sum(residual**2) / 1287.)
        assert row["observation_ids"] == list(observations.ids)
        assert row["role"] == ("production" if row["track"] == "M0" else "reference")


def test_tiny_sample_folds():
    assert make_folds(3, seed=17) == ((0,), (1,), (2,))
    assert make_folds(10) == tuple((i,) for i in range(10))
    assert len(make_folds(11)) == 5
    assert make_folds(12, seed=17) != make_folds(12, seed=18)
    with pytest.raises(EngineError):
        make_folds(2, seed=17)


def test_fold_spy_sees_only_training_rows(linear_batch, monkeypatch):
    import expdoe_dk.bo.evaluation as evaluation
    real_fit = evaluation._fit_surrogate
    seen = []
    def spy(space, observations, **kwargs):
        fitted = real_fit(space, observations, **kwargs)
        seen.append((observations.ids, fitted.transforms))
        return fitted
    monkeypatch.setattr(evaluation, "_fit_surrogate", spy)
    space, batch = linear_batch
    result = evaluate_models(space, batch, tracks={"M0": None}, seed=17)
    for fold, (ids, transforms) in zip(result.folds, seen, strict=True):
        training = [i for i in range(12) if i not in fold]
        assert ids == tuple(f"r{i}" for i in training)
        assert transforms["response_location"] == pytest.approx(100 + 3 * np.mean(training))
        assert transforms["numeric_location"] == pytest.approx([np.mean(training)])
        assert transforms["numeric_scale"] == pytest.approx([np.std(training)])
    for row in result.rows:
        for fold, transforms in zip(result.folds, row["fold_transforms"], strict=True):
            training = [i for i in range(12) if i not in fold]
            assert transforms["numeric_location"] == pytest.approx([np.mean(training)])


@pytest.mark.parametrize("constant", [7., .1])
def test_constant_y_and_exclusions(constant):
    space = Space([Parameter("x", bounds=(0., 4.))], objectives="y")
    batch = ObservationBatch(pd.DataFrame({"x": [0., 1., 2., 3., 4.]}),
        pd.DataFrame({"y": [constant, None, constant, None, constant]}),
        status=pd.Series(["success", "warning", "success", "failed", "success"]),
        ids=("a", "b", "c", "d", "e"))
    result = evaluate_models(space, batch, tracks={"M0": None})
    assert result.exclusions == ({"index": 1, "id": "b", "reason": "warning"},
                                 {"index": 3, "id": "d", "reason": "failed"})
    assert all(row["r2"] is None and "constant" in row["r2_reason"] for row in result.rows)
    assert all(row["observation_ids"] == ["a", "c", "e"] for row in result.rows)


def test_fit_failure_is_typed(linear_batch, monkeypatch):
    import expdoe_dk.bo.surrogates as surrogates
    def fail(*args, **kwargs):
        raise RuntimeError("optimizer failed")
    monkeypatch.setattr(surrogates, "fit_gpytorch_mll", fail)
    with pytest.raises(EngineError) as error:
        fit_surrogate(*linear_batch)
    assert error.value.code == ErrorCode.MODEL_FIT_FAILED


def test_unknown_category_rejected_and_nominal_is_one_hot():
    space = Space([Parameter("c", kind="categorical", values=("A", "B", "C")),
                   Parameter("o", kind="ordinal", values=("low", "mid", "high"))], objectives="y")
    batch = ObservationBatch(pd.DataFrame({"c": ["A", "B", "A"], "o": ["low", "high", "mid"]}),
                             pd.DataFrame({"y": [10., 20., 11.]}))
    fit = fit_surrogate(space, batch)
    assert fit.transforms["features"] == ["c=A", "c=B", "c=C", "o"]
    assert fit.predict(pd.DataFrame({"c": ["C"], "o": ["mid"]})).mean.shape == (1, 1)
    with pytest.raises(EngineError) as error:
        fit.predict(pd.DataFrame({"c": ["missing"], "o": ["mid"]}))
    assert error.value.code == ErrorCode.CONFIG_INVALID


@pytest.mark.parametrize("with_prior", [False, True])
def test_variance_and_predictions_remain_physical(linear_batch, with_prior):
    space, batch = linear_batch
    known = ObservationBatch(batch.X, batch.Y, Yvar=pd.DataFrame({"y": [4.] * 12}))
    knowledge = Knowledge().with_gp_prior("weak") if with_prior else None
    fitted = fit_surrogate(space, known, knowledge=knowledge, seed=13)
    scaled = ObservationBatch(batch.X, batch.Y * 10 + 5,
                              Yvar=pd.DataFrame({"y": [400.] * 12}))
    fitted_scaled = fit_surrogate(space, scaled, knowledge=knowledge, seed=13)
    p, q = fitted.predict(batch.X), fitted_scaled.predict(batch.X)
    np.testing.assert_allclose(q.mean, p.mean * 10 + 5, rtol=1e-5)
    np.testing.assert_allclose(q.variance, p.variance * 100, rtol=1e-5)
    assert fitted.transforms["known_variance"] is True
    assert fitted.transforms["normalized_yvar"] == pytest.approx([4 / 107.25] * 12)
    # Check the actual likelihood, not just the reported normalization metadata:
    # an explicit learned-noise prior cannot override known measurement noise.
    torch.testing.assert_close(fitted._model.likelihood.noise,
        torch.full((12,), 4 / 107.25, dtype=torch.double))


def test_mixed_missing_variance_fails_closed(linear_batch):
    space, batch = linear_batch
    # ObservationBatch already rejects NaN; exercise a future/provider batch at this boundary.
    class PartialVarianceBatch(ObservationBatch):
        @property
        def Yvar(self):
            return pd.DataFrame({"y": [4.] * 11 + [np.nan]})
    partial = PartialVarianceBatch(batch.X, batch.Y)
    with pytest.raises(EngineError) as error:
        fit_surrogate(space, partial)
    assert error.value.code == ErrorCode.CONFIG_INVALID


def test_target_fits_response_not_distance(linear_batch):
    _, batch = linear_batch
    space = Space([Parameter("x", bounds=(0., 11.))], objectives=Objective("y", "target", target=110.))
    prediction = fit_surrogate(space, batch).predict(batch.X)
    assert np.max(abs(prediction.mean[:, 0] - batch.Y.y)) < 2.


@pytest.mark.parametrize("track", ["ridge", "random_forest"])
def test_reference_cannot_be_production_track(linear_batch, track):
    with pytest.raises(EngineError) as error:
        evaluate_models(*linear_batch, tracks={track: None})
    assert error.value.code == ErrorCode.CONFIG_INVALID


def test_reference_dependency_is_lazy_and_actionable(linear_batch, monkeypatch):
    real_import = builtins.__import__
    def missing(name, *args, **kwargs):
        if name.startswith("sklearn"):
            raise ImportError("absent")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", missing)
    fit_surrogate(*linear_batch)
    with pytest.raises(EngineError) as error:
        evaluate_models(*linear_batch, tracks={"M0": None})
    assert error.value.code == ErrorCode.CONFIG_INVALID
    assert "reference-models" in str(error.value) + str(error.value.hints)


@pytest.mark.parametrize("interaction", [False, True])
def test_smooth_and_interaction_oof_are_finite(interaction):
    x = np.linspace(0, 1, 12)
    z = np.array([0., .3, .9, .1, .7, .4, 1., .2, .8, .5, .6, .35])
    y = 10 + np.sin(2 * x) + (x * z * 3 if interaction else z)
    space = Space([Parameter("x", bounds=(0., 1.)), Parameter("z", bounds=(0., 1.))], objectives="y")
    batch = ObservationBatch(pd.DataFrame({"x": x, "z": z}), pd.DataFrame({"y": y}))
    result = evaluate_models(space, batch, tracks={"M0": None}, seed=3)
    assert all(np.isfinite(row["oof_predictions"]).all() for row in result.rows)
    assert all(np.isfinite(row["rmse"]) for row in result.rows)


def test_physical_mean_receives_only_training_response_statistics(linear_batch, monkeypatch):
    import expdoe_dk.bo.surrogates as surrogates
    original = surrogates.build_gp
    seen = []
    def spy(space, knowledge, X, Y, **kwargs):
        model, augmenter = original(space, knowledge, X, Y, **kwargs)
        seen.append((kwargs["response_location"], kwargs["response_scale"],
                     kwargs["runtime"]["mean_components"]))
        expected = -(3 * (1 + X[:, 0] * 11) - kwargs["response_location"]) / kwargs["response_scale"]
        torch.testing.assert_close(model.mean_module(X), expected)
        return model, augmenter
    monkeypatch.setattr(surrogates, "build_gp", spy)
    _, original_batch = linear_batch
    space = Space([Parameter("x", bounds=(1., 12.))], objectives="y")
    batch = ObservationBatch(original_batch.X + 1, original_batch.Y, ids=original_batch.ids)
    knowledge = Knowledge().add_spec(make_pattern_spec(pattern="power_law", version="1.0",
        confidence=1., scope=KnowledgeScope(factors=("x",)), parameters={"scale": 3., "exponent": 1.}))
    result = evaluate_models(space, batch, tracks={"formula": knowledge}, seed=17)
    for fold, (location, scale, artifacts) in zip(result.folds, seen, strict=True):
        training = batch.Y.iloc[[i for i in range(12) if i not in fold]].y
        assert location == pytest.approx(training.mean())
        assert scale == pytest.approx(training.std(ddof=0))
        assert len(artifacts) == 1


@pytest.mark.parametrize("prior", [False, True])
def test_similarity_and_shared_feature_kernel_compose(prior):
    space = Space([Parameter("x", bounds=(0., 3.)),
        Parameter("c", kind="categorical", values=("A", "B", "C"))], objectives="y")
    batch = ObservationBatch(pd.DataFrame({"x": [0., 1., 2., 3.], "c": ["A", "B", "C", "A"]}),
                             pd.DataFrame({"y": [1., 3., 2., 4.]}))
    knowledge = Knowledge().add_spec(make_pattern_spec(pattern="category_similarity", version="1.0",
        confidence=1., scope=KnowledgeScope(factors=("c",)), parameters={"levels": ["C", "A", "B"],
        "matrix": [[1., .3, .2], [.3, 1., .4], [.2, .4, 1.]]}))
    if prior:
        knowledge.with_gp_prior("weak")
    result = fit_surrogate(space, batch, knowledge=knowledge).predict(batch.X)
    assert np.isfinite(result.mean).all()
    assert (result.variance >= 0).all()


def test_virtual_conditions_cover_all_declared_categories(monkeypatch):
    import expdoe_dk.bo.surrogates as surrogates
    original = surrogates.build_gp
    seen = []
    def spy(space, knowledge, X, Y, **kwargs):
        seen.append(X[3:, 1].tolist())
        return original(space, knowledge, X, Y, **kwargs)
    monkeypatch.setattr(surrogates, "build_gp", spy)
    space = Space([Parameter("x", bounds=(0., 1.)),
        Parameter("c", kind="categorical", values=("A", "B", "C", "D"))], objectives="y")
    batch = ObservationBatch(pd.DataFrame({"x": [0., .5, 1.], "c": ["A", "B", "C"]}),
                             pd.DataFrame({"y": [1., 2., 3.]}))
    fit = fit_surrogate(space, batch, knowledge=Knowledge().with_random_augment(n=40), seed=17)
    assert set(seen[0]) == {0., 1., 2., 3.}
    assert fit.transforms["virtual_rows"] == 40


def test_reference_fit_failure_is_typed(linear_batch, monkeypatch):
    from sklearn.linear_model import Ridge
    def fail(*args, **kwargs):
        raise ValueError("reference fit failed")
    monkeypatch.setattr(Ridge, "fit", fail)
    with pytest.raises(EngineError) as error:
        evaluate_models(*linear_batch, tracks={"M0": None})
    assert error.value.code == ErrorCode.MODEL_FIT_FAILED
    assert error.value.details["track"] == "ridge"


def test_public_fit_requires_three_successes():
    space = Space([Parameter("x", bounds=(0., 1.))], objectives="y")
    batch = ObservationBatch(pd.DataFrame({"x": [0., .5, 1.]}), pd.DataFrame({"y": [1., 2., 3.]}),
                             status=pd.Series(["success", "success", "warning"]))
    for operation in (fit_surrogate, lambda s, b: evaluate_models(s, b, tracks={"M0": None})):
        with pytest.raises(EngineError) as error:
            operation(space, batch)
        assert error.value.code == ErrorCode.INSUFFICIENT_DATA


def test_decimal_constant_training_uses_unit_scale():
    space = Space([Parameter("x", bounds=(0., 1.))], objectives="y")
    batch = ObservationBatch(pd.DataFrame({"x": [.1] * 3}), pd.DataFrame({"y": [.1] * 3}))
    fit = fit_surrogate(space, batch)
    assert fit.transforms["numeric_scale"] == [1.]
    assert fit.transforms["response_scale"] == 1.
    np.testing.assert_allclose(fit.predict(batch.X).mean, .1, atol=1e-8)
