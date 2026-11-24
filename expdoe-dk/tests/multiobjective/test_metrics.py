"""Measured maximization metrics, independent of models and preferences."""
import importlib

import numpy as np
import pytest

from expdoe_dk import EngineError, ErrorCode


def api():
    return importlib.import_module("expdoe_dk.bo.multiobjective_metrics")


def test_exact_two_dimensional_hypervolume():
    assert api().hypervolume([[1., 2.], [2., 1.]], [0., 0.]) == 3.


def test_duplicates_dominated_and_below_reference():
    points = np.array([[1., 2.], [2., 1.], [1., 2.], [.5, .5], [-1., 9.]])
    assert api().hypervolume(points, [0., 0.]) == 3.
    # Retain duplicate trial identities in Pareto audit, deduplicate only HV.
    assert api().pareto_mask(points).tolist() == [True, True, True, False, True]


def test_empty_and_reference_boundary():
    assert api().hypervolume(np.empty((0, 2)), [0., 0.]) == 0.
    assert api().pareto_mask(np.empty((0, 2))).tolist() == []
    assert api().hypervolume([[0., 5.], [-1., -1.]], [0., 0.]) == 0.


@pytest.mark.parametrize("points,reference", [
    ([[np.nan, 1.]], [0., 0.]), ([[1., np.inf]], [0., 0.]),
    ([[1., 2.]], [0., np.nan]), ([[1., 2.]], [0.]),
    ([1., 2.], [0., 0.]), ([[1.]], [0.]),
])
def test_metrics_reject_nonfinite_or_misaligned_input(points, reference):
    with pytest.raises(EngineError) as caught:
        api().hypervolume(points, reference)
    assert caught.value.code == ErrorCode.CONFIG_INVALID


def test_pareto_rejects_nan():
    with pytest.raises(EngineError):
        api().pareto_mask(np.array([[np.nan, 2.]]))


def test_oracle_regret_is_benchmark_only_and_rejects_mismatch():
    assert api().hypervolume_regret([[1., 2.]], [[1., 2.], [2., 1.]], [0., 0.]) == 1.
    with pytest.raises(EngineError, match="oracle"):
        api().hypervolume_regret([[2., 2.]], [[1., 1.]], [0., 0.])
    assert api().hypervolume_regret([[1., 1. + 5e-11]], [[1., 1.]], [0., 0.]) == 0.


def test_plateau_is_advisory_after_five_low_gain_updates():
    signal = api().plateau_signal([1., 1.0001, 1.0002, 1.0003, 1.0004, 1.0005])
    assert signal["kind"] == "plateau"
    assert signal["advisory"] is True and signal["stop"] is False
    assert api().plateau_signal([1.] * 5) is None
    assert api().plateau_signal([]) is None
    assert api().plateau_signal([1., 2., 2., 2., 2., 2.]) is None
    assert api().plateau_signal([0., 1e-14, 1e-14, 1e-14, 1e-14, 1e-14]) is None
    assert api().plateau_signal([0.] * 6)["advisory"] is True


@pytest.mark.parametrize("history", [[1., np.nan], [1., np.inf], [-1., 0.], [2., 1.]])
def test_plateau_rejects_invalid_progress(history):
    with pytest.raises(EngineError):
        api().plateau_signal(history)


def test_missing_botorch_metric_is_typed(monkeypatch):
    module = importlib.import_module("botorch.utils.multi_objective.hypervolume")
    monkeypatch.delattr(module, "Hypervolume")
    with pytest.raises(EngineError) as caught:
        api().hypervolume([[1., 2.]], [0., 0.])
    assert caught.value.code == ErrorCode.ACQUISITION_INCOMPATIBLE
