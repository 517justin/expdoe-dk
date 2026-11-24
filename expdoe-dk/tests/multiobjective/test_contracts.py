"""Multiobjective configuration and detached physical prediction contracts."""
from dataclasses import FrozenInstanceError
import json

import numpy as np
import pandas as pd
import pytest

import expdoe_dk as ed
from expdoe_dk import EngineError, ErrorCode, Parameter, Prediction, Space


@pytest.fixture
def config_type():
    # Resolve at test time so missing exports do not hide Prediction failures.
    return ed.MultiObjectiveConfig


@pytest.fixture
def space():
    return Space(params=[Parameter("x", bounds=(0, 1))], objectives=["yield", "cost"])


def test_prediction_keeps_two_physical_outputs():
    assert Prediction(np.zeros((3, 2)), np.ones((3, 2))).mean.shape == (3, 2)
    prediction = Prediction([[42, -7]], [[0.1, 2]])
    assert prediction.to_dict() == {"mean": [[42.0, -7.0]], "variance": [[0.1, 2.0]]}


@pytest.mark.parametrize("shape", [(0, 2), (3, 1), (2, 5)])
def test_prediction_accepts_any_positive_output_count(shape):
    assert Prediction(np.zeros(shape), np.ones(shape)).variance.shape == shape


@pytest.mark.parametrize("mean,variance", [
    (np.zeros((3, 2)), np.ones((3, 1))),
    (np.zeros((3, 2)), np.ones((2, 2))),
    (np.zeros((3, 0)), np.ones((3, 0))),
    (np.zeros((3, 2, 1)), np.ones((3, 2, 1))),
    (np.zeros(2), np.ones(2)),
    ([[1, 2]], [[0, -1]]),
    ([[1, np.nan]], [[0, 1]]),
    ([[1, np.inf]], [[0, 1]]),
    ([[1, 2]], [[0, np.nan]]),
    ([[1, 2]], [[0, np.inf]]),
])
def test_prediction_rejects_invalid_arrays(mean, variance):
    with pytest.raises(ValueError):
        Prediction(mean, variance)


def test_prediction_detaches_inputs_and_returned_arrays():
    mean, variance = np.array([[42., -7.]]), np.array([[0.1, 2.]])
    prediction = Prediction(mean, variance)
    mean[0, 1], variance[0, 1] = 99, -1
    for name in ("mean", "variance"):
        assert not object.__getattribute__(prediction, name).flags.writeable
        leaked = getattr(prediction, name)
        leaked.setflags(write=True)
        leaked[0, 1] = np.nan
    payload = prediction.to_dict()
    payload["mean"][0][1] = 99
    assert prediction.to_dict() == {"mean": [[42.0, -7.0]], "variance": [[0.1, 2.0]]}


def test_multiobjective_result_names_live_in_diagnostics():
    diagnostics = {"objective_names": ["yield", "cost"]}
    result = ed.RecommendationResult(
        pd.DataFrame({"x": [0.5]}), Prediction([[42, 7]], [[1, 2]]), diagnostics,
    )
    diagnostics["objective_names"].reverse()
    result.diagnostics["objective_names"].append("other")
    assert result.to_dict() == {
        "frame": [{"x": 0.5}],
        "predictions": {"mean": [[42.0, 7.0]], "variance": [[1.0, 2.0]]},
        "diagnostics": {"objective_names": ["yield", "cost"]},
    }


@pytest.mark.parametrize("names", [None, [], ["yield"], ["yield", "cost", "extra"],
                                       ["yield", "yield"], ["yield", ""],
                                       ["yield", 1], "yield,cost"])
def test_multiobjective_result_requires_complete_unique_names(names):
    with pytest.raises(ValueError, match="objective_names"):
        ed.RecommendationResult(
            pd.DataFrame({"x": [0.5]}), Prediction([[42, 7]], [[1, 2]]),
            {} if names is None else {"objective_names": names},
        )


def test_default_config_round_trip(config_type, space):
    config = config_type()
    expected = {
        "acquisition": "auto", "noise_mode": "auto", "reference_point": None,
        "preference_weights": None, "min_joint_feasibility": 0.95,
    }
    assert config.to_dict() == expected
    restored = config_type.from_dict(json.loads(json.dumps(expected, allow_nan=False)))
    assert restored == config
    assert restored.validate_for_space(space) is None
    with pytest.raises(FrozenInstanceError):
        config.noise_mode = "known"


@pytest.mark.parametrize("acquisition", ["auto", "q_log_ehvi", "q_log_nehvi", "q_log_nparego"])
@pytest.mark.parametrize("noise_mode", ["auto", "inferred", "known", "noiseless"])
def test_config_preserves_supported_options(config_type, acquisition, noise_mode):
    config = config_type(acquisition=acquisition, noise_mode=noise_mode)
    restored = config_type.from_dict(config.to_dict())
    assert restored.acquisition == acquisition
    assert restored.noise_mode == noise_mode


@pytest.mark.parametrize("field", ["reference_point", "preference_weights"])
def test_config_mappings_are_immutable_and_detached(config_type, space, field):
    source = {"cost": 2, "yield": 1}
    config = config_type(**{field: source})
    config.validate_for_space(space)
    source["cost"] = 99
    assert dict(getattr(config, field)) == {"yield": 1.0, "cost": 2.0}
    with pytest.raises(TypeError):
        getattr(config, field)["cost"] = 99
    payload = config.to_dict()
    restored = config_type.from_dict(payload)
    payload[field]["yield"] = 99
    assert dict(getattr(restored, field)) == {"yield": 1.0, "cost": 2.0}
    assert dict(getattr(config, field)) == {"yield": 1.0, "cost": 2.0}
    json.dumps(restored.to_dict(), allow_nan=False)


@pytest.mark.parametrize("field,value", [
    ("acquisition", "q_ehvi"), ("acquisition", True), ("acquisition", []),
    ("noise_mode", "exact"), ("noise_mode", None), ("noise_mode", {}),
    ("reference_point", [("yield", 1)]), ("preference_weights", [1, 2]),
    ("reference_point", {1: 1}), ("preference_weights", {"": 1}),
    ("min_joint_feasibility", -0.1), ("min_joint_feasibility", 1.01),
])
def test_config_rejects_invalid_fields(config_type, field, value):
    with pytest.raises(EngineError) as error:
        config_type(**{field: value})
    assert error.value.code == ErrorCode.CONFIG_INVALID


@pytest.mark.parametrize("field", ["reference_point", "preference_weights", "min_joint_feasibility"])
@pytest.mark.parametrize("value", [True, False, np.bool_(True), np.nan, np.inf, -np.inf,
                                   "1", None, 1j, 10**1000])
def test_config_rejects_nonfinite_or_nonreal_numbers(config_type, field, value):
    argument = value if field == "min_joint_feasibility" else {"yield": value, "cost": 1}
    with pytest.raises(EngineError) as error:
        config_type(**{field: argument})
    assert error.value.code == ErrorCode.CONFIG_INVALID


@pytest.mark.parametrize("weight", [0, -1])
def test_config_requires_positive_weights(config_type, weight):
    with pytest.raises(EngineError) as error:
        config_type(preference_weights={"yield": 1, "cost": weight})
    assert error.value.code == ErrorCode.CONFIG_INVALID


@pytest.mark.parametrize("probability", [0, 1, np.float64(0.5)])
def test_config_accepts_probability_boundaries_and_physical_reference(config_type, probability):
    config = config_type(reference_point={"yield": -10, "cost": 0},
                         min_joint_feasibility=probability)
    assert config.to_dict()["reference_point"] == {"yield": -10.0, "cost": 0.0}
    assert config.min_joint_feasibility == probability


@pytest.mark.parametrize("field", ["reference_point", "preference_weights"])
@pytest.mark.parametrize("names", [{}, {"yield": 1}, {"yield": 1, "typo": 2},
                                  {"yield": 1, "cost": 2, "extra": 3}])
def test_validate_for_space_requires_exact_objective_names(config_type, space, field, names):
    config = config_type(**{field: names})
    with pytest.raises(EngineError) as error:
        config.validate_for_space(space)
    assert error.value.code == ErrorCode.CONFIG_INVALID
    assert field in str(error.value)


def test_validate_for_space_rejects_single_objective_even_with_defaults(config_type):
    space = Space(params=[Parameter("x", bounds=(0, 1))], objectives="yield")
    with pytest.raises(EngineError) as error:
        config_type().validate_for_space(space)
    assert error.value.code == ErrorCode.CONFIG_INVALID


@pytest.mark.parametrize("payload", [None, [], "{}", {1: 2}, {}, {"acquisition": "auto"}])
def test_from_dict_requires_complete_mapping(config_type, payload):
    with pytest.raises(EngineError) as error:
        config_type.from_dict(payload)
    assert error.value.code == ErrorCode.CONFIG_INVALID


def test_config_rejects_unknown_fields(config_type):
    with pytest.raises(TypeError):
        config_type(unknown=True)
    payload = config_type().to_dict()
    payload["unknown"] = True
    with pytest.raises(EngineError) as error:
        config_type.from_dict(payload)
    assert error.value.code == ErrorCode.CONFIG_INVALID


@pytest.mark.parametrize("field,value", [("acquisition", "unknown"), ("noise_mode", "unknown"),
                                       ("reference_point", {"yield": True}),
                                       ("preference_weights", {"yield": 0}),
                                       ("min_joint_feasibility", np.nan)])
def test_from_dict_validates_values(config_type, field, value):
    payload = config_type().to_dict()
    payload[field] = value
    with pytest.raises(EngineError) as error:
        config_type.from_dict(payload)
    assert error.value.code == ErrorCode.CONFIG_INVALID


def test_public_exports_and_frozen_acquisition_decision(config_type):
    from expdoe_dk.bo import AcquisitionDecision, MultiObjectiveConfig
    from expdoe_dk.bo.contracts import AcquisitionDecision as ContractDecision
    from expdoe_dk.bo.multiobjective import MultiObjectiveConfig as ModuleConfig

    assert config_type is MultiObjectiveConfig is ModuleConfig
    assert ed.AcquisitionDecision is AcquisitionDecision is ContractDecision
    decision = AcquisitionDecision(kind="q_log_nehvi", reason="inferred_noise")
    assert (decision.kind, decision.reason) == ("q_log_nehvi", "inferred_noise")
    with pytest.raises(FrozenInstanceError):
        decision.kind = "q_log_ehvi"
