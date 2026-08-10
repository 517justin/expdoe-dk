import numpy as np
import pytest

from expdoe_dk.domain import Parameter


@pytest.mark.parametrize(
    ("parameter", "physical"),
    [
        (
            Parameter("x", kind="continuous", bounds=(1.0, 100.0), transform="log"),
            [1.0, 10.0, 100.0],
        ),
        (Parameter("count", kind="integer", bounds=(1, 9), step=2), [1, 5, 9]),
        (Parameter("dose", kind="discrete", values=[0.1, 0.3, 0.8]), [0.1, 0.8]),
        (
            Parameter("solvent", kind="categorical", values=["water", "ethanol"]),
            ["water", "ethanol"],
        ),
        (
            Parameter("grade", kind="ordinal", values=["low", "medium", "high"]),
            ["low", "high"],
        ),
    ],
)
def test_parameter_encode_decode_round_trip(parameter, physical):
    """Catches any supported kind losing physical values in a model-frame round trip."""
    encoded = parameter.encode(physical)
    assert parameter.decode(encoded) == physical


def test_log_transform_rejects_nonpositive_bounds():
    """Catches logarithmic parameters accepting an undefined physical domain."""
    with pytest.raises(ValueError, match="positive"):
        Parameter("x", kind="continuous", bounds=(0.0, 10.0), transform="log")


def test_parameter_exposes_model_bounds_cardinality_and_numeric_levels():
    """Catches mixed-space introspection reporting the wrong finite search grid."""
    continuous = Parameter("x", bounds=(1.0, 2.0))
    integer = Parameter("count", kind="integer", bounds=(1, 8), step=2)
    discrete = Parameter("dose", kind="discrete", values=[0.8, 0.1, 0.3])
    categorical = Parameter("solvent", kind="categorical", values=["water", "ethanol"])
    ordinal = Parameter("grade", kind="ordinal", values=["low", "medium", "high"])

    assert continuous.model_bounds == (0.0, 1.0)
    assert continuous.cardinality is None
    assert integer.model_bounds == (0.0, 1.0)
    assert integer.numeric_levels == (1, 3, 5, 7)
    assert integer.cardinality == 4
    assert discrete.numeric_levels == (0.1, 0.3, 0.8)
    assert discrete.cardinality == 3
    assert categorical.model_bounds == (0.0, 1.0)
    assert categorical.cardinality == 2
    assert ordinal.model_bounds == (0.0, 2.0)
    assert ordinal.cardinality == 3


def test_parameter_normalizes_value_levels_to_an_immutable_tuple():
    """Catches caller-owned lists mutating a frozen parameter after construction."""
    values = ["A", "B"]
    parameter = Parameter("binder", kind="categorical", values=values)

    values.append("C")

    assert parameter.values == ("A", "B")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "continuous"},
        {"kind": "continuous", "bounds": (0.0, np.inf)},
        {"kind": "continuous", "bounds": (1.0, 1.0)},
        {"kind": "continuous", "bounds": (0.0, 1.0), "step": 0.1},
        {"kind": "continuous", "bounds": (0.0, 1.0), "values": [0.0, 1.0]},
        {"kind": "integer", "bounds": (1.5, 5)},
        {"kind": "integer", "bounds": (1, 5), "step": 1.5},
        {"kind": "integer", "bounds": (1, 5), "values": [1, 3, 5]},
        {"kind": "discrete", "values": [0.1, 0.1]},
        {"kind": "discrete", "values": [0.1, np.nan]},
        {"kind": "discrete", "bounds": (0.0, 1.0)},
        {"kind": "discrete", "bounds": (0.0, 1.0), "step": 0.2, "values": [0.0, 1.0]},
        {"kind": "categorical", "values": ["A", "A"]},
        {"kind": "categorical", "values": ["A", 2]},
        {"kind": "categorical", "bounds": (0.0, 1.0), "values": ["A", "B"]},
        {"kind": "ordinal", "values": ["low", "low"]},
        {"kind": "ordinal", "values": ["low", ["high"]]},
        {"kind": "ordinal", "values": ["low", "high"], "step": 1},
        {"kind": "categorical", "values": ["A", "B"], "transform": "log"},
        {"kind": "unsupported", "bounds": (0.0, 1.0)},
        {"kind": "continuous", "bounds": (0.0, 1.0), "transform": "square"},
    ],
)
def test_parameter_rejects_invalid_kind_specific_configuration(kwargs):
    """Catches invalid tagged configurations reaching space construction."""
    with pytest.raises(ValueError):
        Parameter("factor", **kwargs)


def test_integer_defaults_to_unit_step_and_discrete_grid_never_exceeds_bounds():
    """Catches finite grids omitting the integer default or stepping past the upper bound."""
    integer = Parameter("count", kind="integer", bounds=(1, 3))
    discrete = Parameter("dose", kind="discrete", bounds=(0.0, 1.0), step=0.3)

    assert integer.numeric_levels == (1, 2, 3)
    assert discrete.numeric_levels == pytest.approx((0.0, 0.3, 0.6, 0.9))


def test_discrete_and_integer_decode_snap_to_the_nearest_physical_level():
    """Catches model decoding returning off-grid physical experiment settings."""
    integer = Parameter("count", kind="integer", bounds=(1, 9), step=2)
    discrete = Parameter("dose", kind="discrete", values=[0.1, 0.3, 0.8])

    assert integer.decode([0.45]) == [5]
    assert discrete.decode([0.5]) == [0.3]


def test_transform_aliases_match_encode_and_decode():
    """Catches the public physical/model transform names diverging from their aliases."""
    parameter = Parameter("x", bounds=(1.0, 100.0), log_scale=True)

    encoded = parameter.to_model([1.0, 10.0, 100.0])

    assert encoded == pytest.approx([0.0, 0.5, 1.0])
    assert parameter.to_physical(encoded) == pytest.approx([1.0, 10.0, 100.0])


def test_log_transform_rejects_nonpositive_explicit_discrete_levels():
    """Catches log scaling explicit discrete values that are outside the log domain."""
    with pytest.raises(ValueError, match="positive"):
        Parameter("dose", kind="discrete", values=[0.0, 1.0], transform="log")
