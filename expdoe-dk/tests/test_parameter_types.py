import numpy as np
import pytest
import torch

from expdoe_dk.domain import Parameter


def test_adjacent_integral_domains_above_float_precision_remain_distinct():
    """Catches construction collapsing adjacent exact integers through float64."""
    low = 2**53

    integer = Parameter("count", kind="integer", bounds=(low, low + 1))
    discrete = Parameter("dose", kind="discrete", values=[low, low + 1])

    assert integer.numeric_levels == (low, low + 1)
    assert discrete.numeric_levels == (low, low + 1)
    assert integer.decode(integer.encode([low, low + 1])) == [low, low + 1]
    assert discrete.decode(discrete.encode([low, low + 1])) == [low, low + 1]


def test_adjacent_integral_log_levels_preserve_their_model_span():
    """Catches log subtraction collapsing adjacent arbitrary-precision levels."""
    low = 2**53
    parameter = Parameter(
        "dose", kind="discrete", values=[low, low + 1], transform="log"
    )

    encoded = parameter.encode([low, low + 1])

    assert encoded == pytest.approx([0.0, 1.0])
    assert parameter.decode(encoded) == [low, low + 1]


def test_integer_midpoint_ties_choose_the_upper_level_on_every_backend():
    """Catches decode, Python, NumPy, and Torch choosing different tie sides."""
    low = 2**53
    parameter = Parameter("count", kind="integer", bounds=(low, low + 4), step=4)
    midpoint = low + 2
    upper = low + 4

    assert parameter.decode([0.5]) == [upper]
    assert parameter.snap([midpoint]).tolist() == [upper]
    assert parameter.snap(np.array([midpoint], dtype=np.int64)).tolist() == [upper]
    assert parameter.snap(np.array([float(midpoint)], dtype=np.float64)).tolist() == [upper]
    assert parameter.snap(torch.tensor([midpoint], dtype=torch.int64)).tolist() == [upper]
    assert parameter.snap(torch.tensor([float(midpoint)], dtype=torch.float64)).tolist() == [
        float(upper)
    ]


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


@pytest.mark.parametrize(
    ("kind", "bounds", "step", "expected"),
    [
        ("integer", (0, 1_000_000_000_000), 2, 500_000_000_001),
        ("discrete", (0.0, 1_000_000_000_000.0), 0.25, 4_000_000_000_001),
    ],
)
def test_large_bounded_cardinality_does_not_enumerate_levels(kind, bounds, step, expected):
    """Catches cardinality allocating an unsafe multi-trillion-level grid."""

    class EnumerationForbiddenParameter(Parameter):
        @property
        def numeric_levels(self):
            raise AssertionError("cardinality enumerated numeric levels")

    parameter = EnumerationForbiddenParameter(
        "factor", kind=kind, bounds=bounds, step=step
    )

    assert parameter.cardinality == expected


def test_parameter_normalizes_value_levels_to_an_immutable_tuple():
    """Catches caller-owned lists mutating a frozen parameter after construction."""
    values = ["A", "B"]
    parameter = Parameter("binder", kind="categorical", values=values)

    values.append("C")

    assert parameter.values == ("A", "B")


@pytest.mark.parametrize(
    ("kind", "values"),
    [
        ("categorical", "AB"),
        ("discrete", b"\x01\x02"),
        ("categorical", {"A", "B"}),
        ("ordinal", frozenset({"low", "high"})),
        ("ordinal", {"low": 0, "high": 1}),
    ],
)
def test_explicit_levels_reject_unordered_or_scalar_iterables(kind, values):
    """Catches level order depending on scalar, set, or mapping iteration."""
    with pytest.raises(ValueError, match="ordered sequence"):
        Parameter("factor", kind=kind, values=values)


def test_explicit_levels_accept_ordered_tuple_input_deterministically():
    """Catches tuple inputs losing their declared categorical order."""
    parameter = Parameter("grade", kind="ordinal", values=("high", "medium", "low"))

    assert parameter.values == ("high", "medium", "low")
    assert parameter.encode(["high", "low"]) == [0.0, 2.0]


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


def test_decimal_bounded_grid_canonicalizes_reachable_high_endpoint():
    """Catches binary rounding pushing a reachable grid endpoint out of bounds."""
    parameter = Parameter("dose", kind="discrete", bounds=(0.0, 0.3), step=0.1)

    assert parameter.numeric_levels == (0.0, 0.1, 0.2, 0.3)
    assert parameter.decode(parameter.encode([0.3])) == [0.3]


def test_negative_decimal_grid_sets_integral_ratio_endpoint_exactly_to_high():
    """Catches signed decimal accumulation leaving a reachable zero off-grid."""
    parameter = Parameter("dose", kind="discrete", bounds=(-0.3, 0.0), step=0.1)

    assert parameter.numeric_levels[-1] == 0.0
    assert parameter.decode(parameter.encode([0.0])) == [0.0]


def test_discrete_membership_does_not_use_magnitude_relative_tolerance():
    """Catches large explicit levels accepting a physically distinct nearby value."""
    parameter = Parameter(
        "dose",
        kind="discrete",
        values=[1_000_000_000_000_000.0, 1_000_000_000_000_010.0],
    )

    with pytest.raises(ValueError, match="declared levels"):
        parameter.encode([1_000_000_000_000_001.0])


def test_discrete_and_integer_decode_snap_to_the_nearest_physical_level():
    """Catches model decoding returning off-grid physical experiment settings."""
    integer = Parameter("count", kind="integer", bounds=(1, 9), step=2)
    discrete = Parameter("dose", kind="discrete", values=[0.1, 0.3, 0.8])

    assert integer.decode([0.45]) == [5]
    assert discrete.decode([0.5]) == [0.3]


@pytest.mark.parametrize("physical", [1_000_000_000_000_000.25, 1_000_000_000_000_001])
def test_integer_encode_requires_exact_grid_membership_at_large_magnitudes(physical):
    """Catches relative float tolerance accepting non-integral or off-grid integers."""
    parameter = Parameter(
        "count",
        kind="integer",
        bounds=(1_000_000_000_000_000, 1_000_000_000_000_004),
        step=2,
    )

    with pytest.raises(ValueError, match="declared levels"):
        parameter.encode([physical])


def test_integer_encode_rejects_exact_value_above_bound_before_float_conversion():
    """Catches float64 rounding an out-of-bounds Python integer onto the upper bound."""
    parameter = Parameter("count", kind="integer", bounds=(0, 2**53))

    with pytest.raises(ValueError, match="within"):
        parameter.encode([2**53 + 1])


def test_integer_round_trip_preserves_python_ints_above_int64():
    """Catches integer snapping overflowing while coercing valid Python ints to int64."""
    physical = [2**63, 2**63 + 2048, 2**63 + 4096]
    parameter = Parameter(
        "count", kind="integer", bounds=(2**63, 2**63 + 4096), step=2048
    )

    encoded = parameter.encode(physical)
    snapped = parameter.snap(np.array([float(2**63 + 2048)]))
    decoded = parameter.decode(encoded)

    assert encoded == [0.0, 0.5, 1.0]
    assert snapped.tolist() == [2**63 + 2048]
    assert type(snapped.tolist()[0]) is int
    assert decoded == physical
    assert all(type(value) is int for value in decoded)


def test_fine_integer_grid_above_int64_keeps_adjacent_python_int_exact():
    """Catches float64 collapsing adjacent valid levels in a large integer grid."""
    low = 2**63
    physical = low + 1
    parameter = Parameter("fine", kind="integer", bounds=(low, low + 4096), step=1)

    encoded = parameter.encode([physical])
    decoded = parameter.decode(encoded)
    snapped = parameter.snap([physical])

    assert encoded == [1 / 4096]
    assert decoded == [physical]
    assert type(decoded[0]) is int
    assert snapped.tolist() == [physical]
    assert type(snapped.tolist()[0]) is int


def test_integer_log_transform_uses_physical_log_coordinates_before_snapping():
    """Catches integer encoding bypassing its declared physical log transform."""
    parameter = Parameter(
        "count", kind="integer", bounds=(1, 9), step=2, transform="log"
    )

    encoded = parameter.encode([1, 3, 5, 7, 9])

    assert encoded == pytest.approx(
        [0.0, 0.5, np.log(5) / np.log(9), np.log(7) / np.log(9), 1.0]
    )
    assert parameter.decode([0.5]) == [3]


def test_nondividing_integer_grid_uses_declared_linear_physical_bounds():
    """Catches integer encoding normalizing against its last reachable level."""
    parameter = Parameter("count", kind="integer", bounds=(1, 8), step=2)

    assert parameter.encode([7]) == [6 / 7]
    assert parameter.decode([0.0, 6 / 7, 1.0]) == [1, 7, 7]


def test_integer_snap_and_decode_return_integer_types():
    """Catches integer factors returning float experiment settings."""
    parameter = Parameter("count", kind="integer", bounds=(1, 9), step=2)

    snapped = parameter.snap(np.array([4.8]))
    decoded = parameter.decode([0.5])

    assert np.issubdtype(snapped.dtype, np.integer)
    assert snapped.tolist() == [5]
    assert decoded == [5]
    assert type(decoded[0]) is int


@pytest.mark.parametrize(
    "parameter",
    [
        Parameter("x", bounds=(0.0, 1.0)),
        Parameter("count", kind="integer", bounds=(1, 9), step=2),
        Parameter("dose", kind="discrete", values=[0.1, 0.3, 0.8]),
    ],
)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_snap_rejects_nonfinite_numeric_values(parameter, value):
    """Catches non-finite physical proposals silently snapping to a valid level."""
    with pytest.raises(ValueError, match="finite"):
        parameter.snap(np.array([value]))


def test_transform_aliases_match_encode_and_decode():
    """Catches the public physical/model transform names diverging from their aliases."""
    parameter = Parameter("x", bounds=(1.0, 100.0), log_scale=True)

    encoded = parameter.to_model([1.0, 10.0, 100.0])

    assert encoded == pytest.approx([0.0, 0.5, 1.0])
    assert parameter.to_physical(encoded) == pytest.approx([1.0, 10.0, 100.0])


def test_wide_linear_bounds_keep_model_and_physical_coordinates_finite():
    """Catches affine span overflow corrupting finite endpoint and midpoint transforms."""
    parameter = Parameter("x", bounds=(-1e308, 1e308))

    encoded = parameter.encode([-1e308, 0.0, 1e308])
    decoded = parameter.decode([0.0, 0.5, 1.0])

    assert encoded == [0.0, 0.5, 1.0]
    assert decoded == [-1e308, 0.0, 1e308]
    assert np.isfinite(encoded).all()
    assert np.isfinite(decoded).all()


def test_wide_log_bounds_keep_model_and_physical_coordinates_finite():
    """Catches positive-ratio overflow corrupting finite logarithmic transforms."""
    parameter = Parameter("x", bounds=(1e-300, 1e300), transform="log")

    encoded = parameter.encode([1e-300, 1.0, 1e300])
    decoded = parameter.decode([0.0, 0.5, 1.0])

    assert encoded == [0.0, 0.5, 1.0]
    assert decoded == [1e-300, 1.0, 1e300]
    assert np.isfinite(encoded).all()
    assert np.isfinite(decoded).all()


@pytest.mark.parametrize("encoded", [np.nan, np.inf, -np.inf, -0.001, 1.001])
def test_decode_rejects_nonfinite_or_out_of_model_bounds(encoded):
    """Catches invalid model coordinates escaping physical-frame validation."""
    parameter = Parameter("x", bounds=(0.0, 10.0))

    with pytest.raises(ValueError):
        parameter.decode([encoded])


@pytest.mark.parametrize(
    ("kwargs", "physical"),
    [
        ({"kind": "integer", "bounds": (1, 9), "step": 2}, [1, 5, 9]),
        ({"kind": "discrete", "values": [0.1, 1.0, 10.0]}, [0.1, 1.0, 10.0]),
    ],
)
def test_positive_finite_kinds_round_trip_with_transform_and_log_scale_alias(
    kwargs, physical
):
    """Catches the declared log aliases diverging for integer or discrete factors."""
    transformed = Parameter("factor", transform="log", **kwargs)
    aliased = Parameter("factor", log_scale=True, **kwargs)

    transformed_encoded = transformed.encode(physical)
    aliased_encoded = aliased.encode(physical)

    assert transformed_encoded == pytest.approx(aliased_encoded)
    assert transformed.decode(transformed_encoded) == physical
    assert aliased.decode(aliased_encoded) == physical


def test_log_transform_rejects_nonpositive_explicit_discrete_levels():
    """Catches log scaling explicit discrete values that are outside the log domain."""
    with pytest.raises(ValueError, match="positive"):
        Parameter("dose", kind="discrete", values=[0.0, 1.0], transform="log")
