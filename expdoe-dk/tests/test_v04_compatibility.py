from pathlib import Path

import expdoe_dk as ed
import numpy as np
import pytest
import torch

from expdoe_dk.domain import LinearConstraint as DomainLinearConstraint
from expdoe_dk.domain import Parameter as DomainParameter


def test_v04_public_imports_and_checkpoint_remain_readable():
    assert ed.Parameter.__name__ == "Parameter"
    assert ed.LinearConstraint.__name__ == "LinearConstraint"
    assert ed.Space.__name__ == "Space"
    assert ed.Result.__name__ == "Result"
    checkpoint = Path(__file__).parent / "fixtures" / "v04_checkpoint.json"
    campaign = ed.Campaign.load_checkpoint(checkpoint)
    assert campaign.space.objectives == ["yield"]
    assert campaign.history_df()["y"].tolist() == [1.5]


def test_legacy_doe_generate_remains_a_dataframe_wrapper():
    from expdoe_dk.doe import generate

    frame = generate(
        ed.Space([ed.Parameter("x", bounds=(0.0, 1.0))]),
        3,
        method="lhs_maximin",
        seed=13,
        n_iterations=20,
    )

    assert frame.__class__.__name__ == "DataFrame"
    assert list(frame.columns) == ["x"]
    assert len(frame) == 3


def test_v04_knowledge_payload_round_trips():
    payload = {
        "strict": False,
        "items": [
            {
                "kind": "monotone",
                "param": "x",
                "effect": "increases_objective",
                "n_pairs_per_dim": 5,
                "epsilon": "auto",
                "delta_norm": 0.5,
            }
        ],
    }
    restored = ed.Knowledge.from_dict(payload)
    assert restored.to_dict() == payload


def test_v04_parameter_sixth_positional_argument_remains_log_scale():
    """Catches the new values field stealing the legacy log_scale position."""
    parameter = ed.Parameter(
        "rate", (1.0, 100.0), "s-1", "continuous", None, True
    )

    assert parameter.log_scale is True
    assert parameter.values is None
    assert parameter.encode([1.0, 10.0, 100.0]) == pytest.approx([0.0, 0.5, 1.0])


def test_v04_numeric_levels_remain_numpy_arrays():
    """Catches the public domain parameter dropping the legacy levels member."""
    discrete = ed.Parameter("dose", (0.0, 1.0), "mL", "discrete", 0.4)
    integer = ed.Parameter("count", bounds=(1, 5), kind="integer", step=2)

    assert isinstance(discrete.levels, np.ndarray)
    assert discrete.levels.tolist() == [0.0, 0.4, 0.8]
    assert isinstance(integer.levels, np.ndarray)
    assert integer.levels.tolist() == [1, 3, 5]
    with pytest.raises(AttributeError):
        ed.Parameter("temperature", bounds=(20.0, 30.0)).levels


def test_v04_linear_constraint_keeps_two_sided_evaluation_contract():
    """Catches the declarative one-sided schema replacing the public v0.4 class."""
    constraint = ed.LinearConstraint(
        coeffs={"a": 1.0, "b": -1.0}, lower=1.0, upper=3.0, name="gap"
    )

    assert constraint.evaluate({"a": 5.0, "b": 3.0}) == 2.0
    assert constraint.satisfied({"a": 5.0, "b": 3.0})
    assert not constraint.satisfied({"a": 7.0, "b": 3.0})
    assert constraint.describe() == "gap"


def test_public_parameter_and_space_are_domain_identities_but_linear_is_adapter():
    """Catches facade re-exports drifting from the composed domain model."""
    from expdoe_dk.domain import Space as DomainSpace

    assert ed.Parameter is DomainParameter
    assert ed.Space is DomainSpace
    assert ed.LinearConstraint is not DomainLinearConstraint


@pytest.mark.parametrize(
    ("parameter", "values", "expected"),
    [
        (ed.Parameter("x", bounds=(0.0, 10.0)), [1.25, 8.75], [1.25, 8.75]),
        (
            ed.Parameter("dose", kind="discrete", values=[0.1, 0.5, 0.9]),
            [0.2, 0.8],
            [0.1, 0.9],
        ),
        (
            ed.Parameter("count", kind="integer", bounds=(1, 9), step=2),
            [1.2, 4.8],
            [1.0, 5.0],
        ),
    ],
)
def test_v04_parameter_snap_preserves_torch_backend_dtype_and_device(
    parameter, values, expected
):
    """Catches legacy tensor snapping returning a host NumPy array."""
    original = torch.tensor(values, dtype=torch.float32)

    snapped = parameter.snap(original)

    assert isinstance(snapped, torch.Tensor)
    assert snapped.dtype == original.dtype
    assert snapped.device == original.device
    torch.testing.assert_close(
        snapped, torch.tensor(expected, dtype=original.dtype, device=original.device)
    )
    if parameter.kind == "continuous":
        assert snapped is original


@pytest.mark.parametrize(
    "parameter",
    [
        ed.Parameter("x", bounds=(0.0, 10.0)),
        ed.Parameter("dose", kind="discrete", values=[0.1, 0.5, 0.9]),
        ed.Parameter("count", kind="integer", bounds=(1, 9), step=2),
    ],
)
def test_v04_parameter_snap_keeps_numpy_inputs_numpy(parameter):
    """Catches backend preservation accidentally changing NumPy callers."""
    snapped = parameter.snap(np.array([0.2, 4.8], dtype=np.float32))

    assert isinstance(snapped, np.ndarray)


@pytest.mark.parametrize(
    "parameter",
    [
        ed.Parameter("x", bounds=(0.0, 10.0)),
        ed.Parameter("dose", kind="discrete", values=[0.1, 0.5, 0.9]),
        ed.Parameter("count", kind="integer", bounds=(1, 9), step=2),
    ],
)
def test_v04_parameter_snap_rejects_nonfinite_torch_inputs(parameter):
    """Catches tensor support bypassing Task 3 finite-input validation."""
    with pytest.raises(ValueError, match="finite"):
        parameter.snap(torch.tensor([float("nan")]))


def test_v04_integer_tensor_snap_stays_exact_above_float_precision():
    """Catches an int64 tensor taking a lossy float64 snapping detour."""
    low = 2**53
    parameter = ed.Parameter(
        "count", kind="integer", bounds=(low, low + 9), step=3
    )
    values = torch.tensor([low + 5], dtype=torch.int64)

    snapped = parameter.snap(values)

    assert torch.equal(snapped, torch.tensor([low + 6], dtype=torch.int64))


@pytest.mark.parametrize(
    ("parameter", "dtype", "values", "expected"),
    [
        (
            ed.Parameter("count", kind="integer", bounds=(10, 20), step=2),
            torch.uint8,
            [0, 9],
            [10, 10],
        ),
        (
            ed.Parameter("count", kind="integer", bounds=(-100, 100), step=10),
            torch.int8,
            [95, 100, 127],
            [100, 100, 100],
        ),
        (
            ed.Parameter("dose", kind="discrete", values=[10, 12, 20]),
            torch.uint8,
            [0, 9],
            [10, 10],
        ),
        (
            ed.Parameter("dose", kind="discrete", values=[-100, 0, 100]),
            torch.int8,
            [95, 100, 127],
            [100, 100, 100],
        ),
    ],
)
def test_integral_tensor_snap_widens_before_arithmetic_and_matches_numpy(
    parameter, dtype, values, expected
):
    """Catches uint8/int8 subtraction and distances wrapping before snap."""
    torch_values = torch.tensor(values, dtype=dtype)
    numpy_values = np.asarray(values, dtype=str(dtype).removeprefix("torch."))

    tensor_snapped = parameter.snap(torch_values)
    numpy_snapped = parameter.snap(numpy_values)

    assert tensor_snapped.dtype == dtype
    assert tensor_snapped.device == torch_values.device
    assert tensor_snapped.tolist() == expected
    assert numpy_snapped.tolist() == expected


@pytest.mark.parametrize(
    ("parameter", "values"),
    [
        (
            ed.Parameter("count", kind="integer", bounds=(100, 200), step=10),
            torch.tensor([127], dtype=torch.int8),
        ),
        (
            ed.Parameter("dose", kind="discrete", values=[0, 300]),
            torch.tensor([255], dtype=torch.uint8),
        ),
    ],
)
def test_integral_tensor_snap_rejects_unrepresentable_result(parameter, values):
    """Catches snapped levels wrapping when cast back to the input dtype."""
    with pytest.raises(ValueError, match="representable|dtype"):
        parameter.snap(values)


def test_int64_tensor_snap_matches_numpy_across_full_signed_range():
    """Catches subtracting INT64_MIN from an int64 tensor in fixed width."""
    info = np.iinfo(np.int64)
    parameter = ed.Parameter(
        "count", kind="integer", bounds=(info.min, info.max), step=2
    )
    tensor_values = torch.tensor([0], dtype=torch.int64)
    numpy_values = np.array([0], dtype=np.int64)

    tensor_snapped = parameter.snap(tensor_values)
    numpy_snapped = parameter.snap(numpy_values)

    assert numpy_snapped.tolist() == [0]
    assert tensor_snapped.tolist() == numpy_snapped.tolist()
    assert tensor_snapped.dtype == tensor_values.dtype
    assert tensor_snapped.device == tensor_values.device


def test_int64_discrete_snap_uses_arbitrary_precision_distances():
    """Catches abs(INT64_MIN) overflowing and appearing nearest to zero."""
    parameter = ed.Parameter(
        "dose", kind="discrete", values=[np.iinfo(np.int64).min, 2**62]
    )
    values = torch.tensor([0], dtype=torch.int64)

    snapped = parameter.snap(values)

    assert torch.equal(snapped, torch.tensor([2**62], dtype=torch.int64))
    assert snapped.device == values.device


def test_uint64_tensor_snap_preserves_unsigned_dtype_and_device():
    """Catches representability checks comparing tensors to uint64 maxima."""
    parameter = ed.Parameter("count", kind="integer", bounds=(0, 10), step=2)
    values = torch.tensor([5], dtype=torch.uint64)

    snapped = parameter.snap(values)

    assert torch.equal(snapped, torch.tensor([6], dtype=torch.uint64))
    assert snapped.dtype == values.dtype
    assert snapped.device == values.device


def test_integral_numpy_and_tensor_discrete_snap_match_above_float_precision():
    """Catches NumPy integral input taking a lossy float64 distance path."""
    low = 2**53 - 100
    high = 2**53 + 100
    parameter = ed.Parameter("dose", kind="discrete", values=[low, high])
    numpy_values = np.array([2**53 + 1], dtype=np.int64)
    tensor_values = torch.tensor([2**53 + 1], dtype=torch.int64)

    numpy_snapped = parameter.snap(numpy_values)
    tensor_snapped = parameter.snap(tensor_values)

    assert numpy_snapped.tolist() == [high]
    assert tensor_snapped.tolist() == numpy_snapped.tolist()
    assert numpy_snapped.dtype == numpy_values.dtype


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (np.array(15, dtype=np.uint8), 16),
        (np.empty((0, 2), dtype=np.int16), np.empty((0, 2), dtype=np.int16)),
        (
            np.array([[0, 9], [11, 20]], dtype=np.uint8),
            np.array([[10, 10], [12, 20]], dtype=np.uint8),
        ),
    ],
)
def test_integral_numpy_snap_preserves_dtype_and_shape(values, expected):
    """Catches exact ndarray snapping flattening or widening representable data."""
    parameter = ed.Parameter("count", kind="integer", bounds=(10, 20), step=2)

    snapped = parameter.snap(values)

    assert snapped.dtype == values.dtype
    assert snapped.shape == values.shape
    np.testing.assert_array_equal(snapped, expected)


def test_integral_numpy_snap_uses_lossless_object_when_dtype_cannot_hold_result():
    """Catches unrepresentable exact levels wrapping back into a narrow ndarray."""
    parameter = ed.Parameter("dose", kind="discrete", values=[0, 300])

    snapped = parameter.snap(np.array([255], dtype=np.uint8))

    assert snapped.dtype == object
    assert snapped.tolist() == [300]
    assert type(snapped.tolist()[0]) is int


@pytest.mark.parametrize(
    "values",
    [
        np.array([2**64 + 1], dtype=object),
        [2**64 + 1],
        (2**64 + 1,),
    ],
)
def test_arbitrary_precision_object_and_sequence_snap_stays_exact(values):
    """Catches object/list/tuple inputs taking a lossy float64 distance path."""
    base = 2**64
    parameter = ed.Parameter(
        "dose", kind="discrete", values=[base - 4096, base + 4096]
    )

    snapped = parameter.snap(values)

    assert isinstance(snapped, np.ndarray)
    assert snapped.dtype == object
    assert snapped.tolist() == [base + 4096]
    assert type(snapped.tolist()[0]) is int


@pytest.mark.parametrize(
    "values",
    [
        np.array([2**53 + 1], dtype=object),
        [2**53 + 1],
        (2**53 + 1,),
    ],
)
def test_object_and_sequence_snap_match_above_float_precision(values):
    """Catches exact sequence parity regressing below the uint64 boundary."""
    base = 2**53
    parameter = ed.Parameter(
        "dose", kind="discrete", values=[base - 100, base + 100]
    )

    snapped = parameter.snap(values)

    assert snapped.tolist() == [base + 100]
    assert all(type(value) is int for value in snapped.reshape(-1).tolist())


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (
            np.array(2**64 + 1, dtype=object),
            np.array(2**64 + 4096, dtype=object),
        ),
        (
            np.empty((0, 2), dtype=object),
            np.empty((0, 2), dtype=object),
        ),
        (
            np.array(
                [
                    [2**64 - 5000, 2**64 + 1],
                    [2**64 + 5000, 2**64],
                ],
                dtype=object,
            ),
            np.array(
                [
                    [2**64 - 4096, 2**64 + 4096],
                    [2**64 + 4096, 2**64 - 4096],
                ],
                dtype=object,
            ),
        ),
    ],
)
def test_integral_object_array_snap_preserves_shape(values, expected):
    """Catches scalar, empty, or multidimensional object arrays losing shape."""
    base = 2**64
    parameter = ed.Parameter(
        "dose", kind="discrete", values=[base - 4096, base + 4096]
    )

    snapped = parameter.snap(values)

    assert snapped.dtype == object
    assert snapped.shape == values.shape
    np.testing.assert_array_equal(snapped, expected)


@pytest.mark.parametrize(
    "values",
    [
        [True, False],
        [1, True],
        (np.int64(1), np.bool_(False)),
        np.array([True, False], dtype=bool),
        np.array([1, True], dtype=object),
    ],
)
def test_non_tensor_snap_rejects_boolean_and_mixed_boolean_inputs(values):
    """Catches booleans being silently interpreted as exact integers."""
    parameter = ed.Parameter("dose", kind="discrete", values=[0, 2])

    with pytest.raises(ValueError, match="boolean|numeric"):
        parameter.snap(values)


@pytest.mark.parametrize(
    "parameter",
    [
        ed.Parameter("x", bounds=(0.0, 10.0)),
        ed.Parameter("count", kind="integer", bounds=(1, 9), step=2),
        ed.Parameter("dose", kind="discrete", values=[0.1, 0.5, 0.9]),
    ],
)
def test_numpy_snap_rejects_empty_boolean_arrays(parameter):
    """Catches empty bool ndarrays bypassing element-wise bool validation."""
    with pytest.raises(ValueError, match="boolean|numeric"):
        parameter.snap(np.empty((0, 2), dtype=bool))
