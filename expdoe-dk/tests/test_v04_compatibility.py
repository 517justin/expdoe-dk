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
        "count", kind="integer", bounds=(low, low + 4), step=2
    )
    values = torch.tensor([low + 3], dtype=torch.int64)

    snapped = parameter.snap(values)

    assert torch.equal(snapped, torch.tensor([low + 2], dtype=torch.int64))
