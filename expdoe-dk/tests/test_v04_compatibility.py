from pathlib import Path

import expdoe_dk as ed
import numpy as np
import pytest

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
