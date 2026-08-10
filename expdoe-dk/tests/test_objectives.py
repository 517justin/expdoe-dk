import numpy as np
import pytest

from expdoe_dk import Objective
from expdoe_dk.domain import normalize_objectives
from expdoe_dk.errors import EngineError, ErrorCode


def test_target_range_utility_is_zero_inside_and_negative_outside():
    """Catches target-range utility treating in-range values as a penalty."""
    objective = Objective("viscosity", direction="target", target=(10.0, 12.0))

    actual = objective.to_utility(np.array([9.0, 10.0, 11.0, 12.0, 14.0]))

    assert actual.tolist() == [-1.0, 0.0, 0.0, 0.0, -2.0]


def test_objective_rejects_target_for_maximize():
    """Catches maximize objectives silently accepting target-only configuration."""
    with pytest.raises(EngineError) as caught:
        Objective("yield", direction="maximize", target=90.0)

    assert caught.value.code is ErrorCode.CONFIG_INVALID


def test_point_target_utility_penalizes_distance_on_either_side():
    """Catches point target utility favoring values on one side of the target."""
    objective = Objective("density", direction="target", target=5.0)

    actual = objective.to_utility(np.array([3.0, 5.0, 6.5]))

    assert actual.tolist() == [-2.0, 0.0, -1.5]


def test_objective_rejects_boolean_priority_even_though_bool_is_an_int():
    """Catches bool priorities bypassing the non-negative integer rank rule."""
    with pytest.raises(EngineError) as caught:
        Objective("yield", direction="maximize", priority=True)

    assert caught.value.code is ErrorCode.CONFIG_INVALID


def test_normalize_objectives_broadcasts_flag_and_preserves_explicit_objective():
    """Catches normalization overwriting an explicitly configured objective."""
    explicit = Objective("cost", direction="minimize", priority=2)

    actual = normalize_objectives(("yield", explicit), maximize=True)

    assert actual == (
        Objective("yield", direction="maximize"),
        explicit,
    )


def test_normalize_objectives_rejects_mismatched_direction_flags():
    """Catches truncating zip behavior when objectives and flags have different lengths."""
    with pytest.raises(EngineError) as caught:
        normalize_objectives(("yield", "cost"), maximize=(True,))

    assert caught.value.code is ErrorCode.CONFIG_INVALID


def test_engine_error_keeps_code_message_details_and_hints_for_serialization():
    """Catches typed errors dropping structured context needed by engine callers."""
    error = EngineError(
        ErrorCode.SPACE_INFEASIBLE,
        "No feasible rows",
        details={"constraint": "A >= B"},
        hints=("Relax the lower bound",),
    )

    assert error.code.value == "SPACE_INFEASIBLE"
    assert error.message == "No feasible rows"
    assert error.details == {"constraint": "A >= B"}
    assert error.hints == ("Relax the lower bound",)
