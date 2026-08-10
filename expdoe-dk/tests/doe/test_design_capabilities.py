import pytest

from expdoe_dk import Parameter, Space, suggest_design
from expdoe_dk.errors import EngineError, ErrorCode


@pytest.fixture
def mixed_space():
    return Space(
        [
            Parameter("temperature", bounds=(300.0, 400.0)),
            Parameter("binder", kind="categorical", values=["A", "B"]),
        ]
    )


def test_auto_selects_sobol_for_nominal_categorical_space(mixed_space):
    batch = suggest_design(
        mixed_space, 6, method="auto", seed=4, return_diagnostics=True
    )

    assert batch.diagnostics.requested_method == "auto"
    assert batch.diagnostics.effective_method == "sobol"
    assert batch.diagnostics.selection_rule == "auto:nominal-categorical->sobol"


def test_explicit_d_optimal_rejects_categorical_without_fallback(mixed_space):
    with pytest.raises(EngineError) as caught:
        suggest_design(mixed_space, 6, method="d_optimal", seed=4)

    assert caught.value.code is ErrorCode.CONFIG_INVALID
    assert "sobol" in caught.value.details["compatible_methods"]
