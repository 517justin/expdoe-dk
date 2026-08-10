import pandas as pd
import pytest

from expdoe_dk import Parameter, Space, suggest_design
from expdoe_dk.domain import ExpressionConstraint
from expdoe_dk.doe.design import candidate_keys
from expdoe_dk.errors import EngineError, ErrorCode


@pytest.fixture
def numeric_space():
    return Space([Parameter("x", bounds=(0.0, 1.0))])


@pytest.fixture
def mixed_space():
    return Space(
        [
            Parameter("temperature", bounds=(300.0, 400.0)),
            Parameter("binder", kind="categorical", values=["A", "B"]),
            Parameter("grade", kind="ordinal", values=[1, None]),
        ]
    )


@pytest.fixture
def existing_mixed_rows():
    return pd.DataFrame(
        {
            "temperature": [350.0],
            "binder": pd.Series(["A"], dtype=object),
            "grade": pd.Series([None], dtype=object),
        }
    )


@pytest.fixture
def tiny_discrete_space():
    return Space(
        [
            Parameter("a", kind="discrete", values=[0.0, 1.0]),
            Parameter("b", kind="discrete", values=[0.0, 1.0]),
        ]
    )


def test_compiled_safety_constraints_filter_initial_design(numeric_space):
    safe = ExpressionConstraint("safe-lower", "x >= 0.2")
    forbidden_complement = ExpressionConstraint(
        "forbidden-upper-complement", "x <= 0.8"
    )

    batch = suggest_design(
        numeric_space,
        8,
        method="sobol",
        seed=9,
        parameter_constraints=(safe, forbidden_complement),
        knowledge_sources=("safe-1", "forbidden-1"),
        return_diagnostics=True,
    )

    assert batch.frame["x"].between(0.2, 0.8, inclusive="both").all()
    assert batch.diagnostics.knowledge_sources == ("safe-1", "forbidden-1")


def test_mixed_design_is_deterministic_balanced_and_avoids_existing(
    mixed_space, existing_mixed_rows
):
    first = suggest_design(
        mixed_space,
        6,
        method="sobol",
        seed=7,
        existing=existing_mixed_rows,
        return_diagnostics=True,
    )
    second = suggest_design(
        mixed_space,
        6,
        method="sobol",
        seed=7,
        existing=existing_mixed_rows,
        return_diagnostics=True,
    )

    pd.testing.assert_frame_equal(first.frame, second.frame)
    assert not candidate_keys(first.frame, mixed_space) & candidate_keys(
        existing_mixed_rows, mixed_space
    )
    counts = first.frame["binder"].value_counts()
    assert counts.max() - counts.min() <= 1


def test_finite_space_shortage_is_structured(tiny_discrete_space):
    with pytest.raises(EngineError) as caught:
        suggest_design(tiny_discrete_space, 5, method="sobol", seed=2)

    assert caught.value.code is ErrorCode.SPACE_INFEASIBLE
    assert caught.value.details["requested"] == 5
    assert caught.value.details["available_cardinality"] == 4
