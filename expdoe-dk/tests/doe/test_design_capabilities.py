import pandas as pd
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


def test_auto_selects_lhs_maximin_without_nominal_categorical_factor():
    batch = suggest_design(
        Space([Parameter("rank", kind="ordinal", values=[0, 1, 2])]),
        2,
        method="auto",
        seed=4,
        return_diagnostics=True,
    )

    assert batch.diagnostics.effective_method == "lhs_maximin"
    assert batch.diagnostics.selection_rule == "auto:no-nominal-categorical->lhs_maximin"


def test_explicit_d_optimal_rejects_categorical_without_fallback(mixed_space):
    with pytest.raises(EngineError) as caught:
        suggest_design(mixed_space, 6, method="d_optimal", seed=4)

    assert caught.value.code is ErrorCode.CONFIG_INVALID
    assert "sobol" in caught.value.details["compatible_methods"]


@pytest.mark.parametrize(
    ("parameter", "methods"),
    [
        (Parameter("x", bounds=(0.0, 1.0)), tuple()),
        (Parameter("x", kind="integer", bounds=(0, 2)), tuple()),
        (Parameter("x", kind="discrete", values=[0.0, 1.0, 2.0]), tuple()),
        (Parameter("x", kind="ordinal", values=[0, 1, 2]), ("d_optimal",)),
        (Parameter("x", kind="categorical", values=["a", "b"]), ("lhs_maximin", "lhs_random", "d_optimal")),
    ],
)
def test_method_kind_matrix_accepts_only_declared_capabilities(parameter, methods):
    space = Space([parameter])
    all_methods = (
        "lhs_maximin", "lhs_random", "sobol", "halton", "d_optimal", "random_uniform"
    )
    for method in all_methods:
        if method in methods:
            with pytest.raises(EngineError) as caught:
                suggest_design(space, 1, method=method, seed=2)
            assert caught.value.code is ErrorCode.CONFIG_INVALID
        else:
            design = suggest_design(space, 1, method=method, seed=2)
            assert len(design) == 1


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"n": 0}, "positive integer"),
        ({"n": 1, "seed": -1}, "non-negative integer"),
        ({"n": 1, "method": "unknown"}, "incompatible"),
        ({"n": 1, "existing": "not-a-frame"}, "pandas DataFrame"),
    ],
)
def test_invalid_design_inputs_are_structured(kwargs, fragment):
    with pytest.raises(EngineError, match=fragment):
        suggest_design(Space([Parameter("x", bounds=(0.0, 1.0))]), **kwargs)


def test_pending_rows_are_canonicalized_and_avoided():
    space = Space([Parameter("x", kind="integer", bounds=(0, 2))])

    design = suggest_design(
        space, 2, method="sobol", seed=6, pending=pd.DataFrame({"x": [0.0]})
    )

    assert 0 not in design["x"].tolist()


def test_diagnostics_include_the_complete_stable_contract(mixed_space):
    batch = suggest_design(mixed_space, 2, method="sobol", seed=11, return_diagnostics=True)
    diagnostics = batch.diagnostics

    assert diagnostics.requested_method == "sobol"
    assert diagnostics.effective_method == "sobol"
    assert diagnostics.selection_rule == "explicit"
    assert diagnostics.seed == 11
    assert len(diagnostics.constraint_digest) == 64
    assert diagnostics.knowledge_sources == ()
    assert diagnostics.factor_encodings["binder"] == "categorical:index"
    assert diagnostics.level_counts["binder"]
    assert diagnostics.minimum_model_distance is not None
    assert diagnostics.rejection_counts
    assert len(diagnostics.candidate_keys) == 2
