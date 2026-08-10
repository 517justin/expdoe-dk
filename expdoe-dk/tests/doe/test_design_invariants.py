import numpy as np
import pandas as pd
import pytest

from expdoe_dk import Parameter, Space, suggest_design
from expdoe_dk.domain import CategoricalCombinationConstraint, ExpressionConstraint
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


def test_existing_numeric_values_are_canonicalized_before_avoidance():
    space = Space([Parameter("x", kind="discrete", values=[0.0, 1.0, 2.0])])

    design = suggest_design(
        space, 2, method="sobol", seed=3, existing=pd.DataFrame({"x": [0]})
    )

    assert design["x"].tolist() == [1.0, 2.0]
    assert not candidate_keys(design, space) & candidate_keys(
        pd.DataFrame({"x": [0]}), space
    )


def test_d_optimal_deduplicates_and_avoids_before_exact_selection():
    space = Space([Parameter("x", kind="integer", bounds=(0, 4))])

    design = suggest_design(
        space, 2, method="d_optimal", seed=1, existing=pd.DataFrame({"x": [0]})
    )

    assert len(design) == 2
    assert 0 not in design["x"].tolist()
    assert design["x"].nunique() == 2


def test_finite_d_optimal_eligibility_rejections_are_counted_once():
    space = Space([Parameter("x", kind="integer", bounds=(0, 4))])

    batch = suggest_design(
        space,
        2,
        method="d_optimal",
        seed=1,
        existing=pd.DataFrame({"x": [0]}),
        return_diagnostics=True,
    )

    assert batch.diagnostics.rejection_counts["avoided"] == 1
    assert batch.diagnostics.rejection_counts["duplicate"] == 0
    assert batch.diagnostics.rejection_counts["constraint"] == 0


def test_d_optimal_matches_the_declared_greedy_log_determinant_criterion():
    space = Space([Parameter("x", kind="integer", bounds=(0, 4))])
    universe = pd.DataFrame({"x": [1, 2, 3, 4]})
    model = space.physical_to_model(universe).cpu().numpy()
    matrix = np.column_stack((np.ones(len(model)), model, model**2))
    information = np.eye(matrix.shape[1]) * 1e-12
    expected = []
    remaining = set(range(len(matrix)))
    for _ in range(2):
        best = max(
            remaining,
            key=lambda index: np.linalg.slogdet(
                information + np.outer(matrix[index], matrix[index])
            )[1],
        )
        expected.append(best)
        information += np.outer(matrix[best], matrix[best])
        remaining.remove(best)

    selected = suggest_design(
        space,
        2,
        method="d_optimal",
        seed=1,
        existing=pd.DataFrame({"x": [0]}),
    )

    assert selected["x"].tolist() == universe.iloc[expected]["x"].tolist()


def test_lhs_random_uses_each_requested_stratum_once():
    space = Space([Parameter("x", bounds=(0.0, 1.0))])

    design = suggest_design(space, 8, method="lhs_random", seed=6)

    assert sorted(np.floor(design["x"].to_numpy() * 8).astype(int).tolist()) == list(
        range(8)
    )


def test_valid_constrained_lhs_keeps_requested_strata():
    space = Space(
        [Parameter("x", bounds=(0.0, 1.0))],
        constraints=[ExpressionConstraint("domain-safe", "x >= 0")],
    )

    design = suggest_design(space, 8, method="lhs_random", seed=6, n_restarts=2)

    assert sorted(np.floor(design["x"].to_numpy() * 8).astype(int).tolist()) == list(
        range(8)
    )


@pytest.mark.parametrize("method", ["lhs_random", "lhs_maximin"])
def test_constrained_lhs_never_silently_substitutes_random_pool(method):
    space = Space(
        [Parameter("x", bounds=(0.0, 1.0))],
        constraints=[ExpressionConstraint("tight", "x <= 0.1")],
    )

    with pytest.raises(EngineError) as caught:
        suggest_design(space, 8, method=method, seed=6, n_restarts=2)

    assert caught.value.code is ErrorCode.SPACE_INFEASIBLE
    assert caught.value.details["method"] == method
    assert caught.value.details["seed"] == 6
    assert caught.value.details["requested"] == 8


def test_lhs_maximin_is_distinct_from_random_lhs_for_same_seed():
    space = Space(
        [Parameter("x", bounds=(0.0, 1.0)), Parameter("y", bounds=(0.0, 1.0))]
    )

    random = suggest_design(space, 6, method="lhs_random", seed=8)
    maximin = suggest_design(space, 6, method="lhs_maximin", seed=8)

    assert not random.equals(maximin)


def test_finite_spaces_keep_method_specific_semantics():
    space = Space(
        [
            Parameter("x", kind="integer", bounds=(0, 3)),
            Parameter("y", kind="integer", bounds=(0, 3)),
        ]
    )
    methods = ("lhs_random", "lhs_maximin", "sobol", "halton", "d_optimal", "random_uniform")
    designs = {
        method: suggest_design(space, 4, method=method, seed=12)
        for method in methods
    }

    for method, design in designs.items():
        pd.testing.assert_frame_equal(
            design, suggest_design(space, 4, method=method, seed=12)
        )
        assert suggest_design(
            space, 4, method=method, seed=12, return_diagnostics=True
        ).diagnostics.effective_method == method
    assert not designs["random_uniform"].equals(
        suggest_design(space, 4, method="random_uniform", seed=13)
    )
    assert len({tuple(map(tuple, design.to_numpy())) for design in designs.values()}) >= 3


def test_balance_is_jointly_feasible_across_ordinal_factors():
    space = Space(
        [
            Parameter("left", kind="ordinal", values=[0, 1]),
            Parameter("right", kind="ordinal", values=[0, 1]),
        ],
        constraints=[
            CategoricalCombinationConstraint(
                "forbid-high-high", forbidden=({"left": 1, "right": 1},)
            )
        ],
    )

    design = suggest_design(space, 2, method="sobol", seed=4)

    assert {tuple(row) for row in design.to_numpy()} == {(0, 1), (1, 0)}


def test_design_batch_is_defensive_and_detached_from_inputs():
    space = Space([Parameter("x", kind="discrete", values=[0.0, 1.0, 2.0])])
    existing = pd.DataFrame({"x": [0.0]})
    batch = suggest_design(
        space, 2, method="sobol", seed=5, existing=existing, return_diagnostics=True
    )

    exposed = batch.frame
    exposed.loc[0, "x"] = 0.0
    existing.loc[0, "x"] = 2.0

    assert 0.0 not in batch.frame["x"].tolist()
    with pytest.raises(TypeError):
        batch.diagnostics.factor_encodings["x"] = "changed"
    with pytest.raises(TypeError):
        batch.diagnostics.level_counts["x"] = {}
    with pytest.raises(TypeError):
        batch.diagnostics.rejection_counts["avoided"] = 99


def test_finite_constraint_rejections_are_reported():
    space = Space(
        [
            Parameter("x", kind="integer", bounds=(0, 1)),
            Parameter("y", kind="integer", bounds=(0, 1)),
        ],
        constraints=[ExpressionConstraint("only-origin", "x + y <= 0")],
    )

    with pytest.raises(EngineError) as caught:
        suggest_design(space, 2, method="sobol", seed=4)

    assert caught.value.code is ErrorCode.SPACE_INFEASIBLE
    assert caught.value.details["rejections"]["constraint"] == 3


def test_historical_existing_rows_need_only_parameter_domain_validity():
    space = Space([Parameter("x", kind="discrete", values=[0.0, 1.0])])
    safe = ExpressionConstraint("safe", "x >= 1")

    design = suggest_design(
        space,
        1,
        method="sobol",
        seed=1,
        parameter_constraints=(safe,),
        existing=pd.DataFrame({"x": [0.0]}),
    )

    assert design["x"].tolist() == [1.0]


def test_soft_parameter_constraints_are_recorded_without_biasing_initial_design():
    space = Space([Parameter("x", bounds=(0.0, 1.0))])
    soft = ExpressionConstraint("preference", "x >= 0.5", hard=False, penalty="hinge", weight=1.0)

    baseline = suggest_design(space, 8, method="sobol", seed=5, return_diagnostics=True)
    softened = suggest_design(
        space,
        8,
        method="sobol",
        seed=5,
        parameter_constraints=(soft,),
        return_diagnostics=True,
    )

    pd.testing.assert_frame_equal(baseline.frame, softened.frame)
    assert baseline.diagnostics.constraint_digest != softened.diagnostics.constraint_digest


def test_requested_row_limit_fails_without_sampling():
    from expdoe_dk.doe.design import MAX_REQUESTED_ROWS

    with pytest.raises(EngineError) as caught:
        suggest_design(Space([Parameter("x", bounds=(0.0, 1.0))]), MAX_REQUESTED_ROWS + 1)

    assert caught.value.code is ErrorCode.CONFIG_INVALID
    assert caught.value.details["limit"] == MAX_REQUESTED_ROWS
