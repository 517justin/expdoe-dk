import pandas as pd
import pytest
import torch

from expdoe_dk import Objective, Parameter, Space, suggest_design
from expdoe_dk.domain import (
    CategoricalCombinationConstraint,
    ExpressionConstraint,
    LinearConstraint as DomainLinearConstraint,
    OutcomeConstraint,
)
from expdoe_dk.errors import EngineError, ErrorCode


def test_mixed_space_round_trip_preserves_parameter_order_and_physical_values():
    """Catches mixed transforms dropping strings or following caller column order."""
    space = Space(
        params=[
            Parameter("temperature", bounds=(300.0, 400.0)),
            Parameter("grade", kind="ordinal", values=["low", "medium", "high"]),
            Parameter("binder", kind="categorical", values=["A", "B"]),
        ],
        objectives=[Objective("strength", "maximize")],
    )
    frame = pd.DataFrame(
        {"binder": ["B"], "temperature": [325.0], "grade": ["medium"]}
    )

    model = space.physical_to_model(frame)
    restored = space.model_to_physical(model)

    assert model.dtype == torch.float64
    assert model.tolist() == [[0.25, 1.0, 1.0]]
    assert restored.to_dict("records") == [
        {"temperature": 325.0, "grade": "medium", "binder": "B"}
    ]


def test_mixed_sobol_design_is_deterministic_and_decodes_declared_levels():
    """Catches low-discrepancy designs leaking model codes into physical output."""
    space = Space(
        params=[
            Parameter("temperature", bounds=(300.0, 400.0)),
            Parameter("grade", kind="ordinal", values=["low", "medium", "high"]),
            Parameter("binder", kind="categorical", values=["A", "B"]),
        ]
    )

    first = suggest_design(space, n=6, method="sobol", seed=4)
    second = suggest_design(space, n=6, method="sobol", seed=4)

    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == ["temperature", "grade", "binder"]
    assert set(first["grade"]) <= {"low", "medium", "high"}
    assert set(first["binder"]) <= {"A", "B"}


def test_legacy_space_constructor_still_exposes_objective_lists():
    """Catches explicit Objective storage breaking the v0.4 list properties."""
    space = Space(
        [Parameter("x", bounds=(0.0, 1.0))],
        objectives="yield",
        maximize=True,
    )

    assert space.objectives == ["yield"]
    assert space.maximize == [True]
    assert space.objective_specs == (Objective("yield", "maximize"),)


def test_feasibility_mask_applies_only_hard_parameter_constraints():
    """Catches soft preferences being promoted into hard feasibility filters."""
    space = Space(
        [Parameter("x", bounds=(0.0, 1.0))],
        constraints=[
            ExpressionConstraint("hard", "x >= 0.25"),
            ExpressionConstraint(
                "preferred", "x <= 0.5", hard=False, penalty="hinge", weight=2.0
            ),
        ],
    )
    frame = pd.DataFrame({"x": [0.1, 0.4, 0.9]})

    assert space.feasibility_mask(frame).tolist() == [False, True, True]


def test_with_constraints_returns_a_validated_copy_without_mutating_source():
    """Catches fluent constraint composition mutating or dropping space metadata."""
    source = Space(
        [Parameter("x", bounds=(0.0, 1.0))],
        objectives=[Objective("yield", "minimize")],
    )
    limit = DomainLinearConstraint("limit", {"x": 1.0}, "<=", 0.5)

    constrained = source.with_constraints(limit)

    assert source.constraints == []
    assert constrained.constraints == [limit]
    assert constrained.objective_specs == source.objective_specs
    assert constrained.feasibility_mask(pd.DataFrame({"x": [0.25, 0.75]})).tolist() == [
        True,
        False,
    ]


def test_space_serialization_round_trips_mixed_domain_types_strictly():
    """Catches schema envelopes losing tagged factors or declarative constraints."""
    space = Space(
        params=[
            Parameter("temperature", bounds=(300.0, 400.0), unit="K"),
            Parameter("binder", kind="categorical", values=["A", "B"]),
        ],
        constraints=[
            DomainLinearConstraint(
                "temperature-limit", {"temperature": 1.0}, "<=", 375.0
            )
        ],
        objectives=[Objective("strength", "maximize", unit="MPa")],
        outcome_constraints=[
            OutcomeConstraint("minimum-strength", "strength", ">=", 20.0)
        ],
    )

    payload = space.to_dict()
    restored = Space.from_dict(payload)

    assert payload["schema_version"] == "1.0"
    assert payload["engine_version"] == "0.5.0"
    assert restored.to_dict() == payload
    assert restored.model_to_physical(
        restored.physical_to_model(
            pd.DataFrame({"temperature": [350.0], "binder": ["B"]})
        )
    ).to_dict("records") == [{"temperature": 350.0, "binder": "B"}]


@pytest.mark.parametrize(
    ("field", "value"),
    [("schema_version", "99.0"), ("engine_version", "9.9.9")],
)
def test_space_restoration_rejects_unknown_schema_or_engine_version(field, value):
    """Catches incompatible serialized spaces being accepted as current state."""
    payload = Space([Parameter("x", bounds=(0.0, 1.0))]).to_dict()
    payload[field] = value

    with pytest.raises(EngineError) as caught:
        Space.from_dict(payload)

    assert caught.value.code is ErrorCode.CHECKPOINT_INCOMPATIBLE


def test_space_restores_unversioned_v04_payload():
    """Catches the strict v0.5 loader rejecting readable v0.4 checkpoints."""
    restored = Space.from_dict(
        {
            "params": [
                {
                    "name": "x",
                    "bounds": [1.0, 100.0],
                    "unit": "",
                    "kind": "continuous",
                    "step": None,
                    "log_scale": True,
                }
            ],
            "constraints": [
                {
                    "coeffs": {"x": 1.0},
                    "lower": 2.0,
                    "upper": 90.0,
                    "name": "window",
                }
            ],
            "objectives": ["yield"],
            "maximize": [False],
        }
    )

    assert restored.objectives == ["yield"]
    assert restored.maximize == [False]
    assert restored.params[0].log_scale is True
    assert restored.constraints[0].satisfied({"x": 50.0})
    assert not restored.constraints[0].satisfied({"x": 100.0})


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Space(
            [
                Parameter("x", bounds=(0.0, 1.0)),
                Parameter("x", bounds=(1.0, 2.0)),
            ]
        ),
        lambda: Space(
            [Parameter("x", bounds=(0.0, 1.0))],
            objectives=[Objective("yield", "maximize"), Objective("yield", "minimize")],
        ),
        lambda: Space(
            [Parameter("x", bounds=(0.0, 1.0))],
            constraints=[
                DomainLinearConstraint("limit", {"x": 1.0}, "<=", 1.0),
                ExpressionConstraint("limit", "x >= 0"),
            ],
        ),
    ],
)
def test_space_rejects_duplicate_domain_names(factory):
    """Catches ambiguous factor, objective, or constraint lookup namespaces."""
    with pytest.raises((ValueError, EngineError), match="unique|duplicate"):
        factory()


def test_space_validates_constraint_factor_and_outcome_references():
    """Catches declarative cross-references escaping space construction."""
    parameter = Parameter("binder", kind="categorical", values=["A", "B"])

    with pytest.raises(EngineError) as factor_error:
        Space(
            [parameter],
            constraints=[
                CategoricalCombinationConstraint(
                    "unknown-factor", forbidden=({"typo": "A"},)
                )
            ],
        )
    with pytest.raises(EngineError) as objective_error:
        Space(
            [parameter],
            objectives=[Objective("strength", "maximize")],
            outcome_constraints=[
                OutcomeConstraint("unknown-objective", "yield", ">=", 0.0)
            ],
        )

    assert factor_error.value.code is ErrorCode.CONFIG_INVALID
    assert objective_error.value.code is ErrorCode.CONFIG_INVALID


def test_numerical_aliases_round_trip_and_mixed_tensor_output_is_explicit():
    """Catches legacy tensor aliases silently coercing string-valued factors."""
    numerical = Space([Parameter("x", bounds=(10.0, 20.0))])
    physical = torch.tensor([[12.5]], dtype=torch.float64)

    assert torch.equal(
        numerical.unit_to_physical(numerical.physical_to_unit(physical)), physical
    )

    mixed = Space(
        [Parameter("binder", kind="categorical", values=["A", "B"])]
    )
    with pytest.raises(
        TypeError,
        match="unit_to_physical tensor output is unavailable.*model_to_physical",
    ):
        mixed.unit_to_physical(torch.tensor([[0.0]], dtype=torch.float64))


def test_numerical_aliases_preserve_legacy_one_row_vector_shape():
    """Catches DataFrame delegation promoting a legacy d-vector into a matrix."""
    space = Space(
        [
            Parameter("temperature", bounds=(300.0, 400.0)),
            Parameter("time", bounds=(10.0, 30.0)),
        ]
    )
    physical = torch.tensor([325.0, 20.0], dtype=torch.float64)

    model = space.physical_to_unit(physical)

    assert model.shape == (2,)
    torch.testing.assert_close(
        model, torch.tensor([0.25, 0.5], dtype=torch.float64)
    )
    assert torch.equal(space.unit_to_physical(model), physical)
