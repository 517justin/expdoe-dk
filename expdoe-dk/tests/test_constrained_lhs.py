"""
Constrained + discrete DoE: feasibility and grid correctness across methods.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from expdoe_dk import LinearConstraint, Parameter, Space, suggest_design
from expdoe_dk.domain import CategoricalCombinationConstraint
from expdoe_dk.doe.constrained import _pool_greedy_maximin


def _space_chem():
    return Space(
        params=[
            Parameter("T", bounds=(60.0, 120.0), unit="°C"),
            Parameter("time", bounds=(10.0, 180.0), unit="min"),
            Parameter("conc_A", bounds=(1.0, 10.0), unit="mL",
                      kind="discrete", step=1.0),
            Parameter("conc_B", bounds=(1.0, 10.0), unit="mL",
                      kind="discrete", step=1.0),
        ],
        constraints=[
            LinearConstraint(coeffs={"conc_A": 1.0, "conc_B": -1.0}, lower=1.0),
        ],
    )


@pytest.mark.parametrize(
    "method", ["lhs_maximin", "lhs_random", "sobol", "halton", "random_uniform"]
)
def test_methods_produce_feasible_discrete_designs(method):
    space = _space_chem()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = suggest_design(space, n=8, method=method, seed=42,
                            n_iterations=600)
    # 1. feasibility
    assert bool(space.feasibility_mask(df).all().item())
    # 2. discrete columns on grid
    for col in ("conc_A", "conc_B"):
        assert np.allclose(df[col].values, df[col].round().values)
        assert df[col].between(1.0, 10.0).all()
    # 3. column count + row count
    assert list(df.columns) == space.param_names
    assert len(df) == 8
    # 4. constraint check on every row
    assert (df["conc_A"] - df["conc_B"] >= 1.0 - 1e-9).all()


def test_maximin_beats_random_in_spread():
    space = Space(
        params=[
            Parameter("x", bounds=(0.0, 1.0)),
            Parameter("y", bounds=(0.0, 1.0)),
            Parameter("z", bounds=(0.0, 1.0)),
        ]
    )
    from scipy.spatial.distance import pdist

    df_max = suggest_design(space, n=12, method="lhs_maximin", seed=42,
                            n_iterations=1500)
    df_rand = suggest_design(space, n=12, method="random_uniform", seed=42)
    d_max = float(pdist(df_max.to_numpy()).min())
    d_rand = float(pdist(df_rand.to_numpy()).min())
    # maximin should produce a strictly larger minimum pairwise distance.
    assert d_max > d_rand


def test_discrete_step_that_does_not_divide_range_never_snaps_past_bounds():
    p = Parameter("dose", bounds=(0.0, 1.0), kind="discrete", step=0.6)

    assert p.levels.tolist() == [0.0, 0.6]
    snapped = p.snap(np.array([0.0, 0.59, 0.9, 1.0]))

    assert np.all(snapped <= 1.0)
    assert snapped.tolist() == [0.0, 0.6, 0.6, 0.6]


def test_sobol_mixed_constraint_filtering_stays_in_physical_frame():
    """Catches categorical constraint checks receiving model-space numeric codes."""
    space = Space(
        params=[
            Parameter("temperature", bounds=(300.0, 400.0)),
            Parameter("binder", kind="categorical", values=["A", "B"]),
        ],
        constraints=[
            CategoricalCombinationConstraint(
                "forbidden-binder", forbidden=({"binder": "B"},)
            )
        ],
    )

    design = suggest_design(space, n=4, method="sobol", seed=11)

    assert design["binder"].tolist() == ["A"] * 4
    assert bool(space.feasibility_mask(design).all())
    pd.testing.assert_frame_equal(
        design,
        space.model_to_physical(space.physical_to_model(design)),
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "method",
    [
        "lhs_maximin",
        "lhs_random",
        "sobol",
        "halton",
        "d_optimal",
        "random_uniform",
    ],
)
def test_all_current_methods_return_one_deterministic_row(method):
    """Catches maximin distance reduction assuming at least two rows."""
    space = Space([Parameter("x", bounds=(0.0, 1.0))])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = suggest_design(space, n=1, method=method, seed=17, n_iterations=20)
        second = suggest_design(space, n=1, method=method, seed=17, n_iterations=20)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 1
    assert first["x"].between(0.0, 1.0).all()
    assert bool(space.feasibility_mask(first).all())


@pytest.mark.parametrize(
    "method",
    [
        "lhs_maximin",
        "lhs_random",
        "sobol",
        "halton",
        "d_optimal",
        "random_uniform",
    ],
)
def test_all_current_methods_preserve_heterogeneous_ordinal_scalars(method):
    """Catches accepted record reconstruction converting None into NaN."""
    space = Space([Parameter("grade", kind="ordinal", values=[1, None])])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        design = suggest_design(
            space, n=8, method=method, seed=19, n_iterations=20
        )

    values = design["grade"].tolist()
    assert design["grade"].dtype == object
    assert any(value is None for value in values)
    assert any(type(value) is int and value == 1 for value in values)
    assert all(value is None or (type(value) is int and value == 1) for value in values)
    assert bool(space.feasibility_mask(design).all())
    space.physical_to_model(design)


def test_candidate_pool_preserves_heterogeneous_ordinal_scalars():
    """Catches the maximin fallback rebuilding mixed records through inference."""
    space = Space([Parameter("grade", kind="ordinal", values=[1, None])])

    design = _pool_greedy_maximin(
        space, n=8, pool_factor=4, max_resample=2, seed=23
    )

    assert design["grade"].dtype == object
    assert bool(space.feasibility_mask(design).all())
    space.physical_to_model(design)
