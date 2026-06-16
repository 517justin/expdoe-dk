"""
Constrained + discrete DoE: feasibility and grid correctness across methods.
"""
import warnings

import numpy as np
import pytest

from expdoe_dk import LinearConstraint, Parameter, Space, suggest_design


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
