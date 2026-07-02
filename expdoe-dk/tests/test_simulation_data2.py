from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_DIR = REPO_ROOT / "experiments" / "simulation_data2"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    old_path = list(sys.path)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
    return module


@pytest.fixture(scope="module")
def oracles():
    return load_module("simulation_data2_oracles", SIM_DIR / "_oracles.py")


def test_make_problem_supports_only_4d_and_6d(oracles):
    p4 = oracles.make_problem(4)
    p6 = oracles.make_problem(6)

    assert p4.dim == 4
    assert p4.n_doe == 6
    assert p4.n_iter == 15
    assert p4.space.param_names == ["T", "time", "pH", "catalyst"]

    assert p6.dim == 6
    assert p6.n_doe == 10
    assert p6.n_iter == 20
    assert p6.space.param_names == [
        "T", "time", "pH", "catalyst", "solvent_A", "additive"
    ]

    with pytest.raises(ValueError, match="Unsupported dim"):
        oracles.make_problem(2)


def test_discrete_levels_stay_within_bounds(oracles):
    for dim in (4, 6):
        problem = oracles.make_problem(dim)
        for param in problem.space.params:
            assert param.kind == "discrete"
            lo, hi = param.bounds
            assert param.levels.min() >= lo
            assert param.levels.max() <= hi


def test_6d_constraints_classify_handpicked_rows(oracles):
    problem = oracles.make_problem(6)
    feasible = pd.DataFrame([{
        "T": 90.0,
        "time": 95.0,
        "pH": 7.0,
        "catalyst": 2.5,
        "solvent_A": 55.0,
        "additive": 5.0,
    }])
    too_much_total = feasible.assign(solvent_A=80.0, additive=10.0)
    too_little_solvent = feasible.assign(solvent_A=30.0, additive=5.0)
    too_much_loading = feasible.assign(catalyst=3.0, additive=10.0)

    assert bool(problem.space.feasibility_mask(feasible).all().item())
    assert not bool(problem.space.feasibility_mask(too_much_total).all().item())
    assert not bool(problem.space.feasibility_mask(too_little_solvent).all().item())
    assert not bool(problem.space.feasibility_mask(too_much_loading).all().item())


def test_noiseless_oracles_are_deterministic_and_noisy_oracles_have_expected_length(oracles):
    p4 = oracles.make_problem(4)
    df = pd.DataFrame([
        {"T": 90.0, "time": 90.0, "pH": 7.0, "catalyst": 2.5},
        {"T": 100.0, "time": 120.0, "pH": 8.0, "catalyst": 3.0},
    ])

    y1 = p4.noiseless(df)
    y2 = p4.noiseless(df)
    assert np.allclose(y1, y2)

    noisy = p4.oracle(df, rng=np.random.default_rng(0))
    assert noisy.shape == (2,)
    assert not np.allclose(noisy, y1)


def test_true_opt_and_hint_are_consistent(oracles):
    for dim in (4, 6):
        problem = oracles.make_problem(dim)
        hint = pd.DataFrame([problem.true_x_hint])
        assert bool(problem.space.feasibility_mask(hint).all().item())
        clean = float(problem.noiseless(hint)[0])
        assert clean <= problem.true_opt + 1e-9
        assert clean >= 0.95 * problem.true_opt


@pytest.fixture(scope="module")
def exp02():
    return load_module("simulation_data2_exp02", SIM_DIR / "02_knowledge_comparison.py")


def test_auto_select_doe_method_reads_experiment_01_summary(tmp_path, exp02):
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    summary = pd.DataFrame({
        "method": ["sobol", "lhs_random", "halton"],
        "gap_final": [0.3, 0.1, 0.2],
    }).set_index("method")
    summary.to_csv(out_dir / "experiment_01_doe_methods_6d_summary.csv")

    selected = exp02.resolve_doe_method("auto", dim=6, out_dir=out_dir)

    assert selected == "lhs_random"


def test_auto_select_doe_method_falls_back_to_sobol(tmp_path, exp02):
    selected = exp02.resolve_doe_method("auto", dim=4, out_dir=tmp_path)

    assert selected == "sobol"


def test_manual_doe_method_override_is_returned_directly(tmp_path, exp02):
    selected = exp02.resolve_doe_method("halton", dim=4, out_dir=tmp_path)

    assert selected == "halton"
