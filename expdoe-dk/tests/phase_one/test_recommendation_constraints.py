"""Hard linear constraints must shape continuous recommendation, not just reject it."""

import numpy as np
import pandas as pd
import pytest
import subprocess
import sys
import textwrap

from expdoe_dk import (
    EngineError, ErrorCode, LinearConstraint as LegacyLinearConstraint,
    ObservationBatch, Parameter, PendingBatch, Space, candidate_keys, recommend,
)
from expdoe_dk.domain import LinearConstraint


def _observations(**factors):
    return ObservationBatch(pd.DataFrame(factors), pd.DataFrame({"y": [0., 1., 0.5]}))


def _assert_unseen_batch(space, result, observations, q, pending=None):
    keys = candidate_keys(result.frame, space)
    assert len(result.frame) == len(keys) == q
    assert not keys & candidate_keys(observations.X, space)
    if pending is not None:
        assert not keys & candidate_keys(pending.X, space)
    assert space.feasibility_mask(result.frame).all()


@pytest.mark.parametrize("equality", [
    LinearConstraint("sum", {"a": 1., "b": 1.}, "==", 1.),
    LegacyLinearConstraint({"a": 1., "b": 1.}, lower=1., upper=1.),
], ids=["domain", "legacy"])
def test_continuous_equality_returns_unique_unseen_batch(equality):
    space = Space([Parameter("a", bounds=(0., 1.)), Parameter("b", bounds=(0., 1.))],
                  objectives="y", constraints=[equality])
    observations = _observations(a=[0., 0.25, 1.], b=[1., 0.75, 0.])
    pending = PendingBatch(("mid",), pd.DataFrame({"a": [0.5], "b": [0.5]}))

    result = recommend(space, observations, pending=pending, q=4, seed=17)

    _assert_unseen_batch(space, result, observations, q=4, pending=pending)
    np.testing.assert_allclose(result.frame.a + result.frame.b, 1., atol=1e-9, rtol=0)


def test_mixed_search_substitutes_physical_levels_and_skips_infeasible_subspaces():
    space = Space([
        Parameter("a", bounds=(2., 4.)),
        Parameter("n", kind="integer", bounds=(1, 3), transform="log"),
        Parameter("b", bounds=(-4., -2.)),
    ], objectives="y", constraints=[LinearConstraint("balance", {"a": 2., "n": 3., "b": -1.}, "==", 14.)])
    # n=1 and n=2 admit a line segment; n=3 is infeasible.
    observations = _observations(a=[3.75, 4., 2.5], n=[1, 1, 2], b=[-3.5, -3., -3.])

    result = recommend(space, observations, q=3, seed=17)

    _assert_unseen_batch(space, result, observations, q=3)
    assert set(result.frame.n) <= {1, 2}
    np.testing.assert_allclose(2 * result.frame.a + 3 * result.frame.n - result.frame.b,
                               14., atol=1e-9, rtol=0)


def test_equality_can_fix_all_continuous_coordinates_per_subspace():
    space = Space([Parameter("n", kind="integer", bounds=(1, 5)),
                   Parameter("a", bounds=(0., 5.))], objectives="y",
                  constraints=[LinearConstraint("sum", {"n": 1., "a": 1.}, "==", 5.)])
    observations = _observations(n=[1, 2, 3], a=[4., 3., 2.])

    result = recommend(space, observations, q=2, seed=17)

    _assert_unseen_batch(space, result, observations, q=2)
    assert set(result.frame.itertuples(index=False, name=None)) == {(4, 1.), (5, 0.)}


def test_seeded_search_respects_affine_bounds_along_the_equality():
    space = Space([Parameter("a", bounds=(0., 1.)), Parameter("b", bounds=(0., 1.))],
                  objectives="y", constraints=[
        LinearConstraint("sum", {"a": 1., "b": 1.}, "==", 1.),
        LegacyLinearConstraint({"a": 1.}, lower=0.1234, upper=0.1235),
    ])
    observations = _observations(a=[0., 0.25, 1.], b=[1., 0.75, 0.])

    result = recommend(space, observations, q=3, seed=17)

    _assert_unseen_batch(space, result, observations, q=3)
    assert result.frame.a.between(0.1234 - 1e-9, 0.1235 + 1e-9).all()


def test_infeasible_continuous_equalities_report_no_candidate():
    space = Space([Parameter("a", bounds=(0., 1.)), Parameter("b", bounds=(0., 1.))],
                  objectives="y", constraints=[LinearConstraint("sum", {"a": 1., "b": 1.}, "==", 3.)])
    with pytest.raises(EngineError) as error:
        recommend(space, _observations(a=[0., 0.25, 1.], b=[1., 0.75, 0.]), seed=17)
    assert error.value.code == ErrorCode.NO_FEASIBLE_CANDIDATE


def test_redundant_equalities_do_not_remove_free_dimensions():
    space = Space([Parameter(name, bounds=(0., 1.)) for name in ("a", "b", "c")],
                  objectives="y", constraints=[
        LinearConstraint("sum", {"a": 1., "b": 1.}, "==", 1.),
        LinearConstraint("scaled_sum", {"a": 2., "b": 2.}, "==", 2.),
        LinearConstraint("fixed", {"c": 1.}, "==", 0.5),
    ])
    observations = _observations(a=[0., 0.25, 1.], b=[1., 0.75, 0.], c=[0.5] * 3)

    result = recommend(space, observations, q=3, seed=17)

    _assert_unseen_batch(space, result, observations, q=3)
    np.testing.assert_allclose(result.frame.a + result.frame.b, 1., atol=1e-9, rtol=0)
    np.testing.assert_allclose(result.frame.c, 0.5, atol=1e-9, rtol=0)


@pytest.mark.parametrize("face", ["bounds", "inequalities"])
def test_boundary_affine_hull_keeps_unseen_free_coordinate(face):
    constraints = [LinearConstraint("sum", {"a": 1., "b": 1.}, "==", 0.)]
    lower = 0. if face == "bounds" else -1.
    if face == "inequalities":
        constraints.extend([
            LinearConstraint("nonnegative_a", {"a": 1.}, ">=", 0.),
            LinearConstraint("nonnegative_b", {"b": 1.}, ">=", 0.),
        ])
    space = Space([Parameter("a", bounds=(lower, 1.)), Parameter("b", bounds=(lower, 1.)),
                   Parameter("c", bounds=(0., 1.))], objectives="y", constraints=constraints)
    observations = _observations(a=[0.] * 3, b=[0.] * 3, c=[0., 0.5, 1.])

    result = recommend(space, observations, q=3, seed=17)

    _assert_unseen_batch(space, result, observations, q=3)
    assert not set(result.frame.c) & {0., 0.5, 1.}
    np.testing.assert_allclose(result.frame[["a", "b"]], 0., atol=1e-9, rtol=0)


@pytest.mark.parametrize("coefficient,delta", [(1., 1e-10), (1000., 1e-10)])
def test_redundant_equalities_keep_physical_tolerance_after_scaling(coefficient, delta):
    space = Space([Parameter("x", bounds=(0., 1.)), Parameter("free", bounds=(0., 1.))],
                  objectives="y", constraints=[
        LinearConstraint("first", {"x": 1.}, "==", 0.5),
        LinearConstraint("second", {"x": coefficient}, "==", coefficient * (0.5 + delta)),
    ])
    observations = _observations(x=[0.5 + delta] * 3, free=[0., 0.5, 1.])
    assert space.feasibility_mask(observations.X).all()

    result = recommend(space, observations, q=3, seed=17)

    _assert_unseen_batch(space, result, observations, q=3)
    assert (abs(result.frame.x - 0.5) <= 1e-9).all()
    assert (abs(coefficient * result.frame.x - coefficient * (0.5 + delta)) <= 1e-9).all()


@pytest.mark.parametrize("boundary,target", [("bounds", -1e-10), ("inequality", 0.5 + 1e-10)])
def test_independent_equality_tolerance_respects_hard_boundaries(boundary, target):
    constraints = [LinearConstraint("near_boundary", {"x": 1.}, "==", target)]
    physical_x = 0. if boundary == "bounds" else 0.5
    if boundary == "inequality":
        constraints.append(LinearConstraint("upper", {"x": 1.}, "<=", 0.5))
    space = Space([Parameter("x", bounds=(0., 1.)), Parameter("free", bounds=(0., 1.))],
                  objectives="y", constraints=constraints)
    observations = _observations(x=[physical_x] * 3, free=[0., 0.5, 1.])
    assert space.feasibility_mask(observations.X).all()

    result = recommend(space, observations, q=3, seed=17)

    _assert_unseen_batch(space, result, observations, q=3)
    assert result.frame.x.between(0., 0.5).all()
    assert (abs(result.frame.x - target) <= 1e-9).all()


def test_equalities_outside_physical_tolerance_remain_infeasible():
    space = Space([Parameter("x", bounds=(0., 1.)), Parameter("free", bounds=(0., 1.))],
                  objectives="y", constraints=[
        LinearConstraint("first", {"x": 1.}, "==", 0.5),
        LinearConstraint("second", {"x": 1.}, "==", 0.5 + 1e-7),
    ])
    with pytest.raises(EngineError) as error:
        recommend(space, _observations(x=[0.5] * 3, free=[0., 0.5, 1.]), seed=17)
    assert error.value.code == ErrorCode.NO_FEASIBLE_CANDIDATE


def test_linear_feasibility_lp_failure_is_not_reported_as_infeasibility(monkeypatch):
    from scipy.optimize import OptimizeResult

    def fail(*args, **kwargs):
        return OptimizeResult(status=4, success=False, message="Injected LP numerical failure")

    monkeypatch.setattr("expdoe_dk.bo.linear_search.linprog", fail)
    space = Space([Parameter("a", bounds=(0., 1.)), Parameter("b", bounds=(0., 1.))],
                  objectives="y", constraints=[LinearConstraint("sum", {"a": 1., "b": 1.}, "==", 1.)])
    with pytest.raises(EngineError) as error:
        recommend(space, _observations(a=[0., .5, 1.], b=[1., .5, 0.]), seed=17)
    assert error.value.code == ErrorCode.ACQUISITION_INCOMPATIBLE
    assert "LP numerical failure" in str(error.value)


def test_missing_new_botorch_error_does_not_break_import_or_ordinary_recommendation():
    # Preload dependencies so removing the new symbol emulates the old public API,
    # not an internally corrupted installation of the current BoTorch version.
    script = textwrap.dedent("""
        import sys
        import expdoe_dk
        for name in list(sys.modules):
            if name == "expdoe_dk" or name.startswith("expdoe_dk."):
                del sys.modules[name]
        import botorch.exceptions.errors as errors
        if hasattr(errors, "InfeasibilityError"):
            del errors.InfeasibilityError
        import pandas as pd
        from expdoe_dk import ObservationBatch, Parameter, Space, recommend
        space = Space([Parameter("x", bounds=(0., 1.))], objectives="y")
        observations = ObservationBatch(pd.DataFrame({"x": [0., .5, 1.]}),
                                        pd.DataFrame({"y": [0., 1., .5]}))
        result = recommend(space, observations, seed=17)
        assert len(result.frame) == 1
        assert result.frame.x.item() not in {0., .5, 1.}
        catalog = pd.DataFrame({"x": [.25, .75]})
        result = recommend(space, observations, candidates=catalog,
                           candidate_ids=["a", "b"], q=2, seed=17)
        assert set(result.frame.x) == {.25, .75}
        assert set(result.candidate_ids) == {"a", "b"}
    """)
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr


def test_old_sampler_api_has_explicit_equality_capability_error(monkeypatch):
    import expdoe_dk.bo.linear_search as search
    from botorch.utils import sampling

    class OldSampler:
        def __init__(self, inequality_constraints=None, equality_constraints=None,
                     bounds=None, interior_point=None, n_burnin=200):
            pytest.fail("Unsupported sampler must be detected before construction")

    monkeypatch.setattr(sampling, "HitAndRunPolytopeSampler", OldSampler)
    if hasattr(search, "HitAndRunPolytopeSampler"):
        monkeypatch.setattr(search, "HitAndRunPolytopeSampler", OldSampler)
    space = Space([Parameter("a", bounds=(0., 1.)), Parameter("b", bounds=(0., 1.))],
                  objectives="y", constraints=[LinearConstraint("sum", {"a": 1., "b": 1.}, "==", 1.)])
    observations = _observations(a=[0., .5, 1.], b=[1., .5, 0.])
    with pytest.raises(EngineError) as error:
        recommend(space, observations, seed=17)
    assert error.value.code == ErrorCode.ACQUISITION_INCOMPATIBLE
    assert "catalog" in str(error.value).lower()
    assert "botorch" in str(error.value).lower()
    result = recommend(space, observations, candidates=pd.DataFrame({"a": [.25], "b": [.75]}),
                       candidate_ids=["feasible"], seed=17)
    assert result.candidate_ids == ("feasible",)


def test_constrained_optimizer_warning_does_not_return_seed_pool(monkeypatch):
    from botorch.generation import gen

    space = Space([Parameter("a", bounds=(0., 1.)), Parameter("b", bounds=(0., 1.))],
                  objectives="y", constraints=[LinearConstraint("sum", {"a": 1., "b": 1.}, "==", 1.)])
    observations = _observations(a=[0., 0.25, 1.], b=[1., 0.75, 0.])
    minimize = gen.minimize_with_timeout

    def fail(*args, **kwargs):
        result = minimize(*args, **kwargs)
        result.update(success=False, status=6, message="Injected singular matrix in LSQ subproblem")
        return result

    monkeypatch.setattr(gen, "minimize_with_timeout", fail)
    with pytest.raises(EngineError) as error:
        recommend(space, observations, seed=17)
    assert error.value.code == ErrorCode.ACQUISITION_INCOMPATIBLE
    assert "singular matrix" in str(error.value)


@pytest.mark.parametrize("log_options", [{"transform": "log"}, {"log_scale": True}])
def test_nonaffine_continuous_equality_recommends_catalog_alternative(log_options):
    space = Space([Parameter("a", bounds=(1., 4.), **log_options), Parameter("b", bounds=(0., 3.))],
                  objectives="y", constraints=[LinearConstraint("sum", {"a": 1., "b": 1.}, "==", 4.)])
    observations = _observations(a=[1., 2., 4.], b=[3., 2., 0.])
    with pytest.raises(EngineError) as error:
        recommend(space, observations, seed=17)
    assert error.value.code in {ErrorCode.CONFIG_INVALID, ErrorCode.ACQUISITION_INCOMPATIBLE}
    assert "catalog" in str(error.value).lower()


def test_catalog_keeps_original_nonaffine_equality_rows_without_optimization(monkeypatch):
    space = Space([Parameter("a", bounds=(1., 4.), transform="log"), Parameter("b", bounds=(0., 3.))],
                  objectives="y", constraints=[LinearConstraint("sum", {"a": 1., "b": 1.}, "==", 4.)])
    observations = _observations(a=[1., 2., 4.], b=[3., 2., 0.])
    catalog = pd.DataFrame({"a": [1.3, 3.1, 3.7], "b": [2.7, 0.9, 0.4]})
    ids = ["first", "second", "infeasible"]

    def unexpected_optimizer(*args, **kwargs):
        pytest.fail("Catalog rows must not enter continuous optimization")

    monkeypatch.setattr("expdoe_dk.bo.recommendation.optimize_acqf", unexpected_optimizer)
    result = recommend(space, observations, candidates=catalog, candidate_ids=ids, q=2, seed=17)

    assert set(result.candidate_ids) == {"first", "second"}
    pd.testing.assert_frame_equal(result.frame, catalog.iloc[[ids.index(id_) for id_ in result.candidate_ids]]
                                  .reset_index(drop=True), check_exact=True)
