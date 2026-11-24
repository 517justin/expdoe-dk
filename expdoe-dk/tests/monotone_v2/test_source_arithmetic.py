"""Local signed/variance budgets, not a universal absolute tolerance."""
import importlib
import importlib.util

import mpmath
import numpy as np
import pytest

from expdoe_dk.errors import EngineError
from .test_source import frozen_fit, assert_local_accuracy, tensor


def arithmetic():
    name = "expdoe_dk.bo.directional_source_arithmetic"
    assert importlib.util.find_spec(name) is not None, "MP arithmetic is not implemented"
    return importlib.import_module(name)


@pytest.mark.parametrize("value", ["1e-330", "1e-310", "1e310"])
def test_positive_variance_range_fails_before_any_outer_rescaling(value):
    a = arithmetic()
    c = mpmath.mp.clone()
    c.dps = 80
    with pytest.raises(EngineError) as caught:
        a.round_scalar(c, c.mpf(value), c.mpf(0), c.mpf(value), variance=True)
    assert caught.value.details["reason"] == "source_output_range"


@pytest.mark.parametrize("above_maximum", [False, True])
def test_variance_mp_upper_bound_even_when_cast_is_finite(above_maximum):
    a = arithmetic()
    c = mpmath.mp.clone()
    c.dps = 100
    maximum = float.fromhex("0x1.fffffffffffffp+1023")
    value = c.mpf(5.)*c.mpf(3.5953862697246315e307) if above_maximum else c.mpf(maximum)
    assert float(value) == maximum
    assert (value > c.mpf(maximum)) == above_maximum
    # Signed outputs retain section 14's finite-cast, local absolute budget.
    assert a.round_scalar(c, value, c.zero, value) == maximum
    if above_maximum:
        with pytest.raises(EngineError) as caught:
            a.round_scalar(c, value, c.zero, value, variance=True)
        assert caught.value.details["reason"] == "source_output_range"
    else:
        assert a.round_scalar(c, value, c.zero, value, variance=True) == maximum


@pytest.mark.parametrize("above_maximum", [False, True])
def test_conditional_variance_mp_upper_bound_with_certified_zero_row(above_maximum):
    a = arithmetic()
    maximum = float.fromhex("0x1.fffffffffffffp+1023")
    scale, diagonal = (5., 3.5953862697246315e307) if above_maximum else (1., maximum)
    assert scale*diagonal == maximum
    model = a.Conditional([[]], [0.], [0.], [1.],
        {"kind": None, "lengthscale": [], "scales": [scale],
         "tables": [[[1., 0., 0.], [0., diagonal, 0.], [0., 0., 0.]]]}, [[0]])
    if above_maximum:
        with pytest.raises(EngineError) as caught:
            model.moments([[], []], [[1], [2]], [0., 0.])
        assert caught.value.details["reason"] == "source_output_range"
    else:
        result = model.moments([[], []], [[1], [2]], [0., 0.])
        assert result.covariance == [[maximum, 0.], [0., 0.]]


@pytest.mark.parametrize("value", ["0", "1e-330", "-1e-330", "1e-310", "-1e-310"])
def test_signed_cancellation_and_small_outputs_use_absolute_budget(value):
    a = arithmetic()
    c = mpmath.mp.clone()
    c.dps = 80
    assert a.round_scalar(c, c.mpf(value), c.mpf(0), c.mpf("1e-290")) == float(value)


def test_small_variance_cannot_borrow_large_neighbour_budget():
    a = arithmetic()
    c = mpmath.mp.clone()
    c.dps = 80
    assert a.round_scalar(c, c.mpf("1e20"), c.mpf(1), c.mpf("1e20"), variance=True)
    with pytest.raises(a.PrecisionFailure):
        a.round_scalar(c, c.mpf("1e-100"), c.mpf("1e-110"), c.mpf("1e-100"), variance=True)


def test_zero_scale_requires_zero_error_not_just_agreeing_zeros():
    a = arithmetic()
    c = mpmath.mp.clone()
    c.dps = 80
    assert a.round_scalar(c, c.zero, c.zero, c.zero) == 0.
    with pytest.raises(a.PrecisionFailure):
        a.round_scalar(c, c.zero, c.mpf("1e-90"), c.zero)


def test_conditional_solves_frozen_operands_with_local_bounds(frozen_fit):
    a = arithmetic()
    assert hasattr(a, "Conditional"), "audited conditional is not implemented"
    data, _ = frozen_fit
    train = np.asarray(data["training"])
    queries = np.asarray(data["query"])
    y = np.asarray(data["physical_gain"])
    p = data["parameters"]
    constant = p["mean_module.raw_constant"]
    source = a.Conditional(
        ((train - train.mean()) / train.std()).tolist(),
        (-(y - y.mean()) / y.std()).tolist(), [constant] * len(train), data["noise"],
        {"kind": "matern", "lengthscale": p["covar_module.kernel.base_kernel.raw_lengthscale"][0],
         "scales": [p["covar_module.kernel.raw_outputscale"]], "tables": []},
        [[] for _ in train])
    result = source.moments(((queries - train.mean()) / train.std()).tolist(),
                            [[] for _ in queries], [constant] * len(queries))
    assert_local_accuracy(data, tensor(result.mean), tensor(result.covariance))
    assert result.digits in (80, 110, 160)


@pytest.mark.parametrize("table", [[[1., .1], [.10000000000001, 1.]], [[1., 1.000000000001], [1.000000000001, 1.]]])
def test_category_table_must_be_exactly_symmetric_and_psd(table):
    a = arithmetic()
    with pytest.raises(EngineError) as caught:
        a.Conditional([[0.], [1.]], [0., 1.], [0., 0.], [.1, .1],
                      {"kind": "rbf", "lengthscale": [1.], "scales": [], "tables": [table]}, [[0], [1]])
    assert caught.value.details["reason"] == "invalid_source"


def test_zero_categorical_prior_row_has_mean_not_observation_target():
    a = arithmetic()
    model = a.Conditional([[], []], [3., 4.], [.2, .2], [.1, .1],
                          {"kind": None, "lengthscale": [], "scales": [], "tables": [[[0., 0.], [0., 1.]]]}, [[0], [1]])
    result = model.moments([[], []], [[0], [1]], [.2, .2])
    assert result.mean[0] == .2
    assert result.covariance[0] == [0., 0.]
    assert result.covariance[1][0] == 0.
    assert result.covariance[1][1] > 0


def test_endpoint_cache_matches_uncached_and_has_no_query_keys():
    a = arithmetic()
    assert hasattr(a.Conditional, "set_endpoints"), "bounded endpoint cache missing"
    args = ([[0.], [.5], [1.]], [.1, -.2, .3], [0.] * 3, [.03] * 3,
            {"kind": "matern", "lengthscale": [.8], "scales": [1.2], "tables": []}, [[], [], []])
    cached, plain = a.Conditional(*args), a.Conditional(*args)
    endpoints = [[.15], [.8], [.15]]
    cached.set_endpoints(endpoints, [[], [], []])
    for value in [.2, .4, .6]:
        x = [[value]]+endpoints
        expected = plain.moments(x, [[]] * 4, [0.] * 4)
        actual = cached.moments(x, [[]] * 4, [0.] * 4)
        assert actual.mean == expected.mean
        assert actual.covariance == expected.covariance
    assert set(cached._states) <= {50, 80, 110, 160}
    assert len(cached._solvers) <= 6


def test_actual_precision_escalation_and_cap_without_mocked_solver():
    a = arithmetic()
    def source(separation):
        return a.Conditional([[0.], [separation]], [0., 0.], [0., 0.], [0., 0.],
                              {"kind": "rbf", "lengthscale": [1.], "scales": [], "tables": []}, [[], []])
    result = source(1e-30).moments([[.7]], [[]], [0.])
    assert result.digits == 160 and result.covariance[0][0] > 0
    with pytest.raises(EngineError) as caught:
        source(1e-100).moments([[.7]], [[]], [0.])
    assert caught.value.details["reason"] == "source_precision_exhausted"


def test_actual_fitted_variance_underflows_before_a_later_affine_can_rescue_it():
    a = arithmetic()
    model = a.Conditional([[0.]], [0.], [0.], [1.],
                          {"kind": "rbf", "lengthscale": [1.], "scales": [1e-200, 1e-130], "tables": []}, [[]])
    with pytest.raises(EngineError) as caught:
        model.moments([[.7]], [[]], [0.])
    assert caught.value.details["reason"] == "source_output_range"


def test_actual_heterogeneous_variances_use_their_own_scales():
    a = arithmetic()
    model = a.Conditional([[]], [0.], [0.], [1.],
                          {"kind": None, "lengthscale": [], "scales": [],
                           "tables": [[[1e20, 0., 0.], [0., 1e-100, 0.], [0., 0., 1.]]]}, [[2]])
    result = model.moments([[], []], [[0], [1]], [0., 0.])
    assert result.covariance == [[1e20, 0.], [0., 1e-100]]


def test_exact_singular_category_training_fails_directly():
    a = arithmetic()
    with pytest.raises(EngineError) as caught:
        model = a.Conditional([[], []], [1., 2.], [0., 0.], [0., 0.],
                              {"kind": None, "lengthscale": [], "scales": [],
                               "tables": [[[1., 1.], [1., 1.]]]}, [[0], [1]])
        model.moments([[]], [[0]], [0.])
    assert caught.value.details["reason"] == "singular_training_covariance"


@pytest.mark.parametrize("train,table", [([[0.], [1.]], [[0., 0.], [0., 1.]]),
                                         ([[0.], [0.]], [[1., 1.], [1., 1.]])])
def test_proven_singular_categorical_relations_with_continuous_factor_fail_directly(train, table):
    a = arithmetic()
    with pytest.raises(EngineError) as caught:
        model = a.Conditional(train, [1., 2.], [0., 0.], [0., 0.],
                              {"kind": "rbf", "lengthscale": [1.], "scales": [1.3],
                               "tables": [table]}, [[0], [1]])
        model.moments([[.3]], [[0]], [0.])
    assert caught.value.details["reason"] == "singular_training_covariance"
