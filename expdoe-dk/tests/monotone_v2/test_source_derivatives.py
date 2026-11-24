"""Independent MP forward differences for kernel-only VJP/double backward."""
import mpmath
import pytest

from .test_source_arithmetic import arithmetic


@pytest.mark.parametrize("kind", ["rbf", "matern"])
def test_equal_nonzero_targets_and_mean_have_exact_zero_mean_derivatives(kind):
    a = arithmetic()
    model = a.Conditional([[0.], [.5], [1.]], [2.] * 3, [2.] * 3, [.03] * 3,
                          {"kind": kind, "lengthscale": [.8], "scales": [], "tables": []},
                          [[], [], []])
    x, categories, mean = [[.2], [.7]], [[], []], [2., 2.]
    assert model.moments(x, categories, mean).mean == mean
    assert model.vjp(x, categories, mean, [1., -.3], [[0., 0.], [0., 0.]]) == [[0.], [0.]]
    hv, dh, _ = model.hvp(x, categories, mean, [1., -.3], [[0., 0.], [0., 0.]], [[.4], [-.2]])
    assert hv == [[0.], [0.]]
    assert dh == [0., 0.]


@pytest.mark.parametrize("kind", ["rbf", "matern"])
def test_symmetric_noncoincident_training_accepts_cancelled_mean_gradient(kind):
    a = arithmetic()
    model = a.Conditional([[-.5], [.5]], [1., 1.], [0., 0.], [.03, .03],
                          {"kind": kind, "lengthscale": [.8], "scales": [], "tables": []}, [[], []])
    gradient = model.vjp([[0.]], [[]], [0.], [1.], [[0.]])
    assert abs(gradient[0][0]) < 1e-14
    hv, dh, dG = model.hvp([[0.]], [[]], [0.], [1.], [[0.]], [[1.]])
    assert hv[0][0] != 0.
    assert abs(dh[0]) < 1e-14
    assert abs(dG[0][0]) < 1e-14


@pytest.mark.parametrize("kind", ["rbf", "matern"])
def test_leaf_hessian_keeps_its_own_absolute_term_scale(kind):
    a = arithmetic()
    c = mpmath.mp.clone()
    c.dps = 80
    x = 1.+2**-20 if kind == "rbf" else .7
    specification = {"kind": kind, "lengthscale": [1.], "scales": [2.], "tables": []}
    _, hessian = a._kernel_jet(c, specification, [x], [0.], [], [], 2)
    z = c.mpf(x)
    if kind == "rbf":
        expected_scale = 2*c.exp(-z*z/2)*(z*z+1)
    else:
        radius = c.sqrt(5*z*z)
        expected_scale = 2*c.mpf(5)/3*c.exp(-radius)*(1+radius+5*z*z)
    assert abs(hessian[0][0].scale/expected_scale-1) < c.mpf("1e-70")


@pytest.mark.parametrize("kind", ["rbf", "matern"])
@pytest.mark.parametrize("loss_kind", ["covariance", "mixed", "skew", "zero", "known_noiseless"])
def test_mp_contractions_match_independent_forward_differences(kind, loss_kind):
    a = arithmetic()
    assert hasattr(a.Conditional, "vjp"), "MP derivative contractions are not implemented"
    train = [[-.7, .2], [.1, -.4], [.9, .8]]
    x = [[.2, .3], [.2, .3]]
    targets, noise, mean = [.1, -.3, .7], [.03] * 3, .15
    known = None
    if loss_kind == "known_noiseless":
        noise, x, known = [0.] * 3, [train[0][:], [.2, .3]], [0, -1]
    ell = [.8, 1.2]
    model = a.Conditional(train, targets, [mean] * 3, noise,
                          {"kind": kind, "lengthscale": ell, "scales": [1.1], "tables": []},
                          [[], [], []])
    h = [0., 0.] if loss_kind != "mixed" else [.2, -.3]
    G = [[.3, -.7], [.2, .8]]
    if loss_kind == "skew":
        G = [[0., .4], [-.4, 0.]]
    if loss_kind == "zero":
        G = [[0., 0.], [0., 0.]]
    v = [[.3, -.6], [-.2, .5]]
    actual = model.vjp(x, [[], []], [mean] * 2, h, G, known=known)
    hv, dh, dG = model.hvp(x, [[], []], [mean] * 2, h, G, v, known=known)
    c = mpmath.mp.clone()
    c.dps = 100

    def kernel(u, w):
        r2 = c.fsum(((c.mpf(s) - c.mpf(t))/c.mpf(l))**2 for s, t, l in zip(u, w, ell))
        if kind == "rbf":
            return c.mpf(1.1) * c.exp(-r2/2)
        r = c.sqrt(5*r2)
        return c.mpf(1.1) * (1+r+r*r/3) * c.exp(-r)

    A = c.matrix([[kernel(u, w) for w in train] for u in train])
    for i in range(3):
        A[i, i] += c.mpf(noise[i])
    inverse = A**-1

    def moments(z):
        B = c.matrix([[kernel(u, w) for w in train] for u in z])
        m = c.ones(2, 1)*c.mpf(mean) + B*inverse*c.matrix([c.mpf(y)-c.mpf(mean) for y in targets])
        C = c.matrix([[kernel(u, w) for w in z] for u in z]) - B*inverse*B.T
        return m, C

    def loss(z):
        m, C = moments(z)
        return c.fsum(c.mpf(h[i])*m[i] for i in range(2)) + c.fsum(
            c.mpf(G[i][j])*C[i, j] for i in range(2) for j in range(2))

    step = c.mpf("1e-20")
    for i in range(2):
        for d in range(2):
            plus, minus = [[c.mpf(z) for z in row] for row in x], [[c.mpf(z) for z in row] for row in x]
            plus[i][d] += step
            minus[i][d] -= step
            expected = (loss(plus)-loss(minus))/(2*step)
            assert abs(c.mpf(actual[i][d])-expected) < c.mpf("1e-14")
    plus = [[c.mpf(z)+step*c.mpf(vv) for z, vv in zip(row, vr)] for row, vr in zip(x, v)]
    minus = [[c.mpf(z)-step*c.mpf(vv) for z, vv in zip(row, vr)] for row, vr in zip(x, v)]
    second = (loss(plus)-2*loss(x)+loss(minus))/step**2
    actual_second = c.fsum(c.mpf(hv[i][d])*c.mpf(v[i][d]) for i in range(2) for d in range(2))
    assert abs(actual_second-second) < c.mpf("1e-14")
    mp, Cp = moments(plus)
    mm, Cm = moments(minus)
    for i in range(2):
        assert abs(c.mpf(dh[i])-(mp[i]-mm[i])/(2*step)) < c.mpf("1e-14")
        for j in range(2):
            assert abs(c.mpf(dG[i][j])-(Cp[i,j]-Cm[i,j])/(2*step)) < c.mpf("1e-14")
    if loss_kind in ("skew", "zero"):
        assert actual == [[0., 0.], [0., 0.]]
        assert hv == [[0., 0.], [0., 0.]]
