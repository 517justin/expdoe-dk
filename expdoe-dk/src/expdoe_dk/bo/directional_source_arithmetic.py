"""Bounded MP formulas for an owned frozen GP, with local error estimates.

No live Torch models are inspected here. The error model is an engineering
estimate, not an interval enclosure or a claim about arbitrary MP internals.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from ..errors import EngineError, ErrorCode


_MAX_NORMAL = float.fromhex("0x1.fffffffffffffp+1023")


class PrecisionFailure(Exception):
    """Retry this same source at the next approved precision pair."""


def _failure(reason, **details):
    return EngineError(ErrorCode.MODEL_FIT_FAILED, "Frozen GP arithmetic failed validation",
                       details={"reason": reason, **details})


def round_scalar(c, value, error, scale, *, variance=False):
    """Section-14 total-error budget, including actual binary64 conversion."""
    if not all(c.isfinite(v) for v in (value, error, scale)) or error < 0 or scale < 0:
        raise PrecisionFailure("undefined local bound")
    if variance and value <= 0:
        raise PrecisionFailure("unresolved positive variance")
    if variance and (value < c.mpf(2)**-1022 or value > c.mpf(_MAX_NORMAL)):
        raise _failure("source_output_range")
    result = float(value)
    if not math.isfinite(result):
        raise _failure("source_output_range")
    if variance and result < math.ldexp(1., -1022):
        raise _failure("source_output_range")
    if error + abs(c.mpf(result) - value) > 32 * c.mpf(2)**-53 * scale:
        raise PrecisionFailure("local output budget")
    return result


def audit_affine(values, scales, offsets, products, actual, errors, variances):
    """Check the fixed map on its actual binary64 operands and intermediates.

    Samples enter at this boundary; this does not certify their root or MC law.
    Moments may additionally carry their incoming construction error estimates.
    """
    import mpmath
    if any(not math.isfinite(v) for group in (values, scales, offsets, products, actual) for v in group):
        raise _failure("source_output_range")
    last = None
    for low_digits, high_digits in Conditional.schedule:
        low, high = mpmath.mp.clone(), mpmath.mp.clone()
        low.dps, high.dps = low_digits, high_digits
        result = []
        try:
            for x, a, b, p, f, incoming, variance in zip(values, scales, offsets, products, actual, errors, variances):
                c = high
                product = c.mpf(a)*c.mpf(x)
                reference = product+c.mpf(b)
                comparison = low.mpf(a)*low.mpf(x)+low.mpf(b)
                F = abs(product)+abs(c.mpf(b))
                error = (abs(c.mpf(a))*c.mpf(incoming) + abs(reference-c.mpf(comparison))
                         + 8*_gamma(c, 2)*F)
                if variance and x != 0:
                    if (product <= 0 or reference <= 0 or p <= 0 or f <= 0
                            or min(product, reference, c.mpf(p), c.mpf(f)) < c.mpf(2)**-1022
                            or max(product, reference) > c.mpf(_MAX_NORMAL)):
                        raise _failure("source_output_range")
                # A multiply's rounding is separately inspected against this
                # same fixed-map local scale, not a neighbouring output's scale.
                if abs(c.mpf(p)-product) > 32*c.mpf(2)**-53*F:
                    raise PrecisionFailure("affine intermediate budget")
                total = error+abs(c.mpf(f)-reference)
                if total > 32*c.mpf(2)**-53*F:
                    raise PrecisionFailure("affine local budget")
                result.append(total)
            return result
        except PrecisionFailure as exc:
            last = str(exc)
    raise _failure("source_precision_exhausted", error=last)


def preflight_affine(values, scales, offsets, errors, variances):
    products = [x*a for x, a in zip(values, scales)]
    nominal = [p+b for p, b in zip(products, offsets)]
    propagated = audit_affine(values, scales, offsets, products, nominal, errors, variances)
    return nominal, propagated


@dataclass(frozen=True)
class _Number:
    value: object
    error: object
    scale: object


def _exact(c, value):
    value = c.mpf(value)
    return _Number(value, c.zero, abs(value))


def _gamma(c, count):
    product = count * c.eps
    if product >= 1:
        raise PrecisionFailure("operation count")
    return product / (1 - product)


def _sum(c, terms):
    terms = tuple(terms)
    scale = c.fsum(abs(t.value) if t.value or not t.error else t.scale for t in terms)
    return _Number(c.fsum(t.value for t in terms),
                   c.fsum(t.error for t in terms) + 8 * _gamma(c, 2 * len(terms)) * scale,
                   scale)


def _neg(a):
    return _Number(-a.value, a.error, a.scale)


def _mul(c, a, b):
    scale = abs(a.value * b.value)
    if not scale and (a.error or b.error):
        scale = a.scale * b.scale
    error = abs(a.value) * b.error + abs(b.value) * a.error + a.error * b.error
    return _Number(a.value * b.value, error + 8 * _gamma(c, 1) * scale, scale)


def _div(c, a, b):
    margin = abs(b.value) - b.error
    if margin <= 0:
        raise PrecisionFailure("division sensitivity")
    value = a.value / b.value
    error = (a.error + abs(value) * b.error) / margin
    return _Number(value, error + 8 * _gamma(c, 1) * abs(value), abs(value))


def _sqrt(c, a):
    if not a.value and not a.error:
        return _exact(c, 0)
    if a.value <= a.error:
        raise PrecisionFailure("sqrt sensitivity")
    value = c.sqrt(a.value)
    error = a.error / (2 * c.sqrt(a.value - a.error)) + 8 * c.eps * abs(value)
    return _Number(value, error, abs(value))


def _exp(c, a):
    value = c.exp(a.value)
    return _Number(value, abs(value) * c.expm1(a.error) + 8 * c.eps * abs(value), abs(value))


def _compared(c, high, low):
    return _Number(high.value, high.error + abs(high.value - c.mpf(low.value)), high.scale)


def _kernel(c, specification, x, y, categories_x, categories_y):
    one = _exact(c, 1)
    if specification["kind"] is None:
        value = one
    else:
        squared = []
        for a, b, length in zip(x, y, specification["lengthscale"]):
            # Equality is an algebraic identity over exact binary64 operands.
            delta = (_exact(c, 0) if a == b else
                     _sum(c, (_exact(c, a), _exact(c, -b))))
            normalized = _div(c, delta, _exact(c, length))
            squared.append(_mul(c, normalized, normalized))
        radius2 = _sum(c, squared)
        if specification["kind"] == "rbf":
            value = _exp(c, _mul(c, _exact(c, -.5), radius2))
        elif specification["kind"] == "matern":
            radius = _sqrt(c, _mul(c, _exact(c, 5), radius2))
            polynomial = _sum(c, (one, radius, _div(c, _mul(c, radius, radius), _exact(c, 3))))
            value = _mul(c, polynomial, _exp(c, _neg(radius)))
        else:
            raise _failure("invalid_source", component="stationary kernel")
        # The reviewed per-leaf count is conservative, in addition to explicit
        # exp/sqrt sensitivities above; it does not count hidden solver work.
        value = _Number(value.value,
                        value.error + 8 * _gamma(c, 256 + 32 * len(x)) * value.scale,
                        value.scale)
    for table, i, j in zip(specification["tables"], categories_x, categories_y):
        value = _mul(c, value, _exact(c, table[i][j]))
    for scale in specification["scales"]:
        value = _mul(c, value, _exact(c, scale))
    return value


def _kernel_jet(c, specification, x, y, categories_x, categories_y, order):
    d = len(x)
    zero, one = _exact(c, 0), _exact(c, 1)
    if specification["kind"] is None:
        return [zero] * d, [[zero] * d for _ in range(d)]
    delta = [zero if a == b else _sum(c, (_exact(c, a), _exact(c, -b))) for a, b in zip(x, y)]
    inverse = [_div(c, one, _mul(c, _exact(c, l), _exact(c, l))) for l in specification["lengthscale"]]
    v = [_mul(c, a, b) for a, b in zip(delta, inverse)]
    r2 = _sum(c, [_mul(c, a, b) for a, b in zip(delta, v)])
    def scaled(a, b):
        product = _mul(c, a, b)
        return _Number(product.value, product.error, a.scale*b.scale)
    if specification["kind"] == "rbf":
        common = _exp(c, _mul(c, _exact(c, -.5), r2))
        gradient = [_neg(_mul(c, common, z)) for z in v]
        hessian = [[scaled(common, _sum(c, (_mul(c, v[a], v[b]),
                                            _neg(inverse[a]) if a == b else zero)))
                    for b in range(d)] for a in range(d)] if order == 2 else []
    else:
        ar = _sqrt(c, _mul(c, _exact(c, 5), r2))
        common = _mul(c, _div(c, _exact(c, -5), _exact(c, 3)), _exp(c, _neg(ar)))
        polynomial = _sum(c, (one, ar))
        gradient = [_mul(c, common, _mul(c, polynomial, z)) for z in v]
        hessian = [[scaled(common, _sum(c, (
            _mul(c, polynomial, inverse[a]) if a == b else zero,
            _neg(_mul(c, _exact(c, 5), _mul(c, v[a], v[b]))))))
            for b in range(d)] for a in range(d)] if order == 2 else []

    def finish(value):
        value = _Number(value.value, value.error + 8*_gamma(c, 256+32*d)*value.scale, value.scale)
        for table, i, j in zip(specification["tables"], categories_x, categories_y):
            value = scaled(value, _exact(c, table[i][j]))
        for scale in specification["scales"]:
            value = scaled(value, _exact(c, scale))
        return value

    return [finish(v) for v in gradient], [[finish(v) for v in row] for row in hessian]


def _matrix_values(c, matrix):
    return c.matrix([[entry.value for entry in row] for row in matrix])


def _norm(c, matrix):
    return max(c.fsum(abs(matrix[i, j]) for j in range(matrix.cols)) for i in range(matrix.rows))


class _PrecisionState:
    def __init__(self, owner, digits):
        import mpmath
        self.c = c = mpmath.mp.clone()
        c.dps = digits
        self.A = [[_kernel(c, owner.kernel, x, y, cx, cy)
                   for y, cy in zip(owner.train, owner.categories)]
                  for x, cx in zip(owner.train, owner.categories)]
        for i, noise in enumerate(owner.noise):
            self.A[i][i] = _sum(c, (self.A[i][i], _exact(c, noise)))
        # Equal owned binary64 operands subtract exactly, including nonzero
        # constants. This certifies the RHS, not a moving-query derivative mask.
        self.f = [(_exact(c, 0) if y == m else _sum(c, (_exact(c, y), _exact(c, -m))))
                  for y, m in zip(owner.targets, owner.train_mean)]
        self.matrix = _matrix_values(c, self.A)
        try:
            self.factor = c.cholesky(self.matrix)
            self.inverse = self.solve(c.eye(len(owner.train)))
        except (ValueError, ZeroDivisionError) as exc:
            raise PrecisionFailure("training Cholesky") from exc

    def solve(self, rhs):
        c = self.c
        result = c.matrix(rhs.rows, rhs.cols)
        for j in range(rhs.cols):
            result[:, j] = c.lu_solve(self.factor.T, c.lu_solve(self.factor, rhs[:, j]))
        return result


class _AuditedSolve:
    def __init__(self, high, low):
        self.state = high
        self.c = c = high.c
        self.A = [[_compared(c, h, l) for h, l in zip(hr, lr)]
                  for hr, lr in zip(high.A, low.A)]
        self.delta = c.matrix([[v.error for v in row] for row in self.A])
        n = high.matrix.rows
        residual = c.eye(n) - high.matrix * high.inverse
        allowance = c.matrix(n, n)
        for i in range(n):
            for j in range(n):
                scale = (int(i == j) + c.fsum(abs(high.matrix[i, k] * high.inverse[k, j])
                                            for k in range(n)))
                allowance[i, j] = 8 * _gamma(c, 2*n + 1) * scale
        inverse_norm = _norm(c, high.inverse)
        rho = _norm(c, residual) + _norm(c, allowance) + _norm(c, self.delta) * inverse_norm
        if rho >= c.mpf(.5):
            raise PrecisionFailure("inverse residual")
        self.M = inverse_norm / (1 - rho)
        condition = _norm(c, high.matrix) * self.M
        if 8 * _gamma(c, 2*n + 1) * condition >= c.mpf(1) / 16:
            raise PrecisionFailure("training conditioning")

    def solve(self, rhs):
        c, n = self.c, len(rhs)
        values = _matrix_values(c, rhs)
        solution = self.state.solve(values)
        result = [[None] * values.cols for _ in range(n)]
        for j in range(values.cols):
            residuals = []
            for i in range(n):
                terms = [self.state.matrix[i, k] * solution[k, j] for k in range(n)]
                residual = abs(values[i, j] - c.fsum(terms))
                rounding = 8 * _gamma(c, 2*n + 1) * (abs(values[i, j]) + c.fsum(map(abs, terms)))
                operand = rhs[i][j].error + c.fsum(self.delta[i, k] * abs(solution[k, j])
                                                  for k in range(n))
                residuals.append(residual + rounding + operand)
            bound = self.M * max(residuals)
            for i in range(n):
                result[i][j] = _Number(solution[i, j], bound, abs(solution[i, j]))
        return result


@dataclass(frozen=True)
class Moments:
    mean: list
    covariance: list
    mean_error: list
    covariance_error: list
    digits: int


class Conditional:
    """One independent batch's conditional, with bounded training/endpoint caches."""
    schedule = ((50, 80), (80, 110), (110, 160))

    def __init__(self, train, targets, train_mean, noise, kernel, categories):
        from copy import deepcopy
        self.train, self.targets, self.train_mean, self.noise, self.kernel, self.categories = deepcopy(
            (train, targets, train_mean, noise, kernel, categories))
        n = len(train)
        if not n or any(len(v) != n for v in (targets, train_mean, noise, categories)):
            raise _failure("invalid_source", component="training shapes")
        numbers = [v for row in train for v in row] + targets + train_mean + noise
        numbers += kernel["lengthscale"] + kernel["scales"]
        if (not all(math.isfinite(v) for v in numbers) or any(v < 0 for v in noise)
                or any(v <= 0 for v in kernel["lengthscale"] + kernel["scales"])):
            raise _failure("invalid_source", component="parameters or noise")
        for table in kernel["tables"]:
            self._validate_table(table)
        self._validate_categories(categories)
        # Within coincident stationary-feature groups the stationary factor is
        # exactly one. A null principal block of this PSD Gram matrix certifies
        # singular training A, including zero-prior rows and category relations.
        groups = {}
        for i, point in enumerate(train):
            groups.setdefault(() if kernel["kind"] is None else tuple(point), []).append(i)
        for indices in groups.values():
            training_matrix = []
            for i in indices:
                row = []
                for j in indices:
                    value = Fraction(1)
                    for table, a, b in zip(kernel["tables"], categories[i], categories[j]):
                        value *= Fraction(table[a][b])
                    for scale in kernel["scales"]:
                        value *= Fraction(scale)
                    row.append(value + (Fraction(noise[i]) if i == j else 0))
                training_matrix.append(row)
            if self._rational_psd_rank(training_matrix) < len(indices):
                raise _failure("singular_training_covariance")
        self._states = {}
        self._solvers = {}
        self._endpoints = []
        self._endpoint_categories = []

    @staticmethod
    def _validate_table(table):
        n = len(table)
        if (not n or any(len(row) != n for row in table)
                or any(not math.isfinite(v) for row in table for v in row)
                or any(table[i][j] != table[j][i] for i in range(n) for j in range(n))):
            raise _failure("invalid_source", component="categorical symmetry/shape")
        # Binary64 tables are rational. Exact symmetric elimination accepts PSD
        # nullspaces without clipping a tolerated but genuinely negative mode.
        matrix = [[Fraction(v) for v in row] for row in table]
        Conditional._rational_psd_rank(matrix)

    @staticmethod
    def _rational_psd_rank(matrix):
        n, rank = len(matrix), 0
        for k in range(n):
            pivot = matrix[k][k]
            if pivot < 0 or (pivot == 0 and any(matrix[i][k] for i in range(k+1, n))):
                raise _failure("invalid_source", component="categorical PSD")
            if pivot:
                rank += 1
                for i in range(k+1, n):
                    for j in range(i, n):
                        matrix[i][j] -= matrix[i][k]*matrix[j][k]/pivot
                        matrix[j][i] = matrix[i][j]
        return rank

    def _validate_categories(self, rows):
        tables = self.kernel["tables"]
        if any(len(row) != len(tables) or any(type(i) is not int or not 0 <= i < len(table)
                                             for i, table in zip(row, tables)) for row in rows):
            raise _failure("invalid_source", component="categorical indices")

    def set_endpoints(self, endpoints, categories):
        if self._states or self._endpoints:
            raise _failure("source_state_changed", component="endpoint generation")
        from copy import deepcopy
        self._validate_categories(categories)
        self._endpoints, self._endpoint_categories = deepcopy((endpoints, categories))

    def _state(self, digits):
        if digits not in self._states:
            self._states[digits] = _PrecisionState(self, digits)
        return self._states[digits]

    def _solver(self, low_digits, high_digits):
        key = low_digits, high_digits
        if key not in self._solvers:
            high, low = self._state(high_digits), self._state(low_digits)
            solver = _AuditedSolve(high, low)
            f = [[_compared(high.c, h, l)] for h, l in zip(high.f, low.f)]
            self._solvers[key] = solver, solver.solve(f)
        return self._solvers[key]

    def _raw(self, state, x, categories):
        c = state.c
        e = len(self._endpoints)
        if e and (x[-e:] != self._endpoints or categories[-e:] != self._endpoint_categories):
            raise _failure("source_state_changed", component="endpoint operands")
        if e and not hasattr(state, "endpoint_raw"):
            E, EC = self._endpoints, self._endpoint_categories
            state.endpoint_raw = (
                [[_kernel(c, self.kernel, a, b, ca, cb) for b, cb in zip(self.train, self.categories)]
                 for a, ca in zip(E, EC)],
                [[_kernel(c, self.kernel, a, b, ca, cb) for b, cb in zip(E, EC)] for a, ca in zip(E, EC)])
        q = len(x)-e
        B = [[_kernel(c, self.kernel, a, b, ca, cb)
              for b, cb in zip(self.train, self.categories)] for a, ca in zip(x[:q], categories[:q])]
        if e:
            B += state.endpoint_raw[0]
        D = [[None] * len(x) for _ in x]
        for i in range(len(x)):
            for j in range(len(x)):
                D[i][j] = (state.endpoint_raw[1][i-q][j-q] if i >= q and j >= q else
                           _kernel(c, self.kernel, x[i], x[j], categories[i], categories[j]))
        return B, D

    def _cross_solve(self, solver, B, identities):
        n, e, total = len(self.train), len(self._endpoints), len(B)
        q = total-e
        if e and not hasattr(solver, "endpoint_P"):
            solver.endpoint_P = solver.solve([[B[j][i] for j in range(q, total)] for i in range(n)])
        if not q:
            P = [row[:] for row in solver.endpoint_P]
        else:
            P = solver.solve([[B[j][i] for j in range(q)] for i in range(n)])
            if e:
                P = [row + solver.endpoint_P[i] for i, row in enumerate(P)]
        # If b_j is exactly a noise-free training column, A^-1 b_j is a
        # coordinate vector. This solve identity is not a derivative mask.
        for j, match in enumerate(identities):
            if match >= 0:
                for i in range(n):
                    P[i][j] = _exact(solver.c, int(i == match))
        return P

    def _kernel_matches(self, x, categories):
        return [next((i for i, (t, ct, noise) in enumerate(zip(self.train, self.categories, self.noise))
                      if noise == 0 and z == t and cz == ct), -1) for z, cz in zip(x, categories)]

    def _evaluate(self, solver, alpha, raw, mean, identities, comparison=None):
        c, n = solver.c, len(self.train)
        B, D = raw
        if comparison is not None:
            B, D = ([[ _compared(c, h, l) for h, l in zip(hr, lr)]
                     for hr, lr in zip(H, L)] for H, L in zip(raw, comparison))
        q = len(B)
        P = self._cross_solve(solver, B, identities)
        e, variable = len(self._endpoints), q-len(self._endpoints)
        cached = hasattr(solver, "endpoint_moments")
        mu, covariance = [], []
        for i in range(q):
            mu.append(solver.endpoint_moments[0][i-variable] if cached and i >= variable else
                      _sum(c, [_exact(c, mean[i])] + [_mul(c, B[i][t], alpha[t][0]) for t in range(n)]))
            covariance.append([solver.endpoint_moments[1][i-variable][j-variable]
                               if cached and i >= variable and j >= variable else
                               _sum(c, [D[i][j]] + [_neg(_mul(c, B[i][t], P[t][j])) for t in range(n)])
                               for j in range(q)])
        if e and not cached:
            solver.endpoint_moments = mu[variable:], [row[variable:] for row in covariance[variable:]]
        return mu, covariance

    def _jets(self, solver, alpha, raw, x, categories, order, nvar, low_state=None, low_raw=None):
        c, n, q, d = solver.c, len(self.train), len(x), len(x[0])
        B = raw[0]
        if low_raw is not None:
            B = [[_compared(c, h, l) for h, l in zip(hr, lr)] for hr, lr in zip(B, low_raw[0])]
        P = self._cross_solve(solver, B, self._kernel_matches(x, categories))

        def jet(a, b, ca, cb):
            g, H = _kernel_jet(c, self.kernel, a, b, ca, cb, order)
            if low_state is not None:
                lg, lH = _kernel_jet(low_state.c, self.kernel, a, b, ca, cb, order)
                g = [_compared(c, h, l) for h, l in zip(g, lg)]
                H = [[_compared(c, h, l) for h, l in zip(hr, lr)] for hr, lr in zip(H, lH)]
            for value in g + [value for row in H for value in row]:
                round_scalar(c, value.value, value.error, value.scale)
            return g, H

        train_jets = [[jet(x[i], t, categories[i], ct) for t, ct in zip(self.train, self.categories)]
                      for i in range(nvar)]
        query_jets = [[jet(x[i], z, categories[i], cz) for z, cz in zip(x, categories)]
                      for i in range(nvar)]
        derivatives = {}
        solved = (solver.solve([[train_jets[j][t][0][b] for j in range(nvar) for b in range(d)]
                                for t in range(n)]) if order == 2 else None)
        for i in range(nvar):
            for a in range(d):
                di = [train_jets[i][t][0][a] for t in range(n)]
                derivatives["u", i, a] = _sum(c, [_mul(c, di[t], alpha[t][0]) for t in range(n)])
                for j in range(q):
                    derivatives["J", i, j, a] = _sum(c, [query_jets[i][j][0][a]] +
                        [_neg(_mul(c, di[t], P[t][j])) for t in range(n)])
                if order == 2:
                    for b in range(d):
                        ei = [train_jets[i][t][1][a][b] for t in range(n)]
                        derivatives["h", i, a, b] = _sum(c, [_mul(c, ei[t], alpha[t][0]) for t in range(n)])
                        for j in range(q):
                            derivatives["U", i, j, a, b] = _sum(c, [query_jets[i][j][1][a][b]] +
                                [_neg(_mul(c, ei[t], P[t][j])) for t in range(n)])
                            if j < nvar:
                                derivatives["V", i, j, a, b] = _sum(c, [_neg(query_jets[i][j][1][a][b])] +
                                    [_neg(_mul(c, di[t], solved[t][j*d+b])) for t in range(n)])
        return derivatives

    @staticmethod
    def _contract(c, jets, operation, q, d):
        kind, h, G, v, nvar = operation
        def exact(x):
            return _exact(c, x)
        def product(*args):
            result = exact(1)
            for term in args:
                product = _mul(c, result, term)
                result = _Number(product.value, product.error, result.scale*term.scale)
            return result
        def finish(terms):
            result = _sum(c, terms)
            # Expand the fixed u/J/h/U/V scalar formula through its cotangent
            # products; a tiny rounded cancellation must not discard its terms.
            scale = c.fsum(term.scale for term in terms)
            return round_scalar(c, result.value, result.error, scale)
        gradient = []
        for i in range(nvar):
            row = []
            for a in range(d):
                if kind == "vjp":
                    terms = [product(exact(h[i]), jets["u", i, a])]
                    for j in range(q):
                        # Expand the two cotangents before summing: skew or
                        # cancelling losses retain their own local term scale.
                        terms.extend(product(exact(weight), jets["J", i, j, a])
                                     for weight in (G[i][j], G[j][i]))
                else:
                    terms = [product(exact(h[i]), jets["h", i, a, b], exact(v[i][b])) for b in range(d)]
                    for j in range(q):
                        for b in range(d):
                            for weight in (G[i][j], G[j][i]):
                                terms.append(product(exact(weight), jets["U", i, j, a, b], exact(v[i][b])))
                                if j < nvar:
                                    terms.append(product(exact(weight), jets["V", i, j, a, b], exact(v[j][b])))
                row.append(finish(terms))
            gradient.append(row)
        if kind == "vjp":
            return gradient
        dh = [finish([product(jets["u", i, a], exact(v[i][a])) for a in range(d)])
              if i < nvar else 0. for i in range(q)]
        dG = []
        for i in range(q):
            row = []
            for j in range(q):
                terms = []
                if i < nvar:
                    terms += [product(jets["J", i, j, a], exact(v[i][a])) for a in range(d)]
                if j < nvar:
                    terms += [product(jets["J", j, i, a], exact(v[j][a])) for a in range(d)]
                row.append(finish(terms))
            dG.append(row)
        return gradient, dh, dG

    def vjp(self, x, categories, mean, h, G, *, nvar=None, known=None):
        return self.moments(x, categories, mean, known,
                            _operation=("vjp", h, G, None, len(x) if nvar is None else nvar))

    def hvp(self, x, categories, mean, h, G, v, *, nvar=None, known=None):
        return self.moments(x, categories, mean, known,
                            _operation=("hvp", h, G, v, len(x) if nvar is None else nvar))

    def moments(self, x, categories, mean, known=None, *, _operation=None):
        if len(x) != len(categories) or len(x) != len(mean):
            raise _failure("invalid_source", component="query shape")
        self._validate_categories(categories)
        identities = self._kernel_matches(x, categories)
        known = [-1] * len(x) if known is None else list(known)
        for i, row in enumerate(categories):
            if any(all(v == 0 for v in table[index]) for index, table in zip(row, self.kernel["tables"])):
                known[i] = -2
        last = None
        for low_digits, high_digits in self.schedule:
            try:
                high, low = self._state(high_digits), self._state(low_digits)
                solver, alpha = self._solver(low_digits, high_digits)
                low_solver, low_alpha = self._solver(low_digits, low_digits)
                high_raw, low_raw = self._raw(high, x, categories), self._raw(low, x, categories)
                hm, hC = self._evaluate(solver, alpha, high_raw, mean, identities, low_raw)
                lm, lC = self._evaluate(low_solver, low_alpha, low_raw, mean, identities)
                c, q = high.c, len(x)
                hm = [_compared(c, h, l) for h, l in zip(hm, lm)]
                hC = [[_compared(c, h, l) for h, l in zip(hr, lr)] for hr, lr in zip(hC, lC)]
                for i, match in enumerate(known):
                    if match != -1:
                        hm[i] = _exact(c, self.targets[match] if match >= 0 else mean[i])
                        for j in range(q):
                            hC[i][j] = hC[j][i] = _exact(c, 0)
                result_mean = [round_scalar(c, v.value, v.error, v.scale) for v in hm]
                for i in range(q):
                    if known[i] == -1:
                        round_scalar(c, hC[i][i].value, hC[i][i].error, hC[i][i].value, variance=True)
                covariance, errors = [], []
                for i in range(q):
                    row, erow, normalized = [], [], c.zero
                    for j in range(q):
                        v = hC[i][j]
                        budget_scale = c.sqrt(hC[i][i].value * hC[j][j].value)
                        row.append(round_scalar(c, v.value, v.error, budget_scale))
                        erow.append(v.error + abs(c.mpf(row[-1]) - v.value))
                        if budget_scale:
                            normalized += abs(v.value - c.mpf(lC[i][j].value)) / budget_scale
                            if abs(v.value - hC[j][i].value) > 32*c.mpf(2)**-53*budget_scale:
                                raise PrecisionFailure("scaled symmetry")
                    if normalized > q*32*c.mpf(2)**-53:
                        raise PrecisionFailure("scaled precision comparison")
                    covariance.append(row)
                    errors.append(erow)
                if _operation is not None:
                    order = 1 if _operation[0] == "vjp" else 2
                    nvar = _operation[-1]
                    jets = self._jets(solver, alpha, high_raw, x, categories, order, nvar, low, low_raw)
                    low_jets = self._jets(low_solver, low_alpha, low_raw, x, categories, order, nvar)
                    jets = {key: _compared(c, value, low_jets[key]) for key, value in jets.items()}
                    for key, value in jets.items():
                        try:
                            round_scalar(c, value.value, value.error, value.scale)
                        except PrecisionFailure as exc:
                            raise PrecisionFailure(f"derivative {key}: {exc}") from exc
                    return self._contract(c, jets, _operation, q, len(x[0]))
                return Moments(result_mean, covariance,
                               [v.error + abs(c.mpf(r)-v.value) for r, v in zip(result_mean, hm)],
                               errors, high_digits)
            except PrecisionFailure as exc:
                last = str(exc)
        raise _failure("source_precision_exhausted", error=last)
