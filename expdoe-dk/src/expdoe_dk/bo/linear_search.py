"""Translate physical linear equalities and seed BoTorch's constrained search."""
from __future__ import annotations

import math
import inspect
import warnings
from dataclasses import dataclass

import pandas as pd
import numpy as np
import torch
from botorch.exceptions.warnings import OptimizationWarning
from botorch.generation import gen_candidates_scipy
from scipy.linalg import qr
from scipy.optimize import linprog

from ..domain import LinearConstraint
from ..errors import EngineError, ErrorCode
from ..space import LinearConstraint as LegacyLinearConstraint

_PHYSICAL_TOL = 1e-9  # Matches both physical equality predicates.
_BOUNDS_ROUNDOFF = 1e-12

_BotorchRow = tuple[torch.Tensor, torch.Tensor, float]


@dataclass(frozen=True)
class ContinuousSubspace:
    fixed: dict[int, float]
    equalities: tuple[_BotorchRow, ...] = ()
    inequalities: tuple[_BotorchRow, ...] = ()
    seeds: torch.Tensor | None = None


@dataclass(frozen=True)
class _AffineRow:
    equality: bool
    slopes: torch.Tensor
    offsets: tuple[float, ...]
    fixed_terms: tuple[tuple[int, float], ...]
    bound: float


def _physical_rows(space):
    """Yield physical rows in BoTorch's equality or >= convention."""
    for constraint in space.constraints:
        if isinstance(constraint, LinearConstraint) and constraint.hard:
            sign = -1 if constraint.operator == "<=" else 1
            yield constraint.coefficients, sign, constraint.operator == "==", constraint.bound
        elif isinstance(constraint, LegacyLinearConstraint):
            lo, hi = constraint.lower, constraint.upper
            if lo == hi:
                yield constraint.coeffs, 1, True, lo
            else:
                if math.isfinite(lo):
                    yield constraint.coeffs, 1, False, lo - _PHYSICAL_TOL
                if math.isfinite(hi):
                    yield constraint.coeffs, -1, False, hi + _PHYSICAL_TOL


def compile_linear_search(space):
    """Leave finite-only and inequality-only searches on the existing path."""
    continuous = [parameter for parameter in space.params if parameter.kind == "continuous"]
    physical = tuple(_physical_rows(space)) if continuous else ()
    if not any(equality for _, _, equality, _ in physical):
        return ()

    nonaffine = {parameter.name for parameter in continuous
                 if parameter.transform == "log" or parameter.log_scale}
    indices = {parameter.name: index for index, parameter in enumerate(space.params)}
    continuous_names = {parameter.name for parameter in continuous}
    rows = []
    for coefficients, sign, equality, bound in physical:
        involved = sorted(name for name in nonaffine if coefficients.get(name, 0) != 0)
        if involved:
            if equality:
                raise EngineError(ErrorCode.ACQUISITION_INCOMPATIBLE,
                    "Continuous linear equality search requires affine coordinates; "
                    "provide a feasible candidate catalog with candidates= instead",
                    details={"nonaffine_parameters": involved})
            # Non-affine inequalities are still enforced in physical space.
            continue

        slopes, offsets = [], []
        for parameter in continuous:
            coefficient = coefficients.get(parameter.name, 0)
            if coefficient:
                low, high = parameter.bounds
                slopes.append(sign * coefficient * (high - low))
                offsets.append(sign * coefficient * low)
            else:
                slopes.append(0.)
        fixed_terms = tuple((indices[name], sign * coefficient)
                            for name, coefficient in coefficients.items()
                            if name not in continuous_names and coefficient)
        if not all(math.isfinite(value) for value in (
            bound, *slopes, *offsets, *(value for _, value in fixed_terms),
        )):
            raise EngineError(ErrorCode.CONFIG_INVALID,
                "Linear constraint cannot be represented in finite model coordinates; "
                "provide a feasible candidate catalog with candidates= instead")
        rows.append(_AffineRow(equality, torch.tensor(slopes, dtype=torch.double),
                               tuple(offsets), fixed_terms, sign * bound))
    return tuple(rows)


def _substituted_rows(space, rows, fixed):
    needed = {index for row in rows for index, _ in row.fixed_terms}
    physical = {index: float(space.params[index].decode([fixed[index]])[0]) for index in needed}
    equalities, inequalities, tolerances = [], [], []
    for row in rows:
        rhs = math.fsum([row.bound, *(-offset for offset in row.offsets),
                        *(-coefficient * physical[index] for index, coefficient in row.fixed_terms)])
        scale = row.slopes.abs().max().item()
        if scale == 0:
            # A finite-only constraint either admits this whole subspace or none of it.
            if (row.equality and abs(rhs) > _PHYSICAL_TOL) or (not row.equality and rhs > 0):
                return None
            continue
        target = equalities if row.equality else inequalities
        rhs /= scale
        if not math.isfinite(rhs):
            raise ValueError("Linear constraint has a non-finite model-coordinate bound")
        target.append((row.slopes / scale, rhs))
        if row.equality:
            tolerances.append(_PHYSICAL_TOL / scale)
    return equalities, inequalities, tolerances


def _dense(rows, sign=1):
    if rows:
        coefficients, rhs = zip(*rows)
        return (sign * torch.stack(coefficients),
                sign * torch.tensor(rhs, dtype=torch.double).unsqueeze(-1))
    return None


def _sparse(rows, continuous_indices):
    result = []
    for coefficients, rhs in rows:
        nonzero = coefficients.nonzero().flatten()
        result.append((continuous_indices[nonzero], coefficients[nonzero], rhs))
    return tuple(result)


def _solve_lp(objective, **kwargs):
    result = linprog(objective, method="highs",
                     options={"primal_feasibility_tolerance": 1e-10,
                              "dual_feasibility_tolerance": 1e-10}, **kwargs)
    if result.status == 2:
        return None
    if not result.success or not np.isfinite(result.x).all() or not math.isfinite(result.fun):
        raise ValueError(f"Linear feasibility LP failed: {result.message}")
    return result


def _independent_equalities(rows):
    if len(rows) < 2:
        return rows
    coefficients, rhs = _dense(rows)
    u, singular, vh = torch.linalg.svd(coefficients, full_matrices=False)
    rtol = max(coefficients.shape) * torch.finfo(coefficients.dtype).eps
    rank = int((singular > singular[0] * rtol).sum())
    if rank == len(rows):
        return rows
    _, _, pivots = qr(coefficients.numpy().T, pivoting=True, mode="economic")
    independent = pivots[:rank]
    reduced_rhs = u[:, :rank].T @ rhs
    residual = rhs - u[:, :rank] @ reduced_rhs
    roundoff = 8 * rtol * max(1., rhs.abs().max().item())
    if residual.abs().max() > roundoff:
        return None
    # Preserve exact input coefficients/bounds instead of rotating tight faces
    # into SVD coordinates that can move a strict inequality by a few ulps.
    return [rows[index] for index in independent]


def _polytope_rows(inequalities, bounds):
    dimensions = bounds.shape[1]
    eye = torch.eye(dimensions, dtype=bounds.dtype)
    return list(inequalities) + list(zip(eye, bounds[0].tolist())) + list(zip(-eye, (-bounds[1]).tolist()))


def _lp_constraints(equalities, rows):
    matrix, rhs = _dense(rows, sign=-1)
    eq = _dense(equalities)
    return {"A_ub": matrix.numpy(), "b_ub": rhs.flatten().numpy(),
            "A_eq": None if eq is None else eq[0].numpy(),
            "b_eq": None if eq is None else eq[1].flatten().numpy(),
            "bounds": [(None, None)] * matrix.shape[1]}


def _consistent_slice(equalities, tolerances, inequalities, bounds):
    """Prefer exact equations; otherwise find a slice inside physical tolerances."""
    rows = _polytope_rows(inequalities, bounds)
    dimensions = bounds.shape[1]
    exact = _independent_equalities(equalities)
    if exact is not None:
        constraints = _lp_constraints(exact, rows)
        point = _solve_lp(np.zeros(dimensions), **constraints)
        # LP feasibility tolerances must not silently widen a hard boundary.
        roundoff = 64 * np.finfo(float).eps * dimensions
        if (point is not None
                and np.all(constraints["A_ub"] @ point.x <= constraints["b_ub"])
                and (constraints["A_eq"] is None
                     or np.all(abs(constraints["A_eq"] @ point.x - constraints["b_eq"]) <= roundoff))):
            return exact
    constraints = _lp_constraints([], rows)
    origin = _solve_lp(np.zeros(dimensions), **constraints)
    if origin is None:
        return None
    coefficients, rhs = _dense(equalities)
    tolerance = torch.tensor(tolerances, dtype=rhs.dtype)
    center = torch.from_numpy(origin.x)
    matrix = coefficients / tolerance[:, None]
    target = (rhs.flatten() - coefficients @ center) / tolerance
    hard_matrix = torch.from_numpy(constraints["A_ub"])
    hard_rhs = torch.from_numpy(constraints["b_ub"]) - hard_matrix @ center
    # HiGHS minimizes the largest equality residual / physical tolerance, while
    # all inequalities and bounds remain unchanged. Displacements avoid large RHS.
    augmented = torch.cat([
        torch.cat([matrix, -torch.ones(len(equalities), 1, dtype=rhs.dtype)], dim=1),
        torch.cat([-matrix, -torch.ones(len(equalities), 1, dtype=rhs.dtype)], dim=1),
        torch.cat([hard_matrix, torch.zeros(len(rows), 1, dtype=rhs.dtype)], dim=1),
    ])
    solution = _solve_lp([0.] * dimensions + [1.], A_ub=augmented.numpy(),
                         b_ub=torch.cat([target, -target, hard_rhs]).numpy(),
                         bounds=list(zip((bounds[0] - center).tolist(),
                                         (bounds[1] - center).tolist())) + [(0., None)])
    if solution is None or solution.fun > 1.:
        return None
    point = center + torch.from_numpy(solution.x[:dimensions])
    adjusted_rhs = coefficients @ point
    if ((adjusted_rhs - rhs.flatten()).abs() > tolerance).any():
        raise ValueError("Equality consistency exceeds physical tolerance after LP")
    if (hard_matrix @ point > torch.from_numpy(constraints["b_ub"])).any():
        raise ValueError("Equality consistency LP violated a hard inequality or bound")
    return _independent_equalities(list(zip(coefficients, adjusted_rhs.tolist())))


def _affine_hull(equalities, inequalities, bounds):
    """Certify universally tight rows by maximizing their slack over the polytope."""
    dimensions = bounds.shape[1]
    rows = _polytope_rows(inequalities, bounds)
    kwargs = _lp_constraints(equalities, rows)
    feasible = _solve_lp(np.zeros(dimensions), **kwargs)
    if feasible is None:
        return None
    augmented = list(equalities)
    for row, bound in rows:
        # A single LP vertex being active is not evidence of a tight face.
        maximum = _solve_lp(-row.numpy(), **kwargs)
        if maximum is None:
            raise ValueError("Feasible polytope became infeasible while certifying its affine hull")
        max_slack = -maximum.fun - bound
        roundoff = 64 * torch.finfo(bounds.dtype).eps * dimensions * max(1., abs(bound))
        if abs(max_slack) <= roundoff:
            augmented.append((row, bound))
    reduced = _independent_equalities(augmented)
    if reduced is None:
        raise ValueError("Certified affine hull has inconsistent equalities")
    return reduced


def _sampler_type():
    # Keep ordinary imports and catalog search usable on the declared old floor.
    from botorch.utils.sampling import HitAndRunPolytopeSampler
    if not {"seed", "n_thinning"} <= set(inspect.signature(HitAndRunPolytopeSampler).parameters):
        raise EngineError(ErrorCode.ACQUISITION_INCOMPATIBLE,
            "Installed BoTorch lacks the seeded polytope sampler API required for equality search; "
            "provide a feasible candidate catalog with candidates= instead")
    return HitAndRunPolytopeSampler


def _fixed_hull_coordinates(equalities, continuous):
    """Eliminate singleton implications of the already certified affine hull."""
    fixed = {}
    while True:
        previous = len(fixed)
        for coefficients, rhs in equalities:
            terms = [(i, value) for i, value in enumerate(coefficients.tolist()) if value]
            remaining = [(i, value) for i, value in terms if i not in fixed]
            if len(remaining) == 1:
                index, coefficient = remaining[0]
                value = (rhs - math.fsum(c * fixed[i] for i, c in terms if i in fixed)) / coefficient
                if not math.isfinite(value):
                    raise ValueError("Certified fixed coordinate is non-finite")
                fixed[index] = value
        if len(fixed) == previous:
            return {continuous[index]: value for index, value in fixed.items()}


def linear_seed_pool(space, bounds, continuous, combinations, rows, samples, seed):
    """Sample free coordinates only, then restore fixed model coordinates."""
    continuous_indices = torch.tensor(continuous, dtype=torch.long)
    free_bounds = bounds[:, continuous_indices]
    sampler_type = _sampler_type()
    frames, subspaces = [], []
    feasible_count = 0
    for fixed in combinations:
        reduced = _substituted_rows(space, rows, fixed)
        if reduced is None:
            continue
        equalities, inequalities, tolerances = reduced
        equalities = _consistent_slice(equalities, tolerances, inequalities, free_bounds)
        if equalities is None:
            continue
        equalities = _affine_hull(equalities, inequalities, free_bounds)
        if equalities is None:
            continue
        # Infeasibility has been checked by the LP. Any sampler error is numerical.
        sampler = sampler_type(
            bounds=free_bounds, equality_constraints=_dense(equalities),
            inequality_constraints=_dense(inequalities, sign=-1),
            n_burnin=32, n_thinning=4, seed=seed,
        )
        feasible_count += 1
        if sampler.nullC.shape[1] == 0:
            # Continuous bounds are already [0, 1]; the LP point is the entire subspace.
            points = sampler.x0.T
        else:
            points = sampler.draw(n=samples)
        if (not torch.isfinite(points).all() or (points < free_bounds[0] - _BOUNDS_ROUNDOFF).any()
                or (points > free_bounds[1] + _BOUNDS_ROUNDOFF).any()):
            raise ValueError("Linear constraint sampler returned invalid model coordinates")
        points = points.clamp(free_bounds[0], free_bounds[1])
        coordinates = torch.empty((len(points), space.n_dims), dtype=bounds.dtype)
        coordinates[:, continuous_indices] = points
        for index, value in fixed.items():
            coordinates[:, index] = value
        frames.append(space.model_to_physical(coordinates))
        if sampler.nullC.shape[1]:
            # Tight bounds plus equalities can leave SLSQP with degenerate active
            # constraints. BoTorch removes fixed features and substitutes them in
            # the original constraints; do not leave certified constant axes free.
            optimizer_fixed = {**fixed, **_fixed_hull_coordinates(equalities, continuous)}
            subspaces.append(ContinuousSubspace(optimizer_fixed, _sparse(equalities, continuous_indices),
                                               _sparse(inequalities, continuous_indices), coordinates))

    pool = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=space.param_names)
    return pool, tuple(subspaces), feasible_count


def generate_linear_candidates(*args, **kwargs):
    # optimize_acqf captures warnings from its generator even when retry is off.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", OptimizationWarning)
        result = gen_candidates_scipy(*args, **kwargs)
    for item in caught:
        if issubclass(item.category, OptimizationWarning):
            raise item.message
        warnings.warn_explicit(item.message, item.category, item.filename, item.lineno)
    return result
