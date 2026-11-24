"""Versioned input-only locations for soft monotone directional evidence.

Endpoints are ordered (minus, plus) in Space model coordinates, before any
kernel feature expansion. They need not belong to a recommendation catalog;
parameter feasibility does not imply outcome safety. No responses are created.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import torch
from torch import Tensor

from ..domain import ObservationBatch, Space
from ..errors import EngineError, ErrorCode
from ..knowledge.artifacts import OptimizationArtifacts
from .knowledge_runtime import check_directional_composition


@dataclass(frozen=True, init=False)
class DirectionalPairs:
    """Detached evidence tensors with per-pair provenance and snapshot metadata."""

    endpoints: Tensor
    signs: Tensor
    widths: Tensor
    softness: Tensor
    weights: Tensor
    pattern_ids: tuple[str, ...]
    _diagnostics: list[dict] = field(repr=False)

    def __init__(self, endpoints, signs, widths, softness, weights, pattern_ids, diagnostics):
        for name, value in (("endpoints", endpoints), ("signs", signs),
                            ("widths", widths), ("softness", softness), ("weights", weights)):
            object.__setattr__(self, name, value.detach().clone())
        object.__setattr__(self, "pattern_ids", tuple(pattern_ids))
        object.__setattr__(self, "_diagnostics", deepcopy(diagnostics))

    @property
    def diagnostics(self) -> list[dict]:
        return deepcopy(self._diagnostics)


def _invalid(message, **details):
    return EngineError(ErrorCode.CONFIG_INVALID, message, details=details)


def _roundtrip_pairs(space, centers, dim, epsilon, parameter_constraints):
    endpoints = centers[:, None, :].expand(-1, 2, -1).clone()
    low, high = space.model_bounds_tensor[:, dim]
    endpoints[:, 0, dim] = (centers[:, dim] - epsilon).clamp(low, high)
    endpoints[:, 1, dim] = (centers[:, dim] + epsilon).clamp(low, high)
    physical = space.model_to_physical(endpoints.reshape(-1, space.n_dims))
    actual = space.physical_to_model(physical).reshape(-1, 2, space.n_dims)
    feasible = space.feasibility_mask(physical)
    hard = [constraint for constraint in parameter_constraints if constraint.hard]
    if hard:
        rows = physical.to_dict(orient="records")
        feasible &= torch.tensor([all(constraint.satisfied(row) for constraint in hard)
                                  for row in rows], dtype=torch.bool)
    unchanged = actual[:, 0, :] == actual[:, 1, :]
    unchanged[:, dim] = True
    valid = (feasible.reshape(-1, 2).all(dim=1) & unchanged.all(dim=1)
             & (actual[:, 1, dim] > actual[:, 0, dim]))
    return actual, valid


def build_directional_pairs(
    space: Space,
    observations: ObservationBatch,
    artifacts: OptimizationArtifacts,
    *,
    seed: int,
    parameter_constraints=(),
) -> DirectionalPairs:
    """Build exact requested counts, or fail closed with pattern provenance.

    ``artifacts`` is one validated, projected output. ``parameter_constraints``
    must include the full runtime's global parameter constraints, not merely
    that output's projection. Only successful observed X rows are accepted.

    Generator v2.0 tries observed rows first, then at most 4096 scrambled Sobol
    centers per active spec. Each spec restarts a private engine with ``seed``;
    inactive declarations cannot shift any active spec's stream. Sobol draws
    are in blocks of 64, in double precision, over the full model bounds.
    """
    if not isinstance(space, Space) or not isinstance(observations, ObservationBatch):
        raise TypeError("space and observations must be Space and ObservationBatch")
    if not isinstance(artifacts, OptimizationArtifacts):
        raise TypeError("artifacts must be OptimizationArtifacts")
    if type(seed) is not int or seed < 0:
        raise _invalid("seed must be a non-negative integer")
    if space.n_objectives != 1:
        raise _invalid("Directional pairs require a projected scalar Space")
    status = observations.status
    if status is not None and not status.eq("success").all():
        raise _invalid("Directional pairs require successful observations")
    parameter_constraints = tuple(parameter_constraints)
    if any(not callable(getattr(item, "satisfied", None)) or
           type(getattr(item, "hard", None)) is not bool for item in parameter_constraints):
        raise _invalid("parameter_constraints must be executable parameter constraints")
    check_directional_composition(artifacts)
    directional = sorted((artifact for artifact in artifacts.virtual_observations
                          if artifact.kind == "monotone_sign_v2"),
                         key=lambda artifact: artifact.source_pattern_id)
    for artifact in directional:
        dim = artifact.payload["dim"]
        if (dim >= space.n_dims or space.params[dim].kind not in {"continuous", "integer", "discrete"}
                or artifact.source_pattern != "monotone" or artifact.source_version != "2.0"):
            raise _invalid("Invalid monotone v2 factor or provenance",
                           pattern_id=artifact.source_pattern_id)

    endpoints, signs, widths, softness, weights, pattern_ids, diagnostics = [], [], [], [], [], [], []
    observed = None
    for artifact in directional:
        payload = artifact.payload
        dim, requested = payload["dim"], payload["n_pairs_per_dim"]
        confidence = payload["confidence"]
        diagnostic = {
            "pattern_id": artifact.source_pattern_id, "generator_version": "2.0",
            "dim": dim, "requested": requested, "accepted": 0,
            "status": "inactive" if confidence == 0 else "active",
            "observed_centers_attempted": 0, "generated_centers": 0,
            "generated_centers_attempted": 0,
            "location_domain": "declared_parameter_domain",
            "catalog_membership_required": False, "outcome_safety_guaranteed": False,
        }
        diagnostics.append(diagnostic)
        if confidence == 0:
            continue
        if observed is None:
            observed = space.physical_to_model(observations.X)
        accepted, seen = [], set()

        def accept(centers, source):
            if not len(centers):
                return
            actual, valid = _roundtrip_pairs(space, centers, dim, payload["epsilon"],
                                             parameter_constraints)
            for pair, keep in zip(actual, valid):
                diagnostic[source] += 1
                if not keep:
                    continue
                key = tuple(pair.flatten().tolist())
                if key in seen:
                    continue
                seen.add(key)
                accepted.append(pair)
                if len(accepted) == requested:
                    break

        accept(observed, "observed_centers_attempted")
        if len(accepted) < requested:
            engine = torch.quasirandom.SobolEngine(space.n_dims, scramble=True, seed=seed)
            low, high = space.model_bounds_tensor
            while len(accepted) < requested and diagnostic["generated_centers"] < 4096:
                centers = low + (high - low) * engine.draw(64, dtype=torch.double)
                diagnostic["generated_centers"] += len(centers)
                accept(centers, "generated_centers_attempted")
        diagnostic["accepted"] = len(accepted)
        if len(accepted) != requested:
            raise _invalid("Cannot generate the requested feasible directional pairs",
                           **diagnostic, reason="no_feasible_directional_pairs")
        for pair in accepted:
            endpoints.append(pair)
            signs.append(1. if payload["direction"] == "increasing" else -1.)
            widths.append(pair[1, dim] - pair[0, dim])
            softness.append(payload["softness"])
            weights.append(confidence / requested)
            pattern_ids.append(artifact.source_pattern_id)
    return DirectionalPairs(
        torch.stack(endpoints) if endpoints else torch.empty((0, 2, space.n_dims), dtype=torch.double),
        torch.tensor(signs, dtype=torch.double), torch.tensor(widths, dtype=torch.double),
        torch.tensor(softness, dtype=torch.double), torch.tensor(weights, dtype=torch.double),
        tuple(pattern_ids), diagnostics,
    )


__all__ = ["DirectionalPairs", "build_directional_pairs"]
