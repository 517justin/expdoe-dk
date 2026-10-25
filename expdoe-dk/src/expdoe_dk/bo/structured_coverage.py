"""Minimum physical-data coverage and whole-set structured mean composition.

These guards do not fit a model, test agreement, or certify identifiability.
They intentionally do not activate any numerical consumer.
"""
from __future__ import annotations

import math

import numpy as np

from ..domain import ObservationBatch, Space
from ..errors import EngineError, ErrorCode
from ..knowledge.artifacts import ARTIFACT_CATEGORIES, OptimizationArtifact, OptimizationArtifacts
from .structured_mean import _validate


STRUCTURED_KINDS = frozenset({"saturation_mean_v2", "threshold_mean_v2", "signed_interaction_mean_v2"})
_SIGNS = {"increasing": 1., "decreasing": -1., "flat": 0.}
_CUTOFF = 1e-10


def check_structured_composition(space: Space, artifacts: OptimizationArtifacts) -> None:
    """Validate the entire active declaration set, independently of data."""
    structured = []
    legacy = []
    for category in ARTIFACT_CATEGORIES:
        for artifact in getattr(artifacts, category):
            if artifact.kind in STRUCTURED_KINDS:
                _validate(space, artifact)
                if category != "mean_components":
                    raise EngineError(ErrorCode.KNOWLEDGE_INVALID,
                        "Structured means require the mean_components category", details={
                            "pattern_ids": [artifact.source_pattern_id], "category": category})
                structured.append(artifact)
            elif category in {"mean_components", "virtual_observations"}:
                legacy.append(artifact)
    if not structured:
        return
    seen = {}
    for artifact in structured:
        payload = artifact.payload
        factors = tuple(payload["factors"])
        key = (tuple(payload["objectives"]), factors)
        previous = seen.get(key)
        if previous is not None:
            raise EngineError(ErrorCode.KNOWLEDGE_CONFLICT,
                "Duplicate or opposing structured declarations", details={
                    "pattern_ids": [previous.source_pattern_id, artifact.source_pattern_id],
                    "factors": list(factors), "objectives": list(payload["objectives"]),
                    "remediation": "Retain one shape per factor/response and one interaction per unordered pair/response."})
        seen[key] = artifact
    if legacy:
        raise EngineError(ErrorCode.KNOWLEDGE_INVALID,
            "Legacy shape means or virtual observations cannot be composed with v2 means", details={
                "reason": "unsupported_composition",
                "pattern_ids": list(dict.fromkeys(a.source_pattern_id for a in structured + legacy)),
                "remediation": "Disable legacy shape/virtual declarations or evaluate them in a separate legacy-only model."})


def _svd(columns, names):
    matrix = np.column_stack(columns).astype(np.float64, copy=False)
    singular = np.linalg.svd(matrix, compute_uv=False)
    tolerance = float(singular[0] * _CUTOFF) if singular.size else 0.
    rank = int(np.count_nonzero(singular > tolerance))
    # Name columns that are dependent on preceding columns at the same cutoff.
    deficient = []
    previous_rank = 0
    for index, name in enumerate(names):
        values = np.linalg.svd(matrix[:, :index + 1], compute_uv=False)
        current_rank = int(np.count_nonzero(values > tolerance))
        if current_rank <= previous_rank:
            deficient.append(name)
        previous_rank = current_rank
    return {"rank": rank, "n_columns": len(names), "columns": list(names),
            "singular_values": singular.tolist(), "tolerance": tolerance,
            "deficient_columns": deficient, "status": "passed" if rank == len(names) else "deficient"}


def _basis(artifacts, coordinates, n_rows, width):
    columns, names = [np.ones(n_rows, dtype=np.float64)], ["intercept"]
    shaped = set()
    interacting = set()
    interactions = []
    for artifact in artifacts:
        payload = artifact.payload
        factors, parameters = payload["factors"], payload["parameters"]
        identity = artifact.source_pattern_id
        if artifact.kind == "signed_interaction_mean_v2":
            interacting.update(factors)
            sign = (1. if artifact.source_pattern == "synergy" else -1.) * (1. if parameters["response_effect"] == "higher" else -1.)
            interactions.append((sign * coordinates[factors[0]] * coordinates[factors[1]], f"{identity}:interaction"))
            continue
        factor = factors[0]
        shaped.add(factor)
        u = coordinates[factor]
        coordinate = payload["coordinates"][0]
        if artifact.kind == "saturation_mean_v2":
            k = (parameters["half_response"] - parameters["origin"]) / coordinate["range"]
            columns.append(_SIGNS[parameters["direction"]] * u / (k + u))
            names.append(f"{identity}:amplitude")
        else:
            h = (parameters["threshold"] - coordinate["lower"]) / coordinate["range"]
            H = width * (np.logaddexp(0., (u - h) / width) - np.logaddexp(0., -h / width))
            for branch, values in (("below", u - H), ("above", H)):
                sign = _SIGNS[parameters[f"{branch}_behavior"]]
                if sign:  # An exact flat branch has no fitted slope parameter.
                    columns.append(sign * values)
                    names.append(f"{identity}:{branch}")
    for factor in coordinates:  # Canonical Space order; nuisance main effects once.
        if factor in interacting - shaped:
            columns.append(coordinates[factor])
            names.append(f"{factor}:linear")
    for values, name in interactions:
        columns.append(values)
        names.append(name)
    return columns, names


def check_structured_coverage(
    space: Space, observations: ObservationBatch | None,
    artifacts: tuple[OptimizationArtifact, ...], *,
    widths: tuple[float, ...] = (0.02, 0.05, 0.15),
) -> dict:
    """Check successful physical rows and full initial mean designs via float64 SVD."""
    check_structured_composition(space, OptimizationArtifacts(mean_components=artifacts))
    structured = tuple(a for a in artifacts if a.kind in STRUCTURED_KINDS)
    result = {"coverage_status": "not_evaluated" if structured else "not_applicable",
        "empirical_evidence": "not_evaluated", "parameter_identifiability": "not_certified",
        "pattern_ids": [a.source_pattern_id for a in structured], "eligible_starts": [],
        "coverage": [], "starts": [], "svd_relative_cutoff": _CUTOFF}
    if observations is None or not structured:
        return result
    if not widths or any(type(w) not in (int, float) or not math.isfinite(w) or not .01 <= w <= .25 for w in widths):
        raise EngineError(ErrorCode.CONFIG_INVALID, "Structured coverage widths must be in [.01, .25]")
    successful = observations.successful()
    X = successful.X
    n_rows = len(X)
    result["successful_rows"] = n_rows

    def insufficient(message, remediation):
        result["coverage_status"] = "deficient"
        result["remediation"] = remediation
        raise EngineError(ErrorCode.INSUFFICIENT_DATA, message, details=result)

    if n_rows < 6:
        insufficient("Structured means require at least six successful rows",
                     "Collect at least six successful observations; warning/failed rows do not count.")
    required_responses = {objective for a in structured for objective in a.payload["objectives"]}
    if not required_responses <= set(successful.Y.columns):
        raise EngineError(ErrorCode.KNOWLEDGE_INVALID, "Observed response does not match the structured declarations",
            details={"pattern_ids": result["pattern_ids"], "objectives": sorted(required_responses)})
    required_factors = {factor for a in structured for factor in a.payload["factors"]}
    coordinates = {}
    for factor in space.param_names:
        if factor not in required_factors:
            continue
        source = next(a for a in structured if factor in a.payload["factors"])
        index = list(source.payload["factors"]).index(factor)
        coordinate = source.payload["coordinates"][index]
        # Subtract exact Python integer anchors before float conversion, including
        # adjacent large integers. Physical values also avoid log-coordinate rank changes.
        coordinates[factor] = np.asarray([(x - coordinate["lower"]) / coordinate["range"]
                                         for x in X[factor].tolist()], dtype=np.float64)
    for artifact in structured:
        payload = artifact.payload
        diagnostic = {"pattern_id": artifact.source_pattern_id, "factors": list(payload["factors"])}
        if artifact.kind == "signed_interaction_mean_v2":
            first, second = payload["factors"]
            u, v = coordinates[first], coordinates[second]
            levels = {factor: int(X[factor].nunique()) for factor in (first, second)}
            diagnostic.update(_svd([np.ones(n_rows), u, v, u*v], ["intercept", first, second, f"{first}*{second}"]))
            diagnostic["distinct_levels"] = levels
            if min(levels.values()) < 2:
                diagnostic["status"] = "deficient"
        else:
            factor = payload["factors"][0]
            transition = payload["parameters"]["half_response" if artifact.kind == "saturation_mean_v2" else "threshold"]
            values = X[factor]
            left, right = int(values[values < transition].nunique()), int(values[values > transition].nunique())
            diagnostic.update(left_distinct=left, right_distinct=right,
                              status="passed" if min(left, right) >= 2 else "deficient")
        result["coverage"].append(diagnostic)
    if any(c["status"] == "deficient" for c in result["coverage"]):
        insufficient("Structured factor/interaction coverage is deficient",
            "Collect two distinct values strictly on each side of each transition, and independent joint interaction combinations with two levels per factor.")
    for index, width in enumerate(widths):
        columns, names = _basis(structured, coordinates, n_rows, width)
        start = {"index": index, "width": float(width), **_svd(columns, names)}
        result["starts"].append(start)
        if start["status"] == "passed":
            result["eligible_starts"].append(index)
    if not result["eligible_starts"]:
        result["deficient_columns"] = list(dict.fromkeys(name for start in result["starts"] for name in start["deficient_columns"]))
        insufficient("Complete structured mean basis is rank deficient at every start",
            "Collect independent joint factor combinations to separate the named columns, or simplify redundant declarations.")
    result["coverage_status"] = "passed"
    return result


__all__ = ["check_structured_composition", "check_structured_coverage"]
