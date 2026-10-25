"""Fail-closed routing from registry declarations to numerical consumers.

Model consumers receive model-frame inputs (Space performs linear/log input
encoding). Physical response means are normalized before fitting; posterior
outcomes must be inverted to physical units before outcome constraints, then
converted to objective utilities before acquisition preferences. Safety
constraints are independent of model choice and must also be passed to M0.
No provider loading or fitted model is exposed here.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import math

import numpy as np

from expdoe_dk.domain import ObservationBatch, Space, constraint_from_dict
from expdoe_dk.errors import EngineError, ErrorCode
from expdoe_dk.knowledge import Knowledge
from expdoe_dk.knowledge.artifacts import ARTIFACT_CATEGORIES, OptimizationArtifact


def _invalid(artifact: OptimizationArtifact, reason: str) -> EngineError:
    return EngineError(ErrorCode.KNOWLEDGE_INVALID, reason, details={
        "pattern_id": artifact.source_pattern_id, "pattern": artifact.source_pattern,
        "kind": artifact.kind, "reason": reason,
    })


def validate_runtime_artifact(space: Space, category: str, artifact: OptimizationArtifact) -> None:
    """Validate numerical preconditions beyond a registry's JSON schema."""
    from .structured_coverage import STRUCTURED_KINDS
    if artifact.kind in STRUCTURED_KINDS:
        from .structured_mean import _validate
        _validate(space, artifact)
        if category != "mean_components":
            raise _invalid(artifact, "Structured means require mean_components")
        return
    payload = artifact.payload
    try:
        def finite(value, *, positive=False):
            if type(value) not in (float, int) or not math.isfinite(value) or (positive and value <= 0):
                raise ValueError("Expected a finite" + (" positive" if positive else "") + " number")

        def dimension(value):
            if type(value) is not int or not 0 <= value < space.n_dims:
                raise ValueError("Artifact dimension is outside the space")

        if artifact.kind in {"power_law_descriptor", "exponential_descriptor", "quadratic_valley_descriptor", "periodic_descriptor"}:
            dimension(payload["dimension"])
            factor = space.param_by_name(payload["factor"])
            if space.param_names[payload["dimension"]] != factor.name or factor.kind not in {"continuous", "integer", "discrete"}:
                raise ValueError("Mean descriptor requires its declared numeric factor")
            names = {"power_law_descriptor": ("exponent", "scale"),
                     "exponential_descriptor": ("rate", "amplitude"),
                     "quadratic_valley_descriptor": ("center", "width"),
                     "periodic_descriptor": ("phase", "period")}[artifact.kind]
            for name in names:
                finite(payload["parameters"][name], positive=name in {"scale", "amplitude", "width", "period"})
        elif artifact.kind == "gp_prior":
            for name in ("lengthscale", "outputscale", "noise"):
                if len(payload[name]) != 2:
                    raise ValueError("Gamma prior requires concentration and rate")
                for value in payload[name]:
                    finite(value, positive=True)
        elif artifact.kind == "monotone":
            dimension(payload["dim"])
            if payload["direction"] not in {"increasing", "decreasing"}:
                raise ValueError("Unknown monotonic direction")
            if payload["epsilon"] != "auto":
                finite(payload["epsilon"], positive=True)
            finite(payload["delta_norm"], positive=True)
            if type(payload["n_pairs_per_dim"]) is not int or payload["n_pairs_per_dim"] <= 0:
                raise ValueError("Monotone pair count must be a positive integer")
        elif artifact.kind == "random_augment":
            if type(payload["n"]) is not int or payload["n"] <= 0:
                raise ValueError("Random augmentation count must be a positive integer")
        elif artifact.kind == "arrhenius_mean":
            dimension(payload["temp_dim_index"])
            finite(payload["activation_energy"])
            finite(payload["amplitude_init"])
        elif artifact.kind == "quadratic_mean":
            if payload["input_dim"] != space.n_dims:
                raise ValueError("Quadratic input dimension must match space")
            for name in ("curvature_signs", "centers"):
                if len(payload[name]) != space.n_dims:
                    raise ValueError("Quadratic parameters must match space dimension")
                for value in payload[name]:
                    finite(value)
        elif artifact.kind == "target_range_constraint":
            if payload["objective"] not in space.objectives or payload["hard"] is not False:
                raise ValueError("Target range requires a known objective and soft semantics")
            finite(payload["lower"])
            finite(payload["upper"])
            if payload["lower"] >= payload["upper"]:
                raise ValueError("Target range must be ordered")
        elif artifact.kind == "objective_priority_preference":
            if len(payload["objectives"]) != len(payload["weights"]) or not payload["weights"]:
                raise ValueError("Preference objectives and weights must match")
            for objective, weight in zip(payload["objectives"], payload["weights"], strict=True):
                if objective not in space.objectives:
                    raise ValueError("Unknown preference objective")
                finite(weight)
                if weight < 0:
                    raise ValueError("Preference weights cannot be negative")
            if not math.isclose(sum(payload["weights"]), 1.):
                raise ValueError("Preference weights must be normalized")
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise _invalid(artifact, f"Invalid {artifact.kind} payload: {error}") from error
    if artifact.kind == "category_similarity_descriptor" and category == "kernel_components":
        try:
            dimension(payload["dimension"])
            factor = space.param_by_name(payload["factor"])
            if (space.param_names[payload["dimension"]] != payload["factor"]
                    or factor.kind != "categorical"):
                raise ValueError("Similarity descriptor requires the dimension of its declared categorical factor")
            parameters = payload["parameters"]
            levels = parameters["levels"]
            matrix = np.asarray(parameters["matrix"], dtype=float)
            if (matrix.shape != (len(levels), len(levels)) or not np.isfinite(matrix).all()
                    or not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0)):
                raise ValueError("Similarity matrix must be finite, square and symmetric")
            if (matrix < 0).any() or (matrix > 1).any():
                raise ValueError("Similarities must be bounded in [0, 1]")
            if np.linalg.eigvalsh(matrix).min() < -1e-10:
                raise ValueError("Similarity matrix must be positive semidefinite")
            encoded = factor.encode(levels)
            if sorted(encoded) != list(range(len(factor.values or ()))):
                raise ValueError("Similarity matrix must specify every declared level; missing covariances are undefined")
        except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as error:
            raise _invalid(artifact, str(error)) from error
    if artifact.kind == "hard_parameter_constraint":
        try:
            constraint = constraint_from_dict(artifact.to_dict()["payload"]["constraint"], allowed_names=space.param_names)
            if not constraint.hard:
                raise ValueError("Hard safety artifact cannot contain a soft constraint")
        except (KeyError, TypeError, ValueError, EngineError) as error:
            raise _invalid(artifact, str(error)) from error
    if artifact.kind == "knowledge_validation" and payload.get("state") == "skipped":
        raise _invalid(artifact, "Enabled knowledge cannot be skipped by validation")


@dataclass(frozen=True)
class _Consumer:
    artifact: OptimizationArtifact

    @property
    def source_pattern_id(self):
        return self.artifact.source_pattern_id

    @property
    def source_pattern(self):
        return self.artifact.source_pattern

    @property
    def source_version(self):
        return self.artifact.source_version

    @property
    def kind(self):
        return self.artifact.kind

    @property
    def payload(self):
        return self.artifact.payload


@dataclass(frozen=True)
class RuntimeParameterConstraint(_Consumer):
    constraint: object

    @property
    def name(self):
        return self.constraint.name

    @property
    def hard(self):
        return self.constraint.hard

    @property
    def penalty(self):
        return self.constraint.penalty

    @property
    def weight(self):
        return self.constraint.weight

    def satisfied(self, values):
        return self.constraint.satisfied(values)

    def to_dict(self):
        return self.constraint.to_dict()


@dataclass(frozen=True)
class RuntimeOutcomeRange(_Consumer):
    """Physical outcome interval; soft preference, not a hard rejection rule."""

    @property
    def hard(self):
        return self.payload["hard"]

    def margin(self, values: Mapping[str, object]) -> float:
        value = float(values[self.payload["objective"]])
        if not math.isfinite(value):
            raise _invalid(self.artifact, "Outcome must be finite")
        return min(value - self.payload["lower"], self.payload["upper"] - value)

    def satisfied(self, values: Mapping[str, object]) -> bool:
        return self.margin(values) >= 0


@dataclass(frozen=True)
class RuntimePreference(_Consumer):
    """Weighted sum of already-oriented objective utilities, not raw outcomes."""

    def __call__(self, utilities: Mapping[str, float]) -> float:
        value = sum(float(utilities[name]) * weight for name, weight in
                    zip(self.payload["objectives"], self.payload["weights"], strict=True))
        if not math.isfinite(value):
            raise _invalid(self.artifact, "Objective utilities must be finite")
        return value


def _check_conflicts(artifacts) -> None:
    seen: dict[tuple, OptimizationArtifact] = {}
    for category in ("priors", "virtual_observations", "kernel_components"):
        for artifact in getattr(artifacts, category):
            if artifact.kind == "gp_prior":
                key, value = ("gp_prior",), artifact.payload
            elif artifact.kind == "monotone":
                key, value = ("monotone", artifact.payload["dim"]), artifact.payload["direction"]
            elif artifact.kind == "category_similarity_descriptor":
                key, value = ("similarity", artifact.payload["factor"]), artifact.payload["parameters"]
            else:
                continue
            previous = seen.get(key)
            previous_value = (previous.payload["direction"] if artifact.kind == "monotone"
                              else previous.payload["parameters"] if artifact.kind == "category_similarity_descriptor"
                              else previous.payload) if previous else None
            if previous is not None and previous_value != value:
                raise EngineError(ErrorCode.KNOWLEDGE_CONFLICT, "Conflicting runtime knowledge", details={
                    "pattern_ids": [previous.source_pattern_id, artifact.source_pattern_id],
                    "kind": artifact.kind,
                })
            seen[key] = artifact


def compile_runtime(space: Space, knowledge: Knowledge,
                    observations: ObservationBatch | None) -> dict[str, tuple]:
    """Return every supported artifact, with executable declarative consumers.

    Unknown kinds/categories and insufficiently specified descriptors fail
    closed. Insufficient empirical coverage never removes hard safety rules.
    Registry artifacts retain provenance; wrappers expose their original
    ``artifact`` plus the interface needed by a downstream numerical consumer.
    """
    from .capabilities import inspect_capabilities

    if not isinstance(space, Space) or not isinstance(knowledge, Knowledge):
        raise TypeError("space and knowledge must be Space and Knowledge")
    if observations is not None and not isinstance(observations, ObservationBatch):
        raise TypeError("observations must be ObservationBatch or None")
    # Compile the whole declaration set first so certified conflicts preserve
    # their KNOWLEDGE_CONFLICT code, rather than being reduced to an inventory.
    artifacts = knowledge.compile(space, observations)
    from .structured_coverage import check_structured_composition
    check_structured_composition(space, artifacts)
    report = inspect_capabilities(space, knowledge)
    if report.blocked:
        raise EngineError(ErrorCode.KNOWLEDGE_INVALID, "Knowledge has no supported runtime route",
                          details=report.to_dict())
    from .capabilities import ARTIFACT_CONSUMERS

    runtime = {}
    for category in ARTIFACT_CATEGORIES:
        routed = []
        for artifact in getattr(artifacts, category):
            if (category, artifact.kind) not in ARTIFACT_CONSUMERS:
                raise _invalid(artifact, f"Unsupported artifact category {category}")
            validate_runtime_artifact(space, category, artifact)
            if category == "parameter_constraints":
                constraint = constraint_from_dict(artifact.to_dict()["payload"]["constraint"],
                                                  allowed_names=space.param_names)
                routed.append(RuntimeParameterConstraint(artifact, constraint))
            elif category == "outcome_constraints":
                routed.append(RuntimeOutcomeRange(artifact))
            elif category == "acquisition_preferences":
                routed.append(RuntimePreference(artifact))
            else:
                routed.append(artifact)
        runtime[category] = tuple(routed)
    _check_conflicts(artifacts)
    return runtime


__all__ = ["compile_runtime"]
