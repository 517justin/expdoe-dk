"""Separate full-space safety/preferences from named physical-response models."""
from __future__ import annotations

from dataclasses import dataclass

from expdoe_dk.domain import ObservationBatch, Space
from expdoe_dk.errors import EngineError, ErrorCode
from expdoe_dk.knowledge import Knowledge
from expdoe_dk.knowledge.artifacts import ARTIFACT_CATEGORIES, OptimizationArtifacts
from expdoe_dk.knowledge.specs import KnowledgePatternSpec


_GLOBAL_PATTERNS = frozenset({"safe_region", "forbidden_region", "target_range",
                              "objective_priority", "tradeoff"})
_MODEL_CATEGORIES = frozenset({"mean_components", "kernel_components", "priors",
                               "virtual_observations", "diagnostics"})
_GLOBAL_CATEGORIES = frozenset({"parameter_constraints", "outcome_constraints",
                                "acquisition_preferences", "diagnostics"})


@dataclass(frozen=True)
class _ObjectiveRoute:
    objective_name: str | None
    space: Space
    observations: ObservationBatch | None
    knowledge: Knowledge


def _subset(knowledge, specs):
    # The public envelope carries restored provider requirements as well as
    # loaded registry provenance; constructing Knowledge(registry) alone loses it.
    envelope = knowledge.to_envelope()
    envelope["specs"] = [spec.to_dict() for spec in specs]
    return Knowledge.from_envelope(envelope, registry=knowledge.registry)


def _scope_error(spec, reason):
    return EngineError(ErrorCode.KNOWLEDGE_INVALID, "Unsupported objective knowledge scope", details={
        "pattern_id": spec.pattern_id, "pattern": spec.pattern, "version": spec.version,
        "objective_name": spec.scope.objectives[0] if len(spec.scope.objectives) == 1 else None,
        "scope": spec.scope.to_dict(), "reason": reason,
    })


def _knowledge_routes(space, observations, knowledge):
    global_specs = []
    model_specs = {name: [] for name in space.objectives}
    for spec in knowledge.specs:
        if not spec.enabled:
            continue
        scope = spec.scope
        if set(scope.objectives) - set(space.objectives):
            raise _scope_error(spec, "unknown_objective_scope")
        if scope.conditions is not None or scope.region is not None:
            raise _scope_error(spec, "unsupported_conditional_scope")
        if set(scope.factors) - set(space.param_names):
            raise _scope_error(spec, "unknown_factor_scope")
        if spec.pattern in _GLOBAL_PATTERNS:
            if spec.pattern in {"safe_region", "forbidden_region"} and scope.objectives:
                raise _scope_error(spec, "parameter_safety_requires_global_scope")
            if spec.pattern in {"target_range", "objective_priority", "tradeoff"} and scope.factors:
                raise _scope_error(spec, "preference_factor_scope_unsupported")
            global_specs.append(spec)
        elif spec.pattern == "category_similarity" and not scope.objectives:
            for specs in model_specs.values():
                specs.append(spec)
        else:
            if not scope.objectives:
                raise _scope_error(spec, "missing_objective_scope")
            for name in scope.objectives:
                payload = spec.to_dict()
                payload["scope"]["objectives"] = [name]
                model_specs[name].append(KnowledgePatternSpec.from_dict(payload))

    routes = [_ObjectiveRoute(None, space, observations, _subset(knowledge, global_specs))]
    if observations is not None and set(observations.Y.columns) != set(space.objectives):
        raise EngineError(ErrorCode.CONFIG_INVALID, "Observation objectives must exactly match Space", details={
            "objective_names": list(space.objectives), "observed_names": list(observations.Y.columns)})
    for objective in space.objective_specs:
        projected = Space(space.params, constraints=space.constraints, objectives=(objective,))
        batch = None
        if observations is not None:
            batch = observations._subset(objectives=[objective.name])
        routes.append(_ObjectiveRoute(objective.name, projected, batch,
                                      _subset(knowledge, model_specs[objective.name])))
    return tuple(routes)


def _compile_route(route: _ObjectiveRoute) -> OptimizationArtifacts:
    from .capabilities import ARTIFACT_CONSUMERS
    from .knowledge_runtime import _check_conflicts, validate_runtime_artifact
    from .structured_coverage import check_structured_composition

    try:
        artifacts = route.knowledge.compile(route.space, route.observations)
        check_structured_composition(route.space, artifacts)
        allowed = _GLOBAL_CATEGORIES if route.objective_name is None else _MODEL_CATEGORIES
        specs = {spec.pattern_id: spec for spec in route.knowledge.specs}
        numerical_sources = set()
        for category in ARTIFACT_CATEGORIES:
            for artifact in getattr(artifacts, category):
                if category not in allowed or (category, artifact.kind) not in ARTIFACT_CONSUMERS:
                    raise EngineError(ErrorCode.KNOWLEDGE_INVALID, "Unsupported objective artifact route", details={
                        "pattern_id": artifact.source_pattern_id, "kind": artifact.kind,
                        "category": category, "reason": "unsupported_artifact"})
                spec = specs[artifact.source_pattern_id]
                if artifact.kind in {"gp_prior", "random_augment"} and spec.scope.factors:
                    raise _scope_error(spec, "unsupported_factor_scope")
                validate_runtime_artifact(route.space, category, artifact)
                if category != "diagnostics":
                    numerical_sources.add(artifact.source_pattern_id)
        missing = [spec.pattern_id for spec in route.knowledge.specs
                   if spec.enabled and spec.pattern_id not in numerical_sources]
        if missing:
            raise EngineError(ErrorCode.KNOWLEDGE_INVALID, "Knowledge has no numerical artifact", details={
                "pattern_ids": missing, "reason": "no_numerical_artifact"})
        _check_conflicts(artifacts, multiobjective=route.objective_name is None)
        return artifacts
    except EngineError as error:
        details = dict(error.details)
        details["objective_name"] = route.objective_name
        details["scope"] = {spec.pattern_id: spec.scope.to_dict() for spec in route.knowledge.specs}
        if not any(key in details for key in ("pattern_id", "pattern_ids", "source_pattern_ids")):
            details["pattern_ids"] = [spec.pattern_id for spec in route.knowledge.specs]
        raise EngineError(error.code, str(error), details=details, hints=error.hints) from error


def project_objective(
    space: Space, observations: ObservationBatch | None, knowledge: Knowledge, objective_name: str,
) -> tuple[Space, ObservationBatch | None, Knowledge]:
    """Project a named output without changing input encoding or provenance.

    In multi-output mode, validate scopes and response composition before selection.
    First call ``compile_runtime(full_space, ...)`` to validate global declarations
    and obtain their consumers once. Compile this returned Knowledge against its
    projected Space for the GP; projection never recompiles global safety.
    A single-output call is an identity operation for legacy compatibility.
    """
    if not isinstance(space, Space) or not isinstance(knowledge, Knowledge):
        raise TypeError("space and knowledge must be Space and Knowledge")
    if observations is not None and not isinstance(observations, ObservationBatch):
        raise TypeError("observations must be ObservationBatch or None")
    if objective_name not in space.objectives:
        raise EngineError(ErrorCode.CONFIG_INVALID, "Unknown projection objective", details={
            "objective_name": objective_name, "objective_names": list(space.objectives)})
    if len(space.objectives) == 1:
        return space, observations, knowledge
    routes = _knowledge_routes(space, observations, knowledge)
    for route in routes:
        if route.objective_name is not None:
            _compile_route(route)
    selected = next(route for route in routes if route.objective_name == objective_name)
    return selected.space, selected.observations, selected.knowledge


__all__ = ["project_objective"]
