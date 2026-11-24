"""Inventory actual consumers of registry-emitted optimization artifacts.

Registry discovery and serialization are intentionally broader than numerical
support. Unknown kinds fail closed; adding a compiler does not silently make a
pattern usable by the Phase 1 optimizer.
"""
from __future__ import annotations

from expdoe_dk.domain import Space
from expdoe_dk.errors import EngineError
from expdoe_dk.knowledge import Knowledge
from expdoe_dk.knowledge.artifacts import ARTIFACT_CATEGORIES

from .contracts import CapabilityReport


# Only these kinds have a wired numerical consumer in the v0.5 baseline.
# E2 may expand this manifest when it actually wires additional consumers.
ARTIFACT_CONSUMERS: dict[tuple[str, str], str] = {
    ("mean_components", "arrhenius_mean"): "bo.gp.build_mean_from_artifacts",
    ("mean_components", "quadratic_mean"): "bo.gp.build_mean_from_artifacts",
    ("mean_components", "power_law_descriptor"): "bo.gp.build_mean_from_artifacts",
    ("mean_components", "exponential_descriptor"): "bo.gp.build_mean_from_artifacts",
    ("mean_components", "quadratic_valley_descriptor"): "bo.gp.build_mean_from_artifacts",
    ("mean_components", "periodic_descriptor"): "bo.gp.build_mean_from_artifacts",
    ("mean_components", "saturation_mean_v2"): "bo.gp.build_mean_from_artifacts",
    ("mean_components", "threshold_mean_v2"): "bo.gp.build_mean_from_artifacts",
    ("mean_components", "signed_interaction_mean_v2"): "bo.gp.build_mean_from_artifacts",
    ("kernel_components", "category_similarity_descriptor"): "bo.gp.build_kernel_from_artifacts",
    ("parameter_constraints", "hard_parameter_constraint"): "bo.knowledge_runtime.RuntimeParameterConstraint",
    ("outcome_constraints", "target_range_constraint"): "bo.knowledge_runtime.RuntimeOutcomeRange",
    ("acquisition_preferences", "objective_priority_preference"): "bo.knowledge_runtime.RuntimePreference",
    ("acquisition_preferences", "tradeoff_preference"): "bo.knowledge_runtime.RuntimePreference",
    ("virtual_observations", "monotone"): "bo.gp.build_virtual_observation_augmenter",
    ("virtual_observations", "random_augment"): "bo.loop.Campaign.ask",
    ("priors", "gp_prior"): "bo.gp.build_prior_modules",
    ("diagnostics", "knowledge_validation"): "bo.capabilities.inspect_capabilities",
    ("diagnostics", "safety_feasibility"): "bo.capabilities.inspect_capabilities",
}


def inspect_capabilities(space: Space, knowledge: Knowledge) -> CapabilityReport:
    """Classify every enabled pattern instance against Phase 1 consumers.

    This is an inspection boundary, not a provider loader or a model fitter.
    Multi-output declarations are checked against their named response or global
    route. Incompatible declarations and kinds without consumers are blocked.
    """
    if not isinstance(space, Space):
        raise TypeError("space must be a Space")
    if not isinstance(knowledge, Knowledge):
        raise TypeError("knowledge must be a Knowledge")
    if len(space.objectives) > 1:
        return _inspect_multiobjective(space, knowledge)

    applied: list[str] = []
    blocked: list[str] = []
    diagnostics: list[dict] = []
    for spec in knowledge.specs:
        if not spec.enabled:
            continue
        identity = {"pattern_id": spec.pattern_id, "pattern": spec.pattern,
                    "version": spec.version}
        if len(space.objectives) != 1:
            blocked.append(spec.pattern_id)
            diagnostics.append({**identity, "reason": "multi_objective_unsupported"})
            continue
        try:
            definition = knowledge.registry.resolve(spec.pattern, spec.version)
            validation = knowledge.registry.validate(spec, space)
            compatibility = definition.compatibility(spec, space)
            if not compatibility.compatible:
                blocked.append(spec.pattern_id)
                diagnostics.append({**identity, "reason": "incompatible_space",
                                    "details": list(compatibility.reasons)})
                continue
            if validation.state == "invalid":
                blocked.append(spec.pattern_id)
                diagnostics.append({**identity, "reason": "invalid_knowledge",
                                    "details": list(validation.errors)})
                continue
            artifacts = knowledge.registry.compile_many((spec,), space)
        except EngineError as error:
            blocked.append(spec.pattern_id)
            diagnostics.append({**identity, "reason": "invalid_knowledge",
                                "code": error.code.value, "details": str(error)})
            continue

        has_numerical_artifact = False
        unsupported = False
        for category in ARTIFACT_CATEGORIES:
            for artifact in getattr(artifacts, category):
                if category != "diagnostics":
                    has_numerical_artifact = True
                consumer = ARTIFACT_CONSUMERS.get((category, artifact.kind))
                entry = {**identity, "artifact_kind": artifact.kind,
                         "category": category}
                from .structured_coverage import STRUCTURED_KINDS
                if artifact.kind in STRUCTURED_KINDS:
                    entry.update(coverage_status="not_evaluated", empirical_evidence="not_evaluated")
                if consumer is None:
                    unsupported = True
                    diagnostics.append({**entry, "reason": "unsupported_artifact"})
                else:
                    from .knowledge_runtime import validate_runtime_artifact
                    try:
                        validate_runtime_artifact(space, category, artifact)
                    except EngineError as error:
                        unsupported = True
                        diagnostics.append({**entry, "reason": "unsupported_artifact",
                                            "details": str(error)})
                    else:
                        diagnostics.append({**entry, "consumer": consumer,
                                            "validation_state": validation.state})
        if not has_numerical_artifact:
            unsupported = True
            diagnostics.append({**identity, "reason": "no_numerical_artifact"})
        (blocked if unsupported else applied).append(spec.pattern_id)
    # Composition is a whole-declaration property, never a per-pattern check.
    if any(spec.enabled and spec.version == "2.0" and spec.pattern in
           {"saturation", "threshold", "synergy", "antagonism"} for spec in knowledge.specs):
        from .structured_coverage import check_structured_composition
        try:
            check_structured_composition(space, knowledge.compile(space))
        except EngineError as error:
            affected = error.details.get("pattern_ids", [])
            if not affected:
                affected = [error.details.get("pattern_id")] if error.details.get("pattern_id") else [s.pattern_id for s in knowledge.specs if s.enabled]
            for identity in affected:
                if identity in applied:
                    applied.remove(identity)
                if identity not in blocked:
                    blocked.append(identity)
            diagnostics.append({"pattern_ids": affected, "reason": error.details.get("reason", "invalid_knowledge"),
                                "code": error.code.value, "details": error.details})
    return CapabilityReport(tuple(applied), tuple(blocked), tuple(diagnostics))


def _inspect_multiobjective(space, knowledge):
    from .objective_routing import _compile_route, _knowledge_routes, _subset
    from .structured_coverage import STRUCTURED_KINDS

    applied, blocked, diagnostics = [], [], []

    def record_error(error, fallback):
        affected = error.details.get("pattern_ids", error.details.get("source_pattern_ids", []))
        if not affected:
            affected = [error.details["pattern_id"]] if "pattern_id" in error.details else fallback
        for identity in affected:
            if identity in applied:
                applied.remove(identity)
            if identity not in blocked:
                blocked.append(identity)
        diagnostics.append({**error.details, "reason": error.details.get("reason", "invalid_knowledge"),
                            "code": error.code.value, "details": str(error)})

    for spec in knowledge.specs:
        if not spec.enabled:
            continue
        try:
            routes = _knowledge_routes(space, None, _subset(knowledge, (spec,)))
            for route in routes:
                artifacts = _compile_route(route)
                for category in ARTIFACT_CATEGORIES:
                    for artifact in getattr(artifacts, category):
                        entry = {"pattern_id": spec.pattern_id, "pattern": spec.pattern,
                                 "version": spec.version, "objective_name": route.objective_name,
                                 "scope": spec.scope.to_dict(), "artifact_kind": artifact.kind,
                                 "category": category, "consumer": ARTIFACT_CONSUMERS[(category, artifact.kind)]}
                        if artifact.kind in STRUCTURED_KINDS:
                            entry.update(coverage_status="not_evaluated", empirical_evidence="not_evaluated")
                        diagnostics.append(entry)
        except EngineError as error:
            record_error(error, [spec.pattern_id])
        else:
            applied.append(spec.pattern_id)
    # Per-pattern success cannot certify composition, prior or global conflicts.
    try:
        routes = _knowledge_routes(space, None, knowledge)
    except EngineError:
        pass  # The per-pattern pass already recorded malformed scopes.
    else:
        for route in routes:
            try:
                _compile_route(route)
            except EngineError as error:
                record_error(error, [spec.pattern_id for spec in route.knowledge.specs])
    return CapabilityReport(tuple(applied), tuple(blocked), tuple(diagnostics))


__all__ = ["ARTIFACT_CONSUMERS", "inspect_capabilities"]
