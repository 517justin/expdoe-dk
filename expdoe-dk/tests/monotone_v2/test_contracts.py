from dataclasses import replace
import json

import pytest

from expdoe_dk.bo import knowledge_runtime
from expdoe_dk.bo.capabilities import inspect_capabilities
from expdoe_dk.bo.objective_routing import _knowledge_routes
from expdoe_dk.domain import Parameter, Space
from expdoe_dk.errors import EngineError, ErrorCode
from expdoe_dk.knowledge import Knowledge
from expdoe_dk.knowledge.artifacts import OptimizationArtifact, OptimizationArtifacts
from expdoe_dk.knowledge.patterns import physics
from expdoe_dk.knowledge.specs import KnowledgeScope, make_pattern_spec


DEFAULTS = {
    "direction": "increasing", "n_pairs_per_dim": 5,
    "epsilon": 0.05, "softness": 1.0,
}


def _space(*, maximize=True, objectives="y"):
    return Space([Parameter(name, bounds=(0., 1.)) for name in ("x", "z", "w")],
                 objectives=objectives, maximize=maximize)


def _spec(*, parameters=None, confidence=1., factors=("x",), objectives=(), **kwargs):
    return make_pattern_spec(
        pattern="monotone", version="2.0",
        parameters=dict(DEFAULTS) if parameters is None else parameters,
        confidence=confidence, scope=KnowledgeScope(factors=factors, objectives=objectives),
        **kwargs,
    )


def _artifact(*, pattern_id="KP-test", **payload):
    values = {"dim": 0, **DEFAULTS, "confidence": 1.0, **payload}
    return OptimizationArtifact("monotone_sign_v2", values, pattern_id, "monotone", "2.0")


def _check(*artifacts):
    return knowledge_runtime.check_directional_composition(
        OptimizationArtifacts(virtual_observations=artifacts))


def test_v2_defaults_and_exact_envelope():
    knowledge = Knowledge()
    assert knowledge.with_monotone_v2("x", "increases_objective") is knowledge
    spec, = knowledge.specs
    assert (spec.pattern, spec.version) == ("monotone", "2.0")
    assert spec.to_dict() == {
        "pattern_id": spec.pattern_id, "pattern": "monotone", "version": "2.0",
        "parameters": DEFAULTS,
        "scope": {"factors": ["x"], "objectives": [], "conditions": None, "region": None},
        "confidence": 1.0, "evidence": [], "enabled": True,
    }
    assert knowledge.to_envelope() == {
        "schema_version": "1.0", "engine_version": "0.5.1", "strict": False,
        "specs": [spec.to_dict()], "providers": [],
    }
    with pytest.raises(TypeError):
        spec.parameters["epsilon"] = .2
    detached = spec.to_dict()
    detached["parameters"]["epsilon"] = .2
    assert spec.parameters["epsilon"] == .05
    artifact, = knowledge.compile(_space()).virtual_observations
    assert artifact.to_dict() == {
        "kind": "monotone_sign_v2",
        "payload": {"dim": 0, **DEFAULTS, "direction": "decreasing", "confidence": 1.0},
        "source_pattern_id": spec.pattern_id, "source_pattern": "monotone", "source_version": "2.0",
    }
    assert physics.monotone_v2_definition().version == "2.0"
    old, = Knowledge().with_monotone("x", "increases_objective").specs
    assert old.version == "1.0"
    assert dict(old.parameters) == {
        "direction": "increasing", "n_pairs_per_dim": 5, "epsilon": "auto", "delta_norm": .5,
    }


INVALID_PARAMETERS = [
    ("n_pairs_per_dim", value) for value in (True, False, 0, 33, 1.5, 5.0, float("nan"), float("inf"))
] + [
    ("epsilon", value) for value in (True, False, 0, -.1, .5, "auto", float("nan"), float("inf"))
] + [
    ("softness", value) for value in (True, False, 0, -1, float("nan"), float("inf"))
]


@pytest.mark.parametrize("name,value", INVALID_PARAMETERS)
def test_helper_rejects_invalid_parameters_without_appending(name, value):
    knowledge = Knowledge()
    with pytest.raises((TypeError, ValueError)):
        knowledge.with_monotone_v2("x", "increases_objective", **{name: value})
    assert knowledge.specs == ()


@pytest.mark.parametrize("name,value", INVALID_PARAMETERS + [("direction", "sideways"), ("delta_norm", .5)])
def test_full_declaration_rejects_invalid_parameters(name, value):
    Knowledge().registry.resolve("monotone", "2.0")
    with pytest.raises((TypeError, ValueError, EngineError)):
        spec = _spec(parameters={**DEFAULTS, name: value})
        Knowledge().add_spec(spec).compile(_space())


@pytest.mark.parametrize("confidence", [True, False, -.1, 1.1, float("nan"), float("inf")])
def test_confidence_is_finite_nonboolean_fraction(confidence):
    with pytest.raises((TypeError, ValueError)):
        Knowledge().with_monotone_v2("x", "increases_objective", confidence=confidence)
    with pytest.raises((TypeError, ValueError)):
        _spec(confidence=confidence)


@pytest.mark.parametrize("kwargs", [{"delta_norm": .5}, {"effect": "increasing"}, {"objective": ""}])
def test_helper_rejects_unknown_or_invalid_arguments(kwargs):
    with pytest.raises((TypeError, ValueError)):
        Knowledge().with_monotone_v2("x", **{"effect": "increases_objective", **kwargs})


@pytest.mark.parametrize("count,epsilon,softness,confidence", [(1, .001, .001, 0.), (32, .499, 2., 1.)])
def test_valid_boundary_values(count, epsilon, softness, confidence):
    knowledge = Knowledge().with_monotone_v2(
        "x", "increases_objective", n_pairs_per_dim=count, epsilon=epsilon,
        softness=softness, confidence=confidence)
    artifact, = knowledge.compile(_space()).virtual_observations
    assert artifact.payload["n_pairs_per_dim"] == count
    assert artifact.payload["confidence"] == confidence
    result, = knowledge.registry.validate_many(knowledge.specs, _space())
    assert result.effective_confidence == confidence
    if confidence == 0:
        assert "inactive" in result.summary.lower()
    _check(artifact)


@pytest.mark.parametrize("maximize,effect,direction", [
    (True, "increases_objective", "decreasing"), (False, "increases_objective", "increasing"),
    (True, "decreases_objective", "increasing"), (False, "decreases_objective", "decreasing"),
])
def test_compiler_uses_internal_direction(maximize, effect, direction):
    artifact, = Knowledge().with_monotone_v2("x", effect).compile(_space(maximize=maximize)).virtual_observations
    assert artifact.payload["direction"] == direction


@pytest.mark.parametrize("factors,objectives", [((), ()), (("x", "z"), ()), (("missing",), ()), (("x",), ("missing",))])
def test_unknown_or_invalid_scopes_reject(factors, objectives):
    knowledge = Knowledge().add_spec(_spec(factors=factors, objectives=objectives))
    result, = knowledge.registry.validate_many(knowledge.specs, _space())
    assert result.state == "invalid"
    with pytest.raises(EngineError):
        knowledge.compile(_space())


@pytest.mark.parametrize("parameter", [
    Parameter("x", kind="integer", bounds=(0, 10)),
    Parameter("x", kind="discrete", values=[0., .5, 1.]),
])
def test_numeric_factor_kinds_compile(parameter):
    artifacts = Knowledge().add_spec(_spec()).compile(Space([parameter], objectives="y"))
    assert artifacts.virtual_observations[0].payload["dim"] == 0


@pytest.mark.parametrize("kind", ["categorical", "ordinal"])
def test_nonnumeric_factor_kinds_reject(kind):
    space = Space([Parameter("x", kind=kind, values=["a", "b"])], objectives="y")
    with pytest.raises(EngineError):
        Knowledge().add_spec(_spec()).compile(space)


def test_multiobjective_scope_projection_retains_identity_and_separates_composition():
    space = _space(objectives=["yield", "cost"], maximize=[True, False])
    knowledge = Knowledge().add_spec(_spec(objectives=("yield", "cost"), pattern_id="KP-shared"))
    result, = knowledge.registry.validate_many(knowledge.specs, space)
    assert result.state == "valid"
    routes = _knowledge_routes(space, None, knowledge)[1:]
    for route, direction in zip(routes, ("decreasing", "increasing"), strict=True):
        artifacts = route.knowledge.compile(route.space)
        assert artifacts.virtual_observations[0].source_pattern_id == "KP-shared"
        assert artifacts.virtual_observations[0].payload["direction"] == direction
        assert knowledge_runtime.check_directional_composition(artifacts) is None
    with pytest.raises(EngineError):
        knowledge.compile(space)
    missing_scope = Knowledge().add_spec(_spec())
    assert missing_scope.registry.validate_many(missing_scope.specs, space)[0].state == "invalid"


def test_versions_roundtrip_without_legacy_item_or_auto_rescue_leakage():
    knowledge = (Knowledge().with_gp_prior(lengthscale="strong")
                 .with_monotone("x", "increases_objective")
                 .with_monotone_v2("z", "decreases_objective", epsilon=.01, confidence=0.))
    envelope = json.loads(json.dumps(knowledge.to_envelope()))
    restored = Knowledge.from_envelope(envelope)
    assert restored.to_envelope() == envelope
    assert len(restored.items_of("monotone")) == 1
    restored.validate(auto_rescue=True)
    restored.compile(_space())
    assert restored.to_envelope() == envelope
    envelope["specs"][-1]["version"] = "3.0"
    with pytest.raises(EngineError, match="Unknown pattern"):
        Knowledge.from_envelope(envelope).compile(_space())


@pytest.mark.parametrize("confidence", [0., 1.])
def test_capability_and_runtime_remain_blocked_until_consumer_exists(confidence, monkeypatch):
    from expdoe_dk.bo.capabilities import ARTIFACT_CONSUMERS
    monkeypatch.delitem(ARTIFACT_CONSUMERS, ("virtual_observations", "monotone_sign_v2"), raising=False)
    knowledge = Knowledge().with_monotone_v2("x", "increases_objective", confidence=confidence)
    assert inspect_capabilities(_space(), knowledge).blocked
    with pytest.raises(EngineError) as caught:
        knowledge_runtime.compile_runtime(_space(), knowledge, None)
    assert caught.value.code is ErrorCode.KNOWLEDGE_INVALID


@pytest.mark.parametrize("direction", ["increasing", "decreasing"])
def test_duplicate_or_opposed_factor_rejects_with_config_invalid(direction):
    first = _artifact()
    second = _artifact(direction=direction, pattern_id="KP-second")
    with pytest.raises(EngineError) as caught:
        _check(first, second)
    assert caught.value.code is ErrorCode.CONFIG_INVALID
    assert caught.value.details["pattern_ids"] == ["KP-test", "KP-second"]


@pytest.mark.parametrize("legacy_kind", ["monotone", "random_augment"])
@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_virtual_response_and_directional_rejects(legacy_kind, reverse):
    legacy = OptimizationArtifact(legacy_kind, {}, "KP-old", legacy_kind, "1.0")
    artifacts = (_artifact(), legacy)
    with pytest.raises(EngineError) as caught:
        _check(*(artifacts[::-1] if reverse else artifacts))
    assert caught.value.code is ErrorCode.CONFIG_INVALID
    assert set(caught.value.details["pattern_ids"]) == {"KP-test", "KP-old"}
    _check(legacy)
    _check(_artifact())


def test_active_pair_budget_is_per_output_and_inactive_factors_do_not_count():
    first = _artifact(n_pairs_per_dim=32)
    second = _artifact(dim=1, n_pairs_per_dim=32, pattern_id="KP-second")
    inactive = _artifact(dim=2, n_pairs_per_dim=32, confidence=0., pattern_id="KP-inactive")
    assert _check(first, second, inactive) is None
    with pytest.raises(EngineError) as caught:
        _check(first, second, replace(inactive, payload={
            **dict(inactive.payload), "confidence": .01, "n_pairs_per_dim": 1,
        }))
    assert caught.value.code is ErrorCode.CONFIG_INVALID
    _check(first, second)
    _check(replace(inactive, payload={**dict(inactive.payload), "confidence": 1.}))


@pytest.mark.parametrize("payload", [
    {"n_pairs_per_dim": 0}, {"n_pairs_per_dim": 33}, {"n_pairs_per_dim": True},
    {"epsilon": 0}, {"epsilon": .5}, {"epsilon": "auto"}, {"epsilon": True},
    {"softness": 0}, {"softness": True}, {"direction": "wrong"},
    {"dim": -1}, {"dim": True}, {"delta_norm": .5},
])
def test_inactive_malformed_payload_still_rejects(payload):
    with pytest.raises(EngineError) as caught:
        _check(_artifact(confidence=0., **payload))
    assert caught.value.code is ErrorCode.CONFIG_INVALID


@pytest.mark.parametrize("confidence", [True, -1., 1.01])
def test_composition_rejects_malformed_confidence(confidence):
    with pytest.raises(EngineError) as caught:
        _check(_artifact(confidence=confidence))
    assert caught.value.code is ErrorCode.CONFIG_INVALID


def test_composition_rejects_missing_payload_field():
    artifact = _artifact()
    payload = dict(artifact.payload)
    del payload["softness"]
    with pytest.raises(EngineError) as caught:
        _check(replace(artifact, payload=payload))
    assert caught.value.code is ErrorCode.CONFIG_INVALID


def test_legacy_conflict_code_is_unchanged():
    knowledge = (Knowledge().with_monotone("x", "increases_objective")
                 .with_monotone("x", "decreases_objective"))
    with pytest.raises(EngineError) as caught:
        knowledge_runtime.compile_runtime(_space(), knowledge, None)
    assert caught.value.code is ErrorCode.KNOWLEDGE_CONFLICT


def test_full_declaration_rejects_overflowing_softness():
    knowledge = Knowledge().add_spec(_spec(parameters={**DEFAULTS, "softness": 10 ** 400}))
    with pytest.raises(EngineError):
        knowledge.compile(_space())


def test_composition_rejects_overflowing_softness_with_config_invalid():
    with pytest.raises(EngineError) as caught:
        _check(_artifact(softness=10 ** 400))
    assert caught.value.code is ErrorCode.CONFIG_INVALID


def test_different_outputs_can_use_different_versions():
    space = _space(objectives=["yield", "cost"], maximize=[True, False])
    knowledge = Knowledge().with_monotone_v2("x", "increases_objective", objective="yield")
    knowledge.add_spec(make_pattern_spec(
        pattern="monotone", version="1.0", confidence=1.,
        parameters={"direction": "decreasing", "n_pairs_per_dim": 5, "epsilon": .1, "delta_norm": .5},
        scope=KnowledgeScope(factors=("x",), objectives=("cost",)),
    ))
    for route in _knowledge_routes(space, None, knowledge)[1:]:
        assert knowledge_runtime.check_directional_composition(route.knowledge.compile(route.space)) is None


@pytest.mark.parametrize("qualifier", ["conditions", "region"])
def test_unimplemented_qualified_scopes_reject(qualifier):
    spec = _spec()
    scoped = replace(spec, parameters=dict(spec.parameters),
                     scope=KnowledgeScope(factors=("x",), **{qualifier: {"x": .5}}))
    with pytest.raises(EngineError):
        Knowledge().add_spec(scoped).compile(_space())
