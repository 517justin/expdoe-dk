"""Version-explicit declarations stay separate from numerical consumer support."""
import json

import numpy as np
import pytest

from expdoe_dk import EngineError, ErrorCode, Knowledge, Parameter, Space, inspect_capabilities
from expdoe_dk.knowledge.patterns import builtin_pattern_definitions
from expdoe_dk.knowledge.specs import KnowledgeScope, make_pattern_spec


PATTERNS = ("saturation", "threshold", "synergy", "antagonism")


def numeric_space():
    return Space([Parameter("x", bounds=(0, 1)), Parameter("z", bounds=(0, 1))],
                 objectives="y")


def declaration(pattern, **changes):
    parameters = {
        "saturation": {"direction": "increasing", "half_response": .5, "origin": 0},
        "threshold": {"threshold": .5, "below_behavior": "increasing", "above_behavior": "flat"},
        "synergy": {"reference": "anchored_additive", "response_effect": "higher"},
        "antagonism": {"reference": "anchored_additive", "response_effect": "lower"},
    }[pattern]
    fields = dict(pattern=pattern, version="2.0", parameters=parameters,
                  scope=KnowledgeScope(factors=("x",) if pattern in PATTERNS[:2]
                                       else ("x", "z"), objectives=("y",)), confidence=1)
    fields.update(changes)
    return make_pattern_spec(**fields)


@pytest.fixture(params=PATTERNS)
def v2_knowledge(request):
    pattern = request.param
    if pattern == "saturation":
        return Knowledge().with_saturation("x", direction="increasing", half_response=.5,
                                           version="2.0", objective="y", origin=0)
    if pattern == "threshold":
        return Knowledge().with_threshold("x", threshold=.5, below_behavior="increasing",
                                          above_behavior="flat", version="2.0", objective="y")
    return Knowledge().with_interaction("x", "z", kind=pattern, version="2.0", objective="y",
                                       reference="anchored_additive", response_effect="higher")


def test_helpers_opt_in_to_v2_without_changing_the_v1_default(v2_knowledge):
    assert Knowledge().with_saturation("x", direction="increasing", half_response=.5).specs[0].version == "1.0"
    assert v2_knowledge.specs[0].version == "2.0"
    assert v2_knowledge.specs[0].scope.objectives == ("y",)
    assert len({d.pattern for d in builtin_pattern_definitions()}) == 23
    assert len(builtin_pattern_definitions()) == 27


def test_each_v2_artifact_is_still_capability_blocked(v2_knowledge):
    report = inspect_capabilities(numeric_space(), v2_knowledge)
    assert report.applied == ()
    assert report.blocked == (v2_knowledge.specs[0].pattern_id,)
    assert any(item.get("reason") == "unsupported_artifact" for item in report.diagnostics)


@pytest.mark.parametrize("pattern,kind", [("saturation", "saturation_mean_v2"),
    ("threshold", "threshold_mean_v2"), ("synergy", "signed_interaction_mean_v2"),
    ("antagonism", "signed_interaction_mean_v2")])
def test_compilers_emit_physical_coordinates_and_exact_provenance(pattern, kind):
    spec = declaration(pattern)
    compiled = Knowledge().add(spec).compile(numeric_space())
    artifact, = compiled.mean_components
    factors = ["x"] if pattern in PATTERNS[:2] else ["x", "z"]
    assert artifact.to_dict() == {
        "kind": kind, "source_pattern_id": spec.pattern_id,
        "source_pattern": pattern, "source_version": "2.0",
        "payload": {"factors": factors, "dimensions": [0] if len(factors) == 1 else [0, 1],
            "objectives": ["y"], "parameters": spec.to_dict()["parameters"], "confidence": 1.,
            "coordinates": [{"factor": name, "lower": 0, "upper": 1, "range": 1,
                             "transform": "linear"} for name in factors]},
    }
    assert compiled.kernel_components == compiled.virtual_observations == ()


@pytest.mark.parametrize("pattern", PATTERNS[2:])
def test_interaction_factors_are_canonicalized_in_space_order(pattern):
    spec = declaration(pattern, scope=KnowledgeScope(factors=("z", "x"), objectives=("y",)))
    artifact, = Knowledge().add(spec).compile(numeric_space()).mean_components
    assert artifact.payload["factors"] == ("x", "z")
    assert artifact.payload["dimensions"] == (0, 1)


def test_coordinates_retain_physical_log_bounds_and_effective_transform():
    space = Space([Parameter("x", bounds=(1, 100), log_scale=True)], objectives="y")
    spec = declaration("saturation", parameters={"direction": "decreasing", "origin": 1,
                                                "half_response": 10})
    artifact, = Knowledge().add(spec).compile(space).mean_components
    assert artifact.to_dict()["payload"]["coordinates"] == [
        {"factor": "x", "lower": 1, "upper": 100, "range": 99, "transform": "log"}]


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("objectives", [(), ("unknown",), ("y", "other")])
def test_v2_requires_one_explicit_known_response(pattern, objectives):
    original = declaration(pattern)
    spec = declaration(pattern, scope=KnowledgeScope(factors=original.scope.factors,
                                                     objectives=objectives))
    with pytest.raises(EngineError) as caught:
        Knowledge().add(spec).compile(numeric_space())
    assert caught.value.code is ErrorCode.KNOWLEDGE_INVALID


@pytest.mark.parametrize("pattern", PATTERNS)
def test_v2_rejects_multiple_physical_responses(pattern):
    space = Space(numeric_space().params, objectives=["y", "other"])
    with pytest.raises(EngineError) as caught:
        Knowledge().add(declaration(pattern)).compile(space)
    assert caught.value.code is ErrorCode.KNOWLEDGE_INVALID


@pytest.mark.parametrize("pattern", PATTERNS)
def test_v2_rejects_categorical_scoped_factor(pattern):
    space = Space([Parameter("x", kind="categorical", values=["a", "b"]),
                   Parameter("z", bounds=(0, 1))], objectives="y")
    with pytest.raises(EngineError) as caught:
        Knowledge().add(declaration(pattern)).compile(space)
    assert caught.value.code is ErrorCode.KNOWLEDGE_INVALID


@pytest.mark.parametrize("pattern", PATTERNS)
def test_enabled_zero_confidence_is_invalid_but_disabled_round_trips(pattern):
    with pytest.raises(EngineError) as caught:
        Knowledge().add(declaration(pattern, confidence=0)).compile(numeric_space())
    assert caught.value.code is ErrorCode.KNOWLEDGE_INVALID
    disabled = Knowledge().add(declaration(pattern, confidence=0, enabled=False))
    restored = Knowledge.from_envelope(disabled.to_envelope())
    assert restored.to_envelope() == disabled.to_envelope()
    assert restored.compile(numeric_space()).mean_components == ()


@pytest.mark.parametrize("pattern,key,value", [
    ("saturation", "origin", .1), ("saturation", "half_response", 0),
    ("saturation", "half_response", 1), ("saturation", "direction", "higher"),
    ("threshold", "threshold", 0), ("threshold", "threshold", 1),
    ("threshold", "above_behavior", "increasing"),
    ("synergy", "reference", "bliss"), ("antagonism", "reference", "bliss"),
    ("synergy", "response_effect", "maximize"), ("antagonism", "response_effect", "minimize"),
])
def test_invalid_v2_semantics_raise_structured_errors(pattern, key, value):
    parameters = declaration(pattern).to_dict()["parameters"]
    parameters[key] = value
    with pytest.raises(EngineError) as caught:
        Knowledge().add(declaration(pattern, parameters=parameters)).compile(numeric_space())
    assert caught.value.code is ErrorCode.KNOWLEDGE_INVALID


@pytest.mark.parametrize("pattern", PATTERNS)
def test_v2_schemas_reject_missing_parameters(pattern):
    for key in declaration(pattern).parameters:
        parameters = declaration(pattern).to_dict()["parameters"]
        del parameters[key]
        with pytest.raises(EngineError) as caught:
            Knowledge().add(declaration(pattern, parameters=parameters)).compile(numeric_space())
        assert caught.value.code is ErrorCode.KNOWLEDGE_INVALID


@pytest.mark.parametrize("pattern", PATTERNS[2:])
def test_registry_rejects_repeated_interaction_factor(pattern):
    # Model deserialization already rejects duplicates; defend the runtime boundary too.
    scope = KnowledgeScope(factors=("x", "z"), objectives=("y",))
    object.__setattr__(scope, "factors", ("x", "x"))
    with pytest.raises(EngineError) as caught:
        Knowledge().add(declaration(pattern, scope=scope)).compile(numeric_space())
    assert caught.value.code is ErrorCode.KNOWLEDGE_INVALID


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_registry_rejects_nonfinite_v2_values_at_runtime(pattern, value):
    spec = declaration(pattern)
    object.__setattr__(spec, "confidence", value)
    with pytest.raises(EngineError) as caught:
        Knowledge().add(spec).compile(numeric_space())
    assert caught.value.code is ErrorCode.KNOWLEDGE_INVALID


@pytest.mark.parametrize("pattern,key", [("saturation", "origin"), ("saturation", "half_response"),
                                        ("threshold", "threshold")])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_registry_rejects_nonfinite_locations(pattern, key, value):
    spec = declaration(pattern)
    parameters = spec.to_dict()["parameters"]
    parameters[key] = value
    object.__setattr__(spec, "parameters", parameters)
    with pytest.raises(EngineError) as caught:
        Knowledge().add(spec).compile(numeric_space())
    assert caught.value.code is ErrorCode.KNOWLEDGE_INVALID


@pytest.mark.parametrize("pattern", PATTERNS)
def test_nonfinite_physical_range_is_rejected_with_structured_error(pattern):
    space = Space([Parameter("x", bounds=(-1e308, 1e308)), Parameter("z", bounds=(0, 1))],
                  objectives="y")
    parameters = declaration(pattern).to_dict()["parameters"]
    if pattern == "saturation":
        parameters["origin"] = -1e308
    with pytest.raises(EngineError) as caught:
        Knowledge().add(declaration(pattern, parameters=parameters)).compile(space)
    assert caught.value.code is ErrorCode.KNOWLEDGE_INVALID


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("scalar,lower,upper", [
    (np.int64, 0, 10), (np.float32, 0., 10.),
    (np.int64, 9007199254740993, 9007199254741003),
])
def test_numpy_bounds_compile_as_builtin_coordinates_without_integer_precision_loss(
        pattern, scalar, lower, upper):
    space = Space([Parameter("x", kind="integer" if scalar is np.int64 else "continuous",
                             bounds=(scalar(lower), scalar(upper))),
                   Parameter("z", bounds=(0, 1))], objectives="y")
    parameters = declaration(pattern).to_dict()["parameters"]
    if pattern == "saturation":
        parameters.update(origin=lower, half_response=lower + 5)
    elif pattern == "threshold":
        parameters["threshold"] = lower + 5
    artifact, = Knowledge().add(declaration(pattern, parameters=parameters)).compile(space).mean_components
    coordinate = artifact.to_dict()["payload"]["coordinates"][0]
    assert coordinate == {"factor": "x", "lower": lower, "upper": upper,
                          "range": 10, "transform": "linear"}
    expected_type = int if scalar is np.int64 else float
    assert all(type(coordinate[key]) is expected_type for key in ("lower", "upper", "range"))
    json.dumps(artifact.to_dict(), allow_nan=False)


@pytest.mark.parametrize("pattern,key", [("saturation", "origin"), ("saturation", "half_response"),
                                        ("threshold", "threshold")])
@pytest.mark.parametrize("value", [10**400, -(10**400)])
def test_oversized_integer_parameters_raise_structured_registry_error(pattern, key, value):
    parameters = declaration(pattern).to_dict()["parameters"]
    parameters[key] = value
    with pytest.raises(EngineError) as caught:
        Knowledge().add(declaration(pattern, parameters=parameters)).compile(numeric_space())
    assert caught.value.code is ErrorCode.KNOWLEDGE_INVALID


@pytest.mark.parametrize("pattern", PATTERNS)
def test_oversized_integer_bounds_raise_structured_registry_error(pattern):
    space = Space([Parameter("x", kind="integer", bounds=(10**400, 10**400 + 10)),
                   Parameter("z", bounds=(0, 1))], objectives="y")
    with pytest.raises(EngineError) as caught:
        Knowledge().add(declaration(pattern)).compile(space)
    assert caught.value.code is ErrorCode.KNOWLEDGE_INVALID


@pytest.mark.parametrize("pattern,key", [
    ("saturation", "origin"), ("saturation", "half_response"), ("saturation", "confidence"),
    ("threshold", "threshold"), ("threshold", "confidence"),
    ("synergy", "confidence"), ("antagonism", "confidence"),
])
@pytest.mark.parametrize("value", [10**400, -(10**400)])
def test_oversized_integer_helper_arguments_raise_value_error(pattern, key, value):
    spec = declaration(pattern)
    kwargs = dict(spec.to_dict()["parameters"], version="2.0", objective="y")
    kwargs[key] = value
    with pytest.raises(ValueError):
        if pattern == "saturation":
            Knowledge().with_saturation("x", **kwargs)
        elif pattern == "threshold":
            Knowledge().with_threshold("x", **kwargs)
        else:
            Knowledge().with_interaction("x", "z", kind=pattern, **kwargs)


@pytest.mark.parametrize("method,args,kwargs", [
    ("with_saturation", ("x",), {"direction": "increasing", "half_response": .5}),
    ("with_threshold", ("x",), {"threshold": .5, "below_behavior": "flat", "above_behavior": "increasing"}),
    ("with_interaction", ("x", "z"), {"kind": "synergy"}),
])
def test_helpers_reject_unknown_versions_and_missing_v2_semantics(method, args, kwargs):
    with pytest.raises(ValueError):
        getattr(Knowledge(), method)(*args, **kwargs, version="9.0")
    with pytest.raises(ValueError):
        getattr(Knowledge(), method)(*args, **kwargs, version="2.0")
    with pytest.raises(ValueError):
        getattr(Knowledge(), method)(*args, **kwargs, objective="y")


@pytest.mark.parametrize("extra", [{"origin": 0}, {"objective": "y", "origin": 0}])
def test_v1_saturation_rejects_v2_only_arguments(extra):
    with pytest.raises(ValueError):
        Knowledge().with_saturation("x", direction="increasing", half_response=.5, **extra)


@pytest.mark.parametrize("extra", [{"reference": "anchored_additive"}, {"response_effect": "higher"}])
def test_v1_interaction_rejects_v2_only_arguments(extra):
    with pytest.raises(ValueError):
        Knowledge().with_interaction("x", "z", kind="synergy", **extra)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_v2_helpers_reject_nonfinite_or_boolean_location_with_value_error(value):
    with pytest.raises(ValueError):
        Knowledge().with_saturation("x", direction="increasing", half_response=value,
                                   origin=0, version="2.0", objective="y")


def test_v1_envelope_is_canonically_identical_after_restore():
    knowledge = (Knowledge().with_saturation("x", direction="increasing", half_response=.5)
                 .with_threshold("x", threshold=.5, below_behavior="flat", above_behavior="flat")
                 .with_interaction("x", "z", kind="antagonism"))
    before = json.dumps(knowledge.to_envelope(), sort_keys=True, separators=(",", ":"))
    restored = Knowledge.from_envelope(json.loads(before))
    assert json.dumps(restored.to_envelope(), sort_keys=True, separators=(",", ":")) == before
    assert all(spec.version == "1.0" for spec in restored.specs)


@pytest.mark.parametrize("pattern,words", [("saturation", ("half-response", "onset", "anchor")),
    ("threshold", ("midpoint", "asymptotically flat", "anchor")),
    ("synergy", ("higher", "additive", "anchor")),
    ("antagonism", ("lower", "additive", "anchor"))])
def test_renderers_explain_physical_meaning_and_confirmation(pattern, words):
    knowledge = Knowledge().add(declaration(pattern))
    text = knowledge.registry.render(knowledge.specs[0], numeric_space()).lower()
    assert all(word in text for word in words)
    assert "confirmation" in text
