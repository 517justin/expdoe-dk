import pytest
import torch
import pandas as pd
from jsonschema import Draft202012Validator

from expdoe_dk import EngineError, ErrorCode, Knowledge, Parameter, Space
from expdoe_dk.bo.gp import build_gp
from expdoe_dk.knowledge.artifacts import (
    ARTIFACT_CATEGORIES,
    OptimizationArtifact,
    OptimizationArtifacts,
)
from expdoe_dk.knowledge.guard import CompatibilityResult, KnowledgeValidationResult
from expdoe_dk.knowledge.registry import KnowledgePatternDefinition, PatternRegistry
from expdoe_dk.knowledge.shape import (
    ArrheniusMeanFrozen,
    CombinedMean,
    QuadraticMeanFrozen,
)
from expdoe_dk.knowledge.specs import KnowledgeScope, make_pattern_spec


def _provider_definition(pattern, version, compiler):
    def validator(spec, space, observations=None):
        return KnowledgeValidationResult(
            pattern_id=spec.pattern_id,
            state="valid",
            summary="valid",
        )

    return KnowledgePatternDefinition(
        pattern=pattern,
        version=version,
        family="provider",
        schema={},
        compiler=compiler,
        validator=validator,
        renderer=lambda spec, space: f"{pattern}@{version}",
        compatibility=lambda spec, space: CompatibilityResult(compatible=True),
    )


def test_legacy_helpers_emit_versioned_specs_and_old_payload():
    knowledge = Knowledge().with_arrhenius("temperature").with_monotone(
        "time", effect="increases_objective"
    )

    assert [(spec.pattern, spec.version) for spec in knowledge.specs] == [
        ("arrhenius", "1.0"),
        ("monotone", "1.0"),
    ]
    restored = Knowledge.from_dict(knowledge.to_dict())
    assert restored.to_dict() == knowledge.to_dict()


def test_default_registry_is_fresh_and_retains_all_five_legacy_builtins():
    first = Knowledge()
    second = Knowledge()

    assert first.registry is not second.registry
    legacy_keys = {
        ("arrhenius", "1.0"),
        ("gp_prior", "1.0"),
        ("monotone", "1.0"),
        ("quadratic_peak", "1.0"),
        ("random_augment", "1.0"),
    }
    registered_keys = {
        (item.pattern, item.version) for item in first.registry.definitions()
    }
    assert legacy_keys <= registered_keys
    assert all(first.registry.resolve(*key) is not None for key in legacy_keys)


def test_builtin_schemas_accept_every_helper_generated_parameter_payload():
    knowledge = (
        Knowledge()
        .with_arrhenius("temperature")
        .with_quadratic_peak("concentration", center=0.5)
        .with_monotone("time", effect="increases_objective")
        .with_gp_prior("medium")
        .with_random_augment(20)
    )

    for spec in knowledge.specs:
        definition = knowledge.registry.resolve(spec.pattern, spec.version)
        validator = Draft202012Validator(definition.schema)
        assert list(validator.iter_errors(spec.to_dict()["parameters"])) == []


@pytest.mark.parametrize(
    "parameter",
    [
        Parameter("dose", bounds=(1.0, 100.0), transform="log"),
        Parameter(
            "dose", kind="discrete", values=[1.0, 10.0, 100.0], transform="log"
        ),
    ],
)
def test_quadratic_peak_center_uses_the_parameter_model_transform(parameter):
    """Catches peak compilation linearly unpacking bounds instead of encoding."""
    knowledge = Knowledge().with_quadratic_peak("dose", center=10.0)
    space = Space([parameter], objectives="yield")

    artifacts = knowledge.compile(space)

    assert artifacts.mean_components[0].payload["centers"] == (pytest.approx(0.5),)


def test_quadratic_peak_rejects_a_center_outside_discrete_levels_before_compile():
    """A discrete center must be a declared level, not merely inside its range."""
    knowledge = Knowledge().with_quadratic_peak("dose", center=0.5)
    space = Space(
        [Parameter("dose", kind="discrete", values=[0.0, 1.0])],
        objectives="yield",
    )

    with pytest.raises(EngineError) as caught:
        knowledge.compile(space)

    assert caught.value.code is ErrorCode.KNOWLEDGE_INVALID
    assert "center" in repr(caught.value.details).casefold()


def test_explicit_registry_is_used_as_is_and_empty_compile_is_empty():
    registry = PatternRegistry()
    knowledge = Knowledge(registry=registry)
    space = Space([Parameter("x", bounds=(0.0, 1.0))])

    assert knowledge.registry is registry
    assert registry.definitions() == ()
    assert knowledge.compile(space) == OptimizationArtifacts()


def test_specs_are_an_immutable_tuple_and_pattern_ids_are_unique():
    knowledge = Knowledge().with_arrhenius("temperature")
    declaration = knowledge.specs[0]

    assert isinstance(knowledge.specs, tuple)
    with pytest.raises(AttributeError):
        knowledge.specs.append(declaration)
    with pytest.raises(ValueError, match="pattern_id"):
        knowledge.add(declaration)


def test_disabled_legacy_declaration_is_audit_only_not_effective():
    """Catches disabled declarations leaking into v0.4 effective knowledge."""
    enabled = Knowledge().with_random_augment(3).specs[0]
    disabled = make_pattern_spec(
        pattern_id=enabled.pattern_id,
        pattern=enabled.pattern,
        version=enabled.version,
        parameters=enabled.to_dict()["parameters"],
        scope=enabled.scope,
        confidence=enabled.confidence,
        enabled=False,
    )
    knowledge = Knowledge().add(disabled)

    assert knowledge.specs == (disabled,)
    assert knowledge.items == []
    assert knowledge.items_of("random_augment") == []
    assert not knowledge.has_kind("random_augment")
    assert knowledge.to_dict() == {"strict": False, "items": []}


def test_repeated_identical_helpers_get_stable_occurrence_distinct_ids():
    first = Knowledge().with_arrhenius("temperature").with_arrhenius("temperature")
    equivalent = (
        Knowledge().with_arrhenius("temperature").with_arrhenius("temperature")
    )

    first_ids = [spec.pattern_id for spec in first.specs]
    assert len(set(first_ids)) == 2
    assert [spec.pattern_id for spec in equivalent.specs] == first_ids
    assert [
        spec.pattern_id for spec in Knowledge.from_dict(first.to_dict()).specs
    ] == first_ids
    assert first.to_dict()["items"] == [
        {
            "kind": "arrhenius",
            "param": "temperature",
            "frozen": True,
            "activation_energy": 1.0,
            "amplitude_init": -1.0,
        },
        {
            "kind": "arrhenius",
            "param": "temperature",
            "frozen": True,
            "activation_energy": 1.0,
            "amplitude_init": -1.0,
        },
    ]


def test_old_payload_with_repeated_identical_items_round_trips_with_stable_ids():
    payload = {
        "strict": False,
        "items": [
            {"kind": "random_augment", "n": 20},
            {"kind": "random_augment", "n": 20},
        ],
    }

    first = Knowledge.from_dict(payload)
    equivalent = Knowledge.from_dict(payload)

    assert first.to_dict() == payload
    first_ids = [spec.pattern_id for spec in first.specs]
    assert len(set(first_ids)) == 2
    assert [spec.pattern_id for spec in equivalent.specs] == first_ids


def test_provider_arrhenius_v2_stays_out_of_legacy_payload():
    registry = PatternRegistry()
    registry.register(
        _provider_definition(
            "arrhenius",
            "2.0",
            lambda spec, space, observations=None: OptimizationArtifacts(),
        )
    )
    knowledge = Knowledge(registry=registry).add(
        make_pattern_spec(
            pattern="arrhenius",
            version="2.0",
            parameters={"provider_model": "custom"},
            scope=KnowledgeScope(factors=("temperature",)),
            confidence=1.0,
        )
    )

    assert knowledge.items == []
    assert not knowledge.has_kind("arrhenius")
    assert knowledge.to_dict() == {"strict": False, "items": []}
    assert knowledge.to_specs_dict()["specs"][0]["version"] == "2.0"
    knowledge.drop("arrhenius")
    assert len(knowledge.specs) == 1


def test_provider_monotone_v2_compiles_without_builtin_auto_epsilon_handling():
    def compiler(spec, space, observations=None):
        return OptimizationArtifacts(
            diagnostics=(
                OptimizationArtifact(
                    kind="provider_monotone",
                    payload={"compiled": True},
                    source_pattern_id=spec.pattern_id,
                    source_pattern=spec.pattern,
                    source_version=spec.version,
                ),
            )
        )

    registry = PatternRegistry()
    registry.register(_provider_definition("monotone", "2.0", compiler))
    knowledge = Knowledge(registry=registry).add(
        make_pattern_spec(
            pattern="monotone",
            version="2.0",
            parameters={"provider_direction": "custom"},
            scope=KnowledgeScope(factors=("time",)),
            confidence=1.0,
        )
    )
    space = Space([Parameter("time", bounds=(0.0, 1.0))])

    assert knowledge.validate(auto_rescue=True) is knowledge
    artifacts = knowledge.compile(space)

    assert artifacts.diagnostics[0].to_dict() == {
        "kind": "provider_monotone",
        "payload": {"compiled": True},
        "source_pattern_id": knowledge.specs[0].pattern_id,
        "source_pattern": "monotone",
        "source_version": "2.0",
    }
    assert not knowledge.has_kind("monotone")
    assert knowledge.to_dict() == {"strict": False, "items": []}


def test_all_legacy_helpers_serialize_separately_as_versioned_specs():
    knowledge = (
        Knowledge()
        .with_arrhenius("temperature", activation_energy=2.0, amplitude_init=-3.0)
        .with_quadratic_peak("concentration", center=4.0, direction="valley")
        .with_monotone("time", effect="decreases_objective", epsilon=0.1)
        .with_gp_prior("strong")
        .with_random_augment(7)
        .strict()
    )

    assert knowledge.to_dict() == {
        "strict": True,
        "items": [
            {
                "kind": "arrhenius",
                "param": "temperature",
                "frozen": True,
                "activation_energy": 2.0,
                "amplitude_init": -3.0,
            },
            {
                "kind": "quadratic_peak",
                "param": "concentration",
                "center": 4.0,
                "direction": "valley",
                "frozen": True,
            },
            {
                "kind": "monotone",
                "param": "time",
                "effect": "decreases_objective",
                "n_pairs_per_dim": 5,
                "epsilon": 0.1,
                "delta_norm": 0.5,
            },
            {"kind": "gp_prior", "lengthscale": "strong"},
            {"kind": "random_augment", "n": 7},
        ],
    }
    specs_payload = knowledge.to_specs_dict()
    assert list(specs_payload) == ["specs"]
    assert [item["pattern"] for item in specs_payload["specs"]] == [
        "arrhenius",
        "quadratic_peak",
        "monotone",
        "gp_prior",
        "random_augment",
    ]


def test_builtin_compilers_emit_exact_provenance_and_numerical_descriptors():
    space = Space(
        [
            Parameter("temperature", bounds=(300.0, 400.0)),
            Parameter("time", bounds=(0.0, 10.0)),
        ],
        objectives="yield",
        maximize=True,
    )
    knowledge = (
        Knowledge()
        .with_arrhenius("temperature", activation_energy=2.5, amplitude_init=-1.5)
        .with_quadratic_peak("time", center=2.5, direction="peak")
        .with_monotone(
            "time",
            effect="increases_objective",
            n_pairs_per_dim=3,
            epsilon=0.1,
            delta_norm=0.7,
        )
        .with_gp_prior("strong")
        .with_random_augment(9)
    )

    artifacts = knowledge.compile(space)

    assert artifacts.mean_components[0].to_dict()["payload"] == {
        "temp_dim_index": 0,
        "activation_energy": 2.5,
        "amplitude_init": -1.5,
    }
    assert artifacts.mean_components[1].to_dict()["payload"] == {
        "input_dim": 2,
        "curvature_signs": [0.0, 1.0],
        "centers": [0.5, 0.25],
    }
    assert artifacts.virtual_observations[0].to_dict()["payload"] == {
        "dim": 1,
        "direction": "decreasing",
        "n_pairs_per_dim": 3,
        "epsilon": 0.1,
        "delta_norm": 0.7,
    }
    assert artifacts.priors[0].to_dict()["payload"] == {
        "lengthscale": [6.0, 15.0],
        "outputscale": [3.0, 1.5],
        "noise": [3.0, 500.0],
    }
    assert artifacts.virtual_observations[1].to_dict()["payload"] == {"n": 9}
    for category in ARTIFACT_CATEGORIES:
        for artifact in getattr(artifacts, category):
            source = next(
                spec
                for spec in knowledge.specs
                if spec.pattern_id == artifact.source_pattern_id
            )
            assert (artifact.source_pattern, artifact.source_version) == (
                source.pattern,
                source.version,
            )


def test_gp_build_consumes_provider_compiled_artifacts():
    def compiler(spec, space, observations=None):
        return OptimizationArtifacts(
            mean_components=(
                OptimizationArtifact(
                    kind="arrhenius_mean",
                    payload={
                        "temp_dim_index": 0,
                        "activation_energy": 4.5,
                        "amplitude_init": -2.5,
                    },
                    source_pattern_id=spec.pattern_id,
                    source_pattern=spec.pattern,
                    source_version=spec.version,
                ),
            )
        )

    def validator(spec, space, observations=None):
        return KnowledgeValidationResult(
            pattern_id=spec.pattern_id,
            state="valid",
            summary="valid",
        )

    registry = PatternRegistry()
    registry.register(
        KnowledgePatternDefinition(
            pattern="provider_mean",
            version="1.0",
            family="provider",
            schema={},
            compiler=compiler,
            validator=validator,
            renderer=lambda spec, space: "provider mean",
            compatibility=lambda spec, space: CompatibilityResult(compatible=True),
        )
    )
    knowledge = Knowledge(registry=registry).add(
        make_pattern_spec(
            pattern="provider_mean",
            version="1.0",
            parameters={},
            scope=KnowledgeScope(factors=("temperature",)),
            confidence=1.0,
        )
    )
    space = Space([Parameter("temperature", bounds=(300.0, 400.0))])
    train_x = torch.tensor([[0.1], [0.5], [0.9]], dtype=torch.float64)
    train_y = torch.tensor([[0.0], [1.0], [0.0]], dtype=torch.float64)

    model, augmenter = build_gp(space, knowledge, train_x, train_y)

    assert isinstance(model.mean_module, ArrheniusMeanFrozen)
    assert float(model.mean_module.activation_energy) == pytest.approx(4.5)
    assert float(model.mean_module.amplitude.detach()) == pytest.approx(-2.5)
    assert augmenter is None
    assert knowledge.items == []
    assert knowledge.to_dict() == {"strict": False, "items": []}


def test_gp_build_preserves_builtin_numerical_components():
    space = Space(
        [
            Parameter("temperature", bounds=(300.0, 400.0)),
            Parameter("time", bounds=(0.0, 10.0)),
        ],
        objectives="yield",
        maximize=True,
    )
    knowledge = (
        Knowledge()
        .with_arrhenius("temperature", activation_energy=2.5, amplitude_init=-1.5)
        .with_quadratic_peak("time", center=2.5, direction="peak")
        .with_monotone(
            "time",
            effect="increases_objective",
            epsilon=0.1,
            delta_norm=0.7,
        )
        .with_gp_prior("strong")
    )
    train_x = torch.tensor([[0.1, 0.2], [0.5, 0.5], [0.9, 0.8]], dtype=torch.float64)
    train_y = torch.tensor([[0.0], [1.0], [0.0]], dtype=torch.float64)

    model, augmenter = build_gp(space, knowledge, train_x, train_y)

    assert isinstance(model.mean_module, CombinedMean)
    assert isinstance(model.mean_module.means[0], ArrheniusMeanFrozen)
    assert isinstance(model.mean_module.means[1], QuadraticMeanFrozen)
    torch.testing.assert_close(
        model.mean_module(torch.tensor([[0.5, 0.25]], dtype=torch.float64)),
        torch.tensor([-0.0101069204986282], dtype=torch.float64),
    )
    assert model.covar_module.base_kernel.lengthscale_prior.concentration == 6.0
    assert model.covar_module.base_kernel.lengthscale_prior.rate == 15.0
    assert model.covar_module.outputscale_prior.concentration == 3.0
    assert model.covar_module.outputscale_prior.rate == 1.5
    assert model.likelihood.noise_covar.noise_prior.concentration == 3.0
    assert model.likelihood.noise_covar.noise_prior.rate == 500.0
    assert augmenter is not None
    assert augmenter.dims == {1: "decreasing"}
    assert augmenter.epsilon == pytest.approx(0.1)
    assert augmenter.delta_norm == pytest.approx(0.7)


def test_campaign_consumes_provider_compiled_virtual_observations(monkeypatch):
    from expdoe_dk import Campaign
    from expdoe_dk.bo import loop as loop_mod

    def compiler(spec, space, observations=None):
        provenance = {
            "source_pattern_id": spec.pattern_id,
            "source_pattern": spec.pattern,
            "source_version": spec.version,
        }
        return OptimizationArtifacts(
            virtual_observations=(
                OptimizationArtifact(
                    kind="monotone",
                    payload={
                        "dim": 0,
                        "direction": "decreasing",
                        "n_pairs_per_dim": 2,
                        "epsilon": 0.1,
                        "delta_norm": 0.5,
                    },
                    **provenance,
                ),
                OptimizationArtifact(
                    kind="random_augment",
                    payload={"n": 3},
                    **provenance,
                ),
            )
        )

    def validator(spec, space, observations=None):
        return KnowledgeValidationResult(
            pattern_id=spec.pattern_id,
            state="valid",
            summary="valid",
        )

    registry = PatternRegistry()
    registry.register(
        KnowledgePatternDefinition(
            pattern="provider_virtual",
            version="1.0",
            family="provider",
            schema={},
            compiler=compiler,
            validator=validator,
            renderer=lambda spec, space: "provider virtual observations",
            compatibility=lambda spec, space: CompatibilityResult(compatible=True),
        )
    )
    knowledge = Knowledge(registry=registry).add(
        make_pattern_spec(
            pattern="provider_virtual",
            version="1.0",
            parameters={},
            scope=KnowledgeScope(factors=("x",)),
            confidence=1.0,
        )
    )
    campaign = Campaign(
        Space([Parameter("x", bounds=(0.0, 1.0))]), knowledge, seed=11
    )
    campaign.tell(pd.DataFrame({"x": [0.1, 0.5, 0.9]}), [0.0, 1.0, 0.0])
    training_rows: list[int] = []
    real_build_gp = loop_mod.build_gp

    def recording_build_gp(space, knowledge, train_x, train_y):
        training_rows.append(len(train_x))
        return real_build_gp(space, knowledge, train_x, train_y)

    monkeypatch.setattr(loop_mod, "build_gp", recording_build_gp)
    monkeypatch.setattr(loop_mod, "fit_gpytorch_mll", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        loop_mod,
        "optimize_acqf",
        lambda *_args, **_kwargs: (
            torch.tensor([[0.25]], dtype=torch.float64),
            None,
        ),
    )

    campaign.ask(q=1, iteration=4)

    assert training_rows == [3, 7, 10]
