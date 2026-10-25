"""Campaign persistence contract: physical data, deterministic resume, safe JSON."""
import json

import numpy as np
import pandas as pd
import pytest

from expdoe_dk import (
    Campaign, EngineError, ErrorCode, Evidence, Knowledge, KnowledgeScope,
    Objective, ObservationBatch, Parameter, PendingBatch, Space, make_pattern_spec,
)


@pytest.fixture
def populated_campaign():
    space = Space([Parameter("x", kind="integer", bounds=(0, 12))], objectives="yield")
    knowledge = Knowledge().add(make_pattern_spec(
        pattern="gp_prior", version="1.0", parameters={"lengthscale": "medium"},
        scope=KnowledgeScope(), confidence=0.8,
        evidence=(Evidence("measurement", "lab-book-17"),),
    ))
    campaign = Campaign(space, knowledge, seed=17, validate=False)
    campaign.tell_batch(ObservationBatch(
        pd.DataFrame({"x": [0, 2, 4, 6, 8]}),
        pd.DataFrame({"yield": [1., 3., 2., 99., None]}),
        Yvar=pd.DataFrame({"yield": [0.1, 0.2, 0.1, 0.4, 0.5]}),
        status=pd.Series(["success", "success", "success", "warning", "failed"]),
        ids=("o0", "o2", "o4", "o6", "o8"),
    ))
    campaign.set_pending(PendingBatch(("p10", "p11"), pd.DataFrame({"x": [10, 11]})))
    campaign.ask(q=1, iteration=7)
    campaign._rng.random(4)  # Exercise non-initial PCG64 progression as well.
    return campaign


def test_restore_preserves_next_batch(populated_campaign):
    state = populated_campaign.to_state()
    restored = Campaign.from_state(state)
    assert restored.to_state() == state
    actual = restored.ask(q=2)
    pd.testing.assert_frame_equal(actual, populated_campaign.ask(q=2))
    assert not set(actual.x) & {0, 2, 4, 6, 8, 10, 11}
    np.testing.assert_array_equal(restored._rng.random(5), populated_campaign._rng.random(5))


def test_checkpoint_writer_identifies_all_envelopes_as_051():
    campaign = Campaign(Space([Parameter("x", bounds=(0., 1.))]), seed=7)

    state = campaign.to_state()

    assert state["engine_version"] == "0.5.1"
    assert state["space"]["engine_version"] == "0.5.1"
    assert state["knowledge"]["engine_version"] == "0.5.1"


@pytest.mark.parametrize("section", [None, "space", "knowledge"])
def test_checkpoint_rejects_unknown_future_patch_with_typed_error(section):
    campaign = Campaign(Space([Parameter("x", bounds=(0., 1.))]), seed=7)
    state = campaign.to_state()
    envelope = state if section is None else state[section]
    envelope["engine_version"] = "0.5.99"

    with pytest.raises(EngineError) as caught:
        Campaign.from_state(state)

    assert caught.value.code is ErrorCode.CHECKPOINT_INCOMPATIBLE


def test_restore_rejects_unknown_schema(populated_campaign):
    state = populated_campaign.to_state()
    state["schema_version"] = "99.0"
    with pytest.raises(EngineError):
        Campaign.from_state(state)


def test_physical_metadata_roundtrip_and_legacy_history(populated_campaign, tmp_path):
    state = populated_campaign.to_state()
    assert state["schema_version"] == "2.1"
    batch = state["observations"][0]
    assert batch["Y"] == [[1.0], [3.0], [2.0], [99.0], [None]]
    assert batch["Yvar"] == [[0.1], [0.2], [0.1], [0.4], [0.5]]
    assert batch["status"] == ["success", "success", "success", "warning", "failed"]
    assert batch["ids"] == ["o0", "o2", "o4", "o6", "o8"]
    assert state["knowledge"]["specs"][0]["evidence"] == [
        {"kind": "measurement", "reference": "lab-book-17"}]
    path = tmp_path / "campaign.json"
    populated_campaign.save_checkpoint(path)
    assert json.loads(path.read_text()) == state
    restored = Campaign.load_checkpoint(path)
    assert restored.to_state() == state
    assert restored.history_df().to_dict("list") == {
        "x": [0, 2, 4], "y": [1., 3., 2.], "kind": ["doe"] * 3}
    assert restored.finalize().best_y_physical == 3.


def test_empty_history_roundtrips_and_pending_can_clear():
    campaign = Campaign(Space([Parameter("x", bounds=(0., 1.))]), seed=0)
    campaign.set_pending(PendingBatch(("p",), pd.DataFrame({"x": [0.5]})))
    restored = Campaign.from_state(campaign.to_state())
    assert restored.to_state() == campaign.to_state()
    assert restored.history_df().empty
    with pytest.raises(EngineError) as caught:
        restored.ask()
    assert caught.value.code is ErrorCode.INSUFFICIENT_DATA
    restored.set_pending(None)
    assert restored.to_state()["pending"] is None


@pytest.mark.parametrize("target, expected", [(5., 4.), ((4., 6.), 4.)])
def test_target_best_result_uses_utility_but_keeps_physical_y(target, expected):
    campaign = Campaign(Space([Parameter("x", bounds=(0., 1.))],
        objectives=Objective("yield", "target", target=target)), validate=False)
    campaign.tell(pd.DataFrame({"x": [0., 0.5, 1.]}), [1., 4., 9.])
    restored = Campaign.from_state(campaign.to_state())
    assert restored.space.objective_specs == campaign.space.objective_specs
    assert restored.finalize().best_y_physical == expected
    assert restored.history_df().y.tolist() == [1., 4., 9.]
    assert restored._y_internal == pytest.approx(-campaign.space.objective_specs[0].to_utility(np.array([1., 4., 9.])))


def test_legacy_tell_delegates_to_batch_and_retains_kinds():
    campaign = Campaign(Space([Parameter("x", bounds=(0., 1.))]), validate=False)
    campaign.tell(pd.DataFrame({"x": [0., 0.5]}), [1., 2.])
    campaign.tell(pd.DataFrame({"x": [1.]}), [3.])
    assert campaign.history_df().kind.tolist() == ["doe", "doe", "bo"]
    assert len(campaign.to_state()["observations"]) == 2


def test_pending_repeated_conditions_are_preserved_and_excluded(populated_campaign):
    populated_campaign.set_pending(PendingBatch(
        ("p1", "p2", "p3"), pd.DataFrame({"x": [10, 10, 0]})))
    restored = Campaign.from_state(populated_campaign.to_state())
    assert restored.to_state()["pending"]["ids"] == ["p1", "p2", "p3"]
    assert not set(restored.ask(q=2).x) & {0, 2, 4, 6, 8, 10}


@pytest.mark.parametrize("method, value", [
    ("tell_batch", None), ("set_pending", pd.DataFrame({"x": [1]})),
    ("tell_batch", ObservationBatch(pd.DataFrame({"wrong": [1]}), pd.DataFrame({"y": [2]}))),
    ("tell_batch", ObservationBatch(pd.DataFrame({"x": [1]}), pd.DataFrame({"wrong": [2]}))),
    ("set_pending", PendingBatch(("bad",), pd.DataFrame({"x": [2.]}))),
])
def test_batch_boundaries_reject_invalid_input_without_mutation(method, value):
    campaign = Campaign(Space([Parameter("x", bounds=(0., 1.))]))
    before = campaign.to_state()
    with pytest.raises(EngineError):
        getattr(campaign, method)(value)
    assert campaign.to_state() == before


def test_multiobjective_new_boundary_fails_explicitly():
    with pytest.warns(UserWarning):
        campaign = Campaign(Space([Parameter("x", bounds=(0., 1.))], objectives=["a", "b"]))
    with pytest.raises(EngineError):
        campaign.to_state()
    with pytest.raises(EngineError):
        campaign.tell_batch(ObservationBatch(pd.DataFrame({"x": [0.]}), pd.DataFrame({"a": [1.], "b": [2.]})))


@pytest.fixture
def schema_20_fixture():
    # Actual 0.5.0 wire fixture, not generated by the reader/writer under test.
    return {
        "schema_version": "2.0", "engine_version": "0.5.0",
        "space": {"schema_version": "1.0", "engine_version": "0.5.0",
            "params": [{"name": "x", "kind": "continuous", "bounds": [0., 1.],
                "values": None, "step": None, "unit": "", "transform": "linear"}],
            "constraints": [], "objectives": [{"name": "yield", "direction": "maximize",
                "target": None, "unit": "", "priority": 0}], "outcome_constraints": []},
        "knowledge": {"schema_version": "1.0", "engine_version": "0.5.0",
            "strict": False, "specs": [], "providers": []},
        "seed": 7, "history": [{"x": 0.25, "y": 1.5, "kind": "doe"}], "trial_kind": ["doe"],
    }


def test_actual_050_schema20_remains_readable(schema_20_fixture):
    restored = Campaign.from_state(schema_20_fixture)
    assert restored.history_df().y.tolist() == [1.5]
    assert restored.seed == 7
    assert restored.to_state()["schema_version"] == "2.1"
    assert restored.to_state()["engine_version"] == "0.5.1"
    assert restored.to_state()["space"]["engine_version"] == "0.5.1"
    assert restored.to_state()["knowledge"]["engine_version"] == "0.5.1"
    assert schema_20_fixture["engine_version"] == "0.5.0"
    assert schema_20_fixture["space"]["engine_version"] == "0.5.0"
    assert schema_20_fixture["knowledge"]["engine_version"] == "0.5.0"


def test_knowledge_050_explicit_compatibility_survives_reader_patch_bump(schema_20_fixture, monkeypatch):
    import expdoe_dk.knowledge as module
    monkeypatch.setattr(module, "ENGINE_VERSION", "0.5.1")
    assert Knowledge.from_envelope(schema_20_fixture["knowledge"]).specs == ()


def test_real_050_campaign_restores_with_patch_version_domain_reader(schema_20_fixture, monkeypatch):
    import expdoe_dk.domain.space as domain_module
    import expdoe_dk.knowledge as knowledge_module
    monkeypatch.setattr(domain_module, "ENGINE_VERSION", "0.5.1")
    monkeypatch.setattr(knowledge_module, "ENGINE_VERSION", "0.5.1")
    restored = Campaign.from_state(schema_20_fixture)
    assert restored.history_df().y.tolist() == [1.5]
    assert restored.space.objectives == ["yield"]
    assert schema_20_fixture["space"]["engine_version"] == "0.5.0"
    rewritten = restored.space.to_dict()
    assert {key: value for key, value in rewritten.items() if key != "engine_version"} == {
        key: value for key, value in schema_20_fixture["space"].items() if key != "engine_version"}
    assert Space.from_dict(rewritten).to_dict() == rewritten


@pytest.mark.parametrize("path, value", [
    (("engine_version",), "0.5.99"),
    (("knowledge", "engine_version"), "0.5.99"),
    (("space", "engine_version"), "0.5.99"),
    (("seed",), True), (("seed",), -1),
    (("rng_state", "bit_generator"), "evil.module:load"),
    (("rng_state", "state", "state"), -1),
    (("ask_iteration",), -1),
    (("pending", "ids"), ["duplicate", "duplicate"]),
    (("observations", 0, "Yvar"), [[-1.]] * 5),
    (("observations", 0, "status"), ["unknown"] * 5),
    (("observations", 0, "ids"), ["duplicate"] * 5),
    (("observations", 0, "Y"), [[None]] * 5),
    (("knowledge", "specs", 0, "parameters"), {"lengthscale": "invalid"}),
    (("knowledge", "specs", 0, "version"), "99.0"),
    (("knowledge", "specs", 0, "enabled"), "yes"),
    (("space", "params", 0, "bounds"), "invalid"),
])
def test_malformed_state_is_typed(populated_campaign, path, value):
    state = populated_campaign.to_state()
    node = state
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(EngineError):
        Campaign.from_state(state)


def test_state_preflight_checks_late_fields_before_domain_construction(populated_campaign, monkeypatch):
    state = populated_campaign.to_state()
    state["pending"]["ids"] = [False, "p"]
    def forbidden(*args, **kwargs):
        pytest.fail("Domain constructor ran before complete declarative preflight")
    monkeypatch.setattr(Space, "from_dict", forbidden)
    monkeypatch.setattr(Knowledge, "from_envelope", forbidden)
    with pytest.raises(EngineError):
        Campaign.from_state(state)


@pytest.mark.parametrize("operation", ["fsync", "replace"])
def test_atomic_write_failure_preserves_previous_file(populated_campaign, tmp_path, monkeypatch, operation):
    import expdoe_dk.bo.checkpoint as checkpoint
    path = tmp_path / "saved.json"
    populated_campaign.save_checkpoint(path)
    before = path.read_bytes()
    populated_campaign.set_pending(None)
    def fail(*args, **kwargs):
        raise OSError("injected write failure")
    monkeypatch.setattr(checkpoint.os, operation, fail)
    with pytest.raises(OSError, match="injected"):
        populated_campaign.save_checkpoint(path)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_json_read_rejects_nonfinite_and_duplicate_keys(tmp_path):
    path = tmp_path / "bad.json"
    for text in ('{"seed": NaN}', '{"seed": 1, "seed": 2}', '[]'):
        path.write_text(text)
        with pytest.raises(EngineError):
            Campaign.load_checkpoint(path)


def test_pending_exact_integer_survives_mixed_numeric_columns():
    base = 2**53
    campaign = Campaign(Space([Parameter("count", kind="integer", bounds=(base, base + 8)),
        Parameter("fraction", bounds=(0., 1.))]))
    campaign.set_pending(PendingBatch(("exact",), pd.DataFrame({"count": [base + 1], "fraction": [0.5]})))
    state = campaign.to_state()
    assert state["pending"]["X"] == [[base + 1, 0.5]]
    assert type(state["pending"]["X"][0][0]) is int
    assert Campaign.from_state(state).to_state() == state


@pytest.fixture
def opaque_provider_campaign():
    envelope = Knowledge().add(make_pattern_spec(pattern="custom", version="3.0",
        parameters={"strength": 0.5}, scope=KnowledgeScope(), confidence=0.8)).to_envelope()
    envelope["providers"] = [{"distribution_name": "example", "canonical_distribution_name": "example",
        "distribution_version": "2.4.1", "entry_point_name": "example",
        "definitions": [{"pattern": "custom", "version": "3.0", "schema_digest": "a" * 64}]}]
    campaign = Campaign(Space([Parameter("x", bounds=(0., 1.))]), Knowledge.from_envelope(envelope))
    campaign.tell(pd.DataFrame({"x": [0.5]}), [1.])
    return campaign


def test_legacy_provider_archive_is_explicit_and_preserves_provenance(opaque_provider_campaign, tmp_path):
    campaign = opaque_provider_campaign
    with pytest.raises(EngineError, match="provider"):
        campaign.to_state()
    path = tmp_path / "legacy.json"
    with pytest.warns(UserWarning, match="archival schema 2.0"):
        campaign.save_checkpoint(path)
    assert json.loads(path.read_text())["schema_version"] == "2.0"
    restored = Campaign.load_checkpoint(path)
    assert restored.knowledge.to_envelope() == campaign.knowledge.to_envelope()
    pd.testing.assert_frame_equal(restored.history_df(), campaign.history_df())


@pytest.mark.parametrize("metadata", ["pending", "variance", "rng", "validation"])
def test_legacy_archive_refuses_lossy_downgrade(opaque_provider_campaign, tmp_path, metadata):
    campaign = opaque_provider_campaign
    if metadata == "pending":
        campaign.set_pending(PendingBatch(("p",), pd.DataFrame({"x": [0.7]})))
    elif metadata == "variance":
        campaign.tell_batch(ObservationBatch(pd.DataFrame({"x": [0.8]}), pd.DataFrame({"y": [2.]}),
            Yvar=pd.DataFrame({"y": [0.1]})))
    elif metadata == "rng":
        campaign._rng.random()
    else:
        campaign.validate_active = False
    with pytest.raises(EngineError, match="downgrade"):
        campaign.save_checkpoint(tmp_path / "legacy.json")
    assert not list(tmp_path.iterdir())


def test_new_state_refuses_enabled_external_provider_version(populated_campaign, opaque_provider_campaign):
    state = populated_campaign.to_state()
    state["knowledge"] = opaque_provider_campaign.knowledge.to_envelope()
    with pytest.raises(EngineError, match="provider"):
        Campaign.from_state(state)


def test_legacy_multiobjective_archive_still_roundtrips(tmp_path):
    with pytest.warns(UserWarning):
        campaign = Campaign(Space([Parameter("x", bounds=(0., 1.))], objectives=["a", "b"]))
    campaign.tell(pd.DataFrame({"x": [0.5]}), [2.])
    path = tmp_path / "legacy.json"
    with pytest.warns(UserWarning, match="archival schema 2.0"):
        campaign.save_checkpoint(path)
    with pytest.warns(UserWarning):
        restored = Campaign.load_checkpoint(path)
    assert restored.space.objectives == ["a", "b"]
    assert restored.finalize().best_y_physical == 2.
    with pytest.raises(EngineError):
        restored.to_state()


def test_mixed_variance_is_persisted_but_never_silently_dropped_at_fit():
    campaign = Campaign(Space([Parameter("x", kind="integer", bounds=(0, 12))]))
    campaign.tell(pd.DataFrame({"x": [0, 1, 2]}), [1., 2., 3.])
    campaign.tell_batch(ObservationBatch(pd.DataFrame({"x": [3]}), pd.DataFrame({"y": [4.]}),
        Yvar=pd.DataFrame({"y": [0.25]}), ids=("measured",)))
    state = campaign.to_state()
    restored = Campaign.from_state(state)
    assert restored.to_state() == state
    with pytest.raises(EngineError, match="Mixed missing variance"):
        restored.ask()


def test_duplicate_observation_ids_across_tells_do_not_mutate_history(populated_campaign):
    state = populated_campaign.to_state()
    with pytest.raises(EngineError, match="IDs"):
        populated_campaign.tell_batch(ObservationBatch(pd.DataFrame({"x": [1]}),
            pd.DataFrame({"yield": [8.]}), ids=("o0",)))
    assert populated_campaign.to_state() == state


@pytest.mark.parametrize("field,value", [("engine_version", []), ("engine_version", None)])
def test_invalid_knowledge_version_type_is_typed(field, value):
    envelope = Knowledge().to_envelope()
    envelope[field] = value
    with pytest.raises(EngineError):
        Knowledge.from_envelope(envelope)


@pytest.mark.parametrize("location", ["space", "knowledge"])
def test_malicious_expression_is_rejected_before_domain_construction(populated_campaign, monkeypatch, location):
    state = populated_campaign.to_state()
    bad_ast = {"type": "call", "function": "eval", "args": []}
    if location == "space":
        state["space"]["constraints"] = [{"name": "bad", "kind": "expression", "hard": True, "ast": bad_ast}]
    else:
        state["knowledge"]["specs"][0].update(pattern="safe_region", parameters={
            "constraint": {"kind": "expression", "hard": True, "ast": bad_ast}})
    def forbidden(*args, **kwargs):
        pytest.fail("Malformed expression reached a domain constructor")
    monkeypatch.setattr(Space, "from_dict", forbidden)
    with pytest.raises(EngineError):
        Campaign.from_state(state)


def test_fractional_encoding_of_iteration_is_not_accepted(populated_campaign):
    state = populated_campaign.to_state()
    state["ask_iteration"] = 8.0
    with pytest.raises(EngineError):
        Campaign.from_state(state)


def test_finalize_keeps_exact_integer_physical_condition():
    base = 2**53
    campaign = Campaign(Space([Parameter("count", kind="integer", bounds=(base, base + 8)),
        Parameter("fraction", bounds=(0., 1.))]))
    campaign.tell(pd.DataFrame({"count": [base + 1], "fraction": [0.5]}), [1.5])
    restored = Campaign.from_state(campaign.to_state())
    assert restored.finalize().best_x_physical == {"count": base + 1, "fraction": 0.5}


def test_explicit_empty_batch_with_known_variance_roundtrips():
    campaign = Campaign(Space([Parameter("x", bounds=(0., 1.))]))
    campaign.tell_batch(ObservationBatch(pd.DataFrame({"x": pd.Series(dtype=float)}),
        pd.DataFrame({"y": pd.Series(dtype=float)}), Yvar=pd.DataFrame({"y": pd.Series(dtype=float)})))
    state = campaign.to_state()
    assert Campaign.from_state(state).to_state() == state


def test_v04_reader_keeps_optional_trial_kind_migration():
    state = {"space": {"params": [{"name": "x", "bounds": [0., 1.]}]},
        "knowledge": {}, "seed": 0, "history": [{"x": 0.5, "y": 1.}]}
    campaign = Campaign.from_state(state)
    assert campaign.history_df().to_dict("records") == [{"x": 0.5, "y": 1., "kind": "doe"}]


@pytest.mark.parametrize("boundary", ["state", "file"])
@pytest.mark.parametrize("exact,dtype", [(2**53 + 1, "Int64"), (2**63 + 1, "UInt64"), (2**80 + 1, object)])
def test_nullable_integer_outcomes_keep_exact_physical_values(tmp_path, boundary, exact, dtype):
    campaign = Campaign(Space([Parameter("x", bounds=(0., 1.))]))
    campaign.tell_batch(ObservationBatch(
        pd.DataFrame({"x": [0., 1.]}),
        pd.DataFrame({"y": pd.Series([exact, pd.NA], dtype=dtype)}),
        status=pd.Series(["success", "failed"]),
    ))
    state = campaign.to_state()
    assert state["observations"][0]["Y"] == [[exact], [None]]
    if boundary == "state":
        restored = Campaign.from_state(state)
    else:
        path = tmp_path / "nullable-integer.json"
        campaign.save_checkpoint(path)
        restored = Campaign.load_checkpoint(path)
    assert restored.to_state()["observations"][0]["Y"] == [[exact], [None]]
    assert restored.to_state() == state


@pytest.mark.parametrize("kind", [[], {}])
@pytest.mark.parametrize("boundary", ["state", "file"])
def test_legacy_knowledge_discriminator_requires_string(tmp_path, kind, boundary):
    state = {"space": {"params": [{"name": "x", "bounds": [0., 1.]}]},
        "knowledge": {"items": [{"kind": kind}]}, "seed": 0, "history": []}
    with pytest.raises(EngineError) as caught:
        if boundary == "state":
            Campaign.from_state(state)
        else:
            path = tmp_path / "bad-legacy.json"
            path.write_text(json.dumps(state))
            Campaign.load_checkpoint(path)
    assert caught.value.code is ErrorCode.CHECKPOINT_INCOMPATIBLE
