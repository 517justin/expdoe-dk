"""Declarative Campaign checkpoints; no executable payloads or provider loading.

2.1 persists physical batches and PCG64 state. 2.0 and unversioned v0.4 are
archival readers (and a legacy-only writer for multiobjective/provider archives).
All JSON structure is checked before constructing Space, Knowledge or Campaign.
Domain readers then apply their semantic invariants before a Campaign is made.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import tempfile

from jsonschema import Draft202012Validator, ValidationError, validators
import pandas as pd

from ..domain import ObservationBatch, PendingBatch, Space
from ..domain.constraints import _AstBudget, _canonical_node
from ..errors import EngineError, ErrorCode
from ..knowledge import Knowledge, COMPATIBLE_ENGINE_VERSIONS
from ..knowledge.patterns import builtin_pattern_definitions

CHECKPOINT_SCHEMA_VERSION = "2.1"
_Validator = validators.extend(Draft202012Validator, type_checker=
    Draft202012Validator.TYPE_CHECKER.redefine("integer", lambda checker, value: type(value) is int))


def _object(properties, *, optional=()):
    return {"type": "object", "properties": properties,
            "required": [key for key in properties if key not in optional],
            "additionalProperties": False}


def _array(items, **kwargs):
    return {"type": "array", "items": items, **kwargs}


_STRING = {"type": "string"}
_NAME = {"type": "string", "minLength": 1}
_NUMBER = {"type": "number"}
_BOOL = {"type": "boolean"}
_INTEGER = {"type": "integer", "minimum": 0}
_SCALAR = {"type": ["string", "number", "boolean", "null"]}
_PAIR = _array(_NUMBER, minItems=2, maxItems=2)
_NAMES = _array(_NAME, uniqueItems=True)
_NULL = {"type": "null"}


def _nullable(schema):
    return {"anyOf": [schema, _NULL]}


_PARAMETER = _object({
    "name": _NAME, "kind": {"enum": ["continuous", "integer", "discrete", "categorical", "ordinal"]},
    "bounds": _nullable(_PAIR), "values": _nullable(_array(_SCALAR)),
    "step": _nullable(_NUMBER), "unit": _STRING, "transform": {"enum": ["linear", "log"]},
})
_OBJECTIVE = _object({"name": _NAME, "direction": {"enum": ["maximize", "minimize", "target"]},
    "target": {"anyOf": [_NUMBER, _PAIR, _NULL]}, "unit": _STRING, "priority": _INTEGER})
_COEFFICIENTS = {"type": "object", "minProperties": 1, "additionalProperties": _NUMBER}


def _constraint_schemas():
    variants = []
    for hard in (True, False):
        common = {"name": _NAME, "hard": {"const": hard}}
        if not hard:
            common.update(penalty=_NAME, weight={"type": "number", "exclusiveMinimum": 0})
        for kind, fields in {
            "linear": {"coefficients": _COEFFICIENTS, "operator": {"enum": ["<=", ">=", "=="]}, "bound": _NUMBER},
            "expression": {"ast": {"type": "object"}},
            "outcome": {"objective": _NAME, "operator": {"enum": ["<=", ">=", "=="]}, "bound": _NUMBER},
        }.items():
            variants.append(_object({**common, "kind": {"const": kind}, **fields}))
        for field in ("allowed", "forbidden"):
            variants.append(_object({**common, "kind": {"const": "categorical_combination"},
                field: _array({"type": "object", "minProperties": 1, "additionalProperties": _SCALAR}, minItems=1)}))
    variants.append(_object({"kind": {"const": "legacy_linear"}, "name": _STRING,
        "coeffs": _COEFFICIENTS, "lower": _nullable(_NUMBER), "upper": _nullable(_NUMBER)}))
    return {"anyOf": variants}


_SPACE = _object({"schema_version": {"const": "1.0"},
    "engine_version": {"enum": list(COMPATIBLE_ENGINE_VERSIONS)},
    "params": _array(_PARAMETER, minItems=1), "objectives": _array(_OBJECTIVE, minItems=1),
    "constraints": _array(_constraint_schemas()), "outcome_constraints": _array(_constraint_schemas())})
_SCOPE = _object({"factors": _NAMES, "objectives": _NAMES,
    "conditions": _nullable({"type": "object"}), "region": _nullable({"type": "object"})})
_SPEC = _object({"pattern_id": _NAME, "pattern": _NAME, "version": _NAME,
    "parameters": {"type": "object"}, "scope": _SCOPE,
    "confidence": {"type": "number", "minimum": 0, "maximum": 1}, "enabled": _BOOL,
    "evidence": _array(_object({"kind": _NAME, "reference": _NAME}))}, optional=("pattern_id",))
_PROVIDER = _object({"distribution_name": _NAME, "canonical_distribution_name": _NAME,
    "entry_point_name": _NAME, "distribution_version": _NAME,
    "definitions": _array(_object({"pattern": _NAME, "version": _NAME,
        "schema_digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"}}))})
_KNOWLEDGE = _object({"schema_version": {"const": "1.0"},
    "engine_version": {"enum": list(COMPATIBLE_ENGINE_VERSIONS)},
    "strict": _BOOL, "specs": _array(_SPEC), "providers": _array(_PROVIDER)})
_MATRIX = _array(_array(_SCALAR))
_BATCH = _object({"X": _MATRIX, "Y": _MATRIX, "Yvar": _nullable(_array(_array({"type": "number", "minimum": 0}))),
    "status": _nullable(_array({"enum": ["success", "warning", "failed"]})), "ids": _NAMES})
_PENDING = _nullable(_object({"X": _MATRIX, "ids": _NAMES}))
_RNG = _object({"bit_generator": {"const": "PCG64"},
    "state": _object({"state": {"type": "integer", "minimum": 0, "maximum": 2**128 - 1},
        "inc": {"type": "integer", "minimum": 1, "maximum": 2**128 - 1}}),
    "has_uint32": {"enum": [0, 1]}, "uinteger": {"type": "integer", "minimum": 0, "maximum": 2**32 - 1}})
_VALIDATION = _object({"active": _BOOL, "interval": {"type": "integer", "minimum": 1},
    "min_obs": {"type": "integer", "minimum": 3}, "auto_rescue": _BOOL,
    "last_n": _INTEGER, "emitted": _array(_array(_STRING, minItems=2, maxItems=2), uniqueItems=True)})
_COMMON = {"schema_version": {"enum": ["2.0", "2.1"]},
    "engine_version": {"enum": list(COMPATIBLE_ENGINE_VERSIONS)}, "space": _SPACE,
    "knowledge": _KNOWLEDGE, "seed": _INTEGER, "trial_kind": _array({"enum": ["doe", "bo"]})}
_STATE = _object({**_COMMON, "observations": _array(_BATCH), "pending": _PENDING,
    "rng_state": _RNG, "ask_iteration": _INTEGER, "validation": _VALIDATION})
_LEGACY = _object({**_COMMON, "history": _array({"type": "object"})})


def _invalid(message):
    return EngineError(ErrorCode.CHECKPOINT_INCOMPATIBLE, message)


def _strict_json(value):
    """Reject objects with conversion hooks, non-finite values and non-string keys."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _strict_json(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _strict_json(item)
        return
    raise _invalid("State must contain only finite built-in JSON values")


def _validate_schema(value, schema):
    try:
        _Validator(schema).validate(value)
    except ValidationError as error:
        raise _invalid(f"Invalid checkpoint field {list(error.absolute_path)!r}: {error.message}") from error


def _validate_knowledge(payload, *, archival):
    definitions = {(d.pattern, d.version): d for d in builtin_pattern_definitions()}
    ids = [s.get("pattern_id") for s in payload["specs"] if "pattern_id" in s]
    if len(set(ids)) != len(ids):
        raise _invalid("Duplicate knowledge pattern IDs")
    provider_keys = {(d["pattern"], d["version"]) for p in payload["providers"] for d in p["definitions"]}
    for spec in payload["specs"]:
        key = (spec["pattern"], spec["version"])
        definition = definitions.get(key)
        if definition is not None:
            _validate_schema(spec["parameters"], dict(definition.schema))
        elif spec["enabled"] and (not archival or key not in provider_keys):
            raise _invalid(f"Unsupported enabled knowledge/provider version {key!r}; explicit provider reload required")


def _check_rows(batch, names, objectives):
    n = len(batch["X"])
    if len(batch["Y"]) != n or any(len(row) != len(names) for row in batch["X"]) or any(len(row) != len(objectives) for row in batch["Y"]):
        raise _invalid("Observation matrix dimensions do not match declared space")
    for key in ("Yvar", "status"):
        if batch[key] is not None and len(batch[key]) != n:
            raise _invalid(f"Observation {key} length does not match X")
    if batch["ids"] and len(batch["ids"]) != n:
        raise _invalid("Observation ids length does not match X")
    if batch["Yvar"] is not None and any(len(row) != len(objectives) for row in batch["Yvar"]):
        raise _invalid("Observation variance columns do not match objectives")
    status = batch["status"] or ["success"] * n
    for row, flag in zip(batch["Y"], status):
        if flag == "success" and any(type(value) not in (float, int) for value in row):
            raise _invalid("Successful objective values must be finite non-boolean numbers")


def validate_state(state):
    """Validate all declarative fields first; return an isolated wire payload."""
    _strict_json(state)
    if type(state) is not dict:
        raise _invalid("Campaign checkpoint must be a JSON object")
    state = deepcopy(state)
    schema = state.get("schema_version")
    if schema not in (None, "2.0", "2.1"):
        raise _invalid(f"Unsupported Campaign schema version {schema!r}")
    if schema is None:
        _validate_v04(state)
        return state
    _validate_schema(state, _STATE if schema == "2.1" else _LEGACY)
    _validate_knowledge(state["knowledge"], archival=schema == "2.0")
    names = [p["name"] for p in state["space"]["params"]]
    objectives = [o["name"] for o in state["space"]["objectives"]]
    # Reuse the bounded, non-evaluating AST validator before any domain object
    # construction. JSON Schema alone cannot enforce the interpreter allowlist.
    for constraint in (*state["space"]["constraints"], *state["space"]["outcome_constraints"]):
        if constraint["kind"] == "expression":
            _canonical_node(constraint["ast"], frozenset(names), 1, _AstBudget())
    for spec in state["knowledge"]["specs"]:
        if (spec["pattern"], spec["version"]) in {("safe_region", "1.0"), ("forbidden_region", "1.0")}:
            _canonical_node(spec["parameters"]["constraint"]["ast"],
                frozenset(spec["scope"]["factors"] or names), 1, _AstBudget())
    if schema == "2.1":
        if len(objectives) != 1:
            raise _invalid("Phase-1 Campaign state requires exactly one objective")
        count = 0
        all_ids = []
        for batch in state["observations"]:
            _check_rows(batch, names, objectives)
            count += (batch["status"] or ["success"] * len(batch["X"])).count("success")
            all_ids.extend(batch["ids"])
        if len(set(all_ids)) != len(all_ids):
            raise _invalid("Observation IDs must be unique across batches")
        pending = state["pending"]
        if pending is not None and (len(pending["ids"]) != len(pending["X"]) or any(len(row) != len(names) for row in pending["X"])):
            raise _invalid("Pending matrix dimensions/ids do not match space")
        if state["rng_state"]["state"]["inc"] % 2 != 1:
            raise _invalid("PCG64 increment must be odd")
        if state["validation"]["last_n"] > count:
            raise _invalid("Validation progress exceeds successful observation count")
    else:
        count = _validate_history(state, names)
    if len(state["trial_kind"]) != count:
        raise _invalid("trial_kind length does not match successful history")
    return state


def _validate_history(state, names):
    for row in state["history"]:
        _validate_schema(row, _object({**{name: _SCALAR for name in names}, "y": _NUMBER, "kind": {"enum": ["doe", "bo"]}}))
    if [row["kind"] for row in state["history"]] != state["trial_kind"]:
        raise _invalid("History kind and trial_kind disagree")
    return len(state["history"])


def _validate_v04(state):
    parameter = _object({"name": _NAME, "bounds": _PAIR, "unit": _STRING,
        "kind": {"enum": ["continuous", "discrete", "integer"]}, "step": _nullable(_NUMBER), "log_scale": _BOOL},
        optional=("unit", "kind", "step", "log_scale"))
    space = _object({"params": _array(parameter, minItems=1), "constraints": _array(_object({
        "coeffs": _COEFFICIENTS, "lower": _nullable(_NUMBER), "upper": _nullable(_NUMBER), "name": _STRING},
        optional=("lower", "upper", "name"))), "objectives": {"anyOf": [_NAME, _NAMES]},
        "maximize": {"anyOf": [_BOOL, _array(_BOOL)]}}, optional=("constraints", "objectives", "maximize"))
    knowledge = _object({"strict": _BOOL, "items": _array({"type": "object"})}, optional=("strict", "items"))
    _validate_schema(state, _object({"space": space, "knowledge": knowledge, "seed": _INTEGER,
        "history": _array({"type": "object"}), "trial_kind": _array({"enum": ["doe", "bo"]})},
        optional=("trial_kind",)))
    state.setdefault("trial_kind", ["doe"] * len(state["history"]))
    if len(state["trial_kind"]) != len(state["history"]):
        raise _invalid("Legacy trial_kind length does not match history")
    for row, kind in zip(state["history"], state["trial_kind"]):
        row.setdefault("kind", kind)
    legacy_items = {
        "arrhenius": {"param": _NAME, "frozen": _BOOL, "activation_energy": _NUMBER, "amplitude_init": _NUMBER},
        "quadratic_peak": {"param": _NAME, "center": _NUMBER, "direction": {"enum": ["peak", "valley"]}, "frozen": _BOOL},
        "monotone": {"param": _NAME, "effect": {"enum": ["increases_objective", "decreases_objective"]},
            "n_pairs_per_dim": {"type": "integer", "minimum": 1}, "epsilon": {"anyOf": [_NUMBER, {"const": "auto"}]}, "delta_norm": _NUMBER},
        "random_augment": {"n": {"type": "integer", "minimum": 0}},
        "gp_prior": {"lengthscale": {"enum": ["weak", "medium", "strong"]}},
    }
    required = {"arrhenius": ("param",), "quadratic_peak": ("param", "center"), "monotone": ("param", "effect")}
    for item in state["knowledge"].get("items", []):
        kind = item.get("kind")
        if type(kind) is not str or kind not in legacy_items:
            raise _invalid(f"Unknown legacy knowledge kind {kind!r}")
        fields = legacy_items[kind]
        _validate_schema(item, _object({"kind": {"const": kind}, **fields},
            optional=set(fields) - set(required.get(kind, ()))))
    count = _validate_history(state, [p["name"] for p in state["space"]["params"]])
    if len(state["trial_kind"]) != count:
        raise _invalid("Legacy trial_kind length does not match history")


def batch_to_state(batch):
    def matrix(frame):
        if frame is None:
            return None
        # Missing unsuccessful outcomes are represented as JSON null, never NaN.
        return frame.astype(object).where(pd.notna(frame), None).values.tolist()
    return {"X": matrix(batch.X), "Y": matrix(batch.Y), "Yvar": matrix(batch.Yvar),
        "status": None if batch.status is None else batch.status.tolist(), "ids": list(batch.ids)}


def batch_from_state(raw, space):
    # Object dtype prevents initial numeric inference; pd.NA also prevents the
    # ObservationBatch object-column deepcopy/map from inferring float64 for
    # integers plus missing values. JSON serialization maps pd.NA back to null.
    outcomes = pd.DataFrame(
        [[pd.NA if value is None else value for value in row] for row in raw["Y"]],
        columns=space.objectives, dtype=object,
    )
    return ObservationBatch(pd.DataFrame(raw["X"], columns=space.param_names),
        outcomes,
        Yvar=None if raw["Yvar"] is None else pd.DataFrame(raw["Yvar"], columns=space.objectives, dtype=float),
        status=None if raw["status"] is None else pd.Series(raw["status"], dtype=object), ids=raw["ids"])


def restore(state, campaign_class):
    state = validate_state(state)
    try:
        # Migration seam for a patch reader: preflight has already checked the
        # outer checkpoint AND both nested envelopes against the exact table.
        # Normalize only the Space reader's version tag, never semantic fields.
        from ..domain.space import ENGINE_VERSION as reader_version
        space_payload = deepcopy(state["space"])
        if "schema_version" in space_payload:
            space_payload["engine_version"] = reader_version
        space = Space.from_dict(space_payload)
        knowledge = (Knowledge.from_dict(state["knowledge"]) if "schema_version" not in state
                     else Knowledge.from_envelope(state["knowledge"]))
        if state.get("schema_version") != "2.1":
            campaign = campaign_class(space, knowledge, seed=state["seed"])
            history = pd.DataFrame(state["history"])
            if not history.empty:
                campaign.tell(history[space.param_names], history["y"].to_numpy())
                campaign._trial_kind = state["trial_kind"]
            return campaign
        batches = [batch_from_state(batch, space) for batch in state["observations"]]
        raw_pending = state["pending"]
        pending = None if raw_pending is None else PendingBatch(raw_pending["ids"],
            pd.DataFrame(raw_pending["X"], columns=space.param_names))
        for batch in batches:
            validate_frame(space, batch.X)
        if pending is not None:
            validate_frame(space, pending.X)
        validation = state["validation"]
        campaign = campaign_class(space, knowledge, seed=state["seed"], validate=False,
            validation_interval=validation["interval"], validation_min_obs=validation["min_obs"],
            auto_rescue=validation["auto_rescue"])
        for batch in batches:
            campaign.tell_batch(batch)
        campaign._trial_kind = state["trial_kind"]
        campaign.set_pending(pending)
        campaign._rng.bit_generator.state = state["rng_state"]
        campaign._ask_iteration = state["ask_iteration"]
        campaign.validate_active = validation["active"]
        campaign._last_validation_n = validation["last_n"]
        campaign._validation_emitted = {tuple(item) for item in validation["emitted"]}
        return campaign
    except EngineError:
        raise
    except (TypeError, ValueError, KeyError, OverflowError) as error:
        raise _invalid(f"Invalid Campaign state: {error}") from error


def validate_frame(space, frame):
    if list(frame.columns) != list(space.param_names):
        raise EngineError(ErrorCode.CONFIG_INVALID, "Physical columns must exactly match space.param_names")
    try:
        space.physical_to_model(frame)
    except (TypeError, ValueError, KeyError, OverflowError) as error:
        raise EngineError(ErrorCode.CONFIG_INVALID, f"Invalid physical conditions: {error}") from error


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise _invalid(f"Duplicate JSON key {key!r}")
            result[key] = value
        return result
    def constant(value):
        raise _invalid(f"Non-finite JSON value {value!r}")
    try:
        return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, UnicodeError) as error:
        raise _invalid(f"Invalid checkpoint JSON: {error}") from error


def atomic_save(path, state):
    """Durable temporary sibling, strict readback validation, then atomic replace.

    No application lock is acquired. Any failure before replace preserves the
    existing target and removes the temporary sibling.
    """
    state = validate_state(state)
    encoded = json.dumps(state, indent=2, allow_nan=False)
    target = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                prefix=f".{target.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if validate_state(read_json(temporary)) != state:
            raise _invalid("Written checkpoint did not validate identically")
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
