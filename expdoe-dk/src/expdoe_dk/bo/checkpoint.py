"""Declarative Campaign checkpoints; no executable payloads or provider loading.

2.2 persists multiobjective campaigns with lossless physical frame metadata.
2.1 persists scalar batches and PCG64 state. 2.0 and unversioned v0.4 are
archival readers (and a legacy-only writer for opaque provider archives).
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
import numpy as np
import pandas as pd

from ..domain import ObservationBatch, PendingBatch, Space
from ..domain.constraints import _AstBudget, _canonical_node
from ..errors import EngineError, ErrorCode
from ..knowledge import Knowledge, COMPATIBLE_ENGINE_VERSIONS
from ..knowledge.patterns import builtin_pattern_definitions

CHECKPOINT_SCHEMA_VERSION = "2.1"
MULTIOBJECTIVE_SCHEMA_VERSION = "2.2"
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

# These are finite allowlists, never dtype names interpreted from arbitrary input.
_NUMPY_DTYPES = {name: np.dtype(name) for name in (
    "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    "float16", "float32", "float64")}
_FRAME_DTYPES = {"object": np.dtype(object), **_NUMPY_DTYPES,
    **{name: cls() for name, cls in (
        ("Int8", pd.Int8Dtype), ("Int16", pd.Int16Dtype), ("Int32", pd.Int32Dtype),
        ("Int64", pd.Int64Dtype), ("UInt8", pd.UInt8Dtype), ("UInt16", pd.UInt16Dtype),
        ("UInt32", pd.UInt32Dtype), ("UInt64", pd.UInt64Dtype),
        ("Float32", pd.Float32Dtype), ("Float64", pd.Float64Dtype), ("boolean", pd.BooleanDtype))}}
_STRING_DTYPES = {f"string[{storage}]:{missing}": (storage, missing)
                  for storage in ("python", "pyarrow") for missing in ("NA", "nan")}
_MISSING = _object({"missing": {"enum": ["NA", "nan", "+inf", "-inf"]}})
_CELL = {"anyOf": [_SCALAR, _MISSING, _object({"scalar": {"const": "numpy"},
    "dtype": {"enum": list(_NUMPY_DTYPES)}, "value": {"anyOf": [_SCALAR, _MISSING]}})]}
_FRAME = _object({"columns": _NAMES,
    "dtypes": _array({"enum": [*_FRAME_DTYPES, *_STRING_DTYPES]}), "data": _array(_array(_CELL))})
_MULTI_BATCH = _object({"X": _FRAME, "Y": _FRAME, "Yvar": _nullable(_FRAME),
    "status": _BATCH["properties"]["status"], "ids": _NAMES})
_CONFIG = _object({
    "acquisition": {"enum": ["auto", "q_log_ehvi", "q_log_nehvi", "q_log_nparego"]},
    "noise_mode": {"enum": ["auto", "inferred", "known", "noiseless"]},
    "reference_point": _nullable({"type": "object", "additionalProperties": _NUMBER}),
    "preference_weights": _nullable({"type": "object", "additionalProperties": _NUMBER}),
    "min_joint_feasibility": {"type": "number", "minimum": 0, "maximum": 1}})
_GEOMETRY = _object({"objective_names": _NAMES,
    **{name: _array(_NUMBER) for name in ("location", "scale", "report_reference",
        "acquisition_reference", "acquisition_ideal")},
    "source_digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    "source_row_count": {"type": "integer", "minimum": 3},
    "preference_weights": _nullable(_array(_NUMBER)),
    "preference_provenance": _array({"anyOf": [
        _object({"source": {"enum": ["config", "objective_priority"]}}),
        _object({"source": {"const": "knowledge"}, "source_pattern_id": _NAME,
                 "source_pattern": _NAME, "source_version": _NAME})]}),
    "physical_reference": _CONFIG["properties"]["reference_point"],
    "reference_method": {"enum": ["physical", "observed_feasible_min_minus_margin"]}})
_DIAGNOSTIC = _object({"objective_name": _NAME, "pattern_id": _NAME, "pattern": _NAME,
    "status": {"enum": ["checked", "not_checked", "unsupported"]},
    "violation": _nullable(_BOOL), "reason": _nullable(_NAME)})
_MULTI_STATE = _object({**_COMMON, "schema_version": {"const": MULTIOBJECTIVE_SCHEMA_VERSION},
    # Task 10 owns the release allowlist expansion; 2.2 has no historical writer.
    "engine_version": {"const": "0.5.1"},
    "observations": _array(_MULTI_BATCH), "observation_ids": _array(_NAMES),
    "batch_kinds": _array({"enum": ["doe", "bo"]}),
    "pending": _nullable(_object({"X": _FRAME, "ids": _NAMES})),
    "catalog": _nullable(_object({"X": _FRAME, "ids": _NAMES})),
    "multiobjective": _CONFIG, "geometry": _nullable(_GEOMETRY),
    "rng_state": _RNG, "ask_iteration": _INTEGER,
    "validation": _object({**_VALIDATION["properties"],
        "emitted": _array(_array(_NAME, minItems=4, maxItems=4), uniqueItems=True),
        "diagnostics": _array(_DIAGNOSTIC)})})


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
    if schema not in (None, "2.0", "2.1", MULTIOBJECTIVE_SCHEMA_VERSION):
        raise _invalid(f"Unsupported Campaign schema version {schema!r}")
    if schema is None:
        _validate_v04(state)
        return state
    _validate_schema(state, _MULTI_STATE if schema == MULTIOBJECTIVE_SCHEMA_VERSION
                     else _STATE if schema == "2.1" else _LEGACY)
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
    if schema == MULTIOBJECTIVE_SCHEMA_VERSION:
        from .loop import Campaign
        _restore_multiobjective(state, Campaign)
        return state
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
    # Preserve scalar identities before constructing immutable batches.
    # JSON serialization maps missing unsuccessful outcomes back to null.
    outcomes = pd.DataFrame(
        [[pd.NA if value is None else value for value in row] for row in raw["Y"]],
        columns=space.objectives, dtype=object,
    )
    return ObservationBatch(pd.DataFrame(raw["X"], columns=space.param_names, dtype=object),
        outcomes,
        Yvar=None if raw["Yvar"] is None else pd.DataFrame(raw["Yvar"], columns=space.objectives, dtype=float),
        status=None if raw["status"] is None else pd.Series(raw["status"], dtype=object), ids=raw["ids"])


def _cell_to_state(value):
    if isinstance(value, np.generic):
        dtype = value.dtype.name
        if dtype not in _NUMPY_DTYPES:
            raise _invalid(f"Unsupported checkpoint scalar dtype {dtype!r}")
        return {"scalar": "numpy", "dtype": dtype, "value": _cell_to_state(value.item())}
    if value is pd.NA:
        return {"missing": "NA"}
    if type(value) is float and not math.isfinite(value):
        return {"missing": "nan" if math.isnan(value) else "+inf" if value > 0 else "-inf"}
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise _invalid(f"Unsupported checkpoint scalar type {type(value).__name__}")


def _cell_from_state(value):
    if type(value) is not dict:
        return value
    if "missing" in value:
        return {"NA": pd.NA, "nan": float("nan"), "+inf": float("inf"), "-inf": -float("inf")}[value["missing"]]
    decoded = _NUMPY_DTYPES[value["dtype"]].type(_cell_from_state(value["value"]))
    if _cell_to_state(decoded) != value:
        raise _invalid("Lossy numpy scalar encoding")
    return decoded


def _dtype_name(dtype):
    if isinstance(dtype, pd.StringDtype):
        missing = "NA" if dtype.na_value is pd.NA else "nan"
        name = f"string[{dtype.storage}]:{missing}"
        if name in _STRING_DTYPES:
            return name
    name = str(dtype)
    if name in _FRAME_DTYPES and dtype == _FRAME_DTYPES[name]:
        return name
    raise _invalid(f"Unsupported checkpoint column dtype {dtype!r}")


def frame_to_state(frame):
    """Encode columns separately: row-wise coercion can round large integers."""
    columns = []
    dtypes = []
    for _, series in frame.items():
        dtype = _dtype_name(series.dtype)
        dtypes.append(dtype)
        columns.append([_cell_to_state(value.item() if dtype != "object"
            and isinstance(value, np.generic) else value) for value in series.tolist()])
    return {"columns": list(frame.columns), "dtypes": dtypes,
            "data": [list(row) for row in zip(*columns)]}


def _frame_from_state(raw, names):
    if (raw["columns"] != list(names) or len(raw["dtypes"]) != len(names)
            or any(len(row) != len(names) for row in raw["data"])):
        raise _invalid("Frame columns/dimensions must exactly match the declared space")
    columns = {}
    for index, (name, dtype_name) in enumerate(zip(names, raw["dtypes"], strict=True)):
        if dtype_name in _STRING_DTYPES:
            storage, missing = _STRING_DTYPES[dtype_name]
            dtype = (pd.StringDtype(storage=storage) if missing == "NA" else
                     pd.StringDtype(storage=storage, na_value=np.nan))
        else:
            dtype = _FRAME_DTYPES[dtype_name]
        columns[name] = pd.Series([_cell_from_state(row[index]) for row in raw["data"]], dtype=dtype)
    frame = pd.DataFrame(columns)
    # Coercible is not lossless: reject rounding, truncation and marker changes.
    if json.dumps(frame_to_state(frame), sort_keys=True) != json.dumps(raw, sort_keys=True):
        raise _invalid("Column dtype changes checkpoint values or missing markers")
    return frame


def multiobjective_batch_to_state(batch):
    return {"X": frame_to_state(batch.X), "Y": frame_to_state(batch.Y),
        "Yvar": None if batch.Yvar is None else frame_to_state(batch.Yvar),
        "status": None if batch.status is None else batch.status.tolist(), "ids": list(batch.ids)}


def multiobjective_to_state(campaign):
    campaign._check_frozen_declaration()
    from ..domain.space import ENGINE_VERSION
    return validate_state({"schema_version": MULTIOBJECTIVE_SCHEMA_VERSION,
        "engine_version": ENGINE_VERSION, "space": campaign.space.to_dict(),
        "knowledge": campaign.knowledge.to_envelope(), "seed": campaign.seed,
        "observations": [multiobjective_batch_to_state(batch) for batch in campaign._observation_batches],
        "observation_ids": [list(ids) for ids in campaign._observation_ids],
        "batch_kinds": list(campaign._batch_kinds), "trial_kind": list(campaign._trial_kind),
        "pending": None if campaign._pending is None else {
            "X": frame_to_state(campaign._pending.X), "ids": list(campaign._pending.ids)},
        "catalog": None if campaign._catalog is None else {
            "X": frame_to_state(campaign._catalog), "ids": list(campaign._catalog_ids)},
        "multiobjective": campaign.multiobjective.to_dict(),
        "geometry": None if campaign._geometry is None else campaign._geometry.to_dict(),
        "rng_state": campaign._rng.bit_generator.state, "ask_iteration": campaign._ask_iteration,
        "validation": {"active": campaign.validate_active, "interval": campaign.validation_interval,
            "min_obs": campaign.validation_min_obs, "auto_rescue": campaign.auto_rescue,
            "last_n": campaign._last_validation_n,
            "emitted": [list(key) for key in sorted(campaign._validation_emitted)],
            "diagnostics": campaign.validation_diagnostics}})


def _restore_multiobjective(state, campaign_class):
    """Semantic validation without fitting, asking, or replaying empirical callbacks."""
    from .multiobjective import MultiObjectiveConfig
    from .objective_geometry import ObjectiveGeometry, resolve_geometry
    from .knowledge_runtime import compile_runtime

    try:
        if any(state[key]["engine_version"] != state["engine_version"] for key in ("space", "knowledge")):
            raise _invalid("Multiobjective engine versions must agree across all envelopes")
        space = Space.from_dict(state["space"])
        knowledge = Knowledge.from_envelope(state["knowledge"])
        config = MultiObjectiveConfig.from_dict(state["multiobjective"])
        config.validate_for_space(space)
        batches = []
        for raw in state["observations"]:
            batch = ObservationBatch(_frame_from_state(raw["X"], space.param_names),
                _frame_from_state(raw["Y"], space.objectives),
                Yvar=None if raw["Yvar"] is None else _frame_from_state(raw["Yvar"], space.objectives),
                status=None if raw["status"] is None else pd.Series(raw["status"], dtype=object), ids=raw["ids"])
            validate_frame(space, batch.X)
            batches.append(batch)
        ids = state["observation_ids"]
        kinds = state["batch_kinds"]
        expected_batch_kinds = ["doe"] + ["bo"] * (len(batches) - 1) if batches else []
        if len(ids) != len(batches) or kinds != expected_batch_kinds:
            raise _invalid("Batch kinds/effective IDs must align with observation ingestion order")
        expected_kinds = []
        for batch, effective, kind in zip(batches, ids, kinds, strict=True):
            if len(effective) != len(batch.X) or (batch.ids and list(batch.ids) != effective):
                raise _invalid("Effective IDs must match original explicit IDs and row counts")
            expected_kinds.extend([kind] * len(batch.successful().X))
        all_ids = [identifier for batch_ids in ids for identifier in batch_ids]
        if len(set(all_ids)) != len(all_ids) or state["trial_kind"] != expected_kinds:
            raise _invalid("Duplicate trial identity or inconsistent trial kinds")
        validation = state["validation"]
        if validation["last_n"] > len(expected_kinds) or state["rng_state"]["state"]["inc"] % 2 != 1:
            raise _invalid("Invalid validation progress or PCG64 increment")
        specs = {spec.pattern_id: spec for spec in knowledge.specs}
        for objective, pattern_id, kind, parameter in validation["emitted"]:
            spec = specs.get(pattern_id)
            if (objective not in space.objectives or parameter not in space.param_names or spec is None
                    or kind != spec.pattern or kind not in {"monotone", "arrhenius", "quadratic_peak"}
                    or spec.version != "1.0" or not spec.scope.factors or parameter != spec.scope.factors[0]
                    or (spec.scope.objectives and objective not in spec.scope.objectives)):
                raise _invalid("Invalid empirical validation warning identity")
        for diagnostic in validation["diagnostics"]:
            spec = specs.get(diagnostic["pattern_id"])
            if (diagnostic["objective_name"] not in space.objectives or spec is None
                    or diagnostic["pattern"] != spec.pattern
                    or (diagnostic["status"] == "checked") != (diagnostic["violation"] is not None)
                    or (diagnostic["status"] == "checked") != (diagnostic["reason"] is None)):
                raise _invalid("Invalid empirical validation diagnostic")
        campaign = campaign_class(space, knowledge, seed=state["seed"], validate=False,
            validation_interval=validation["interval"], validation_min_obs=validation["min_obs"],
            auto_rescue=validation["auto_rescue"], multiobjective=config)
        campaign._observation_batches = batches
        campaign._observation_ids = [tuple(item) for item in ids]
        campaign._batch_kinds = list(kinds)
        campaign._trial_kind = list(expected_kinds)
        pending = state["pending"]
        if pending is not None:
            if set(pending["ids"]).intersection(all_ids):
                raise _invalid("A trial cannot be both observed and pending")
            campaign.set_pending(PendingBatch(pending["ids"], _frame_from_state(pending["X"], space.param_names)))
        geometry = state["geometry"]
        if (geometry is None) != (state["ask_iteration"] == 0) or (geometry is None and state["catalog"] is not None):
            raise _invalid("Geometry/catalog must agree with successful ask progress")
        if geometry is not None:
            geometry = ObjectiveGeometry.from_dict(geometry)
            remaining = geometry.source_row_count
            prefix, prefix_ids = [], []
            for batch, effective in zip(batches, ids, strict=True):
                if remaining == 0:
                    break
                count = min(remaining, len(batch.X))
                prefix.append(batch._subset(rows=list(range(count))))
                prefix_ids.append(tuple(effective[:count]))
                remaining -= count
            if remaining:
                raise _invalid("Geometry source prefix exceeds measured history")
            # Later mixed-variance batches may be archival-only. Combine only the
            # original source prefix so they cannot change or invalidate geometry.
            campaign._observation_batches, campaign._observation_ids = prefix, prefix_ids
            observations = campaign._combined_observations()
            campaign._observation_batches, campaign._observation_ids = batches, [tuple(item) for item in ids]
            expected = resolve_geometry(space, observations, compile_runtime(space, knowledge, observations), config)
            if expected.to_dict() != geometry.to_dict():
                raise _invalid("Frozen geometry or source digest does not match its measured prefix")
            campaign._geometry = geometry
            campaign._resolved_declaration = campaign._declaration()
        catalog = state["catalog"]
        if catalog is not None:
            if catalog["ids"] != sorted(catalog["ids"]):
                raise _invalid("Bound catalog IDs must be in canonical order")
            frame = _frame_from_state(catalog["X"], space.param_names)
            campaign._catalog, campaign._catalog_ids = campaign._prepare_campaign_catalog(frame, catalog["ids"])
        campaign._rng.bit_generator.state = state["rng_state"]
        campaign._ask_iteration = state["ask_iteration"]
        campaign.validate_active = validation["active"]
        campaign._last_validation_n = validation["last_n"]
        campaign._validation_emitted = {tuple(item) for item in validation["emitted"]}
        campaign._validation_diagnostics = deepcopy(validation["diagnostics"])
        return campaign
    except EngineError as error:
        if error.code is ErrorCode.CHECKPOINT_INCOMPATIBLE:
            raise
        raise _invalid(f"Invalid multiobjective checkpoint: {error}") from error
    except (TypeError, ValueError, KeyError, OverflowError, ImportError) as error:
        raise _invalid(f"Invalid multiobjective checkpoint: {error}") from error


def restore(state, campaign_class):
    state = validate_state(state)
    if state.get("schema_version") == MULTIOBJECTIVE_SCHEMA_VERSION:
        return _restore_multiobjective(state, campaign_class)
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
            if space.n_objectives > 1:
                raise EngineError(ErrorCode.CHECKPOINT_INCOMPATIBLE,
                    "Legacy multiobjective archive is missing objective measurements; "
                    "read the raw archive instead of restoring a Campaign")
            campaign = campaign_class(space, knowledge, seed=state["seed"])
            history = pd.DataFrame(state["history"])
            if not history.empty:
                campaign.tell(history[space.param_names], history["y"].to_numpy())
                campaign._trial_kind = state["trial_kind"]
            return campaign
        batches = [batch_from_state(batch, space) for batch in state["observations"]]
        raw_pending = state["pending"]
        pending = None if raw_pending is None else PendingBatch(raw_pending["ids"],
            pd.DataFrame(raw_pending["X"], columns=space.param_names, dtype=object))
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
