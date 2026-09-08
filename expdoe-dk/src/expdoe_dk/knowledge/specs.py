"""Immutable declarative knowledge pattern specifications."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias


JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
_FrozenJSONValue: TypeAlias = (
    JSONScalar | tuple["_FrozenJSONValue", ...] | Mapping[str, "_FrozenJSONValue"]
)


def _require_string(value: object, context: str, *, nonempty: bool = False) -> str:
    if type(value) is not str:
        raise TypeError(f"{context} must be a string")
    if nonempty and not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _freeze_json(value: object, context: str) -> _FrozenJSONValue:
    """Validate, detach, and recursively freeze one strict JSON value."""
    if value is None or type(value) in (str, bool, int):
        return value  # type: ignore[return-value]
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{context} must contain only finite JSON numbers")
        return value
    if type(value) is list:
        return tuple(
            _freeze_json(item, f"{context}[{index}]")
            for index, item in enumerate(value)
        )
    if type(value) is dict:
        frozen: dict[str, _FrozenJSONValue] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{context} keys must be strings")
            frozen[key] = _freeze_json(item, f"{context}.{key}")
        return MappingProxyType(frozen)
    raise TypeError(
        f"{context} must contain only built-in JSON dict/list/scalar values"
    )


def _freeze_json_object(value: object, context: str) -> Mapping[str, _FrozenJSONValue]:
    if type(value) is not dict:
        raise TypeError(f"{context} must be a JSON object")
    frozen = _freeze_json(value, context)
    assert isinstance(frozen, Mapping)
    return frozen


def _thaw_json(value: _FrozenJSONValue) -> JSONValue:
    """Return a detached built-in JSON representation of a frozen value."""
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _require_exact_fields(
    payload: object,
    expected: set[str],
    context: str,
    *,
    optional: set[str] = frozenset(),
) -> dict[str, object]:
    if type(payload) is not dict:
        raise TypeError(f"{context} must be a JSON object")
    if any(type(key) is not str for key in payload):
        raise TypeError(f"{context} keys must be strings")
    required = expected - optional
    missing = required - set(payload)
    unknown = set(payload) - expected
    if missing:
        raise ValueError(f"{context} is missing fields {sorted(missing)!r}")
    if unknown:
        raise ValueError(f"{context} has unknown fields {sorted(unknown)!r}")
    return payload


def _normalize_names(value: object, context: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{context} must be an ordered sequence of strings")
    names = tuple(value)
    for name in names:
        _require_string(name, context, nonempty=True)
    if len(set(names)) != len(names):
        raise ValueError(f"{context} names must be unique")
    return names  # type: ignore[return-value]


def _normalize_confidence(value: object, context: str) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{context} must be a finite non-boolean real number")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{context} must be finite and in [0, 1]")
    return normalized


@dataclass(frozen=True)
class Evidence:
    """Non-executable provenance for a knowledge declaration."""

    kind: str
    reference: str

    def __post_init__(self) -> None:
        _require_string(self.kind, "Evidence.kind", nonempty=True)
        _require_string(self.reference, "Evidence.reference", nonempty=True)

    def to_dict(self) -> dict[str, JSONValue]:
        return {"kind": self.kind, "reference": self.reference}

    @classmethod
    def from_dict(cls, payload: object) -> "Evidence":
        item = _require_exact_fields(payload, {"kind", "reference"}, "Evidence")
        return cls(kind=item["kind"], reference=item["reference"])  # type: ignore[arg-type]


@dataclass(frozen=True)
class KnowledgeScope:
    """Ordered names and optional physical qualifiers affected by a pattern."""

    factors: tuple[str, ...] = ()
    objectives: tuple[str, ...] = ()
    conditions: dict[str, JSONValue] | None = None
    region: dict[str, JSONValue] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "factors", _normalize_names(self.factors, "KnowledgeScope.factors")
        )
        object.__setattr__(
            self,
            "objectives",
            _normalize_names(self.objectives, "KnowledgeScope.objectives"),
        )
        if self.conditions is not None:
            object.__setattr__(
                self,
                "conditions",
                _freeze_json_object(self.conditions, "KnowledgeScope.conditions"),
            )
        if self.region is not None:
            object.__setattr__(
                self,
                "region",
                _freeze_json_object(self.region, "KnowledgeScope.region"),
            )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "factors": list(self.factors),
            "objectives": list(self.objectives),
            "conditions": (
                None
                if self.conditions is None
                else _thaw_json(self.conditions)  # type: ignore[arg-type]
            ),
            "region": (
                None if self.region is None else _thaw_json(self.region)  # type: ignore[arg-type]
            ),
        }

    @classmethod
    def from_dict(cls, payload: object) -> "KnowledgeScope":
        item = _require_exact_fields(
            payload,
            {"factors", "objectives", "conditions", "region"},
            "KnowledgeScope",
        )
        if type(item["factors"]) is not list:
            raise TypeError("KnowledgeScope.factors must be a JSON array")
        if type(item["objectives"]) is not list:
            raise TypeError("KnowledgeScope.objectives must be a JSON array")
        return cls(
            factors=tuple(item["factors"]),
            objectives=tuple(item["objectives"]),
            conditions=item["conditions"],  # type: ignore[arg-type]
            region=item["region"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class KnowledgePatternSpec:
    """A versioned immutable declaration resolved through a pattern registry."""

    pattern_id: str
    pattern: str
    version: str
    parameters: dict[str, JSONValue]
    scope: KnowledgeScope
    confidence: float
    evidence: tuple[Evidence, ...] = ()
    enabled: bool = True

    def __post_init__(self) -> None:
        _require_string(self.pattern_id, "KnowledgePatternSpec.pattern_id", nonempty=True)
        _require_string(self.pattern, "KnowledgePatternSpec.pattern", nonempty=True)
        _require_string(self.version, "KnowledgePatternSpec.version", nonempty=True)
        object.__setattr__(
            self,
            "parameters",
            _freeze_json_object(self.parameters, "KnowledgePatternSpec.parameters"),
        )
        if not isinstance(self.scope, KnowledgeScope):
            raise TypeError("KnowledgePatternSpec.scope must be a KnowledgeScope")
        object.__setattr__(
            self,
            "confidence",
            _normalize_confidence(self.confidence, "KnowledgePatternSpec.confidence"),
        )
        if isinstance(self.evidence, (str, bytes, bytearray)) or not isinstance(
            self.evidence, Sequence
        ):
            raise TypeError("KnowledgePatternSpec.evidence must be an ordered sequence")
        evidence = tuple(self.evidence)
        if not all(isinstance(item, Evidence) for item in evidence):
            raise TypeError("KnowledgePatternSpec.evidence must contain Evidence instances")
        object.__setattr__(self, "evidence", evidence)
        if type(self.enabled) is not bool:
            raise TypeError("KnowledgePatternSpec.enabled must be a bool")

        # This is also an invariant check on future serialization changes.
        json.dumps(self.to_dict(), allow_nan=False)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "pattern_id": self.pattern_id,
            "pattern": self.pattern,
            "version": self.version,
            "parameters": _thaw_json(self.parameters),  # type: ignore[arg-type]
            "scope": self.scope.to_dict(),
            "confidence": self.confidence,
            "evidence": [item.to_dict() for item in self.evidence],
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "KnowledgePatternSpec":
        expected = {
            "pattern_id",
            "pattern",
            "version",
            "parameters",
            "scope",
            "confidence",
            "evidence",
            "enabled",
        }
        item = _require_exact_fields(
            payload,
            expected,
            "KnowledgePatternSpec",
            optional={"pattern_id"},
        )
        if type(item["evidence"]) is not list:
            raise TypeError("KnowledgePatternSpec.evidence must be a JSON array")
        evidence = tuple(Evidence.from_dict(value) for value in item["evidence"])
        scope = KnowledgeScope.from_dict(item["scope"])
        pattern_id = None
        if "pattern_id" in item:
            pattern_id = _require_string(
                item["pattern_id"],
                "KnowledgePatternSpec.pattern_id",
                nonempty=True,
            )
        return make_pattern_spec(
            pattern_id=pattern_id,
            pattern=item["pattern"],  # type: ignore[arg-type]
            version=item["version"],  # type: ignore[arg-type]
            parameters=item["parameters"],  # type: ignore[arg-type]
            scope=scope,
            confidence=item["confidence"],  # type: ignore[arg-type]
            evidence=evidence,
            enabled=item["enabled"],  # type: ignore[arg-type]
        )


def pattern_id_for(normalized_content: dict[str, JSONValue]) -> str:
    """Return the stable identifier for canonical normalized JSON content."""
    frozen = _freeze_json_object(normalized_content, "pattern_id content")
    canonical = json.dumps(
        _thaw_json(frozen),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"KP-{hashlib.sha256(canonical).hexdigest()[:12]}"


def make_pattern_spec(
    *,
    pattern: str,
    version: str,
    parameters: dict[str, JSONValue],
    scope: KnowledgeScope,
    confidence: float,
    evidence: Sequence[Evidence] = (),
    enabled: bool = True,
    pattern_id: str | None = None,
) -> KnowledgePatternSpec:
    """Construct a spec, deriving an identifier only when none was supplied."""
    effective_id = pattern_id
    if effective_id is None:
        provisional = KnowledgePatternSpec(
            pattern_id="provisional",
            pattern=pattern,
            version=version,
            parameters=parameters,
            scope=scope,
            confidence=confidence,
            evidence=tuple(evidence),
            enabled=enabled,
        )
        normalized = provisional.to_dict()
        del normalized["pattern_id"]
        effective_id = pattern_id_for(normalized)
    return KnowledgePatternSpec(
        pattern_id=effective_id,
        pattern=pattern,
        version=version,
        parameters=parameters,
        scope=scope,
        confidence=confidence,
        evidence=tuple(evidence),
        enabled=enabled,
    )


__all__ = [
    "Evidence",
    "JSONScalar",
    "JSONValue",
    "KnowledgePatternSpec",
    "KnowledgeScope",
    "make_pattern_spec",
    "pattern_id_for",
]
