"""Explicit, allow-listed loading of installed knowledge-pattern providers."""
from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any

from expdoe_dk.errors import EngineError, ErrorCode

from .definition import KnowledgePatternDefinition
from .registry import PatternRegistry


ENTRY_POINT_GROUP = "expdoe_dk.knowledge_patterns"


@dataclass(frozen=True)
class ProviderDefinitionRecord:
    pattern: str
    version: str

    def to_dict(self) -> dict[str, str]:
        return {"pattern": self.pattern, "version": self.version}


@dataclass(frozen=True)
class ProviderRecord:
    distribution_name: str
    canonical_distribution_name: str
    entry_point_name: str
    distribution_version: str
    definitions: tuple[ProviderDefinitionRecord, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "distribution_name": self.distribution_name,
            "canonical_distribution_name": self.canonical_distribution_name,
            "entry_point_name": self.entry_point_name,
            "distribution_version": self.distribution_version,
            "definitions": [item.to_dict() for item in self.definitions],
        }


@dataclass(frozen=True)
class ProviderLoadReport:
    providers: tuple[ProviderRecord, ...]

    def to_dict(self) -> dict[str, object]:
        return {"providers": [item.to_dict() for item in self.providers]}


@dataclass(frozen=True)
class _EntryMetadata:
    entry: Any
    distribution_name: str
    canonical_distribution_name: str
    entry_point_name: str
    distribution_version: str


def canonical_distribution_name(name: str) -> str:
    """Return the deterministic PEP 503 spelling of a distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _configuration_error(message: str, **details: object) -> EngineError:
    return EngineError(ErrorCode.CONFIG_INVALID, message, details=dict(details))


def _knowledge_error(
    message: str,
    *,
    stage: str,
    metadata: _EntryMetadata | None = None,
    error: Exception | None = None,
    **details: object,
) -> EngineError:
    payload: dict[str, object] = {"stage": stage}
    if metadata is not None:
        payload.update(
            {
                "distribution": metadata.distribution_name,
                "canonical_distribution": metadata.canonical_distribution_name,
                "entry_point": metadata.entry_point_name,
                "distribution_version": metadata.distribution_version,
            }
        )
    if error is not None:
        payload["error_type"] = type(error).__name__
    payload.update(details)
    return EngineError(ErrorCode.KNOWLEDGE_INVALID, message, details=payload)


def _requested_distributions(allowed: Collection[str]) -> set[str]:
    if isinstance(allowed, (str, bytes)) or not isinstance(allowed, Collection):
        raise _configuration_error(
            "allowed must be a collection of installed distribution names"
        )
    requested: set[str] = set()
    for index, name in enumerate(allowed):
        if not isinstance(name, str) or not name:
            raise _configuration_error(
                "Allowed distribution names must be non-empty strings",
                index=index,
            )
        if not re.fullmatch(
            r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?", name
        ):
            raise _configuration_error(
                "Allowed distribution name is not a valid project name",
                index=index,
                distribution=name,
            )
        requested.add(canonical_distribution_name(name))
    return requested


def _metadata_for(entry: object, index: int) -> _EntryMetadata:
    try:
        entry_name = entry.name
        value = entry.value
        group = entry.group
        distribution = entry.dist
        distribution_name = distribution.name
        distribution_version = distribution.version
        loader = entry.load
    except Exception as error:
        raise _knowledge_error(
            "Malformed pattern provider entry-point metadata",
            stage="metadata",
            error=error,
            entry_index=index,
        ) from None

    fields = {
        "entry_point_name": entry_name,
        "entry_point_value": value,
        "entry_point_group": group,
        "distribution_name": distribution_name,
        "distribution_version": distribution_version,
    }
    for field, item in fields.items():
        if not isinstance(item, str) or not item:
            raise _knowledge_error(
                "Malformed pattern provider entry-point metadata",
                stage="metadata",
                entry_index=index,
                field=field,
            )
    if group != ENTRY_POINT_GROUP:
        raise _knowledge_error(
            "Pattern provider entry point belongs to the wrong group",
            stage="metadata",
            entry_index=index,
            field="entry_point_group",
            group=group,
        )
    if not callable(loader):
        raise _knowledge_error(
            "Pattern provider entry point has no callable loader",
            stage="metadata",
            entry_index=index,
            field="load",
        )
    if not re.fullmatch(
        r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?",
        distribution_name,
    ):
        raise _knowledge_error(
            "Pattern provider distribution name is invalid",
            stage="metadata",
            entry_index=index,
            field="distribution_name",
        )
    return _EntryMetadata(
        entry=entry,
        distribution_name=distribution_name,
        canonical_distribution_name=canonical_distribution_name(
            distribution_name
        ),
        entry_point_name=entry_name,
        distribution_version=distribution_version,
    )


def _load_definitions(metadata: _EntryMetadata) -> tuple[KnowledgePatternDefinition, ...]:
    try:
        provider = metadata.entry.load()
    except Exception as error:
        raise _knowledge_error(
            "Pattern provider could not be loaded",
            stage="load",
            metadata=metadata,
            error=error,
        ) from None
    if not callable(provider):
        raise _knowledge_error(
            "Loaded pattern provider is not callable",
            stage="provider",
            metadata=metadata,
            actual_type=type(provider).__name__,
        )
    try:
        returned = provider()
    except Exception as error:
        raise _knowledge_error(
            "Pattern provider factory failed",
            stage="factory",
            metadata=metadata,
            error=error,
        ) from None
    if returned is None or isinstance(returned, (str, bytes, Mapping)):
        raise _knowledge_error(
            "Pattern provider returned an invalid definitions collection",
            stage="result",
            metadata=metadata,
            actual_type=type(returned).__name__,
        )
    try:
        definitions = tuple(returned)
    except Exception as error:
        raise _knowledge_error(
            "Pattern provider returned an invalid definitions collection",
            stage="result",
            metadata=metadata,
            error=error,
        ) from None
    for index, definition in enumerate(definitions):
        if not isinstance(definition, KnowledgePatternDefinition):
            raise _knowledge_error(
                "Pattern provider returned a non-definition value",
                stage="definition",
                metadata=metadata,
                definition_index=index,
                actual_type=type(definition).__name__,
            )
    return tuple(sorted(definitions, key=lambda item: (item.pattern, item.version)))


def _preflight_definitions(
    loaded: list[tuple[_EntryMetadata, tuple[KnowledgePatternDefinition, ...]]],
    registry: PatternRegistry,
) -> None:
    existing = {
        (definition.pattern, definition.version) for definition in registry.definitions()
    }
    seen: dict[tuple[str, str], _EntryMetadata] = {}
    for metadata, definitions in loaded:
        for index, definition in enumerate(definitions):
            key = (definition.pattern, definition.version)
            if key in existing:
                raise _knowledge_error(
                    "Provider definition duplicates an existing registry definition",
                    stage="duplicate",
                    metadata=metadata,
                    pattern=definition.pattern,
                    version=definition.version,
                    duplicate_with="registry",
                )
            if key in seen:
                previous = seen[key]
                raise _knowledge_error(
                    "Pattern providers returned duplicate definitions",
                    stage="duplicate",
                    metadata=metadata,
                    pattern=definition.pattern,
                    version=definition.version,
                    duplicate_with={
                        "distribution": previous.distribution_name,
                        "entry_point": previous.entry_point_name,
                    },
                )
            seen[key] = metadata

            validation_registry = PatternRegistry()
            try:
                validation_registry.register(definition)
            except Exception as error:
                underlying = (
                    error.details
                    if isinstance(error, EngineError)
                    else {"error_type": type(error).__name__}
                )
                raise _knowledge_error(
                    "Pattern provider returned an invalid definition",
                    stage="definition",
                    metadata=metadata,
                    definition_index=index,
                    pattern=definition.pattern,
                    version=definition.version,
                    validation=underlying,
                ) from None


def load_pattern_providers(
    allowed: Collection[str], registry: PatternRegistry
) -> ProviderLoadReport:
    """Load pattern definitions only from explicitly allowed distributions."""
    if not isinstance(registry, PatternRegistry):
        raise _configuration_error("registry must be a PatternRegistry")
    requested = _requested_distributions(allowed)
    if not requested:
        return ProviderLoadReport(())
    try:
        entries = tuple(entry_points(group=ENTRY_POINT_GROUP))
    except Exception as error:
        raise _knowledge_error(
            "Pattern provider metadata could not be enumerated",
            stage="enumeration",
            error=error,
        ) from None
    metadata_entries = tuple(
        _metadata_for(entry, index) for index, entry in enumerate(entries)
    )
    provenance_keys: set[tuple[str, str]] = set()
    for metadata in metadata_entries:
        key = (
            metadata.canonical_distribution_name,
            metadata.entry_point_name,
        )
        if key in provenance_keys:
            raise _knowledge_error(
                "Duplicate pattern provider entry-point metadata",
                stage="metadata",
                metadata=metadata,
            )
        provenance_keys.add(key)
    available = {
        metadata.canonical_distribution_name for metadata in metadata_entries
    }
    missing = sorted(requested - available)
    if missing:
        raise EngineError(
            ErrorCode.EXTENSION_NOT_ALLOWED,
            "Pattern provider distributions are not installed",
            details={
                "requested_distributions": sorted(requested),
                "missing_distributions": missing,
            },
        )

    selected = sorted(
        (
            metadata
            for metadata in metadata_entries
            if metadata.canonical_distribution_name in requested
        ),
        key=lambda metadata: (
            metadata.canonical_distribution_name,
            metadata.entry_point_name,
        ),
    )
    loaded = [
        (metadata, _load_definitions(metadata)) for metadata in selected
    ]
    _preflight_definitions(loaded, registry)

    records = tuple(
        ProviderRecord(
            distribution_name=metadata.distribution_name,
            canonical_distribution_name=metadata.canonical_distribution_name,
            entry_point_name=metadata.entry_point_name,
            distribution_version=metadata.distribution_version,
            definitions=tuple(
                ProviderDefinitionRecord(item.pattern, item.version)
                for item in definitions
            ),
        )
        for metadata, definitions in loaded
    )
    definitions_to_register = tuple(
        definition
        for _, definitions in loaded
        for definition in definitions
    )
    try:
        registry.register_many(definitions_to_register)
    except Exception as error:
        raise _knowledge_error(
            "Pattern provider definitions could not be committed",
            stage="commit",
            error=error,
            providers=[
                {
                    "distribution": metadata.distribution_name,
                    "canonical_distribution": (
                        metadata.canonical_distribution_name
                    ),
                    "entry_point": metadata.entry_point_name,
                    "distribution_version": metadata.distribution_version,
                }
                for metadata, _ in loaded
            ],
            definitions=[
                {"pattern": item.pattern, "version": item.version}
                for item in definitions_to_register
            ],
        ) from None
    return ProviderLoadReport(records)


__all__ = [
    "ENTRY_POINT_GROUP",
    "ProviderDefinitionRecord",
    "ProviderLoadReport",
    "ProviderRecord",
    "canonical_distribution_name",
    "load_pattern_providers",
]
