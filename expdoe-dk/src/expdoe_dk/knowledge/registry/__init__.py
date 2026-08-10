"""Declarative knowledge registry public surface."""

from .definition import (
    CompatibilityRule,
    KnowledgeCompiler,
    KnowledgePatternDefinition,
    KnowledgeRenderer,
    KnowledgeValidator,
)
from .registry import PatternRegistry
from .providers import (
    ENTRY_POINT_GROUP,
    ProviderDefinitionRecord,
    ProviderLoadReport,
    ProviderRecord,
    canonical_distribution_name,
    load_pattern_providers,
)

__all__ = [
    "CompatibilityRule",
    "KnowledgeCompiler",
    "KnowledgePatternDefinition",
    "KnowledgeRenderer",
    "KnowledgeValidator",
    "PatternRegistry",
    "ENTRY_POINT_GROUP",
    "ProviderDefinitionRecord",
    "ProviderLoadReport",
    "ProviderRecord",
    "canonical_distribution_name",
    "load_pattern_providers",
]
