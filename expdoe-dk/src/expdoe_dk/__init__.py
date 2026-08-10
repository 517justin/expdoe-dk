"""
expdoe-dk: Experimental DoE + BO with Domain Knowledge injection.

Quick start:
    >>> import expdoe_dk as ed
    >>> space = ed.Space(
    ...     params=[ed.Parameter("T", bounds=(60, 120), unit="°C"),
    ...             ed.Parameter("conc", bounds=(1, 10), unit="mL",
    ...                          kind="discrete", step=1.0)],
    ...     constraints=[],
    ...     objectives="yield",
    ...     maximize=True,
    ... )
    >>> campaign = ed.Campaign(space, knowledge=None, seed=0)
    >>> doe = campaign.suggest_doe(n=8)
"""

from .space import Parameter, LinearConstraint, Space
from .knowledge import Knowledge
from .bo import Campaign, Result
from .doe import (
    DesignBatch,
    DesignDiagnostics,
    compatible_design_methods,
    suggest_design,
)
from .domain import (
    CategoricalCombinationConstraint,
    Constraint,
    ExpressionConstraint,
    Objective,
    ObservationBatch,
    OutcomeConstraint,
    PendingBatch,
    constraint_from_dict,
    normalize_objectives,
)
from .domain.space import ENGINE_VERSION, SPACE_SCHEMA_VERSION
from .errors import EngineError, ErrorCode
from .knowledge.registry import (
    ENTRY_POINT_GROUP,
    KnowledgePatternDefinition,
    PatternRegistry,
    ProviderDefinitionRecord,
    ProviderLoadReport,
    ProviderRecord,
    canonical_distribution_name,
    load_pattern_providers,
)
from .knowledge.artifacts import OptimizationArtifact, OptimizationArtifacts
from .knowledge.guard import CompatibilityResult, KnowledgeValidationResult
from .knowledge.specs import (
    Evidence,
    KnowledgePatternSpec,
    KnowledgeScope,
    make_pattern_spec,
)

__all__ = [
    "Parameter",
    "LinearConstraint",
    "Space",
    "Knowledge",
    "Campaign",
    "Result",
    "suggest_design",
    "DesignBatch",
    "DesignDiagnostics",
    "compatible_design_methods",
    "Objective",
    "ObservationBatch",
    "Constraint",
    "ExpressionConstraint",
    "CategoricalCombinationConstraint",
    "OutcomeConstraint",
    "PendingBatch",
    "constraint_from_dict",
    "normalize_objectives",
    "EngineError",
    "ErrorCode",
    "ENGINE_VERSION",
    "SPACE_SCHEMA_VERSION",
    "Evidence",
    "KnowledgePatternSpec",
    "KnowledgeScope",
    "make_pattern_spec",
    "OptimizationArtifact",
    "OptimizationArtifacts",
    "CompatibilityResult",
    "KnowledgeValidationResult",
    "KnowledgePatternDefinition",
    "PatternRegistry",
    "ENTRY_POINT_GROUP",
    "ProviderDefinitionRecord",
    "ProviderLoadReport",
    "ProviderRecord",
    "canonical_distribution_name",
    "load_pattern_providers",
]

__version__ = "0.5.0"
