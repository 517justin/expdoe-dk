"""Complete built-in registry for versioned domain-knowledge patterns."""

from .categorical import category_similarity_definition, ordinal_categories_definition
from .feasibility import forbidden_region_definition, safe_region_definition
from .interaction import (
    antagonism_definition,
    conditional_effect_definition,
    ratio_optimum_definition,
    synergy_definition,
)
from .multiobjective import (
    objective_priority_definition,
    target_range_definition,
    tradeoff_definition,
)
from .physics import arrhenius_definition, monotone_definition
from .prior import GP_PRIOR_PRESETS, gp_prior_definition
from .shape import (
    exponential_definition,
    optimum_range_definition,
    periodic_definition,
    power_law_definition,
    quadratic_peak_definition,
    quadratic_valley_definition,
    random_augment_definition,
    saturation_definition,
    threshold_definition,
)


def builtin_pattern_definitions():
    """Construct a fresh set so every default registry is application-local."""
    return (
        arrhenius_definition(),
        antagonism_definition(),
        category_similarity_definition(),
        conditional_effect_definition(),
        exponential_definition(),
        forbidden_region_definition(),
        gp_prior_definition(),
        quadratic_peak_definition(),
        quadratic_valley_definition(),
        monotone_definition(),
        objective_priority_definition(),
        optimum_range_definition(),
        ordinal_categories_definition(),
        periodic_definition(),
        power_law_definition(),
        random_augment_definition(),
        ratio_optimum_definition(),
        safe_region_definition(),
        saturation_definition(),
        synergy_definition(),
        target_range_definition(),
        threshold_definition(),
        tradeoff_definition(),
    )


__all__ = ["GP_PRIOR_PRESETS", "builtin_pattern_definitions"]
