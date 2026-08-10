"""Built-in registry definitions for the v0.4 knowledge helpers."""

from .physics import arrhenius_definition, monotone_definition
from .prior import GP_PRIOR_PRESETS, gp_prior_definition
from .shape import quadratic_peak_definition, random_augment_definition


def builtin_pattern_definitions():
    """Construct a fresh set so every default registry is application-local."""
    return (
        arrhenius_definition(),
        quadratic_peak_definition(),
        monotone_definition(),
        gp_prior_definition(),
        random_augment_definition(),
    )


__all__ = ["GP_PRIOR_PRESETS", "builtin_pattern_definitions"]
