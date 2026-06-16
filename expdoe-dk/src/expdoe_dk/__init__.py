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
from .doe import generate as suggest_design

__all__ = [
    "Parameter",
    "LinearConstraint",
    "Space",
    "Knowledge",
    "Campaign",
    "Result",
    "suggest_design",
]

__version__ = "0.4.0"
