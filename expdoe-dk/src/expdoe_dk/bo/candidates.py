"""Validation of a fixed physical catalog, without reconstructing its rows."""
from collections.abc import Sequence

import pandas as pd
import numpy as np

from ..domain import Space
from ..doe.design import candidate_keys
from ..errors import EngineError, ErrorCode


def _physical_key(row):
    # Numeric int/float values compare exactly in Python, without rounding large
    # integers to floats. Booleans remain distinct ordinal/categorical levels.
    return tuple((isinstance(value, (bool, np.bool_)), value) for value in row)


def prepare_catalog(space: Space, frame: pd.DataFrame, ids: Sequence[str] | None, *,
                    references: Sequence[pd.DataFrame] = ()):
    """Return ID-sorted original rows and a physical-key to stable-ID mapping."""
    def invalid(message):
        return EngineError(ErrorCode.CONFIG_INVALID, message)

    if not isinstance(frame, pd.DataFrame):
        raise invalid("candidates must be a pandas DataFrame")
    if not frame.columns.is_unique or set(frame.columns) != set(space.param_names):
        raise invalid("Candidate columns must exactly match Space parameters; keep IDs and outcomes separate")
    if isinstance(ids, (str, bytes)) or not isinstance(ids, Sequence):
        raise invalid("candidate_ids must be an ordered sequence of unique non-empty strings")
    ids = tuple(ids)
    if len(ids) != len(frame) or any(not isinstance(value, str) or not value for value in ids):
        raise invalid("candidate_ids must contain one non-empty string per candidate row")
    if len(set(ids)) != len(ids):
        raise invalid("candidate_ids must be unique")
    try:
        space.physical_to_model(frame)
    except (TypeError, ValueError, OverflowError) as error:
        raise invalid(f"Invalid candidate physical values: {error}") from error
    order = sorted(range(len(ids)), key=ids.__getitem__)
    ordered = frame.iloc[order][space.param_names].copy(deep=True).reset_index(drop=True)
    key_to_id, raw_by_key, physical_keys = {}, {}, set()
    for position, original_position in enumerate(order):
        row = ordered.iloc[[position]]
        raw = _physical_key(next(row.itertuples(index=False, name=None)))
        if raw in physical_keys:
            raise invalid("Duplicate physical candidates are ambiguous; use one ID per condition")
        physical_keys.add(raw)
        key = next(iter(candidate_keys(row, space)))
        if key in raw_by_key and raw_by_key[key] != raw:
            raise invalid("Model-coordinate collision between distinct physical candidates; "
                          "the current encoding cannot distinguish these formulas")
        raw_by_key[key] = raw
        key_to_id[key] = ids[original_position]
    for reference in references:
        for position in range(len(reference)):
            row = reference.iloc[[position]][space.param_names]
            key = next(iter(candidate_keys(row, space)))
            raw = _physical_key(next(row.itertuples(index=False, name=None)))
            if key in raw_by_key and raw_by_key[key] != raw:
                raise invalid("Model-coordinate collision between a candidate and an observed or pending "
                              "condition; the current encoding cannot distinguish these formulas")
    return ordered, key_to_id
