"""Canonical dataframe schemas shared across the sports tooling suite.

Two dataframes flow through every stage of this package:

``events``
    One row per on-ball action. The pitch is normalized to 105x68 meters with the
    team in possession always attacking left-to-right, so ``x`` grows toward the
    goal being attacked. ``t`` is seconds elapsed within the match.

``tracking``
    One row per player per frame, plus a ``team == "ball"`` row carrying the ball
    position. Sampled at 10Hz or better; upstream loaders resample when a source
    delivers something coarser.

The column lists below are the contract. Loaders emit exactly these columns in
this order, and :func:`validate_events` / :func:`validate_tracking` are the
gatekeepers every entry point calls before doing any real work.
"""

from __future__ import annotations

from typing import Final

import pandas as pd
from pandas.api import types as pdt

__all__ = [
    "EVENT_COLUMNS",
    "TRACKING_COLUMNS",
    "SchemaError",
    "validate_events",
    "validate_tracking",
]

EVENT_COLUMNS: Final[tuple[str, ...]] = (
    "match_id",
    "competition",
    "season",
    "team_id",
    "team",
    "player_id",
    "player",
    "t",
    "period",
    "type",
    "subtype",
    "x",
    "y",
    "end_x",
    "end_y",
    "outcome",
    "pressure",
    "body_part",
    "extra",
)

TRACKING_COLUMNS: Final[tuple[str, ...]] = (
    "match_id",
    "t",
    "period",
    "frame",
    "team",
    "player_id",
    "jersey",
    "x",
    "y",
    "vx",
    "vy",
    "confidence",
)

# Expected dtype family per column. The check is deliberately by family rather
# than by exact numpy dtype: int32 vs int64 and object vs string are both fine,
# a string where a float belongs is not.
_EVENT_DTYPES: Final[dict[str, str]] = {
    "t": "numeric",
    "period": "integer",
    "x": "numeric",
    "y": "numeric",
    "end_x": "numeric",
    "end_y": "numeric",
    "pressure": "boolean",
}

_TRACKING_DTYPES: Final[dict[str, str]] = {
    "t": "numeric",
    "period": "integer",
    "frame": "integer",
    "team": "string",
    "x": "numeric",
    "y": "numeric",
    "vx": "numeric",
    "vy": "numeric",
    "confidence": "numeric",
}

TRACKING_TEAMS: Final[frozenset[str]] = frozenset({"home", "away", "ball"})

PITCH_LENGTH: Final[float] = 105.0
PITCH_WIDTH: Final[float] = 68.0


class SchemaError(ValueError):
    """Raised when a dataframe does not match the canonical schema.

    The message lists every problem found, not just the first one, so a caller
    fixing a loader sees the whole picture in a single run.
    """


def _family_matches(series: pd.Series, family: str) -> bool:
    """Return whether ``series`` belongs to the named dtype family."""
    if family == "numeric":
        return bool(pdt.is_numeric_dtype(series) and not pdt.is_bool_dtype(series))
    if family == "integer":
        return bool(pdt.is_integer_dtype(series))
    if family == "boolean":
        return bool(pdt.is_bool_dtype(series))
    if family == "string":
        return bool(pdt.is_string_dtype(series) or pdt.is_object_dtype(series))
    raise ValueError(f"unknown dtype family: {family!r}")


def _validate(
    df: pd.DataFrame,
    columns: tuple[str, ...],
    dtypes: dict[str, str],
    kind: str,
) -> pd.DataFrame:
    """Shared column/dtype check for both canonical frames.

    Args:
        df: Frame to validate.
        columns: Full set of canonical columns for this frame kind.
        dtypes: Mapping of column name to expected dtype family.
        kind: Human-readable frame name used in error messages.

    Returns:
        The same dataframe, with columns reordered canonically.

    Raises:
        SchemaError: If columns are missing, unexpected, or wrongly typed.
    """
    if not isinstance(df, pd.DataFrame):
        raise SchemaError(f"{kind}: expected a pandas DataFrame, got {type(df).__name__}")

    problems: list[str] = []

    present = list(df.columns)
    missing = [c for c in columns if c not in present]
    extra = [c for c in present if c not in columns]
    if missing:
        problems.append(f"missing columns: {', '.join(missing)}")
    if extra:
        problems.append(f"unexpected columns: {', '.join(extra)}")

    wrong: list[str] = []
    for column, family in dtypes.items():
        if column in missing or column not in present:
            continue
        series = df[column]
        if not _family_matches(series, family):
            wrong.append(f"{column} (expected {family}, got {series.dtype})")
    if wrong:
        problems.append(f"wrong dtypes: {'; '.join(wrong)}")

    if problems:
        raise SchemaError(f"{kind} schema invalid -> " + " | ".join(problems))

    return df.loc[:, list(columns)]


def validate_events(df: pd.DataFrame) -> pd.DataFrame:
    """Validate an ``events`` dataframe against the canonical schema.

    Args:
        df: Candidate events frame.

    Returns:
        The frame with columns reordered to :data:`EVENT_COLUMNS`.

    Raises:
        SchemaError: If columns are missing, unexpected, or wrongly typed.
    """
    return _validate(df, EVENT_COLUMNS, _EVENT_DTYPES, "events")


def validate_tracking(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a ``tracking`` dataframe against the canonical schema.

    In addition to the column and dtype contract this checks that ``team`` only
    contains ``home``, ``away`` or ``ball``.

    Args:
        df: Candidate tracking frame.

    Returns:
        The frame with columns reordered to :data:`TRACKING_COLUMNS`.

    Raises:
        SchemaError: If columns are missing, unexpected, wrongly typed, or if
            ``team`` holds a value outside :data:`TRACKING_TEAMS`.
    """
    out = _validate(df, TRACKING_COLUMNS, _TRACKING_DTYPES, "tracking")

    seen = set(out["team"].dropna().unique())
    bad = sorted(str(v) for v in seen - TRACKING_TEAMS)
    if bad:
        raise SchemaError(
            "tracking schema invalid -> team must be one of "
            f"{sorted(TRACKING_TEAMS)}, found: {', '.join(bad)}"
        )

    return out
