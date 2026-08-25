"""Pitch phase labelling.

Splits the pitch into build-up, middle and final third from the perspective of
the team in possession, so counter-press numbers can be compared like for like
across different areas of the pitch.
"""

from __future__ import annotations

from typing import Final, Literal

import pandas as pd

Phase = Literal["build_up", "middle", "final_third"]

PHASES: Final[tuple[str, str, str]] = ("build_up", "middle", "final_third")

# Thresholds along the 105m pitch length, attacking left-to-right.
BUILD_UP_MAX_X: Final[float] = 35.0
MIDDLE_MAX_X: Final[float] = 70.0


def phase_for_x(x: float) -> Phase:
    """Label a single pitch location.

    Args:
        x: Distance in meters along the pitch, ``0`` to ``105``, attacking
            left-to-right for the team in possession.

    Returns:
        The phase that ``x`` falls in.
    """
    raise NotImplementedError


def label_phases(df: pd.DataFrame, x_col: str = "x") -> pd.DataFrame:
    """Add a ``phase`` column to a frame of pitch locations.

    Args:
        df: Frame containing a pitch-x column.
        x_col: Name of the column holding the x coordinate.

    Returns:
        A copy of ``df`` with a ``phase`` column appended.
    """
    raise NotImplementedError
