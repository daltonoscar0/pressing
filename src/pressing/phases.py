"""Pitch phase labelling: build-up, middle, final third.

The pitch is cut into three equal thirds **from the perspective of the team
that had the ball**, so that a loss 20 m from your own goal is always
"build-up" whichever way you happen to be kicking. Concretely, a loss location
``x`` is first turned into an attacking fraction::

    u = x / pitch_length          if that team attacks +x
    u = 1 - x / pitch_length      if that team attacks -x

so ``u = 0`` is the team's own goal line and ``u = 1`` is the goal it is
attacking. The cuts are at exactly ``1/3`` and ``2/3``:

===============  ====================
``u``            phase
===============  ====================
``u < 1/3``      ``build_up``
``1/3 <= u <  2/3``  ``middle``
``u >= 2/3``     ``final_third``
===============  ====================

Both boundaries are **left-closed**: a location at exactly ``1/3`` is
``middle`` and one at exactly ``2/3`` is ``final_third``. Real losses never land
exactly on a boundary, but the convention has to be stated so the tests can pin
it and so the two implementations here (metres and fractions) cannot drift.

The canonical project frame has the HOME team attacking ``+x`` for the whole
match, so ``attack_direction`` is ``+1`` for home and ``-1`` for away and is
read from ``DataFrame.attrs["attack_direction"]`` rather than assumed.
"""

from __future__ import annotations

from typing import Final, Literal

import numpy as np
import pandas as pd

from pressing.schema import PITCH_LENGTH

__all__ = [
    "BUILD_UP_MAX_FRACTION",
    "BUILD_UP_MAX_X",
    "MIDDLE_MAX_FRACTION",
    "MIDDLE_MAX_X",
    "PHASES",
    "Phase",
    "attacking_fraction",
    "label_loss_phases",
    "label_phases",
    "phase_for_fraction",
    "phase_for_x",
    "phase_of_loss",
]

Phase = Literal["build_up", "middle", "final_third"]

PHASES: Final[tuple[str, str, str]] = ("build_up", "middle", "final_third")

#: Cuts as fractions of the pitch length, measured from the team's own goal.
BUILD_UP_MAX_FRACTION: Final[float] = 1.0 / 3.0
MIDDLE_MAX_FRACTION: Final[float] = 2.0 / 3.0

#: The same cuts in metres on the canonical 105 m pitch, kept for readability at
#: call sites. Derived, never hand-typed, so the two can never disagree.
BUILD_UP_MAX_X: Final[float] = PITCH_LENGTH * BUILD_UP_MAX_FRACTION
MIDDLE_MAX_X: Final[float] = PITCH_LENGTH * MIDDLE_MAX_FRACTION


def phase_for_fraction(u: float) -> Phase | None:
    """Label an attacking fraction of the pitch.

    Args:
        u: Distance from the team's own goal line as a fraction of the pitch
            length, ``0`` at its own goal and ``1`` at the goal it attacks.
            Values slightly outside ``[0, 1]`` are allowed -- players and the
            ball do go behind the goal lines -- and clamp to the nearest phase.

    Returns:
        The phase, or ``None`` when ``u`` is not finite.
    """
    if u is None or not np.isfinite(u):
        return None
    if u < BUILD_UP_MAX_FRACTION:
        return "build_up"
    if u < MIDDLE_MAX_FRACTION:
        return "middle"
    return "final_third"


def phase_for_x(x: float, pitch_length: float = PITCH_LENGTH) -> Phase | None:
    """Label a single pitch location already normalised to attack ``+x``.

    Args:
        x: Metres along the pitch, ``0`` at the team's own goal line and
            ``pitch_length`` at the goal it is attacking.
        pitch_length: Pitch length in metres.

    Returns:
        The phase that ``x`` falls in, or ``None`` when ``x`` is not finite.
    """
    return phase_for_fraction(float(x) / float(pitch_length))


def attacking_fraction(
    x: float,
    attack_direction: int,
    pitch_length: float = PITCH_LENGTH,
) -> float:
    """Convert a canonical x to a fraction measured from a team's own goal.

    Args:
        x: Canonical metres along the pitch, ``0`` at the home team's own goal.
        attack_direction: ``+1`` if the team of interest attacks ``+x``, ``-1``
            otherwise.
        pitch_length: Pitch length in metres.

    Returns:
        ``x / pitch_length`` when the team attacks ``+x``, its mirror otherwise.

    Raises:
        ValueError: If ``attack_direction`` is not ``+1`` or ``-1``.
    """
    if attack_direction not in (1, -1):
        raise ValueError(f"attack_direction must be +1 or -1, got {attack_direction!r}")
    fraction = float(x) / float(pitch_length)
    return fraction if attack_direction == 1 else 1.0 - fraction


def phase_of_loss(
    x: float,
    attack_direction: int,
    pitch_length: float = PITCH_LENGTH,
) -> Phase | None:
    """Label a loss location from the losing team's point of view.

    Args:
        x: Canonical metres along the pitch.
        attack_direction: ``+1`` if the LOSING team attacks ``+x``.
        pitch_length: Pitch length in metres.

    Returns:
        The phase, or ``None`` when ``x`` is not finite.
    """
    if x is None or not np.isfinite(x):
        return None
    return phase_for_fraction(attacking_fraction(x, attack_direction, pitch_length))


def _fractions(
    x: np.ndarray,
    directions: np.ndarray,
    pitch_length: float,
) -> np.ndarray:
    """Vectorised :func:`attacking_fraction` over aligned arrays."""
    fraction = x / float(pitch_length)
    return np.where(directions == 1, fraction, 1.0 - fraction)


def _labels(fractions: np.ndarray) -> np.ndarray:
    """Vectorised :func:`phase_for_fraction` over an array of fractions.

    Unlabelled rows come back as ``nan`` rather than ``None``: pandas coerces
    ``None`` to ``nan`` on assignment into an object column anyway, so emitting
    ``nan`` here keeps the array and the resulting column identical.
    """
    out = np.full(fractions.shape, np.nan, dtype=object)
    finite = np.isfinite(fractions)
    out[finite & (fractions < BUILD_UP_MAX_FRACTION)] = "build_up"
    out[finite & (fractions >= BUILD_UP_MAX_FRACTION) & (fractions < MIDDLE_MAX_FRACTION)] = (
        "middle"
    )
    out[finite & (fractions >= MIDDLE_MAX_FRACTION)] = "final_third"
    return out


def label_phases(
    df: pd.DataFrame,
    x_col: str = "x",
    *,
    attack_direction: int | None = None,
    direction_col: str | None = None,
    pitch_length: float = PITCH_LENGTH,
) -> pd.DataFrame:
    """Add ``x_norm`` and ``phase`` columns to a frame of pitch locations.

    Args:
        df: Frame containing a pitch-x column in canonical metres.
        x_col: Name of the column holding the x coordinate.
        attack_direction: ``+1`` / ``-1`` applied to every row. Defaults to
            ``+1``, i.e. ``x_col`` is already normalised to attack ``+x``.
        direction_col: Name of a per-row direction column, which takes
            precedence over ``attack_direction`` when given.
        pitch_length: Pitch length in metres.

    Returns:
        A copy of ``df`` with ``x_norm`` (the attacking fraction) and ``phase``
        appended. Rows with a non-finite x get ``nan`` in ``phase``.

    Raises:
        KeyError: If ``x_col`` or ``direction_col`` is missing.
    """
    if x_col not in df.columns:
        raise KeyError(f"{x_col!r} not in frame; have {list(df.columns)}")
    out = df.copy()
    out.attrs.update(df.attrs)
    if out.empty:
        out["x_norm"] = pd.Series(dtype="float64")
        out["phase"] = pd.Series(dtype="object")
        return out

    if direction_col is not None:
        if direction_col not in df.columns:
            raise KeyError(f"{direction_col!r} not in frame; have {list(df.columns)}")
        directions = df[direction_col].to_numpy(dtype="int64")
    else:
        directions = np.full(len(df), 1 if attack_direction is None else int(attack_direction))

    fractions = _fractions(df[x_col].to_numpy(dtype="float64"), directions, pitch_length)
    out["x_norm"] = fractions
    out["phase"] = _labels(fractions)
    return out


def label_loss_phases(
    losses: pd.DataFrame,
    attack_direction: dict[str, int] | None = None,
    *,
    pitch_length: float = PITCH_LENGTH,
) -> pd.DataFrame:
    """Label a loss table by the LOSING team's attacking direction.

    Args:
        losses: Loss table carrying ``x`` and ``losing_team``.
        attack_direction: ``{"home": +1, "away": -1}``. Defaults to
            ``losses.attrs["attack_direction"]`` and then to the canonical
            home-attacks-``+x`` convention.
        pitch_length: Pitch length in metres.

    Returns:
        A copy of ``losses`` with ``attack_direction``, ``x_norm`` and ``phase``
        columns appended.
    """
    directions = dict(
        attack_direction
        if attack_direction is not None
        else losses.attrs.get("attack_direction", {"home": 1, "away": -1})
    )
    out = losses.copy()
    out.attrs.update(losses.attrs)
    if out.empty:
        out["attack_direction"] = pd.Series(dtype="int64")
        out["x_norm"] = pd.Series(dtype="float64")
        out["phase"] = pd.Series(dtype="object")
        return out
    out["attack_direction"] = [
        int(directions.get(str(team), 1)) for team in losses["losing_team"]
    ]
    return label_phases(
        out,
        "x",
        direction_col="attack_direction",
        pitch_length=pitch_length,
    )
