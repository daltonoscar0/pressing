"""The counter-press metric.

For each possession loss, two numbers describe what happened next:

``recovery``
    The share of pitch control the losing team wins back within five seconds of
    the turnover, normalized so that an instant regain scores ~1.0 and a clean
    break away from the press scores ~0.0.

``epv_conceded``
    The expected possession value the team that won the ball accrues over the
    same window.

Results aggregate per player (weighted by proximity to the loss), per team, and
per phase of the pitch.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

RECOVERY_WINDOW_S: float = 5.0


@dataclass(frozen=True)
class PossessionLoss:
    """A single turnover and its aftermath.

    Attributes:
        match_id: Match the loss belongs to.
        t: Time of the turnover in seconds.
        period: Match period.
        losing_team: ``"home"`` or ``"away"``; the team that gave the ball away.
        x: Turnover location along the pitch, in meters.
        y: Turnover location across the pitch, in meters.
        phase: Pitch phase, one of ``"build_up"``, ``"middle"``, ``"final_third"``.
        recovery: Control share won back within the window, in ``[0, 1]``.
        epv_conceded: EPV accrued by the team that won the ball, over the window.
    """

    match_id: str
    t: float
    period: int
    losing_team: str
    x: float
    y: float
    phase: str
    recovery: float
    epv_conceded: float


def find_possession_losses(
    tracking: pd.DataFrame,
    events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Locate every possession loss in a match.

    Args:
        tracking: Canonical tracking frame for the match.
        events: Optional canonical events frame. When present, turnovers are
            taken from the event stream; otherwise they are inferred from ball
            possession changes in the tracking data.

    Returns:
        One row per loss with ``match_id``, ``t``, ``period``, ``losing_team``,
        ``x``, ``y``.
    """
    raise NotImplementedError


def counterpress_value(
    tracking: pd.DataFrame,
    loss_t: float,
    losing_team: str,
    window_s: float = RECOVERY_WINDOW_S,
) -> tuple[float, float]:
    """Score one possession loss.

    Args:
        tracking: Canonical tracking frame covering at least ``window_s``
            seconds after ``loss_t``.
        loss_t: Time of the turnover in seconds.
        losing_team: ``"home"`` or ``"away"``; the team that gave the ball away.
        window_s: Length of the recovery window in seconds.

    Returns:
        ``(recovery, epv_conceded)``.
    """
    raise NotImplementedError


def per_player(
    losses: pd.DataFrame,
    tracking: pd.DataFrame,
    radius_m: float = 15.0,
) -> pd.DataFrame:
    """Aggregate counter-press performance per player.

    A player is credited with a loss when they are within ``radius_m`` of the
    turnover at the moment it happens.

    Args:
        losses: Scored losses, as returned by :func:`score_match`.
        tracking: Canonical tracking frame for the match.
        radius_m: Proximity threshold in meters.

    Returns:
        One row per player with loss count, mean recovery, and EPV conceded.
    """
    raise NotImplementedError


def per_team(losses: pd.DataFrame) -> pd.DataFrame:
    """Aggregate counter-press performance per team.

    Args:
        losses: Scored losses, as returned by :func:`score_match`.

    Returns:
        One row per team with loss count, mean recovery, and EPV conceded.
    """
    raise NotImplementedError


def per_phase(losses: pd.DataFrame) -> pd.DataFrame:
    """Aggregate counter-press performance per pitch phase.

    Args:
        losses: Scored losses, as returned by :func:`score_match`.

    Returns:
        One row per ``(team, phase)`` with loss count and mean recovery.
    """
    raise NotImplementedError


def score_match(tracking: pd.DataFrame, events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Find and score every possession loss in a match.

    Args:
        tracking: Canonical tracking frame for the match.
        events: Optional canonical events frame used to locate turnovers.

    Returns:
        One row per loss, with the fields of :class:`PossessionLoss`.
    """
    raise NotImplementedError
