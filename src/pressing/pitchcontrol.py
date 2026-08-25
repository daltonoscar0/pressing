"""Probabilistic pitch control surfaces.

Implements the Spearman-style model: every player is given a time-to-intercept
for each target location from their current position and velocity, those times
are turned into arrival probabilities with a logistic ramp, and control is
integrated forward along the ball flight to give each team's share of every
cell on the pitch.

Validated against the Friends of Tracking reference implementation on Metrica
match 1; see ``validation/`` for the correlation actually achieved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PitchControlParams:
    """Model constants for the pitch control surface.

    Attributes:
        max_speed: Player top speed in m/s.
        reaction_time: Seconds a player keeps their current velocity before
            accelerating toward the target.
        max_acceleration: Player acceleration in m/s^2.
        tti_sigma: Width of the logistic ramp on time-to-intercept, in seconds.
        lambda_att: Ball-control rate for the attacking team.
        lambda_def: Ball-control rate for the defending team.
        int_dt: Integration timestep in seconds.
        max_int_time: Integration horizon in seconds.
        grid_n_x: Grid cells along the pitch length.
        grid_n_y: Grid cells across the pitch width.
    """

    max_speed: float = 5.0
    reaction_time: float = 0.7
    max_acceleration: float = 7.0
    tti_sigma: float = 0.45
    lambda_att: float = 3.99
    lambda_def: float = 4.31
    int_dt: float = 0.04
    max_int_time: float = 10.0
    grid_n_x: int = 50
    grid_n_y: int = 32


def time_to_intercept(
    positions: np.ndarray,
    velocities: np.ndarray,
    target: np.ndarray,
    params: PitchControlParams | None = None,
) -> np.ndarray:
    """Time for each player to reach ``target``.

    Args:
        positions: ``(n_players, 2)`` array of player positions in meters.
        velocities: ``(n_players, 2)`` array of player velocities in m/s.
        target: ``(2,)`` target location in meters.
        params: Model constants. Defaults to :class:`PitchControlParams`.

    Returns:
        ``(n_players,)`` array of intercept times in seconds.
    """
    raise NotImplementedError


def control_surface(
    tracking_frame: pd.DataFrame,
    attacking_team: str,
    params: PitchControlParams | None = None,
) -> np.ndarray:
    """Compute the attacking team's pitch control over a grid.

    Args:
        tracking_frame: Rows of a canonical tracking frame for a single instant,
            including the ``team == "ball"`` row.
        attacking_team: ``"home"`` or ``"away"``; the team whose control is returned.
        params: Model constants. Defaults to :class:`PitchControlParams`.

    Returns:
        ``(grid_n_y, grid_n_x)`` array of control probabilities in ``[0, 1]``
        for the attacking team.
    """
    raise NotImplementedError


def control_share(
    surface: np.ndarray,
    region: tuple[float, float, float, float] | None = None,
) -> float:
    """Mean control across the pitch or a rectangular region.

    Args:
        surface: Control surface from :func:`control_surface`.
        region: Optional ``(x_min, y_min, x_max, y_max)`` box in meters. When
            omitted the whole pitch is used.

    Returns:
        Mean control probability in ``[0, 1]``.
    """
    raise NotImplementedError
