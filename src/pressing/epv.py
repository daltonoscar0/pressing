"""Expected possession value surface.

A simple EPV grid: the value of possessing the ball at each location on the
pitch, used to weight pitch control so that regaining the ball in a dangerous
area counts for more than regaining it in front of your own keeper.
"""

from __future__ import annotations

import numpy as np


def load_epv_grid(path: str | None = None) -> np.ndarray:
    """Load the EPV grid.

    Args:
        path: Optional path to a grid file. Defaults to the bundled grid.

    Returns:
        ``(n_y, n_x)`` array of possession values, oriented left-to-right for
        the team in possession.
    """
    raise NotImplementedError


def epv_at(grid: np.ndarray, x: float, y: float) -> float:
    """Look up the EPV at a pitch location.

    Args:
        grid: EPV grid from :func:`load_epv_grid`.
        x: Distance in meters along the pitch, ``0`` to ``105``.
        y: Distance in meters across the pitch, ``0`` to ``68``.

    Returns:
        Expected possession value at that location.
    """
    raise NotImplementedError


def expected_epv(control: np.ndarray, grid: np.ndarray) -> float:
    """Control-weighted EPV over the whole pitch.

    Args:
        control: Pitch control surface for the team of interest.
        grid: EPV grid on the same geometry.

    Returns:
        Sum of ``control * grid``, the EPV a team currently commands.
    """
    raise NotImplementedError
