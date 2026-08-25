"""Matplotlib figures used in the report.

Every function here returns a figure rather than writing a file, so the report
builder controls sizing and placement and the same figures can be reused in the
validation notebook.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def draw_pitch(ax: object, length: float = 105.0, width: float = 68.0) -> object:
    """Draw pitch markings onto an existing axes.

    Args:
        ax: Matplotlib axes to draw on.
        length: Pitch length in meters.
        width: Pitch width in meters.

    Returns:
        The same axes, for chaining.
    """
    raise NotImplementedError


def control_heatmap(
    surface: np.ndarray,
    tracking_frame: pd.DataFrame,
    title: str = "",
) -> Figure:
    """Plot a pitch control surface with player positions on top.

    Args:
        surface: Control surface from :func:`pressing.pitchcontrol.control_surface`.
        tracking_frame: Tracking rows for the instant being drawn.
        title: Optional panel title.

    Returns:
        The figure.
    """
    raise NotImplementedError


def loss_map(losses: pd.DataFrame, team: str) -> Figure:
    """Plot every possession loss for a team, colored by recovery.

    Args:
        losses: Scored losses for the match.
        team: Team to plot.

    Returns:
        The figure.
    """
    raise NotImplementedError


def phase_bars(phase_table: pd.DataFrame) -> Figure:
    """Plot mean recovery by pitch phase for both teams.

    Args:
        phase_table: Aggregated table from :func:`pressing.counterpress.per_phase`.

    Returns:
        The figure.
    """
    raise NotImplementedError


def value_scatter(season: pd.DataFrame) -> Figure:
    """Scatter counter-press per 90 against market value.

    Args:
        season: Season table joined to market values.

    Returns:
        The figure.
    """
    raise NotImplementedError
