"""Every matplotlib figure the report is made of.

Each renderer returns a :class:`matplotlib.figure.Figure` and nothing else: no
file is written, no window is opened, and ``pyplot`` is never imported. That is
deliberate. ``pyplot`` keeps a process-global registry of figures, so a report
that builds forty panels through it leaks forty figures unless every caller
remembers to close them; building bare :class:`~matplotlib.figure.Figure`
objects with an explicit Agg canvas has no registry to leak into and is safe to
call from a worker process. The report module owns sizing and placement, and the
validation notebook can reuse the identical figures.

The visual language
-------------------
One place defines the look -- the palette, the type scale, the line weights and
the team colour pair -- and every figure draws from it, so six pages of PDF read
as one system rather than seven unrelated charts. See the constants below; the
rc settings are applied through :func:`styled`, a context manager, so nothing is
mutated globally on import.

**Team colours are blue (home) and orange (away).** That pair is the standard
colour-blind-safe choice: the two hues differ on the blue-yellow axis, which is
the axis red-green deficiencies leave intact, so deuteranopes and protanopes
(around 8% of men) separate them as easily as anyone else -- unlike the red/green
pairing football graphics reach for by default. The pitch-control diverging map
is built from those same two colours with a neutral grey midpoint, so a coach
reads "blue grass is the blue team's grass" without consulting a key.

Coordinates
-----------
Everything here speaks the canonical project frame: metres on a 105 x 68 pitch,
origin **bottom-left**, ``x`` in ``[0, 105]``, ``y`` in ``[0, 68]``, ``y``
increasing upward, the home team attacking ``+x`` for the whole match.

Control surfaces are ``(n_y, n_x)`` with **row 0 at the bottom touchline**, so
they are drawn with ``origin="lower"`` and ``extent=(0, length, 0, width)``.
Getting that backwards flips the heatmap top-for-bottom while leaving a
completely plausible-looking picture, which is the single easiest silent error
in this module -- :func:`control_heatmap` is covered by an explicit
pixel-sampling regression test for it.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Final

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Ellipse, Rectangle

from pressing.pitchcontrol import players_from_tracking
from pressing.schema import PITCH_LENGTH, PITCH_WIDTH

__all__ = [
    "AWAY_COLOUR",
    "CENTRE_CIRCLE_R",
    "CORNER_R",
    "GOAL_DEPTH",
    "GOAL_WIDTH",
    "HOME_COLOUR",
    "INK",
    "PENALTY_AREA_DEPTH",
    "PENALTY_AREA_WIDTH",
    "PENALTY_ARC_R",
    "PENALTY_SPOT_DIST",
    "RC",
    "SEASON_COLUMNS",
    "SIX_YARD_DEPTH",
    "SIX_YARD_WIDTH",
    "SURFACE",
    "TEAM_COLOURS",
    "ball_from_tracking",
    "control_cmap",
    "control_heatmap",
    "draw_pitch",
    "lock_pitch_aspect",
    "loss_map",
    "moment_caption",
    "moment_panel",
    "moment_strip",
    "ordered_teams",
    "phase_bars",
    "pitch_figure",
    "positions_from_tracking",
    "recovery_cmap",
    "recovery_sparkline",
    "season_scatter",
    "styled",
    "team_colour",
    "team_label",
    "value_scatter",
]

# --------------------------------------------------------------------------- #
# The visual language: one definition, every figure
# --------------------------------------------------------------------------- #

#: Page and panel background. Matches the report's paper.
SURFACE: Final[str] = "#ffffff"

#: Ink scale. Primary for titles and numbers, soft for labels, muted for
#: anything that must recede (gridlines, footnotes, unscored points).
INK: Final[str] = "#14140f"
INK_SOFT: Final[str] = "#52514e"
INK_MUTED: Final[str] = "#8f8e86"

#: Structure. Axis rules and gridlines are hairlines in a warm grey so the data
#: is always the darkest thing on the panel.
AXIS_COLOUR: Final[str] = "#c9cbc4"
GRID_COLOUR: Final[str] = "#e7e8e2"

#: The pitch itself: a barely-there fill and restrained markings. The pitch is
#: a backdrop for the data, never a subject.
PITCH_FILL: Final[str] = "#eff2ea"
PITCH_LINE: Final[str] = "#b0b5a6"
#: Markings drawn *over* a control surface need to survive both the pale
#: midband and the saturated poles, so they are mid-grey rather than white.
PITCH_LINE_OVER: Final[str] = "#4c4c46"

#: Team identity. Blue / orange: the colour-blind-safe pair (see module
#: docstring). Home is blue because home attacks +x, and left-to-right is how a
#: coach reads a pitch diagram.
HOME_COLOUR: Final[str] = "#2a78d6"
AWAY_COLOUR: Final[str] = "#eb6834"
TEAM_COLOURS: Final[dict[str, str]] = {"home": HOME_COLOUR, "away": AWAY_COLOUR}

#: Neutral midpoint for every diverging ramp. Grey, never a hue: the middle of a
#: diverging scale has to read as "nothing to see here".
NEUTRAL_MID: Final[str] = "#f0efec"

#: Poles of the recovery ramp. Deliberately NOT the team colours: on the loss
#: map colour already means "how much did we win back", so team identity is
#: carried by marker shape instead and the two encodings never collide.
RECOVERY_LOW: Final[str] = "#c1352f"
RECOVERY_HIGH: Final[str] = "#3d3f8f"

#: Accent used for a single highlighted mark (season-scatter outliers).
ACCENT: Final[str] = "#eb6834"

#: Mixed toward this to make the deep end of a diverging ramp.
_DEEP: Final[str] = "#0d1114"

#: Type scale. DejaVu Sans ships with matplotlib, so it is present offline on a
#: fresh clone and it is the same face the PDF registers for player names.
FONT_STACK: Final[list[str]] = ["DejaVu Sans"]
FONT_SIZE: Final[float] = 8.5
TITLE_SIZE: Final[float] = 10.5
LABEL_SIZE: Final[float] = 8.0
TICK_SIZE: Final[float] = 7.5
CAPTION_SIZE: Final[float] = 7.5

#: Line weights, in points.
PITCH_LW: Final[float] = 0.9
AXIS_LW: Final[float] = 0.8
HAIRLINE: Final[float] = 0.7
SERIES_LW: Final[float] = 1.8

#: Marker sizes, in points.
PLAYER_MS: Final[float] = 8.0
BALL_MS: Final[float] = 5.5

#: Default render resolution. 150 dpi is the measured sweet spot for the PDF:
#: visually clean at A4 and half the file size of 200 dpi for no print gain.
DEFAULT_DPI: Final[int] = 150

#: rc settings applied by :func:`styled`. Kept as data so a notebook can do
#: ``matplotlib.rcParams.update(plots.RC)`` and get the same look inline.
RC: Final[dict[str, Any]] = {
    "figure.facecolor": SURFACE,
    "figure.dpi": DEFAULT_DPI,
    "savefig.facecolor": SURFACE,
    "savefig.dpi": DEFAULT_DPI,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS_COLOUR,
    "axes.linewidth": AXIS_LW,
    "axes.labelcolor": INK_SOFT,
    "axes.labelsize": LABEL_SIZE,
    "axes.titlecolor": INK,
    "axes.titlelocation": "left",
    "axes.titlepad": 6.0,
    "axes.titlesize": TITLE_SIZE,
    "axes.titleweight": "bold",
    "axes.spines.right": False,
    "axes.spines.top": False,
    "font.family": "sans-serif",
    "font.sans-serif": FONT_STACK,
    "font.size": FONT_SIZE,
    "grid.color": GRID_COLOUR,
    "grid.linewidth": HAIRLINE,
    "legend.fontsize": TICK_SIZE,
    "legend.frameon": False,
    "legend.handletextpad": 0.5,
    "legend.labelcolor": INK_SOFT,
    "lines.linewidth": SERIES_LW,
    "lines.solid_capstyle": "round",
    "xtick.color": INK_SOFT,
    "xtick.direction": "out",
    "xtick.labelsize": TICK_SIZE,
    "xtick.major.size": 3.0,
    "xtick.major.width": AXIS_LW,
    "ytick.color": INK_SOFT,
    "ytick.direction": "out",
    "ytick.labelsize": TICK_SIZE,
    "ytick.major.size": 3.0,
    "ytick.major.width": AXIS_LW,
}


@contextmanager
def styled(overrides: dict[str, Any] | None = None) -> Iterator[None]:
    """Apply the project rc settings for the duration of the block.

    Every renderer in this module wraps itself in this, so figures built here
    look identical whatever the caller's rcParams are, and the caller's
    rcParams are restored on the way out.

    Args:
        overrides: Extra rc entries applied on top of :data:`RC`.

    Yields:
        None.
    """
    settings = dict(RC)
    if overrides:
        settings.update(overrides)
    with mpl.rc_context(settings):
        yield


def _mix(colour: str, other: str, weight: float) -> str:
    """Blend two colours in linear RGB.

    Args:
        colour: Base colour.
        other: Colour to blend toward.
        weight: Fraction of ``other``, in ``[0, 1]``.

    Returns:
        A hex string.
    """
    a = np.asarray(mcolors.to_rgb(colour))
    b = np.asarray(mcolors.to_rgb(other))
    return mcolors.to_hex(tuple((1.0 - weight) * a + weight * b))


def _luminance(colour: str) -> float:
    """Relative luminance of a colour, for picking readable overlay text."""
    r, g, b = mcolors.to_rgb(colour)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def ordered_teams(teams: Any) -> list[str]:
    """Put teams in the house order: home, then away, then anything else.

    Every panel that shows both sides uses this, so the legend order, the bar
    order and the line order never disagree between one page and the next.

    Args:
        teams: Any iterable of team keys, possibly with repeats or NaNs.

    Returns:
        A de-duplicated list in the house order.
    """
    seen = {str(t) for t in teams if isinstance(t, str) or not pd.isna(t)}
    rank = {"home": 0, "away": 1}
    return sorted(seen, key=lambda t: (rank.get(t, 2), t))


def team_colour(team: str | None) -> str:
    """The identity colour for a team.

    Args:
        team: ``"home"``, ``"away"`` or anything else.

    Returns:
        The team's colour, or the muted ink for an unknown team.
    """
    return TEAM_COLOURS.get(str(team), INK_MUTED)


def team_label(team: str | None, labels: dict[str, str] | None = None) -> str:
    """Display name for a team.

    Args:
        team: ``"home"`` or ``"away"``.
        labels: Optional real names keyed by ``"home"`` / ``"away"``.

    Returns:
        The supplied name, or the capitalised side.
    """
    key = str(team)
    if labels and key in labels:
        return str(labels[key])
    return key.replace("_", " ").capitalize()


def control_cmap(
    attacking_team: str = "home",
    defending_team: str | None = None,
) -> mcolors.LinearSegmentedColormap:
    """Diverging colormap for a pitch-control surface.

    The poles are the two teams' identity colours and the midpoint is neutral
    grey, so the surface and the player markers speak the same language: 1.0 is
    "the attacking team owns this", 0.0 is "the defending team owns it", 0.5 is
    genuinely contested and recedes toward the paper.

    Args:
        attacking_team: Team whose control the surface measures.
        defending_team: The other team. Defaults to the opposite side.

    Returns:
        A 256-step diverging colormap.
    """
    other = defending_team or ("away" if str(attacking_team) == "home" else "home")
    att = team_colour(attacking_team)
    dfd = team_colour(other)
    # The poles are barely deeper than the team colours themselves. Most of a
    # real surface is settled, so a heavy pole turns two thirds of the panel
    # into a slab of ink; keeping the ends close to the marker colour means
    # "this grass is that team's" reads instantly and the eye is still drawn to
    # the contested seam, which is the only part anyone can act on.
    return mcolors.LinearSegmentedColormap.from_list(
        f"pressing_control_{attacking_team}",
        [_mix(dfd, _DEEP, 0.22), dfd, NEUTRAL_MID, att, _mix(att, _DEEP, 0.22)],
        N=256,
    )


def recovery_cmap() -> mcolors.LinearSegmentedColormap:
    """Diverging colormap for the 5 s recovery share.

    Centred on 0.5, which the metric defines as a dead-even standoff: below it
    the opponent walked away from the press, above it the press owned the
    ground that mattered.

    Returns:
        A 256-step diverging colormap.
    """
    return mcolors.LinearSegmentedColormap.from_list(
        "pressing_recovery",
        [
            _mix(RECOVERY_LOW, _DEEP, 0.35),
            RECOVERY_LOW,
            NEUTRAL_MID,
            RECOVERY_HIGH,
            _mix(RECOVERY_HIGH, _DEEP, 0.35),
        ],
        N=256,
    )


# --------------------------------------------------------------------------- #
# Pitch geometry
# --------------------------------------------------------------------------- #

# Real Laws-of-the-Game dimensions in metres. The canonical 105 x 68 pitch is a
# full-size pitch, so these are used as-is; on any other pitch size they are
# scaled by length/105 and width/68 independently, which is why the circles are
# drawn as ellipses rather than circles.
PENALTY_AREA_DEPTH: Final[float] = 16.5
PENALTY_AREA_WIDTH: Final[float] = 40.32
SIX_YARD_DEPTH: Final[float] = 5.5
SIX_YARD_WIDTH: Final[float] = 18.32
PENALTY_SPOT_DIST: Final[float] = 11.0
PENALTY_ARC_R: Final[float] = 9.15
CENTRE_CIRCLE_R: Final[float] = 9.15
GOAL_WIDTH: Final[float] = 7.32
GOAL_DEPTH: Final[float] = 2.0
CORNER_R: Final[float] = 1.0
SPOT_R: Final[float] = 0.25


def draw_pitch(
    ax: Axes,
    length: float = PITCH_LENGTH,
    width: float = PITCH_WIDTH,
    *,
    fill: str | None = PITCH_FILL,
    line_colour: str = PITCH_LINE,
    linewidth: float = PITCH_LW,
    alpha: float = 1.0,
    margin: float = 3.2,
    zorder: float = 3.0,
    axis_off: bool = True,
) -> Axes:
    """Draw a full set of pitch markings onto an existing axes.

    Touchlines, halfway line, centre circle and spot, both penalty areas and
    six-yard boxes, penalty spots and arcs, corner arcs and the goals -- all at
    real proportions, in the canonical bottom-left metre frame.

    Args:
        ax: Axes to draw on.
        length: Pitch length in metres.
        width: Pitch width in metres.
        fill: Turf colour, or None to leave the axes background alone (which is
            what you want when a control surface is already drawn underneath).
        line_colour: Colour of the markings.
        linewidth: Marking line width in points.
        alpha: Opacity of the markings.
        margin: Metres of breathing room outside the touchlines. Must clear
            :data:`GOAL_DEPTH` or the goals are cropped.
        zorder: Draw order of the markings.
        axis_off: Whether to hide the axes frame and ticks.

    Returns:
        The same axes, for chaining.
    """
    sx = length / PITCH_LENGTH
    sy = width / PITCH_WIDTH
    line_kw: dict[str, Any] = {
        "color": line_colour,
        "linewidth": linewidth,
        "alpha": alpha,
        "zorder": zorder,
        "solid_capstyle": "butt",
    }
    patch_kw: dict[str, Any] = {
        "facecolor": "none",
        "edgecolor": line_colour,
        "linewidth": linewidth,
        "alpha": alpha,
        "zorder": zorder,
    }

    if fill is not None:
        ax.add_patch(
            Rectangle((0.0, 0.0), length, width, facecolor=fill, edgecolor="none", zorder=0.0)
        )

    # Touchlines and halfway line. The boundary carries slightly more weight
    # than the internal markings so the pitch reads as a frame, not a diagram.
    boundary_kw = {
        **patch_kw,
        "linewidth": linewidth * 1.25,
        "edgecolor": _mix(line_colour, _DEEP, 0.2),
    }
    ax.add_patch(Rectangle((0.0, 0.0), length, width, **boundary_kw))
    ax.plot([length / 2.0, length / 2.0], [0.0, width], **line_kw)

    # Centre circle and spot.
    ax.add_patch(
        Ellipse(
            (length / 2.0, width / 2.0),
            2.0 * CENTRE_CIRCLE_R * sx,
            2.0 * CENTRE_CIRCLE_R * sy,
            **patch_kw,
        )
    )
    ax.add_patch(
        Ellipse(
            (length / 2.0, width / 2.0),
            2.0 * SPOT_R * sx,
            2.0 * SPOT_R * sy,
            facecolor=line_colour,
            edgecolor="none",
            alpha=alpha,
            zorder=zorder,
        )
    )

    # The penalty arc is the part of a 9.15 m circle around the penalty spot
    # that lies outside the box. cos(theta) = (16.5 - 11) / 9.15 -> +/-53.05 deg.
    arc_cos = (PENALTY_AREA_DEPTH - PENALTY_SPOT_DIST) / PENALTY_ARC_R
    arc_half = float(np.degrees(np.arccos(arc_cos)))

    for side in (0, 1):
        # side 0 is the left goal, side 1 the right; `sign` points into the pitch.
        goal_x = 0.0 if side == 0 else length
        sign = 1.0 if side == 0 else -1.0

        for depth, box_width in (
            (PENALTY_AREA_DEPTH, PENALTY_AREA_WIDTH),
            (SIX_YARD_DEPTH, SIX_YARD_WIDTH),
        ):
            half = box_width * sy / 2.0
            ax.add_patch(
                Rectangle(
                    (min(goal_x, goal_x + sign * depth * sx), width / 2.0 - half),
                    depth * sx,
                    2.0 * half,
                    **patch_kw,
                )
            )

        spot_x = goal_x + sign * PENALTY_SPOT_DIST * sx
        ax.add_patch(
            Ellipse(
                (spot_x, width / 2.0),
                2.0 * SPOT_R * sx,
                2.0 * SPOT_R * sy,
                facecolor=line_colour,
                edgecolor="none",
                alpha=alpha,
                zorder=zorder,
            )
        )
        if side == 0:
            theta1, theta2 = -arc_half, arc_half
        else:
            theta1, theta2 = 180.0 - arc_half, 180.0 + arc_half
        ax.add_patch(
            Arc(
                (spot_x, width / 2.0),
                2.0 * PENALTY_ARC_R * sx,
                2.0 * PENALTY_ARC_R * sy,
                theta1=theta1,
                theta2=theta2,
                **patch_kw,
            )
        )

        # Goal, drawn as a shallow box behind the goal line.
        half_goal = GOAL_WIDTH * sy / 2.0
        ax.add_patch(
            Rectangle(
                (min(goal_x, goal_x - sign * GOAL_DEPTH), width / 2.0 - half_goal),
                GOAL_DEPTH,
                2.0 * half_goal,
                **{**patch_kw, "linewidth": linewidth * 1.2},
            )
        )

    for (cx, cy), start in (
        ((0.0, 0.0), 0.0),
        ((length, 0.0), 90.0),
        ((length, width), 180.0),
        ((0.0, width), 270.0),
    ):
        ax.add_patch(
            Arc(
                (cx, cy),
                2.0 * CORNER_R * sx,
                2.0 * CORNER_R * sy,
                theta1=start,
                theta2=start + 90.0,
                **patch_kw,
            )
        )

    ax.set_xlim(-margin, length + margin)
    ax.set_ylim(-margin, width + margin)
    lock_pitch_aspect(ax)
    if axis_off:
        ax.set_axis_off()
    return ax


def lock_pitch_aspect(ax: Axes) -> None:
    """Give the axes BOX the pitch's own aspect, not just the data.

    ``set_aspect("equal", adjustable="box")`` alone leaves the axes occupying
    its whole allocated cell and draws a correctly-proportioned pitch inside
    it, so a colourbar runs the full height of the cell while the pitch fills
    only part of it -- which is where the dead space beside a panel comes from,
    and ``bbox_inches="tight"`` cannot trim it because the colourbar is really
    that tall. Pinning the box aspect instead makes the cell the same shape as
    the pitch, so the colourbar matches the pitch height and the tight bbox is
    honest. The data limits are unchanged: ``adjustable="datalim"`` has nothing
    to adjust once the box already has the data's aspect.

    Args:
        ax: A pitch axes whose limits are already set.
    """
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    span_x, span_y = abs(x1 - x0), abs(y1 - y0)
    if span_x <= 0 or span_y <= 0:
        return
    ax.set_box_aspect(span_y / span_x)
    ax.set_aspect("equal", adjustable="datalim")


def pitch_figure(
    figsize: tuple[float, float] = (7.6, 5.0),
    *,
    dpi: int = DEFAULT_DPI,
    length: float = PITCH_LENGTH,
    width: float = PITCH_WIDTH,
    **pitch_kwargs: Any,
) -> tuple[Figure, Axes]:
    """Build a figure holding a single pitch.

    Args:
        figsize: Figure size in inches.
        dpi: Render resolution.
        length: Pitch length in metres.
        width: Pitch width in metres.
        **pitch_kwargs: Forwarded to :func:`draw_pitch`.

    Returns:
        ``(figure, axes)`` with the markings already drawn.
    """
    fig = _new_figure(figsize, dpi)
    ax = fig.add_subplot(1, 1, 1)
    draw_pitch(ax, length, width, **pitch_kwargs)
    return fig, ax


def _new_figure(figsize: tuple[float, float], dpi: int) -> Figure:
    """Create a constrained-layout figure with an Agg canvas attached.

    ``pyplot`` is never used, so nothing accumulates in a global registry and
    the result is safe to build inside a worker process.

    Args:
        figsize: Figure size in inches.
        dpi: Render resolution.

    Returns:
        A bare figure.
    """
    fig = Figure(figsize=figsize, dpi=dpi, layout="constrained")
    fig.set_layout_engine("constrained", w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)
    FigureCanvasAgg(fig)
    return fig


# --------------------------------------------------------------------------- #
# Shared mark drawing
# --------------------------------------------------------------------------- #


def _as_xy(points: Any) -> np.ndarray:
    """Coerce a positions argument to an ``(n, 2)`` float array.

    Accepts None, an ``(n, 2)`` array-like, a single ``(2,)`` point, or a
    dataframe with ``x`` and ``y`` columns.

    Args:
        points: Positions in canonical metres.

    Returns:
        An ``(n, 2)`` array, possibly empty.
    """
    if points is None:
        return np.empty((0, 2), dtype=float)
    if isinstance(points, pd.DataFrame):
        return points.loc[:, ["x", "y"]].to_numpy(dtype=float).reshape(-1, 2)
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return np.empty((0, 2), dtype=float)
    return arr.reshape(-1, 2)


def _draw_players(
    ax: Axes,
    positions: Any,
    colour: str,
    *,
    labels: Sequence[str] | None = None,
    velocities: Any = None,
    lead_s: float = 0.8,
    marker: str = "o",
    size: float = PLAYER_MS,
    label: str | None = None,
    zorder: float = 5.0,
) -> None:
    """Draw one team's markers, optionally with jersey numbers and velocities.

    Args:
        ax: Axes to draw on.
        positions: ``(n, 2)`` positions in canonical metres.
        colour: Team colour.
        labels: Optional per-player text drawn inside the marker.
        velocities: Optional ``(n, 2)`` velocities in m/s, drawn as a short
            leader line showing where the player is heading.
        lead_s: Seconds of travel the velocity leader represents.
        marker: Matplotlib marker.
        size: Marker size in points.
        label: Legend label.
        zorder: Draw order.
    """
    xy = _as_xy(positions)
    if len(xy) == 0:
        return
    vel = _as_xy(velocities)
    if len(vel) == len(xy):
        for (px, py), (vx, vy) in zip(xy, vel, strict=True):
            if not np.isfinite([vx, vy]).all():
                continue
            ax.plot(
                [px, px + vx * lead_s],
                [py, py + vy * lead_s],
                color=colour,
                linewidth=1.1,
                alpha=0.75,
                zorder=zorder - 0.5,
                solid_capstyle="round",
            )
    ax.plot(
        xy[:, 0],
        xy[:, 1],
        linestyle="none",
        marker=marker,
        markersize=size,
        # A shade deeper than the team colour, and deeper than the colormap's
        # own pole, so a player standing in ground his team completely controls
        # still separates from the surface underneath him.
        markerfacecolor=_mix(colour, _DEEP, 0.32),
        markeredgecolor=SURFACE,
        # The ring separates overlapping markers, so it has to scale with the
        # marker: a fixed 1.1 pt ring eats a 4 pt thumbnail marker alive.
        markeredgewidth=max(0.45, size * 0.14),
        label=label,
        zorder=zorder,
    )
    if labels is not None:
        text_colour = SURFACE if _luminance(_mix(colour, _DEEP, 0.32)) < 0.6 else INK
        for (px, py), text in zip(xy, labels, strict=False):
            ax.text(
                px,
                py,
                str(text),
                ha="center",
                va="center",
                fontsize=size * 0.55,
                color=text_colour,
                zorder=zorder + 0.5,
            )


def _draw_ball(ax: Axes, ball: Any, *, size: float = BALL_MS, zorder: float = 6.0) -> None:
    """Draw the ball, if it is known at this instant.

    Args:
        ax: Axes to draw on.
        ball: ``(2,)`` ball position in canonical metres, or None.
        size: Marker size in points.
        zorder: Draw order.
    """
    xy = _as_xy(ball)
    if len(xy) == 0 or not np.isfinite(xy[0]).all():
        return
    ax.plot(
        xy[0, 0],
        xy[0, 1],
        linestyle="none",
        marker="o",
        markersize=size,
        markerfacecolor=SURFACE,
        markeredgecolor=INK,
        markeredgewidth=max(0.6, size * 0.24),
        zorder=zorder,
    )


def _direction_note(ax: Axes, text: str, *, length: float) -> None:
    """Put a small attacking-direction note under the pitch.

    Args:
        ax: Pitch axes.
        text: The note, e.g. ``"Home attacking"``.
        length: Pitch length in metres, used to place the note.
    """
    ax.annotate(
        text,
        xy=(length / 2.0, -0.5),
        xycoords="data",
        ha="center",
        va="top",
        fontsize=CAPTION_SIZE - 0.5,
        color=INK_MUTED,
    )


def _caption(fig: Figure, text: str | None) -> None:
    """Attach a caption line beneath the figure.

    ``supxlabel`` is used rather than free text so the constrained layout
    engine reserves room for it and ``bbox_inches="tight"`` cannot clip it.

    Args:
        fig: Figure to caption.
        text: Caption, or None / empty to skip.
    """
    if not text:
        return
    fig.supxlabel(text, fontsize=CAPTION_SIZE, color=INK_SOFT, wrap=True)


def positions_from_tracking(tracking_frame: pd.DataFrame, team: str) -> np.ndarray:
    """Positions of one team at one instant, in canonical metres.

    Args:
        tracking_frame: Rows of a canonical tracking frame for a single frame.
        team: ``"home"`` or ``"away"``.

    Returns:
        ``(n, 2)`` array of in-frame player positions.
    """
    return players_from_tracking(tracking_frame, team).positions


def ball_from_tracking(tracking_frame: pd.DataFrame) -> np.ndarray | None:
    """Ball position at one instant, or None when the ball is untracked.

    Args:
        tracking_frame: Rows of a canonical tracking frame for a single frame.

    Returns:
        A ``(2,)`` array, or None.
    """
    rows = tracking_frame[tracking_frame["team"] == "ball"]
    if rows.empty:
        return None
    xy = rows.loc[:, ["x", "y"]].to_numpy(dtype=float)[0]
    return None if not np.isfinite(xy).all() else xy


# --------------------------------------------------------------------------- #
# 1. Pitch control heatmap
# --------------------------------------------------------------------------- #


def _draw_control(
    ax: Axes,
    control: np.ndarray,
    *,
    attacking_team: str,
    length: float,
    width: float,
    centre: float,
    vmin: float,
    vmax: float,
    interpolation: str,
) -> mpl.image.AxesImage:
    """Draw a control surface onto a pitch axes, right way up.

    Args:
        ax: Axes to draw on.
        control: ``(n_y, n_x)`` surface with row 0 at the BOTTOM touchline.
        attacking_team: Team whose control the surface measures.
        length: Pitch length in metres.
        width: Pitch width in metres.
        centre: Value that maps to the neutral midpoint.
        vmin: Lower end of the scale.
        vmax: Upper end of the scale.
        interpolation: ``imshow`` interpolation.

    Returns:
        The image artist, for the colourbar.

    Raises:
        ValueError: If ``control`` is not two-dimensional.
    """
    surface = np.asarray(control, dtype=float)
    if surface.ndim != 2:
        raise ValueError(f"control surface must be 2-D (n_y, n_x), got shape {surface.shape}")
    # origin="lower" is the whole ballgame: row 0 of the surface is the bottom
    # touchline. Without it the heatmap is flipped and still looks plausible.
    return ax.imshow(
        surface,
        origin="lower",
        extent=(0.0, length, 0.0, width),
        cmap=control_cmap(attacking_team),
        norm=mcolors.TwoSlopeNorm(vcenter=centre, vmin=vmin, vmax=vmax),
        interpolation=interpolation,
        alpha=0.9,
        zorder=1.0,
        aspect="auto",
    )


def _control_colorbar(
    fig: Figure,
    image: mpl.image.AxesImage,
    ax: Axes,
    *,
    attacking_team: str,
    team_labels: dict[str, str] | None,
    centre: float,
) -> None:
    """Add the control colourbar, labelled with team names rather than numbers.

    Args:
        fig: Figure to attach to.
        image: The surface image.
        ax: Axes the bar sits beside.
        attacking_team: Team whose control the surface measures.
        team_labels: Optional real team names.
        centre: The neutral value.
    """
    other = "away" if str(attacking_team) == "home" else "home"
    bar = _horizontal_bar(fig, image, ax)
    bar.set_ticks([0.0, centre, 1.0])
    bar.set_ticklabels(
        [team_label(other, team_labels), "even", team_label(attacking_team, team_labels)]
    )
    bar.ax.tick_params(length=0, labelsize=TICK_SIZE, colors=INK_SOFT)
    bar.outline.set_visible(False)


def _horizontal_bar(fig: Figure, mappable: Any, ax: Axes, shrink: float = 0.55) -> Any:
    """Add the house-style colourbar: horizontal, short, under the pitch.

    Vertical is the matplotlib default and the wrong choice here. A pitch axes
    is pinned to the pitch's own aspect (see :func:`lock_pitch_aspect`), so a
    vertical bar -- which the layout engine stretches to the full cell height,
    not the axes height -- ends up taller than the pitch and drags a band of
    dead space along with it. A short horizontal bar under the pitch also reads
    like what it is: a line running from one team to the other.

    Args:
        fig: Figure to attach to.
        mappable: The artist the bar describes.
        ax: Axes the bar sits under.
        shrink: Bar length as a fraction of the pitch width.

    Returns:
        The colourbar.
    """
    return fig.colorbar(
        mappable,
        ax=ax,
        orientation="horizontal",
        location="bottom",
        fraction=0.045,
        pad=0.015,
        shrink=shrink,
        aspect=34,
    )


def control_heatmap(
    control: np.ndarray,
    *,
    ball: Any = None,
    attackers: Any = None,
    defenders: Any = None,
    attacking_team: str = "home",
    attacker_labels: Sequence[str] | None = None,
    defender_labels: Sequence[str] | None = None,
    attacker_velocities: Any = None,
    defender_velocities: Any = None,
    title: str = "",
    caption: str | None = None,
    colorbar: bool = True,
    team_labels: dict[str, str] | None = None,
    figsize: tuple[float, float] = (7.6, 5.0),
    dpi: int = DEFAULT_DPI,
    length: float = PITCH_LENGTH,
    width: float = PITCH_WIDTH,
    centre: float = 0.5,
    vmin: float = 0.0,
    vmax: float = 1.0,
    interpolation: str = "bilinear",
    ax: Axes | None = None,
) -> Figure:
    """Plot a pitch-control surface with both teams and the ball on top.

    The surface is drawn with a diverging colormap centred on ``centre``, whose
    poles are the two teams' identity colours, so the heatmap and the markers
    agree without a key. Any grid resolution works: the array is stretched over
    the pitch extent, which is exactly right for the cell-centred grids
    :func:`pressing.pitchcontrol.make_grid` produces.

    Args:
        control: ``(n_y, n_x)`` control probability for ``attacking_team``,
            **row 0 at the bottom touchline** (the orientation
            :func:`pressing.pitchcontrol.pitch_control` returns).
        ball: ``(2,)`` ball position in canonical metres, or None.
        attackers: ``(n, 2)`` positions of the team in possession.
        defenders: ``(n, 2)`` positions of the other team.
        attacking_team: ``"home"`` or ``"away"``; the team the surface measures.
        attacker_labels: Optional jersey numbers drawn inside the markers.
        defender_labels: Same for the defending team.
        attacker_velocities: Optional ``(n, 2)`` velocities in m/s.
        defender_velocities: Same for the defending team.
        title: Panel title.
        caption: Optional caption line beneath the panel.
        colorbar: Whether to draw the colourbar.
        team_labels: Optional real team names keyed by ``"home"`` / ``"away"``.
        figsize: Figure size in inches. Ignored when ``ax`` is given.
        dpi: Render resolution. Ignored when ``ax`` is given.
        length: Pitch length in metres.
        width: Pitch width in metres.
        centre: Control value that maps to the neutral midpoint.
        vmin: Lower end of the colour scale.
        vmax: Upper end of the colour scale.
        interpolation: ``imshow`` interpolation. ``"bilinear"`` smooths the
            coarse demo grid; pass ``"nearest"`` to see the actual cells.
        ax: Draw into an existing axes instead of making a figure.

    Returns:
        The figure the surface was drawn on.

    Raises:
        ValueError: If ``control`` is not two-dimensional.
    """
    with styled():
        if ax is None:
            fig, ax = pitch_figure(
                figsize,
                dpi=dpi,
                length=length,
                width=width,
                fill=None,
                line_colour=PITCH_LINE_OVER,
                alpha=0.35,
            )
        else:
            owner = ax.get_figure()
            assert owner is not None
            fig = owner
            draw_pitch(ax, length, width, fill=None, line_colour=PITCH_LINE_OVER, alpha=0.35)

        image = _draw_control(
            ax,
            control,
            attacking_team=attacking_team,
            length=length,
            width=width,
            centre=centre,
            vmin=vmin,
            vmax=vmax,
            interpolation=interpolation,
        )
        # imshow forces aspect="auto" above so the pitch aspect stays ours.
        lock_pitch_aspect(ax)

        other = "away" if str(attacking_team) == "home" else "home"
        _draw_players(
            ax,
            defenders,
            team_colour(other),
            labels=defender_labels,
            velocities=defender_velocities,
            label=team_label(other, team_labels),
        )
        _draw_players(
            ax,
            attackers,
            team_colour(attacking_team),
            labels=attacker_labels,
            velocities=attacker_velocities,
            label=team_label(attacking_team, team_labels),
        )
        _draw_ball(ax, ball)

        if title:
            ax.set_title(title)
        if colorbar:
            _control_colorbar(
                fig,
                image,
                ax,
                attacking_team=attacking_team,
                team_labels=team_labels,
                centre=centre,
            )
        _caption(fig, caption)
    return fig


# --------------------------------------------------------------------------- #
# 2. Loss map
# --------------------------------------------------------------------------- #


def _mirror(xy: np.ndarray, length: float, width: float) -> np.ndarray:
    """Rotate positions 180 degrees so a team attacking -x now attacks +x.

    Args:
        xy: ``(n, 2)`` positions in canonical metres.
        length: Pitch length in metres.
        width: Pitch width in metres.

    Returns:
        A new ``(n, 2)`` array.
    """
    out = np.asarray(xy, dtype=float).copy()
    out[:, 0] = length - out[:, 0]
    out[:, 1] = width - out[:, 1]
    return out


def loss_map(
    losses: pd.DataFrame,
    team: str | None = None,
    *,
    value: str = "recovery",
    size_by: str | None = "epv_conceded",
    team_labels: dict[str, str] | None = None,
    title: str | None = None,
    caption: str | None = None,
    colorbar: bool = True,
    show_thirds: bool = True,
    orient_for_team: bool = True,
    figsize: tuple[float, float] = (7.6, 5.2),
    dpi: int = DEFAULT_DPI,
    length: float = PITCH_LENGTH,
    width: float = PITCH_WIDTH,
    ax: Axes | None = None,
) -> Figure:
    """Plot every possession loss, coloured by 5 s recovery.

    Colour is the recovery share on a diverging ramp centred at 0.5, the value
    the metric defines as a dead-even standoff. Marker area is the EPV the
    opponent gained, so the losses that actually cost something are the ones
    that draw the eye. Losses that could not be scored are still drawn -- as
    small hollow grey rings -- because silently dropping them would flatter the
    map.

    Args:
        losses: Scored losses from :func:`pressing.counterpress.score_losses`.
            Needs ``x``, ``y``, ``losing_team`` and ``value``; ``scored`` and
            ``size_by`` are used when present.
        team: Restrict to one losing team, or None for both.
        value: Column mapped to colour.
        size_by: Column mapped to marker area, or None for a constant size.
        team_labels: Optional real team names.
        title: Panel title. Defaults to a generated one.
        caption: Caption line. Defaults to a generated loss count.
        colorbar: Whether to draw the colourbar.
        show_thirds: Whether to mark the build-up / middle / final-third cuts.
        orient_for_team: When one team is selected, mirror the pitch so that
            team attacks ``+x`` -- a coach reads their own attack left to right.
        figsize: Figure size in inches.
        dpi: Render resolution.
        length: Pitch length in metres.
        width: Pitch width in metres.
        ax: Draw into an existing axes instead of making a figure.

    Returns:
        The figure.
    """
    with styled():
        if ax is None:
            fig, ax = pitch_figure(figsize, dpi=dpi, length=length, width=width)
        else:
            owner = ax.get_figure()
            assert owner is not None
            fig = owner
            draw_pitch(ax, length, width)

        table = losses if team is None else losses[losses["losing_team"] == team]
        table = table.reset_index(drop=True)

        if show_thirds:
            for cut in (length / 3.0, 2.0 * length / 3.0):
                ax.plot(
                    [cut, cut],
                    [0.0, width],
                    color=PITCH_LINE,
                    linewidth=HAIRLINE,
                    linestyle=(0, (2, 3)),
                    zorder=2.0,
                )

        image = None
        n_scored = 0
        if not table.empty:
            xy = table.loc[:, ["x", "y"]].to_numpy(dtype=float)
            flip = (
                orient_for_team
                and team is not None
                # Away attacks -x in the canonical frame, so its losses need
                # mirroring for the "we attack right" reading to hold.
                and str(team) == "away"
            )
            if flip:
                xy = _mirror(xy, length, width)

            colour_values = pd.to_numeric(table[value], errors="coerce").to_numpy(dtype=float)
            good = np.isfinite(xy).all(axis=1) & np.isfinite(colour_values)
            if "scored" in table.columns:
                good &= table["scored"].fillna(False).to_numpy(dtype=bool)
            n_scored = int(good.sum())

            sizes = np.full(len(table), 34.0)
            if size_by is not None and size_by in table.columns:
                magnitude = np.abs(
                    pd.to_numeric(table[size_by], errors="coerce").to_numpy(dtype=float)
                )
                top = float(np.nanmax(magnitude)) if np.isfinite(magnitude).any() else 0.0
                if top > 0:
                    sizes = 14.0 + 96.0 * np.nan_to_num(magnitude) / top

            markers = {"home": "o", "away": "^"}
            teams_present = (
                [str(team)] if team is not None else ordered_teams(table["losing_team"])
            )
            for side in teams_present:
                pick = good & (table["losing_team"].to_numpy() == side)
                if not pick.any():
                    continue
                image = ax.scatter(
                    xy[pick, 0],
                    xy[pick, 1],
                    c=colour_values[pick],
                    s=sizes[pick],
                    marker=markers.get(side, "o"),
                    cmap=recovery_cmap(),
                    norm=mcolors.TwoSlopeNorm(vcenter=0.5, vmin=0.0, vmax=1.0),
                    edgecolors=SURFACE,
                    linewidths=0.6,
                    zorder=5.0,
                )
            missed = ~good & np.isfinite(xy).all(axis=1)
            if missed.any():
                ax.scatter(
                    xy[missed, 0],
                    xy[missed, 1],
                    s=18.0,
                    facecolors="none",
                    edgecolors=INK_MUTED,
                    linewidths=0.7,
                    zorder=4.0,
                )
            # Proxy handles in neutral grey: on this figure colour already means
            # recovery, so a legend swatch taking its colour from the data would
            # claim a second, false meaning. Shape is the key, not colour.
            handles = []
            if team is None and len(teams_present) > 1:
                handles += [
                    Line2D(
                        [],
                        [],
                        linestyle="none",
                        marker=markers.get(side, "o"),
                        markerfacecolor=INK_MUTED,
                        markeredgecolor=SURFACE,
                        markersize=6.0,
                        label=team_label(side, team_labels),
                    )
                    for side in teams_present
                ]
            if missed.any():
                handles.append(
                    Line2D(
                        [],
                        [],
                        linestyle="none",
                        marker="o",
                        markerfacecolor="none",
                        markeredgecolor=INK_MUTED,
                        markersize=5.0,
                        label="not scored",
                    )
                )
            if handles:
                ax.legend(
                    handles=handles,
                    loc="lower right",
                    bbox_to_anchor=(1.0, 1.0),
                    ncols=len(handles),
                )

        if title is None:
            who = "both teams" if team is None else team_label(team, team_labels)
            # Short on purpose: the colourbar already says what colour means, and
            # the legend shares this row.
            title = f"Possession losses — {who}"
        ax.set_title(title)

        if team is not None and orient_for_team:
            _direction_note(ax, f"{team_label(team, team_labels)} attacking →", length=length)
        elif team is None:
            _direction_note(
                ax,
                f"← {team_label('away', team_labels)} attacking    |    "
                f"{team_label('home', team_labels)} attacking →",
                length=length,
            )

        if colorbar and image is not None:
            bar = _horizontal_bar(fig, image, ax)
            bar.set_label("5 s recovery share", size=LABEL_SIZE, color=INK_SOFT)
            bar.set_ticks([0.0, 0.5, 1.0])
            bar.ax.tick_params(length=0, labelsize=TICK_SIZE, colors=INK_SOFT)
            bar.outline.set_visible(False)

        if caption is None:
            unit = "loss" if len(table) == 1 else "losses"
            caption = f"{len(table)} possession {unit}, {n_scored} scored"
            if size_by is not None:
                caption += f"; marker area = |{size_by.replace('_', ' ')}|"
        _caption(fig, caption)
    return fig


# --------------------------------------------------------------------------- #
# 3. Phase bars
# --------------------------------------------------------------------------- #


def phase_bars(
    phase_table: pd.DataFrame,
    *,
    value: str = "mean_recovery",
    team_labels: dict[str, str] | None = None,
    title: str = "Recovery by pitch phase",
    figsize: tuple[float, float] = (7.6, 3.0),
    dpi: int = DEFAULT_DPI,
) -> Figure:
    """Plot mean recovery by pitch phase for both teams.

    Phase is not a nuisance variable here, it is the dominant one: measured
    mean recovery in Metrica game 1 runs 0.24-0.28 in build-up against 0.70 in
    the final third. Comparing two teams within a phase is meaningful; across
    phases it is not, which is why the bars are grouped by phase.

    Args:
        phase_table: Output of :func:`pressing.counterpress.per_phase`.
        value: Column to plot.
        team_labels: Optional real team names.
        title: Panel title.
        figsize: Figure size in inches.
        dpi: Render resolution.

    Returns:
        The figure.
    """
    with styled():
        fig = _new_figure(figsize, dpi)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(title)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True)
        ax.set_ylabel(value.replace("_", " "))

        if phase_table.empty:
            _empty(ax, "No phase data")
            return fig

        phases = list(dict.fromkeys(phase_table["phase"].tolist()))
        teams = ordered_teams(phase_table["team"])
        positions = np.arange(len(phases), dtype=float)
        span = 0.72
        bar_w = span / max(len(teams), 1)

        for i, side in enumerate(teams):
            block = phase_table[phase_table["team"] == side].set_index("phase")
            heights = [
                float(block[value].get(phase, np.nan)) if value in block.columns else np.nan
                for phase in phases
            ]
            offset = -span / 2.0 + bar_w * (i + 0.5)
            ax.bar(
                positions + offset,
                heights,
                width=bar_w * 0.86,
                color=team_colour(side),
                edgecolor=SURFACE,
                linewidth=1.0,
                label=team_label(side, team_labels),
                zorder=3.0,
            )
            for x, height in zip(positions + offset, heights, strict=True):
                if np.isfinite(height):
                    ax.annotate(
                        f"{height:.2f}",
                        xy=(x, height),
                        xytext=(0, 2),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=TICK_SIZE - 0.5,
                        color=INK_SOFT,
                    )

        ax.axhline(0.5, color=INK_MUTED, linewidth=HAIRLINE, linestyle=(0, (3, 3)), zorder=2.0)
        ax.annotate(
            "even",
            xy=(1.0, 0.5),
            xycoords=("axes fraction", "data"),
            xytext=(-2, 3),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=TICK_SIZE - 0.5,
            color=INK_MUTED,
        )
        ax.set_xticks(positions)
        ax.set_xticklabels([p.replace("_", " ") for p in phases])
        ax.set_xlim(-0.6, len(phases) - 0.4)
        tallest = max(
            (float(bar.get_height()) for group in ax.containers for bar in group),
            default=0.0,
        )
        if np.isfinite(tallest) and tallest > 0:
            # Headroom for the value labels sitting on top of the bars.
            ax.set_ylim(0.0, max(tallest, 0.5) * 1.12)
        if len(teams) > 1:
            ax.legend(loc="upper left", ncols=len(teams))
    return fig


def _declutter(fig: Figure, annotations: Sequence[Any], max_steps: int = 8) -> None:
    """Nudge overlapping point labels apart, in points, after a trial draw.

    Only a handful of labels are ever placed, so the cost of one extra draw is
    trivial next to the cost of two player names printed on top of each other.
    Offsets are moved in typographic points rather than pixels, so the result
    survives being re-rendered at a different dpi.

    Args:
        fig: Figure the annotations belong to.
        annotations: Annotations in placement priority order.
        max_steps: How many times one label may be pushed down.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed: list[Any] = []
    to_points = 72.0 / float(fig.dpi)
    for ann in annotations:
        box = ann.get_window_extent(renderer=renderer)
        for _ in range(max_steps):
            if not any(box.overlaps(other) for other in placed):
                break
            dx, dy = ann.get_position()
            ann.set_position((dx, dy - box.height * to_points - 1.5))
            box = ann.get_window_extent(renderer=renderer)
        placed.append(box)


def _empty(ax: Axes, message: str) -> None:
    """Render a bare "nothing to show" state on an axes.

    Args:
        ax: Axes to annotate.
        message: What is missing.
    """
    ax.text(
        0.5,
        0.5,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=FONT_SIZE,
        color=INK_MUTED,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# --------------------------------------------------------------------------- #
# 4-5. Moment panels
# --------------------------------------------------------------------------- #


def moment_caption(
    loss: pd.Series | dict[str, Any],
    *,
    recovering_player: str | None = None,
    team_labels: dict[str, str] | None = None,
) -> str:
    """Build the one-line caption for a moment panel.

    Args:
        loss: A row of :data:`pressing.counterpress.LOSS_SCORE_COLUMNS`.
        recovering_player: Player who won the ball back, if known.
        team_labels: Optional real team names.

    Returns:
        A caption of the form ``"58'12\" - Home lost to Player 15 - recovery
        0.62 - EPV swing +0.0031"``.
    """
    row = dict(loss)
    parts: list[str] = []

    seconds = row.get("t")
    if seconds is not None and np.isfinite(float(seconds)):
        total = int(round(float(seconds)))
        parts.append(f"{total // 60}'{total % 60:02d}\"")

    losing = row.get("losing_team")
    lost_by = row.get("losing_player")
    who = team_label(losing, team_labels)
    parts.append(f"{who} lost the ball" + (f" ({lost_by})" if lost_by else ""))

    if recovering_player:
        parts.append(f"won by {recovering_player}")

    recovery = row.get("recovery")
    if recovery is not None and np.isfinite(float(recovery)):
        parts.append(f"5 s recovery {float(recovery):.2f}")

    epv = row.get("epv_conceded")
    if epv is not None and np.isfinite(float(epv)):
        parts.append(f"EPV swing {float(epv):+.4f}")

    phase = row.get("phase")
    if phase:
        parts.append(str(phase).replace("_", " "))

    return "  ·  ".join(parts)


def moment_panel(
    control: np.ndarray,
    *,
    ball: Any = None,
    attackers: Any = None,
    defenders: Any = None,
    attacking_team: str = "home",
    attacker_labels: Sequence[str] | None = None,
    defender_labels: Sequence[str] | None = None,
    attacker_velocities: Any = None,
    defender_velocities: Any = None,
    title: str = "",
    caption: str = "",
    colorbar: bool = True,
    team_labels: dict[str, str] | None = None,
    figsize: tuple[float, float] = (5.4, 4.2),
    dpi: int = DEFAULT_DPI,
    length: float = PITCH_LENGTH,
    width: float = PITCH_WIDTH,
    interpolation: str = "bilinear",
) -> Figure:
    """One counter-press moment: pitch, control surface, markers, ball, caption.

    This is the panel pages 3-5 of the report are built from. The default
    ``figsize`` is the half-width A4 panel; pass a wider one for the full-width
    hero panel on page 5 and nothing else changes.

    Args:
        control: ``(n_y, n_x)`` control surface, row 0 at the bottom touchline.
        ball: ``(2,)`` ball position, or None.
        attackers: ``(n, 2)`` positions of the team in possession.
        defenders: ``(n, 2)`` positions of the counter-pressing team.
        attacking_team: Team the surface measures -- the team that WON the ball.
        attacker_labels: Optional jersey numbers.
        defender_labels: Optional jersey numbers.
        attacker_velocities: Optional ``(n, 2)`` velocities in m/s.
        defender_velocities: Optional ``(n, 2)`` velocities in m/s.
        title: Panel title, e.g. ``"Moment 3 - t0 + 2.0 s"``.
        caption: Caption line, usually from :func:`moment_caption`.
        colorbar: Whether to draw the colourbar. Turn it off in tight 2-up
            layouts where one shared key is enough.
        team_labels: Optional real team names.
        figsize: Figure size in inches. ``(5.4, 4.2)`` is half-width A4.
        dpi: Render resolution.
        length: Pitch length in metres.
        width: Pitch width in metres.
        interpolation: ``imshow`` interpolation.

    Returns:
        The figure.
    """
    return control_heatmap(
        control,
        ball=ball,
        attackers=attackers,
        defenders=defenders,
        attacking_team=attacking_team,
        attacker_labels=attacker_labels,
        defender_labels=defender_labels,
        attacker_velocities=attacker_velocities,
        defender_velocities=defender_velocities,
        title=title,
        caption=caption,
        colorbar=colorbar,
        team_labels=team_labels,
        figsize=figsize,
        dpi=dpi,
        length=length,
        width=width,
        interpolation=interpolation,
    )


def moment_strip(
    surfaces: Sequence[np.ndarray],
    *,
    offsets: Sequence[float] = (0.0, 2.0, 5.0),
    balls: Sequence[Any] | None = None,
    attackers: Sequence[Any] | None = None,
    defenders: Sequence[Any] | None = None,
    attacking_team: str = "home",
    title: str = "",
    caption: str = "",
    team_labels: dict[str, str] | None = None,
    figsize: tuple[float, float] = (7.6, 2.4),
    dpi: int = DEFAULT_DPI,
    length: float = PITCH_LENGTH,
    width: float = PITCH_WIDTH,
    interpolation: str = "bilinear",
) -> Figure:
    """A thumbnail strip showing the surface evolve through the window.

    Three small pitches side by side -- ``t0``, ``t0+2s``, ``t0+5s`` by default
    -- so the coach sees whether the press closed the space or the opponent
    walked out of it. This is the page-5 companion to the hero panel.

    Args:
        surfaces: One ``(n_y, n_x)`` control surface per thumbnail.
        offsets: Seconds after ``t0`` each surface represents. Used for titles.
        balls: Optional ball position per thumbnail.
        attackers: Optional ``(n, 2)`` attacker positions per thumbnail.
        defenders: Optional ``(n, 2)`` defender positions per thumbnail.
        attacking_team: Team the surfaces measure.
        title: Strip title, drawn above the first thumbnail.
        caption: Caption line beneath the strip.
        team_labels: Optional real team names.
        figsize: Figure size in inches.
        dpi: Render resolution.
        length: Pitch length in metres.
        width: Pitch width in metres.
        interpolation: ``imshow`` interpolation.

    Returns:
        The figure.

    Raises:
        ValueError: If ``surfaces`` is empty.
    """
    if len(surfaces) == 0:
        raise ValueError("moment_strip needs at least one surface")

    with styled():
        fig = _new_figure(figsize, dpi)
        axes = fig.subplots(1, len(surfaces), squeeze=False)[0]
        other = "away" if str(attacking_team) == "home" else "home"

        for i, (ax, surface) in enumerate(zip(axes, surfaces, strict=True)):
            draw_pitch(
                ax,
                length,
                width,
                fill=None,
                line_colour=PITCH_LINE_OVER,
                alpha=0.3,
                linewidth=PITCH_LW * 0.8,
                margin=GOAL_DEPTH + 0.8,
            )
            _draw_control(
                ax,
                surface,
                attacking_team=attacking_team,
                length=length,
                width=width,
                centre=0.5,
                vmin=0.0,
                vmax=1.0,
                interpolation=interpolation,
            )
            lock_pitch_aspect(ax)
            if defenders is not None and i < len(defenders):
                _draw_players(ax, defenders[i], team_colour(other), size=PLAYER_MS * 0.72)
            if attackers is not None and i < len(attackers):
                _draw_players(ax, attackers[i], team_colour(attacking_team), size=PLAYER_MS * 0.72)
            if balls is not None and i < len(balls):
                _draw_ball(ax, balls[i], size=BALL_MS * 0.7)

            offset = float(offsets[i]) if i < len(offsets) else float("nan")
            stamp = "t₀" if offset == 0.0 else f"t₀ + {offset:g} s"
            ax.set_title(stamp, fontsize=CAPTION_SIZE, color=INK_SOFT, fontweight="normal")

        if title:
            fig.suptitle(title, fontsize=TITLE_SIZE, color=INK, fontweight="bold", ha="left", x=0.0)
        if caption:
            _caption(fig, caption)
        else:
            _caption(fig, f"Control shown for {team_label(attacking_team, team_labels)}")
    return fig


# --------------------------------------------------------------------------- #
# 6. Recovery sparkline
# --------------------------------------------------------------------------- #


def _rolling_time_mean(
    t: np.ndarray, values: np.ndarray, window_s: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centred rolling mean over an irregular time series.

    Args:
        t: Sample times in seconds, any order.
        values: Sample values, same length as ``t``.
        window_s: Full window width in seconds.

    Returns:
        ``(t_sorted, mean, count)``, all sorted by time. Points with no
        finite value are excluded from the means but keep their place in time.
    """
    finite = np.isfinite(t) & np.isfinite(values)
    t = np.asarray(t, dtype=float)[finite]
    values = np.asarray(values, dtype=float)[finite]
    order = np.argsort(t, kind="stable")
    t, values = t[order], values[order]
    if t.size == 0:
        return t, values, np.zeros(0, dtype=int)
    half = window_s / 2.0
    lo = np.searchsorted(t, t - half, side="left")
    hi = np.searchsorted(t, t + half, side="right")
    cumulative = np.concatenate([[0.0], np.cumsum(values)])
    count = hi - lo
    mean = (cumulative[hi] - cumulative[lo]) / np.maximum(count, 1)
    return t, mean, count


def recovery_sparkline(
    losses: pd.DataFrame,
    *,
    value: str = "recovery",
    window_min: float = 7.0,
    team_labels: dict[str, str] | None = None,
    title: str = "Counter-press recovery through the match",
    show_points: bool = True,
    figsize: tuple[float, float] = (7.6, 2.3),
    dpi: int = DEFAULT_DPI,
) -> Figure:
    """Rolling counter-press recovery for both teams over match time.

    One line per team: the mean recovery over a centred window of
    ``window_min`` minutes, plotted against the minute the ball was lost. The
    dotted rule at 0.5 is the metric's stated neutral, so a line above it means
    the team was winning its counter-presses over that stretch.

    Args:
        losses: Scored losses from :func:`pressing.counterpress.score_losses`.
            Needs ``t``, ``losing_team`` and ``value``; ``scored`` and
            ``period`` are used when present.
        value: Column to smooth.
        window_min: Full width of the rolling window, in minutes.
        team_labels: Optional real team names.
        title: Panel title.
        show_points: Whether to draw the individual losses faintly underneath.
        figsize: Figure size in inches.
        dpi: Render resolution.

    Returns:
        The figure.
    """
    with styled():
        fig = _new_figure(figsize, dpi)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(title)
        ax.set_xlabel("match minute")
        ax.set_ylabel(value.replace("_", " "))
        ax.set_axisbelow(True)
        # No gridlines: the only reference worth drawing on this panel is the
        # dead-even rule at 0.5, and a gridline sitting under it just doubles it.

        table = losses
        if "scored" in table.columns:
            table = table[table["scored"].fillna(False).astype(bool)]
        if table.empty or value not in table.columns:
            _empty(ax, "No scored losses to plot")
            return fig

        ax.axhline(0.5, color=INK_MUTED, linewidth=HAIRLINE, linestyle=(0, (3, 3)), zorder=2.0)

        if "period" in losses.columns and losses["period"].nunique() > 1:
            first = losses[losses["period"] == losses["period"].min()]
            if not first.empty:
                break_at = float(first["t"].max()) / 60.0
                ax.axvline(break_at, color=AXIS_COLOUR, linewidth=HAIRLINE, zorder=2.0)
                ax.annotate(
                    "half time",
                    xy=(break_at, 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(3, -2),
                    textcoords="offset points",
                    ha="left",
                    va="top",
                    fontsize=TICK_SIZE - 0.5,
                    color=INK_MUTED,
                )

        for side in ordered_teams(table["losing_team"]):
            block = table[table["losing_team"] == side]
            minutes = block["t"].to_numpy(dtype=float) / 60.0
            series = pd.to_numeric(block[value], errors="coerce").to_numpy(dtype=float)
            t_sorted, smoothed, _ = _rolling_time_mean(minutes, series, window_min)
            if t_sorted.size == 0:
                continue
            colour = team_colour(side)
            if show_points:
                ax.plot(
                    minutes,
                    series,
                    linestyle="none",
                    marker="o",
                    markersize=2.6,
                    color=colour,
                    alpha=0.25,
                    zorder=3.0,
                )
            ax.plot(
                t_sorted,
                smoothed,
                color=colour,
                linewidth=SERIES_LW,
                label=team_label(side, team_labels),
                zorder=4.0,
            )

        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.margins(x=0.01)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper left", ncols=2)
        _caption(
            fig,
            f"Centred {window_min:g}-minute rolling mean; dots are individual losses, "
            "the dotted rule is a dead-even press",
        )
    return fig


# --------------------------------------------------------------------------- #
# 7. Season scatter
# --------------------------------------------------------------------------- #

#: The dataframe contract :func:`season_scatter` is written against. Season
#: mode lands later; this is the shape it must produce.
#:
#: ``player``
#:     Display name, used for the outlier labels.
#: ``counterpress_per_90``
#:     The metric, per 90 tracked minutes.
#: ``market_value_eur``
#:     Transfermarkt market value in euros. Strictly positive -- the axis is
#:     logarithmic, so zero and missing values are dropped and counted in the
#:     caption rather than silently plotted at the axis edge.
#: ``minutes`` (optional)
#:     Tracked minutes, mapped to marker area so a 200-minute cameo does not
#:     read as loudly as a full season.
#: ``team`` (optional)
#:     Club, appended to the outlier labels when present.
SEASON_COLUMNS: Final[tuple[str, ...]] = (
    "player",
    "counterpress_per_90",
    "market_value_eur",
    "minutes",
    "team",
)


def season_scatter(
    season: pd.DataFrame,
    *,
    x: str = "market_value_eur",
    y: str = "counterpress_per_90",
    label_col: str = "player",
    size_col: str | None = "minutes",
    n_labels: int = 6,
    title: str = "Counter-press per 90 against market value",
    caption: str | None = None,
    figsize: tuple[float, float] = (7.6, 4.6),
    dpi: int = DEFAULT_DPI,
) -> Figure:
    """Scatter counter-press per 90 against market value, log-x, with a trend.

    Market value spans three orders of magnitude, so the x axis is logarithmic
    and the trend is fitted in log space. The labelled points are the largest
    residuals from that trend in both directions -- the players doing more (or
    less) counter-pressing than their price says they should, which is the only
    reason this chart exists.

    Args:
        season: One row per player. See :data:`SEASON_COLUMNS`.
        x: Market-value column.
        y: Metric column.
        label_col: Column holding the display name.
        size_col: Column mapped to marker area, or None for a constant size.
        n_labels: How many outliers to label.
        title: Panel title.
        caption: Caption line. Defaults to a generated one.
        figsize: Figure size in inches.
        dpi: Render resolution.

    Returns:
        The figure. An empty or all-missing table yields a labelled empty
        state; a single point is drawn with no trend line.
    """
    with styled():
        fig = _new_figure(figsize, dpi)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(title)
        ax.set_xlabel("market value (EUR, log scale)")
        ax.set_ylabel(y.replace("_", " "))
        ax.set_axisbelow(True)
        ax.grid(True)

        if season.empty or x not in season.columns or y not in season.columns:
            _empty(ax, "No season data")
            _caption(fig, caption)
            return fig

        values = pd.to_numeric(season[x], errors="coerce").to_numpy(dtype=float)
        metric = pd.to_numeric(season[y], errors="coerce").to_numpy(dtype=float)
        usable = np.isfinite(values) & np.isfinite(metric) & (values > 0)
        dropped = int((~usable).sum())
        if not usable.any():
            _empty(ax, "No players with a positive market value")
            _caption(fig, caption or f"{dropped} rows dropped: no usable market value")
            return fig

        table = season.loc[usable].reset_index(drop=True)
        values, metric = values[usable], metric[usable]

        sizes = np.full(values.size, 26.0)
        if size_col is not None and size_col in table.columns:
            minutes = pd.to_numeric(table[size_col], errors="coerce").to_numpy(dtype=float)
            top = float(np.nanmax(minutes)) if np.isfinite(minutes).any() else 0.0
            if top > 0:
                sizes = 12.0 + 70.0 * np.nan_to_num(minutes) / top

        ax.set_xscale("log")
        ax.scatter(
            values,
            metric,
            s=sizes,
            color=HOME_COLOUR,
            alpha=0.72,
            edgecolors=SURFACE,
            linewidths=0.6,
            zorder=3.0,
        )

        log_values = np.log10(values)
        fitted = False
        note = ""
        if values.size >= 3 and np.unique(log_values).size >= 2:
            slope, intercept = np.polyfit(log_values, metric, 1)
            grid = np.linspace(log_values.min(), log_values.max(), 64)
            ax.plot(
                10.0**grid,
                slope * grid + intercept,
                color=INK_SOFT,
                linewidth=1.3,
                linestyle=(0, (5, 3)),
                zorder=4.0,
            )
            fitted = True
            residual = metric - (slope * log_values + intercept)
            if np.std(log_values) > 0 and np.std(metric) > 0:
                r = float(np.corrcoef(log_values, metric)[0, 1])
                note = f"r = {r:+.2f} against log₁₀ value"
            worst = np.argsort(-np.abs(residual))[: max(int(n_labels), 0)]
            if worst.size and label_col in table.columns:
                ax.scatter(
                    values[worst],
                    metric[worst],
                    s=sizes[worst],
                    facecolors="none",
                    edgecolors=ACCENT,
                    linewidths=1.4,
                    zorder=5.0,
                )
                labels = []
                for i in worst:
                    name = str(table.loc[i, label_col])
                    if "team" in table.columns and pd.notna(table.loc[i, "team"]):
                        name = f"{name} ({table.loc[i, 'team']})"
                    labels.append(
                        ax.annotate(
                            name,
                            xy=(values[i], metric[i]),
                            xytext=(6, 4),
                            textcoords="offset points",
                            fontsize=TICK_SIZE - 0.5,
                            color=INK,
                            zorder=6.0,
                        )
                    )
                _declutter(fig, labels)

        if note:
            ax.annotate(
                note,
                xy=(1.0, 1.0),
                xycoords="axes fraction",
                xytext=(-2, -2),
                textcoords="offset points",
                ha="right",
                va="top",
                fontsize=TICK_SIZE,
                color=INK_SOFT,
            )

        if caption is None:
            unit = "player" if len(table) == 1 else "players"
            caption = f"{len(table)} {unit}"
            if size_col is not None and size_col in table.columns:
                caption += f"; marker area = {size_col}"
            if dropped:
                caption += f"; {dropped} dropped for a missing or non-positive value"
            if not fitted:
                caption += "; too few points to fit a trend"
        _caption(fig, caption)
    return fig


def value_scatter(season: pd.DataFrame, **kwargs: Any) -> Figure:
    """Alias of :func:`season_scatter`.

    Args:
        season: Season table joined to market values.
        **kwargs: Forwarded to :func:`season_scatter`.

    Returns:
        The figure.
    """
    return season_scatter(season, **kwargs)
