"""The six-page PDF a coach is actually handed.

The template is the product. Everything upstream of here -- the loaders, the
pitch-control model, the five-second metric -- exists so that these six pages
say something true and legible about one match:

1. **Cover.** Fixture, provider and frame rate; a head-to-head KPI table with
   the subject team's column picked out and a delta against the opponent; the
   field-position caveat; and a full-width rolling-recovery sparkline.
2. **Loss map.** Every possession loss on one pitch, in each team's own colour,
   plus the per-phase breakdown table -- build-up / middle / final third by
   losses, mean recovery and EPV swing.
3-4. **Top moments 1-4**, two per page, each a pitch-control panel with a
   caption naming the minute, who lost the ball, who won it back, the recovery
   share and the EPV swing.
5. **Moment 5 in detail.** One full-width panel plus the t0 / t0+2s / t0+5s
   thumbnail strip, so the surface is seen to evolve rather than asserted to.
6. **Player table.** Minutes, losses nearby, recovery, EPV swing and the three
   phase splits, with the method footnote that defines "nearby".

Honesty is part of the layout, not an appendix
----------------------------------------------
Three claims in this report are weaker than they look, so each is printed where
the reader cannot miss it rather than left for a README nobody opens.

* **Recovery is not comparable across pitch phases.** It is dominated by where
  the ball was lost, not by how hard anyone pressed -- :func:`per_phase` in
  :mod:`pressing.counterpress` records the measured spread. The caveat is
  printed on the cover, next to the very numbers it qualifies, and again under
  the phase table on page 2.
* **The EPV grid is not ours.** It is the Friends of Tracking reference grid,
  used as published, and that project publishes no provenance for its values.
  The footnote says exactly that; it is a reference surface, not an estimate
  this package derived. See :data:`EPV_PROVENANCE`.
* **The data is Metrica's.** Credited in the footer of every page.

And when the numbers are not derived from tracking at all -- the ``pressing
proxy`` path, which estimates them from an event stream -- the report says so
three separate ways, because recon proved no single way is reliable. A rotated
alpha watermark extracts as scrambled text and disappears in greyscale
printing; a metadata field is invisible on paper; a footer line can be trimmed
by a bad photocopy. So :func:`build_report` writes all three: the diagonal
watermark, an opaque line in the page footer, and the PDF's ``subject``
metadata. See :data:`PROXY_WATERMARK`.

Two things the renderer has to get right
----------------------------------------
Both were found by rendering pages to pixels, and both pass every test that
only inspects the text layer.

**The watermark is drawn from ``onPageEnd``, never ``onPage``.** A page
template's ``onPage`` hook runs *before* the frame's flowables, so a watermark
drawn there sits underneath them and is completely occluded by a full-width
heatmap -- while still extracting perfectly from the text layer. ``onPageEnd``
runs after, and puts it on top.

**Player names need a real font.** reportlab's built-in Helvetica silently
substitutes a box glyph for anything outside Latin-1, with no exception and no
warning: "Vukovic", "Gundogan" and "Lukasz" all come out damaged in a table a
coach is reading names off. :func:`register_pdf_fonts` registers the DejaVu
Sans faces that ship inside matplotlib -- no new dependency, works offline --
and every style in this module uses them.

Sizing
------
Figures are rendered to in-memory PNGs at :data:`DPI` and wrapped as platypus
images at their *measured* pixel aspect. Nothing here hardcodes an aspect
ratio: ``bbox_inches="tight"`` crops each figure by its own content, so the
height that comes back depends on how much chrome the figure drew, and a
guessed ratio distorts the pitch. Every page's story is additionally wrapped in
a :class:`~reportlab.platypus.KeepInFrame`, so content that would overflow --
a long team name, a squad of thirty, a caption that wraps twice -- is scaled
down instead of spilling onto a seventh page. The report is exactly
:data:`N_PAGES` pages whatever it is fed.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal
from xml.sax.saxutils import escape

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    Image,
    KeepInFrame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from pressing import plots
from pressing.counterpress import (
    DEFAULT_RADIUS_M,
    WEIGHTING_LABELS,
    headline_weighting,
    per_phase,
    per_player,
    per_team,
    summarise_match,
)
from pressing.phases import PHASES
from pressing.pitchcontrol import (
    PitchControlParams,
    PitchGrid,
    PlayerSet,
    make_grid,
    pitch_control_full,
    players_from_tracking,
)
from pressing.possession import POSSESSION_TYPES

__all__ = [
    "BOTTOM_MARGIN",
    "DATA_CREDIT",
    "DPI",
    "EPV_PROVENANCE",
    "FRAME_H",
    "FRAME_W",
    "MARGIN",
    "MOMENT_OFFSETS",
    "N_PAGES",
    "PHASE_CAVEAT",
    "PROXY_SUBTITLE",
    "PROXY_WATERMARK",
    "STRIP_OFFSETS",
    "TOP_MOMENTS",
    "Moment",
    "MomentFrame",
    "build_report",
    "cover_page",
    "loss_map_page",
    "moment_detail_page",
    "moment_pages",
    "player_table_page",
    "print_grid",
    "recovering_players",
    "register_pdf_fonts",
    "render_moments",
]

# --------------------------------------------------------------------------- #
# Page geometry
# --------------------------------------------------------------------------- #

#: A4 portrait, in points.
PAGE_W: Final[float]
PAGE_H: Final[float]
PAGE_W, PAGE_H = A4

MARGIN: Final[float] = 15 * mm
#: Deeper than the others: the footer carries two lines and a rule.
BOTTOM_MARGIN: Final[float] = 18 * mm

FRAME_W: Final[float] = PAGE_W - 2 * MARGIN
FRAME_H: Final[float] = PAGE_H - MARGIN - BOTTOM_MARGIN

#: Gutter between the two columns of a 2-up panel row, in points.
GUTTER: Final[float] = 8.0
HALF_W: Final[float] = (FRAME_W - GUTTER) / 2.0

#: Raster resolution for every embedded figure. Measured over the whole
#: six-page document during recon: 100 dpi 286 KiB, 150 dpi 427 KiB, 200 dpi
#: 591 KiB, 300 dpi 929 KiB, all within 0.7-1.4 s. 150 is the point where more
#: pixels stop being visible on A4.
DPI: Final[int] = 150

#: The report is always exactly this many pages. Every page's story is wrapped
#: in a shrinking :class:`~reportlab.platypus.KeepInFrame` to keep it true.
N_PAGES: Final[int] = 6

#: How many moments the report prints. Four across pages 3-4, the fifth in
#: detail on page 5. The metric is computed at every loss; this is only how
#: many get a picture. Fewer are printed without complaint when a match has
#: fewer scorable losses.
TOP_MOMENTS: Final[int] = 5

#: Offset from ``t0``, in seconds, that a printed moment panel shows. Two
#: seconds in: at ``t0`` every panel looks the same -- the ball is on the
#: crosshair and nobody has moved -- and by ``t0+5s`` the outcome is already
#: settled. The middle of the window is where a press is still a decision.
PANEL_OFFSET: Final[float] = 2.0

#: Offsets the page-5 thumbnail strip shows.
STRIP_OFFSETS: Final[tuple[float, ...]] = (0.0, 2.0, 5.0)

#: Every offset a printed moment needs a surface for.
MOMENT_OFFSETS: Final[tuple[float, ...]] = tuple(sorted({PANEL_OFFSET, *STRIP_OFFSETS}))

#: Cells across the pitch for the panels that actually get printed. The metric
#: runs on a coarse grid over hundreds of losses; the handful of surfaces that
#: reach paper are recomputed at pitch resolution, which is cheap because there
#: are only a handful of them.
PRINT_GRID_NX: Final[int] = 105

#: Most rows the player table will print. The page count is a contract, so the
#: table is capped and says so rather than flowing onto a seventh page.
MAX_PLAYER_ROWS: Final[int] = 30

MomentRank = Literal["costly", "recovered", "conceded"]

# --------------------------------------------------------------------------- #
# The things the report has to say out loud
# --------------------------------------------------------------------------- #

#: Default diagonal watermark for a report built from event-proxy estimates.
PROXY_WATERMARK: Final[str] = "EVENT-PROXY ESTIMATES"

#: The line under it, and the wording repeated in the footer and the metadata.
PROXY_SUBTITLE: Final[str] = "not derived from tracking data"

#: Source credit, printed in the footer of every page. Metrica publishes no
#: licence for its sample data and asks only to be acknowledged; this is the
#: acknowledgement.
DATA_CREDIT: Final[str] = "Tracking and events: Metrica Sports sample data"

#: What the EPV surface is, and what it is not. The Friends of Tracking grid is
#: MIT-licensed and vendored with its notice, but that project publishes no
#: derivation for the values in it, so this package cannot claim them as an
#: estimate of its own and does not.
EPV_PROVENANCE: Final[str] = (
    "EPV surface: the Friends of Tracking reference grid (MIT), used exactly as "
    "published. That project documents no derivation for its values, so it is a "
    "reference surface reproduced here, not an estimate this tool derived."
)

#: The caveat that keeps the headline number from being misread. Printed on the
#: cover beside the KPI table and again under the phase table.
PHASE_CAVEAT: Final[str] = (
    "Read every recovery figure inside its own phase. The dominant driver of this "
    "metric is where the ball was lost, not how hard anyone pressed: lose it in "
    "your own build-up and your team is compressed around its own goal while the "
    "opponent occupies everything else, and the share is small for reasons that "
    "have nothing to do with the press. A build-up recovery and a final-third "
    "recovery are different scales and must not be subtracted from one another."
)

#: What "nearby" means in the player table.
NEARBY_GLOSS: Final[str] = (
    "Losses nearby counts the possession losses where the player was within "
    "{radius:g} m of the turnover at the instant it happened, inclusive, and on the "
    "team that lost the ball. It is a proximity count, not a judgement: being "
    "near the ball when it is lost is not the same as being at fault for it."
)

# --------------------------------------------------------------------------- #
# Ink
# --------------------------------------------------------------------------- #

_INK = colors.HexColor(plots.INK)
_INK_SOFT = colors.HexColor(plots.INK_SOFT)
_INK_MUTED = colors.HexColor(plots.INK_MUTED)
_RULE = colors.HexColor(plots.AXIS_COLOUR)
_ZEBRA = colors.HexColor("#f4f5f1")
_HEADER_FILL = colors.HexColor("#22262a")
_ALERT = colors.HexColor("#c0392b")

_TEAM_INK: Final[dict[str, colors.Color]] = {
    "home": colors.HexColor(plots.HOME_COLOUR),
    "away": colors.HexColor(plots.AWAY_COLOUR),
}


def _team_ink(team: str | None) -> colors.Color:
    """Table ink for a team, matching the figures' identity colours.

    Args:
        team: ``"home"``, ``"away"`` or anything else.

    Returns:
        The team's colour, or the muted ink for an unknown team.
    """
    return _TEAM_INK.get(str(team), _INK_SOFT)


# --------------------------------------------------------------------------- #
# Fonts
# --------------------------------------------------------------------------- #

#: The faces registered with reportlab, and the names the styles refer to.
FONT_REGULAR: Final[str] = "DejaVuSans"
FONT_BOLD: Final[str] = "DejaVuSans-Bold"
FONT_ITALIC: Final[str] = "DejaVuSans-Oblique"

_FONT_FILES: Final[dict[str, str]] = {
    FONT_REGULAR: "DejaVuSans.ttf",
    FONT_BOLD: "DejaVuSans-Bold.ttf",
    FONT_ITALIC: "DejaVuSans-Oblique.ttf",
}

_FALLBACK_FONTS: Final[tuple[str, str, str]] = ("Helvetica", "Helvetica-Bold", "Helvetica-Oblique")


def register_pdf_fonts() -> tuple[str, str, str]:
    """Register the DejaVu Sans faces reportlab needs for real player names.

    reportlab's built-in Helvetica is a Latin-1 font. Asked to draw a character
    it does not have it substitutes a filled box and raises nothing at all --
    "Vukovic" loses its c-acute, "Gundogan" its g-breve, "Lukasz" its slashed
    L -- so a squad list can be quietly corrupted in exactly the column a coach
    reads names off. The DejaVu faces have those glyphs and ship inside
    matplotlib, so registering them costs no dependency and works offline on a
    fresh clone.

    Idempotent: re-registering a face reportlab already knows is a no-op, so
    this is safe to call once per report build.

    Returns:
        ``(regular, bold, italic)`` font names to use. The DejaVu names when
        the faces were found, and Helvetica's -- which will mangle non-Latin-1
        names -- only if matplotlib's data directory is not where it says it
        is, which should not happen.
    """
    root = Path(mpl.get_data_path()) / "fonts" / "ttf"
    registered: list[str] = []
    for name, filename in _FONT_FILES.items():
        path = root / filename
        if not path.is_file():
            return _FALLBACK_FONTS
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(path)))
        registered.append(name)
    pdfmetrics.registerFontFamily(
        FONT_REGULAR, normal=FONT_REGULAR, bold=FONT_BOLD, italic=FONT_ITALIC
    )
    return (registered[0], registered[1], registered[2])


@dataclass(frozen=True)
class _Styles:
    """The paragraph styles one report build draws with."""

    title: ParagraphStyle
    subtitle: ParagraphStyle
    heading: ParagraphStyle
    body: ParagraphStyle
    caption: ParagraphStyle
    note: ParagraphStyle
    caveat: ParagraphStyle


def _build_styles(regular: str, bold: str, italic: str) -> _Styles:
    """Build the style set on top of the registered faces.

    Args:
        regular: Regular face name.
        bold: Bold face name.
        italic: Oblique face name.

    Returns:
        The styles every page uses.
    """
    body = ParagraphStyle(
        "pressing-body",
        fontName=regular,
        fontSize=8.6,
        leading=11.6,
        textColor=_INK_SOFT,
        spaceAfter=0,
    )
    return _Styles(
        title=ParagraphStyle(
            "pressing-title",
            parent=body,
            fontName=bold,
            fontSize=22,
            leading=25,
            textColor=_INK,
            spaceAfter=2,
        ),
        subtitle=ParagraphStyle(
            "pressing-subtitle",
            parent=body,
            fontSize=9.4,
            leading=12.6,
            textColor=_INK_SOFT,
            spaceAfter=10,
        ),
        heading=ParagraphStyle(
            "pressing-heading",
            parent=body,
            fontName=bold,
            fontSize=12.5,
            leading=15,
            textColor=_INK,
            spaceBefore=0,
            spaceAfter=5,
        ),
        body=body,
        caption=ParagraphStyle(
            "pressing-caption",
            parent=body,
            fontSize=7.6,
            leading=9.8,
            textColor=_INK_SOFT,
        ),
        note=ParagraphStyle(
            "pressing-note",
            parent=body,
            fontSize=7.2,
            leading=9.4,
            textColor=_INK_MUTED,
        ),
        caveat=ParagraphStyle(
            "pressing-caveat",
            parent=body,
            fontName=italic,
            fontSize=7.8,
            leading=10.4,
            textColor=_INK_SOFT,
            leftIndent=6,
            borderPadding=0,
        ),
    )


def _p(text: str, style: ParagraphStyle) -> Paragraph:
    """Escape ``text`` and wrap it in a paragraph.

    Everything printed here can contain a player or team name straight from a
    data file, so the markup escape is not optional.

    Args:
        text: Plain text, unescaped.
        style: Paragraph style.

    Returns:
        The paragraph.
    """
    return Paragraph(escape(str(text)), style)


# --------------------------------------------------------------------------- #
# Figures to flowables
# --------------------------------------------------------------------------- #


def _fig_image(fig: Figure, width_pt: float, *, dpi: int = DPI) -> Image:
    """Render a matplotlib figure to an in-memory PNG and place it at a width.

    The height comes from the PNG's own pixel dimensions, never from the
    figure's nominal ``figsize``: ``bbox_inches="tight"`` crops to the drawn
    content, so two figures asked for the same size come back with different
    aspects depending on how much chrome each drew. Assuming a ratio here is
    how a pitch ends up subtly stretched.

    Args:
        fig: Figure to render. Not closed -- these are bare figures with no
            ``pyplot`` registry behind them, so there is nothing to leak.
        width_pt: Target width in points.
        dpi: Raster resolution.

    Returns:
        A platypus image at ``width_pt`` and its measured aspect.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    buf.seek(0)
    pixel_w, pixel_h = ImageReader(buf).getSize()
    buf.seek(0)
    return Image(buf, width=width_pt, height=width_pt * pixel_h / pixel_w)


# --------------------------------------------------------------------------- #
# Small formatting helpers
# --------------------------------------------------------------------------- #


def _num(value: Any, digits: int = 2, *, dash: str = "—") -> str:
    """Format a number, or a dash when it is missing.

    Args:
        value: Anything numeric-ish.
        digits: Decimal places.
        dash: What to print for a missing or non-finite value.

    Returns:
        The formatted number, or ``dash``.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return dash
    if not np.isfinite(number):
        return dash
    return f"{number:.{digits}f}"


def _signed(value: Any, digits: int = 2, *, dash: str = "—") -> str:
    """Format a number with an explicit sign, or a dash when missing.

    Args:
        value: Anything numeric-ish.
        digits: Decimal places.
        dash: What to print for a missing or non-finite value.

    Returns:
        A string such as ``"+0.02"``, or ``dash``. An exact zero comes back
        unsigned: "+0" reads as a rounded-down positive, which it is not.
    """
    text = _num(value, digits, dash=dash)
    if text == dash:
        return text
    if float(text) == 0.0:
        return text.lstrip("-")
    return text if text.startswith("-") else f"+{text}"


def _scaled_epv(value: Any) -> float:
    """An ``epv_conceded`` value in display units, or NaN when it is missing.

    Args:
        value: Raw ``epv_conceded``, or anything that is not a number.

    Returns:
        The value times :data:`pressing.plots.EPV_DISPLAY_SCALE`, or NaN.
    """
    try:
        return float(value) * plots.EPV_DISPLAY_SCALE
    except (TypeError, ValueError):
        return float("nan")


def _pct(value: Any, digits: int = 1, *, dash: str = "—") -> str:
    """Format a fraction in ``[0, 1]`` as a percentage.

    Args:
        value: Fraction.
        digits: Decimal places.
        dash: What to print for a missing value.

    Returns:
        A string such as ``"37.5%"``, or ``dash``.
    """
    text = _num(np.nan if value is None else float(value) * 100.0, digits, dash=dash)
    return text if text == dash else f"{text}%"


def _clock(seconds: Any) -> str:
    """Format match time as ``mm'ss"``.

    Args:
        seconds: Seconds elapsed.

    Returns:
        A clock string, or ``"—"``.
    """
    try:
        total = int(round(float(seconds)))
    except (TypeError, ValueError):
        return "—"
    if total < 0:
        return "—"
    return f"{total // 60}'{total % 60:02d}\""


def _int(value: Any, *, dash: str = "—") -> str:
    """Format an integer count, or a dash when it is missing."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return dash
    return dash if not np.isfinite(number) else str(int(round(number)))


# --------------------------------------------------------------------------- #
# Moments: which ones, and the surfaces they are drawn from
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MomentFrame:
    """One evaluated instant inside a printed moment.

    Attributes:
        offset: Seconds after the turnover.
        frame: Tracking frame number.
        control: ``(n_y, n_x)`` control surface for the team that WON the ball,
            row 0 at the bottom touchline.
        attackers: ``(n, 2)`` positions of the team that won the ball.
        defenders: ``(n, 2)`` positions of the counter-pressing team.
        attacker_velocities: ``(n, 2)`` velocities in m/s, or None.
        defender_velocities: ``(n, 2)`` velocities in m/s, or None.
        ball: ``(2,)`` tracked ball position, or None when untracked.
        ball_last_known: ``(2,)`` last tracked ball position in the same
            period, held forward, or None.
    """

    offset: float
    frame: int
    control: np.ndarray
    attackers: np.ndarray
    defenders: np.ndarray
    attacker_velocities: np.ndarray | None
    defender_velocities: np.ndarray | None
    ball: np.ndarray | None
    ball_last_known: np.ndarray | None


@dataclass(frozen=True)
class Moment:
    """One possession loss, rendered.

    Attributes:
        rank: 1-based position in the printed order.
        loss: The loss row, as a plain dict.
        frames: Evaluated instants, in increasing offset order.
        recovering_player: Who won the ball back inside the window, if the
            event stream says anyone did.
    """

    rank: int
    loss: dict[str, Any]
    frames: tuple[MomentFrame, ...]
    recovering_player: str | None

    def at(self, offset: float) -> MomentFrame | None:
        """The evaluated instant nearest ``offset``, or None if there are none.

        Args:
            offset: Seconds after the turnover.

        Returns:
            The closest :class:`MomentFrame`, or None when the moment has no
            usable instants at all.
        """
        if not self.frames:
            return None
        return min(self.frames, key=lambda item: abs(item.offset - offset))

    @property
    def losing_team(self) -> str:
        """The team that gave the ball away."""
        return str(self.loss.get("losing_team"))

    @property
    def winning_team(self) -> str:
        """The team that won it."""
        return str(self.loss.get("winning_team"))

    @property
    def loss_xy(self) -> np.ndarray | None:
        """Where the ball was lost, in canonical metres, or None."""
        xy = np.array([self.loss.get("x", np.nan), self.loss.get("y", np.nan)], dtype=float)
        return xy if np.isfinite(xy).all() else None


def print_grid(n_x: int = PRINT_GRID_NX) -> PitchGrid:
    """The grid the printed panels are recomputed on.

    Args:
        n_x: Cells along the pitch. The default is one per metre.

    Returns:
        A cell-centred grid over the canonical pitch.
    """
    return make_grid(n_x=n_x)


def recovering_players(
    losses: pd.DataFrame,
    events: pd.DataFrame | None,
    *,
    window_s: float = 5.0,
    fps: float | None = None,
) -> dict[int, str]:
    """Who won the ball back after each loss, from the event stream alone.

    The same rule :func:`pressing.counterpress.score_losses` uses for its
    ``regained_within_5s`` flag -- the first ball-carrying event by the losing
    team inside the closed window, in the same period -- but keeping the name
    rather than the flag, so a moment caption can say who did it.

    Args:
        losses: Scored losses, with ``loss_id``, ``frame``, ``period``,
            ``losing_team`` and ``event_index``.
        events: Canonical events frame for the same match, in event order.
            None returns an empty mapping.
        window_s: Window length in seconds.
        fps: Tracking rate. Defaults to the events frame's own ``attrs``.

    Returns:
        ``{loss_id: player}`` for the losses where a regain was found.
    """
    if events is None or events.empty or losses.empty:
        return {}
    if "event_index" not in losses.columns:
        return {}
    rate = float(fps if fps is not None else events.attrs.get("fps", 25.0))
    ordered = events.reset_index(drop=True)
    start_frame = np.array(
        [int((payload or {}).get("start_frame", -1)) for payload in ordered["extra"]],
        dtype=np.int64,
    )
    carries = np.isin(ordered["type"].to_numpy(), list(POSSESSION_TYPES))
    teams = ordered["team"].to_numpy()
    periods = ordered["period"].to_numpy(dtype=np.int64)
    names = ordered["player"].astype("object").to_numpy()

    window_frames = int(round(window_s * rate))
    out: dict[int, str] = {}
    for row in losses.to_dict("records"):
        try:
            index = int(row["event_index"])
        except (KeyError, TypeError, ValueError):
            continue
        f0 = int(row["frame"])
        limit = f0 + window_frames
        for j in range(index + 1, len(ordered)):
            if start_frame[j] > limit:
                break
            if not carries[j] or start_frame[j] < f0:
                continue
            if periods[j] != int(row["period"]) or teams[j] != row["losing_team"]:
                continue
            name = names[j]
            if name is not None and not pd.isna(name):
                out[int(row["loss_id"])] = str(name)
            break
    return out


def _rank_key(rank: MomentRank) -> tuple[str, bool]:
    """Column and sort direction behind a moment ranking.

    Args:
        rank: Which moments to print.

    Returns:
        ``(column, ascending)``.

    Raises:
        ValueError: On an unknown ranking.
    """
    if rank == "costly":
        return "epv_conceded", False
    if rank == "recovered":
        return "recovery", False
    if rank == "conceded":
        return "recovery", True
    raise ValueError(f"rank must be one of 'costly', 'recovered', 'conceded', got {rank!r}")


#: One line explaining each ranking, for the page heading.
RANK_GLOSS: Final[dict[str, str]] = {
    "costly": (
        "Ranked by EPV swing: the turnovers after which the opponent gained the "
        "most ground over the next five seconds."
    ),
    "recovered": (
        "Ranked by recovery share: the counter-presses that won back the most "
        "ground over the next five seconds."
    ),
    "conceded": (
        "Ranked by recovery share, lowest first: the counter-presses that won "
        "back the least ground over the next five seconds."
    ),
}


def _gk_jerseys(tracking: pd.DataFrame, rows: pd.DataFrame) -> dict[str, str | None]:
    """Map each team's goalkeeper to a jersey number.

    The tracking frame names its goalkeepers by ``player_id``; the pitch-control
    player builder wants a jersey. This resolves one against the other on the
    rows that are already in hand.

    Args:
        tracking: The match's tracking frame, for ``attrs["goalkeeper"]``.
        rows: Any subset of it carrying ``player_id`` and ``jersey``.

    Returns:
        ``{"home": jersey_or_None, "away": jersey_or_None}``.
    """
    keepers = dict(tracking.attrs.get("goalkeeper", {}))
    out: dict[str, str | None] = {"home": None, "away": None}
    if not keepers or rows.empty:
        return out
    lookup = (
        rows.loc[:, ["player_id", "jersey"]]
        .dropna()
        .drop_duplicates("player_id")
        .set_index("player_id")["jersey"]
    )
    for team, keeper in keepers.items():
        if team in out and keeper in lookup.index:
            out[team] = str(lookup.loc[keeper])
    return out


class _BallTrack:
    """Ball positions per frame, with the last-known lookup a panel needs.

    The ball is untracked in a large minority of Metrica frames, so a panel
    that only draws a tracked ball draws no ball at all on many moments. The
    metric's documented ``ball_policy="hold"`` carries the last known position
    forward within a period; this is the same fallback for the picture, and it
    reports which of the two it found so the panel can label it.
    """

    def __init__(self, tracking: pd.DataFrame, *, fps: float, hold_max_s: float = 5.0) -> None:
        """Index the match's ball rows.

        Args:
            tracking: Canonical tracking frame for the match.
            fps: Tracking rate.
            hold_max_s: How far back a held position may be taken from.
        """
        rows = tracking.loc[tracking["team"] == "ball", ["frame", "period", "x", "y"]]
        rows = rows.sort_values("frame", kind="stable")
        finite = np.isfinite(rows["x"].to_numpy(dtype=float)) & np.isfinite(
            rows["y"].to_numpy(dtype=float)
        )
        known = rows.loc[finite]
        self._frame = known["frame"].to_numpy(dtype=np.int64)
        self._period = known["period"].to_numpy(dtype=np.int64)
        self._xy = known.loc[:, ["x", "y"]].to_numpy(dtype=float)
        self._max_gap = int(round(hold_max_s * fps))

    def at(self, frame: int, period: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Ball position at a frame, and the held fallback.

        Args:
            frame: Frame number.
            period: The frame's period; a held position never crosses into it
                from another one.

        Returns:
            ``(tracked, last_known)``. ``tracked`` is the ball's own row when
            it exists; ``last_known`` is the most recent tracked position in
            the same period within the hold limit, or None.
        """
        if self._frame.size == 0:
            return None, None
        slot = int(np.searchsorted(self._frame, frame, side="right")) - 1
        if slot < 0:
            return None, None
        if self._frame[slot] == frame and self._period[slot] == period:
            return self._xy[slot], None
        if (
            self._period[slot] == period
            and 0 < frame - int(self._frame[slot]) <= self._max_gap
        ):
            return None, self._xy[slot]
        return None, None


def render_moments(
    tracking: pd.DataFrame,
    losses: pd.DataFrame,
    team: str | None = None,
    *,
    n: int = TOP_MOMENTS,
    rank: MomentRank = "costly",
    offsets: tuple[float, ...] = MOMENT_OFFSETS,
    grid: PitchGrid | None = None,
    pc_params: PitchControlParams | None = None,
    events: pd.DataFrame | None = None,
    fps: float | None = None,
) -> list[Moment]:
    """Pick the moments the report prints and compute their surfaces.

    Only scored losses are eligible: a loss with no tracked ball, no location or
    a truncated window has no surface to draw and is not silently drawn anyway.
    The surfaces are recomputed here at print resolution rather than reused from
    the metric's coarse demo grid -- a handful of surfaces, so the cost is small
    and the printed panel is the one a reader would want to zoom into.

    Args:
        tracking: Canonical tracking frame for the match.
        losses: Scored losses from :func:`pressing.counterpress.score_losses`.
        team: Restrict to losses by this team, or None for either.
        n: How many moments to return. Fewer come back when fewer qualify.
        rank: Which moments count as "top". See :data:`RANK_GLOSS`.
        offsets: Seconds after ``t0`` to evaluate.
        grid: Grid to compute on. Defaults to :func:`print_grid`.
        pc_params: Pitch-control model constants.
        events: Canonical events, used only to name the recovering player.
        fps: Tracking rate. Defaults to ``tracking.attrs["fps"]``.

    Returns:
        Up to ``n`` moments, best first. Empty when nothing qualifies.
    """
    rate = float(fps if fps is not None else tracking.attrs.get("fps", 25.0))
    grid = print_grid() if grid is None else grid
    pc_params = PitchControlParams() if pc_params is None else pc_params

    table = losses
    if "scored" in table.columns:
        table = table[table["scored"].fillna(False).astype(bool)]
    if team is not None:
        table = table[table["losing_team"] == team]
    column, ascending = _rank_key(rank)
    if column not in table.columns or table.empty:
        return []
    table = table[np.isfinite(pd.to_numeric(table[column], errors="coerce").to_numpy(float))]
    if table.empty:
        return []
    chosen = table.sort_values(column, ascending=ascending, kind="stable").head(int(n))

    wanted: set[int] = set()
    for row in chosen.to_dict("records"):
        for offset in offsets:
            wanted.add(int(row["frame"]) + int(round(offset * rate)))
    rows = tracking.loc[tracking["frame"].isin(wanted)]
    by_frame = {int(key): block for key, block in rows.groupby("frame", sort=False)}
    keepers = _gk_jerseys(tracking, rows)
    ball_track = _BallTrack(tracking, fps=rate)
    names = recovering_players(chosen, events, fps=rate)

    moments: list[Moment] = []
    for position, row in enumerate(chosen.to_dict("records"), start=1):
        frames: list[MomentFrame] = []
        f0 = int(row["frame"])
        period = int(row["period"])
        winner, presser = str(row["winning_team"]), str(row["losing_team"])
        for offset in offsets:
            frame = f0 + int(round(offset * rate))
            block = by_frame.get(frame)
            if block is None or block.empty or int(block["period"].iloc[0]) != period:
                continue
            attackers = _player_set(block, winner, keepers.get(winner))
            defenders = _player_set(block, presser, keepers.get(presser))
            if len(attackers) == 0 or len(defenders) == 0:
                continue
            ball, held = ball_track.at(frame, period)
            surface = pitch_control_full(
                attackers,
                defenders,
                ball_pos=ball if ball is not None else held,
                grid=grid,
                params=pc_params,
                offsides=False,
            ).attacking
            frames.append(
                MomentFrame(
                    offset=float(offset),
                    frame=frame,
                    control=np.asarray(surface, dtype=float),
                    attackers=attackers.positions,
                    defenders=defenders.positions,
                    attacker_velocities=attackers.velocities,
                    defender_velocities=defenders.velocities,
                    ball=ball,
                    ball_last_known=held,
                )
            )
        moments.append(
            Moment(
                rank=position,
                loss=dict(row),
                frames=tuple(frames),
                recovering_player=names.get(int(row.get("loss_id", -1))),
            )
        )
    return moments


def _player_set(block: pd.DataFrame, team: str, gk_jersey: str | None) -> PlayerSet:
    """One team's players at one frame, dropping anyone without a position.

    Args:
        block: Tracking rows for a single frame.
        team: ``"home"`` or ``"away"``.
        gk_jersey: That team's goalkeeper jersey, or None.

    Returns:
        The in-frame players.
    """
    rows = block[block["team"] == team]
    xy = rows.loc[:, ["x", "y"]].to_numpy(dtype=float)
    good = np.isfinite(xy).all(axis=1)
    rows = rows.loc[good]
    return players_from_tracking(rows, team, gk_jersey)


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #


def _table_style(base: list[tuple[Any, ...]], regular: str, bold: str) -> TableStyle:
    """Assemble a table style on the registered faces.

    Args:
        base: Style commands specific to the table.
        regular: Regular face name.
        bold: Bold face name.

    Returns:
        The style.
    """
    return TableStyle(
        [
            ("FONT", (0, 0), (-1, -1), regular, 8.2),
            ("FONT", (0, 0), (-1, 0), bold, 8.0),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3.0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3.0),
            ("LEFTPADDING", (0, 0), (-1, -1), 5.0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5.0),
            *base,
        ]
    )


def _kpi_table(
    losses: pd.DataFrame,
    team: str,
    labels: dict[str, str] | None,
    styles: _Styles,
    fonts: tuple[str, str, str],
) -> Table:
    """The cover's head-to-head KPI table.

    Every row is a metric, every team gets a column, and the last column is the
    subject team minus the opponent -- the delta a coach reads first. The
    subject team's column is set in its own identity colour, the same blue or
    orange it is drawn in on every figure.

    Args:
        losses: Scored losses.
        team: The subject team.
        labels: Optional real team names.
        styles: Paragraph styles.
        fonts: ``(regular, bold, italic)``.

    Returns:
        The table.
    """
    regular, bold, _ = fonts
    teams = per_team(losses).set_index("team")
    sides = plots.ordered_teams(list(teams.index) + [team])
    other = [side for side in sides if side != team]
    opponent = other[0] if other else None
    weighting = WEIGHTING_LABELS.get(headline_weighting(losses), "")

    def stat(side: str | None, column: str) -> float:
        """One team's value for one aggregate column, or NaN."""
        if side is None or side not in teams.index or column not in teams.columns:
            return float("nan")
        return float(teams.loc[side, column])

    def half_counts(side: str | None) -> tuple[int, int]:
        """That team's losses in its own half and in the opponent's."""
        if side is None or "x_norm" not in losses.columns:
            return 0, 0
        block = losses[losses["losing_team"] == side]
        norm = pd.to_numeric(block["x_norm"], errors="coerce").to_numpy(dtype=float)
        own = int(np.sum(np.isfinite(norm) & (norm < 0.5)))
        return own, int(np.sum(np.isfinite(norm))) - own

    own_us, opp_us = half_counts(team)
    own_them, opp_them = half_counts(opponent)

    rows: list[tuple[str, str, str, str]] = []

    def add(
        label: str,
        us: float,
        them: float,
        fmt: Any,
        *,
        delta_fmt: Any | None = None,
    ) -> None:
        """Append one metric row, with the delta rendered in its own units."""
        gap = "—"
        if np.isfinite(us) and np.isfinite(them):
            gap = (delta_fmt or (lambda v: _signed(v, 2)))(us - them)
        rows.append((label, fmt(us), fmt(them), gap))

    add(
        f"Counter-press recovery, 5 s ({weighting})",
        stat(team, "mean_recovery"),
        stat(opponent, "mean_recovery"),
        lambda v: _num(v, 3),
        delta_fmt=lambda v: _signed(v, 3),
    )
    add(
        "Regain within 5 s (event stream)",
        stat(team, "regain_rate"),
        stat(opponent, "regain_rate"),
        _pct,
        # Percentages differ in percentage POINTS, and saying so is the
        # difference between "eight tenths of a point" and "eight tenths of a
        # percent", which are not the same number.
        delta_fmt=lambda v: f"{_signed(v * 100.0, 1)} pp",
    )
    add(
        "Possession losses",
        stat(team, "losses"),
        stat(opponent, "losses"),
        lambda v: _int(v),
        delta_fmt=lambda v: _signed(v, 0),
    )
    rows.append(
        (
            "— lost in own half / opponent half",
            f"{own_us} / {opp_us}",
            f"{own_them} / {opp_them}",
            "—",
        )
    )
    add(
        f"EPV conceded per loss ({plots.EPV_DISPLAY_UNIT})",
        stat(team, "mean_epv_conceded") * plots.EPV_DISPLAY_SCALE,
        stat(opponent, "mean_epv_conceded") * plots.EPV_DISPLAY_SCALE,
        lambda v: _signed(v, 2),
    )
    add(
        "Median time to regain (s)",
        stat(team, "median_time_to_regain"),
        stat(opponent, "median_time_to_regain"),
        lambda v: _num(v, 2),
    )

    header = (
        "",
        plots.team_label(team, labels),
        plots.team_label(opponent, labels) if opponent else "—",
        "Difference",
    )
    data = [
        [_p(header[0], styles.caption), header[1], header[2], header[3]],
        *[[_p(row[0], styles.body), row[1], row[2], row[3]] for row in rows],
    ]
    widths = [FRAME_W * 0.46, FRAME_W * 0.18, FRAME_W * 0.18, FRAME_W * 0.18]
    table = Table(data, colWidths=widths, hAlign="LEFT")
    commands: list[tuple[Any, ...]] = [
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("FONT", (1, 1), (1, -1), bold, 10.5),
        ("TEXTCOLOR", (1, 1), (1, -1), _team_ink(team)),
        ("FONT", (2, 1), (2, -1), regular, 9.5),
        ("TEXTCOLOR", (2, 1), (2, -1), _team_ink(opponent)),
        ("TEXTCOLOR", (3, 1), (3, -1), _INK_SOFT),
        ("TEXTCOLOR", (1, 0), (1, 0), _team_ink(team)),
        ("TEXTCOLOR", (2, 0), (2, 0), _team_ink(opponent)),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, _INK),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, _RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 4.0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.0),
    ]
    table.setStyle(_table_style(commands, regular, bold))
    return table


def _phase_table(
    losses: pd.DataFrame,
    labels: dict[str, str] | None,
    fonts: tuple[str, str, str],
) -> Table:
    """The page-2 breakdown by pitch phase, for both teams.

    Args:
        losses: Scored losses.
        labels: Optional real team names.
        fonts: ``(regular, bold, italic)``.

    Returns:
        The table.
    """
    regular, bold, _ = fonts
    phases = per_phase(losses)
    header = [
        "Phase",
        "Team",
        "Losses",
        "Scored",
        "Mean recovery\n5 s",
        f"EPV swing\n({plots.EPV_DISPLAY_UNIT})",
        "Regain\nwithin 5 s",
    ]
    data: list[list[Any]] = [header]
    ink: list[tuple[Any, ...]] = []
    sides = plots.ordered_teams(phases["team"]) if not phases.empty else []
    for phase in PHASES:
        for side in sides:
            block = phases[(phases["phase"] == phase) & (phases["team"] == side)]
            row_index = len(data)
            if block.empty:
                data.append(
                    [
                        phase.replace("_", " "),
                        plots.team_label(side, labels),
                        "0",
                        "0",
                        "—",
                        "—",
                        "—",
                    ]
                )
                ink.append(("TEXTCOLOR", (1, row_index), (1, row_index), _team_ink(side)))
                continue
            entry = block.iloc[0]
            thin = int(entry["scored_losses"]) < plots.MIN_PHASE_LOSSES
            data.append(
                [
                    phase.replace("_", " "),
                    plots.team_label(side, labels),
                    _int(entry["losses"]),
                    _int(entry["scored_losses"]) + (" (low n)" if thin else ""),
                    _num(entry["mean_recovery"], 3),
                    _signed(_scaled_epv(entry["mean_epv_conceded"]), 2),
                    _pct(entry["regain_rate"], 1),
                ]
            )
            ink.append(("TEXTCOLOR", (1, row_index), (1, row_index), _team_ink(side)))
            if thin:
                ink.append(("TEXTCOLOR", (2, row_index), (-1, row_index), _INK_MUTED))
    if len(data) == 1:
        data.append(["—", "—", "0", "0", "—", "—", "—"])

    widths = [FRAME_W * w for w in (0.17, 0.15, 0.11, 0.13, 0.17, 0.15, 0.12)]
    table = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1)
    commands: list[tuple[Any, ...]] = [
        ("BACKGROUND", (0, 0), (-1, 0), _HEADER_FILL),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONT", (0, 0), (-1, 0), bold, 7.4),
        ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, _RULE),
        *ink,
    ]
    for index in range(1, len(data)):
        if index % 2 == 0:
            commands.append(("BACKGROUND", (0, index), (-1, index), _ZEBRA))
    table.setStyle(_table_style(commands, regular, bold))
    return table


def _player_rows(players: pd.DataFrame, team: str | None) -> pd.DataFrame:
    """The players the table prints, in the order it prints them.

    Args:
        players: Output of :func:`pressing.counterpress.per_player`.
        team: Restrict to one team, or None for everyone.

    Returns:
        At most :data:`MAX_PLAYER_ROWS` rows, most-involved first.
    """
    table = players if team is None else players[players["team"] == team]
    if table.empty:
        return table
    order = ["losses_nearby", "minutes"]
    keys = [column for column in order if column in table.columns]
    if keys:
        table = table.sort_values(keys, ascending=False, kind="stable")
    return table.head(MAX_PLAYER_ROWS)


def _player_table(
    players: pd.DataFrame,
    team: str | None,
    goalkeepers: dict[str, Any],
    styles: _Styles,
    fonts: tuple[str, str, str],
) -> tuple[Table, int, int]:
    """The page-6 squad table.

    Args:
        players: Output of :func:`pressing.counterpress.per_player`.
        team: Restrict to one team, or None for everyone.
        goalkeepers: ``attrs["goalkeeper"]`` from the tracking frame, used for
            the only position this data source actually knows.
        styles: Paragraph styles.
        fonts: ``(regular, bold, italic)``.

    Returns:
        ``(table, printed_rows, available_rows)``.
    """
    regular, bold, _ = fonts
    shown = _player_rows(players, team)
    available = len(players if team is None else players[players["team"] == team])
    header = [
        "#",
        "Player",
        "Pos",
        "Min",
        "Losses\nnearby",
        "Recovery\n5 s",
        f"EPV swing\n({plots.EPV_DISPLAY_UNIT})",
        "Build-up",
        "Middle",
        "Final third",
    ]
    data: list[list[Any]] = [header]
    ink: list[tuple[Any, ...]] = []
    keepers = {str(value) for value in goalkeepers.values()}

    for row in shown.to_dict("records"):
        index = len(data)
        player_id = str(row.get("player_id", ""))
        cells = [
            _int(row.get("jersey"), dash=""),
            _p(player_id, styles.body),
            "GK" if player_id in keepers else "—",
            _int(row.get("minutes")),
            _int(row.get("losses_nearby")),
            _num(row.get("mean_recovery"), 2),
            _signed(_scaled_epv(row.get("epv_conceded")), 2),
        ]
        for phase in PHASES:
            count = float(row.get(f"losses_{phase}", 0) or 0)
            cells.append(_num(row.get(f"recovery_{phase}"), 2))
            if count < 5:
                column = 7 + PHASES.index(phase)
                ink.append(("TEXTCOLOR", (column, index), (column, index), _INK_MUTED))
        data.append(cells)
        ink.append(("TEXTCOLOR", (0, index), (0, index), _team_ink(row.get("team"))))

    if len(data) == 1:
        data.append(["", _p("No players on the pitch in this match", styles.body), *[""] * 8])

    widths = [
        FRAME_W * w
        for w in (0.045, 0.215, 0.055, 0.065, 0.095, 0.095, 0.105, 0.108, 0.108, 0.109)
    ]
    table = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1)
    commands: list[tuple[Any, ...]] = [
        ("BACKGROUND", (0, 0), (-1, 0), _HEADER_FILL),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONT", (0, 0), (-1, 0), bold, 7.2),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
        ("ALIGN", (2, 0), (2, -1), "CENTER"),
        ("FONT", (0, 1), (0, -1), bold, 8.2),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, _RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 2.8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.8),
        *ink,
    ]
    for index in range(1, len(data)):
        if index % 2 == 0:
            commands.append(("BACKGROUND", (0, index), (-1, index), _ZEBRA))
    table.setStyle(_table_style(commands, regular, bold))
    return table, len(data) - 1, available


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #


def cover_page(
    losses: pd.DataFrame,
    team: str,
    *,
    subtitle: str,
    labels: dict[str, str] | None,
    styles: _Styles,
    fonts: tuple[str, str, str],
    title: str = "Counter-press report",
) -> list[Flowable]:
    """Page 1: the fixture, the headline numbers and the caveat that frames them.

    Args:
        losses: Scored losses.
        team: The subject team.
        subtitle: Fixture, provider and frame-rate line.
        labels: Optional real team names.
        styles: Paragraph styles.
        fonts: ``(regular, bold, italic)``.
        title: Report title.

    Returns:
        The page's flowables.
    """
    summary = summarise_match(losses)
    story: list[Flowable] = [
        _p(title, styles.title),
        _p(subtitle, styles.subtitle),
        _kpi_table(losses, team, labels, styles, fonts),
        Spacer(1, 7),
        _p(PHASE_CAVEAT, styles.caveat),
        Spacer(1, 6),
    ]
    # The bars sit directly under the caveat because they are the evidence for
    # it: three pairs of bars an order apart, with the sample size printed on
    # each, say more than the sentence does.
    story.append(
        _fig_image(
            plots.phase_bars(per_phase(losses), team_labels=labels, figsize=(7.6, 3.0)),
            FRAME_W,
        )
    )
    story.append(Spacer(1, 8))
    story.append(
        _fig_image(
            plots.recovery_sparkline(losses, team_labels=labels, figsize=(7.6, 2.4)), FRAME_W
        )
    )
    skipped = int(summary.get("n_skipped", 0) or 0)
    reasons = summary.get("skip_reasons") or {}
    scored_note = (
        f"{summary.get('n_losses', 0)} open-play possession losses, "
        f"{summary.get('n_scored', 0)} of them scored."
    )
    if skipped:
        detail = ", ".join(f"{name}: {count}" for name, count in sorted(reasons.items()))
        scored_note += f" {skipped} could not be scored ({detail}); none were dropped."
    ball_out = summary.get("ball_out_count")
    if ball_out is not None:
        scored_note += (
            f" A further {int(ball_out)} balls went out of play and are counted separately, "
            "since nobody can counter-press a throw-in."
        )
    story += [
        Spacer(1, 4),
        _p(scored_note, styles.note),
        Spacer(1, 6),
        _p(
            "The rest of this report: page 2, every loss on one pitch and the same "
            "losses split by phase. Pages 3–5, the five turnovers this ranking picks "
            "out, with the pitch-control surface behind each one. Page 6, the same "
            "numbers per player, for both squads. "
            + EPV_PROVENANCE,
            styles.note,
        ),
    ]
    return story


def loss_map_page(
    losses: pd.DataFrame,
    *,
    labels: dict[str, str] | None,
    styles: _Styles,
    fonts: tuple[str, str, str],
) -> list[Flowable]:
    """Page 2: every loss on one pitch, then the same losses split by phase.

    Args:
        losses: Scored losses.
        labels: Optional real team names.
        styles: Paragraph styles.
        fonts: ``(regular, bold, italic)``.

    Returns:
        The page's flowables.
    """
    # Tall on purpose. The pitch axes is pinned to the pitch's own aspect, so a
    # short figure is limited by its height and comes back pillarboxed -- the
    # image is full width but the pitch inside it is not. Measured: this figure
    # stops growing at about 6.6 in tall, which is where the pitch becomes
    # width-bound and the panel finally fills the page.
    figure = plots.loss_map(losses, team_labels=labels, figsize=(7.6, 6.8))
    return [
        _p("Where the ball was lost", styles.heading),
        _fig_image(figure, FRAME_W),
        Spacer(1, 6),
        _p("Broken down by pitch phase", styles.heading),
        _phase_table(losses, labels, fonts),
        Spacer(1, 5),
        _p(
            "Phases are measured from the losing team's own goal: build-up is that "
            "team's own third, final third is the opponent's. Compare the two teams "
            "within a row; never subtract one row from another — see the cover for why. "
            + plots.EPV_UNIT_GLOSS,
            styles.note,
        ),
        Spacer(1, 4),
        _p(
            "A grey cross on the map is a loss the metric could not score — no located "
            "ball, or a window cut short by the whistle. They are drawn rather than "
            "dropped, because a map that quietly hides its failures flatters the press. "
            + EPV_PROVENANCE,
            styles.note,
        ),
    ]


#: Width of a printed moment panel on pages 3-4, as a fraction of the frame.
#: A pitch panel is inherently wide and short -- measured tight-bbox aspect is
#: about 0.79 -- so two of them at full frame width do not fit on one page.
#: This is the widest they can be and still leave room for their captions.
MOMENT_PANEL_FRACTION: Final[float] = 0.73


def _centred(content: list[Flowable], width: float) -> Table:
    """Centre a block of flowables in the frame.

    A panel narrower than the frame, left-aligned, reads as a page with a
    margin error on one side. The same panel centred reads as a decision.

    Args:
        content: Flowables to stack.
        width: Block width in points.

    Returns:
        A single-column table holding the block.
    """
    table = Table([[item] for item in content], colWidths=[width], hAlign="CENTER")
    table.setStyle(
        TableStyle(
            [
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def _moment_caption(moment: Moment, labels: dict[str, str] | None, styles: _Styles) -> Paragraph:
    """The line under a printed moment panel.

    Args:
        moment: The moment.
        labels: Optional real team names.
        styles: Paragraph styles.

    Returns:
        A caption paragraph.
    """
    row = moment.loss
    parts = [
        f"{moment.rank}. {_clock(row.get('t'))}",
        f"{plots.team_label(moment.losing_team, labels)} lost the ball"
        + (f" ({row.get('losing_player')})" if row.get("losing_player") else ""),
    ]
    if moment.recovering_player:
        parts.append(f"won back by {moment.recovering_player}")
    else:
        parts.append("not won back inside 5 s")
    parts.append(f"recovery {_num(row.get('recovery'), 2)}")
    swing = plots.format_epv(row.get("epv_conceded"))
    if swing:
        parts.append(f"EPV swing {swing}")
    phase = row.get("phase")
    if phase:
        parts.append(str(phase).replace("_", " "))
    return _p("  ·  ".join(parts), styles.caption)


def _moment_image(
    moment: Moment,
    width_pt: float,
    *,
    labels: dict[str, str] | None,
    figsize: tuple[float, float],
    styles: _Styles,
) -> Flowable:
    """The panel for one moment, or a stated absence.

    Args:
        moment: The moment.
        width_pt: Target width in points.
        labels: Optional real team names.
        figsize: Figure size in inches.
        styles: Paragraph styles.

    Returns:
        An image, or a paragraph saying why there is none.
    """
    frame = moment.at(PANEL_OFFSET)
    if frame is None:
        return _p(
            f"{moment.rank}. No tracking frames inside this moment's window, so no "
            "control surface could be drawn for it.",
            styles.note,
        )
    figure = plots.moment_panel(
        frame.control,
        ball=frame.ball,
        ball_last_known=frame.ball_last_known,
        loss_xy=moment.loss_xy,
        attackers=frame.attackers,
        defenders=frame.defenders,
        attacking_team=moment.winning_team,
        attacker_velocities=frame.attacker_velocities,
        defender_velocities=frame.defender_velocities,
        title=f"{moment.rank}.  {_clock(moment.loss.get('t'))}  ·  t₀ + {frame.offset:g} s",
        caption="",
        team_labels=labels,
        figsize=figsize,
    )
    return _fig_image(figure, width_pt)


def moment_pages(
    moments: list[Moment],
    *,
    team: str,
    rank: MomentRank,
    labels: dict[str, str] | None,
    styles: _Styles,
    per_page: int = 2,
) -> list[list[Flowable]]:
    """Pages 3-4: the printed moments, two to a page.

    Args:
        moments: Moments to print, in order.
        team: The subject team, for the heading.
        rank: Which ranking produced them, for the heading gloss.
        labels: Optional real team names.
        styles: Paragraph styles.
        per_page: Moments on each page.

    Returns:
        One list of flowables per page. Always two pages, so the report's page
        count does not depend on how many moments a match happened to have.
    """
    pages: list[list[Flowable]] = []
    who = plots.team_label(team, labels)
    for page in range(2):
        block = moments[page * per_page : (page + 1) * per_page]
        first, last = page * per_page + 1, (page + 1) * per_page
        story: list[Flowable] = [
            _p(f"Counter-press moments {first}–{last} — {who}", styles.heading)
        ]
        if page == 0:
            story.append(_p(RANK_GLOSS[rank], styles.body))
            story.append(Spacer(1, 6))
        if not block:
            story.append(
                _p(
                    "No further scorable moments in this match. The metric is computed "
                    "at every possession loss; only losses with a located ball and a "
                    "complete window can be drawn.",
                    styles.note,
                )
            )
            pages.append(story)
            continue
        width = FRAME_W * MOMENT_PANEL_FRACTION
        for moment in block:
            story.append(
                _centred(
                    [
                        _moment_image(
                            moment, width, labels=labels, figsize=(6.4, 4.6), styles=styles
                        ),
                        _moment_caption(moment, labels, styles),
                    ],
                    width,
                )
            )
            story.append(Spacer(1, 8))
        story.append(
            _p(
                "Surface is the ball-winning team's pitch control two seconds after the "
                "turnover; the crosshair is where the ball was lost. "
                + plots.EPV_UNIT_GLOSS,
                styles.note,
            )
        )
        pages.append(story)
    return pages


def _moment_stats(
    moment: Moment,
    weighting: str,
    fonts: tuple[str, str, str],
) -> Table:
    """The strip of numbers under the enlarged moment.

    The panel shows one instant; these are the whole window, so the picture and
    the metric can be checked against each other rather than taken on trust.

    Args:
        moment: The moment.
        weighting: Headline weighting, which decides whether the two endpoint
            shares are read off the area or the EPV columns.
        fonts: ``(regular, bold, italic)``.

    Returns:
        A two-row table.
    """
    regular, bold, _ = fonts
    row = moment.loss
    regain = row.get("time_to_regain")
    back = (
        f"{float(regain):.2f} s"
        if regain is not None and np.isfinite(float(regain))
        else "not inside 5 s"
    )
    labels = [
        "Control share at t₀",
        "at t₀ + 5 s",
        "Mean over the window",
        f"EPV swing ({plots.EPV_DISPLAY_UNIT})",
        "Ball back after",
    ]
    values = [
        _num(row.get(f"share_t0_{weighting}"), 2),
        _num(row.get(f"share_end_{weighting}"), 2),
        _num(row.get("recovery"), 2),
        _signed(_scaled_epv(row.get("epv_conceded")), 2),
        back,
    ]
    table = Table(
        [labels, values], colWidths=[FRAME_W / len(labels)] * len(labels), hAlign="LEFT"
    )
    table.setStyle(
        _table_style(
            [
                ("FONT", (0, 0), (-1, 0), regular, 7.2),
                ("TEXTCOLOR", (0, 0), (-1, 0), _INK_MUTED),
                ("FONT", (0, 1), (-1, 1), bold, 12.0),
                ("TEXTCOLOR", (0, 1), (-1, 1), _team_ink(moment.losing_team)),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("LINEABOVE", (0, 0), (-1, 0), 0.6, _RULE),
                ("LINEBELOW", (0, 1), (-1, 1), 0.6, _RULE),
                ("TOPPADDING", (0, 0), (-1, -1), 3.0),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 0.0),
                ("BOTTOMPADDING", (0, 1), (-1, 1), 4.0),
                ("LEFTPADDING", (0, 0), (0, -1), 0.0),
            ],
            regular,
            bold,
        )
    )
    return table


def moment_detail_page(
    moment: Moment | None,
    *,
    team: str,
    labels: dict[str, str] | None,
    styles: _Styles,
    fonts: tuple[str, str, str],
    weighting: str = "area",
) -> list[Flowable]:
    """Page 5: one moment large, plus the window as a thumbnail strip.

    Args:
        moment: The moment to enlarge, or None when the match had fewer than
            five scorable ones.
        team: The subject team, for the heading.
        labels: Optional real team names.
        styles: Paragraph styles.
        fonts: ``(regular, bold, italic)``.
        weighting: Headline weighting, for the endpoint shares.

    Returns:
        The page's flowables.
    """
    who = plots.team_label(team, labels)
    if moment is None:
        return [
            _p(f"Counter-press moment in detail — {who}", styles.heading),
            _p(
                "This match had fewer scorable possession losses than the report prints "
                "panels for, so there is no fifth moment to enlarge. Nothing has been "
                "substituted for it.",
                styles.body,
            ),
        ]
    story: list[Flowable] = [
        _p(
            f"Moment {moment.rank} in detail — {_clock(moment.loss.get('t'))} — {who}",
            styles.heading,
        ),
        _moment_image(moment, FRAME_W, labels=labels, figsize=(7.6, 5.0), styles=styles),
        _moment_caption(moment, labels, styles),
        Spacer(1, 8),
        _moment_stats(moment, weighting, fonts),
        Spacer(1, 10),
    ]
    strip_frames = [moment.at(offset) for offset in STRIP_OFFSETS]
    usable = [frame for frame in strip_frames if frame is not None]
    if usable:
        figure = plots.moment_strip(
            [frame.control for frame in usable],
            offsets=[frame.offset for frame in usable],
            balls=[frame.ball for frame in usable],
            ball_last_known=[frame.ball_last_known for frame in usable],
            loss_xy=moment.loss_xy,
            attackers=[frame.attackers for frame in usable],
            defenders=[frame.defenders for frame in usable],
            attacking_team=moment.winning_team,
            team_labels=labels,
        )
        story.append(_fig_image(figure, FRAME_W))
    story.append(Spacer(1, 4))
    story.append(
        _p(
            "The strip is the same moment at the start, the middle and the end of the "
            "five-second window, so the surface is seen to move rather than asserted to. "
            + EPV_PROVENANCE,
            styles.note,
        )
    )
    return story


def player_table_page(
    players: pd.DataFrame,
    team: str | None,
    *,
    goalkeepers: dict[str, Any],
    labels: dict[str, str] | None,
    radius_m: float,
    styles: _Styles,
    fonts: tuple[str, str, str],
    opponent: str | None = None,
) -> list[Flowable]:
    """Page 6: the squad table and the method note that defines it.

    The opponent's table is printed underneath the subject team's, at the same
    scale. Recovery is heavily driven by where the ball was lost, so a number
    is only worth anything against another number computed the same way in the
    same match -- and the opponent's squad is the only such comparison a single
    match report has.

    Args:
        players: Output of :func:`pressing.counterpress.per_player`.
        team: Restrict the first table to one team, or None for everyone.
        goalkeepers: ``attrs["goalkeeper"]`` from the tracking frame.
        labels: Optional real team names.
        radius_m: The proximity radius the attribution used.
        styles: Paragraph styles.
        fonts: ``(regular, bold, italic)``.
        opponent: The other team, given a second table of its own. None prints
            only the first table.

    Returns:
        The page's flowables.
    """
    table, printed, available = _player_table(players, team, goalkeepers, styles, fonts)
    who = plots.team_label(team, labels) if team else "both teams"
    story: list[Flowable] = [
        _p(f"Player counter-press table — {who}", styles.heading),
        table,
    ]
    if printed < available:
        story.append(
            _p(
                f"Showing the {printed} players most often near a turnover, of "
                f"{available} who played.",
                styles.note,
            )
        )
    if opponent is not None and not players[players["team"] == opponent].empty:
        other, other_printed, other_available = _player_table(
            players, opponent, goalkeepers, styles, fonts
        )
        story += [
            Spacer(1, 10),
            _p(
                f"For comparison — {plots.team_label(opponent, labels)}",
                styles.heading,
            ),
            other,
        ]
        if other_printed < other_available:
            story.append(
                _p(
                    f"Showing {other_printed} of {other_available} who played.",
                    styles.note,
                )
            )
    story += [
        Spacer(1, 9),
        _p(NEARBY_GLOSS.format(radius=radius_m), styles.body),
        Spacer(1, 4),
        _p(
            "Minutes are tracked presence, not an assumed 90, and the tracking clock "
            "runs continuously across half time, so a full match is about 97 minutes on "
            "this scale — per-90 rates must divide by these numbers, not by 90. "
            "Recovery is the mean over that player's nearby losses; the three phase "
            "columns are the same mean split by where the ball was lost, and a phase "
            "column resting on fewer than five losses is greyed because the difference "
            "between two such numbers is noise. Position is shown only where the data "
            "source states one, which for this provider is the goalkeeper and nobody "
            "else. " + plots.EPV_UNIT_GLOSS,
            styles.note,
        ),
        Spacer(1, 4),
        _p(EPV_PROVENANCE, styles.note),
    ]
    return story


# --------------------------------------------------------------------------- #
# Page furniture
# --------------------------------------------------------------------------- #


def _draw_footer(
    canvas: Canvas,
    *,
    fixture: str,
    disclosure: str | None,
    fonts: tuple[str, str, str],
) -> None:
    """Draw the rule, the running head and the source credit.

    The disclosure, when there is one, is drawn here as flat opaque text as
    well as diagonally across the page. That redundancy is the point: the
    diagonal watermark is alpha-blended, so it can vanish in a greyscale
    photocopy, and it is rotated, so text extraction reorders its characters
    and a machine looking for the string will not find it. This line survives
    both.

    Args:
        canvas: Active canvas.
        fixture: Running-head text.
        disclosure: Proxy disclosure, or None.
        fonts: ``(regular, bold, italic)``.
    """
    regular, bold, _ = fonts
    canvas.saveState()
    canvas.setStrokeColor(_RULE)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, 13.4 * mm, PAGE_W - MARGIN, 13.4 * mm)

    canvas.setFont(regular, 7.4)
    canvas.setFillColor(_INK_SOFT)
    canvas.drawString(MARGIN, 9.9 * mm, fixture)
    canvas.drawRightString(
        PAGE_W - MARGIN, 9.9 * mm, f"page {canvas.getPageNumber()} of {N_PAGES}"
    )

    canvas.setFont(regular, 6.6)
    canvas.setFillColor(_INK_MUTED)
    canvas.drawString(MARGIN, 6.4 * mm, DATA_CREDIT)
    if disclosure:
        canvas.setFont(bold, 7.4)
        canvas.setFillColor(_ALERT)
        canvas.drawRightString(PAGE_W - MARGIN, 6.4 * mm, f"{disclosure} — {PROXY_SUBTITLE}")
    canvas.restoreState()


def _draw_watermark(canvas: Canvas, text: str, fonts: tuple[str, str, str]) -> None:
    """Stamp the disclosure diagonally across the page.

    Called from the page template's ``onPageEnd`` hook, never ``onPage``:
    ``onPage`` runs before the frame's flowables are drawn, which puts the
    watermark underneath them, and a full-width heatmap then hides it
    completely while every text-layer test still passes.

    The size is derived from the string rather than fixed, so a longer
    disclosure shrinks to fit the page diagonal instead of running off the
    corners.

    Args:
        canvas: Active canvas.
        text: Watermark text.
        fonts: ``(regular, bold, italic)``.
    """
    _, bold, _ = fonts
    diagonal = float(np.hypot(PAGE_W, PAGE_H))
    angle = float(np.degrees(np.arctan2(PAGE_H, PAGE_W)))
    width_at_1pt = pdfmetrics.stringWidth(text, bold, 1.0) or 1.0
    size = min(40.0, 0.78 * diagonal / width_at_1pt)

    canvas.saveState()
    canvas.translate(PAGE_W / 2.0, PAGE_H / 2.0)
    canvas.rotate(angle)
    canvas.setFont(bold, size)
    canvas.setFillColor(colors.Color(0.78, 0.13, 0.13, alpha=0.16))
    canvas.drawCentredString(0.0, 0.0, text)
    canvas.setFont(bold, size * 0.36)
    canvas.drawCentredString(0.0, -size * 0.80, PROXY_SUBTITLE)
    canvas.restoreState()


# --------------------------------------------------------------------------- #
# The build
# --------------------------------------------------------------------------- #


def _subtitle(
    tracking: pd.DataFrame,
    losses: pd.DataFrame,
    team: str,
    labels: dict[str, str] | None,
    provider: str | None,
    match_date: str | None,
) -> tuple[str, str, str]:
    """The cover subtitle and the running head.

    Nothing is invented: a field the source does not carry -- a kick-off date,
    for most sample data -- is simply not printed, rather than filled with the
    build date, which would read as a match date to anyone holding the paper.

    Args:
        tracking: Canonical tracking frame.
        losses: Scored losses, for the match id when tracking has none.
        team: The subject team.
        labels: Optional real team names.
        provider: Data provider name. Defaults to the frame's ``attrs``.
        match_date: Kick-off date, if the caller knows one.

    Returns:
        ``(subtitle, running_head, fixture)``.
    """
    attrs = tracking.attrs
    fixture = (
        f"{plots.team_label('home', labels)} vs {plots.team_label('away', labels)}"
    )
    source = provider or str(attrs.get("source", "") or "").strip()
    source_name = {"metrica": "Metrica Sports"}.get(source.lower(), source) or "unknown provider"
    match_id = attrs.get("match_id") or losses.attrs.get("match_id")
    fps = attrs.get("fps")

    parts = [fixture]
    if match_date:
        parts.append(str(match_date))
    if match_id:
        parts.append(f"match {match_id}")
    parts.append(f"{source_name} tracking")
    if fps:
        parts.append(f"{float(fps):g} Hz")
    parts.append(f"report for {plots.team_label(team, labels)}")
    head = f"pressing — counter-press report — {fixture}"
    return "  ·  ".join(parts), head, fixture


def build_report(
    tracking: pd.DataFrame,
    losses: pd.DataFrame,
    team: str,
    out_path: Path,
    events: pd.DataFrame | None = None,
    watermark: str | None = None,
    *,
    players: pd.DataFrame | None = None,
    moments: list[Moment] | None = None,
    team_labels: dict[str, str] | None = None,
    provider: str | None = None,
    match_date: str | None = None,
    title: str = "Counter-press report",
    rank: MomentRank = "costly",
    radius_m: float = DEFAULT_RADIUS_M,
    grid: PitchGrid | None = None,
    author: str = "pressing",
) -> Path:
    """Render the six-page report.

    Args:
        tracking: Canonical tracking frame for the match.
        losses: Scored losses from
            :func:`pressing.counterpress.score_losses`.
        team: The team the report is written for, ``"home"`` or ``"away"``.
        out_path: Destination PDF path. Parent directories are created.
        events: Canonical events frame. Optional, and used only to name the
            player who won the ball back in a moment caption.
        watermark: Disclosure to stamp across every page, in the footer and in
            the PDF metadata. :data:`PROXY_WATERMARK` for the event-proxy path.
            None for a report built from real tracking.
        players: Pre-computed per-player aggregates. Computed when omitted.
        moments: Pre-computed moments. Computed when omitted.
        team_labels: Real team names keyed by ``"home"`` / ``"away"``.
        provider: Data provider name for the subtitle. Defaults to the
            tracking frame's own ``attrs["source"]``.
        match_date: Kick-off date, if known. Omitted from the subtitle rather
            than guessed when it is not.
        title: Report title on the cover.
        rank: Which losses get printed as moments. See :data:`RANK_GLOSS`.
        radius_m: Proximity radius used for the player attribution, for the
            method footnote.
        grid: Grid the printed panels are computed on. Defaults to
            :func:`print_grid`.
        author: PDF author metadata.

    Returns:
        The path written.

    Raises:
        ValueError: If ``team`` is not ``"home"`` or ``"away"``.
    """
    if team not in ("home", "away"):
        raise ValueError(f"team must be 'home' or 'away', got {team!r}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fonts = register_pdf_fonts()
    styles = _build_styles(*fonts)
    labels = dict(team_labels) if team_labels else None

    if players is None:
        players = per_player(losses, tracking, radius_m)
    if moments is None:
        moments = render_moments(
            tracking,
            losses,
            team,
            n=TOP_MOMENTS,
            rank=rank,
            grid=grid,
            events=events,
        )

    subtitle, head, fixture = _subtitle(
        tracking, losses, team, labels, provider, match_date
    )
    disclosure = watermark.strip() if watermark and watermark.strip() else None

    pages: list[list[Flowable]] = [
        cover_page(
            losses,
            team,
            subtitle=subtitle,
            labels=labels,
            styles=styles,
            fonts=fonts,
            title=title,
        ),
        loss_map_page(losses, labels=labels, styles=styles, fonts=fonts),
        *moment_pages(moments, team=team, rank=rank, labels=labels, styles=styles),
        moment_detail_page(
            moments[TOP_MOMENTS - 1] if len(moments) >= TOP_MOMENTS else None,
            team=team,
            labels=labels,
            styles=styles,
            fonts=fonts,
            weighting=headline_weighting(losses),
        ),
        player_table_page(
            players,
            team,
            goalkeepers=dict(tracking.attrs.get("goalkeeper", {})),
            labels=labels,
            radius_m=radius_m,
            styles=styles,
            fonts=fonts,
            opponent="away" if team == "home" else "home",
        ),
    ]
    if len(pages) != N_PAGES:  # pragma: no cover - guards a future edit
        raise ValueError(f"the report template must build {N_PAGES} pages, got {len(pages)}")

    subject = (
        f"{disclosure}: {PROXY_SUBTITLE}. Counter-press report."
        if disclosure
        else "Counter-press report from tracking data."
    )
    doc = BaseDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=BOTTOM_MARGIN,
        title=f"{title} — {fixture}",
        author=author,
        subject=subject,
        creator="pressing",
    )

    def on_page(canvas: Canvas, _doc: BaseDocTemplate) -> None:
        """Draw the footer, before the frame's flowables."""
        _draw_footer(canvas, fixture=head, disclosure=disclosure, fonts=fonts)

    def on_page_end(canvas: Canvas, _doc: BaseDocTemplate) -> None:
        """Draw the watermark, after the frame's flowables. See the module docstring."""
        if disclosure:
            _draw_watermark(canvas, disclosure, fonts)

    doc.addPageTemplates(
        [
            PageTemplate(
                id="pressing",
                frames=[Frame(MARGIN, BOTTOM_MARGIN, FRAME_W, FRAME_H, id="body")],
                onPage=on_page,
                onPageEnd=on_page_end,
            )
        ]
    )

    story: list[Flowable] = []
    for index, page in enumerate(pages):
        # Shrink-to-fit rather than spill: the page count is part of the
        # contract, and real data arrives with long names and wrapping captions.
        story.append(KeepInFrame(FRAME_W, FRAME_H, page, mode="shrink"))
        if index < len(pages) - 1:
            story.append(PageBreak())
    doc.build(story)
    return out_path


