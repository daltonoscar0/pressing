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
6. **Player table.** A boxed warning, then minutes, losses nearby, recovery and
   its residual against a conditional expectation, EPV swing and the three
   phase splits, with the method footnote that defines "nearby".

Honesty is part of the layout, not an appendix
----------------------------------------------
Several claims in this report are weaker than they look, so each is printed
where the reader cannot miss it rather than left for a README nobody opens.
Every figure below is measured on the match in hand by
:func:`measure_disclosure` and is never a constant: a report built on other
data prints that data's numbers, or omits the sentence when it cannot compute
one. The values quoted here are the bundled Metrica match, reproducible with
``score_losses`` and :mod:`pressing.baseline`.

* **The per-player recovery column is mostly field position.** A cubic in the
  attacking fraction of the turnover -- position, nothing else -- reproduces
  the printed column at ``r = +0.9771`` over the 27 players it prints one for
  (``R2 = 0.9547``, Spearman ``+0.9402``); the player's own mean turnover
  position alone reaches ``R2 = 0.9583``, and at loss level a straight line in
  ``x_norm`` explains ``R2 = 0.8212`` of 255 scored losses. Page 6 opens with
  that number in a boxed warning above the table, because page 6 is the page
  that leaves the building on its own. See :func:`_rank_warning` and
  :data:`RANK_WARNING_HEADING`.
* **The per-phase columns carry no between-player signal on one match.** The
  report tells the reader to compare within a phase; a 5,000-shuffle
  permutation test, shuffled inside a squad, gives ``p = 0.97`` (build-up),
  ``0.86`` (middle) and ``0.16`` (final third), with the build-up spread
  *smaller* than chance. Each p-value is printed in its own column heading and
  the verdict is in the warning. The residual column is no better --
  ``p = 0.67`` -- and says so.
* **A thin cell prints no number at all.** Below
  :data:`pressing.baseline.MIN_LOSSES_FOR_PLAYER` losses a cell reads
  :data:`SUPPRESSED`; on this match that silences 6 of the 28 rows.
* **The headline metric is not a claim that the ball came back.** Recovery is a
  control share and regain is a possession event; measured over the same 255
  losses they correlate at ``r = +0.071``. The cover stacks the two rows, so a
  tinted note stating that correlation sits between them.
* **Recovery is not comparable across pitch phases.** It is dominated by where
  the ball was lost, not by how hard anyone pressed -- :func:`per_phase` in
  :mod:`pressing.counterpress` records the measured spread. The caveat is
  printed on the cover, next to the very numbers it qualifies, and again under
  the phase table on page 2. The cover also prints one pooled cross-phase
  difference, so directly beneath it the page shows the check its own caveat
  demands: both squads reweighted to the match's phase mix, and whether the
  sign survives. On this match it does, ``+0.027`` standardised against
  ``+0.022`` raw.
* **A loss the map cannot draw is accounted for, not boasted about.** Page 2
  used to promise grey crosses for unscored losses unconditionally; a loss with
  no ball position has nowhere to be drawn and, having no position, no pitch
  third either, so it is missing from the map *and* from the phase table. On
  this match that is exactly the one unscored loss, which is why the cover
  counts 256 and page 2's Losses column adds to 255. Both pages now say so.
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

That shrink has one trap, and it is why every block of prose on the cover and
on page 6 is boxed by :func:`_fixed_width`. ``KeepInFrame`` in ``shrink`` mode
re-wraps its content at ``availWidth * scale`` and then draws the result scaled
back down: a paragraph reflows into the wider measure and comes out full width,
a table with fixed column widths cannot, and an image is fixed too. Mixing them
puts 367 pt of squad table under 496 pt of footnote -- measured, on the page-6
that first carried the warning. Boxed, everything shrinks together and an
overflowing page reads as a slightly smaller page instead of a broken one.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
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

from pressing import baseline, plots
from pressing.counterpress import (
    DEFAULT_RADIUS_M,
    WEIGHTING_LABELS,
    headline_weighting,
    per_phase,
    per_player,
    per_team,
    summarise_match,
)

# The long (loss_id, player_id) credit table behind the page-6 rows. It is
# private to ``counterpress`` because nothing outside that module needed it
# until this one did: the permutation test on page 6 has to shuffle values
# among the losses each player was actually credited with, and re-deriving that
# credit here would risk a page whose warning disagrees with the table above it.
from pressing.counterpress import _attribution as _credited_pressers
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
    "MIN_PERMUTATION_GROUPS",
    "MOMENT_OFFSETS",
    "N_PAGES",
    "PERMUTATION_SEED",
    "PERMUTATION_SHUFFLES",
    "PHASE_CAVEAT",
    "POLARITY_LEGEND",
    "PROXY_SUBTITLE",
    "PROXY_WATERMARK",
    "RANK_WARNING_HEADING",
    "STRIP_OFFSETS",
    "SUPPRESSED",
    "TOP_MOMENTS",
    "Disclosure",
    "Moment",
    "MomentFrame",
    "build_report",
    "cover_page",
    "loss_map_page",
    "measure_disclosure",
    "moment_detail_page",
    "moment_pages",
    "player_table_page",
    "print_grid",
    "recovering_players",
    "register_pdf_fonts",
    "render_moments",
    "with_expected_recovery",
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

#: The page-6 heading a reader has to walk past to reach the squad table.
#:
#: Worded as an instruction rather than a hedge. "This metric has limitations"
#: is read as boilerplate and skipped; a sentence that names the thing the
#: reader was about to do is not.
RANK_WARNING_HEADING: Final[str] = (
    "Before you rank anybody: this table is mostly a map of where each player plays"
)

#: What a per-player cell reads when too few losses stand behind it.
#:
#: Deliberately not a dash and deliberately not blank. A dash reads as "the
#: data is missing", which is false -- the data is there and is too thin to
#: print -- and a blank reads as zero. This says what happened.
SUPPRESSED: Final[str] = "n<{minimum}"

#: Shuffles behind every permutation p-value the report prints. The audit that
#: forced this page used 5,000; matching it makes the two numbers comparable.
PERMUTATION_SHUFFLES: Final[int] = 5000

#: Seed for those shuffles.
#:
#: Fixed for the same reason :data:`pressing.counterpress.BOOTSTRAP_SEED` is:
#: the demo PDF is required to be byte-identical between runs, and a p-value
#: that wandered in the third decimal place would break that on every build.
PERMUTATION_SEED: Final[int] = 20260826

#: Fewest players a permutation test needs before its p-value is printed.
#: Below this the null is built from so few groups that the statistic is not
#: worth putting on paper.
MIN_PERMUTATION_GROUPS: Final[int] = 3

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
#: Panel fill behind a warning. Pale enough to print, dark enough to see.
_ALERT_TINT = colors.HexColor("#fbeeec")

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
    alert: ParagraphStyle
    alert_heading: ParagraphStyle
    dense: ParagraphStyle


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
        # Warnings are set in the body face at body size, not shrunk into the
        # note face. A caveat printed smaller than the number it qualifies is
        # a caveat the layout has already decided is less important.
        alert=ParagraphStyle(
            "pressing-alert",
            parent=body,
            fontSize=8.0,
            leading=10.6,
            textColor=_INK,
            spaceAfter=3,
        ),
        alert_heading=ParagraphStyle(
            "pressing-alert-heading",
            parent=body,
            fontName=bold,
            fontSize=10.6,
            leading=13.0,
            textColor=_ALERT,
            spaceAfter=5,
        ),
        # Page 6 stacks two full squads, a warning and three method notes into
        # one frame. A player-name cell set in the body face leads at 11.6 pt
        # and, over thirty rows, is on its own what pushed the page past the
        # frame -- at which point KeepInFrame shrank the whole page, warning
        # included, to 72% and the tables came out 367 pt wide inside a 510 pt
        # frame. The name cell has to wrap, so it stays a paragraph; it is set
        # in a tighter one.
        dense=ParagraphStyle(
            "pressing-dense",
            parent=body,
            fontSize=7.4,
            leading=8.4,
            textColor=_INK_SOFT,
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
        The formatted number, or ``dash``. A value that rounds to zero comes
        back without a sign: "-0.000" is a rounding artefact of a tiny negative
        number and reads on paper as a measured negative quantity, which it is
        not.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return dash
    if not np.isfinite(number):
        return dash
    text = f"{number:.{digits}f}"
    return text.lstrip("-") if float(text) == 0.0 else text


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
# What the report has to admit about itself
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Disclosure:
    """Every weakness this report prints about itself, measured on its own match.

    Nothing in here is a constant and nothing is copied from a README. Each
    field is computed by :func:`measure_disclosure` from the same losses the
    pages are drawn from, so a report built on a different match prints that
    match's numbers and a report built on a match where the metric behaves
    better says so.

    Attributes:
        n_scored: Scored losses behind every figure below.
        regain_r: Correlation between the headline recovery share and whether
            the ball was actually regained inside the window. The cover stacks
            those two numbers, so the cover states this.
        position_r: Correlation between the printed per-player recovery column
            and a prediction made from turnover position alone -- a cubic in
            the attacking fraction, fitted at loss level, averaged over each
            player's credited losses. NaN when it cannot be fitted.
        position_rho: The same comparison as a rank correlation.
        position_rows: Players both columns exist for.
        printed_rows: Rows the page-6 tables print.
        suppressed_rows: How many of them have too few losses to show a value.
        min_losses: The suppression threshold,
            :data:`pressing.baseline.MIN_LOSSES_FOR_PLAYER`.
        level_interval: Median width of the 95% bootstrap interval on a printed
            recovery value.
        level_span: Distance from the top to the bottom of that column.
        residual_interval: The same median width for the residual column.
        residual_span: The residual column's own top-to-bottom span.
        residual_clear_of_zero: Printed players whose residual interval
            excludes zero.
        residual_p: Permutation p-value for the pooled residual separating
            players by more than chance.
        residual_out_of_sample: Whether the expectation behind the residual was
            fitted without seeing the loss it scores. False on a single match,
            and the page says so.
        phase_p: Permutation p-value per phase for the raw recovery column.
        phase_residual_p: The same for the residual.
        phase_printed: How many of the printed rows actually show a value in
            each phase column. A column that prints nothing is a fact about the
            match the page has to state, not a blank the reader has to notice.
        raw_difference: Subject team's pooled recovery minus the opponent's, as
            the cover prints it.
        standardised_difference: The same difference with both teams reweighted
            to the match's own phase mix -- the check the cover's own caveat
            demands before a pooled difference may be read.
        standardised: Each team's phase-mix-standardised recovery.
        half_counts: ``{team: (own half, opponent half)}`` loss counts, the
            reason the standardisation is not a formality.
        unscored: Losses the metric could not score.
        unscored_drawn: How many of those have a location and so reach the map
            as grey crosses.
        unscored_undrawable: How many have no location at all and therefore
            cannot appear on any map.
        total_losses: Every detected loss, scored or not -- the count the cover
            prints.
        phase_rows: Losses carrying a pitch phase, which is what page 2's phase
            table sums to. A loss with no ball position has no phase either, so
            when the two counts differ the pages must say where the difference
            went rather than let a reader add them up and find 255 under a
            headline of 256.
    """

    n_scored: int = 0
    regain_r: float = float("nan")
    position_r: float = float("nan")
    position_rho: float = float("nan")
    position_rows: int = 0
    printed_rows: int = 0
    suppressed_rows: int = 0
    min_losses: int = baseline.MIN_LOSSES_FOR_PLAYER
    level_interval: float = float("nan")
    level_span: float = float("nan")
    residual_interval: float = float("nan")
    residual_span: float = float("nan")
    residual_clear_of_zero: int = 0
    residual_p: float = float("nan")
    residual_out_of_sample: bool = False
    phase_p: dict[str, float] = field(default_factory=dict)
    phase_residual_p: dict[str, float] = field(default_factory=dict)
    phase_printed: dict[str, int] = field(default_factory=dict)
    raw_difference: float = float("nan")
    standardised_difference: float = float("nan")
    standardised: dict[str, float] = field(default_factory=dict)
    half_counts: dict[str, tuple[int, int]] = field(default_factory=dict)
    unscored: int = 0
    unscored_drawn: int = 0
    unscored_undrawable: int = 0
    total_losses: int = 0
    phase_rows: int = 0

    @property
    def phase_table_short_by(self) -> int:
        """Losses the phase table cannot show because they carry no phase."""
        return max(0, self.total_losses - self.phase_rows)

    @property
    def has_position_check(self) -> bool:
        """Whether the position-only comparison produced a usable number."""
        return bool(np.isfinite(self.position_r)) and self.position_rows >= 3

    @property
    def standardisation_holds(self) -> bool:
        """Whether the pooled team difference keeps its sign once standardised."""
        raw, adjusted = self.raw_difference, self.standardised_difference
        if not (np.isfinite(raw) and np.isfinite(adjusted)):
            return False
        return bool(raw * adjusted > 0.0)


def with_expected_recovery(losses: pd.DataFrame, tracking: pd.DataFrame) -> pd.DataFrame:
    """Attach the conditional baseline's expectation and residual, if it fits.

    The report is the only caller that needs them, so it computes them rather
    than demanding that every path into :func:`build_report` remember to. When
    the baseline cannot be fitted -- no scored loss, a tracking frame that does
    not cover the turnovers -- the frame comes back with the two columns
    present and empty, which is exactly what
    :func:`pressing.counterpress.per_player` expects of a match with no
    baseline, so no page has to branch on it.

    Args:
        losses: Scored losses.
        tracking: Canonical tracking frame for the same match.

    Returns:
        A copy of ``losses`` carrying
        :data:`pressing.baseline.RESIDUAL_COLUMNS`.
    """
    try:
        features = baseline.loss_features(losses, tracking)
        return baseline.add_expected_recovery(losses, features)
    except (KeyError, ValueError, IndexError):
        blank = losses.copy()
        blank.attrs.update(losses.attrs)
        for column in baseline.RESIDUAL_COLUMNS:
            blank[column] = np.nan
        blank.attrs["residual_out_of_sample"] = False
        return blank


def _rank_r(left: np.ndarray, right: np.ndarray) -> float:
    """Spearman correlation, as Pearson on the ranks.

    Args:
        left: First sample.
        right: Second sample, the same length.

    Returns:
        The rank correlation, or NaN when either side is constant.
    """
    if left.size < 3:
        return float("nan")
    a = pd.Series(left).rank().to_numpy(dtype=np.float64)
    b = pd.Series(right).rank().to_numpy(dtype=np.float64)
    if a.std() == 0.0 or b.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    """Pearson correlation over the finite pairs, or NaN.

    Args:
        left: First sample.
        right: Second sample, the same length.

    Returns:
        The correlation, or NaN when fewer than three finite pairs survive or
        either side is constant.
    """
    ok = np.isfinite(left) & np.isfinite(right)
    if int(ok.sum()) < 3:
        return float("nan")
    a, b = left[ok], right[ok]
    if a.std() == 0.0 or b.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _position_only_prediction(losses: pd.DataFrame, tracking: pd.DataFrame) -> np.ndarray:
    """Expected recovery from where the ball was lost and nothing else.

    A cubic in the attacking fraction of the turnover, fitted at loss level.
    This is the model page 6 has to be measured against: if it reproduces the
    printed column, the column is territory.

    Args:
        losses: Scored losses.
        tracking: Canonical tracking frame for the same match.

    Returns:
        One prediction per loss, NaN where the fit could not reach.
    """
    spec = baseline.BaselineSpec(position_degree=3, use_width=False, shape=())
    try:
        features = baseline.loss_features(losses, tracking)
        model = baseline.fit_baseline(features, spec)
        return np.asarray(model.predict(features), dtype=np.float64)
    except (KeyError, ValueError, IndexError, np.linalg.LinAlgError):
        return np.full(len(losses), np.nan)


def _permutation_p(
    values: pd.Series,
    groups: pd.Series,
    strata: pd.Series,
) -> float:
    """Run one between-player permutation test and return its p-value.

    Args:
        values: Per-loss values.
        groups: Player key per loss.
        strata: Block label per loss, normally the team, so that a squad-level
            offset can never be shuffled into looking like a player effect.

    Returns:
        The p-value, or NaN when too few players clear the test's own minimum.
    """
    if values.empty:
        return float("nan")
    result = baseline.permutation_between_player(
        values,
        groups,
        strata=strata,
        n_shuffles=PERMUTATION_SHUFFLES,
        rng=np.random.default_rng(PERMUTATION_SEED),
    )
    if result.n_groups < MIN_PERMUTATION_GROUPS:
        return float("nan")
    return float(result.p_value)


def _standardised_recovery(losses: pd.DataFrame) -> dict[str, float]:
    """Each team's recovery reweighted to the match's own phase mix.

    The cover prints a pooled per-team recovery and a difference between two of
    them, directly above a caveat forbidding cross-phase arithmetic. The two
    squads do not lose the ball in the same places, so the pooled difference is
    only readable if it survives being put on a common phase mix. This computes
    the standardised version so the page can show the check rather than assert
    that it passed.

    Args:
        losses: Scored losses.

    Returns:
        ``{team: standardised mean recovery}``, empty when nothing is scored.
    """
    if losses.empty or "phase" not in losses.columns:
        return {}
    block = losses[losses["scored"].to_numpy(dtype=bool)]
    block = block[np.isfinite(block["recovery"].to_numpy(dtype=np.float64))]
    if block.empty:
        return {}
    weights = block["phase"].value_counts(normalize=True)
    out: dict[str, float] = {}
    for side, rows in block.groupby("losing_team", observed=True):
        means = rows.groupby("phase", observed=True)["recovery"].mean()
        share = weights.reindex(means.index).astype(float)
        total = float(share.sum())
        if total > 0.0:
            out[str(side)] = float((means.astype(float) * share).sum() / total)
    return out


def _half_counts(losses: pd.DataFrame) -> dict[str, tuple[int, int]]:
    """Losses each team conceded in its own half and in the opponent's.

    Args:
        losses: Scored losses.

    Returns:
        ``{team: (own half, opponent half)}``.
    """
    if losses.empty or "x_norm" not in losses.columns:
        return {}
    out: dict[str, tuple[int, int]] = {}
    for side, rows in losses.groupby("losing_team", observed=True):
        norm = pd.to_numeric(rows["x_norm"], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(norm)
        own = int(np.sum(finite & (norm < 0.5)))
        out[str(side)] = (own, int(finite.sum()) - own)
    return out


def measure_disclosure(
    losses: pd.DataFrame,
    players: pd.DataFrame,
    tracking: pd.DataFrame,
    team: str,
    *,
    radius_m: float = DEFAULT_RADIUS_M,
    printed: pd.DataFrame | None = None,
) -> Disclosure:
    """Measure everything the pages have to disclose, on this match.

    This is the function that keeps the warnings honest. The alternative --
    printing the figures an audit once measured on the bundled match -- would
    put a false number on every report built from anything else.

    Args:
        losses: Scored losses, ideally already carrying
            :data:`pressing.baseline.RESIDUAL_COLUMNS`.
        players: Output of :func:`pressing.counterpress.per_player` for the
            same losses.
        tracking: Canonical tracking frame for the match.
        team: The subject team.
        radius_m: The proximity radius the player attribution used.
        printed: The exact rows page 6 will print. Defaults to every player.

    Returns:
        The measured :class:`Disclosure`. Fields that cannot be computed on
        this match come back NaN or empty, and every caller is written so that
        an absent number removes a clause rather than printing a wrong one.
    """
    scored_mask = losses["scored"].to_numpy(dtype=bool) if len(losses) else np.zeros(0, bool)
    recovery = losses["recovery"].to_numpy(dtype=np.float64) if len(losses) else np.zeros(0)
    finite = scored_mask & np.isfinite(recovery)
    regain = (
        pd.to_numeric(losses.get("regained_within_5s"), errors="coerce").to_numpy(dtype=np.float64)
        if "regained_within_5s" in losses.columns
        else np.full(len(losses), np.nan)
    )

    unscored = losses[~scored_mask] if len(losses) else losses
    located = (
        np.isfinite(unscored["x"].to_numpy(dtype=np.float64))
        & np.isfinite(unscored["y"].to_numpy(dtype=np.float64))
        if len(unscored)
        else np.zeros(0, dtype=bool)
    )

    shown = players if printed is None else printed
    flags = (
        shown["reportable"].to_numpy(dtype=bool)
        if "reportable" in shown.columns and len(shown)
        else np.zeros(len(shown), dtype=bool)
    )
    good = shown.loc[flags] if len(shown) else shown
    phase_printed = {
        phase: (
            int(shown[f"reportable_{phase}"].to_numpy(dtype=bool).sum())
            if f"reportable_{phase}" in shown.columns and len(shown)
            else 0
        )
        for phase in PHASES
    }

    def spread(frame: pd.DataFrame, mean: str, low: str, high: str) -> tuple[float, float]:
        """Median interval width and top-to-bottom span of one printed column."""
        if frame.empty or mean not in frame.columns:
            return (float("nan"), float("nan"))
        values = frame[mean].to_numpy(dtype=np.float64)
        widths = frame[high].to_numpy(dtype=np.float64) - frame[low].to_numpy(dtype=np.float64)
        widths = widths[np.isfinite(widths)]
        values = values[np.isfinite(values)]
        return (
            float(np.median(widths)) if widths.size else float("nan"),
            float(values.max() - values.min()) if values.size else float("nan"),
        )

    level_interval, level_span = spread(good, "mean_recovery", "recovery_lo", "recovery_hi")
    residual_interval, residual_span = spread(
        good, "mean_residual", "residual_lo", "residual_hi"
    )
    clear = 0
    if not good.empty and "residual_lo" in good.columns:
        low = good["residual_lo"].to_numpy(dtype=np.float64)
        high = good["residual_hi"].to_numpy(dtype=np.float64)
        clear = int(np.sum(np.isfinite(low) & np.isfinite(high) & ((low > 0.0) | (high < 0.0))))

    # The credit table page 6 is built from, so the tests below shuffle exactly
    # the values the table averages.
    credited = _credited_pressers(losses, tracking, radius_m)
    wanted = ["loss_id", "phase", "recovery", "scored", *baseline.RESIDUAL_COLUMNS]
    source = losses.loc[:, [name for name in wanted if name in losses.columns]]
    joined = credited.merge(source, on="loss_id", how="left")
    if len(joined):
        joined = joined[joined["scored"].fillna(False).to_numpy(dtype=bool)]
        joined = joined.assign(
            key=joined["team"].astype(str) + "/" + joined["player_id"].astype(str)
        )

    position_r = position_rho = float("nan")
    position_rows = 0
    prediction = _position_only_prediction(losses, tracking)
    if len(joined) and np.isfinite(prediction).any() and not shown.empty:
        lookup = pd.Series(prediction, index=losses["loss_id"].to_numpy())
        by_player = joined.assign(
            position_only=joined["loss_id"].map(lookup).to_numpy(dtype=np.float64)
        ).groupby(["team", "player_id"], observed=True)
        pooled = by_player.agg(observed=("recovery", "mean"), predicted=("position_only", "mean"))
        keys = shown.set_index(["team", "player_id"]).index
        pooled = pooled.loc[pooled.index.intersection(keys)]
        observed = pooled["observed"].to_numpy(dtype=np.float64)
        predicted = pooled["predicted"].to_numpy(dtype=np.float64)
        ok = np.isfinite(observed) & np.isfinite(predicted)
        position_rows = int(ok.sum())
        position_r = _pearson(observed, predicted)
        position_rho = _rank_r(observed[ok], predicted[ok])

    phase_p: dict[str, float] = {}
    phase_residual_p: dict[str, float] = {}
    residual_p = float("nan")
    if len(joined):
        if "recovery_residual" in joined.columns:
            residual_p = _permutation_p(
                joined["recovery_residual"], joined["key"], joined["team"]
            )
        for phase in PHASES:
            block = joined[joined["phase"] == phase]
            phase_p[phase] = _permutation_p(block["recovery"], block["key"], block["team"])
            if "recovery_residual" in block.columns:
                phase_residual_p[phase] = _permutation_p(
                    block["recovery_residual"], block["key"], block["team"]
                )

    teams = per_team(losses).set_index("team") if len(losses) else pd.DataFrame()
    standardised = _standardised_recovery(losses)
    other = [side for side in ("home", "away") if side != team]
    opponent = other[0] if other else None

    def team_mean(side: str | None) -> float:
        """One team's pooled recovery, or NaN."""
        if side is None or side not in teams.index or "mean_recovery" not in teams.columns:
            return float("nan")
        return float(teams.loc[side, "mean_recovery"])

    raw_difference = team_mean(team) - team_mean(opponent)
    standardised_difference = float("nan")
    if team in standardised and opponent in standardised:
        standardised_difference = standardised[team] - standardised[opponent]

    return Disclosure(
        n_scored=int(finite.sum()),
        regain_r=_pearson(recovery[finite], regain[finite]) if finite.any() else float("nan"),
        position_r=position_r,
        position_rho=position_rho,
        position_rows=position_rows,
        printed_rows=int(len(shown)),
        suppressed_rows=int(np.sum(~flags)),
        min_losses=int(players.attrs.get("min_losses_for_player", baseline.MIN_LOSSES_FOR_PLAYER)),
        level_interval=level_interval,
        level_span=level_span,
        residual_interval=residual_interval,
        residual_span=residual_span,
        residual_clear_of_zero=clear,
        residual_p=residual_p,
        residual_out_of_sample=bool(losses.attrs.get("residual_out_of_sample", False)),
        phase_p=phase_p,
        phase_residual_p=phase_residual_p,
        phase_printed=phase_printed,
        raw_difference=raw_difference,
        standardised_difference=standardised_difference,
        standardised=standardised,
        half_counts=_half_counts(losses),
        unscored=int(len(unscored)),
        unscored_drawn=int(located.sum()),
        unscored_undrawable=int(len(unscored) - int(located.sum())),
        total_losses=int(len(losses)),
        phase_rows=int(losses["phase"].notna().sum()) if "phase" in losses.columns else 0,
    )


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


def _fixed_width(content: Flowable, width: float = FRAME_W) -> Table:
    """Pin a paragraph to a fixed measure so a page shrink stays uniform.

    ``build_report`` wraps every page in ``KeepInFrame(mode="shrink")``. When a
    page overflows, that re-wraps its content at ``availWidth * scale`` and then
    draws the result scaled down. A paragraph reflows into the wider measure and
    comes back out full width; a table with fixed column widths cannot, so it
    ends up narrower than the prose beneath it -- measured at 367 pt of table
    under 496 pt of footnote on a page-6 that had overflowed to a scale of 0.72.
    Boxing a paragraph makes it shrink with everything else, so an overflowing
    page reads as a slightly smaller page rather than a broken one.

    Args:
        content: The flowable to pin.
        width: Measure in points.

    Returns:
        A borderless single-cell table of exactly ``width``.
    """
    box = Table([[content]], colWidths=[width], hAlign="LEFT")
    box.setStyle(
        TableStyle(
            [
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return box


def _callout(content: list[Flowable], width: float = FRAME_W) -> Table:
    """Box a block of flowables so a reader cannot skim past it.

    A warning set in the same grey note face as the method footnotes is a
    warning nobody reads. This gives it a tinted panel and a thick rule down
    the left edge, which survives a greyscale photocopy as a solid black bar --
    the failure mode the proxy watermark was designed around, applied to the
    one page that had no warning at all.

    Args:
        content: The flowables to box.
        width: Panel width in points.

    Returns:
        A single-cell table carrying the panel styling.
    """
    panel = Table([[content]], colWidths=[width], hAlign="LEFT")
    panel.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), _ALERT_TINT),
                ("LINEBEFORE", (0, 0), (0, -1), 3.0, _ALERT),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7.0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7.0),
                ("TOPPADDING", (0, 0), (-1, -1), 6.0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6.0),
            ]
        )
    )
    return panel


#: Which direction of each cover KPI is the good one, and whether either is.
#:
#: The Difference column used to render every row through one neutral
#: formatter, so "+0.48 s slower to win the ball back" was styled exactly like
#: "+0.8 pp more often regained". Polarity is a property of the metric, so it
#: is declared next to the metric.
_POLARITY: Final[dict[str, str]] = {
    "recovery": "higher",
    "regain": "higher",
    "losses": "neither",
    "halves": "neither",
    "epv": "lower",
    "time_to_regain": "lower",
}

#: What the polarity column prints.
_POLARITY_LABEL: Final[dict[str, str]] = {
    "higher": "higher",
    "lower": "lower",
    "neither": "—",
}

#: How to read the Difference column, printed under it.
#:
#: The colour is the redundant cue and the "Better is" column is the load-bearing
#: one, because a red number is black in a photocopy while the word "lower" is
#: still the word "lower". The last clause is there because a coloured delta
#: reads as a verdict, and none of these gaps has been tested against noise.
POLARITY_LEGEND: Final[str] = (
    "\"Better is\" states which direction of each row is the good one — a property of "
    "the metric, not of this match — and a Difference printed in red is the subject team "
    "on the wrong side of it. Both are directions, not verdicts: no gap in this column "
    "has been tested against chance."
)


def _kpi_table(
    losses: pd.DataFrame,
    team: str,
    labels: dict[str, str] | None,
    styles: _Styles,
    fonts: tuple[str, str, str],
    disclosure: Disclosure | None = None,
) -> Table:
    """The cover's head-to-head KPI table.

    Every row is a metric, every team gets a column, then the subject team
    minus the opponent -- the delta a coach reads first -- and then which
    direction of that delta is the good one. The subject team's column is set
    in its own identity colour, the same blue or orange it is drawn in on
    every figure.

    Two of the rows read as a pair and are not one. "Counter-press recovery"
    and "Regain within 5 s" sit one above the other and describe, apparently,
    the same five seconds; on this data they are very nearly unrelated, and a
    table that stacks them without saying so is inviting the inference. So a
    measured note is spanned across the table directly beneath them.

    Args:
        losses: Scored losses.
        team: The subject team.
        labels: Optional real team names.
        styles: Paragraph styles.
        fonts: ``(regular, bold, italic)``.
        disclosure: Measured weaknesses, for the note under the regain row.
            Omitted, the note is not printed rather than guessed at.

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

    rows: list[tuple[str, str, str, str, str, float]] = []

    def add(
        label: str,
        us: float,
        them: float,
        fmt: Any,
        polarity: str,
        *,
        delta_fmt: Any | None = None,
    ) -> None:
        """Append one metric row, with the delta rendered in its own units."""
        gap = "—"
        difference = float("nan")
        if np.isfinite(us) and np.isfinite(them):
            difference = us - them
            gap = (delta_fmt or (lambda v: _signed(v, 2)))(difference)
        rows.append((label, fmt(us), fmt(them), gap, polarity, difference))

    add(
        f"Counter-press recovery, 5 s ({weighting})",
        stat(team, "mean_recovery"),
        stat(opponent, "mean_recovery"),
        lambda v: _num(v, 3),
        _POLARITY["recovery"],
        delta_fmt=lambda v: _signed(v, 3),
    )
    add(
        "Regain within 5 s (event stream)",
        stat(team, "regain_rate"),
        stat(opponent, "regain_rate"),
        _pct,
        _POLARITY["regain"],
        # Percentages differ in percentage POINTS, and saying so is the
        # difference between "eight tenths of a point" and "eight tenths of a
        # percent", which are not the same number.
        delta_fmt=lambda v: f"{_signed(v * 100.0, 1)} pp",
    )
    # Index of the row the spanned note has to sit under, counting the header.
    regain_row = len(rows)
    add(
        "Possession losses",
        stat(team, "losses"),
        stat(opponent, "losses"),
        lambda v: _int(v),
        _POLARITY["losses"],
        delta_fmt=lambda v: _signed(v, 0),
    )
    rows.append(
        (
            "— lost in own half / opponent half",
            f"{own_us} / {opp_us}",
            f"{own_them} / {opp_them}",
            "—",
            _POLARITY["halves"],
            float("nan"),
        )
    )
    add(
        f"EPV conceded per loss ({plots.EPV_DISPLAY_UNIT})",
        stat(team, "mean_epv_conceded") * plots.EPV_DISPLAY_SCALE,
        stat(opponent, "mean_epv_conceded") * plots.EPV_DISPLAY_SCALE,
        lambda v: _signed(v, 2),
        _POLARITY["epv"],
    )
    add(
        "Median time to regain (s)",
        stat(team, "median_time_to_regain"),
        stat(opponent, "median_time_to_regain"),
        lambda v: _num(v, 2),
        _POLARITY["time_to_regain"],
    )

    header = (
        "",
        plots.team_label(team, labels),
        plots.team_label(opponent, labels) if opponent else "—",
        "Difference",
        "Better is",
    )
    data: list[list[Any]] = [
        [_p(header[0], styles.caption), header[1], header[2], header[3], header[4]]
    ]
    commands: list[tuple[Any, ...]] = []
    note = _regain_note(disclosure, styles)
    for row in rows:
        index = len(data)
        data.append(
            [
                _p(row[0], styles.body),
                row[1],
                row[2],
                row[3],
                _POLARITY_LABEL.get(row[4], "—"),
            ]
        )
        # A delta on the wrong side of a metric whose polarity we have just
        # declared is the one cell on this table a coach must not misread.
        if row[4] != "neither" and np.isfinite(row[5]) and row[5] != 0.0:
            worse = (row[5] < 0.0) if row[4] == "higher" else (row[5] > 0.0)
            if worse:
                commands.append(("TEXTCOLOR", (3, index), (3, index), _ALERT))
        if note is not None and index == regain_row:
            span = len(data)
            data.append([note, "", "", "", ""])
            commands += [
                ("SPAN", (0, span), (-1, span)),
                ("BACKGROUND", (0, span), (-1, span), _ALERT_TINT),
                ("LINEBEFORE", (0, span), (0, span), 3.0, _ALERT),
                ("TOPPADDING", (0, span), (-1, span), 4.0),
                ("BOTTOMPADDING", (0, span), (-1, span), 4.0),
            ]

    widths = [FRAME_W * w for w in (0.415, 0.165, 0.165, 0.150, 0.105)]
    table = Table(data, colWidths=widths, hAlign="LEFT")
    commands += [
        ("ALIGN", (1, 0), (3, -1), "RIGHT"),
        ("ALIGN", (4, 0), (4, -1), "CENTER"),
        ("FONT", (1, 1), (1, -1), bold, 10.5),
        ("TEXTCOLOR", (1, 1), (1, -1), _team_ink(team)),
        ("FONT", (2, 1), (2, -1), regular, 9.5),
        ("TEXTCOLOR", (2, 1), (2, -1), _team_ink(opponent)),
        ("FONT", (4, 1), (4, -1), regular, 7.4),
        ("TEXTCOLOR", (4, 1), (4, -1), _INK_MUTED),
        ("TEXTCOLOR", (1, 0), (1, 0), _team_ink(team)),
        ("TEXTCOLOR", (2, 0), (2, 0), _team_ink(opponent)),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, _INK),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, _RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 4.0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.0),
    ]
    table.setStyle(_table_style(commands, regular, bold))
    return table


def _regain_note(disclosure: Disclosure | None, styles: _Styles) -> Paragraph | None:
    """The spanned row that stops the top two KPI rows being read as a pair.

    Args:
        disclosure: Measured weaknesses.
        styles: Paragraph styles.

    Returns:
        The paragraph, or None when the correlation could not be measured --
        in which case the row is left out rather than filled with a guess.
    """
    if disclosure is None or not np.isfinite(disclosure.regain_r):
        return None
    return _p(
        "These two rows are almost unrelated. Over the "
        f"{disclosure.n_scored} scored losses of this match they correlate at "
        f"r = {_signed(disclosure.regain_r, 2)}: a high recovery share is not a claim "
        "that the ball came back, and the two rows must not be read as one "
        "finding stated twice. Recovery is control of the ground; regain is "
        "possession of the ball.",
        styles.alert,
    )


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


def _p_header(p_value: float) -> str:
    """The permutation p-value as a column sub-heading.

    Args:
        p_value: The p-value, or NaN when the test could not run.

    Returns:
        A short string such as ``"p = 0.97"``, or ``"p n/a"``.
    """
    if not np.isfinite(p_value):
        return "p n/a"
    return f"p = {p_value:.2f}"


def _cell(
    value: Any,
    *,
    printable: bool,
    minimum: int,
    digits: int = 2,
    signed: bool = False,
) -> str:
    """One per-player number, or the reason it is not printed.

    Args:
        value: The number.
        printable: Whether enough losses stand behind it.
        minimum: The threshold, for the suppression text.
        digits: Decimal places.
        signed: Print an explicit ``+`` on positives, for a column whose sign
            is the whole message.

    Returns:
        The formatted number, or :data:`SUPPRESSED`.
    """
    if not printable:
        return SUPPRESSED.format(minimum=minimum)
    return _signed(value, digits) if signed else _num(value, digits)


def _interval(low: Any, high: Any, *, printable: bool) -> str:
    """A bootstrap interval as one narrow cell.

    Args:
        low: Lower bound.
        high: Upper bound.
        printable: Whether the row is above the suppression threshold.

    Returns:
        ``"-0.005 … +0.017"``, or an empty cell for a suppressed row.
    """
    if not printable:
        return ""
    lo, hi = _signed(low, 3), _signed(high, 3)
    if lo == "—" or hi == "—":
        return "—"
    return f"{lo} … {hi}"


def _player_table(
    players: pd.DataFrame,
    team: str | None,
    goalkeepers: dict[str, Any],
    styles: _Styles,
    fonts: tuple[str, str, str],
    disclosure: Disclosure | None = None,
) -> tuple[Table, int, int]:
    """The page-6 squad table.

    Three things changed here after the metric was measured rather than
    described, and each is visible on the paper.

    **Values below the minimum count are not printed.** A mean over six
    turnovers carries a 95% interval wider than the whole column, so printing
    it to two decimal places asserts a precision the number does not have. The
    cell says :data:`SUPPRESSED` instead, which is a fact about the sample and
    not a missing value.

    **The residual is printed beside the level, with its interval.** The level
    is very largely territory; the residual -- what the five seconds added to
    the shape the ball was lost in -- is the only column here that is not. It
    gets its interval next to it because on this data it is small enough that
    the interval is most of what there is to say.

    **The phase headers carry their own permutation p-value.** The report tells
    the reader to compare within a phase. Whether that comparison finds
    anything is a measurable question, and the answer belongs on the column,
    not in a footnote under it.

    Args:
        players: Output of :func:`pressing.counterpress.per_player`.
        team: Restrict to one team, or None for everyone.
        goalkeepers: ``attrs["goalkeeper"]`` from the tracking frame, used for
            the only position this data source actually knows.
        styles: Paragraph styles.
        fonts: ``(regular, bold, italic)``.
        disclosure: Measured weaknesses, for the p-values in the phase headers.

    Returns:
        ``(table, printed_rows, available_rows)``.
    """
    regular, bold, _ = fonts
    shown = _player_rows(players, team)
    available = len(players if team is None else players[players["team"] == team])
    minimum = int(players.attrs.get("min_losses_for_player", baseline.MIN_LOSSES_FOR_PLAYER))
    phase_p = disclosure.phase_p if disclosure is not None else {}
    header = [
        "#",
        "Player",
        "Min",
        "Losses\nnearby",
        "Recovery\n5 s",
        "vs\nexpected",
        "its 95%\ninterval",
        f"EPV swing\n({plots.EPV_DISPLAY_UNIT})",
        *[
            f"{phase.replace('_', ' ').capitalize()}\n"
            f"{_p_header(float(phase_p.get(phase, float('nan'))))}"
            for phase in PHASES
        ],
    ]
    data: list[list[Any]] = [header]
    ink: list[tuple[Any, ...]] = []
    keepers = {str(value) for value in goalkeepers.values()}
    first_phase = len(header) - len(PHASES)

    for row in shown.to_dict("records"):
        index = len(data)
        player_id = str(row.get("player_id", ""))
        printable = bool(row.get("reportable", False))
        cells: list[Any] = [
            _int(row.get("jersey"), dash=""),
            _p(f"{player_id} (GK)" if player_id in keepers else player_id, styles.dense),
            _int(row.get("minutes")),
            _int(row.get("losses_nearby")),
            _cell(row.get("mean_recovery"), printable=printable, minimum=minimum),
            # Signed, because a residual is a direction before it is a size and
            # the interval beside it is signed too.
            _cell(
                row.get("mean_residual"),
                printable=printable,
                minimum=minimum,
                digits=3,
                signed=True,
            ),
            _interval(row.get("residual_lo"), row.get("residual_hi"), printable=printable),
            # Suppressed under the same rule as the rest. It is a per-loss mean
            # over exactly the same losses, so a page that hides Recovery on
            # seven of them and prints an EPV mean on the same seven is telling
            # the reader two different things about one sample -- and makes the
            # warning's "a cell prints only when 11 losses stand behind it"
            # false on its own page.
            _cell(
                _scaled_epv(row.get("epv_conceded")),
                printable=printable,
                minimum=minimum,
                signed=True,
            ),
        ]
        if not printable:
            # Everything derived from too few losses is greyed together, so a
            # suppressed row reads as one decision rather than four dashes.
            ink.append(("TEXTCOLOR", (4, index), (7, index), _INK_MUTED))
        for offset, phase in enumerate(PHASES):
            allowed = bool(row.get(f"reportable_{phase}", False))
            cells.append(_cell(row.get(f"recovery_{phase}"), printable=allowed, minimum=minimum))
            if not allowed:
                column = first_phase + offset
                ink.append(("TEXTCOLOR", (column, index), (column, index), _INK_MUTED))
        data.append(cells)
        ink.append(("TEXTCOLOR", (0, index), (0, index), _team_ink(row.get("team"))))

    if len(data) == 1:
        empty = _p("No players on the pitch in this match", styles.body)
        data.append(["", empty, *[""] * (len(header) - 2)])

    # Measured on the rendered page rather than guessed: at the previous
    # widths the right-aligned "its 95% / interval" header sat two points from
    # a slightly overflowing "EPV swing", and the two read as one word,
    # "its 95%EPV swing". Both columns are wider here and the interval column
    # keeps a real right pad.
    widths = [
        FRAME_W * w
        for w in (0.038, 0.164, 0.048, 0.070, 0.078, 0.072, 0.132, 0.092, 0.102, 0.102, 0.102)
    ]
    table = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1)
    commands: list[tuple[Any, ...]] = [
        ("BACKGROUND", (0, 0), (-1, 0), _HEADER_FILL),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONT", (0, 0), (-1, -1), regular, 7.6),
        ("FONT", (0, 0), (-1, 0), bold, 6.6),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
        ("FONT", (0, 1), (0, -1), bold, 7.8),
        ("FONT", (5, 1), (6, -1), regular, 6.8),
        ("LEFTPADDING", (5, 0), (6, -1), 2.0),
        ("RIGHTPADDING", (5, 0), (5, -1), 2.0),
        ("RIGHTPADDING", (6, 0), (6, -1), 5.0),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, _RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 1.4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.4),
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


def _cover_evidence(
    disclosure: Disclosure | None,
    labels: dict[str, str] | None,
    styles: _Styles,
) -> list[Flowable]:
    """The two checks the cover asserts and never used to show.

    The first attaches a number to the field-position caveat directly above it,
    which the repository stated qualitatively for a long time and quantified
    nowhere. The second is the arithmetic the caveat itself demands: the cover
    prints one pooled cross-phase recovery per team and subtracts them, two
    lines above a sentence saying cross-phase recoveries must not be
    subtracted. The pooled difference is only readable if it survives putting
    both squads on a common phase mix, so the check is run and printed.

    Args:
        disclosure: Measured weaknesses, or None for no evidence lines.
        labels: Optional real team names.
        styles: Paragraph styles.

    Returns:
        Zero, one or two paragraphs.
    """
    if disclosure is None:
        return []
    out: list[Flowable] = []
    if disclosure.has_position_check:
        out.append(
            _p(
                "Measured on this match, not asserted: a model that knows only where "
                "each player's turnovers happened reproduces the whole per-player "
                f"recovery column on page 6 at r = {_signed(disclosure.position_r, 2)} "
                f"over {disclosure.position_rows} players. That column ranks territory. "
                "Page 6 says so at the top, in full.",
                styles.alert,
            )
        )

    if np.isfinite(disclosure.standardised_difference) and np.isfinite(disclosure.raw_difference):
        halves = "; ".join(
            f"{plots.team_label(side, labels)} {own}/{opp}"
            for side, (own, opp) in sorted(disclosure.half_counts.items())
        )
        standardised = ", ".join(
            f"{plots.team_label(side, labels)} {value:.3f}"
            for side, value in sorted(disclosure.standardised.items())
        )
        verdict = (
            "the sign survives, so the recovery row may be read as a difference"
            if disclosure.standardisation_holds
            else "the sign does NOT survive, so the recovery row must not be read as a "
            "difference"
        )
        out.append(
            _p(
                "The recovery row pools three phases and the squads did not lose the ball "
                f"in the same places (own half/opponent half: {halves}). Putting both on "
                f"the match's own phase mix gives {standardised}, a gap of "
                f"{_signed(disclosure.standardised_difference, 3)} where the row above "
                f"prints {_signed(disclosure.raw_difference, 3)} — {verdict}. A "
                "difference of two pooled numbers is not otherwise safe to read.",
                styles.caveat,
            )
        )
    return out


def cover_page(
    losses: pd.DataFrame,
    team: str,
    *,
    subtitle: str,
    labels: dict[str, str] | None,
    styles: _Styles,
    fonts: tuple[str, str, str],
    title: str = "Counter-press report",
    disclosure: Disclosure | None = None,
) -> list[Flowable]:
    """Page 1: the fixture, the headline numbers and the caveats that frame them.

    Args:
        losses: Scored losses.
        team: The subject team.
        subtitle: Fixture, provider and frame-rate line.
        labels: Optional real team names.
        styles: Paragraph styles.
        fonts: ``(regular, bold, italic)``.
        title: Report title.
        disclosure: Measured weaknesses. Every sentence that needs one of its
            numbers is omitted when the number could not be measured.

    Returns:
        The page's flowables.
    """
    summary = summarise_match(losses)
    story: list[Flowable] = [
        _p(title, styles.title),
        _p(subtitle, styles.subtitle),
        _kpi_table(losses, team, labels, styles, fonts, disclosure),
        Spacer(1, 3),
        # Every block of prose on this page is pinned to the frame width for
        # the reason ``_fixed_width`` documents: the cover carries two fixed
        # -width figures and a fixed-width table, and an unpinned paragraph
        # beside them reflows under ``KeepInFrame`` and comes out wider than
        # everything it is captioning.
        _fixed_width(_p(POLARITY_LEGEND, styles.note)),
        Spacer(1, 7),
        _fixed_width(_p(PHASE_CAVEAT, styles.caveat)),
    ]
    evidence = _cover_evidence(disclosure, labels, styles)
    if evidence:
        story += [Spacer(1, 4), *(_fixed_width(item) for item in evidence)]
    story.append(Spacer(1, 6))
    # The bars sit directly under the caveat because they are the evidence for
    # it: three pairs of bars an order apart, with the sample size printed on
    # each, say more than the sentence does.
    story.append(
        _fig_image(
            plots.phase_bars(per_phase(losses), team_labels=labels, figsize=(7.6, 2.7)),
            FRAME_W,
        )
    )
    story.append(Spacer(1, 8))
    story.append(
        _fig_image(
            plots.recovery_sparkline(losses, team_labels=labels, figsize=(7.6, 2.1)), FRAME_W
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
        scored_note += f" {skipped} could not be scored ({detail})."
        # "None were dropped" used to end that sentence and was not true of the
        # pages that follow: a loss with no ball position has no pitch third
        # either, so it is absent from page 2's map and from its phase table
        # while still being counted here. The line now says where it went.
        if disclosure is not None:
            missing = disclosure.unscored_undrawable
            if missing == 0:
                scored_note += " None were dropped: page 2 draws them as grey crosses."
            else:
                if missing == 1:
                    they, carry, them = "That one is", "carries", "it"
                else:
                    they, carry, them = f"{missing} of them are", "carry", "them"
                scored_note += (
                    f" {they} counted in the {disclosure.total_losses} above but "
                    f"{carry} no ball position at all, so there is nowhere on a pitch "
                    f"to draw {them}."
                )
                if disclosure.phase_table_short_by:
                    scored_note += (
                        " With no position there is no pitch third either, so page 2's "
                        f"phase table works from {disclosure.phase_rows}."
                    )
    ball_out = summary.get("ball_out_count")
    if ball_out is not None:
        scored_note += (
            f" A further {int(ball_out)} balls went out of play and are counted separately, "
            "since nobody can counter-press a throw-in."
        )
    story += [
        Spacer(1, 4),
        _fixed_width(_p(scored_note, styles.note)),
        Spacer(1, 6),
        _fixed_width(
            _p(
                "The rest of this report: page 2, every loss on one pitch and the same "
                "losses split by phase. Pages 3–5, the five turnovers this ranking "
                "picks out, with the pitch-control surface behind each one. Page 6, the "
                "same numbers per player, for both squads. " + EPV_PROVENANCE,
                styles.note,
            )
        ),
    ]
    return story


def _unscored_note(disclosure: Disclosure | None) -> str:
    """What the map does and does not show of the losses it could not score.

    The page used to print, unconditionally, that unscored losses "are drawn
    rather than dropped, because a map that quietly hides its failures flatters
    the press". A loss with no ball position has nowhere to be drawn, so it is
    excluded from the cross layer and from the legend -- and on the bundled
    match the single unscored loss is exactly that kind. The boast described
    something the reader could not see. This says what is actually on the
    pitch above it.

    Args:
        disclosure: Measured weaknesses, or None when the counts are unknown.

    Returns:
        A sentence or two, always true of the map beside them.
    """
    if disclosure is None:
        return (
            "A grey cross on the map is a loss the metric could not score. A loss with "
            "no ball position at all cannot be drawn anywhere and is not on this map; "
            "the cover counts it."
        )
    if disclosure.unscored == 0:
        return (
            "Every loss on this map was scored, so there are no grey crosses on it: "
            "there is nothing this metric failed on to draw."
        )
    drawn, missing = disclosure.unscored_drawn, disclosure.unscored_undrawable
    losses_word = "loss" if disclosure.unscored == 1 else "losses"
    if missing == 0:
        return (
            f"{disclosure.unscored} {losses_word} could not be scored — a window cut "
            "short by the whistle, or a ball the metric could not follow — and all "
            f"{drawn} are drawn as grey crosses rather than dropped, because a map that "
            "quietly hides its failures flatters the press."
        )
    # Where the undrawable losses went, stated as arithmetic the reader can do
    # on the page in front of them. The phase table below adds up to fewer than
    # the cover's headline count whenever a loss has no position, because a loss
    # with no position has no pitch third either, and a footnote that does not
    # say so leaves the reader to find the discrepancy themselves.
    elsewhere = ""
    if disclosure.phase_table_short_by:
        elsewhere = (
            " — and no pitch third to file it under, which is why the Losses column in "
            f"the table below adds up to {disclosure.phase_rows} and not the "
            f"{disclosure.total_losses} the cover counts"
        )
    if drawn == 0:
        subject = (
            "The single loss the metric could not score"
            if disclosure.unscored == 1
            else f"The {disclosure.unscored} losses the metric could not score"
        )
        verb = "has" if disclosure.unscored == 1 else "have"
        them = "it" if disclosure.unscored == 1 else "them"
        return (
            f"There are no grey crosses on this map. {subject} {verb} no ball position "
            f"at all, so there is nowhere on a pitch to draw {them}{elsewhere}. Saying "
            "so is better than a legend entry for crosses that are not there."
        )
    them = "it" if missing == 1 else "them"
    return (
        f"{disclosure.unscored} {losses_word} could not be scored. {drawn} of them have "
        "a ball position and are drawn as grey crosses rather than dropped, because a "
        f"map that quietly hides its failures flatters the press. The other {missing} "
        f"had no ball position at all, so there is nowhere on a pitch to draw {them}"
        f"{elsewhere}."
    )


def loss_map_page(
    losses: pd.DataFrame,
    *,
    labels: dict[str, str] | None,
    styles: _Styles,
    fonts: tuple[str, str, str],
    disclosure: Disclosure | None = None,
) -> list[Flowable]:
    """Page 2: every loss on one pitch, then the same losses split by phase.

    Args:
        losses: Scored losses.
        labels: Optional real team names.
        styles: Paragraph styles.
        fonts: ``(regular, bold, italic)``.
        disclosure: Measured weaknesses, so the footnote about unscored losses
            describes this map rather than a map in general.

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
        _p(_unscored_note(disclosure) + " " + EPV_PROVENANCE, styles.note),
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


def _rank_warning(disclosure: Disclosure | None, styles: _Styles) -> Table | None:
    """The boxed warning that opens page 6.

    Page 6 is the page a coach photocopies and circulates, it is a per-player
    ranking, and its headline column is very largely a map of where those
    players' turnovers happened. Until this existed the page carried no
    field-position warning at all -- the knowledge was in the source and never
    reached the paper.

    Every number in it is measured on the match in hand by
    :func:`measure_disclosure`, and every sentence that would need a number the
    match cannot supply is left out rather than softened into a hedge.

    Args:
        disclosure: Measured weaknesses, or None to print no warning.
        styles: Paragraph styles.

    Returns:
        The boxed panel, or None when nothing measurable could be said.
    """
    if disclosure is None:
        return None
    lines: list[str] = []

    if disclosure.has_position_check:
        rho = (
            f", rank correlation {_signed(disclosure.position_rho, 2)}"
            if np.isfinite(disclosure.position_rho)
            else ""
        )
        lines.append(
            "Measured on this match, not asserted: a model given nothing but WHERE "
            "each player's turnovers happened — no pressing, no shape, no identity — "
            f"reproduces the Recovery 5 s column at r = {_signed(disclosure.position_r, 2)} "
            f"over {disclosure.position_rows} players{rho}. Squared, that is "
            f"{disclosure.position_r ** 2 * 100.0:.0f}% of everything separating one "
            "player from another in it. Recovery is the share of the pitch a team held "
            "after a turnover, and control of ground is mostly geography. Sort this "
            "column and you have very largely sorted the squad by where it plays."
        )

    residual = (
        "vs expected takes that out: it is Recovery minus what a team losing the ball "
        "in those places, in those shapes, was already going to get. It is the only "
        "column here that is not territory."
    )
    if disclosure.residual_clear_of_zero == 0 and np.isfinite(disclosure.residual_interval):
        residual += " On this match every printed player's 95% interval on it includes zero."
    elif disclosure.residual_clear_of_zero:
        residual += (
            f" On this match {disclosure.residual_clear_of_zero} of the printed players "
            "have a 95% interval on it that excludes zero."
        )
    if np.isfinite(disclosure.residual_p):
        residual += (
            " Reshuffling the losses at random among a squad's own players spreads them "
            f"this far apart {disclosure.residual_p * 100.0:.0f}% of the time "
            f"({PERMUTATION_SHUFFLES:,} shuffles, p = {disclosure.residual_p:.2f})."
        )
    if not disclosure.residual_out_of_sample:
        residual += (
            " The expectation behind it was fitted on this match, so it is in-sample "
            "and if anything flatters itself."
        )
    lines.append(residual)

    measured = {
        phase: value for phase, value in disclosure.phase_p.items() if np.isfinite(value)
    }
    # A column that prints nothing at all is the sharpest version of the same
    # point, and a reader scanning a wall of n<11 deserves to be told it is the
    # sample and not a rendering fault. Computed here rather than inside the
    # branch below, because a match too thin to run the test at all is exactly
    # the match most likely to have an empty column.
    empty = [phase for phase in PHASES if disclosure.phase_printed.get(phase, 0) == 0]
    empty_note = ""
    if empty and len(empty) < len(PHASES):
        names = " and ".join(phase.replace("_", " ") for phase in empty)
        verb = "column is" if len(empty) == 1 else "columns are"
        empty_note = (
            f" The {names} {verb} empty for a reason, not by mistake: no player on "
            f"either squad was near {disclosure.min_losses} turnovers there."
        )
    elif len(empty) == len(PHASES):
        empty_note = (
            " No phase column prints a value for anybody on this match: no player on "
            f"either squad was near {disclosure.min_losses} turnovers in any one third."
        )
    if measured:
        detail = ", ".join(
            f"p = {value:.2f} in the {phase.replace('_', ' ')}" for phase, value in measured.items()
        )
        strong = [phase for phase, value in measured.items() if value < 0.05]
        verdict = (
            "None of them separates players by more than chance."
            if not strong
            else "Read them one column at a time."
        )
        phase_line = (
            "The three phase columns are the comparison the rest of this report tells "
            f"you to make. By the same test they give {detail}. {verdict} Comparing two "
            "players inside one phase does not rescue the number on one match."
        )
        # When a phase column DOES clear the test, the next question is whether
        # what separated the players there was pressing or position, and the
        # residual answers it. Across the two Metrica matches the one column
        # that clears -- recovery in the final third -- goes from p = 0.0008 to
        # p = 0.88 the moment the expectation is subtracted. So the check is
        # printed beside the finding rather than left for a reader to want.
        residual_detail = ", ".join(
            f"{phase.replace('_', ' ')} p = {disclosure.phase_residual_p[phase]:.2f}"
            for phase in strong
            if np.isfinite(disclosure.phase_residual_p.get(phase, float("nan")))
        )
        if residual_detail:
            phase_line += (
                " Subtracting the expected value from the column that did clear leaves "
                f"{residual_detail}, so what separated those players was where they "
                "stood."
            )
        counts = ", ".join(
            f"{disclosure.phase_printed.get(phase, 0)} in the {phase.replace('_', ' ')}"
            for phase in PHASES
            if phase in measured
        )
        phase_line += (
            f" Of the {disclosure.printed_rows} rows here, the number carrying a printed "
            f"value is {counts}."
        )
        lines.append(phase_line + empty_note)
        empty_note = ""

    marker = SUPPRESSED.format(minimum=disclosure.min_losses)
    silenced = (
        " Here that silences every number on "
        f"{disclosure.suppressed_rows} of these {disclosure.printed_rows} rows."
        if disclosure.suppressed_rows
        else (
            " Every row here clears it for Recovery, vs expected and EPV swing, so "
            f"{marker} can only appear in a phase column."
        )
    )
    thin = (
        f"No cell prints unless {disclosure.min_losses} losses stand behind it — nearby "
        "losses for Recovery, vs expected and EPV swing, losses in that third for a "
        f"phase column. Below that the cell reads {marker}.{silenced}{empty_note}"
    )
    if np.isfinite(disclosure.level_interval) and np.isfinite(disclosure.level_span):
        thin += (
            " Where a value does print, the typical 95% interval on it is "
            f"{disclosure.level_interval:.2f} wide against a column spanning "
            f"{disclosure.level_span:.2f} top to bottom."
        )
    thin += (
        " So: read this table for who was near the ball when it was lost, how often, "
        "and in which third. Those are counts and the counts are solid. Every number "
        "to the right of Losses nearby — the EPV swing included, which is the ground "
        "the opponent gained and therefore territory too — is context for those "
        "counts, not a ranking of players."
    )
    lines.append(thin)

    return _callout(
        [
            _p(RANK_WARNING_HEADING, styles.alert_heading),
            *[_p(line, styles.alert) for line in lines],
        ]
    )


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
    disclosure: Disclosure | None = None,
) -> list[Flowable]:
    """Page 6: the warning, the squad table and the method note that defines it.

    The opponent's table is printed underneath the subject team's, at the same
    scale. Recovery is heavily driven by where the ball was lost, so a number
    is only worth anything against another number computed the same way in the
    same match -- and the opponent's squad is the only such comparison a single
    match report has.

    The warning comes first, above the table, because this is the page that
    leaves the building on its own. A reader who photocopies it and pins it up
    photocopies the warning with it.

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
        disclosure: Measured weaknesses, for the warning and the phase headers.

    Returns:
        The page's flowables.
    """
    table, printed, available = _player_table(
        players, team, goalkeepers, styles, fonts, disclosure
    )
    who = plots.team_label(team, labels) if team else "both teams"
    story: list[Flowable] = []
    warning = _rank_warning(disclosure, styles)
    if warning is not None:
        story += [warning, Spacer(1, 7)]
    story += [
        _p(f"Player counter-press table — {who}", styles.heading),
        table,
    ]
    if printed < available:
        story.append(
            _fixed_width(
                _p(
                    f"Showing the {printed} players most often near a turnover, of "
                    f"{available} who played.",
                    styles.note,
                )
            )
        )
    if opponent is not None and not players[players["team"] == opponent].empty:
        other, other_printed, other_available = _player_table(
            players, opponent, goalkeepers, styles, fonts, disclosure
        )
        story += [
            Spacer(1, 7),
            _p(
                f"For comparison — {plots.team_label(opponent, labels)}",
                styles.heading,
            ),
            other,
        ]
        if other_printed < other_available:
            story.append(
                _fixed_width(
                    _p(
                        f"Showing {other_printed} of {other_available} who played.",
                        styles.note,
                    )
                )
            )
    minimum = int(players.attrs.get("min_losses_for_player", baseline.MIN_LOSSES_FOR_PLAYER))
    # One method note, not three. The suppression rule, the interval and the
    # p-values are all stated in the warning at the top of the page in the
    # words a reader needs them in; repeating them at the bottom cost the page
    # more height than it could spare, and height on this page is paid for by
    # shrinking the warning.
    story += [
        Spacer(1, 6),
        _fixed_width(
            _p(
                NEARBY_GLOSS.format(radius=radius_m)
                + " Minutes are tracked presence, not an assumed 90, and the tracking "
                "clock runs across half time, so a full match is about 97 minutes here: "
                "per-90 rates divide by these numbers, not by 90. Recovery is the mean "
                "over that player's nearby losses, and the phase columns are the same "
                "mean split by where the ball was lost. vs expected subtracts a "
                "conditional expectation fitted on the turnover's position, the losing "
                "team's compactness, its distance from the ball and its defensive line "
                "height; the interval beside it is a 2,000-resample percentile bootstrap "
                "over that player's own losses. p in a phase heading is a "
                f"{PERMUTATION_SHUFFLES:,}-shuffle permutation test — that column's "
                "values reshuffled among the squad's own losses with every player's "
                "count held fixed, and p is how often chance spread the players at least "
                f"as far apart as they actually are. The threshold {minimum} is the "
                "count at which a 95% interval on a mean first becomes narrower than the "
                "whole range of player means. A goalkeeper is marked (GK); this provider "
                "states no other position for anybody. "
                + plots.EPV_UNIT_GLOSS
                + " "
                + EPV_PROVENANCE,
                styles.note,
            )
        ),
    ]
    return story


# --------------------------------------------------------------------------- #
# Page furniture
# --------------------------------------------------------------------------- #


def _draw_footer(
    canvas: Canvas,
    *,
    fixture: str,
    proxy_line: str | None,
    fonts: tuple[str, str, str],
) -> None:
    """Draw the rule, the running head and the source credit.

    The proxy disclosure, when there is one, is drawn here as flat opaque
    text as well as diagonally across the page. That redundancy is the point: the
    diagonal watermark is alpha-blended, so it can vanish in a greyscale
    photocopy, and it is rotated, so text extraction reorders its characters
    and a machine looking for the string will not find it. This line survives
    both.

    Args:
        canvas: Active canvas.
        fixture: Running-head text.
        proxy_line: Event-proxy disclosure, or None. Named apart from the
            report's measured :class:`Disclosure`, which is a different thing
            entirely and used to share this name.
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
    if proxy_line:
        canvas.setFont(bold, 7.4)
        canvas.setFillColor(_ALERT)
        canvas.drawRightString(PAGE_W - MARGIN, 6.4 * mm, f"{proxy_line} — {PROXY_SUBTITLE}")
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

    # The conditional baseline before anything else: per_player reads the
    # residual columns off the loss frame, and the page-6 warning is measured
    # from the same frame the tables are aggregated from.
    losses = with_expected_recovery(losses, tracking)
    if players is None:
        players = per_player(losses, tracking, radius_m)
    other_side = "away" if team == "home" else "home"
    printed_rows = pd.concat(
        [_player_rows(players, team), _player_rows(players, other_side)]
    )
    disclosure = measure_disclosure(
        losses, players, tracking, team, radius_m=radius_m, printed=printed_rows
    )
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
    proxy_line = watermark.strip() if watermark and watermark.strip() else None

    pages: list[list[Flowable]] = [
        cover_page(
            losses,
            team,
            subtitle=subtitle,
            labels=labels,
            styles=styles,
            fonts=fonts,
            title=title,
            disclosure=disclosure,
        ),
        loss_map_page(
            losses, labels=labels, styles=styles, fonts=fonts, disclosure=disclosure
        ),
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
            opponent=other_side,
            disclosure=disclosure,
        ),
    ]
    if len(pages) != N_PAGES:  # pragma: no cover - guards a future edit
        raise ValueError(f"the report template must build {N_PAGES} pages, got {len(pages)}")

    subject = (
        f"{proxy_line}: {PROXY_SUBTITLE}. Counter-press report."
        if proxy_line
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
        _draw_footer(canvas, fixture=head, proxy_line=proxy_line, fonts=fonts)

    def on_page_end(canvas: Canvas, _doc: BaseDocTemplate) -> None:
        """Draw the watermark, after the frame's flowables. See the module docstring."""
        if proxy_line:
            _draw_watermark(canvas, proxy_line, fonts)

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


