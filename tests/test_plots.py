"""Tests for the report's figure renderers.

What is worth testing about a plotting module, and what the rest is taste:

1. **Does it build at all, headlessly.** Every figure here has to render under
   Agg inside a PDF build, at more than one size, without a display.
2. **Is the heatmap the right way up.** A control surface is ``(n_y, n_x)`` with
   row 0 at the BOTTOM touchline. Draw it without ``origin="lower"`` and you get
   a completely plausible-looking picture of the wrong half of the pitch, and no
   assertion about types or shapes will ever catch it. The test here renders to
   pixels and reads the colours back out, which is the only check that actually
   proves the orientation.
3. **Do the degenerate inputs survive, and survive in the right way.** Empty
   tables, single points, NaN coordinates and an untracked ball all occur in
   real matches. None of them may raise -- and none of them may be handled by
   quietly drawing a mark at the origin, dropping a row without saying so, or
   printing a panel with nothing on it, which is what "survives" used to mean
   here. Every degenerate-input test below asserts on artist positions, artist
   counts or rendered pixels, so it fails on a plausible wrong answer and not
   only on an exception.
4. **Does the picture still mean what the caption says.** The encodings are
   under test, not just the plumbing: hue is the team on every figure, marker
   fill carries the sign of a signed quantity and marker area its magnitude,
   an inferred ball is drawn differently from a tracked one, and a scaled
   number is never printed without its unit.
5. **Can the numbers be read off it.** v0.1.0 shipped a season scatter whose
   correlation -- the one statistic the chart exists to print -- was covered by
   a player's name, and six colourbars carrying no digits at all. Neither
   raised anything, both returned a Figure, and both drew every artist they
   promised. So the legibility block near the end asserts on geometry measured
   off the rendered canvas: which boxes overlap which, what falls outside the
   axes, where a leader line runs, and whether a colour ramp has numbers on it.

Nothing here imports ``pyplot``, and one test asserts that the module hasn't
either -- figures built through ``pyplot`` accumulate in a process-global
registry and leak straight through a multi-page report build.
"""

from __future__ import annotations

import math
import sys
import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from matplotlib.ft2font import FT2Font

from pressing import plots
from pressing.schema import PITCH_LENGTH, PITCH_WIDTH

matplotlib.use("Agg")

HALF_WIDTH = (5.4, 4.2)
FULL_WIDTH = (7.6, 5.2)

#: Names whose characters live outside Latin-1. reportlab's built-in Helvetica
#: turns every one of these into a box glyph with no exception raised, and a
#: matplotlib face without the glyphs does the same with only a warning, so the
#: registered face is tested against them directly.
AWKWARD_NAMES = ("Vuković", "Gündoğan", "Łukasz", "Şahin", "Bergström", "Håland")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def surface() -> np.ndarray:
    """A plausible control surface: attackers own the right end."""
    n_y, n_x = 32, 50
    ramp = np.linspace(0.05, 0.95, n_x)
    return np.tile(ramp, (n_y, 1))


@pytest.fixture
def players() -> dict[str, np.ndarray]:
    """Two small teams and a ball, in canonical metres."""
    rng = np.random.default_rng(11)
    return {
        "attackers": np.column_stack(
            [rng.uniform(50.0, 100.0, 11), rng.uniform(5.0, 63.0, 11)]
        ),
        "defenders": np.column_stack([rng.uniform(5.0, 60.0, 11), rng.uniform(5.0, 63.0, 11)]),
        "ball": np.array([61.0, 40.0]),
    }


@pytest.fixture
def losses() -> pd.DataFrame:
    """A loss table with the columns the renderers actually read."""
    rng = np.random.default_rng(3)
    n = 60
    return pd.DataFrame(
        {
            "loss_id": np.arange(n),
            "t": np.sort(rng.uniform(0.0, 5600.0, n)),
            "period": np.where(np.sort(rng.uniform(0.0, 5600.0, n)) > 2800.0, 2, 1),
            "losing_team": rng.choice(["home", "away"], n),
            "x": rng.uniform(0.0, PITCH_LENGTH, n),
            "y": rng.uniform(0.0, PITCH_WIDTH, n),
            "recovery": rng.uniform(0.1, 0.95, n),
            "epv_conceded": rng.normal(0.0, 0.002, n),
            "scored": True,
            "phase": rng.choice(["build_up", "middle", "final_third"], n),
            "losing_player": [f"Player{i}" for i in range(n)],
        }
    )


@pytest.fixture
def season() -> pd.DataFrame:
    """A season table matching the documented contract."""
    rng = np.random.default_rng(5)
    n = 24
    return pd.DataFrame(
        {
            "player": [f"Player {i}" for i in range(n)],
            "team": rng.choice(["Rovers", "United"], n),
            "minutes": rng.uniform(400.0, 3000.0, n),
            "counterpress_per_90": rng.uniform(2.0, 20.0, n),
            "market_value_eur": 10 ** rng.uniform(5.5, 7.8, n),
        }
    )


def render(fig: Figure) -> np.ndarray:
    """Rasterise a figure to an RGB array without touching the filesystem."""
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(float)


def closes(fig: Figure) -> None:
    """Assert a figure tears down cleanly and leaves no pyplot state behind."""
    assert isinstance(fig, Figure)
    fig.clear()
    assert not fig.axes
    assert "matplotlib.pyplot" not in sys.modules, (
        "plots.py must never import pyplot: pyplot figures live in a global "
        "registry and leak through a multi-page report build"
    )


def marker_points(ax: matplotlib.axes.Axes) -> np.ndarray:
    """Every point drawn as a MARKER on an axes, in data coordinates.

    Lines with no marker -- the pitch markings, the velocity leaders, a trend
    -- are excluded, so this is exactly the set of positions the figure is
    claiming something sits at.
    """
    points = [
        line.get_xydata()
        for line in ax.lines
        if line.get_marker() not in ("None", "none", None, "")
    ]
    points += [collection.get_offsets() for collection in ax.collections]
    if not points:
        return np.empty((0, 2), dtype=float)
    return np.vstack([np.asarray(p, dtype=float).reshape(-1, 2) for p in points])


def key_labels(ax: matplotlib.axes.Axes) -> list[str]:
    """Legend labels on an axes, or an empty list when it has no legend."""
    legend = ax.get_legend()
    return [] if legend is None else [text.get_text() for text in legend.get_texts()]


def notes(ax: matplotlib.axes.Axes) -> list[str]:
    """Free text drawn on an axes: annotations, direction notes, empty states."""
    return [text.get_text() for text in ax.texts]


def ink_fraction(fig: Figure) -> float:
    """Fraction of rendered pixels that differ from the paper colour."""
    pixels = render(fig)
    paper = np.asarray(matplotlib.colors.to_rgb(plots.SURFACE)) * 255.0
    return float((np.abs(pixels - paper).max(axis=2) > 6.0).mean())


# --------------------------------------------------------------------------- #
# The visual language
# --------------------------------------------------------------------------- #


def test_team_colours_are_a_distinct_pair() -> None:
    assert plots.team_colour("home") == plots.HOME_COLOUR
    assert plots.team_colour("away") == plots.AWAY_COLOUR
    assert plots.HOME_COLOUR != plots.AWAY_COLOUR
    # Blue vs orange separate on the blue-yellow axis, which red-green
    # deficiencies leave intact; the pair must never collapse toward one hue.
    home = np.asarray(matplotlib.colors.to_rgb(plots.HOME_COLOUR))
    away = np.asarray(matplotlib.colors.to_rgb(plots.AWAY_COLOUR))
    assert abs(float(home[2] - away[2])) > 0.3  # blue channel does the work


def test_team_colour_falls_back_for_unknown_teams() -> None:
    assert plots.team_colour("ball") == plots.INK_MUTED
    assert plots.team_colour(None) == plots.INK_MUTED


def test_team_label_prefers_supplied_names() -> None:
    assert plots.team_label("home") == "Home"
    assert plots.team_label("home", {"home": "Rovers"}) == "Rovers"


def test_ordered_teams_is_home_then_away() -> None:
    assert plots.ordered_teams(["away", "home", "away"]) == ["home", "away"]
    assert plots.ordered_teams(["zebra", "away"]) == ["away", "zebra"]


def test_styled_restores_rcparams() -> None:
    before = matplotlib.rcParams["font.size"]
    with plots.styled():
        assert matplotlib.rcParams["axes.titlelocation"] == "left"
    assert matplotlib.rcParams["font.size"] == before


def test_control_cmap_poles_are_the_team_colours() -> None:
    cmap = plots.control_cmap("home")
    low = np.asarray(cmap(0.0)[:3])
    high = np.asarray(cmap(1.0)[:3])
    mid = np.asarray(cmap(0.5)[:3])
    away = np.asarray(matplotlib.colors.to_rgb(plots.AWAY_COLOUR))
    home = np.asarray(matplotlib.colors.to_rgb(plots.HOME_COLOUR))
    # Poles sit near their team's hue, and the midpoint is a neutral: no hue at
    # the centre of a diverging scale.
    assert float(np.linalg.norm(high - home)) < float(np.linalg.norm(high - away))
    assert float(np.linalg.norm(low - away)) < float(np.linalg.norm(low - home))
    assert float(mid.max() - mid.min()) < 0.06


def test_control_cmap_is_mirrored_for_the_other_team() -> None:
    home_cmap = plots.control_cmap("home")
    away_cmap = plots.control_cmap("away")
    assert np.allclose(home_cmap(1.0), away_cmap(0.0))


# --------------------------------------------------------------------------- #
# Pitch geometry
# --------------------------------------------------------------------------- #


def test_pitch_constants_fit_inside_the_pitch() -> None:
    assert 0 < plots.PENALTY_AREA_WIDTH < PITCH_WIDTH
    assert 0 < plots.SIX_YARD_WIDTH < plots.PENALTY_AREA_WIDTH
    assert 0 < plots.GOAL_WIDTH < plots.SIX_YARD_WIDTH
    assert 0 < plots.SIX_YARD_DEPTH < plots.PENALTY_AREA_DEPTH
    assert 2 * plots.PENALTY_AREA_DEPTH < PITCH_LENGTH
    # The penalty spot is inside its box and the arc reaches beyond it, which is
    # the only reason the arc is visible at all.
    assert plots.SIX_YARD_DEPTH < plots.PENALTY_SPOT_DIST < plots.PENALTY_AREA_DEPTH
    assert plots.PENALTY_SPOT_DIST + plots.PENALTY_ARC_R > plots.PENALTY_AREA_DEPTH
    # Centre circles from both halves must not touch.
    assert 2 * plots.CENTRE_CIRCLE_R < PITCH_LENGTH / 2


def test_drawn_pitch_markings_stay_inside_the_pitch() -> None:
    fig, ax = plots.pitch_figure()
    for patch in ax.patches:
        box = _patch_bounds(patch)
        # Goals are the one marking that legitimately sits outside the field.
        assert box.x0 >= -plots.GOAL_DEPTH - 1e-6
        assert box.x1 <= PITCH_LENGTH + plots.GOAL_DEPTH + 1e-6
        assert box.y0 >= -1e-6
        assert box.y1 <= PITCH_WIDTH + 1e-6
    closes(fig)


def test_draw_pitch_scales_to_a_different_pitch_size() -> None:
    fig, ax = plots.pitch_figure(length=106.0, width=68.0)
    assert ax.get_xlim()[1] > 106.0
    widest = max(_patch_bounds(patch).x1 for patch in ax.patches)
    assert widest == pytest.approx(106.0 + plots.GOAL_DEPTH, abs=0.01)
    closes(fig)


def test_draw_pitch_margin_clears_the_goals() -> None:
    fig, ax = plots.pitch_figure()
    assert ax.get_xlim()[0] <= -plots.GOAL_DEPTH
    assert ax.get_xlim()[1] >= PITCH_LENGTH + plots.GOAL_DEPTH
    closes(fig)


def test_pitch_axes_box_has_the_pitch_aspect() -> None:
    fig, ax = plots.pitch_figure()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    assert ax.get_box_aspect() == pytest.approx((y1 - y0) / (x1 - x0))
    closes(fig)


# --------------------------------------------------------------------------- #
# Heatmap orientation -- the regression test that matters
# --------------------------------------------------------------------------- #


def _patch_bounds(patch: object) -> object:
    """Data-space bounding box of a patch.

    ``Patch.get_extents`` reports DISPLAY coordinates, which say nothing about
    where a marking sits on the pitch.
    """
    return patch.get_patch_transform().transform_path(patch.get_path()).get_extents()


def _pixel_at(fig: Figure, pixels: np.ndarray, x: float, y: float) -> np.ndarray:
    """Colour of the rendered pixel at a canonical pitch position."""
    px, py = fig.axes[0].transData.transform((x, y))
    return pixels[int(round(fig.bbox.height - py)), int(round(px))]


def _pole_colours() -> tuple[np.ndarray, np.ndarray]:
    """RGB of the top and bottom of the home-attacking control scale."""
    cmap = plots.control_cmap("home")
    return np.asarray(cmap(1.0)[:3]) * 255.0, np.asarray(cmap(0.0)[:3]) * 255.0


def test_heatmap_first_row_renders_at_the_bottom_of_the_pitch() -> None:
    """A surface that is 1.0 only in row 0 must paint the BOTTOM touchline.

    Surfaces are ``(n_y, n_x)`` with row 0 at the lowest y. In a rendered image
    the lowest y is the LAST pixel row. Draw the array without
    ``origin="lower"`` and this reverses, silently, while still producing a
    picture that looks like pitch control.
    """
    control = np.zeros((32, 50))
    control[0, :] = 1.0
    fig = plots.control_heatmap(
        control,
        attacking_team="home",
        colorbar=False,
        interpolation="nearest",
        figsize=(6.0, 4.0),
    )
    pixels = render(fig)
    # Sample the exact cell centres of the first and last rows, mid-pitch in x.
    # Positions come from the axes' own data transform, so the test cannot be
    # fooled by margins or by the layout engine moving the axes.
    high, low = _pole_colours()
    bottom = _pixel_at(fig, pixels, PITCH_LENGTH / 2, PITCH_WIDTH / 64)
    top = _pixel_at(fig, pixels, PITCH_LENGTH / 2, PITCH_WIDTH * 63 / 64)
    assert np.linalg.norm(bottom - high) < np.linalg.norm(bottom - low), (
        "row 0 of the surface must render at the BOTTOM of the pitch"
    )
    assert np.linalg.norm(top - low) < np.linalg.norm(top - high)
    closes(fig)


def test_heatmap_last_column_renders_at_the_right_of_the_pitch() -> None:
    """The x axis must not be flipped either."""
    control = np.zeros((32, 50))
    control[:, -1] = 1.0
    fig = plots.control_heatmap(
        control,
        attacking_team="home",
        colorbar=False,
        interpolation="nearest",
        figsize=(6.0, 4.0),
    )
    pixels = render(fig)
    left = _pixel_at(fig, pixels, PITCH_LENGTH / 100, PITCH_WIDTH / 2)
    right = _pixel_at(fig, pixels, PITCH_LENGTH * 99 / 100, PITCH_WIDTH / 2)
    high, low = _pole_colours()
    assert np.linalg.norm(right - high) < np.linalg.norm(right - low)
    assert np.linalg.norm(left - low) < np.linalg.norm(left - high)
    closes(fig)


def test_heatmap_image_declares_lower_origin_and_full_pitch_extent() -> None:
    fig = plots.control_heatmap(np.full((32, 50), 0.5), colorbar=False)
    image = fig.axes[0].images[0]
    assert image.origin == "lower"
    assert tuple(image.get_extent()) == (0.0, PITCH_LENGTH, 0.0, PITCH_WIDTH)
    closes(fig)


@pytest.mark.parametrize("shape", [(8, 12), (32, 50), (68, 105), (17, 23)])
def test_heatmap_accepts_any_grid_resolution(shape: tuple[int, int]) -> None:
    fig = plots.control_heatmap(np.full(shape, 0.5), colorbar=True)
    image = fig.axes[0].images[0]
    assert image.get_array().shape == shape
    assert tuple(image.get_extent()) == (0.0, PITCH_LENGTH, 0.0, PITCH_WIDTH)
    closes(fig)


def test_heatmap_rejects_a_non_2d_surface() -> None:
    with pytest.raises(ValueError, match="2-D"):
        plots.control_heatmap(np.zeros(50))


# --------------------------------------------------------------------------- #
# Headless build at both panel sizes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("figsize", [HALF_WIDTH, FULL_WIDTH])
def test_control_heatmap_builds_at_both_panel_sizes(
    surface: np.ndarray, players: dict[str, np.ndarray], figsize: tuple[float, float]
) -> None:
    fig = plots.control_heatmap(
        surface,
        ball=players["ball"],
        attackers=players["attackers"],
        defenders=players["defenders"],
        attacking_team="home",
        attacker_labels=[str(i) for i in range(11)],
        defender_labels=[str(i) for i in range(11)],
        attacker_velocities=np.zeros((11, 2)),
        defender_velocities=np.full((11, 2), 2.0),
        title="Turnover",
        caption="a caption",
        team_labels={"home": "Rovers", "away": "United"},
        figsize=figsize,
    )
    assert fig.get_size_inches() == pytest.approx(np.asarray(figsize))
    assert render(fig).shape[2] == 3
    closes(fig)


@pytest.mark.parametrize("figsize", [HALF_WIDTH, FULL_WIDTH])
def test_moment_panel_builds_at_both_panel_sizes(
    surface: np.ndarray, players: dict[str, np.ndarray], figsize: tuple[float, float]
) -> None:
    fig = plots.moment_panel(
        surface,
        ball=players["ball"],
        attackers=players["attackers"],
        defenders=players["defenders"],
        attacking_team="away",
        title="Moment 1",
        caption="58'12\"  ·  recovery 0.62",
        figsize=figsize,
    )
    assert fig.get_size_inches() == pytest.approx(np.asarray(figsize))
    render(fig)
    closes(fig)


def test_moment_panel_without_a_colourbar_has_one_axes(surface: np.ndarray) -> None:
    fig = plots.moment_panel(surface, colorbar=False)
    assert len(fig.axes) == 1
    closes(fig)
    fig = plots.moment_panel(surface, colorbar=True)
    assert len(fig.axes) == 2
    closes(fig)


def test_moment_panel_with_no_players_and_no_ball_draws_no_marks_at_all() -> None:
    """Nothing known means nothing drawn -- and the panel says the ball is gone.

    The failure this guards against is not an exception. It is a panel that
    quietly puts a mark at the origin, or an ordinary-looking panel that has
    simply lost the ball and never mentions it.
    """
    fig = plots.moment_panel(np.full((32, 50), 0.5), ball=None, attackers=None, defenders=None)
    ax = fig.axes[0]
    assert marker_points(ax).size == 0, "no marker may be drawn for an unknown position"
    assert "ball untracked here" in key_labels(ax), (
        "a panel with no ball must SAY it has no ball, not just omit it"
    )
    closes(fig)


def test_moment_panel_drops_a_nan_ball_rather_than_drawing_it_at_the_origin() -> None:
    """A NaN ball must leave the panel with no ball mark anywhere.

    Named "ignores" and asserting nothing, this test used to pass just as
    happily on an implementation that drew the NaN ball at (0, 0). It now
    checks the corner pixel and the artist list.
    """
    surface = np.full((32, 50), 0.5)
    fig = plots.moment_panel(surface, ball=np.array([np.nan, np.nan]), colorbar=False)
    ax = fig.axes[0]
    assert marker_points(ax).size == 0
    assert "ball untracked here" in key_labels(ax)
    # Pixel-for-pixel identical to a panel that was never given a ball at all.
    # A NaN drawn at the origin, or anywhere else, breaks this immediately.
    blank = plots.moment_panel(surface, ball=None, colorbar=False)
    assert np.array_equal(render(fig), render(blank))
    closes(fig)
    closes(blank)


def test_moment_caption_reads_a_scored_loss_row() -> None:
    row = pd.Series(
        {
            "t": 2508.5,
            "losing_team": "home",
            "losing_player": "Player6",
            "recovery": 0.85,
            "epv_conceded": 0.00071,
            "phase": "final_third",
        }
    )
    caption = plots.moment_caption(row, recovering_player="Player23")
    assert "41'48\"" in caption
    assert "Player6" in caption
    assert "Player23" in caption
    assert "0.85" in caption
    assert "final third" in caption
    # The EPV swing is scaled and its unit is named. Printed raw at four
    # decimals -- which is what this asserted before -- every real caption in
    # Metrica game 1 reads "+0.0001" or "-0.0000".
    assert f"+0.71 {plots.EPV_DISPLAY_UNIT}" in caption
    assert "+0.0007" not in caption


def test_moment_caption_skips_missing_fields() -> None:
    caption = plots.moment_caption({"losing_team": "away", "recovery": np.nan})
    assert "Away" in caption
    assert "nan" not in caption


def test_moment_strip_renders_three_thumbnails(surface: np.ndarray) -> None:
    fig = plots.moment_strip(
        [surface, surface * 0.5, 1.0 - surface],
        offsets=(0.0, 2.0, 5.0),
        balls=[np.array([50.0, 34.0])] * 3,
        attackers=[np.zeros((3, 2)) + 60.0] * 3,
        defenders=[np.zeros((3, 2)) + 20.0] * 3,
        title="Five seconds",
    )
    # Three pitches plus the shared colourbar.
    assert len(fig.axes) == 4
    stamps = [ax.get_title(loc="left") for ax in fig.axes[:3]]
    assert stamps == ["t₀", "t₀ + 2 s", "t₀ + 5 s"]
    render(fig)
    closes(fig)


def test_moment_strip_works_with_one_thumbnail(surface: np.ndarray) -> None:
    fig = plots.moment_strip([surface], offsets=(0.0,), colorbar=False)
    assert len(fig.axes) == 1
    closes(fig)


def test_moment_strip_rejects_an_empty_sequence() -> None:
    with pytest.raises(ValueError, match="at least one surface"):
        plots.moment_strip([])


# --------------------------------------------------------------------------- #
# Loss map
# --------------------------------------------------------------------------- #


def test_loss_map_builds_for_both_teams(losses: pd.DataFrame) -> None:
    fig = plots.loss_map(losses, team_labels={"home": "Rovers", "away": "United"})
    caption = fig.get_supxlabel()
    assert f"{len(losses)} possession losses" in caption
    assert "60 scored" in caption
    render(fig)
    closes(fig)


def test_loss_map_filters_to_one_team(losses: pd.DataFrame) -> None:
    fig = plots.loss_map(losses, team="home")
    n_home = int((losses["losing_team"] == "home").sum())
    assert f"{n_home} possession losses" in fig.get_supxlabel()
    closes(fig)


def test_loss_map_mirrors_the_away_team_to_attack_right(losses: pd.DataFrame) -> None:
    """Away attacks -x in the canonical frame; a coach reads their own attack
    left to right, so a single-team map is rotated for them."""
    away = losses[losses["losing_team"] == "away"]
    plain = plots.loss_map(away, team="away", orient_for_team=False)
    turned = plots.loss_map(away, team="away", orient_for_team=True)
    plain_x = plain.axes[0].collections[0].get_offsets()[:, 0]
    turned_x = turned.axes[0].collections[0].get_offsets()[:, 0]
    assert np.allclose(np.sort(PITCH_LENGTH - plain_x), np.sort(turned_x))
    closes(plain)
    closes(turned)


def test_loss_map_keeps_unscored_losses_visible(losses: pd.DataFrame) -> None:
    table = losses.copy()
    table.loc[table.index[:5], "scored"] = False
    table.loc[table.index[:5], "recovery"] = np.nan
    fig = plots.loss_map(table)
    assert "55 scored" in fig.get_supxlabel()
    labels = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
    assert "not scored" in labels
    closes(fig)


def test_loss_map_drops_a_nan_loss_instead_of_plotting_it_at_the_origin(
    losses: pd.DataFrame,
) -> None:
    """A loss with no location must not appear anywhere on the pitch.

    Asserting only that this renders would pass on an implementation that
    plotted the NaN loss at (0, 0), which is a corner flag and reads as a
    perfectly ordinary turnover.
    """
    table = losses.copy()
    table.loc[table.index[0], ["x", "y"]] = np.nan
    fig = plots.loss_map(table)
    drawn = marker_points(fig.axes[0])
    assert len(drawn) == len(table) - 1
    assert np.isfinite(drawn).all()
    assert not np.isclose(drawn, 0.0).all(axis=1).any(), (
        "a loss with no location must be dropped, not drawn at the origin"
    )
    closes(fig)


def test_loss_map_survives_an_empty_table(losses: pd.DataFrame) -> None:
    fig = plots.loss_map(losses.head(0))
    assert "0 possession losses" in fig.get_supxlabel()
    render(fig)
    closes(fig)


def test_loss_map_survives_a_single_loss(losses: pd.DataFrame) -> None:
    fig = plots.loss_map(losses.head(1))
    assert "1 possession loss," in fig.get_supxlabel()
    closes(fig)


# --------------------------------------------------------------------------- #
# Sparkline, phase bars
# --------------------------------------------------------------------------- #


def test_recovery_sparkline_draws_one_line_per_team(losses: pd.DataFrame) -> None:
    fig = plots.recovery_sparkline(losses)
    ax = fig.axes[0]
    labelled = [line for line in ax.lines if line.get_label() in {"Home", "Away"}]
    assert len(labelled) == 2
    assert [line.get_label() for line in labelled] == ["Home", "Away"]
    assert ax.get_ylim() == (0.0, 1.0)
    render(fig)
    closes(fig)


def test_recovery_sparkline_marks_half_time(losses: pd.DataFrame) -> None:
    fig = plots.recovery_sparkline(losses)
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert "half time" in texts
    closes(fig)


def test_recovery_sparkline_handles_an_empty_table(losses: pd.DataFrame) -> None:
    fig = plots.recovery_sparkline(losses.head(0))
    assert any("No scored losses" in t.get_text() for t in fig.axes[0].texts)
    closes(fig)


def test_recovery_sparkline_plots_a_single_loss_at_its_own_minute_and_value(
    losses: pd.DataFrame,
) -> None:
    """One loss is one point, at the right minute, at its own value.

    A one-point rolling mean IS the point, so this pins both axes: a smoother
    that silently returned zeros, or a series plotted against the row index
    instead of the clock, would pass a bare "it renders" test.
    """
    one = losses.head(1)
    fig = plots.recovery_sparkline(one)
    ax = fig.axes[0]
    side = str(one["losing_team"].iloc[0])
    lines = [line for line in ax.lines if line.get_label() == plots.team_label(side)]
    assert len(lines) == 1
    xy = lines[0].get_xydata()
    assert xy.shape == (1, 2)
    assert xy[0, 0] == pytest.approx(float(one["t"].iloc[0]) / 60.0)
    assert xy[0, 1] == pytest.approx(float(one["recovery"].iloc[0]))
    closes(fig)


def test_rolling_mean_is_centred_and_counts_correctly() -> None:
    t = np.array([0.0, 10.0, 20.0, 30.0])
    values = np.array([0.0, 1.0, 0.0, 1.0])
    times, mean, count = plots._rolling_time_mean(t, values, 20.0)
    assert np.allclose(times, t)
    # A +/-10 s half window is closed at both ends.
    assert np.allclose(count, [2, 3, 3, 2])
    assert mean[0] == pytest.approx(0.5)
    assert mean[1] == pytest.approx(1.0 / 3.0)


def test_rolling_mean_drops_non_finite_samples() -> None:
    t = np.array([0.0, 1.0, 2.0])
    values = np.array([1.0, np.nan, 1.0])
    times, mean, count = plots._rolling_time_mean(t, values, 10.0)
    assert times.size == 2
    assert np.allclose(mean, 1.0)
    assert np.allclose(count, 2)


def test_phase_bars_groups_by_phase() -> None:
    table = pd.DataFrame(
        {
            "team": ["home", "home", "home", "away", "away", "away"],
            "phase": ["build_up", "middle", "final_third"] * 2,
            "mean_recovery": [0.28, 0.49, 0.70, 0.24, 0.44, 0.70],
        }
    )
    fig = plots.phase_bars(table)
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["build up", "middle", "final third"]
    assert len(ax.containers) == 2
    render(fig)
    closes(fig)


def test_phase_bars_handles_an_empty_table() -> None:
    fig = plots.phase_bars(pd.DataFrame(columns=["team", "phase", "mean_recovery"]))
    assert any("No phase data" in t.get_text() for t in fig.axes[0].texts)
    closes(fig)


# --------------------------------------------------------------------------- #
# Season scatter
# --------------------------------------------------------------------------- #


def test_season_scatter_fits_a_trend_and_labels_outliers(season: pd.DataFrame) -> None:
    fig = plots.season_scatter(season, n_labels=4)
    ax = fig.axes[0]
    assert ax.get_xscale() == "log"
    assert len(ax.lines) == 1  # the fitted trend
    labels = [t.get_text() for t in ax.texts]
    assert sum(text.startswith("Player") for text in labels) == 4
    assert any("r = " in text for text in labels)
    render(fig)
    closes(fig)


def test_season_scatter_survives_an_empty_frame() -> None:
    fig = plots.season_scatter(pd.DataFrame(columns=list(plots.SEASON_COLUMNS)))
    assert any("No season data" in t.get_text() for t in fig.axes[0].texts)
    render(fig)
    closes(fig)


def test_season_scatter_survives_a_single_point(season: pd.DataFrame) -> None:
    fig = plots.season_scatter(season.head(1))
    ax = fig.axes[0]
    assert not ax.lines  # a single point cannot define a trend
    assert "1 player" in fig.get_supxlabel()
    assert "too few points" in fig.get_supxlabel()
    render(fig)
    closes(fig)


def test_season_scatter_drops_non_positive_values_on_a_log_axis(season: pd.DataFrame) -> None:
    table = season.copy()
    table.loc[table.index[:3], "market_value_eur"] = [0.0, -1.0, np.nan]
    fig = plots.season_scatter(table)
    assert "3 dropped" in fig.get_supxlabel()
    closes(fig)


def test_season_scatter_reports_when_nothing_is_usable(season: pd.DataFrame) -> None:
    table = season.copy()
    table["market_value_eur"] = 0.0
    fig = plots.season_scatter(table)
    assert any("positive market value" in t.get_text() for t in fig.axes[0].texts)
    closes(fig)


def test_value_scatter_is_an_alias(season: pd.DataFrame) -> None:
    fig = plots.value_scatter(season, n_labels=0)
    assert fig.axes[0].get_xscale() == "log"
    closes(fig)


# --------------------------------------------------------------------------- #
# Tracking adapters
# --------------------------------------------------------------------------- #


def _tracking_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "team": ["home", "home", "away", "ball"],
            "jersey": ["1", "2", "3", None],
            "player_id": ["h1", "h2", "a1", "ball"],
            "x": [10.0, 20.0, 80.0, 50.0],
            "y": [30.0, 40.0, 20.0, 34.0],
            "vx": [1.0, 0.0, -1.0, 0.0],
            "vy": [0.0, 1.0, 0.0, 0.0],
        }
    )


def test_positions_and_ball_from_tracking() -> None:
    frame = _tracking_frame()
    assert plots.positions_from_tracking(frame, "home").shape == (2, 2)
    assert np.allclose(plots.ball_from_tracking(frame), [50.0, 34.0])


def test_ball_from_tracking_returns_none_when_untracked() -> None:
    frame = _tracking_frame()
    frame.loc[3, ["x", "y"]] = np.nan
    assert plots.ball_from_tracking(frame) is None
    assert plots.ball_from_tracking(frame[frame["team"] != "ball"]) is None


def test_every_figure_type_draws_something_and_closes(
    surface: np.ndarray, losses: pd.DataFrame, season: pd.DataFrame
) -> None:
    """One pass over the whole public surface, the way the report uses it.

    Every figure must actually put ink on the paper. Rendering a blank sheet
    raises nothing at all, so the old version of this test -- which only called
    ``render`` and threw the pixels away -- would have passed on eight empty
    pages. The floors are deliberately loose: the point is "not blank", not a
    pixel-exact fingerprint that breaks on a font update.
    """
    phase_table = pd.DataFrame(
        {
            "team": ["home", "away"],
            "phase": ["middle", "middle"],
            "mean_recovery": [0.5, 0.4],
            "scored_losses": [40, 44],
        }
    )
    figures = {
        "pitch": plots.pitch_figure()[0],
        "control_heatmap": plots.control_heatmap(surface, figsize=HALF_WIDTH),
        "loss_map": plots.loss_map(losses),
        "moment_panel": plots.moment_panel(surface, caption="x"),
        "moment_strip": plots.moment_strip([surface, surface, surface]),
        "sparkline": plots.recovery_sparkline(losses),
        "phase_bars": plots.phase_bars(phase_table),
        "season_scatter": plots.season_scatter(season),
    }
    for name, fig in figures.items():
        assert fig.axes, f"{name} produced a figure with no axes"
        assert ink_fraction(fig) > 0.02, f"{name} rendered an all-but-blank page"
        closes(fig)


# --------------------------------------------------------------------------- #
# The ball fallback
# --------------------------------------------------------------------------- #


def test_resolve_ball_prefers_tracked_then_held_then_says_it_does_not_know() -> None:
    tracked, state = plots.resolve_ball(np.array([10.0, 20.0]), np.array([90.0, 60.0]))
    assert state == "tracked"
    assert np.allclose(tracked, [10.0, 20.0])

    held, state = plots.resolve_ball(None, np.array([90.0, 60.0]))
    assert state == "inferred"
    assert np.allclose(held, [90.0, 60.0])

    held, state = plots.resolve_ball(np.array([np.nan, np.nan]), np.array([90.0, 60.0]))
    assert state == "inferred"
    assert np.allclose(held, [90.0, 60.0])

    nothing, state = plots.resolve_ball(None, None)
    assert state == "unknown"
    assert nothing is None
    assert set(plots.BALL_STATES) == {"tracked", "inferred", "unknown"}


def test_untracked_ball_falls_back_to_the_last_known_position(surface: np.ndarray) -> None:
    """The ball is untracked in 39.1% of Metrica game 1's frames.

    Without this fallback two of the five printed top moments render with no
    ball at all, so the mark has to appear at the held position -- and it has
    to be visibly a DIFFERENT mark, not a tracked ball drawn on a guess.
    """
    held = np.array([70.0, 20.0])
    fig = plots.moment_panel(surface, ball=None, ball_last_known=held)
    ax = fig.axes[0]
    drawn = marker_points(ax)
    assert len(drawn) == 1
    assert np.allclose(drawn[0], held)
    assert "ball (last known)" in key_labels(ax)

    balls = [line for line in ax.lines if line.get_marker() == "o"]
    assert len(balls) == 1
    # Open, where a tracked ball is a filled disc. Fill is the whole signal.
    assert balls[0].get_markerfacecolor() == "none"
    closes(fig)


def test_a_tracked_ball_is_drawn_solid_and_an_inferred_one_is_not(surface: np.ndarray) -> None:
    at = np.array([70.0, 20.0])
    tracked = plots.moment_panel(surface, ball=at)
    inferred = plots.moment_panel(surface, ball=None, ball_last_known=at)

    solid = [line for line in tracked.axes[0].lines if line.get_marker() == "o"]
    open_ring = [line for line in inferred.axes[0].lines if line.get_marker() == "o"]
    assert len(solid) == len(open_ring) == 1
    assert solid[0].get_markerfacecolor() == plots.SURFACE
    assert open_ring[0].get_markerfacecolor() == "none"
    assert open_ring[0].get_markersize() > solid[0].get_markersize()
    assert "ball" in key_labels(tracked.axes[0])
    assert "ball (last known)" in key_labels(inferred.axes[0])
    closes(tracked)
    closes(inferred)


def test_a_panel_with_no_ball_still_marks_the_turnover(surface: np.ndarray) -> None:
    """Never a control panel with no ball AND no indication of the turnover."""
    loss_xy = np.array([80.0, 40.0])
    fig = plots.moment_panel(surface, ball=None, ball_last_known=None, loss_xy=loss_xy)
    ax = fig.axes[0]
    crosses = [line for line in ax.lines if line.get_marker() is plots.TURNOVER_MARKER]
    assert len(crosses) == 1
    assert np.allclose(crosses[0].get_xydata()[0], loss_xy)
    assert "ball lost here" in key_labels(ax)
    assert "ball untracked here" in key_labels(ax)
    closes(fig)


def test_the_turnover_cross_never_hides_the_ball_it_sits_on(surface: np.ndarray) -> None:
    """At t0 the loss location IS the tracked ball, so both land on one point.

    The crosshair therefore has to be open in the middle and drawn under the
    ball. A filled cross on top swallows the ball and the panel silently loses
    the mark it was added to protect.
    """
    at = np.array([60.0, 34.0])
    fig = plots.moment_panel(surface, ball=at, loss_xy=at, interpolation="nearest")
    ax = fig.axes[0]
    cross = next(line for line in ax.lines if line.get_marker() is plots.TURNOVER_MARKER)
    ball = next(line for line in ax.lines if line.get_marker() == "o")
    assert cross.get_markerfacecolor() == "none"
    assert cross.get_zorder() < ball.get_zorder()
    # And the ball really survives to the pixels: its centre is paper-coloured,
    # which neither the ink cross nor the mid-grey surface would leave it.
    pixels = render(fig)
    centre = _pixel_at(fig, pixels, at[0], at[1])
    paper = np.asarray(matplotlib.colors.to_rgb(plots.SURFACE)) * 255.0
    assert np.linalg.norm(centre - paper) < 30.0
    closes(fig)


def test_moment_strip_holds_the_ball_forward_per_thumbnail(surface: np.ndarray) -> None:
    at = np.array([44.0, 30.0])
    fig = plots.moment_strip(
        [surface, surface, surface],
        offsets=(0.0, 2.0, 5.0),
        balls=[at, None, None],
        ball_last_known=[at, at, at],
        loss_xy=at,
    )
    kinds = []
    for ax in fig.axes[:3]:
        ball = next(line for line in ax.lines if line.get_marker() == "o")
        kinds.append("tracked" if ball.get_markerfacecolor() == plots.SURFACE else "inferred")
        assert np.allclose(ball.get_xydata()[0], at)
    assert kinds == ["tracked", "inferred", "inferred"]
    assert "last tracked position, held" in fig.get_supxlabel()
    closes(fig)


def test_moment_strip_says_so_when_the_ball_is_untracked(surface: np.ndarray) -> None:
    fig = plots.moment_strip([surface, surface], offsets=(0.0, 2.0), balls=[None, None])
    for ax in fig.axes[:2]:
        assert not [line for line in ax.lines if line.get_marker() == "o"]
    assert "untracked on 2 thumbnails" in fig.get_supxlabel()
    closes(fig)


# --------------------------------------------------------------------------- #
# The attacking-direction note
# --------------------------------------------------------------------------- #


def test_direction_note_text_names_both_teams_and_both_directions() -> None:
    text = plots.direction_note_text({"home": "Rovers", "away": "United"})
    assert "Rovers" in text and "United" in text
    assert text.index("United") < text.index("Rovers"), "away attacks -x, so it comes first"
    assert "←" in text and "→" in text
    compact = plots.direction_note_text({"home": "Rovers", "away": "United"}, compact=True)
    assert len(compact) < len(text)
    assert "Rovers" in compact and "United" in compact


@pytest.mark.parametrize("builder", ["control_heatmap", "moment_panel"])
def test_control_surfaces_always_say_which_way_the_teams_attack(
    surface: np.ndarray, builder: str
) -> None:
    """A control surface with no direction note is not information.

    "Away United own the right-hand box" means nothing at all until you know
    which goal Away United are running at.
    """
    labels = {"home": "Rovers", "away": "United"}
    fig = getattr(plots, builder)(surface, attacking_team="away", team_labels=labels)
    assert plots.direction_note_text(labels) in notes(fig.axes[0])
    closes(fig)


def test_moment_strip_carries_the_direction_note_once(surface: np.ndarray) -> None:
    labels = {"home": "Rovers", "away": "United"}
    fig = plots.moment_strip([surface] * 3, team_labels=labels)
    printed = [note for ax in fig.axes for note in notes(ax) if "Rovers" in note]
    assert printed == [plots.direction_note_text(labels, compact=True)]
    closes(fig)


def test_direction_note_can_be_turned_off(surface: np.ndarray) -> None:
    fig = plots.moment_panel(surface, direction=False)
    assert not [note for note in notes(fig.axes[0]) if "attacking" in note]
    closes(fig)


def test_loss_map_notes_the_direction_for_one_team_and_for_both(
    losses: pd.DataFrame,
) -> None:
    labels = {"home": "Rovers", "away": "United"}
    both = plots.loss_map(losses, team_labels=labels)
    assert plots.direction_note_text(labels) in notes(both.axes[0])
    one = plots.loss_map(losses, team="away", team_labels=labels)
    assert "United attacking →" in notes(one.axes[0])
    closes(both)
    closes(one)


# --------------------------------------------------------------------------- #
# Reading EPV
# --------------------------------------------------------------------------- #


def test_format_epv_scales_and_names_its_unit() -> None:
    assert plots.EPV_DISPLAY_SCALE == 1000.0
    assert plots.format_epv(0.00071) == f"+0.71 {plots.EPV_DISPLAY_UNIT}"
    assert plots.format_epv(-0.0026) == f"-2.60 {plots.EPV_DISPLAY_UNIT}"
    assert plots.format_epv(0.00071, unit=False) == "+0.71"
    # The whole measured range of the raw quantity in Metrica game 1 must come
    # out as distinguishable numbers, which is the entire point of the scaling.
    low, high = plots.format_epv(-0.0026426), plots.format_epv(0.0040011)
    assert low != high
    assert {low, high}.isdisjoint({"+0.0000", "-0.0000"})


def test_format_epv_returns_nothing_for_a_missing_value() -> None:
    assert plots.format_epv(None) == ""
    assert plots.format_epv(np.nan) == ""
    assert plots.format_epv("not a number") == ""


def test_epv_unit_is_defined_somewhere_a_report_can_print_it() -> None:
    assert plots.EPV_DISPLAY_UNIT in plots.EPV_UNIT_GLOSS
    assert "1000" in plots.EPV_UNIT_GLOSS


def test_loss_map_caption_names_the_epv_unit_in_plain_words(losses: pd.DataFrame) -> None:
    caption = plots.loss_map(losses).get_supxlabel()
    assert plots.EPV_DISPLAY_UNIT in caption
    assert "bigger dot" in caption
    # The old caption was "marker area = |epv conceded|", which is notation, not
    # a caption. Neither the absolute-value bars nor the raw column name belong.
    assert "|" not in caption
    assert "epv_conceded" not in caption


# --------------------------------------------------------------------------- #
# One channel, one meaning
# --------------------------------------------------------------------------- #


def _closer_to(colour: object, target: str, than: str) -> bool:
    rgb = np.asarray(matplotlib.colors.to_rgb(colour))
    a = np.asarray(matplotlib.colors.to_rgb(target))
    b = np.asarray(matplotlib.colors.to_rgb(than))
    return bool(np.linalg.norm(rgb - a) < np.linalg.norm(rgb - b))


def test_recovery_ramp_stays_inside_its_team_hue() -> None:
    """Hue means team on every page, so the recovery ramp may not spend it.

    The ramp this replaced ran red to indigo, which put page 2 in a colour
    language none of the other pages spoke.
    """
    for side, other in (("home", "away"), ("away", "home")):
        ramp = plots.recovery_cmap(side)
        for level in (0.25, 0.5, 0.75, 1.0):
            assert _closer_to(
                ramp(level), plots.team_colour(side), plots.team_colour(other)
            ), f"{side} ramp leaves its own hue at {level}"
        # And lightness, not hue, is what carries the number.
        assert sum(ramp(0.05)[:3]) > sum(ramp(0.95)[:3])


def test_recovery_ramp_never_reaches_the_paper() -> None:
    """A loss the press never recovered from is the last mark that may vanish."""
    paper = np.asarray(matplotlib.colors.to_rgb(plots.SURFACE))
    pitch = np.asarray(matplotlib.colors.to_rgb(plots.PITCH_FILL))
    for side in ("home", "away"):
        low = np.asarray(plots.recovery_cmap(side)(0.0)[:3])
        assert np.linalg.norm(low - paper) > 0.1
        assert np.linalg.norm(low - pitch) > 0.1


def test_loss_map_colours_each_team_in_its_own_hue() -> None:
    """Blue is home on this page exactly as it is on every other page.

    One loss per team, at positions that identify which mark is whose, so the
    hue check cannot be satisfied by matching the wrong collection.
    """
    table = pd.DataFrame(
        {
            "t": [10.0, 20.0],
            "losing_team": ["home", "away"],
            "x": [25.0, 80.0],
            "y": [10.0, 58.0],
            "recovery": [0.95, 0.95],
            "epv_conceded": [0.002, 0.002],
            "scored": [True, True],
        }
    )
    fig = plots.loss_map(table, orient_for_team=False)
    # Scatter face colours are resolved from the colormap at DRAW time; read
    # before that, every mapped collection still reports matplotlib's default
    # blue and a broken figure passes.
    fig.canvas.draw()
    where = {"home": np.array([25.0, 10.0]), "away": np.array([80.0, 58.0])}
    seen = set()
    for collection in fig.axes[0].collections:
        offsets = np.asarray(collection.get_offsets(), dtype=float)
        if len(offsets) != 1:
            continue
        side = next(s for s, xy in where.items() if np.allclose(offsets[0], xy))
        other = "away" if side == "home" else "home"
        face = collection.get_facecolor()[0]
        assert _closer_to(face[:3], plots.team_colour(side), plots.team_colour(other)), (
            f"the {side} loss is not drawn in the {side} hue"
        )
        seen.add(side)
    assert seen == {"home", "away"}
    closes(fig)


def test_loss_map_puts_the_sign_in_the_fill_and_the_size_in_the_area() -> None:
    """Two presses of equal magnitude and opposite sign must not look alike.

    Marker area alone is unsigned, so a press that pushed the opponent
    backwards used to draw exactly as loudly as one that let them make ground.
    """
    table = pd.DataFrame(
        {
            "t": [10.0, 20.0],
            "losing_team": ["home", "home"],
            "x": [30.0, 70.0],
            "y": [20.0, 48.0],
            "recovery": [0.6, 0.6],
            "epv_conceded": [0.003, -0.003],
            "scored": [True, True],
        }
    )
    fig = plots.loss_map(table)
    fig.canvas.draw()
    ax = fig.axes[0]
    def is_filled(collection: object) -> bool:
        face = collection.get_facecolor()
        return bool(len(face)) and float(face[0][3]) > 0.0

    filled = [c for c in ax.collections if is_filled(c)]
    hollow = [c for c in ax.collections if not is_filled(c)]
    assert len(filled) == 1 and len(hollow) == 1
    assert np.allclose(filled[0].get_offsets()[0], [30.0, 20.0])
    assert np.allclose(hollow[0].get_offsets()[0], [70.0, 48.0])
    # Equal magnitude, so equal area: the size says how much, the fill says
    # which way.
    assert float(filled[0].get_sizes()[0]) == pytest.approx(float(hollow[0].get_sizes()[0]))
    assert "solid = " in fig.get_supxlabel() and "open ring = " in fig.get_supxlabel()
    closes(fig)


def test_loss_map_marks_unscored_losses_without_using_a_team_hue(
    losses: pd.DataFrame,
) -> None:
    table = losses.copy()
    table.loc[table.index[:5], "scored"] = False
    table.loc[table.index[:5], "recovery"] = np.nan
    fig = plots.loss_map(table)
    fig.canvas.draw()
    ax = fig.axes[0]
    assert "not scored" in key_labels(ax)
    crosses = [c for c in ax.collections if len(c.get_offsets()) == 5]
    assert crosses, "the five unscored losses must still be drawn"
    face = crosses[0].get_edgecolor()[0]
    assert _closer_to(face[:3], plots.INK_MUTED, plots.HOME_COLOUR)
    assert _closer_to(face[:3], plots.INK_MUTED, plots.AWAY_COLOUR)
    closes(fig)


def test_loss_map_gives_every_team_shown_its_own_labelled_colourbar(
    losses: pd.DataFrame,
) -> None:
    labels = {"home": "Rovers", "away": "United"}
    both = plots.loss_map(losses, team_labels=labels)
    bars = [ax for ax in both.axes if ax.get_label() == "<colorbar>"]
    assert len(bars) == 2
    named = {ax.get_ylabel() for ax in bars}
    assert named == {"Rovers", "United"}
    # Numbers on EVERY bar. These used to read "walked away / even / owned it"
    # with no number anywhere, so a 0.55 dot and a 0.95 dot were the same mark.
    for bar in bars:
        assert [t.get_text() for t in bar.get_xticklabels()] == [
            "0",
            "0.25",
            "0.5",
            "0.75",
            "1",
        ]
    # ...and the words those numbers mean, once, under the stack.
    gloss = [ax.get_xlabel() for ax in bars if ax.get_xlabel()]
    assert gloss == ["5 s recovery share — 0 = walked away, 0.5 = even, 1 = owned it"]
    one = plots.loss_map(losses, team="home", team_labels=labels)
    assert len([ax for ax in one.axes if ax.get_label() == "<colorbar>"]) == 1
    closes(both)
    closes(one)


# --------------------------------------------------------------------------- #
# Sample sizes on the phase bars
# --------------------------------------------------------------------------- #


@pytest.fixture
def phase_table() -> pd.DataFrame:
    """The measured Metrica game 1 shape: build-up rests on 11 and 13 losses."""
    return pd.DataFrame(
        {
            "team": ["home"] * 3 + ["away"] * 3,
            "phase": ["build_up", "middle", "final_third"] * 2,
            "mean_recovery": [0.28, 0.49, 0.70, 0.24, 0.44, 0.70],
            "scored_losses": [11, 60, 57, 13, 53, 61],
        }
    )


def test_phase_bars_print_the_sample_size_on_every_bar(phase_table: pd.DataFrame) -> None:
    printed = notes(plots.phase_bars(phase_table).axes[0])
    for n in phase_table["scored_losses"]:
        assert any(text.startswith(f"n={n}") for text in printed), f"n={n} is not on the chart"


def test_phase_bars_fade_and_flag_a_bar_that_rests_on_too_few_losses(
    phase_table: pd.DataFrame,
) -> None:
    """"Build-up 0.28 versus 0.24" reads as a finding until you see n=11."""
    fig = plots.phase_bars(phase_table, min_n=20)
    ax = fig.axes[0]
    bars = {}
    for container, side in zip(ax.containers, ("home", "away"), strict=True):
        for patch, phase in zip(container, ("build_up", "middle", "final_third"), strict=True):
            bars[(side, phase)] = patch

    thin = bars[("home", "build_up")]
    solid = bars[("home", "middle")]
    assert _closer_to(thin.get_facecolor(), plots.SURFACE, plots.HOME_COLOUR)
    assert _closer_to(solid.get_facecolor(), plots.HOME_COLOUR, plots.SURFACE)
    assert thin.get_linestyle() != solid.get_linestyle()
    assert any("(low n)" in text for text in notes(ax))
    caption = fig.get_supxlabel()
    assert "20" in caption and "2 bars" in caption
    closes(fig)


def test_phase_bars_legend_swatches_stay_solid_when_the_first_bar_is_faded(
    phase_table: pd.DataFrame,
) -> None:
    """A BarContainer takes its legend key from its FIRST bar -- here, a faded one."""
    fig = plots.phase_bars(phase_table)
    ax = fig.axes[0]
    swatches = ax.get_legend().legend_handles
    assert len(swatches) == 2
    for swatch, side in zip(swatches, ("home", "away"), strict=True):
        assert _closer_to(swatch.get_facecolor(), plots.team_colour(side), plots.SURFACE)
    closes(fig)


def test_phase_bars_flag_nothing_when_every_bar_is_well_sampled(
    phase_table: pd.DataFrame,
) -> None:
    fig = plots.phase_bars(phase_table, min_n=5)
    assert "faded" not in fig.get_supxlabel()
    assert not [text for text in notes(fig.axes[0]) if "(low n)" in text]
    closes(fig)


def test_phase_bars_fall_back_to_the_loss_count_and_then_to_no_count() -> None:
    only_losses = pd.DataFrame(
        {"team": ["home"], "phase": ["middle"], "mean_recovery": [0.5], "losses": [7]}
    )
    assert any("n=7" in text for text in notes(plots.phase_bars(only_losses).axes[0]))
    neither = only_losses.drop(columns=["losses"])
    assert not [text for text in notes(plots.phase_bars(neither).axes[0]) if "n=" in text]


# --------------------------------------------------------------------------- #
# The strip earns its height
# --------------------------------------------------------------------------- #


def test_strip_figsize_tracks_the_thumbnails_it_has_to_hold() -> None:
    one = plots.strip_figsize(1)
    three = plots.strip_figsize(3)
    assert one[0] == three[0]
    # Each pitch is pinned to the pitch aspect, so three across are each a
    # third as wide and a third as tall.
    assert one[1] > three[1]
    assert plots.strip_figsize(3, width_in=15.2)[1] > three[1]


def test_moment_strip_default_height_is_mostly_thumbnail(surface: np.ndarray) -> None:
    """Roughly a third of this figure's height used to be blank paper.

    The axes are pinned to the pitch aspect, so a guessed figure height becomes
    dead space that ``bbox_inches="tight"`` cannot reclaim -- the axes really
    is that short.
    """
    fig = plots.moment_strip([surface] * 3, offsets=(0.0, 2.0, 5.0))
    fig.canvas.draw()
    height = fig.get_size_inches()[1]
    used = sum(ax.get_position().height for ax in fig.axes[:3]) / 3.0
    assert used * height > 0.55 * height - plots._STRIP_CHROME_IN
    assert height == pytest.approx(plots.strip_figsize(3)[1])
    closes(fig)


def test_moment_strip_carries_a_colourbar_naming_both_teams(surface: np.ndarray) -> None:
    """As a standalone page-5 flowable, nothing else says which colour is who."""
    fig = plots.moment_strip(
        [surface] * 3, attacking_team="away", team_labels={"home": "Rovers", "away": "United"}
    )
    bars = [ax for ax in fig.axes if ax.get_label() == "<colorbar>"]
    assert len(bars) == 1
    # Both channels: the number on top, who it belongs to underneath. Team
    # names alone -- which is what this printed before -- mean no value can be
    # read off any cell of any surface in the strip.
    assert [t.get_text() for t in bars[0].get_xticklabels()] == [
        "0\nRovers",
        "0.25",
        "0.5\neven",
        "0.75",
        "1\nUnited",
    ]
    assert bars[0].get_xlabel() == "probability United reaches the ball there first"
    closes(fig)


# --------------------------------------------------------------------------- #
# Legibility: can the numbers actually be read off the picture
#
# Every assertion here is geometric and measured off the rendered canvas. A
# test that only calls the renderer is worth nothing against this class of bug:
# v0.1.0's season scatter raised nothing, returned a Figure, drew every artist
# it promised, and shipped a PNG whose headline statistic was covered by a
# player's name.
# --------------------------------------------------------------------------- #


def text_boxes(fig: Figure) -> dict[str, object]:
    """Every drawn annotation's TEXT box, in display pixels, keyed by its string.

    Measured through :class:`matplotlib.text.Text` directly rather than through
    ``plots``'s own helper, so a bug in the helper cannot hide a collision from
    the test that exists to catch it. ``Annotation.get_window_extent`` is not
    usable here: it unions in the leader line, which would report a label as
    occupying the whole corridor back to its marker.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes: dict[str, object] = {}
    for ax in fig.axes:
        for text in ax.texts:
            if not text.get_text():
                continue
            if hasattr(text, "update_positions"):
                text.update_positions(renderer)
            boxes[text.get_text()] = matplotlib.text.Text.get_window_extent(text)
    return boxes


def shared_area(one: object, other: object) -> float:
    """Area two display-space boxes share, in square pixels."""
    dx = min(one.x1, other.x1) - max(one.x0, other.x0)  # type: ignore[attr-defined]
    dy = min(one.y1, other.y1) - max(one.y0, other.y0)  # type: ignore[attr-defined]
    return float(dx * dy) if dx > 0.0 and dy > 0.0 else 0.0


def colourbars(fig: Figure) -> list[matplotlib.axes.Axes]:
    """Every colourbar axes on a figure."""
    return [ax for ax in fig.axes if ax.get_label() == "<colorbar>"]


#: Where the six labelled points sat on the chart that shipped broken: high in
#: the cloud and spread over the top two decades of market value, with the
#: highest of them far enough right that a name growing rightwards from it runs
#: into the corner the correlation note is pinned to. Round numbers picked to
#: reproduce that LAYOUT -- no season data is reproduced here, and none needs
#: to be, since the defect was entirely geometric.
LABELLED_LAYOUT: tuple[tuple[float, float], ...] = (
    (3.0e6, 10.7),
    (3.5e6, 12.0),
    (5.0e6, 12.1),
    (8.0e6, 11.9),
    (1.3e7, 11.4),
    (5.5e7, 10.4),
)


@pytest.fixture
def crowded_season() -> pd.DataFrame:
    """A synthetic of the chart that shipped broken: La Liga 2015/2016.

    Calibrated to the real one rather than invented, because the placement
    problem is entirely a matter of geometry and an arbitrary fixture tests an
    arbitrary geometry. 105 plotted players, market values spread over the
    three decades the real table spans, a cloud around 5.5 per 90, and the six
    largest residuals placed at :data:`LABELLED_LAYOUT`. The names are full
    length on purpose -- "Hernán Arsenio Pérez González (Espanyol)" is over
    300 px wide at print size and has to go somewhere.
    """
    rng = np.random.default_rng(17)
    cloud = 99
    value = np.concatenate(
        [10 ** rng.uniform(5.0, 8.1, cloud), [v for v, _ in LABELLED_LAYOUT]]
    )
    metric = np.concatenate(
        [np.clip(5.5 + rng.normal(0.0, 1.9, cloud), 1.2, 10.0), [m for _, m in LABELLED_LAYOUT]]
    )
    n = value.size
    return pd.DataFrame(
        {
            "player": [f"Aurelio Buendía Number {i}" for i in range(n)],
            "team": [f"Real Sociedad {i % 4}" for i in range(n)],
            "minutes": rng.uniform(400.0, 3000.0, n),
            "counterpress_per_90": metric,
            "market_value_eur": value,
        }
    )


def test_the_correlation_note_is_never_overprinted_by_a_label(
    crowded_season: pd.DataFrame,
) -> None:
    """The one number the chart exists to print has to be readable.

    v0.1.0 annotated the note AFTER ``_declutter`` and left it out of the
    collision set entirely, so on the shipped La Liga PNG "Samuel Castillejo
    Azuaga (Villarreal)" printed straight through "r = -0.09" and neither could
    be read. The note now goes in as a fixed obstacle, so nothing may touch it.
    """
    fig = plots.season_scatter(crowded_season, n_labels=6)
    boxes = text_boxes(fig)
    note = next(text for text in boxes if text.startswith("r = "))
    for text, box in boxes.items():
        if text == note:
            continue
        assert shared_area(boxes[note], box) == 0.0, (
            f"{text!r} is printed over the correlation note {note!r}"
        )
    closes(fig)


def test_point_labels_never_print_over_one_another(crowded_season: pd.DataFrame) -> None:
    """Two player names in the same pixels name nobody."""
    fig = plots.season_scatter(crowded_season, n_labels=6)
    boxes = text_boxes(fig)
    names = sorted(boxes)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            assert shared_area(boxes[first], boxes[second]) == 0.0, (
                f"{first!r} and {second!r} overlap"
            )
    closes(fig)


def test_point_labels_stay_inside_the_axes(crowded_season: pd.DataFrame) -> None:
    """A label off the edge is unattributable, and it wrecks the layout too.

    ``savefig(bbox_inches="tight")`` grows the canvas to swallow the overflow,
    which is what threw the shipped scatter's title off-centre.
    """
    fig = plots.season_scatter(crowded_season, n_labels=6)
    boxes = text_boxes(fig)
    fig.canvas.draw()
    frame = fig.axes[0].get_window_extent(renderer=fig.canvas.get_renderer())
    for text, box in boxes.items():
        escaped = box.width * box.height - shared_area(box, frame)
        assert escaped == 0.0, f"{text!r} hangs {escaped:.0f} px² outside the axes"
    closes(fig)


def test_a_label_that_had_to_move_is_joined_to_its_point(
    crowded_season: pd.DataFrame,
) -> None:
    """Displacement without a leader line is the defect, not the displacement.

    v0.1.0 pushed three names 22.8 pt below their anchors and drew nothing
    joining them, so six orange-ringed points could not be matched to six
    names. Near labels still get no line: a stub through every marker is
    clutter, and the point is right there.
    """
    fig = plots.season_scatter(crowded_season, n_labels=6)
    fig.canvas.draw()
    labelled = [
        text
        for text in fig.axes[0].texts
        if getattr(text, "arrow_patch", None) is not None
    ]
    assert len(labelled) == 6
    for text in labelled:
        moved = math.hypot(*text.get_position()) > plots._LEADER_AT
        assert text.arrow_patch.get_visible() is moved, (
            f"{text.get_text()!r} sits at {text.get_position()} with "
            f"leader visible={text.arrow_patch.get_visible()}"
        )
    closes(fig)


def test_no_leader_line_is_drawn_through_another_label(
    crowded_season: pd.DataFrame,
) -> None:
    """A hairline ruled across a player's name is a strikethrough, not a key."""
    fig = plots.season_scatter(crowded_season, n_labels=6)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = text_boxes(fig)
    for text in fig.axes[0].texts:
        arrow = getattr(text, "arrow_patch", None)
        if arrow is None or not arrow.get_visible():
            continue
        text.update_positions(renderer)
        ends = np.asarray(arrow.get_path().vertices)[[0, -1]]
        for other, box in boxes.items():
            if other == text.get_text():
                continue
            struck = plots._segment_in_box(tuple(ends[0]), tuple(ends[1]), box)
            assert struck == 0.0, (
                f"{text.get_text()!r}'s leader rules {struck:.0f} px across {other!r}"
            )
    closes(fig)


def test_a_labelled_marker_is_never_buried_under_a_different_name(
    crowded_season: pd.DataFrame,
) -> None:
    """The ringed point is the claim; a name on top of it hides the evidence."""
    fig = plots.season_scatter(crowded_season, n_labels=6)
    fig.canvas.draw()
    ax = fig.axes[0]
    # Selected by having a data-space anchor rather than by owning a leader
    # line, so this stays a real test against a build that draws no leaders at
    # all. The correlation note is anchored in axes fractions and drops out.
    anchored = [t for t in ax.texts if getattr(t, "xycoords", None) == "data"]
    assert len(anchored) == 6
    boxes = text_boxes(fig)
    for text in anchored:
        spot = ax.transData.transform(text.xy)
        for other, box in boxes.items():
            if other == text.get_text():
                continue
            assert not (
                box.x0 <= spot[0] <= box.x1 and box.y0 <= spot[1] <= box.y1
            ), f"{text.get_text()!r}'s marker is under the label {other!r}"
    closes(fig)


def test_the_note_reports_the_sample_size_and_an_interval(season: pd.DataFrame) -> None:
    """A bare r cannot say whether a null was measured or merely not ruled out."""
    fig = plots.season_scatter(season)
    note = next(t.get_text() for t in fig.axes[0].texts if t.get_text().startswith("r = "))
    assert f"(n = {len(season)})" in note
    assert "95% CI" in note
    closes(fig)


def test_the_interval_on_the_note_reproduces_the_shipped_la_liga_figure() -> None:
    """Fisher-z against the measured La Liga scatter: r = -0.0927 over 105 players."""
    low, high = plots._pearson_ci(-0.0927, 105)
    assert (round(low, 3), round(high, 3)) == (-0.279, 0.101)
    # Degenerate inputs have no interval and must say so rather than invent one.
    assert all(np.isnan(plots._pearson_ci(0.5, 3)))
    assert all(np.isnan(plots._pearson_ci(1.0, 50)))


@pytest.mark.parametrize("expected_bars", [1])
def test_every_colourbar_on_a_moment_panel_can_be_read_as_a_number(
    surface: np.ndarray, expected_bars: int
) -> None:
    """A colour ramp with no numbers on it cannot separate 0.55 from 0.95."""
    fig = plots.moment_panel(surface, team_labels={"home": "Rovers", "away": "United"})
    bars = colourbars(fig)
    assert len(bars) == expected_bars
    for bar in bars:
        numbers = [t.get_text().split("\n")[0] for t in bar.get_xticklabels()]
        assert len(numbers) >= 3
        assert [float(n) for n in numbers] == sorted(float(n) for n in numbers)
        assert bar.get_xlabel(), "a colourbar with no axis label names no quantity"
    closes(fig)


def test_every_figure_with_a_colourbar_puts_numbers_on_it(
    surface: np.ndarray, losses: pd.DataFrame
) -> None:
    """Swept across every renderer that draws one. None of them had numbers.

    The loss map's bars carry the recovery scale and the control bars carry the
    pitch-control probability; on the shipped PDF neither printed a single
    digit, so no value could be read off any heatmap cell or any dot on any of
    the six pages.
    """
    labels = {"home": "Rovers", "away": "United"}
    figures = {
        "loss_map": plots.loss_map(losses, team_labels=labels),
        "loss_map_one_team": plots.loss_map(losses, team="home", team_labels=labels),
        "control_heatmap": plots.control_heatmap(surface, team_labels=labels),
        "moment_panel": plots.moment_panel(surface, team_labels=labels),
        "moment_strip": plots.moment_strip([surface] * 3, team_labels=labels),
    }
    for name, fig in figures.items():
        bars = colourbars(fig)
        assert bars, f"{name} draws no colourbar"
        for bar in bars:
            numbers = [t.get_text().split("\n")[0] for t in bar.get_xticklabels()]
            assert len(numbers) >= 3, f"{name} has only {numbers} on a colourbar"
            values = [float(n) for n in numbers]
            assert values == sorted(values), f"{name} ticks out of order: {values}"
            assert values[0] == 0.0 and values[-1] == 1.0, (
                f"{name} does not label both ends of the ramp: {values}"
            )
        # Exactly one axis label per stack of bars: the quantity, named once.
        assert len([bar for bar in bars if bar.get_xlabel()]) == 1, (
            f"{name} names the quantity {len([b for b in bars if b.get_xlabel()])} times"
        )
        closes(fig)


def test_the_moment_key_says_which_dots_are_which_team(
    surface: np.ndarray, players: dict[str, np.ndarray]
) -> None:
    """The panel draws 22 coloured dots and used to name none of them."""
    fig = plots.moment_panel(
        surface,
        attackers=players["attackers"],
        defenders=players["defenders"],
        ball=players["ball"],
        attacking_team="home",
        team_labels={"home": "Rovers", "away": "United"},
    )
    assert {"Rovers", "United"} <= set(key_labels(fig.axes[0]))
    closes(fig)


def test_the_moment_key_promises_only_the_teams_it_drew(surface: np.ndarray) -> None:
    """A key entry for a team with no tracked positions is a lie about the frame."""
    fig = plots.moment_panel(surface, team_labels={"home": "Rovers", "away": "United"})
    assert not {"Rovers", "United"} & set(key_labels(fig.axes[0]))
    closes(fig)


def test_the_loss_map_key_puts_a_scale_on_the_dot_sizes(losses: pd.DataFrame) -> None:
    """Marker area is a quantitative channel; prose in the caption is not a key.

    The reference is the 95th percentile of the magnitudes, so the key's top
    entry has to be that value with its unit, not a hardcoded number.
    """
    fig = plots.loss_map(losses, size_by="epv_conceded")
    magnitude = np.abs(losses["epv_conceded"].to_numpy(dtype=float))
    reference = float(np.percentile(magnitude, 95.0))
    keys = key_labels(fig.axes[0])
    assert plots.format_epv(reference) in keys
    assert plots.format_epv(0.0) in keys
    assert sum(plots.EPV_DISPLAY_UNIT in k for k in keys) == 3
    closes(fig)


# --------------------------------------------------------------------------- #
# Fonts
# --------------------------------------------------------------------------- #


def test_register_fonts_adds_matplotlibs_own_dejavu_faces() -> None:
    paths = plots.register_fonts()
    assert paths, "matplotlib ships DejaVu Sans in mpl-data/fonts/ttf"
    assert [path.rsplit("/", 1)[-1] for path in paths] == list(plots.FONT_FILES)
    for path in paths:
        assert path.endswith(".ttf")


def test_the_face_the_figures_use_has_the_glyphs_real_player_names_need() -> None:
    """Helvetica turns "Vuković" into "Vukovi" plus a box, silently.

    matplotlib will at least warn, but a warning in a batch PDF build is a
    warning nobody sees, so the face is checked for the glyphs directly.
    """
    with plots.styled():
        resolved = matplotlib.font_manager.findfont(
            matplotlib.font_manager.FontProperties(
                family=matplotlib.rcParams["font.sans-serif"]
            )
        )
    face = FT2Font(resolved)
    for name in AWKWARD_NAMES:
        for character in name:
            assert face.get_char_index(ord(character)) != 0, (
                f"{resolved} has no glyph for {character!r} in {name!r}"
            )


def test_a_figure_renders_awkward_player_names_without_a_missing_glyph(
    surface: np.ndarray,
) -> None:
    caption = "  ·  ".join(AWKWARD_NAMES)
    fig = plots.moment_panel(surface, title="Łukasz · Gündoğan", caption=caption)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig.canvas.draw()
    missing = [str(w.message) for w in caught if "missing from font" in str(w.message)]
    assert not missing, missing
    assert fig.get_supxlabel() == caption
    closes(fig)


def test_awkward_names_actually_reach_the_pixels(season: pd.DataFrame) -> None:
    """A dropped glyph and a drawn glyph both raise nothing; pixels differ.

    Rendering "Vuković" and "Vukovic" must not produce identical ink -- if it
    does, the accented character never made it onto the page.
    """
    accented = season.copy()
    accented.loc[accented.index[0], "player"] = "Vuković"
    plain = season.copy()
    plain.loc[plain.index[0], "player"] = "Vukovic"
    a = plots.season_scatter(accented, n_labels=len(season), title="")
    b = plots.season_scatter(plain, n_labels=len(season), title="")
    assert any("Vuković" in text.get_text() for text in a.axes[0].texts)
    assert render(a).shape == render(b).shape
    assert not np.array_equal(render(a), render(b))
    closes(a)
    closes(b)
