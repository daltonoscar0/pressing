"""Tests for the report's figure renderers.

Three things are worth testing about a plotting module and the rest is taste:

1. **Does it build at all, headlessly.** Every figure here has to render under
   Agg inside a PDF build, at more than one size, without a display.
2. **Is the heatmap the right way up.** A control surface is ``(n_y, n_x)`` with
   row 0 at the BOTTOM touchline. Draw it without ``origin="lower"`` and you get
   a completely plausible-looking picture of the wrong half of the pitch, and no
   assertion about types or shapes will ever catch it. The test here renders to
   pixels and reads the colours back out, which is the only check that actually
   proves the orientation.
3. **Do the degenerate inputs survive.** Empty tables, single points and NaN
   coordinates all occur in real matches and none of them may raise.

Nothing here imports ``pyplot``, and one test asserts that the module hasn't
either -- figures built through ``pyplot`` accumulate in a process-global
registry and leak straight through a multi-page report build.
"""

from __future__ import annotations

import sys

import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from pressing import plots
from pressing.schema import PITCH_LENGTH, PITCH_WIDTH

matplotlib.use("Agg")

HALF_WIDTH = (5.4, 4.2)
FULL_WIDTH = (7.6, 5.2)


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


def test_moment_panel_survives_missing_players_and_ball(surface: np.ndarray) -> None:
    fig = plots.moment_panel(surface, ball=None, attackers=None, defenders=None)
    render(fig)
    closes(fig)


def test_moment_panel_ignores_a_nan_ball(surface: np.ndarray) -> None:
    fig = plots.moment_panel(surface, ball=np.array([np.nan, np.nan]))
    render(fig)
    closes(fig)


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
    assert "+0.0007" in caption
    assert "final third" in caption


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
    assert len(fig.axes) == 3
    stamps = [ax.get_title(loc="left") for ax in fig.axes]
    assert stamps == ["t₀", "t₀ + 2 s", "t₀ + 5 s"]
    render(fig)
    closes(fig)


def test_moment_strip_works_with_one_thumbnail(surface: np.ndarray) -> None:
    fig = plots.moment_strip([surface], offsets=(0.0,))
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


def test_loss_map_survives_nan_coordinates(losses: pd.DataFrame) -> None:
    table = losses.copy()
    table.loc[table.index[0], ["x", "y"]] = np.nan
    fig = plots.loss_map(table)
    render(fig)
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


def test_recovery_sparkline_handles_a_single_loss(losses: pd.DataFrame) -> None:
    fig = plots.recovery_sparkline(losses.head(1))
    render(fig)
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


def test_every_figure_type_builds_and_closes(
    surface: np.ndarray, losses: pd.DataFrame, season: pd.DataFrame
) -> None:
    """One pass over the whole public surface, the way the report uses it."""
    figures = [
        plots.pitch_figure()[0],
        plots.control_heatmap(surface, figsize=HALF_WIDTH),
        plots.loss_map(losses),
        plots.moment_panel(surface, caption="x"),
        plots.moment_strip([surface, surface, surface]),
        plots.recovery_sparkline(losses),
        plots.phase_bars(
            pd.DataFrame(
                {
                    "team": ["home", "away"],
                    "phase": ["middle", "middle"],
                    "mean_recovery": [0.5, 0.4],
                }
            )
        ),
        plots.season_scatter(season),
    ]
    for fig in figures:
        render(fig)
        closes(fig)
