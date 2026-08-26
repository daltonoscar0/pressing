"""Tests for the counter-press metric.

The synthetic scenes here are built so the right answer is known by
construction rather than by having run the code once and frozen the output:
a team that owns the whole pitch must score ~1, a team that owns none of it
must score ~0, and a scene that is its own mirror image must score exactly the
neutral value.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

import pressing
from pressing.counterpress import (
    _MIN_PARALLEL_LOSSES,
    BALL_POLICIES,
    DEFAULT_TIMEPOINTS,
    DEFAULT_WEIGHTING,
    LOSS_SCORE_COLUMNS,
    MAX_USEFUL_JOBS,
    WEIGHTING_LABELS,
    CounterPressParams,
    _ball_at,
    _build_index,
    alternate_weighting,
    default_jobs,
    headline_weighting,
    minutes_played,
    per_phase,
    per_player,
    per_team,
    score_losses,
    summarise_match,
    team_comparison,
    with_weighting,
)
from pressing.pitchcontrol import make_grid
from pressing.schema import (
    EVENT_COLUMNS,
    PITCH_LENGTH,
    PITCH_WIDTH,
    validate_events,
    validate_tracking,
)

FPS = 25.0
#: A coarse grid keeps the suite quick. It is still symmetric about the centre
#: spot, which is what the standoff test needs.
GRID = make_grid(n_x=25)

#: Neutral value of the AREA-weighted share on a mirror-symmetric standoff.
#: Not a fitted constant: it is 0.5 by construction, because rotating the scene
#: 180 degrees swaps the two teams and maps the grid onto itself.
NEUTRAL_AREA = 0.5

#: The EPV-weighted share has NO scene-independent neutral, which is the point.
#: The EPV weight is oriented toward one goal, so it does not survive the
#: rotation that swaps the two teams: on the seed-7 standoff below it measures
#: 0.380, while the mean over real Metrica losses is 0.80. That is the measured
#: reason the headline weighting is the area share.
STANDOFF_EPV_SEED7 = 0.3802


# --------------------------------------------------------------------------- #
# Synthetic scene builders
# --------------------------------------------------------------------------- #


def _spread(n: int, seed: int) -> np.ndarray:
    """``n`` positions spread deterministically over the whole pitch."""
    rng = np.random.default_rng(seed)
    return np.stack(
        [
            rng.uniform(4.0, PITCH_LENGTH - 4.0, n),
            rng.uniform(4.0, PITCH_WIDTH - 4.0, n),
        ],
        axis=1,
    )


def _huddle(n: int, centre: tuple[float, float], seed: int) -> np.ndarray:
    """``n`` positions packed into a 4 m box, i.e. controlling almost nothing."""
    rng = np.random.default_rng(seed)
    return np.stack(
        [
            centre[0] + rng.uniform(-2.0, 2.0, n),
            centre[1] + rng.uniform(-2.0, 2.0, n),
        ],
        axis=1,
    )


def _rotate(positions: np.ndarray) -> np.ndarray:
    """Rotate a scene 180 degrees about the centre spot."""
    return np.stack(
        [PITCH_LENGTH - positions[:, 0], PITCH_WIDTH - positions[:, 1]], axis=1
    )


def build_tracking(
    blocks: list[dict[str, object]],
    *,
    fps: float = FPS,
    match_id: str = "synthetic",
    ball_missing: set[int] | None = None,
) -> pd.DataFrame:
    """Assemble a canonical tracking frame from static per-period scenes.

    Args:
        blocks: One dict per period with ``frames`` (an int array), ``period``,
            ``home`` and ``away`` ``(n, 2)`` position arrays, ``ball``
            (a ``(2,)`` position or None), and optionally ``home_ids`` /
            ``away_ids`` to override the generated player labels.
        fps: Tracking rate.
        match_id: Value for the ``match_id`` column.
        ball_missing: Frames at which the ball row is omitted entirely, which
            is how the canonical loader represents an untracked ball.

    Returns:
        A validated canonical tracking frame with ``attrs`` populated.
    """
    missing = ball_missing or set()
    rows: list[dict[str, object]] = []
    for block in blocks:
        frames = np.asarray(block["frames"], dtype=np.int64)
        period = int(block["period"])  # type: ignore[call-overload]
        for frame in frames:
            for team, positions in (("home", block["home"]), ("away", block["away"])):
                labels = block.get(f"{team}_ids")
                for i, (x, y) in enumerate(np.asarray(positions, dtype=float)):
                    rows.append(
                        {
                            "match_id": match_id,
                            "t": frame / fps,
                            "period": period,
                            "frame": int(frame),
                            "team": team,
                            "player_id": (
                                f"{team[0].upper()}{i}" if labels is None else labels[i]  # type: ignore[index]
                            ),
                            "jersey": i + (0 if team == "home" else 20),
                            "x": x,
                            "y": y,
                            "vx": 0.0,
                            "vy": 0.0,
                            "confidence": 1.0,
                        }
                    )
            ball = block.get("ball")
            if ball is not None and int(frame) not in missing:
                rows.append(
                    {
                        "match_id": match_id,
                        "t": frame / fps,
                        "period": period,
                        "frame": int(frame),
                        "team": "ball",
                        "player_id": None,
                        "jersey": pd.NA,
                        "x": float(np.asarray(ball)[0]),
                        "y": float(np.asarray(ball)[1]),
                        "vx": 0.0,
                        "vy": 0.0,
                        "confidence": 1.0,
                    }
                )
    frame_out = pd.DataFrame(rows)
    frame_out["period"] = frame_out["period"].astype("int8")
    frame_out["frame"] = frame_out["frame"].astype("int32")
    frame_out["jersey"] = frame_out["jersey"].astype("Int16")
    frame_out["confidence"] = frame_out["confidence"].astype("float32")
    frame_out.attrs.update(
        {
            "match_id": match_id,
            "source": "synthetic",
            "fps": fps,
            "pitch_length": PITCH_LENGTH,
            "pitch_width": PITCH_WIDTH,
            "attack_direction": {"home": 1, "away": -1},
            "goalkeeper": {},
        }
    )
    out = validate_tracking(frame_out)
    out.attrs.update(frame_out.attrs)
    return out


def build_losses(records: list[dict[str, object]], *, fps: float = FPS) -> pd.DataFrame:
    """Assemble a loss table in the shape :func:`score_losses` expects."""
    rows = []
    for i, record in enumerate(records):
        frame = int(record["frame"])  # type: ignore[call-overload]
        rows.append(
            {
                "loss_id": i,
                "t": frame / fps,
                "frame": frame,
                "period": int(record.get("period", 1)),  # type: ignore[call-overload]
                "losing_team": record["losing_team"],
                "winning_team": "away" if record["losing_team"] == "home" else "home",
                "x": float(record.get("x", 52.5)),  # type: ignore[arg-type]
                "y": float(record.get("y", 34.0)),  # type: ignore[arg-type]
                "location_source": record.get("location_source", "tracking_ball"),
                "losing_player": record.get("losing_player", "H0"),
                "subtype": None,
                "event_index": int(record.get("event_index", i)),  # type: ignore[call-overload]
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["fps"] = fps
    out.attrs["attack_direction"] = {"home": 1, "away": -1}
    return out


def empty_events(fps: float = FPS) -> pd.DataFrame:
    """A canonical events frame with no rows, for scenes with no event stream."""
    blank = pd.DataFrame({column: [] for column in EVENT_COLUMNS})
    blank["period"] = blank["period"].astype("int8")
    blank["pressure"] = blank["pressure"].astype(bool)
    out = validate_events(blank)
    out.attrs["fps"] = fps
    return out


def build_events(records: list[dict[str, object]], *, fps: float = FPS) -> pd.DataFrame:
    """Assemble a minimal canonical events frame carrying ``start_frame``."""
    rows = []
    for record in records:
        frame = int(record["frame"])  # type: ignore[call-overload]
        rows.append(
            {
                "match_id": "synthetic",
                "competition": "synthetic",
                "season": pd.NA,
                "team_id": record["team"],
                "team": record["team"],
                "player_id": None,
                "player": None,
                "t": frame / fps,
                "period": int(record.get("period", 1)),  # type: ignore[call-overload]
                "type": record["type"],
                "subtype": None,
                "x": np.nan,
                "y": np.nan,
                "end_x": np.nan,
                "end_y": np.nan,
                "outcome": None,
                "pressure": False,
                "body_part": None,
                "extra": {"start_frame": frame, "end_frame": frame},
            }
        )
    frame_out = pd.DataFrame(rows)
    frame_out["period"] = frame_out["period"].astype("int8")
    out = validate_events(frame_out)
    out.attrs["fps"] = fps
    return out


def _static_scene(
    home: np.ndarray,
    away: np.ndarray,
    ball: tuple[float, float] | None = (52.5, 34.0),
    n_frames: int = 200,
    **kwargs: object,
) -> pd.DataFrame:
    """A single-period scene in which nobody moves."""
    return build_tracking(
        [
            {
                "frames": np.arange(1, n_frames + 1),
                "period": 1,
                "home": home,
                "away": away,
                "ball": None if ball is None else np.array(ball),
            }
        ],
        **kwargs,  # type: ignore[arg-type]
    )


def _score(
    tracking: pd.DataFrame,
    losses: pd.DataFrame,
    params: CounterPressParams | None = None,
) -> pd.DataFrame:
    """Score a synthetic scene on the coarse test grid."""
    return score_losses(
        tracking, empty_events(), losses=losses, params=params, grid=GRID
    )


# --------------------------------------------------------------------------- #
# The three acceptance scenarios
# --------------------------------------------------------------------------- #


def test_instant_regain_scores_one() -> None:
    """A counter-pressing team that owns the whole pitch scores ~1.0.

    The scene: the home team gives the ball away but is spread across the pitch
    while every away player is packed into a 4 m box in one corner. Home wins
    essentially every cell, so its control share -- and therefore its recovery
    -- is at the top of the range.
    """
    tracking = _static_scene(_spread(11, 0), _huddle(11, (3.0, 3.0), 1))
    losses = build_losses([{"frame": 10, "losing_team": "home"}])
    scored = _score(tracking, losses)

    row = scored.iloc[0]
    assert bool(row["scored"])
    assert row["n_timepoints"] == len(DEFAULT_TIMEPOINTS)
    assert row["recovery_area"] > 0.9
    assert row["recovery_epv"] > 0.9
    assert row["recovery"] == pytest.approx(row["recovery_area"])


def test_walking_the_ball_away_scores_zero() -> None:
    """The mirror scenario: the pressers own nothing, so recovery is ~0.0."""
    tracking = _static_scene(_huddle(11, (3.0, 3.0), 1), _spread(11, 0))
    losses = build_losses([{"frame": 10, "losing_team": "home"}])
    scored = _score(tracking, losses)

    row = scored.iloc[0]
    assert bool(row["scored"])
    assert row["recovery_area"] < 0.1
    assert row["recovery_epv"] < 0.1


@pytest.mark.parametrize("seed", [7, 11])
def test_symmetric_standoff_is_neutral(seed: int) -> None:
    """A scene that is its own mirror image scores exactly the neutral value.

    Neutral for the headline metric is 0.5, and it is 0.5 by construction, not
    by calibration: rotating the pitch 180 degrees about the centre spot swaps
    the two teams, maps the grid onto itself and leaves the ball on the centre
    spot, so the two control surfaces are exact reflections and the AREA share
    is exactly one half whatever the arrangement.
    """
    home = _spread(11, seed)
    tracking = _static_scene(home, _rotate(home))
    losses = build_losses([{"frame": 10, "losing_team": "home"}])
    scored = _score(tracking, losses)

    row = scored.iloc[0]
    assert row["recovery"] == pytest.approx(NEUTRAL_AREA, abs=1e-5)
    assert row["recovery_area"] == pytest.approx(NEUTRAL_AREA, abs=1e-5)
    assert row["share_t0_area"] == pytest.approx(NEUTRAL_AREA, abs=1e-5)


def test_the_epv_weighting_has_no_scene_independent_neutral() -> None:
    """The same standoff is NOT 0.5 under the EPV weighting.

    This is not a defect being tolerated, it is the measurement that decided
    the default. The EPV weight is oriented toward the goal the ball-winning
    team attacks, so the rotation that swaps the teams does not preserve it.
    """
    home = _spread(11, 7)
    tracking = _static_scene(home, _rotate(home))
    scored = _score(tracking, build_losses([{"frame": 10, "losing_team": "home"}]))

    row = scored.iloc[0]
    assert row["recovery_area"] == pytest.approx(NEUTRAL_AREA, abs=1e-5)
    assert row["recovery_epv"] == pytest.approx(STANDOFF_EPV_SEED7, abs=0.01)
    assert abs(row["recovery_epv"] - NEUTRAL_AREA) > 0.1


def test_default_weighting_is_the_area_share() -> None:
    """The headline column follows DEFAULT_WEIGHTING, and switching is free."""
    assert DEFAULT_WEIGHTING == "area"
    assert CounterPressParams().weighting == "area"

    home = _spread(11, 7)
    tracking = _static_scene(home, _rotate(home))
    scored = _score(tracking, build_losses([{"frame": 10, "losing_team": "home"}]))
    assert scored["recovery"].iloc[0] == pytest.approx(scored["recovery_area"].iloc[0])

    switched = with_weighting(scored, "epv")
    assert switched["recovery"].iloc[0] == pytest.approx(scored["recovery_epv"].iloc[0])
    assert switched.attrs["counterpress_params"].weighting == "epv"


# --------------------------------------------------------------------------- #
# Window boundaries
# --------------------------------------------------------------------------- #


def test_window_is_closed_at_both_ends() -> None:
    """The sample at exactly t0+5.000 s is inside the window.

    Tracking that ends exactly 125 frames (5.000 s at 25 Hz) after the loss
    yields all six timepoints; one frame shorter yields five.
    """
    losses = build_losses([{"frame": 10, "losing_team": "home"}])
    home, away = _spread(11, 0), _spread(11, 3)

    exact = _score(_static_scene(home, away, n_frames=10 + 125), losses)
    assert exact["n_timepoints"].iloc[0] == 6

    short = _score(_static_scene(home, away, n_frames=10 + 124), losses)
    assert short["n_timepoints"].iloc[0] == 5
    assert bool(short["scored"].iloc[0])


def test_window_never_reads_across_half_time() -> None:
    """A loss near the whistle is scored on its own period only.

    Period 1 is a scene the home team dominates and period 2 is its exact
    inverse. A loss 1 s before the break has only two timepoints inside period
    1; if the later ones leaked across the boundary the recovery would collapse
    toward the period-2 value instead of staying near 1.
    """
    dominant_home = _spread(11, 0)
    huddled_away = _huddle(11, (3.0, 3.0), 1)
    tracking = build_tracking(
        [
            {
                "frames": np.arange(1, 101),
                "period": 1,
                "home": dominant_home,
                "away": huddled_away,
                "ball": np.array([52.5, 34.0]),
            },
            {
                "frames": np.arange(101, 301),
                "period": 2,
                "home": huddled_away,
                "away": dominant_home,
                "ball": np.array([52.5, 34.0]),
            },
        ]
    )

    scored = _score(tracking, build_losses([{"frame": 75, "losing_team": "home"}]))
    row = scored.iloc[0]
    # Frames 75 and 100 are in period 1; 125, 150, 175 and 200 are in period 2.
    assert row["n_timepoints"] == 2
    assert bool(row["scored"])
    assert row["recovery_area"] > 0.9

    # One timepoint on its own is below min_timepoints, so the loss is skipped
    # rather than scored from a single frame.
    late = _score(tracking, build_losses([{"frame": 99, "losing_team": "home"}]))
    assert not bool(late["scored"].iloc[0])
    assert late["skip_reason"].iloc[0] == "window_truncated"
    assert late["n_timepoints"].iloc[0] == 1


def test_regain_window_is_closed_at_five_seconds() -> None:
    """An event at exactly t0+5.000 s is a regain; at t0+5.040 s it is not."""
    tracking = _static_scene(_spread(11, 0), _spread(11, 3), n_frames=400)
    losses = build_losses(
        [
            {"frame": 100, "losing_team": "home", "event_index": 0},
            {"frame": 200, "losing_team": "home", "event_index": 2},
        ]
    )
    events = build_events(
        [
            {"frame": 100, "team": "home", "type": "BALL LOST"},
            {"frame": 225, "team": "home", "type": "RECOVERY"},
            {"frame": 200, "team": "home", "type": "BALL LOST"},
            {"frame": 326, "team": "home", "type": "RECOVERY"},
        ]
    )
    scored = score_losses(tracking, events, losses=losses, grid=GRID)

    assert bool(scored["regained_within_5s"].iloc[0])
    assert scored["time_to_regain"].iloc[0] == pytest.approx(5.0)
    assert not bool(scored["regained_within_5s"].iloc[1])
    assert np.isnan(scored["time_to_regain"].iloc[1])


def test_regain_ignores_the_opponent_and_earlier_events() -> None:
    """Only the losing team's own later ball-carrying events count."""
    tracking = _static_scene(_spread(11, 0), _spread(11, 3), n_frames=400)
    losses = build_losses([{"frame": 100, "losing_team": "home", "event_index": 0}])
    events = build_events(
        [
            {"frame": 100, "team": "home", "type": "BALL LOST"},
            {"frame": 110, "team": "away", "type": "RECOVERY"},
            {"frame": 130, "team": "home", "type": "CHALLENGE"},
        ]
    )
    scored = score_losses(tracking, events, losses=losses, grid=GRID)
    assert not bool(scored["regained_within_5s"].iloc[0])


# --------------------------------------------------------------------------- #
# Missing ball policy
# --------------------------------------------------------------------------- #


def _scene_with_ball_gap(missing: set[int]) -> pd.DataFrame:
    """A static scene whose ball rows are absent at ``missing`` frames."""
    return build_tracking(
        [
            {
                "frames": np.arange(1, 301),
                "period": 1,
                "home": _spread(11, 0),
                "away": _spread(11, 3),
                "ball": np.array([20.0, 10.0]),
            }
        ],
        ball_missing=missing,
    )


def test_hold_policy_carries_the_last_known_ball_forward() -> None:
    """The documented default: hold, and say how many samples were filled."""
    tracking = _scene_with_ball_gap({150})
    losses = build_losses([{"frame": 100, "losing_team": "home"}])

    held = _score(tracking, losses, CounterPressParams(ball_policy="hold"))
    assert held["n_timepoints"].iloc[0] == 6
    assert held["n_ball_filled"].iloc[0] == 1

    index = _build_index(tracking, FPS)
    position, filled = _ball_at(index, 150, "hold", int(5 * FPS))
    assert filled
    assert position == pytest.approx(np.array([20.0, 10.0]))


def test_skip_policy_drops_the_timepoint() -> None:
    """``skip`` produces a shorter window rather than a filled ball."""
    tracking = _scene_with_ball_gap({150})
    losses = build_losses([{"frame": 100, "losing_team": "home"}])
    skipped = _score(tracking, losses, CounterPressParams(ball_policy="skip"))
    assert skipped["n_timepoints"].iloc[0] == 5
    assert skipped["n_ball_filled"].iloc[0] == 0
    assert bool(skipped["scored"].iloc[0])


def test_interpolate_policy_lands_between_the_neighbours() -> None:
    """``interpolate`` uses the surrounding known positions, not the last one."""
    tracking = build_tracking(
        [
            {
                "frames": np.arange(1, 51),
                "period": 1,
                "home": _spread(11, 0),
                "away": _spread(11, 3),
                "ball": np.array([10.0, 10.0]),
            },
            {
                "frames": np.arange(51, 101),
                "period": 1,
                "home": _spread(11, 0),
                "away": _spread(11, 3),
                "ball": np.array([30.0, 30.0]),
            },
        ],
        ball_missing=set(range(41, 61)),
    )
    index = _build_index(tracking, FPS)
    position, filled = _ball_at(index, 50, "interpolate", int(5 * FPS))
    assert filled
    # Frame 40 holds (10, 10) and frame 61 holds (30, 30); frame 50 is 10/21 of
    # the way between them.
    assert position[0] == pytest.approx(10.0 + 20.0 * 10.0 / 21.0)


def test_no_ball_at_the_loss_instant_skips_the_loss() -> None:
    """Under ``skip`` a loss with no ball at t0 is reported, not guessed."""
    tracking = _scene_with_ball_gap({100})
    losses = build_losses([{"frame": 100, "losing_team": "home"}])
    scored = _score(tracking, losses, CounterPressParams(ball_policy="skip"))
    assert not bool(scored["scored"].iloc[0])
    assert scored["skip_reason"].iloc[0] == "no_ball"
    assert np.isnan(scored["recovery"].iloc[0])


def test_hold_will_not_reach_back_further_than_the_limit() -> None:
    """A gap longer than ``ball_hold_max_s`` is not papered over."""
    tracking = _scene_with_ball_gap(set(range(50, 201)))
    losses = build_losses([{"frame": 100, "losing_team": "home"}])
    scored = _score(
        tracking, losses, CounterPressParams(ball_policy="hold", ball_hold_max_s=1.0)
    )
    assert scored["skip_reason"].iloc[0] == "no_ball"


def test_ball_policy_names_are_validated() -> None:
    """A typo in the policy name must fail loudly, not fall back silently."""
    with pytest.raises(ValueError, match="ball_policy"):
        CounterPressParams(ball_policy="hold-last")  # type: ignore[arg-type]
    assert set(BALL_POLICIES) == {"hold", "interpolate", "skip", "none"}


def test_a_loss_without_a_location_is_reported_not_dropped() -> None:
    """Metrica game 1 has exactly one of these; it must keep its row."""
    tracking = _static_scene(_spread(11, 0), _spread(11, 3))
    losses = build_losses(
        [{"frame": 10, "losing_team": "home", "x": np.nan, "y": np.nan}]
    )
    scored = _score(tracking, losses)
    assert len(scored) == 1
    assert not bool(scored["scored"].iloc[0])
    assert scored["skip_reason"].iloc[0] == "no_loss_location"
    assert pd.isna(scored["phase"].iloc[0])


# --------------------------------------------------------------------------- #
# Player attribution
# --------------------------------------------------------------------------- #


def _attribution_scene() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A loss at (50, 34) with home players at 14.9, 15.0, 15.1 and 40 m."""
    home = np.array(
        [
            [50.0 + 14.9, 34.0],
            [50.0 + 15.0, 34.0],
            [50.0 + 15.1, 34.0],
            [50.0 + 40.0, 34.0],
        ]
    )
    tracking = _static_scene(home, _spread(4, 5))
    losses = build_losses([{"frame": 10, "losing_team": "home", "x": 50.0, "y": 34.0}])
    return tracking, losses


def test_attribution_radius_is_inclusive() -> None:
    """14.9 m is in, exactly 15.0 m is in, 15.1 m is out."""
    tracking, losses = _attribution_scene()
    scored = _score(tracking, losses)
    players = per_player(scored, tracking, radius_m=15.0)
    credits = dict(zip(players["player_id"], players["losses_nearby"], strict=True))
    assert credits["H0"] == 1
    assert credits["H1"] == 1
    assert credits["H2"] == 0
    assert credits["H3"] == 0


def test_attribution_radius_is_a_parameter() -> None:
    """Widening the radius pulls the 15.1 m player in."""
    tracking, losses = _attribution_scene()
    scored = _score(tracking, losses)
    players = per_player(scored, tracking, radius_m=16.0)
    credits = dict(zip(players["player_id"], players["losses_nearby"], strict=True))
    assert credits["H2"] == 1
    assert credits["H3"] == 0


def test_attribution_credits_only_the_counter_pressing_team() -> None:
    """The team that WON the ball is not counter-pressing anything."""
    tracking, losses = _attribution_scene()
    scored = _score(tracking, losses)
    players = per_player(scored, tracking, radius_m=200.0)
    away = players[players["team"] == "away"]
    assert int(away["losses_nearby"].sum()) == 0
    assert int(players[players["team"] == "home"]["losses_nearby"].sum()) == 4


# --------------------------------------------------------------------------- #
# Minutes played
# --------------------------------------------------------------------------- #


def test_minutes_come_from_tracked_presence_not_ninety() -> None:
    """A player who leaves at half time gets half the minutes."""
    home = _spread(3, 0)
    tracking = build_tracking(
        [
            {
                "frames": np.arange(1, 101),
                "period": 1,
                "home": home,
                "away": _spread(3, 4),
                "ball": np.array([52.5, 34.0]),
            },
            {
                "frames": np.arange(101, 201),
                "period": 2,
                "home": home[:2],
                "away": _spread(2, 4),
                "ball": np.array([52.5, 34.0]),
            },
        ]
    )
    minutes = minutes_played(tracking).set_index(["team", "player_id"])["minutes"]
    # Frames 1..200 at 25 Hz is a 199-frame interval = 7.96 s.
    assert minutes[("home", "H0")] == pytest.approx(199 / FPS / 60.0)
    assert minutes[("home", "H2")] == pytest.approx(99 / FPS / 60.0)


def test_substitution_overlap_is_not_double_counted() -> None:
    """The outgoing and incoming players share the boundary frame exactly once.

    Metrica writes a substitution as a one-frame overlap: player A's last valid
    frame IS player B's first. Measuring a span as the closed interval
    ``last - first`` makes the pair sum to the true elapsed time.
    """
    scene, opponent = _spread(1, 0), _spread(1, 9)
    tracking = build_tracking(
        [
            # H0 plays frames 1..50 and H1 plays 50..100: both are present at
            # frame 50, exactly as Metrica writes a substitution.
            {
                "frames": np.arange(1, 51),
                "period": 1,
                "home": scene,
                "home_ids": ["H0"],
                "away": opponent,
                "away_ids": ["A0"],
                "ball": np.array([52.5, 34.0]),
            },
            {
                "frames": np.arange(50, 101),
                "period": 1,
                "home": scene,
                "home_ids": ["H1"],
                "away": opponent,
                "away_ids": ["A0"],
                "ball": None,
            },
        ]
    )

    minutes = minutes_played(tracking).set_index(["team", "player_id"])["minutes"]
    assert minutes[("home", "H0")] == pytest.approx(49 / FPS / 60.0)
    assert minutes[("home", "H1")] == pytest.approx(50 / FPS / 60.0)
    # The pair sums to the full 99-frame interval, with the shared frame counted
    # exactly once rather than twice.
    total = minutes[("home", "H0")] + minutes[("home", "H1")]
    assert total == pytest.approx(minutes[("away", "A0")])
    assert total == pytest.approx(99 / FPS / 60.0)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def _two_team_scene() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A scene with one loss per team, in different thirds."""
    tracking = _static_scene(_spread(11, 0), _spread(11, 3), n_frames=400)
    losses = build_losses(
        [
            {"frame": 100, "losing_team": "home", "x": 10.0, "y": 34.0},
            {"frame": 200, "losing_team": "away", "x": 10.0, "y": 34.0},
        ]
    )
    return tracking, _score(tracking, losses)


def test_scored_table_has_the_declared_columns() -> None:
    """LOSS_SCORE_COLUMNS is the contract the report builds against."""
    _, scored = _two_team_scene()
    assert tuple(scored.columns) == LOSS_SCORE_COLUMNS


def test_phase_uses_each_losing_team_own_direction() -> None:
    """The same x is build-up for home and final third for away."""
    _, scored = _two_team_scene()
    phases = dict(zip(scored["losing_team"], scored["phase"], strict=True))
    assert phases["home"] == "build_up"
    assert phases["away"] == "final_third"


def test_per_team_and_comparison() -> None:
    """Per-team aggregation and the cover-page head-to-head."""
    _, scored = _two_team_scene()
    teams = per_team(scored).set_index("team")
    assert list(teams.index) == ["away", "home"]
    assert int(teams.loc["home", "losses"]) == 1

    comparison = team_comparison(scored).set_index("metric")
    assert comparison.loc["losses", "difference"] == pytest.approx(0.0)
    assert comparison.loc["mean_recovery", "difference"] == pytest.approx(
        teams.loc["home", "mean_recovery"] - teams.loc["away", "mean_recovery"]
    )


def test_per_phase_emits_every_phase_for_every_team() -> None:
    """The report's bar chart always wants three bars per team."""
    _, scored = _two_team_scene()
    table = per_phase(scored)
    assert len(table) == 6
    assert set(table["phase"]) == {"build_up", "middle", "final_third"}
    empty = table[(table["team"] == "home") & (table["phase"] == "middle")]
    assert int(empty["losses"].iloc[0]) == 0


def test_summary_counts_scored_and_skipped() -> None:
    """Nothing disappears: skipped losses are counted and explained."""
    tracking = _static_scene(_spread(11, 0), _spread(11, 3))
    losses = build_losses(
        [
            {"frame": 10, "losing_team": "home"},
            {"frame": 20, "losing_team": "away", "x": np.nan, "y": np.nan},
        ]
    )
    scored = _score(tracking, losses)
    summary = summarise_match(scored)
    assert summary["n_losses"] == 2
    assert summary["n_scored"] == 1
    assert summary["n_skipped"] == 1
    assert summary["skip_reasons"] == {"no_loss_location": 1}


# --------------------------------------------------------------------------- #
# Parameter validation
# --------------------------------------------------------------------------- #


def test_timepoints_must_start_at_the_loss_instant() -> None:
    """Every window is anchored at t0; a set that is not is a bug."""
    with pytest.raises(ValueError, match="start at 0.0"):
        CounterPressParams(timepoints=(1.0, 2.0))


def test_timepoints_must_increase() -> None:
    """Duplicated or unsorted timepoints would silently reweight the mean."""
    with pytest.raises(ValueError, match="strictly increasing"):
        CounterPressParams(timepoints=(0.0, 2.0, 1.0))


def test_weighting_is_validated() -> None:
    """Only the two documented weightings exist."""
    with pytest.raises(ValueError, match="weighting"):
        CounterPressParams(weighting="epv-ish")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="weighting"):
        with_weighting(pd.DataFrame(), "neither")  # type: ignore[arg-type]


def test_denser_timepoints_are_available() -> None:
    """The 1 Hz default is a budget choice, not a hard-coded limit."""
    tracking = _static_scene(_spread(11, 0), _spread(11, 3), n_frames=400)
    losses = build_losses([{"frame": 100, "losing_team": "home"}])
    params = CounterPressParams(timepoints=(0.0, 0.5, 1.0, 1.5, 2.0))
    scored = _score(tracking, losses, params)
    assert scored["n_timepoints"].iloc[0] == 5


# --------------------------------------------------------------------------- #
# The weighting contract
# --------------------------------------------------------------------------- #


def test_the_headline_and_the_alternate_are_never_the_same_column() -> None:
    """The bug this guards: two identical columns under two headings.

    ``mean_recovery`` is the headline weighting and ``mean_recovery_alt`` is the
    other one, so under the shipped ``"area"`` default the alternate must be the
    EPV share -- a different number -- rather than a second copy of the area
    share printed beside it.
    """
    tracking, scored = _two_team_scene()
    assert headline_weighting(scored) == DEFAULT_WEIGHTING

    for table in (per_team(scored), per_phase(scored)):
        assert "mean_recovery_area" not in table.columns
        assert table.attrs["weighting"] == "area"
        assert table.attrs["alt_weighting"] == "epv"
        rows = table[table["scored_losses"] > 0]
        assert len(rows) > 0
        assert not np.allclose(rows["mean_recovery"], rows["mean_recovery_alt"])

    players = per_player(scored, tracking, radius_m=200.0)
    credited = players[players["losses_nearby"] > 0]
    assert len(credited) > 0
    assert not np.allclose(credited["mean_recovery"], credited["mean_recovery_alt"])
    assert players.attrs["weighting"] == "area"


def test_switching_the_weighting_swaps_the_two_columns() -> None:
    """``with_weighting`` re-points the headline; the alternate follows it."""
    _, scored = _two_team_scene()
    area = per_team(scored).set_index("team")
    epv = per_team(with_weighting(scored, "epv")).set_index("team")

    assert epv.attrs["weighting"] == "epv"
    assert epv.attrs["alt_weighting"] == "area"
    assert np.allclose(epv["mean_recovery"], area["mean_recovery_alt"])
    assert np.allclose(epv["mean_recovery_alt"], area["mean_recovery"])


def test_the_weighting_is_readable_without_attrs() -> None:
    """Pandas drops ``attrs`` freely, so the columns are the fallback source."""
    _, scored = _two_team_scene()
    stripped = scored.copy()
    stripped.attrs.clear()
    assert headline_weighting(stripped) == "area"

    switched = with_weighting(scored, "epv").copy()
    switched.attrs.clear()
    assert headline_weighting(switched) == "epv"

    assert alternate_weighting("area") == "epv"
    assert alternate_weighting("epv") == "area"
    assert set(WEIGHTING_LABELS) == {"area", "epv"}


def test_the_summary_says_which_weighting_it_used() -> None:
    """A headline number a coach reads must carry its own definition."""
    _, scored = _two_team_scene()
    assert summarise_match(scored)["weighting"] == "area"
    assert summarise_match(with_weighting(scored, "epv"))["weighting"] == "epv"


# --------------------------------------------------------------------------- #
# Parallel scoring
# --------------------------------------------------------------------------- #

#: A grid coarse enough that this file can afford to score dozens of losses.
COARSE_GRID = make_grid(n_x=10)


def _moving_scene(n_frames: int) -> pd.DataFrame:
    """A scene in which every frame is a different, deterministic formation.

    Nobody stands still, so no two losses in it can score the same number by
    accident -- which is what makes comparing two runs of it worth anything.
    """
    return build_tracking(
        [
            {
                "frames": np.array([frame]),
                "period": 1,
                "home": _spread(11, 1000 + frame),
                "away": _spread(11, 5000 + frame),
                "ball": np.array([52.5, 34.0]),
            }
            for frame in range(1, n_frames + 1)
        ]
    )


def test_one_worker_and_many_agree_exactly() -> None:
    """``jobs`` is a wall-clock knob and nothing else.

    Every numeric column must be bit-identical between the serial path and the
    process pool, NaNs included. Scoring enough losses to cross
    ``_MIN_PARALLEL_LOSSES`` is the point: below it the pool is skipped and the
    comparison would be vacuous.
    """
    n = _MIN_PARALLEL_LOSSES
    tracking = _moving_scene(n * 5 + 200)
    losses = build_losses(
        [
            {"frame": 10 + 5 * i, "losing_team": "home" if i % 2 else "away"}
            for i in range(n)
        ]
    )
    serial = score_losses(
        tracking, empty_events(), losses=losses, grid=COARSE_GRID, jobs=1
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parallel = score_losses(
            tracking, empty_events(), losses=losses, grid=COARSE_GRID, jobs=2
        )
    # Without this the test would happily compare the serial path with itself.
    assert not [w for w in caught if "falling back" in str(w.message)]

    assert len(serial) == n
    assert bool(serial["scored"].all())
    # Not a scene where every loss happens to score the same thing.
    assert serial["recovery"].nunique() == n

    assert list(serial.columns) == list(parallel.columns)
    for column in serial.columns:
        left, right = serial[column], parallel[column]
        if pd.api.types.is_bool_dtype(left) or not pd.api.types.is_numeric_dtype(left):
            assert left.equals(right), column
        else:
            assert np.array_equal(
                left.to_numpy(dtype=float), right.to_numpy(dtype=float), equal_nan=True
            ), column


def test_a_handful_of_losses_never_starts_a_pool() -> None:
    """Spawning six processes to score two losses is a loss, so we do not."""
    tracking = _static_scene(_spread(11, 0), _spread(11, 3), n_frames=400)
    losses = build_losses([{"frame": 100, "losing_team": "home"}])
    scored = score_losses(
        tracking, empty_events(), losses=losses, grid=COARSE_GRID, jobs=MAX_USEFUL_JOBS
    )
    assert bool(scored["scored"].iloc[0])
    assert scored.attrs["jobs"] == MAX_USEFUL_JOBS


def test_default_jobs_is_capped_and_at_least_one() -> None:
    """The cap is measured: scaling is flat past six workers on this machine."""
    assert 1 <= default_jobs() <= MAX_USEFUL_JOBS


def test_the_matplotlib_cache_guard_stands_aside_when_it_should(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The package sets MPLCONFIGDIR at import; here is when it must not.

    This lives with the counter-press tests because the demo budget it protects
    is this module's budget: on a fresh clone the font-cache build is the
    single largest item in front of the first surface being computed.
    """
    monkeypatch.setenv("MPLCONFIGDIR", "/somewhere/the/user/chose")
    assert pressing.use_repo_matplotlib_cache() is None

    monkeypatch.delenv("MPLCONFIGDIR")
    monkeypatch.setitem(sys.modules, "matplotlib", object())
    assert pressing.use_repo_matplotlib_cache() is None
    assert "MPLCONFIGDIR" not in os.environ
    # Nothing to warm either: this process already has matplotlib loaded.
    assert pressing.warm_matplotlib_cache() is None


def test_the_matplotlib_cache_is_repo_local_and_gitignored() -> None:
    """``out/`` is already in .gitignore, and the cache is regenerable."""
    cache = pressing.matplotlib_cache_dir()
    assert cache is not None
    assert cache.name == ".mplcache"
    assert cache.parent.name == "out"
    assert (cache.parents[1] / "pyproject.toml").is_file()


def test_jobs_must_be_a_positive_count() -> None:
    """Zero or negative workers is a typo, not a request for the serial path."""
    tracking = _static_scene(_spread(11, 0), _spread(11, 3))
    losses = build_losses([{"frame": 10, "losing_team": "home"}])
    with pytest.raises(ValueError, match="jobs must be a positive integer"):
        score_losses(tracking, empty_events(), losses=losses, grid=GRID, jobs=0)
