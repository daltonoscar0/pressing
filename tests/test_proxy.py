"""Tests for the event-only proxy model.

Every test here builds its own StatsBomb-shaped input by hand, so the suite runs
on a fresh clone with no network and no ``data/raw/``. StatsBomb data is never
committed, so a test that needed a real events file would be a test that never
runs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pressing import proxy
from pressing.pitchcontrol import make_grid

# --------------------------------------------------------------------------- #
# Hand-built fixtures
# --------------------------------------------------------------------------- #

RED = "Red FC"
BLUE = "Blue United"


def event(
    index: int,
    period: int,
    timestamp: str,
    type_name: str,
    team: str,
    possession: int,
    possession_team: str,
    *,
    location: list[float] | None = None,
    play_pattern: str = "Regular Play",
    under_pressure: bool = False,
    counterpress: bool = False,
    duration: float | None = 0.5,
    uuid: str | None = None,
) -> dict[str, Any]:
    """Build one StatsBomb-shaped event dict.

    Args:
        index: Event ordering key.
        period: Match period.
        timestamp: PERIOD-RELATIVE ``HH:MM:SS.mmm`` timestamp, as StatsBomb
            stores it.
        type_name: ``type.name``.
        team: Team the event belongs to.
        possession: Possession number.
        possession_team: Team credited with the possession.
        location: ``[x, y]`` on the 120 x 80 pitch, or None.
        play_pattern: ``play_pattern.name``.
        under_pressure: The ``under_pressure`` flag.
        counterpress: The ``counterpress`` flag.
        duration: Event duration in seconds, or None.
        uuid: Event id. Defaults to ``e{index}``.

    Returns:
        A dict with the keys :func:`pressing.proxy.load_statsbomb_events` reads.
    """
    payload: dict[str, Any] = {
        "id": uuid or f"e{index}",
        "index": index,
        "period": period,
        "timestamp": timestamp,
        # Faithful to StatsBomb: `timestamp` above is period-RELATIVE while
        # `minute` here is match-ABSOLUTE, so period 2 restarts the one and not
        # the other. That mismatch is the trap the tests below pin.
        "minute": 45 * (period - 1) + int(timestamp.split(":")[1]),
        "second": int(float(timestamp.split(":")[2])),
        "type": {"id": 1, "name": type_name},
        "team": {"id": 1 if team == RED else 2, "name": team},
        "possession": possession,
        "possession_team": {
            "id": 1 if possession_team == RED else 2,
            "name": possession_team,
        },
        "play_pattern": {"id": 1, "name": play_pattern},
    }
    if location is not None:
        payload["location"] = location
    if duration is not None:
        payload["duration"] = duration
    if under_pressure:
        payload["under_pressure"] = True
    if counterpress:
        payload["counterpress"] = True
    return payload


@pytest.fixture
def turnover_events() -> list[dict[str, Any]]:
    """A four-possession match with exactly one open-play turnover.

    Possession 1 is Red building up and losing the ball; possession 2 is Blue
    winning it back at ``x = 30`` in Blue's own frame and playing three actions
    inside the five-second window. Possession 3 is Red again but restarts
    ``From Throw In``, so the loss ending possession 2 is a dead ball and must
    NOT be counted. Possession 4 sits in period 2 to exercise the period reset.

    Returns:
        Raw StatsBomb-shaped events in file order.
    """
    return [
        # -- possession 1: Red has the ball and gives it away ----------------
        event(1, 1, "00:00:00.000", "Pass", RED, 1, RED, location=[60.0, 40.0]),
        event(2, 1, "00:00:02.000", "Ball Receipt*", RED, 1, RED, location=[75.0, 40.0],
              under_pressure=True),
        # -- possession 2: Blue wins it at x=30 in Blue's frame --------------
        event(3, 1, "00:00:03.000", "Ball Recovery", BLUE, 2, BLUE, location=[30.0, 40.0]),
        event(4, 1, "00:00:04.000", "Pressure", RED, 2, BLUE, location=[89.0, 40.0],
              counterpress=True),
        event(5, 1, "00:00:05.500", "Pass", BLUE, 2, BLUE, location=[35.0, 30.0],
              under_pressure=True),
        event(6, 1, "00:00:07.000", "Ball Receipt*", BLUE, 2, BLUE, location=[50.0, 30.0]),
        # 10 s in: outside the window.
        event(7, 1, "00:00:13.000", "Pass", BLUE, 2, BLUE, location=[70.0, 30.0]),
        # -- possession 3: Red, but from a throw in -> dead ball -------------
        event(8, 1, "00:00:20.000", "Pass", RED, 3, RED, location=[40.0, 1.0],
              play_pattern="From Throw In"),
        event(9, 1, "00:45:00.000", "Half End", RED, 3, RED, location=None, duration=None),
        # -- possession 4: period 2, timestamp restarts at zero --------------
        event(10, 2, "00:00:00.000", "Pass", BLUE, 4, BLUE, location=[60.0, 40.0]),
        event(11, 2, "00:00:01.000", "Ball Receipt*", BLUE, 4, BLUE, location=[70.0, 40.0]),
        event(12, 2, "00:00:02.000", "Interception", RED, 5, RED, location=[45.0, 40.0]),
        event(13, 2, "00:00:03.000", "Pass", RED, 5, RED, location=[50.0, 45.0]),
    ]


@pytest.fixture
def events_frame(turnover_events: list[dict[str, Any]], tmp_path: Path) -> pd.DataFrame:
    """The hand-built events written to disk and read back through the loader.

    Args:
        turnover_events: Raw events.
        tmp_path: pytest temporary directory.

    Returns:
        The parsed events frame.
    """
    path = tmp_path / "99.json"
    path.write_text(json.dumps(turnover_events))
    return proxy.load_statsbomb_events(path)


def freeze(teammates: int, opponents: int, *, keeper: bool = True,
           uuid: str = "e3", area: list[float] | None = None) -> proxy.FreezeFrame:
    """Build a freeze-frame with players spread across the pitch.

    Args:
        teammates: How many players on the event's own team.
        opponents: How many on the other team.
        keeper: Whether the first opponent is flagged as a goalkeeper.
        uuid: Event id the frame belongs to.
        area: Flat ``[x1, y1, x2, y2, ...]`` visible polygon, or None.

    Returns:
        The frame.
    """
    n = teammates + opponents
    xs = np.linspace(20.0, 100.0, n)
    ys = np.linspace(10.0, 70.0, n)
    is_teammate = np.array([True] * teammates + [False] * opponents)
    keeps = np.zeros(n, bool)
    if keeper and opponents:
        keeps[teammates] = True
    polygon = (
        np.asarray(area, dtype=float).reshape(-1, 2)
        if area is not None
        else np.zeros((0, 2))
    )
    return proxy.FreezeFrame(
        event_uuid=uuid,
        positions=np.column_stack([xs, ys]),
        teammate=is_teammate,
        actor=np.zeros(n, bool),
        keeper=keeps,
        visible_area=polygon,
    )


# --------------------------------------------------------------------------- #
# The period-boundary time trap
# --------------------------------------------------------------------------- #


def test_absolute_seconds_does_not_jump_at_the_period_boundary(
    turnover_events: list[dict[str, Any]],
) -> None:
    """Period 2 must continue from the end of period 1, not restart or jump.

    StatsBomb's ``timestamp`` is period-relative and its ``minute`` is
    match-absolute. Mixing them produces a spurious gap of roughly 2758 s at
    every half boundary, silently. This pins the correct behaviour.
    """
    times = proxy.absolute_seconds(turnover_events)
    periods = np.array([e["period"] for e in turnover_events])

    # Period 1 ends at 45:00 = 2700 s, so period 2's first event is at 2700 s.
    assert times[periods == 1].max() == pytest.approx(2700.0)
    assert times[periods == 2].min() == pytest.approx(2700.0)

    # The gap ACROSS the boundary is zero, not the ~2758 s the bug produces.
    boundary = int(np.argmax(periods == 2))
    assert times[boundary] - times[boundary - 1] == pytest.approx(0.0)

    # And time never runs backwards anywhere.
    assert np.diff(times).min() >= 0.0

    # The trap is real in this fixture, so the test is not asserting something
    # that was never at risk: the raw period-relative timestamp DOES restart,
    # and adding the match-absolute `minute` to it jumps by 45 minutes.
    raw = np.array([proxy._parse_timestamp(e["timestamp"]) for e in turnover_events])
    assert raw[boundary] < raw[boundary - 1]

    mixed = raw + np.array([e["minute"] * 60.0 for e in turnover_events])
    assert mixed[boundary] - mixed[boundary - 1] == pytest.approx(-2700.0)


def test_absolute_seconds_is_monotonic_and_starts_at_zero(
    turnover_events: list[dict[str, Any]],
) -> None:
    """The first event is time zero and time never runs backwards."""
    times = proxy.absolute_seconds(turnover_events)
    assert times[0] == pytest.approx(0.0)
    assert np.all(np.diff(times) >= 0.0)


def test_absolute_seconds_handles_an_empty_file() -> None:
    """No events is an empty array, not a crash."""
    assert proxy.absolute_seconds([]).shape == (0,)


def test_parse_timestamp_rejects_a_malformed_string() -> None:
    """A timestamp that is not ``HH:MM:SS`` raises rather than parsing to junk."""
    with pytest.raises(ValueError, match="malformed"):
        proxy._parse_timestamp("12:34")


def test_five_second_window_never_spans_the_half(events_frame: pd.DataFrame) -> None:
    """A turnover late in period 1 must not pull period 2 events into its window.

    The last period-1 event sits at 2700 s and period 2's first is also at
    2700 s, so a window that ignored ``period`` would splice the halves.
    """
    features = proxy.build_features(events_frame)
    period_two = features[features["period"] == 2]
    assert len(period_two) == 1
    # Blue lose it to Red's interception; only Red's two period-2 events are in
    # the window, never anything from period 1.
    assert period_two["n_events_5s"].iloc[0] == 2


# --------------------------------------------------------------------------- #
# Feature extraction against a known answer
# --------------------------------------------------------------------------- #


def test_turnover_detection_finds_only_open_play_losses(events_frame: pd.DataFrame) -> None:
    """Possession-team changes count; dead-ball restarts do not.

    The fixture has three possession-team changes: Red -> Blue (open play),
    Blue -> Red (``From Throw In``, dead ball) and Blue -> Red in period 2
    (open play). Only two survive.
    """
    turnovers = proxy.find_turnovers(events_frame)
    assert len(turnovers) == 2
    assert turnovers["losing_team"].tolist() == [RED, BLUE]
    assert turnovers["winning_team"].tolist() == [BLUE, RED]
    assert turnovers["period"].tolist() == [1, 2]


def test_turnover_detection_does_not_use_a_type_whitelist(
    events_frame: pd.DataFrame,
) -> None:
    """The losing event is a ``Ball Receipt*``, which no whitelist would catch.

    Recon measured that the last on-ball event of a possession is
    ``Ball Receipt*`` 65 times out of 196 in a real match, so a whitelist
    misclassifies a third of losses.
    """
    turnovers = proxy.find_turnovers(events_frame)
    losing_event = events_frame.set_index("index").loc[turnovers["loss_index"].iloc[0]]
    assert losing_event["type"] == "Ball Receipt*"


def test_turnover_location_is_in_the_losing_teams_frame(
    events_frame: pd.DataFrame,
) -> None:
    """Blue recover at x=30 in Blue's frame, so Red lost it at x=90 in Red's.

    StatsBomb re-orients per event so the event's own team attacks ``+x``. The
    turnover location must be expressed from the LOSING team's own goal, which
    is what the phase split means by ``x``. Getting this wrong is silent: the
    number still lands on the pitch, just at the other end.
    """
    turnovers = proxy.find_turnovers(events_frame)
    first = turnovers.iloc[0]
    assert first["sb_x"] == pytest.approx(120.0 - 30.0)
    assert first["x"] == pytest.approx(90.0 * 105.0 / 120.0)
    # y is mirrored too, then flipped for the bottom-left origin.
    assert first["sb_y"] == pytest.approx(80.0 - 40.0)
    assert first["y"] == pytest.approx(40.0 * 68.0 / 80.0)
    assert first["phase"] == "final_third"


def test_features_match_a_hand_counted_answer(events_frame: pd.DataFrame) -> None:
    """Every spec-named feature on the first turnover, counted by hand.

    Window is ``[3.0, 8.0]`` s and holds events 3, 4, 5 and 6. Event 7 is at
    13 s and is out. The next three actions after the turnover event are 4, 5
    and 6.
    """
    features = proxy.build_features(events_frame)
    row = features[features["period"] == 1].iloc[0]

    assert row["t"] == pytest.approx(3.0)
    assert row["time_to_next_event"] == pytest.approx(1.0)
    assert row["n_events_5s"] == 4
    assert row["n_loser_events_5s"] == 1  # only Red's Pressure at 4 s
    assert row["n_winner_events_5s"] == 3

    assert row["next_1_type"] == "Pressure"
    assert row["next_2_type"] == "Pass"
    assert row["next_3_type"] == "Ball Receipt*"
    assert row["next_1_by_loser"] == 1.0
    assert row["next_2_by_loser"] == 0.0
    assert row["next_3_by_loser"] == 0.0

    # Pressure flags: events 2 (outside), 5 in-window under_pressure; event 4
    # in-window counterpress.
    assert row["under_pressure_count"] == 1
    assert row["counterpress_count"] == 1
    assert row["pressure_event_count"] == 1
    assert row["loss_event_type"] == "Ball Receipt*"
    assert row["loss_under_pressure"] == 1.0

    assert row["next_play_pattern"] == "Regular Play"
    assert row["regained_within_5s"] == 0.0
    assert row["losing_possession_events"] == 2
    assert row["losing_possession_duration"] == pytest.approx(2.0)


def test_ball_progress_is_measured_in_one_orientation(
    events_frame: pd.DataFrame,
) -> None:
    """Ball movement uses the winner's frame throughout, mirroring where needed.

    The window's first located event is Blue's recovery at ``x = 30`` and its
    last is Blue's receipt at ``x = 50``, both already in Blue's frame, so the
    ball made 20 StatsBomb units of forward progress for Blue. Red's Pressure
    event at ``x = 89`` in RED's frame sits between them and must be mirrored to
    ``x = 31`` rather than read as a 59-unit jump.
    """
    features = proxy.build_features(events_frame)
    row = features[features["period"] == 1].iloc[0]
    assert row["ball_forward_5s"] == pytest.approx(20.0 * 105.0 / 120.0)
    assert row["ball_distance_5s"] == pytest.approx(
        float(np.hypot(20.0, 10.0)) * 105.0 / 120.0
    )


def test_features_never_read_a_freeze_frame() -> None:
    """The proxy must run on a match with events and no 360 at all.

    A feature derived from freeze-frame positions would make the model
    unusable on exactly the matches it exists for, so the source of
    :func:`~pressing.proxy.build_features` is checked for any reference to the
    freeze-frame machinery.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(proxy.build_features).strip())
    body = tree.body[0].body  # type: ignore[attr-defined]
    if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]  # drop the docstring, which legitimately says "freeze"
    code = "\n".join(ast.unparse(node) for node in body)
    for banned in ("freeze", "FreezeFrame", "visible_area", "teammate", "keeper"):
        assert banned not in code, f"build_features refers to {banned!r}"

    # And the signature takes no freeze-frame argument at all.
    assert set(inspect.signature(proxy.build_features).parameters) == {
        "events",
        "turnovers",
        "window_s",
    }


def test_build_features_on_an_empty_match_returns_the_right_columns() -> None:
    """No turnovers gives an empty frame with the full schema, not a crash."""
    empty = pd.DataFrame(columns=list(proxy.EVENT_FIELDS))
    out = proxy.build_features(empty)
    assert out.empty
    assert list(out.columns) == list(proxy.FEATURE_META_COLUMNS) + list(proxy.FEATURE_COLUMNS)


# --------------------------------------------------------------------------- #
# Coordinates and geometry
# --------------------------------------------------------------------------- #


def test_sb_to_canonical_maps_the_corners() -> None:
    """StatsBomb's top-left origin becomes the canonical bottom-left one."""
    corners = np.array([[0.0, 0.0], [120.0, 80.0], [60.0, 40.0]])
    out = proxy.sb_to_canonical(corners)
    assert out[0] == pytest.approx([0.0, 68.0])
    assert out[1] == pytest.approx([105.0, 0.0])
    assert out[2] == pytest.approx([52.5, 34.0])


def test_orient_to_mirrors_only_the_other_team() -> None:
    """A frame already in the target team's orientation is left alone."""
    point = np.array([30.0, 20.0])
    assert proxy.orient_to(point, RED, RED) == pytest.approx(point)
    assert proxy.orient_to(point, BLUE, RED) == pytest.approx([90.0, 60.0])


def test_points_in_polygon_matches_a_known_rectangle() -> None:
    """Ray casting against a rectangle with points inside, outside and beyond."""
    rect = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]])
    pts = np.array([[5.0, 2.5], [-1.0, 2.5], [11.0, 2.5], [5.0, 6.0], [1.0, 1.0]])
    assert proxy.points_in_polygon(pts, rect).tolist() == [True, False, False, False, True]


def test_points_in_polygon_with_no_polygon_keeps_everything() -> None:
    """A degenerate polygon means "no truncation", not "nothing visible"."""
    pts = np.array([[1.0, 1.0], [50.0, 50.0]])
    assert proxy.points_in_polygon(pts, np.zeros((0, 2))).all()
    assert proxy.points_in_polygon(pts, np.array([[0.0, 0.0], [1.0, 1.0]])).all()


def test_points_in_polygon_area_matches_the_shoelace_area() -> None:
    """The mask a label uses must integrate to the polygon's real area.

    A rotation-flipped or transposed mask still produces a plausible-looking
    share, so the mask is checked against an independent area computation.
    """
    grid = make_grid(n_x=50)
    poly = proxy.sb_to_canonical(
        np.array([[16.0, 80.0], [48.0, 1.0], [74.0, 1.0], [104.0, 80.0]])
    )
    mask_fraction = proxy.points_in_polygon(grid.targets, poly).mean()
    x, y = poly[:, 0], poly[:, 1]
    shoelace = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    assert mask_fraction == pytest.approx(shoelace / (105.0 * 68.0), abs=0.01)


# --------------------------------------------------------------------------- #
# Freeze-frame label
# --------------------------------------------------------------------------- #


def test_freeze_frame_players_resolves_teams_through_the_event_team() -> None:
    """``teammate`` is relative to the event, so the mapping must be inverted.

    A Pressure event by the counter-pressing team has ``teammate=True`` on the
    PRESSERS. Reading it as the winner's players would swap the two teams and
    return one minus the right answer.
    """
    frame = freeze(teammates=4, opponents=6, keeper=False)

    # Event belongs to the winner: teammates are the winner.
    winner, presser = proxy.freeze_frame_players(frame, BLUE, BLUE, RED)
    assert len(winner) == 4
    assert len(presser) == 6

    # Same frame, event belongs to the loser: the mapping flips.
    winner, presser = proxy.freeze_frame_players(frame, RED, BLUE, RED)
    assert len(winner) == 6
    assert len(presser) == 4


def test_freeze_frame_players_mirrors_a_losing_teams_frame() -> None:
    """A frame from the loser's event is rotated so the winner attacks +x."""
    frame = proxy.FreezeFrame(
        event_uuid="x",
        positions=np.array([[30.0, 20.0], [90.0, 60.0]]),
        teammate=np.array([True, False]),
        actor=np.zeros(2, bool),
        keeper=np.zeros(2, bool),
        visible_area=np.zeros((0, 2)),
    )
    winner, _ = proxy.freeze_frame_players(frame, BLUE, BLUE, RED)
    mirrored, _ = proxy.freeze_frame_players(frame, RED, BLUE, RED)
    # Winner's single player, seen from each side, must land on the same spot.
    assert winner.positions[0] == pytest.approx(
        proxy.sb_to_canonical(np.array([30.0, 20.0]))
    )
    assert mirrored.positions[0] == pytest.approx(
        proxy.sb_to_canonical(np.array([120.0 - 90.0, 80.0 - 60.0]))
    )


def test_freeze_frame_players_gives_every_player_zero_velocity() -> None:
    """A freeze-frame carries no velocity, so none may be invented."""
    winner, presser = proxy.freeze_frame_players(freeze(5, 5), BLUE, BLUE, RED)
    assert np.all(winner.velocities == 0.0)
    assert np.all(presser.velocities == 0.0)


def test_freeze_frame_players_rejects_an_unrelated_event_team() -> None:
    """An event belonging to neither side of the turnover is a bug, not a skip."""
    with pytest.raises(ValueError, match="neither"):
        proxy.freeze_frame_players(freeze(5, 5), "Green Town", BLUE, RED)


def test_two_mirror_image_teams_split_control_evenly() -> None:
    """A symmetric frame must score exactly 0.5, whatever the orientation.

    This is the label's dead-even reference point. If the team mapping or the
    mirroring were wrong the number would drift off 0.5.
    """
    positions = np.array([[40.0, 30.0], [50.0, 50.0], [60.0, 30.0],
                          [80.0, 50.0], [70.0, 30.0], [60.0, 50.0]])
    mirrored = np.column_stack([120.0 - positions[:, 0], 80.0 - positions[:, 1]])
    frame = proxy.FreezeFrame(
        event_uuid="sym",
        positions=np.vstack([positions, mirrored]),
        teammate=np.array([True] * 6 + [False] * 6),
        actor=np.zeros(12, bool),
        keeper=np.zeros(12, bool),
        visible_area=np.zeros((0, 2)),
    )
    winner, presser = proxy.freeze_frame_players(frame, BLUE, BLUE, RED)
    from pressing.pitchcontrol import pitch_control_full

    result = pitch_control_full(
        winner, presser, ball_pos=np.array([52.5, 34.0]), grid=make_grid(n_x=50)
    )
    share = proxy._share_from_surfaces(result.attacking, result.defending)
    assert share == pytest.approx(0.5, abs=1e-9)


def test_label_needs_at_least_two_frames(events_frame: pd.DataFrame) -> None:
    """A one-frame window is unlabelled, not scored off the loss instant alone.

    Averaging one instant would measure the geometry at the turnover and say
    nothing at all about the five seconds that follow.
    """
    frames = {"e3": freeze(6, 6, uuid="e3")}
    labelled = proxy.label_turnovers(events_frame, frames)
    first = labelled[labelled["period"] == 1].iloc[0]
    assert not bool(first["labelled"])
    assert first["skip_reason"] == "too_few_frames"
    assert np.isnan(first["recovery_360"])


def test_label_with_no_360_at_all_is_reported_not_guessed(
    events_frame: pd.DataFrame,
) -> None:
    """An events file with no matching freeze-frames yields null labels.

    This is the graceful-degradation case: 360 coverage is per-event and per
    competition, so the pipeline has to survive a match with none.
    """
    labelled = proxy.label_turnovers(events_frame, {})
    assert len(labelled) == 2
    assert not labelled["labelled"].any()
    assert set(labelled["skip_reason"]) == {"no_360"}
    assert labelled["recovery_360"].isna().all()


def test_label_averages_the_window_and_stays_in_range(
    events_frame: pd.DataFrame,
) -> None:
    """Three frames in the window give a labelled row inside ``[0, 1]``."""
    frames = {
        "e3": freeze(6, 5, uuid="e3"),
        "e5": freeze(6, 5, uuid="e5"),
        "e6": freeze(6, 5, uuid="e6"),
    }
    labelled = proxy.label_turnovers(events_frame, frames)
    first = labelled[labelled["period"] == 1].iloc[0]
    assert bool(first["labelled"])
    assert first["n_frames"] == 3
    assert 0.0 <= first["recovery_360"] <= 1.0
    assert 0.0 <= first["recovery_360_visible"] <= 1.0
    assert first["frame_span_s"] == pytest.approx(4.0)


def test_label_skips_a_frame_with_too_few_visible_players(
    events_frame: pd.DataFrame,
) -> None:
    """Pitch control against one visible defender is not a defence."""
    frames = {
        "e3": freeze(6, 1, uuid="e3", keeper=False),
        "e5": freeze(6, 1, uuid="e5", keeper=False),
        "e6": freeze(6, 1, uuid="e6", keeper=False),
    }
    labelled = proxy.label_turnovers(events_frame, frames)
    first = labelled[labelled["period"] == 1].iloc[0]
    assert not bool(first["labelled"])
    assert first["n_frames"] == 0


def test_label_params_rejects_a_velocity_model_it_cannot_honour() -> None:
    """Only zero velocity is defensible from a freeze-frame."""
    with pytest.raises(ValueError, match="no velocity information"):
        proxy.LabelParams(velocity="constant")
    with pytest.raises(ValueError, match="weighting"):
        proxy.LabelParams(weighting="epv")
    with pytest.raises(ValueError, match="min_frames"):
        proxy.LabelParams(min_frames=0)


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #


@pytest.fixture
def toy_dataset() -> pd.DataFrame:
    """A synthetic dataset with a genuine signal, ten matches deep.

    The label is a smooth function of ``x`` plus a smaller contribution from
    the pressure count plus noise, so a working model must beat the mean and
    the x-only baseline must land in between. Match ids are dense so a split
    that accidentally worked row-wise would still show up.

    Returns:
        A frame carrying every feature column plus ``recovery_360``.
    """
    rng = np.random.default_rng(7)
    n_per_match, n_matches = 60, 10
    rows: list[dict[str, Any]] = []
    types = ["Pass", "Pressure", "Ball Receipt*", "Duel", "Carry"]
    for match in range(n_matches):
        for i in range(n_per_match):
            x = float(rng.uniform(0.0, 105.0))
            pressure = int(rng.integers(0, 5))
            row: dict[str, Any] = {
                "match_id": 1000 + match,
                "turnover_id": f"{1000 + match}:{i}",
                "t": float(i * 30),
                "losing_team": "A",
                "winning_team": "B",
                "phase": proxy._phase_of(x),
                "recovery_360": float(
                    np.clip(0.25 + 0.004 * x + 0.02 * pressure + rng.normal(0, 0.05), 0, 1)
                ),
            }
            for name in proxy.NUMERIC_FEATURES:
                row[name] = float(rng.normal(0, 1))
            for name in proxy.CATEGORICAL_FEATURES:
                row[name] = str(rng.choice(types))
            row["x"] = x
            row["counterpress_count"] = float(pressure)
            rows.append(row)
    return pd.DataFrame(rows)


def test_split_by_match_holds_out_whole_matches(toy_dataset: pd.DataFrame) -> None:
    """No match id may appear on both sides of the split.

    Splitting by turnover would put two losses from the same match five seconds
    apart on opposite sides, which the spec calls disqualifying.
    """
    train_mask, test_mask = proxy.split_by_match(toy_dataset, test_fraction=0.3, seed=1)
    assert (train_mask | test_mask).all()
    assert not (train_mask & test_mask).any()

    train_matches = set(toy_dataset.loc[train_mask, "match_id"])
    test_matches = set(toy_dataset.loc[test_mask, "match_id"])
    assert train_matches & test_matches == set()
    assert train_matches | test_matches == set(toy_dataset["match_id"])
    assert len(test_matches) == 3


def test_split_by_match_is_reproducible_and_seed_sensitive(
    toy_dataset: pd.DataFrame,
) -> None:
    """The same seed gives the same split; a different seed usually does not."""
    a = proxy.split_by_match(toy_dataset, seed=1)[1]
    b = proxy.split_by_match(toy_dataset, seed=1)[1]
    c = proxy.split_by_match(toy_dataset, seed=2)[1]
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_split_by_match_rejects_impossible_requests(toy_dataset: pd.DataFrame) -> None:
    """A fraction outside (0, 1) or a single match cannot produce a split."""
    with pytest.raises(ValueError, match="test_fraction"):
        proxy.split_by_match(toy_dataset, test_fraction=0.0)
    with pytest.raises(ValueError, match="at least 2 matches"):
        proxy.split_by_match(toy_dataset[toy_dataset["match_id"] == 1000])


def test_split_always_leaves_at_least_one_match_on_each_side() -> None:
    """A tiny request still holds something out rather than emptying a side."""
    frame = pd.DataFrame({"match_id": [1, 1, 2, 2, 3, 3]})
    train_mask, test_mask = proxy.split_by_match(frame, test_fraction=0.01)
    assert test_mask.sum() > 0
    assert train_mask.sum() > 0


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


def test_train_refuses_to_take_the_match_id_as_a_feature(
    toy_dataset: pd.DataFrame,
) -> None:
    """A per-match offset cannot exist at prediction time and would leak."""
    with pytest.raises(ValueError, match="must not be a feature"):
        proxy.train(
            toy_dataset,
            toy_dataset["recovery_360"],
            feature_names=("x", "match_id"),
            categorical_names=(),
        )


def test_train_refuses_unlabelled_rows(toy_dataset: pd.DataFrame) -> None:
    """Missing targets must be dropped by the caller, not silently imputed."""
    targets = toy_dataset["recovery_360"].copy()
    targets.iloc[3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        proxy.train(toy_dataset, targets)


def test_predictions_have_the_right_shape_and_range(toy_dataset: pd.DataFrame) -> None:
    """One prediction per row, every one a share inside ``[0, 1]``."""
    train_mask, test_mask = proxy.split_by_match(toy_dataset, seed=3)
    model = proxy.train(
        toy_dataset[train_mask], toy_dataset.loc[train_mask, "recovery_360"]
    )
    predictions = model.predict(toy_dataset[test_mask])
    assert predictions.shape == (int(test_mask.sum()),)
    assert np.isfinite(predictions).all()
    assert predictions.min() >= 0.0
    assert predictions.max() <= 1.0


def test_the_model_recovers_a_signal_it_was_given(toy_dataset: pd.DataFrame) -> None:
    """On data with a known structure the held-out r must be clearly positive.

    Not a claim about real football -- it is the check that the training,
    encoding and prediction path is wired up at all. A model that returned
    noise would pass every shape test above and fail this one.
    """
    train_mask, test_mask = proxy.split_by_match(toy_dataset, seed=5)
    model = proxy.train(
        toy_dataset[train_mask], toy_dataset.loc[train_mask, "recovery_360"]
    )
    metrics = proxy.evaluate(
        model, toy_dataset[test_mask], toy_dataset.loc[test_mask, "recovery_360"]
    )
    assert metrics.r > 0.8
    assert metrics.n_matches == 3
    assert metrics.n_losses == int(test_mask.sum())


def test_unseen_categories_encode_to_minus_one(toy_dataset: pd.DataFrame) -> None:
    """A category absent from training must not shift every other code.

    Re-deriving the vocabulary at prediction time is the bug this guards: it
    produces plausible predictions rather than an error.
    """
    model = proxy.train(toy_dataset, toy_dataset["recovery_360"])
    unseen = toy_dataset.head(3).copy()
    unseen["next_1_type"] = "Bicycle Kick That Does Not Exist"
    encoded = model.encode(unseen)
    column = model.feature_names.index("next_1_type")
    assert (encoded[:, column] == -1).all()
    assert np.isfinite(model.predict(unseen)).all()


def test_encode_reports_a_missing_column_by_name(toy_dataset: pd.DataFrame) -> None:
    """A dropped feature raises with the column named, not a shape error."""
    model = proxy.train(toy_dataset, toy_dataset["recovery_360"])
    with pytest.raises(KeyError, match="ball_forward_5s"):
        model.encode(toy_dataset.drop(columns=["ball_forward_5s"]))


def test_score_predictions_reports_nan_for_a_constant_baseline(
    toy_dataset: pd.DataFrame,
) -> None:
    """A mean predictor has no correlation; reporting 0.0 would invent a number."""
    constant = np.full(len(toy_dataset), 0.5)
    metrics = proxy.score_predictions(
        constant, toy_dataset["recovery_360"], toy_dataset
    )
    assert np.isnan(metrics.r)
    assert np.isnan(metrics.spearman)
    assert metrics.mae > 0.0


def test_compare_models_reports_every_baseline(toy_dataset: pd.DataFrame) -> None:
    """The comparison table carries the mean, phase, x-only and full models."""
    train_mask, test_mask = proxy.split_by_match(toy_dataset, seed=11)
    model = proxy.train(
        toy_dataset[train_mask], toy_dataset.loc[train_mask, "recovery_360"]
    )
    table = proxy.compare_models(
        toy_dataset, toy_dataset["recovery_360"], train_mask, test_mask, model
    )
    assert table["model"].tolist() == ["mean", "phase", "turnover_x", "full"]
    assert np.isnan(table.loc[table["model"] == "mean", "r"].iloc[0])
    # The toy label is mostly x, so x alone must already be strong.
    assert table.loc[table["model"] == "turnover_x", "r"].iloc[0] > 0.7


def test_model_round_trips_through_the_artefact(
    toy_dataset: pd.DataFrame, tmp_path: Path
) -> None:
    """Saving and loading must give bit-identical predictions."""
    model = proxy.train(toy_dataset, toy_dataset["recovery_360"], metadata={"note": "toy"})
    path = proxy.save_model(model, tmp_path / "toy.pkl.gz")
    assert path.exists()

    reloaded = proxy.load_bundled_model(path)
    assert reloaded.feature_names == model.feature_names
    assert reloaded.vocabularies == model.vocabularies
    assert reloaded.metadata["note"] == "toy"
    assert np.array_equal(reloaded.predict(toy_dataset), model.predict(toy_dataset))


def test_loading_a_missing_artefact_says_how_to_make_one(tmp_path: Path) -> None:
    """The error points at the training command, not just at a path."""
    with pytest.raises(FileNotFoundError, match="MODEL_CARD"):
        proxy.load_bundled_model(tmp_path / "absent.pkl.gz")


def test_an_unknown_model_variant_is_rejected_by_name() -> None:
    """A typo must not silently resolve to the deployable model."""
    with pytest.raises(ValueError, match="unknown model variant"):
        proxy.bundled_model_path("held_out")


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def test_the_dataset_row_order_does_not_depend_on_completion_order() -> None:
    """Row order is a correctness property here, not a tidiness one.

    ``HistGradientBoostingRegressor`` draws its early-stopping validation slice
    out of the design matrix by position, so two frames with identical content in
    a different row order fit to different trees. The v0.1.0 parallel dataset
    build appended each match as its worker finished, which made the training set
    a race: measured on the 6,195 real labelled rows, three fresh processes on the
    canonical order all gave held-out ``r`` = 0.369162 stopping at iteration 79
    with byte-identical predictions, while a row shuffle alone gave 0.369387,
    0.374414 and 0.368491 stopping at 85, 69 and 85. Assembly must therefore be
    order-blind.
    """
    parts = [
        pd.DataFrame({"match_id": [7, 7, 7], "loss_index": [0, 1, 2]}),
        pd.DataFrame({"match_id": [3, 3], "loss_index": [0, 1]}),
        pd.DataFrame({"match_id": [5], "loss_index": [0]}),
    ]
    canonical = proxy._canonical_row_order(parts)
    for order in ([1, 0, 2], [2, 1, 0], [0, 2, 1]):
        shuffled = proxy._canonical_row_order([parts[i] for i in order])
        pd.testing.assert_frame_equal(canonical, shuffled)
    assert canonical["match_id"].tolist() == [3, 3, 5, 7, 7, 7]
    # Within a match the natural order survives; the sort is stable.
    assert canonical["loss_index"].tolist() == [0, 1, 0, 0, 1, 2]


def test_an_empty_assembly_gives_an_empty_frame_not_an_exception() -> None:
    """A cache with nothing in it is a legitimate, reportable state."""
    assert proxy._canonical_row_order([]).empty


def test_refitting_the_same_rows_twice_gives_identical_predictions(
    toy_dataset: pd.DataFrame,
) -> None:
    """With the row order fixed, the fit is reproducible."""
    first = proxy.train(toy_dataset, toy_dataset["recovery_360"])
    second = proxy.train(toy_dataset, toy_dataset["recovery_360"])
    assert np.array_equal(first.predict(toy_dataset), second.predict(toy_dataset))
    assert first.estimator.n_iter_ == second.estimator.n_iter_


def test_saving_the_same_model_twice_gives_byte_identical_files(
    toy_dataset: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deterministic fit is not a reproducible file, and this is the gap.

    Up to v0.1.1 ``save_model`` went through ``gzip.open``, which writes
    ``time.time()`` into gzip header bytes 4:8. Two writes of one unchanged model
    a second apart therefore differed at byte offset 4 and nowhere else, which is
    how the committed artefacts kept changing under a fit that had already been
    made deterministic. The shipped split artefact carried mtime 1787773230.
    """
    model = proxy.train(toy_dataset, toy_dataset["recovery_360"], metadata={"note": "toy"})
    first = (tmp_path / "a") / "toy.pkl.gz"
    second = (tmp_path / "b") / "toy.pkl.gz"
    # Move the clock a year between the two writes rather than sleeping. Under
    # the old code path this alone made the files differ.
    monkeypatch.setattr(time, "time", lambda: 1_700_000_000.0)
    proxy.save_model(model, first)
    monkeypatch.setattr(time, "time", lambda: 1_800_000_000.0)
    proxy.save_model(model, second)
    assert first.read_bytes() == second.read_bytes()


def test_the_artefact_header_carries_no_timestamp(
    toy_dataset: pd.DataFrame, tmp_path: Path
) -> None:
    """Header bytes 4:8 must read zero, the value gzip reserves for "unknown"."""
    model = proxy.train(toy_dataset, toy_dataset["recovery_360"])
    header = proxy.save_model(model, tmp_path / "toy.pkl.gz").read_bytes()[:10]
    assert header[:2] == b"\x1f\x8b"
    assert int.from_bytes(header[4:8], "little") == proxy.ARTEFACT_GZIP_MTIME == 0


def test_the_shipped_artefacts_carry_no_timestamp_of_any_kind() -> None:
    """No committed artefact may record a clock, in its header or its metadata.

    A wall-clock field inside the pickle is worse than one in the header: it is
    compressed, so it changes the file's length as well as its content. The same
    model saved with three different ``trained_utc`` strings came out at 124,366,
    124,366 and 124,356 bytes.
    """
    for variant in ("refit", "split"):
        path = proxy.bundled_model_path(variant)
        if not path.exists():  # pragma: no cover - artefact-free checkout
            pytest.skip("no bundled proxy models in this checkout")
        assert int.from_bytes(path.read_bytes()[4:8], "little") == 0, variant
        metadata = proxy.load_bundled_model(variant=variant).metadata
        assert "trained_utc" not in metadata, variant
        assert not [
            key for key in metadata if "utc" in key.lower() or "timestamp" in key.lower()
        ], variant


# --------------------------------------------------------------------------- #
# Honest scoring
# --------------------------------------------------------------------------- #


def _with_split_metadata(
    model: proxy.ProxyModel, frame: pd.DataFrame, train_mask: np.ndarray, fitted_on: str
) -> proxy.ProxyModel:
    """Stamp a model with the split metadata a real training run records.

    Args:
        model: The fitted model to stamp.
        frame: The dataset it came from.
        train_mask: Boolean training rows.
        fitted_on: ``"train_matches"`` or ``"all_labelled"``.

    Returns:
        The same model, mutated.
    """
    model.metadata = {
        "fitted_on": fitted_on,
        "train_match_ids": sorted(int(v) for v in frame.loc[train_mask, "match_id"].unique()),
        "held_out_match_ids": sorted(
            int(v) for v in frame.loc[~train_mask, "match_id"].unique()
        ),
    }
    return model


def test_score_held_out_refuses_a_model_that_saw_the_held_out_rows(
    toy_dataset: pd.DataFrame,
) -> None:
    """The reproduction trap this exists to close.

    The deployable artefact is refitted on every labelled row, so feeding it the
    documented held-out match ids and reading the correlation is in-sample. On
    the real dataset that returns r = 0.6151 against an honest 0.3692, with
    nothing to signal the mistake. Documenting it is not enough; it must raise.
    """
    train_mask, _ = proxy.split_by_match(toy_dataset, seed=3)
    refit = _with_split_metadata(
        proxy.train(toy_dataset, toy_dataset["recovery_360"]),
        toy_dataset, train_mask, "all_labelled",
    )
    with pytest.raises(ValueError, match="in-sample"):
        proxy.score_held_out(refit, toy_dataset, toy_dataset["recovery_360"])


def test_score_held_out_scores_the_split_model_on_its_own_held_out_matches(
    toy_dataset: pd.DataFrame,
) -> None:
    """The honest path returns the same metrics as scoring the test rows by hand."""
    train_mask, test_mask = proxy.split_by_match(toy_dataset, seed=3)
    split = _with_split_metadata(
        proxy.train(toy_dataset[train_mask], toy_dataset.loc[train_mask, "recovery_360"]),
        toy_dataset, train_mask, "train_matches",
    )
    honest = proxy.score_held_out(split, toy_dataset, toy_dataset["recovery_360"])
    by_hand = proxy.evaluate(
        split, toy_dataset[test_mask], toy_dataset.loc[test_mask, "recovery_360"]
    )
    assert honest == by_hand
    assert honest.n_losses == int(test_mask.sum())
    assert honest.n_matches == len(proxy.held_out_match_ids(split))


def test_score_held_out_reports_a_model_that_cannot_say_what_it_saw(
    toy_dataset: pd.DataFrame,
) -> None:
    """An artefact from before the split metadata must not be scored on a guess."""
    model = proxy.train(toy_dataset, toy_dataset["recovery_360"])
    with pytest.raises(KeyError, match="held_out_match_ids"):
        proxy.score_held_out(model, toy_dataset, toy_dataset["recovery_360"])
    with pytest.raises(KeyError, match="train_match_ids"):
        proxy.training_match_ids(model)


def test_score_held_out_refuses_a_frame_with_none_of_its_matches(
    toy_dataset: pd.DataFrame,
) -> None:
    """Scoring the wrong competition must fail loudly, not return one row."""
    train_mask, _ = proxy.split_by_match(toy_dataset, seed=3)
    split = _with_split_metadata(
        proxy.train(toy_dataset[train_mask], toy_dataset.loc[train_mask, "recovery_360"]),
        toy_dataset, train_mask, "train_matches",
    )
    elsewhere = toy_dataset.assign(match_id=toy_dataset["match_id"] + 500_000)
    with pytest.raises(ValueError, match="held-out matches appear"):
        proxy.score_held_out(split, elsewhere, elsewhere["recovery_360"])


def test_the_shipped_artefacts_record_which_matches_each_one_saw() -> None:
    """Both committed artefacts must know their own split, and disagree correctly."""
    if not proxy.bundled_model_path("split").exists():  # pragma: no cover
        pytest.skip("no bundled proxy models in this checkout")

    refit = proxy.load_bundled_model(variant="refit")
    split = proxy.load_bundled_model(variant="split")

    assert refit.metadata["fitted_on"] == "all_labelled"
    assert split.metadata["fitted_on"] == "train_matches"
    train_ids = set(proxy.training_match_ids(split))
    held_out = set(proxy.held_out_match_ids(split))
    assert train_ids and held_out
    assert train_ids.isdisjoint(held_out)
    assert train_ids | held_out == set(proxy.training_match_ids(refit)) | set(
        proxy.held_out_match_ids(refit)
    )
    assert len(train_ids) + len(held_out) == refit.metadata["n_matches"]


# --------------------------------------------------------------------------- #
# The phase caveat
# --------------------------------------------------------------------------- #


def test_the_phase_caveat_quotes_the_models_own_means(toy_dataset: pd.DataFrame) -> None:
    """A model's recorded phase means win over the package default."""
    model = proxy.train(
        toy_dataset,
        toy_dataset["recovery_360"],
        metadata={"predicted_phase_means": {
            "build_up": 0.111, "middle": 0.222, "final_third": 0.333,
        }},
    )
    assert proxy.phase_reference_means(model) == {
        "build_up": 0.111, "middle": 0.222, "final_third": 0.333,
    }
    caveat = " ".join(proxy.recovery_phase_caveat(model))
    assert "build-up 0.111" in caveat and "final third 0.333" in caveat
    assert "+0.222" in caveat  # this model's own gradient, 0.333 - 0.111


def test_the_phase_caveat_never_offers_the_tracking_scale_as_the_proxys() -> None:
    """The v0.1.0 header printed 0.26 / 0.47 / 0.70 above proxy numbers.

    Those are the tracking metric's phase means. The proxy's own gradient across
    pitch thirds is about a sixth of the tracking metric's, so a reader who
    scaled against them read the proxy's own average as a remarkable press. The
    tracking figures may still be named, but only inside the sentence that tells
    you not to scale against them.
    """
    scale_line, warning_line = proxy.recovery_phase_caveat()
    for value in (f"{v:.2f}" for v in proxy.TRACKING_PHASE_MEANS.values()):
        assert value not in scale_line
    assert "do NOT scale" in warning_line
    assert "0.26" in warning_line and "0.70" in warning_line
    means = proxy.phase_reference_means()
    assert means["final_third"] - means["build_up"] < 0.25 * (
        proxy.TRACKING_PHASE_MEANS["final_third"] - proxy.TRACKING_PHASE_MEANS["build_up"]
    )


def test_predict_returns_one_row_per_turnover_with_no_epv(
    toy_dataset: pd.DataFrame, events_frame: pd.DataFrame
) -> None:
    """The event-only path emits estimates and refuses to invent an EPV number."""
    model = proxy.train(toy_dataset, toy_dataset["recovery_360"])
    out = proxy.predict(model, events_frame)
    assert len(out) == 2
    assert out["recovery"].between(0.0, 1.0).all()
    assert out["epv_conceded"].isna().all()
    assert out["estimated"].all()


def test_predict_on_a_match_with_no_turnovers_is_empty_not_broken() -> None:
    """A file with no open-play losses yields an empty frame with the schema."""
    model_input = pd.DataFrame(columns=list(proxy.EVENT_FIELDS))
    dummy = proxy.ProxyModel(
        estimator=None,
        feature_names=proxy.FEATURE_COLUMNS,
        categorical_names=proxy.CATEGORICAL_FEATURES,
        vocabularies={c: () for c in proxy.CATEGORICAL_FEATURES},
        target="recovery_360",
    )
    out = proxy.predict(dummy, model_input)
    assert out.empty
    assert "recovery" in out.columns


# --------------------------------------------------------------------------- #
# Fetching, offline
# --------------------------------------------------------------------------- #


def test_statsbomb_urls_point_at_the_open_data_layout() -> None:
    """The cache mirrors ``data/`` exactly, so a URL is a path join."""
    assert proxy.statsbomb_url("events", "1.json").endswith("/data/events/1.json")
    assert proxy.statsbomb_url("three-sixty", "1.json").endswith("/data/three-sixty/1.json")
    assert proxy.statsbomb_url("", "competitions.json").endswith("/data/competitions.json")


def test_fetch_returns_a_cached_file_without_touching_the_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cached file is never re-downloaded, which the rules require."""
    target = tmp_path / "events" / "7.json"
    target.parent.mkdir(parents=True)
    target.write_text("[]")

    def explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("the network must not be touched for a cached file")

    monkeypatch.setattr(proxy.urllib.request, "urlopen", explode)
    assert proxy.fetch_statsbomb("events", "7.json", tmp_path) == target


def test_fetch_without_download_permission_raises(tmp_path: Path) -> None:
    """Offline callers get a clear error rather than a hang."""
    with pytest.raises(FileNotFoundError, match="not cached"):
        proxy.fetch_statsbomb("events", "7.json", tmp_path, allow_download=False)


def test_matches_with_360_needs_the_tree_listing(tmp_path: Path) -> None:
    """Enumeration is driven off the manifest, never off a directory scan.

    274 orphan event files belong to no season; a directory scan would pull
    them in with no competition metadata.
    """
    with pytest.raises(FileNotFoundError, match="tree.json"):
        proxy.matches_with_360(tmp_path)


def test_matches_with_360_lists_only_matches_that_have_a_frame_file(
    tmp_path: Path,
) -> None:
    """A match without a 360 blob is not a labelled match."""
    (tmp_path / "matches" / "11").mkdir(parents=True)
    (tmp_path / "tree.json").write_text(
        json.dumps({"tree": [
            {"path": "data/three-sixty/500.json", "size": 100},
            {"path": "data/events/500.json", "size": 10},
            {"path": "data/events/501.json", "size": 10},
        ]})
    )
    (tmp_path / "matches" / "11" / "90.json").write_text(
        json.dumps([
            {
                "match_id": mid,
                "competition": {"competition_name": "La Liga"},
                "season": {"season_name": "2020/2021"},
                "home_team": {"home_team_name": "H"},
                "away_team": {"away_team_name": "A"},
                "match_date": "2021-01-01",
            }
            for mid in (500, 501)
        ])
    )
    listing = proxy.matches_with_360(tmp_path)
    assert listing["match_id"].tolist() == [500]
    assert listing["bytes_360"].tolist() == [100]


def test_build_match_dataset_says_which_file_is_missing(tmp_path: Path) -> None:
    """A partly cached match names the absent file rather than failing late."""
    with pytest.raises(FileNotFoundError, match="not cached"):
        proxy.build_match_dataset(12345, tmp_path)
