"""Tests for :mod:`pressing.possession`.

Most cases are built from hand-written event sequences so the expected answer is
obvious by inspection. Two integration cases run against the truncated real
Metrica CSVs under ``tests/fixtures/metrica``. Everything is offline.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pressing import _io
from pressing.possession import (
    LOSS_COLUMNS,
    POSSESSION_COLUMNS,
    count_ball_outs,
    detect_losses,
    segment_possessions,
)

FIXTURES = Path(__file__).parent / "fixtures" / "metrica"

# (team, type, subtype, period, start_frame, end_frame, from, to, sx, sy, ex, ey)
Row = tuple


def _events(tmp_path: Path, rows: list[Row], *, game: int = 9) -> pd.DataFrame:
    """Write a Metrica-shaped events CSV and load it back canonically."""
    path = tmp_path / f"Sample_Game_{game}_RawEventsData.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Team", "Type", "Subtype", "Period", "Start Frame", "Start Time [s]",
                "End Frame", "End Time [s]", "From", "To",
                "Start X", "Start Y", "End X", "End Y",
            ]
        )
        for team, kind, subtype, period, start, end, sender, receiver, sx, sy, ex, ey in rows:
            writer.writerow(
                [
                    team, kind, subtype, period,
                    start, start / _io.METRICA_FPS, end, end / _io.METRICA_FPS,
                    sender, receiver, sx, sy, ex, ey,
                ]
            )
    return _io.load_metrica_events(game, tmp_path, flip_period=None)


def _ball_track(frames: dict[int, tuple[float, float]]) -> pd.DataFrame:
    """Build a minimal tracking frame holding only ball positions, in metres."""
    return pd.DataFrame(
        {
            "team": ["ball"] * len(frames),
            "frame": list(frames),
            "x": [xy[0] for xy in frames.values()],
            "y": [xy[1] for xy in frames.values()],
        }
    )


# --------------------------------------------------------------------------- #
# Possession segmentation
# --------------------------------------------------------------------------- #


def test_segments_maximal_same_team_runs(tmp_path: Path) -> None:
    events = _events(
        tmp_path,
        [
            ("Home", "PASS", "", 1, 10, 20, "Player1", "Player2", 0.5, 0.5, 0.6, 0.5),
            ("Home", "PASS", "", 1, 20, 30, "Player2", "Player3", 0.6, 0.5, 0.7, 0.5),
            ("Home", "BALL LOST", "INTERCEPTION", 1, 30, 45, "Player3", "", 0.7, 0.5, 0.8, 0.5),
            ("Away", "RECOVERY", "INTERCEPTION", 1, 46, 46, "Player20", "", 0.8, 0.5, 0.0, 0.0),
            ("Away", "PASS", "", 1, 50, 60, "Player20", "Player21", 0.8, 0.5, 0.9, 0.5),
            ("Away", "SHOT", "ON TARGET-SAVED", 1, 70, 80, "Player21", "", 0.9, 0.5, 1.0, 0.5),
        ],
    )
    possessions = segment_possessions(events)

    assert list(possessions.columns) == list(POSSESSION_COLUMNS)
    assert possessions["team"].tolist() == ["home", "away"]
    assert possessions["n_events"].tolist() == [3, 3]
    assert possessions["end_type"].tolist() == ["BALL LOST", "SHOT"]
    assert possessions["start_index"].tolist() == [0, 3]
    assert possessions["end_index"].tolist() == [2, 5]
    assert possessions["start_t"].tolist() == pytest.approx([0.4, 1.84])
    # The first possession ends when the ball is actually lost: End Frame 45.
    assert possessions["end_t"].tolist() == pytest.approx([1.8, 3.2])
    assert possessions["duration"].tolist() == pytest.approx([1.4, 1.36])


def test_challenge_fault_and_card_do_not_split_a_possession(tmp_path: Path) -> None:
    """A CHALLENGE is logged for one side only, so counting it would split runs."""
    events = _events(
        tmp_path,
        [
            ("Home", "PASS", "", 1, 10, 20, "Player1", "Player2", 0.5, 0.5, 0.6, 0.5),
            ("Away", "CHALLENGE", "TACKLE-LOST", 1, 21, 21, "Player20", "", 0.6, 0.5, 0.0, 0.0),
            ("Home", "FAULT RECEIVED", "", 1, 21, 21, "Player2", "", 0.6, 0.5, 0.0, 0.0),
            ("Away", "CARD", "YELLOW", 1, 22, 22, "Player20", "", 0.0, 0.0, 0.0, 0.0),
            ("Home", "PASS", "", 1, 30, 40, "Player2", "Player3", 0.6, 0.5, 0.7, 0.5),
        ],
    )
    possessions = segment_possessions(events)
    assert len(possessions) == 1
    assert possessions["team"].item() == "home"
    assert possessions["n_events"].item() == 2


def test_possessions_break_at_half_time(tmp_path: Path) -> None:
    """The same team can end one half and kick the next one off; that is two."""
    events = _events(
        tmp_path,
        [
            ("Home", "PASS", "", 1, 10, 20, "Player1", "Player2", 0.5, 0.5, 0.6, 0.5),
            ("Home", "BALL LOST", "FORCED-END HALF", 1, 30, 30, "Player2", "", 0.6, 0.5, 0.0, 0.0),
            ("Home", "SET PIECE", "KICK OFF", 2, 5000, 5000, "Player1", "", 0.0, 0.0, 0.0, 0.0),
            ("Home", "PASS", "", 2, 5000, 5010, "Player1", "Player2", 0.5, 0.5, 0.6, 0.5),
        ],
    )
    possessions = segment_possessions(events)
    assert len(possessions) == 2
    assert possessions["period"].tolist() == [1, 2]
    assert possessions["duration"].max() < 5.0, "no possession may span the interval"


def test_segment_possessions_handles_an_empty_frame() -> None:
    empty = pd.DataFrame(columns=["type", "team", "period", "t", "extra"])
    assert segment_possessions(empty).empty


# --------------------------------------------------------------------------- #
# Loss detection
# --------------------------------------------------------------------------- #


def test_loss_fires_on_end_frame_not_start_frame(tmp_path: Path) -> None:
    events = _events(
        tmp_path,
        [
            ("Home", "BALL LOST", "INTERCEPTION", 1, 100, 137, "Player3", "", 0.7, 0.5, 0.8, 0.5),
            ("Away", "RECOVERY", "INTERCEPTION", 1, 138, 138, "Player20", "", 0.8, 0.5, 0.0, 0.0),
        ],
    )
    losses = detect_losses(events)

    assert list(losses.columns) == list(LOSS_COLUMNS)
    assert len(losses) == 1
    assert losses["frame"].item() == 137, "the ball changes hands at the End Frame"
    assert losses["t"].item() == pytest.approx(137 / 25.0)
    assert losses["losing_team"].item() == "home"
    assert losses["winning_team"].item() == "away"
    assert losses["losing_player"].item() == "Player3"


def test_end_half_ball_lost_is_not_a_turnover(tmp_path: Path) -> None:
    events = _events(
        tmp_path,
        [
            ("Home", "BALL LOST", "INTERCEPTION", 1, 100, 137, "Player3", "", 0.7, 0.5, 0.8, 0.5),
            ("Away", "BALL LOST", "FORCED-END HALF", 1, 200, 210, "Player20", "",
             0.5, 0.5, 0.5, 0.5),
        ],
    )
    losses = detect_losses(events)
    assert len(losses) == 1
    assert losses["losing_team"].item() == "home"


def test_ball_out_is_counted_but_never_a_loss(tmp_path: Path) -> None:
    events = _events(
        tmp_path,
        [
            ("Home", "BALL OUT", "", 1, 100, 110, "Player3", "", 0.7, 0.5, 1.0, 0.5),
            ("Home", "BALL OUT", "THROW IN", 1, 200, 210, "Player4", "", 0.7, 0.0, 0.7, 0.0),
            ("Home", "BALL LOST", "INTERCEPTION", 1, 300, 320, "Player3", "", 0.7, 0.5, 0.8, 0.5),
        ],
    )
    losses = detect_losses(events)
    assert len(losses) == 1
    assert count_ball_outs(events) == 2
    assert losses.attrs["ball_out_count"] == 2


def test_location_prefers_the_tracked_ball(tmp_path: Path) -> None:
    events = _events(
        tmp_path,
        [("Home", "BALL LOST", "INTERCEPTION", 1, 100, 137, "Player3", "", 0.7, 0.5, 0.2, 0.2)],
    )
    tracking = _ball_track({137: (60.0, 20.0)})
    losses = detect_losses(events, tracking)
    assert losses["location_source"].item() == "tracking_ball"
    assert losses["x"].item() == pytest.approx(60.0)
    assert losses["y"].item() == pytest.approx(20.0)


def test_location_falls_back_to_the_event_end(tmp_path: Path) -> None:
    events = _events(
        tmp_path,
        [("Home", "BALL LOST", "INTERCEPTION", 1, 100, 137, "Player3", "", 0.7, 0.5, 0.2, 0.25)],
    )
    tracking = _ball_track({136: (60.0, 20.0)})  # nothing at the loss frame
    losses = detect_losses(events, tracking)
    assert losses["location_source"].item() == "event_end"
    assert losses["x"].item() == pytest.approx(0.2 * 105.0)
    assert losses["y"].item() == pytest.approx((1.0 - 0.25) * 68.0)


def test_location_falls_back_to_the_next_event_start(tmp_path: Path) -> None:
    events = _events(
        tmp_path,
        [
            ("Home", "BALL LOST", "INTERCEPTION", 1, 100, 137, "Player3", "", 0.7, 0.5, "", ""),
            ("Away", "RECOVERY", "INTERCEPTION", 1, 138, 138, "Player20", "", 0.4, 0.6, "", ""),
        ],
    )
    losses = detect_losses(events, _ball_track({}))
    assert losses["location_source"].item() == "next_event_start"
    assert losses["x"].item() == pytest.approx(0.4 * 105.0)
    assert losses["y"].item() == pytest.approx((1.0 - 0.6) * 68.0)


def test_an_unlocatable_loss_is_reported_rather_than_guessed(tmp_path: Path) -> None:
    events = _events(
        tmp_path,
        [("Home", "BALL LOST", "INTERCEPTION", 1, 100, 137, "Player3", "", "", "", "", "")],
    )
    losses = detect_losses(events, _ball_track({}))
    assert losses["location_source"].item() == "none"
    assert np.isnan(losses["x"].item())
    assert np.isnan(losses["y"].item())


def test_losses_come_back_ordered_by_frame(tmp_path: Path) -> None:
    events = _events(
        tmp_path,
        [
            ("Home", "BALL LOST", "INTERCEPTION", 1, 300, 320, "Player3", "", 0.7, 0.5, 0.7, 0.5),
            ("Away", "BALL LOST", "THEFT", 1, 100, 110, "Player20", "", 0.3, 0.5, 0.3, 0.5),
        ],
    )
    losses = detect_losses(events)
    assert losses["frame"].tolist() == [110, 320]
    assert losses["loss_id"].tolist() == [0, 1]


# --------------------------------------------------------------------------- #
# Against the real (truncated) Metrica files
# --------------------------------------------------------------------------- #


def test_real_slice_loss_uses_the_tracked_ball() -> None:
    """Game 2's slice holds one BALL LOST whose End X/Y is NaN in the raw file."""
    match = _io.load_metrica_match(2, FIXTURES)
    losses = detect_losses(match.events, match.tracking)

    assert len(losses) == 1
    row = losses.iloc[0]
    assert row["frame"] == 68031
    assert row["period"] == 2
    assert row["losing_team"] == "home"
    assert row["winning_team"] == "away"
    assert row["losing_player"] == "Player4"
    assert row["location_source"] == "tracking_ball"

    ball = match.tracking[
        (match.tracking["team"] == "ball") & (match.tracking["frame"] == 68031)
    ]
    assert row["x"] == pytest.approx(ball["x"].item())
    assert row["y"] == pytest.approx(ball["y"].item())


def test_real_slice_possessions_respect_the_team_runs() -> None:
    match = _io.load_metrica_match(2, FIXTURES)
    possessions = segment_possessions(match.events)
    assert possessions["team"].tolist() == ["away", "home"]
    assert possessions["period"].tolist() == [1, 2]
    assert possessions["end_type"].tolist() == ["PASS", "BALL LOST"]
