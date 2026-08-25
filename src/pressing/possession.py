"""Possession segmentation and open-play loss detection.

Two things happen here, and both rest on findings verified against the raw
Metrica event files rather than on what the event names suggest.

Possessions are maximal same-team runs over the six event types that actually
carry the ball. CHALLENGE, FAULT RECEIVED and CARD are excluded: a CHALLENGE is
logged as a single row for one side only 211 times out of 233 in game 1, so
including it splits possessions that never ended.

A loss fires at a ``BALL LOST`` row's **End Frame**, not its Start Frame. The
Start Frame is when the losing player's action began; the End Frame is the
moment and place the ball actually changed hands, which is why the paired
opponent ``RECOVERY`` starts within 0-2 frames of it and carries the same
coordinates. The single ``END HALF``-tagged ``BALL LOST`` per match is the
whistle, not a turnover, and is excluded. ``BALL OUT`` is a dead ball, so
counter-pressing it is meaningless; it is counted separately and reported next
to the losses so a coach sees the whole turnover picture.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from pressing._io import METRICA_FPS, event_extra

__all__ = [
    "BALL_OUT_TYPE",
    "LOSS_COLUMNS",
    "POSSESSION_COLUMNS",
    "POSSESSION_TYPES",
    "count_ball_outs",
    "detect_losses",
    "segment_possessions",
]

#: The event types that carry the ball and therefore define a possession run.
POSSESSION_TYPES: Final[frozenset[str]] = frozenset(
    {"PASS", "SHOT", "BALL LOST", "BALL OUT", "RECOVERY", "SET PIECE"}
)

BALL_OUT_TYPE: Final[str] = "BALL OUT"
_BALL_LOST_TYPE: Final[str] = "BALL LOST"
#: Subtype token marking the ``BALL LOST`` row that is really the half-time
#: whistle. Subtypes are hyphen-joined compounds, so this is a token test.
_END_HALF_TOKEN: Final[str] = "END HALF"

POSSESSION_COLUMNS: Final[tuple[str, ...]] = (
    "possession_id",
    "team",
    "period",
    "start_t",
    "end_t",
    "duration",
    "n_events",
    "start_index",
    "end_index",
    "end_type",
)

LOSS_COLUMNS: Final[tuple[str, ...]] = (
    "loss_id",
    "t",
    "frame",
    "period",
    "losing_team",
    "winning_team",
    "x",
    "y",
    "location_source",
    "losing_player",
    "subtype",
    "event_index",
)

_OPPONENT: Final[dict[str, str]] = {"home": "away", "away": "home"}


def segment_possessions(events: pd.DataFrame) -> pd.DataFrame:
    """Segment a canonical events frame into maximal same-team possessions.

    Only :data:`POSSESSION_TYPES` participate; every other row is dropped before
    the runs are cut, so a challenge logged for one side alone does not split a
    possession that never ended.

    Args:
        events: Canonical events frame, in event order.

    Returns:
        A dataframe with :data:`POSSESSION_COLUMNS`, one row per possession.
        ``start_index``/``end_index`` are positional indices into ``events``.
    """
    if events.empty:
        return pd.DataFrame(columns=list(POSSESSION_COLUMNS))

    ball = events.reset_index(drop=True)
    mask = ball["type"].isin(POSSESSION_TYPES).to_numpy()
    positions = np.flatnonzero(mask)
    if positions.size == 0:
        return pd.DataFrame(columns=list(POSSESSION_COLUMNS))

    run = ball.loc[positions].reset_index(drop=True)
    team = run["team"].to_numpy()
    period = run["period"].to_numpy()
    new_run = np.empty(len(run), dtype=bool)
    new_run[0] = True
    # A run breaks on a change of team, and also at half time: in Metrica game 1
    # the home team both ends period 1 in possession and kicks period 2 off, so
    # a pure team-change cut would splice a single 1,600 s "possession" across
    # the interval.
    new_run[1:] = (team[1:] != team[:-1]) | (period[1:] != period[:-1])

    starts = np.flatnonzero(new_run)
    ends = np.append(starts[1:], len(run)) - 1

    end_t = event_extra(run, "end_t").to_numpy(dtype=np.float64)
    start_t = run["t"].to_numpy(dtype=np.float64)
    # RECOVERY and SET PIECE rows carry End Frame == Start Frame, and one
    # kick-off row in game 1 carries End Frame 0, so a possession's end time is
    # the later of its last event's start and end.
    run_end_t = np.fmax(start_t, np.where(np.isfinite(end_t), end_t, start_t))

    out = pd.DataFrame(
        {
            "possession_id": np.arange(len(starts), dtype=np.int32),
            "team": team[starts],
            "period": run["period"].to_numpy()[starts],
            "start_t": start_t[starts],
            "end_t": run_end_t[ends],
            "n_events": ends - starts + 1,
            "start_index": positions[starts].astype(np.int32),
            "end_index": positions[ends].astype(np.int32),
            "end_type": run["type"].to_numpy()[ends],
        }
    )
    out["duration"] = out["end_t"] - out["start_t"]
    out.attrs.update(events.attrs)
    return out.loc[:, list(POSSESSION_COLUMNS)]


def count_ball_outs(events: pd.DataFrame) -> int:
    """Count ``BALL OUT`` events.

    These are turnovers, but dead-ball ones: the ball leaves the field and the
    other team restarts it, so there is nothing to counter-press. The report
    shows the count beside the open-play losses.

    Args:
        events: Canonical events frame.

    Returns:
        The number of ``BALL OUT`` rows.
    """
    if events.empty:
        return 0
    return int((events["type"] == BALL_OUT_TYPE).sum())


def detect_losses(
    events: pd.DataFrame,
    tracking: pd.DataFrame | None = None,
    *,
    fps: float | None = None,
) -> pd.DataFrame:
    """Detect open-play possession losses and locate each one.

    The trigger instant is the ``BALL LOST`` row's End Frame. Location is
    resolved in this order, and the column ``location_source`` records which one
    won so the report never has to guess:

    1. ``tracking_ball`` - the tracked ball position at that frame. Available at
       253/256 loss frames in game 1 and 232/232 in game 2.
    2. ``event_end`` - the ``BALL LOST`` End X/Y.
    3. ``next_event_start`` - the following event's Start X/Y.
    4. ``none`` - nothing available; ``x``/``y`` are NaN. One loss in game 1,
       none in game 2.

    Args:
        events: Canonical events frame for one match.
        tracking: Canonical tracking frame for the same match. When ``None`` the
            first fallback is skipped.
        fps: Frame rate used to turn frames into seconds. Defaults to the
            ``fps`` recorded in ``events.attrs``, else Metrica's 25 Hz.

    Returns:
        A dataframe with :data:`LOSS_COLUMNS`, one row per open-play loss,
        ordered by frame. ``DataFrame.attrs["ball_out_count"]`` carries the
        separately reported ``BALL OUT`` count.
    """
    rate = float(fps if fps is not None else events.attrs.get("fps", METRICA_FPS))

    ball = events.reset_index(drop=True)
    tokens = event_extra(ball, "subtype_tokens")
    is_loss = (ball["type"] == _BALL_LOST_TYPE) & ~tokens.map(
        lambda tok: _END_HALF_TOKEN in (tok or frozenset())
    )
    positions = np.flatnonzero(is_loss.to_numpy())

    frames = event_extra(ball, "end_frame").to_numpy(dtype="float64")[positions].astype(np.int64)
    losing_team = ball["team"].to_numpy()[positions]

    out = pd.DataFrame(
        {
            "loss_id": np.arange(positions.size, dtype=np.int32),
            "t": frames / rate,
            "frame": frames.astype(np.int32),
            "period": ball["period"].to_numpy()[positions],
            "losing_team": losing_team,
            "winning_team": [_OPPONENT.get(str(team), "unknown") for team in losing_team],
            "x": np.nan,
            "y": np.nan,
            "location_source": "none",
            "losing_player": ball["player_id"].to_numpy()[positions],
            "subtype": ball["subtype"].to_numpy()[positions],
            "event_index": positions.astype(np.int32),
        }
    )

    _fill_from_tracking(out, tracking)
    _fill_from_columns(
        out,
        ball["end_x"].to_numpy(dtype="float64")[positions],
        ball["end_y"].to_numpy(dtype="float64")[positions],
        "event_end",
    )
    following = np.minimum(positions + 1, len(ball) - 1)
    valid_next = positions + 1 < len(ball)
    next_x = np.where(valid_next, ball["x"].to_numpy(dtype="float64")[following], np.nan)
    next_y = np.where(valid_next, ball["y"].to_numpy(dtype="float64")[following], np.nan)
    _fill_from_columns(out, next_x, next_y, "next_event_start")

    out = out.sort_values("frame", kind="stable", ignore_index=True)
    out["loss_id"] = np.arange(len(out), dtype=np.int32)
    out.attrs.update(events.attrs)
    out.attrs["ball_out_count"] = count_ball_outs(events)
    return out.loc[:, list(LOSS_COLUMNS)]


def _fill_from_tracking(losses: pd.DataFrame, tracking: pd.DataFrame | None) -> None:
    """Fill still-unlocated losses from the tracked ball position, in place."""
    if tracking is None or tracking.empty or losses.empty:
        return
    ball = tracking[tracking["team"] == "ball"]
    if ball.empty:
        return
    positions = ball.drop_duplicates("frame").set_index("frame")[["x", "y"]]
    looked_up = positions.reindex(losses["frame"].to_numpy())
    _fill_from_columns(
        losses,
        looked_up["x"].to_numpy(dtype="float64"),
        looked_up["y"].to_numpy(dtype="float64"),
        "tracking_ball",
    )


def _fill_from_columns(
    losses: pd.DataFrame,
    xs: np.ndarray,
    ys: np.ndarray,
    source: str,
) -> None:
    """Fill still-unlocated losses from a candidate coordinate pair, in place."""
    if losses.empty:
        return
    unset = losses["location_source"].to_numpy() == "none"
    usable = unset & np.isfinite(xs) & np.isfinite(ys)
    if not usable.any():
        return
    losses.loc[usable, "x"] = xs[usable]
    losses.loc[usable, "y"] = ys[usable]
    losses.loc[usable, "location_source"] = source
