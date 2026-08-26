"""The counter-press metric: how much a team wins back in the five seconds
after it gives the ball away.

What the number means, in plain language
----------------------------------------
The moment your team loses the ball is ``t0``. For the next five seconds you
are the counter-pressing team and the opponent is trying to get away from you.
We freeze the match at six instants -- ``t0``, ``t0+1s``, ``t0+2s``, ``t0+3s``,
``t0+4s``, ``t0+5s`` -- and at each one we compute a pitch-control surface: for
every 2 x 2 m patch of grass, the probability that your team would win the ball
if it arrived there, and the same for the opponent.

Your **control share** at an instant is your slice of that surface::

    share(t) = sum(w * yours) / sum(w * (yours + theirs))

``w`` is a per-cell weight. Two are computed for every loss and both are
reported:

``recovery_area``
    ``w = 1``. Every patch of grass counts the same, so this is the plain share
    of the pitch you control.

``recovery_epv``
    ``w = EPV``, the expected-possession-value of the ball being at that patch
    **for the opponent**, i.e. weighted toward your own goal, which is where
    they are trying to get to. Control of the grass behind them is nearly
    worthless and this weighting says so.

Whichever weighting is used, the number is the **average control share over the
six instants**::

    recovery = mean( share(t0), share(t0+1s), ..., share(t0+5s) )

Read it as a probability. ``1.0`` means you owned everything that mattered for
the whole five seconds -- an instant regain. ``0.0`` means you owned nothing --
they walked away from you. ``0.5`` is a dead-even standoff.

Why the headline is the AREA share
----------------------------------
The EPV weighting sounds better and measures worse, so :data:`DEFAULT_WEIGHTING`
is ``"area"`` and ``recovery_epv`` rides alongside. Weighting toward the goal
the opponent is attacking puts nearly all the weight inside the counter-pressing
team's own penalty box -- exactly the cells its own defensive block and
goalkeeper control almost by definition, and the pitch-control model gives a
defending goalkeeper a ball-control rate three times an outfielder's, which
amplifies it further. Measured over all 255 scored losses in Metrica game 1:

============  ======  ========  ======  ======  ======
weighting     min     median    mean    max     sd
============  ======  ========  ======  ======  ======
``area``      0.1697  0.5629    0.5559  0.9150  0.1767
``epv``       0.4351  0.8388    0.8025  0.9761  0.1233
============  ======  ========  ======  ======  ======

Game 2 is the same story (area sd 0.1925 against EPV 0.1361). The EPV share
never goes below 0.39, sits three quarters of the way up its range, and buys no
discrimination for it: the two rank losses almost identically (Pearson r =
0.957 in game 1, 0.953 in game 2) and correlate with the event-stream regain
flag equally weakly (0.07 / 0.08 in game 1). A number a coach reads against a
stated neutral of 0.5 is worth more than a number that lives between 0.75 and
0.95. :func:`with_weighting` switches the headline column with no recomputation.

Which column is which, at both levels
-------------------------------------
Per loss, :data:`LOSS_SCORE_COLUMNS` carries ``recovery_area`` and
``recovery_epv`` side by side plus ``recovery``, which is deliberately an alias
for whichever of the two the active weighting selects. That is what makes
:func:`with_weighting` free: both numbers are already there.

The aggregates must not repeat that trick, because a table prints its column
names. So :func:`per_team`, :func:`per_phase` and :func:`per_player` emit
``mean_recovery`` -- always the headline weighting -- and ``mean_recovery_alt``
-- always the other one. The two are never the same number under two headings.
The frame's ``attrs["weighting"]`` and ``attrs["alt_weighting"]`` name them, and
:data:`WEIGHTING_LABELS` turns either name into words fit for a heading. Use
:func:`headline_weighting` to read the weighting back off any scored frame.

Read every recovery number WITHIN its phase
-------------------------------------------
The single largest driver of this metric is where on the pitch the ball was
lost, not how hard anyone pressed. Lose it in your own third and your team is
compressed around your own goal while the opponent occupies everything else, so
your area share is small; lose it in theirs and the geometry inverts. Measured
mean ``recovery_area`` in game 1, as the range over the two teams: build-up
0.24-0.28, middle 0.44-0.49, final third 0.70. The EPV weighting does not fix
this -- the same three splits, grouped the same way, are 0.52-0.58, 0.75-0.78
and 0.89. This is why :func:`per_phase` exists and why the report must never put
a build-up recovery next to a final-third one and call the difference pressing.

Two companions travel with it:

``epv_conceded``
    How much expected possession value the opponent picked up. It is their
    control-weighted mean EPV averaged over ``t0+1s .. t0+5s`` minus their value
    at ``t0``: positive means they made ground on you, negative means your press
    pushed them backwards. The underlying quantity is ``mean(EPV * control)``
    over the grid -- a PER-CELL mean, so it does not change if the grid is
    refined, but it is consequently small (Metrica game 1: -0.0026 to +0.0040
    per loss). Multiply by the cell count for the pitch integral if a report
    wants a larger number; do not compare integrals computed at different
    resolutions.

``regained_within_5s``
    The plain, checkable yes/no from the event stream: did your team have the
    ball again inside five seconds? The continuous ``recovery`` is the
    interesting number; this is the one you can argue about with a coach while
    watching the clip, so it is always carried alongside.

Design choices that materially change the number
------------------------------------------------
**Six timepoints, not 10 Hz.** Sampling the window at the tracking rate would
mean 51 surfaces per loss. Measured: 10 Hz over ~300 losses costs 121-161 s of
pure control computation, which alone blows the two-minute demo budget, while
six timepoints costs 65 s serial (14 s on six workers) for the same 300 losses.
The timepoint set is :data:`DEFAULT_TIMEPOINTS` and is a parameter
(:attr:`CounterPressParams.timepoints`), so anyone who wants the expensive
version can ask for it.

**Who counts as "in possession" for the surface.** The pitch-control model is
mildly asymmetric: the defending team's goalkeeper gets a much larger
ball-control rate. For the whole window we treat the team that WON the ball as
the team in possession, fixed at ``t0``, even if the counter-press succeeds
halfway through. That keeps every one of the six surfaces computed the same way,
so differences between them come from players moving and nothing else.

**The window is closed at both ends.** ``[t0, t0+5s]``. The sample at exactly
``t0+5.000 s`` is included, and an opponent-losing-the-ball-back event at
exactly ``t0+5.000 s`` counts as a regain. Timepoints are also confined to the
loss's own period: a loss four seconds before half time is scored on the two or
three frames that exist before the whistle, never on frames from the second
half.

**Missing ball.** The ball is untracked in 39-41% of Metrica frames (game 1
39.1%, game 2 41.0%). Coverage inside a counter-press window is better than
that but still far from complete: at the six sample points it is 85.2% in game
1 and 79.7% in game 2, and across every frame of the five seconds 83.4% and
76.7%. The ball position only enters the model as a travel time, but it does
enter it. The policy is a parameter (:attr:`CounterPressParams.ball_policy`)
and the default is ``"hold"``: carry the last known ball position forward,
within the same period, for at most
:attr:`CounterPressParams.ball_hold_max_s` seconds. The
justification is that the NaN runs are dead-ball stretches -- the ball really is
lying roughly where it went out -- so holding is closer to the truth than
either dropping the timepoint (which would silently average different numbers
of samples across losses, making them non-comparable) or interpolating (which
invents ball motion through positions the ball never occupied, e.g. straight
across the pitch to a throw-in). ``"interpolate"``, ``"skip"`` and ``"none"``
(the reference behaviour of assuming zero travel time) are all available.

Nothing is silently dropped: a loss that cannot be scored stays in the table
with NaN metrics and a ``skip_reason`` drawn from :data:`SKIP_REASONS`.

Aggregation
-----------
``per_player`` credits a loss to every counter-pressing player within
:attr:`CounterPressParams.radius_m` (15 m by default) of the loss location at
``t0``, inclusive of the boundary. Minutes come from each player's actual
tracked presence, not an assumed 90.

``per_phase`` splits by :mod:`pressing.phases` -- build-up, middle, final third,
measured from the losing team's own goal. ``per_team`` and ``team_comparison``
give the head-to-head the report's cover page needs.

Cost
----
Scoring a match is the expensive part and it parallelises cleanly, so
:func:`score_losses` takes ``jobs``. Measured on Metrica game 1 (256 losses, 255
scoreable, six timepoints each, 50 x 32 grid), whole-call wall clock, best of
three: 15.09 s at ``jobs=1``, 9.46 s at 2, 7.40 s at 4, 6.22 s at 6, 6.94 s at
10. It stops improving past six because this machine is four performance cores
plus six efficiency cores, which is why :data:`MAX_USEFUL_JOBS` is six.

Inside that, the scoring loop alone is 14.26 s serial against 3.21 s on six
workers -- 4.4x -- and the remaining 2.31 s is the process pool starting up,
measured with an empty pool of the same size. Pool start-up is why
:data:`_MIN_PARALLEL_LOSSES` exists: a handful of losses is not worth it.

``jobs=1`` is a fully supported path, not a fallback, and every column it
produces is bit-identical to any other ``jobs`` value -- verified over all 256
game 1 losses at 1, 2, 4, 6 and 10 workers, all 31 columns.
"""

from __future__ import annotations

import multiprocessing
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field, replace
from typing import Any, Final, Literal

import numpy as np
import pandas as pd

from pressing._io import METRICA_FPS
from pressing.epv import epv_surface
from pressing.phases import PHASES, label_loss_phases
from pressing.pitchcontrol import (
    PitchControlParams,
    PitchGrid,
    PlayerSet,
    make_grid,
    pitch_control_full,
)
from pressing.possession import POSSESSION_TYPES, detect_losses
from pressing.schema import PITCH_LENGTH

__all__ = [
    "BALL_POLICIES",
    "DEFAULT_RADIUS_M",
    "DEFAULT_TIMEPOINTS",
    "DEFAULT_WEIGHTING",
    "LOSS_SCORE_COLUMNS",
    "MAX_USEFUL_JOBS",
    "RECOVERY_WINDOW_S",
    "SKIP_REASONS",
    "WEIGHTING_LABELS",
    "BallPolicy",
    "CounterPressParams",
    "PossessionLoss",
    "Weighting",
    "alternate_weighting",
    "counterpress_value",
    "default_jobs",
    "find_possession_losses",
    "headline_weighting",
    "minutes_played",
    "per_phase",
    "per_player",
    "per_team",
    "score_losses",
    "score_match",
    "summarise_match",
    "team_comparison",
    "with_weighting",
]

#: Length of the counter-press window in seconds.
RECOVERY_WINDOW_S: Final[float] = 5.0

#: Offsets from ``t0`` at which pitch control is evaluated, in seconds. Six
#: samples at 1 Hz rather than 51 at 10 Hz: measured, 10 Hz over 300 losses is
#: 121 s of control computation on six workers (161 s for 400 losses) against a
#: 120 s budget for the whole demo, while these six are 65 s serial / 14 s on
#: six workers for the same 300 losses. The window is CLOSED, so 5.0 is in.
DEFAULT_TIMEPOINTS: Final[tuple[float, ...]] = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)

#: Radius in metres inside which a counter-pressing player is credited with a
#: loss. Inclusive: 15.0 m exactly is inside.
DEFAULT_RADIUS_M: Final[float] = 15.0

Weighting = Literal["epv", "area"]
BallPolicy = Literal["hold", "interpolate", "skip", "none"]

#: Which control share the headline ``recovery`` column reports. ``"area"``:
#: the EPV weighting saturates high and adds no discrimination. See the module
#: docstring for the measured distributions behind that choice.
DEFAULT_WEIGHTING: Final[Weighting] = "area"

#: Human-readable label for each weighting, for table headings and captions.
#: An aggregate table's ``mean_recovery`` column is the headline weighting and
#: ``mean_recovery_alt`` is the other one; the frame's ``attrs["weighting"]``
#: and ``attrs["alt_weighting"]`` say which is which, and these are the words
#: to print above them.
WEIGHTING_LABELS: Final[dict[str, str]] = {
    "area": "area-weighted",
    "epv": "EPV-weighted",
}

BALL_POLICIES: Final[tuple[str, ...]] = ("hold", "interpolate", "skip", "none")

#: Worker cap for :func:`score_losses`. This machine is four performance cores
#: plus six efficiency ones, and scoring Metrica game 1 measured 6.22 s on six
#: workers against 6.94 s on ten: past six, more workers are slower, not faster.
MAX_USEFUL_JOBS: Final[int] = 6

#: Fewest scoreable losses worth starting a process pool for. Starting an empty
#: six-worker pool measured 2.31 s and scoring one loss on the default grid
#: measured 56 ms serial at 4.4x parallel, which puts break-even at about 53
#: losses; 64 keeps a margin. A real match has 230-260 losses and is never
#: near this, so the constant exists to stop a two-loss test scene paying for a
#: pool it cannot possibly amortise.
_MIN_PARALLEL_LOSSES: Final[int] = 64

#: Every reason a loss can fail to be scored. A skipped loss keeps its row.
SKIP_REASONS: Final[tuple[str, ...]] = (
    "no_loss_location",
    "no_ball",
    "window_truncated",
    "empty_team",
)

_TEAMS: Final[tuple[str, str]] = ("home", "away")
_OPPONENT: Final[dict[str, str]] = {"home": "away", "away": "home"}

LOSS_SCORE_COLUMNS: Final[tuple[str, ...]] = (
    "loss_id",
    "match_id",
    "t",
    "frame",
    "period",
    "losing_team",
    "winning_team",
    "x",
    "y",
    "x_norm",
    "phase",
    "location_source",
    "losing_player",
    "subtype",
    "event_index",
    "n_timepoints",
    "n_ball_filled",
    "recovery",
    "recovery_area",
    "recovery_epv",
    "share_t0_area",
    "share_t0_epv",
    "share_end_area",
    "share_end_epv",
    "epv_winner_t0",
    "epv_winner_mean",
    "epv_conceded",
    "regained_within_5s",
    "time_to_regain",
    "scored",
    "skip_reason",
)


@dataclass(frozen=True)
class CounterPressParams:
    """Every knob of the counter-press metric, in one place.

    Attributes:
        window_s: Length of the counter-press window in seconds. Only used to
            bound the event-stream regain test; the control samples are
            ``timepoints``.
        timepoints: Offsets from ``t0``, in seconds, at which pitch control is
            evaluated. See :data:`DEFAULT_TIMEPOINTS` for the budget reasoning.
        radius_m: Metres inside which a counter-pressing player is credited
            with a loss, inclusive of the boundary.
        weighting: Which share the headline ``recovery`` column carries,
            ``"epv"`` or ``"area"``. Both are always computed; the default is
            :data:`DEFAULT_WEIGHTING`.
        ball_policy: What to do when the ball is untracked at a timepoint. See
            the module docstring.
        ball_hold_max_s: How far back ``"hold"`` (and how far either way
            ``"interpolate"``) will look for a real ball position, in seconds.
        min_timepoints: Fewest usable samples a loss needs to be scored at all.
            Two -- ``t0`` plus one later instant -- is the minimum that says
            anything about a five-second window.
        pitch_length: Pitch length in metres, for the phase split.
        fps: Tracking rate. Defaults to the frame's own ``attrs["fps"]``.
    """

    window_s: float = RECOVERY_WINDOW_S
    timepoints: tuple[float, ...] = DEFAULT_TIMEPOINTS
    radius_m: float = DEFAULT_RADIUS_M
    weighting: Weighting = "area"
    ball_policy: BallPolicy = "hold"
    ball_hold_max_s: float = 5.0
    min_timepoints: int = 2
    pitch_length: float = PITCH_LENGTH
    fps: float | None = None

    def __post_init__(self) -> None:
        """Validate the knobs.

        Raises:
            ValueError: On an unknown weighting or ball policy, an empty or
                non-monotonic timepoint set, a negative radius, or
                ``min_timepoints`` below one.
        """
        if self.weighting not in ("epv", "area"):
            raise ValueError(f"weighting must be 'epv' or 'area', got {self.weighting!r}")
        if self.ball_policy not in BALL_POLICIES:
            raise ValueError(
                f"ball_policy must be one of {BALL_POLICIES}, got {self.ball_policy!r}"
            )
        points = tuple(float(p) for p in self.timepoints)
        if not points:
            raise ValueError("timepoints must not be empty")
        if points[0] != 0.0:
            raise ValueError(f"timepoints must start at 0.0 (the loss instant), got {points[0]}")
        if any(b <= a for a, b in zip(points, points[1:], strict=False)):
            raise ValueError(f"timepoints must be strictly increasing, got {points}")
        object.__setattr__(self, "timepoints", points)
        if self.radius_m < 0:
            raise ValueError(f"radius_m must be non-negative, got {self.radius_m}")
        if self.min_timepoints < 1:
            raise ValueError(f"min_timepoints must be at least 1, got {self.min_timepoints}")


@dataclass(frozen=True)
class PossessionLoss:
    """A single turnover and its aftermath.

    Attributes:
        match_id: Match the loss belongs to.
        t: Time of the turnover in seconds.
        period: Match period.
        losing_team: ``"home"`` or ``"away"``; the team that gave the ball away.
        x: Turnover location along the pitch, in metres.
        y: Turnover location across the pitch, in metres.
        phase: Pitch phase, one of ``"build_up"``, ``"middle"``, ``"final_third"``.
        recovery: Control share won back within the window, in ``[0, 1]``.
        epv_conceded: EPV accrued by the team that won the ball, over the window.
    """

    match_id: str
    t: float
    period: int
    losing_team: str
    x: float
    y: float
    phase: str | None
    recovery: float
    epv_conceded: float


# --------------------------------------------------------------------------- #
# Fast tracking access
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _TrackingIndex:
    """Numpy views over a canonical tracking frame, indexed by frame number.

    A match is millions of rows and a match is hundreds of losses times six
    timepoints, so ``tracking[tracking.frame == f]`` inside the loop is a
    quadratic disaster. Everything here is precomputed once: contiguous row
    slices per frame, and a dense per-frame ball position with the forward and
    backward nearest-known-ball indices already resolved.
    """

    frame_min: int
    frame_max: int
    fps: float
    row_start: np.ndarray
    row_stop: np.ndarray
    period: np.ndarray
    team_code: np.ndarray
    is_gk: np.ndarray
    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    player_id: np.ndarray
    ball_x: np.ndarray
    ball_y: np.ndarray
    ball_prev: np.ndarray
    ball_next: np.ndarray
    attack_direction: dict[str, int] = field(default_factory=lambda: {"home": 1, "away": -1})

    def has_frame(self, frame: int) -> bool:
        """Whether ``frame`` exists in the tracking data."""
        if frame < self.frame_min or frame > self.frame_max:
            return False
        slot = frame - self.frame_min
        return bool(self.row_stop[slot] > self.row_start[slot])

    def period_of(self, frame: int) -> int:
        """Period number of ``frame``, or ``-1`` when the frame is absent."""
        if frame < self.frame_min or frame > self.frame_max:
            return -1
        return int(self.period[frame - self.frame_min])

    def player_set(self, frame: int, team: str) -> PlayerSet:
        """Build a :class:`PlayerSet` for one team at one frame."""
        slot = frame - self.frame_min
        lo, hi = int(self.row_start[slot]), int(self.row_stop[slot])
        code = _TEAMS.index(team)
        sel = self.team_code[lo:hi] == code
        return PlayerSet.from_arrays(
            positions=np.stack([self.x[lo:hi][sel], self.y[lo:hi][sel]], axis=1),
            velocities=np.stack([self.vx[lo:hi][sel], self.vy[lo:hi][sel]], axis=1),
            is_gk=self.is_gk[lo:hi][sel],
            ids=self.player_id[lo:hi][sel],
        )

    def team_positions(self, frame: int, team: str) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(player_ids, positions)`` for one team at one frame."""
        slot = frame - self.frame_min
        lo, hi = int(self.row_start[slot]), int(self.row_stop[slot])
        code = _TEAMS.index(team)
        sel = self.team_code[lo:hi] == code
        return (
            self.player_id[lo:hi][sel],
            np.stack([self.x[lo:hi][sel], self.y[lo:hi][sel]], axis=1),
        )


def _runs(labels: np.ndarray) -> list[tuple[int, int]]:
    """Half-open index ranges of equal consecutive labels."""
    if labels.size == 0:
        return []
    edges = np.flatnonzero(np.diff(labels)) + 1
    bounds = np.concatenate(([0], edges, [labels.size]))
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:], strict=True)]


def _build_index(tracking: pd.DataFrame, fps: float) -> _TrackingIndex:
    """Precompute the frame-indexed numpy views used by the scoring loop.

    Args:
        tracking: Canonical long tracking frame for one match.
        fps: Tracking rate in frames per second.

    Returns:
        A :class:`_TrackingIndex`.

    Raises:
        ValueError: If ``tracking`` is empty.
    """
    if tracking.empty:
        raise ValueError("cannot index an empty tracking frame")
    ordered = tracking.sort_values("frame", kind="stable")

    frames = ordered["frame"].to_numpy(dtype=np.int64)
    frame_min, frame_max = int(frames[0]), int(frames[-1])
    slots = np.arange(frame_min, frame_max + 1, dtype=np.int64)
    row_start = np.searchsorted(frames, slots, side="left")
    row_stop = np.searchsorted(frames, slots, side="right")

    teams = ordered["team"].to_numpy()
    team_code = np.full(len(ordered), 2, dtype=np.int8)
    team_code[teams == "home"] = 0
    team_code[teams == "away"] = 1

    goalkeeper = dict(tracking.attrs.get("goalkeeper", {}))
    player_id = ordered["player_id"].to_numpy()
    is_gk = np.zeros(len(ordered), dtype=bool)
    for team, keeper in goalkeeper.items():
        if team in _TEAMS and keeper is not None:
            is_gk |= (team_code == _TEAMS.index(team)) & (player_id == keeper)

    # Dense per-frame period and ball position.
    period = np.full(slots.size, -1, dtype=np.int16)
    filled = row_stop > row_start
    period[filled] = ordered["period"].to_numpy(dtype=np.int16)[row_start[filled]]

    ball_x = np.full(slots.size, np.nan)
    ball_y = np.full(slots.size, np.nan)
    ball_rows = ordered[ordered["team"] == "ball"]
    if not ball_rows.empty:
        ball_slot = ball_rows["frame"].to_numpy(dtype=np.int64) - frame_min
        ball_x[ball_slot] = ball_rows["x"].to_numpy(dtype=np.float64)
        ball_y[ball_slot] = ball_rows["y"].to_numpy(dtype=np.float64)

    ball_prev, ball_next = _nearest_known_ball(ball_x, period)

    return _TrackingIndex(
        frame_min=frame_min,
        frame_max=frame_max,
        fps=float(fps),
        row_start=row_start,
        row_stop=row_stop,
        period=period,
        team_code=team_code,
        is_gk=is_gk,
        x=ordered["x"].to_numpy(dtype=np.float64),
        y=ordered["y"].to_numpy(dtype=np.float64),
        vx=ordered["vx"].to_numpy(dtype=np.float64),
        vy=ordered["vy"].to_numpy(dtype=np.float64),
        player_id=player_id,
        ball_x=ball_x,
        ball_y=ball_y,
        ball_prev=ball_prev,
        ball_next=ball_next,
        attack_direction=dict(tracking.attrs.get("attack_direction", {"home": 1, "away": -1})),
    )


def _nearest_known_ball(ball_x: np.ndarray, period: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index of the nearest known ball position before / after each frame slot.

    The search never crosses a period boundary, so a loss late in the first half
    can never borrow a ball position from the second.

    Args:
        ball_x: Dense per-frame ball x, NaN where the ball is untracked.
        period: Dense per-frame period, ``-1`` for absent frames.

    Returns:
        ``(prev, next)``. ``prev[i]`` is the largest ``j <= i`` in the same
        period with a known ball, or ``-1``; ``next[i]`` is the smallest
        ``j >= i``, or ``len(ball_x)``.
    """
    n = ball_x.size
    known = np.isfinite(ball_x)
    idx = np.arange(n)
    prev = np.where(known, idx, -1)
    nxt = np.where(known, idx, n)
    for start, stop in _runs(period):
        np.maximum.accumulate(prev[start:stop], out=prev[start:stop])
        block = nxt[start:stop][::-1].copy()
        np.minimum.accumulate(block, out=block)
        nxt[start:stop] = block[::-1]
    return prev, nxt


_SKIP = object()


def _ball_at(
    index: _TrackingIndex,
    frame: int,
    policy: BallPolicy,
    max_gap_frames: int,
) -> tuple[np.ndarray | None, bool] | object:
    """Resolve the ball position at one frame under the missing-ball policy.

    Args:
        index: Precomputed tracking index.
        frame: Frame number.
        policy: One of :data:`BALL_POLICIES`.
        max_gap_frames: How far ``hold``/``interpolate`` may reach.

    Returns:
        ``(position, was_filled)`` where ``position`` is a ``(2,)`` array or
        ``None`` (meaning "assume zero ball travel time"), or the module-private
        ``_SKIP`` sentinel when this timepoint cannot be used.
    """
    slot = frame - index.frame_min
    x, y = index.ball_x[slot], index.ball_y[slot]
    if np.isfinite(x) and np.isfinite(y):
        return np.array([x, y]), False
    if policy == "none":
        return None, True
    if policy == "skip":
        return _SKIP
    prev = int(index.ball_prev[slot])
    if policy == "interpolate":
        nxt = int(index.ball_next[slot])
        if (
            prev >= 0
            and nxt < index.ball_x.size
            and slot - prev <= max_gap_frames
            and nxt - slot <= max_gap_frames
        ):
            weight = (slot - prev) / (nxt - prev)
            return (
                np.array(
                    [
                        index.ball_x[prev] + weight * (index.ball_x[nxt] - index.ball_x[prev]),
                        index.ball_y[prev] + weight * (index.ball_y[nxt] - index.ball_y[prev]),
                    ]
                ),
                True,
            )
    if prev >= 0 and slot - prev <= max_gap_frames:
        return np.array([index.ball_x[prev], index.ball_y[prev]]), True
    return _SKIP


# --------------------------------------------------------------------------- #
# Scoring one loss
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Sample:
    """One evaluated timepoint inside a counter-press window."""

    offset: float
    share_area: float
    share_epv: float
    epv_winner: float
    ball_filled: bool


@dataclass(frozen=True)
class _FrameWork:
    """Everything one timepoint needs, cut free of the tracking frame.

    A few hundred bytes of player arrays rather than a view into the match's
    millions of rows, so a window can be shipped to another process for a
    fraction of what pickling the whole index would cost.
    """

    offset: float
    winner: PlayerSet
    presser: PlayerSet
    ball: np.ndarray | None
    ball_filled: bool


@dataclass(frozen=True)
class _LossWork:
    """One loss's whole counter-press window, ready to score."""

    attack_direction: int
    weighting: Weighting
    samples: tuple[_FrameWork, ...]


def _evaluate_frame(
    winner: PlayerSet,
    presser: PlayerSet,
    ball: np.ndarray | None,
    grid: PitchGrid,
    pc_params: PitchControlParams,
    epv_weight: np.ndarray,
) -> tuple[float, float, float]:
    """Compute both control shares and the winner's EPV at one instant.

    Args:
        winner: Team treated as in possession -- the one that won the ball.
        presser: The counter-pressing team.
        ball: ``(2,)`` ball position or ``None``.
        grid: Pitch-control grid.
        pc_params: Pitch-control model constants.
        epv_weight: ``(n_y, n_x)`` EPV surface oriented for the winning team.

    Returns:
        ``(share_area, share_epv, epv_winner)``. ``share_*`` are the losing
        team's normalised control shares; ``epv_winner`` is the winning team's
        control-weighted mean EPV.

    Raises:
        ValueError: If either team has no in-frame players.
    """
    result = pitch_control_full(
        winner,
        presser,
        ball_pos=ball,
        grid=grid,
        params=pc_params,
        offsides=False,
    )
    won = result.attacking.astype(np.float64)
    pressed = result.defending.astype(np.float64)
    total = won + pressed

    denom_area = float(total.sum())
    share_area = float(pressed.sum()) / denom_area if denom_area > 0 else np.nan
    denom_epv = float((epv_weight * total).sum())
    share_epv = float((epv_weight * pressed).sum()) / denom_epv if denom_epv > 0 else np.nan
    return share_area, share_epv, float((epv_weight * won).mean())


def _blank_row() -> dict[str, Any]:
    """The metric columns of a loss that could not be scored."""
    return {
        "n_timepoints": 0,
        "n_ball_filled": 0,
        "recovery": np.nan,
        "recovery_area": np.nan,
        "recovery_epv": np.nan,
        "share_t0_area": np.nan,
        "share_t0_epv": np.nan,
        "share_end_area": np.nan,
        "share_end_epv": np.nan,
        "epv_winner_t0": np.nan,
        "epv_winner_mean": np.nan,
        "epv_conceded": np.nan,
        "scored": False,
        "skip_reason": None,
    }


def _prepare_loss(
    loss: dict[str, Any],
    index: _TrackingIndex,
    params: CounterPressParams,
) -> _LossWork | dict[str, Any]:
    """Cut one loss's window out of the tracking frame, or explain why not.

    This is the cheap half of scoring: it resolves the timepoints, the ball
    policy and the two player sets, and decides every skip reason. The
    expensive half -- the pitch-control surfaces -- is :func:`_score_work`,
    which touches nothing but the small arrays this returns. Splitting them is
    what makes ``jobs > 1`` possible without shipping the whole match to each
    worker, and it means the serial and parallel paths run identical code.

    Args:
        loss: One loss row as a dict.
        index: Precomputed tracking index.
        params: Metric knobs.

    Returns:
        A :class:`_LossWork` when the loss can be scored, otherwise the
        finished metric columns of a skipped row.
    """
    if not (np.isfinite(loss["x"]) and np.isfinite(loss["y"])):
        return {**_blank_row(), "skip_reason": "no_loss_location"}

    losing_team = str(loss["losing_team"])
    winning_team = str(loss["winning_team"])
    if losing_team not in _TEAMS or winning_team not in _TEAMS:
        return {**_blank_row(), "skip_reason": "empty_team"}

    f0 = int(loss["frame"])
    period0 = index.period_of(f0)
    max_gap = int(round(params.ball_hold_max_s * index.fps))

    samples: list[_FrameWork] = []
    for offset in params.timepoints:
        frame = f0 + int(round(offset * index.fps))
        if not index.has_frame(frame) or index.period_of(frame) != period0:
            continue
        ball = _ball_at(index, frame, params.ball_policy, max_gap)
        if ball is _SKIP:
            if offset == 0.0:
                return {**_blank_row(), "skip_reason": "no_ball"}
            continue
        position, was_filled = ball  # type: ignore[misc]
        winner = index.player_set(frame, winning_team)
        presser = index.player_set(frame, losing_team)
        # The only ValueError pitch_control_full raises is an empty side, and
        # it raises it before computing anything, so the check belongs here.
        if len(winner) == 0 or len(presser) == 0:
            if offset == 0.0:
                return {**_blank_row(), "skip_reason": "empty_team"}
            continue
        samples.append(_FrameWork(offset, winner, presser, position, was_filled))

    if not samples or samples[0].offset != 0.0:
        return {**_blank_row(), "skip_reason": "window_truncated"}
    if len(samples) < params.min_timepoints:
        return {
            **_blank_row(),
            "n_timepoints": len(samples),
            "skip_reason": "window_truncated",
        }
    return _LossWork(
        attack_direction=int(index.attack_direction.get(winning_team, 1)),
        weighting=params.weighting,
        samples=tuple(samples),
    )


def _score_work(
    work: _LossWork,
    grid: PitchGrid,
    pc_params: PitchControlParams,
    epv_cache: dict[int, np.ndarray],
) -> dict[str, Any]:
    """Turn one prepared window into the metric columns of its row.

    Args:
        work: Prepared window from :func:`_prepare_loss`.
        grid: Pitch-control grid.
        pc_params: Pitch-control model constants.
        epv_cache: Per-direction EPV surfaces, filled in place.

    Returns:
        The metric columns for the loss's row.
    """
    direction = work.attack_direction
    epv_weight = epv_cache.get(direction)
    if epv_weight is None:
        epv_weight = epv_surface(attack_direction=direction, shape=grid.shape)
        epv_cache[direction] = epv_weight

    samples = [
        _Sample(
            item.offset,
            *_evaluate_frame(
                item.winner, item.presser, item.ball, grid, pc_params, epv_weight
            ),
            item.ball_filled,
        )
        for item in work.samples
    ]

    area = np.array([s.share_area for s in samples])
    epv = np.array([s.share_epv for s in samples])
    winner_epv = np.array([s.epv_winner for s in samples])

    recovery_area = float(np.nanmean(area))
    recovery_epv = float(np.nanmean(epv))
    later = winner_epv[1:]
    epv_mean = float(np.nanmean(later)) if later.size else float(winner_epv[0])

    return {
        "n_timepoints": len(samples),
        "n_ball_filled": int(sum(s.ball_filled for s in samples)),
        "recovery": recovery_epv if work.weighting == "epv" else recovery_area,
        "recovery_area": recovery_area,
        "recovery_epv": recovery_epv,
        "share_t0_area": float(area[0]),
        "share_t0_epv": float(epv[0]),
        "share_end_area": float(area[-1]),
        "share_end_epv": float(epv[-1]),
        "epv_winner_t0": float(winner_epv[0]),
        "epv_winner_mean": epv_mean,
        "epv_conceded": epv_mean - float(winner_epv[0]),
        "scored": True,
        "skip_reason": None,
    }


# --------------------------------------------------------------------------- #
# Running the scoring, serially or across processes
# --------------------------------------------------------------------------- #


def default_jobs() -> int:
    """How many workers :func:`score_losses` uses when ``jobs`` is not given.

    ``min(MAX_USEFUL_JOBS, os.cpu_count())``. The cap is measured, not
    defensive: see :data:`MAX_USEFUL_JOBS`.

    Returns:
        A worker count of at least one.
    """
    return max(1, min(MAX_USEFUL_JOBS, os.cpu_count() or 1))


#: Per-worker state, set once by :func:`_worker_init` so the grid, the model
#: constants and the EPV surfaces cross the process boundary once instead of
#: once per chunk.
_WORKER: dict[str, Any] = {}


def _worker_init(grid: PitchGrid, pc_params: PitchControlParams) -> None:
    """Initialise one pool worker.

    Args:
        grid: Pitch-control grid.
        pc_params: Pitch-control model constants.
    """
    _WORKER["grid"] = grid
    _WORKER["pc_params"] = pc_params
    _WORKER["epv_cache"] = {}


def _score_chunk(chunk: list[_LossWork]) -> list[dict[str, Any]]:
    """Score a contiguous run of prepared losses inside a pool worker.

    Args:
        chunk: Prepared windows, in table order.

    Returns:
        One metric-column dict per window, in the same order.
    """
    return [
        _score_work(work, _WORKER["grid"], _WORKER["pc_params"], _WORKER["epv_cache"])
        for work in chunk
    ]


def _chunked(work: list[_LossWork], n_chunks: int) -> list[list[_LossWork]]:
    """Split ``work`` into at most ``n_chunks`` contiguous, order-preserving runs."""
    if n_chunks <= 1 or len(work) <= 1:
        return [work]
    size = -(-len(work) // n_chunks)
    return [work[i : i + size] for i in range(0, len(work), size)]


def _run_scoring(
    work: list[_LossWork],
    grid: PitchGrid,
    pc_params: PitchControlParams,
    jobs: int,
) -> list[dict[str, Any]]:
    """Score every prepared window, in table order, on ``jobs`` processes.

    The parallel path is an optimisation and never a requirement. It is skipped
    for one worker and for a handful of losses, and if the pool cannot be built
    or dies mid-flight the work is redone serially with a warning -- a spawn
    problem on an unfamiliar machine must not be able to break the demo. Order
    is preserved and every worker runs the same :func:`_score_work` on the same
    inputs, so the output is identical whatever ``jobs`` is.

    One caller-side requirement comes with ``jobs > 1``: macOS and Windows start
    workers by SPAWNING, which re-imports the caller's ``__main__`` module, so a
    script that calls this at import time re-runs itself in every worker. Guard
    the entry point with ``if __name__ == "__main__":`` -- the console script
    installed for ``pressing`` already is. A worker that gets here anyway is
    detected below and quietly goes serial rather than recursing.

    Args:
        work: Prepared windows, in table order.
        grid: Pitch-control grid.
        pc_params: Pitch-control model constants.
        jobs: Worker count.

    Returns:
        One metric-column dict per window, in the same order.
    """

    def serial() -> list[dict[str, Any]]:
        cache: dict[int, np.ndarray] = {}
        return [_score_work(item, grid, pc_params, cache) for item in work]

    if jobs <= 1 or len(work) < _MIN_PARALLEL_LOSSES:
        return serial()
    if getattr(multiprocessing.current_process(), "_inheriting", False):
        # A spawning worker is importing its parent's __main__ and has reached
        # this call. Starting a pool from here is what multiprocessing itself
        # refuses to do; go serial instead of recursing.
        return serial()
    # Four chunks per worker: enough to even out a straggler, few enough that
    # the per-chunk pickle overhead stays invisible.
    chunks = _chunked(work, jobs * 4)
    try:
        with ProcessPoolExecutor(
            max_workers=jobs,
            initializer=_worker_init,
            initargs=(grid, pc_params),
        ) as pool:
            return [row for part in pool.map(_score_chunk, chunks) for row in part]
    except (OSError, BrokenProcessPool, RuntimeError) as exc:
        warnings.warn(
            f"parallel scoring on {jobs} workers failed ({exc!r}); "
            "falling back to the serial path",
            RuntimeWarning,
            stacklevel=2,
        )
        return serial()


# --------------------------------------------------------------------------- #
# Event-stream regain
# --------------------------------------------------------------------------- #


def _regain_flags(
    losses: pd.DataFrame,
    events: pd.DataFrame,
    fps: float,
    window_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Did the losing team have the ball again inside the window?

    Uses only the event stream, so it is the interpretable companion to the
    continuous control-based ``recovery``. A regain is the first ball-carrying
    event (:data:`pressing.possession.POSSESSION_TYPES`) by the losing team that
    comes after the ``BALL LOST`` row and starts within the CLOSED window
    ``[t0, t0 + window_s]`` in the same period.

    Args:
        losses: Loss table with ``frame``, ``period``, ``losing_team`` and
            ``event_index``.
        events: Canonical events frame for the same match, in event order.
        fps: Tracking rate.
        window_s: Window length in seconds.

    Returns:
        ``(regained, time_to_regain)``, aligned to ``losses``.
    """
    n = len(losses)
    regained = np.zeros(n, dtype=bool)
    delay = np.full(n, np.nan)
    if n == 0 or events.empty:
        return regained, delay

    ordered = events.reset_index(drop=True)
    start_frame = np.array(
        [int((payload or {}).get("start_frame", -1)) for payload in ordered["extra"]],
        dtype=np.int64,
    )
    types = ordered["type"].to_numpy()
    teams = ordered["team"].to_numpy()
    periods = ordered["period"].to_numpy(dtype=np.int64)
    carries = np.isin(types, list(POSSESSION_TYPES))

    window_frames = int(round(window_s * fps))
    loss_index = losses["event_index"].to_numpy(dtype=np.int64)
    loss_frame = losses["frame"].to_numpy(dtype=np.int64)
    loss_team = losses["losing_team"].to_numpy()
    loss_period = losses["period"].to_numpy(dtype=np.int64)

    for i in range(n):
        f0 = loss_frame[i]
        limit = f0 + window_frames
        for j in range(int(loss_index[i]) + 1, len(ordered)):
            if start_frame[j] > limit:
                break
            if not carries[j] or start_frame[j] < f0:
                continue
            if periods[j] != loss_period[i] or teams[j] != loss_team[i]:
                continue
            regained[i] = True
            delay[i] = (start_frame[j] - f0) / fps
            break
    return regained, delay


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def find_possession_losses(
    tracking: pd.DataFrame,
    events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Locate every open-play possession loss in a match.

    A thin pass-through to :func:`pressing.possession.detect_losses`, which owns
    the definition: the ``BALL LOST`` row's END frame, excluding the single
    half-time whistle, located from the tracked ball with documented fallbacks.

    Args:
        tracking: Canonical tracking frame for the match.
        events: Canonical events frame. Required -- turnovers come from the
            event stream, not from tracking alone.

    Returns:
        One row per loss with :data:`pressing.possession.LOSS_COLUMNS`.

    Raises:
        ValueError: If ``events`` is None.
    """
    if events is None:
        raise ValueError(
            "possession losses are detected from the event stream; pass an events frame"
        )
    return detect_losses(events, tracking)


def score_losses(
    tracking: pd.DataFrame,
    events: pd.DataFrame,
    losses: pd.DataFrame | None = None,
    params: CounterPressParams | None = None,
    grid: PitchGrid | None = None,
    pc_params: PitchControlParams | None = None,
    jobs: int | None = None,
) -> pd.DataFrame:
    """Score every possession loss in a match.

    Args:
        tracking: Canonical tracking frame for the match.
        events: Canonical events frame for the same match.
        losses: Pre-detected losses. Defaults to
            :func:`pressing.possession.detect_losses`.
        params: Metric knobs. Defaults to :class:`CounterPressParams`.
        grid: Pitch-control grid. Defaults to the 50 x 32 demo grid.
        pc_params: Pitch-control model constants.
        jobs: Worker processes to score on. ``None`` means
            :func:`default_jobs`; ``1`` forces the serial path. The result is
            identical either way -- only the wall clock changes.

    Returns:
        A dataframe with :data:`LOSS_SCORE_COLUMNS`, one row per loss. Losses
        that could not be scored keep their row with NaN metrics, ``scored``
        False and a ``skip_reason``. ``DataFrame.attrs`` carries
        ``ball_out_count``, ``counterpress_params``, ``n_skipped``,
        ``weighting`` and ``alt_weighting``.

    Raises:
        ValueError: If ``jobs`` is not a positive integer.
    """
    params = CounterPressParams() if params is None else params
    workers = default_jobs() if jobs is None else int(jobs)
    if workers < 1:
        raise ValueError(f"jobs must be a positive integer or None, got {jobs!r}")
    grid = make_grid() if grid is None else grid
    pc_params = PitchControlParams() if pc_params is None else pc_params
    fps = float(
        params.fps
        if params.fps is not None
        else tracking.attrs.get("fps", events.attrs.get("fps", METRICA_FPS))
    )

    if losses is None:
        losses = detect_losses(events, tracking, fps=fps)
    losses = losses.reset_index(drop=True)

    index = _build_index(tracking, fps)
    labelled = label_loss_phases(
        losses,
        index.attack_direction,
        pitch_length=params.pitch_length,
    )

    prepared = [_prepare_loss(row, index, params) for row in labelled.to_dict("records")]
    work = [item for item in prepared if isinstance(item, _LossWork)]
    computed = iter(_run_scoring(work, grid, pc_params, workers))
    scored = [
        next(computed) if isinstance(item, _LossWork) else item for item in prepared
    ]

    out = labelled.copy()
    out.attrs.update(losses.attrs)
    for column in (
        "n_timepoints",
        "n_ball_filled",
        "recovery",
        "recovery_area",
        "recovery_epv",
        "share_t0_area",
        "share_t0_epv",
        "share_end_area",
        "share_end_epv",
        "epv_winner_t0",
        "epv_winner_mean",
        "epv_conceded",
        "scored",
        "skip_reason",
    ):
        out[column] = [row[column] for row in scored]

    regained, delay = _regain_flags(out, events, fps, params.window_s)
    out["regained_within_5s"] = regained
    out["time_to_regain"] = delay
    if "match_id" not in out.columns:
        out["match_id"] = tracking.attrs.get("match_id")

    out.attrs["fps"] = fps
    out.attrs["attack_direction"] = dict(index.attack_direction)
    out.attrs["counterpress_params"] = params
    out.attrs["weighting"] = params.weighting
    out.attrs["alt_weighting"] = alternate_weighting(params.weighting)
    out.attrs["jobs"] = workers
    out.attrs["n_skipped"] = int((~out["scored"]).sum())
    out.attrs["skip_reasons"] = (
        out.loc[~out["scored"], "skip_reason"].value_counts().to_dict()
        if not out.empty
        else {}
    )
    return out.loc[:, list(LOSS_SCORE_COLUMNS)]


def score_match(
    tracking: pd.DataFrame,
    events: pd.DataFrame | None = None,
    params: CounterPressParams | None = None,
    grid: PitchGrid | None = None,
    pc_params: PitchControlParams | None = None,
    jobs: int | None = None,
) -> pd.DataFrame:
    """Find and score every possession loss in a match.

    Args:
        tracking: Canonical tracking frame for the match.
        events: Canonical events frame used to locate turnovers. Required.
        params: Metric knobs. Defaults to :class:`CounterPressParams`.
        grid: Pitch-control grid. Defaults to the 50 x 32 demo grid.
        pc_params: Pitch-control model constants.
        jobs: Worker processes. See :func:`score_losses`.

    Returns:
        A dataframe with :data:`LOSS_SCORE_COLUMNS`.

    Raises:
        ValueError: If ``events`` is None.
    """
    if events is None:
        raise ValueError(
            "possession losses are detected from the event stream; pass an events frame"
        )
    return score_losses(
        tracking, events, params=params, grid=grid, pc_params=pc_params, jobs=jobs
    )


def counterpress_value(
    tracking: pd.DataFrame,
    loss_t: float,
    losing_team: str,
    window_s: float = RECOVERY_WINDOW_S,
    *,
    params: CounterPressParams | None = None,
    grid: PitchGrid | None = None,
    pc_params: PitchControlParams | None = None,
    ball_xy: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Score one possession loss in isolation.

    The single-loss entry point, mostly useful for tests and for interactive
    inspection of one moment. Whole matches should go through
    :func:`score_losses`, which builds the frame index once instead of per loss.

    Args:
        tracking: Canonical tracking frame covering at least ``window_s``
            seconds after ``loss_t``.
        loss_t: Time of the turnover in seconds.
        losing_team: ``"home"`` or ``"away"``; the team that gave the ball away.
        window_s: Length of the recovery window in seconds. Ignored when
            ``params`` supplies its own timepoints.
        params: Metric knobs. Defaults to :class:`CounterPressParams` with its
            timepoints truncated to ``window_s``.
        grid: Pitch-control grid.
        pc_params: Pitch-control model constants.
        ball_xy: Loss location override. Defaults to the tracked ball at the
            nearest frame to ``loss_t``.

    Returns:
        ``(recovery, epv_conceded)``. Both are NaN if the loss cannot be scored.

    Raises:
        ValueError: If ``losing_team`` is not ``"home"`` or ``"away"``.
    """
    if losing_team not in _TEAMS:
        raise ValueError(f"losing_team must be 'home' or 'away', got {losing_team!r}")
    if params is None:
        params = CounterPressParams(
            window_s=window_s,
            timepoints=tuple(p for p in DEFAULT_TIMEPOINTS if p <= window_s),
        )
    grid = make_grid() if grid is None else grid
    pc_params = PitchControlParams() if pc_params is None else pc_params
    fps = float(params.fps if params.fps is not None else tracking.attrs.get("fps", METRICA_FPS))

    index = _build_index(tracking, fps)
    frame = int(round(loss_t * fps))
    if not index.has_frame(frame):
        candidates = np.arange(index.frame_min, index.frame_max + 1)
        frame = int(candidates[np.argmin(np.abs(candidates - frame))])

    slot = frame - index.frame_min
    if ball_xy is None:
        x, y = float(index.ball_x[slot]), float(index.ball_y[slot])
    else:
        x, y = float(ball_xy[0]), float(ball_xy[1])

    row = {
        "x": x,
        "y": y,
        "frame": frame,
        "losing_team": losing_team,
        "winning_team": _OPPONENT[losing_team],
    }
    prepared = _prepare_loss(row, index, params)
    result = (
        _score_work(prepared, grid, pc_params, {})
        if isinstance(prepared, _LossWork)
        else prepared
    )
    return float(result["recovery"]), float(result["epv_conceded"])


# --------------------------------------------------------------------------- #
# The weighting contract
# --------------------------------------------------------------------------- #


def alternate_weighting(weighting: Weighting) -> Weighting:
    """The weighting that is not ``weighting``.

    Args:
        weighting: ``"area"`` or ``"epv"``.

    Returns:
        The other one.

    Raises:
        ValueError: If ``weighting`` is not a known weighting.
    """
    if weighting == "area":
        return "epv"
    if weighting == "epv":
        return "area"
    raise ValueError(f"weighting must be 'epv' or 'area', got {weighting!r}")


def headline_weighting(losses: pd.DataFrame) -> Weighting:
    """Which weighting the ``recovery`` column of ``losses`` currently carries.

    Read from ``attrs["counterpress_params"]``, which :func:`score_losses` sets
    and :func:`with_weighting` updates. Pandas drops ``attrs`` across some
    operations, so a frame that has lost them is checked against its own
    columns before falling back to :data:`DEFAULT_WEIGHTING`.

    Args:
        losses: Scored losses.

    Returns:
        ``"area"`` or ``"epv"``.
    """
    params = losses.attrs.get("counterpress_params")
    if isinstance(params, CounterPressParams):
        return params.weighting
    stated = losses.attrs.get("weighting")
    if stated in ("area", "epv"):
        return stated  # type: ignore[return-value]
    if not losses.empty and {"recovery", "recovery_epv"} <= set(losses.columns):
        headline = losses["recovery"].to_numpy(dtype=np.float64)
        weighted = losses["recovery_epv"].to_numpy(dtype=np.float64)
        if np.array_equal(headline, weighted, equal_nan=True):
            return "epv"
    return DEFAULT_WEIGHTING


# --------------------------------------------------------------------------- #
# Minutes and player attribution
# --------------------------------------------------------------------------- #


def minutes_played(tracking: pd.DataFrame, fps: float | None = None) -> pd.DataFrame:
    """Minutes each player was actually on the pitch, from tracking presence.

    Not an assumed 90. A player's rows in the canonical frame exist exactly
    while they are on the pitch -- verified against the raw Metrica files, every
    player's valid span is perfectly contiguous, so a missing row means "not
    playing", never "off camera".

    Substitutions overlap by exactly one frame: the outgoing and incoming player
    are both present at the boundary. That is why a span's duration is measured
    as ``(last_frame - first_frame) / fps``, the length of the closed interval,
    rather than ``(last - first + 1)``. Player A's interval ends at the boundary
    frame and player B's begins there, so the pair sum to the true elapsed time
    with no double count.

    One honest caveat about the scale: Metrica's frame clock runs continuously
    across half time -- period 2 starts at the frame after period 1 ends, with
    no gap -- so these are TRACKED minutes over the whole ~96.7 minute match
    clock, not the 90 of a nominal match. Per-90 rates must divide by these
    numbers, not by 90.

    The implementation is one sort and three ``bincount`` calls rather than a
    groupby loop, because iterating the groups of a multi-million row frame
    materialises a dataframe block per player. Measured on Metrica game 1
    (3,190,138 player rows, 28 players), best of three: 1.02 s for the groupby
    version against 0.37 s here, and the two frames compare equal. The sort is
    on a single packed ``player * frame_span + frame`` key, which is 0.157 s
    against 0.318 s for ``lexsort`` plus the gather it needs.

    Args:
        tracking: Canonical tracking frame for the match.
        fps: Tracking rate. Defaults to ``tracking.attrs["fps"]``.

    Returns:
        One row per player with ``team``, ``player_id``, ``jersey``,
        ``first_frame``, ``last_frame``, ``n_frames`` and ``minutes``.
    """
    rate = float(fps if fps is not None else tracking.attrs.get("fps", METRICA_FPS))
    columns = [
        "team",
        "player_id",
        "jersey",
        "first_frame",
        "last_frame",
        "n_frames",
        "minutes",
    ]
    players = tracking.loc[tracking["team"] != "ball", ["team", "player_id", "jersey", "frame"]]
    if players.empty:
        return pd.DataFrame(columns=columns)

    frame = players["frame"].to_numpy(dtype=np.int64)
    jersey = players["jersey"].to_numpy(dtype=np.float64, na_value=np.nan)
    team_code, team_names = pd.factorize(players["team"])
    player_code, player_names = pd.factorize(players["player_id"])
    code, pairs = pd.factorize(team_code.astype(np.int64) * len(player_names) + player_code)
    n_players = len(pairs)

    # One int64 key per row, player in the high part and frame in the low part,
    # so a single np.sort orders the rows by player and then by frame without
    # ever materialising a permutation to gather through.
    frame_min = int(frame.min())
    span = np.int64(int(frame.max()) - frame_min + 1)
    packed = np.sort(code.astype(np.int64) * span + (frame - frame_min))
    key, sorted_frame = packed // span, packed % span + frame_min

    changed = np.empty(key.size, dtype=bool)
    changed[0] = True
    changed[1:] = key[1:] != key[:-1]

    # A run is a stretch of consecutive frames for one player; its length is
    # the CLOSED interval last - first, so a substitution's shared boundary
    # frame is not counted twice. See the docstring.
    run_start = changed | np.concatenate(([False], np.diff(sorted_frame) > 1))
    starts = np.flatnonzero(run_start)
    ends = np.concatenate((starts[1:] - 1, [key.size - 1]))
    seconds = np.bincount(
        key[starts],
        weights=(sorted_frame[ends] - sorted_frame[starts]) / rate,
        minlength=n_players,
    )

    distinct = changed | np.concatenate(([False], np.diff(sorted_frame) != 0))
    n_frames = np.bincount(key, weights=distinct, minlength=n_players).astype(np.int64)

    key_starts = np.flatnonzero(changed)
    key_ends = np.concatenate((key_starts[1:] - 1, [key.size - 1]))
    first_frame = np.zeros(n_players, dtype=np.int64)
    last_frame = np.zeros(n_players, dtype=np.int64)
    first_frame[key[key_starts]] = sorted_frame[key_starts]
    last_frame[key[key_ends]] = sorted_frame[key_ends]

    # Latest assignment wins, so walking the valid rows backwards leaves each
    # player's FIRST non-null jersey in row order -- the same one the groupby
    # version took.
    shirt = np.full(n_players, np.nan)
    valid = np.flatnonzero(np.isfinite(jersey))[::-1]
    shirt[code[valid]] = jersey[valid]

    rows = [
        {
            "team": str(team_names[pair // len(player_names)]),
            "player_id": str(player_names[pair % len(player_names)]),
            "jersey": int(shirt[i]) if np.isfinite(shirt[i]) else pd.NA,
            "first_frame": int(first_frame[i]),
            "last_frame": int(last_frame[i]),
            "n_frames": int(n_frames[i]),
            "minutes": float(seconds[i]) / 60.0,
        }
        for i, pair in enumerate(pairs)
    ]
    return pd.DataFrame(rows).sort_values(["team", "player_id"], ignore_index=True)


def _attribution(
    losses: pd.DataFrame,
    tracking: pd.DataFrame,
    radius_m: float,
) -> pd.DataFrame:
    """Long table of ``(loss_id, player_id)`` for every credited presser.

    Only the few hundred frames a loss actually lands on are touched, so this
    needs no :class:`_TrackingIndex` and does not pay for building one.

    Args:
        losses: Scored losses.
        tracking: Canonical tracking frame for the match.
        radius_m: Inclusive radius in metres.

    Returns:
        Columns ``loss_id``, ``team``, ``player_id``, ``distance_m``, sorted by
        ``(loss_id, player_id)``.
    """
    columns = ["loss_id", "team", "player_id", "distance_m"]
    usable = losses.loc[
        np.isfinite(losses["x"].to_numpy(dtype=np.float64))
        & np.isfinite(losses["y"].to_numpy(dtype=np.float64))
        & losses["losing_team"].isin(_TEAMS).to_numpy(dtype=bool),
        ["loss_id", "frame", "losing_team", "x", "y"],
    ]
    if usable.empty:
        return pd.DataFrame(columns=columns)

    wanted = np.unique(usable["frame"].to_numpy(dtype=np.int64))
    rows = tracking.loc[
        tracking["team"].isin(_TEAMS).to_numpy(dtype=bool)
        & tracking["frame"].isin(wanted).to_numpy(dtype=bool),
        ["frame", "team", "player_id", "x", "y"],
    ]
    if rows.empty:
        return pd.DataFrame(columns=columns)

    joined = usable.merge(rows, on="frame", how="inner", suffixes=("", "_player"))
    joined = joined[joined["team"].to_numpy() == joined["losing_team"].to_numpy()]
    distance = np.hypot(
        joined["x_player"].to_numpy(dtype=np.float64) - joined["x"].to_numpy(dtype=np.float64),
        joined["y_player"].to_numpy(dtype=np.float64) - joined["y"].to_numpy(dtype=np.float64),
    )
    near = np.isfinite(distance) & (distance <= radius_m)
    out = pd.DataFrame(
        {
            "loss_id": joined["loss_id"].to_numpy()[near].astype(np.int64),
            "team": joined["team"].to_numpy()[near].astype(str),
            "player_id": joined["player_id"].to_numpy()[near].astype(str),
            "distance_m": distance[near],
        }
    )
    return out.sort_values(["loss_id", "player_id"], ignore_index=True)


def per_player(
    losses: pd.DataFrame,
    tracking: pd.DataFrame,
    radius_m: float = DEFAULT_RADIUS_M,
    *,
    fps: float | None = None,
) -> pd.DataFrame:
    """Aggregate counter-press performance per player.

    A player is credited with a loss when they are within ``radius_m`` of the
    turnover location at the instant it happens -- inclusive, so exactly
    ``radius_m`` counts -- and they are on the team that LOST the ball. Only
    scored losses contribute to the means; a loss that was skipped still counts
    toward nothing at all.

    ``mean_recovery`` is the headline weighting and ``mean_recovery_alt`` is the
    other one, exactly as in :func:`per_team`. See that function for why the two
    are never the same column under two names.

    Args:
        losses: Scored losses from :func:`score_losses`.
        tracking: Canonical tracking frame for the match.
        radius_m: Proximity threshold in metres.
        fps: Tracking rate. Defaults to ``tracking.attrs["fps"]``.

    Returns:
        One row per player with ``team``, ``player_id``, ``jersey``,
        ``minutes``, ``losses_nearby``, ``mean_distance_m``, ``mean_recovery``,
        ``mean_recovery_alt``, ``epv_conceded``, ``regain_rate``,
        ``losses_per_90``, and per-phase ``losses_<phase>`` /
        ``recovery_<phase>`` splits. ``attrs["weighting"]`` and
        ``attrs["alt_weighting"]`` name the two recovery columns.
    """
    rate = float(fps if fps is not None else tracking.attrs.get("fps", METRICA_FPS))
    weighting = headline_weighting(losses)
    alternate = alternate_weighting(weighting)
    minutes = minutes_played(tracking, rate)
    if losses.empty:
        # Same columns as a populated table, so the report never has to branch.
        blank = minutes.assign(
            losses_nearby=0,
            scored_losses=0,
            mean_distance_m=np.nan,
            mean_recovery=np.nan,
            mean_recovery_alt=np.nan,
            epv_conceded=np.nan,
            regain_rate=np.nan,
            losses_per_90=np.nan,
        )
        for phase in PHASES:
            blank[f"losses_{phase}"] = 0
            blank[f"recovery_{phase}"] = np.nan
        blank.attrs["weighting"] = weighting
        blank.attrs["alt_weighting"] = alternate
        return blank

    credited = _attribution(losses, tracking, radius_m)

    metrics = losses.loc[
        :,
        [
            "loss_id",
            "phase",
            "recovery",
            f"recovery_{alternate}",
            "epv_conceded",
            "regained_within_5s",
            "scored",
        ],
    ]
    joined = credited.merge(metrics, on="loss_id", how="left")

    grouped = joined.groupby(["team", "player_id"], observed=True, sort=True)
    table = grouped.agg(
        losses_nearby=("loss_id", "size"),
        mean_distance_m=("distance_m", "mean"),
        mean_recovery=("recovery", "mean"),
        mean_recovery_alt=(f"recovery_{alternate}", "mean"),
        epv_conceded=("epv_conceded", "mean"),
        regain_rate=("regained_within_5s", "mean"),
        scored_losses=("scored", "sum"),
    ).reset_index()

    for phase in PHASES:
        block = joined[joined["phase"] == phase]
        counts = block.groupby(["team", "player_id"], observed=True).agg(
            **{
                f"losses_{phase}": ("loss_id", "size"),
                f"recovery_{phase}": ("recovery", "mean"),
            }
        )
        table = table.merge(counts.reset_index(), on=["team", "player_id"], how="left")
        table[f"losses_{phase}"] = table[f"losses_{phase}"].fillna(0).astype(int)

    out = minutes.merge(table, on=["team", "player_id"], how="left")
    out["losses_nearby"] = out["losses_nearby"].fillna(0).astype(int)
    out["scored_losses"] = out["scored_losses"].fillna(0).astype(int)
    for phase in PHASES:
        out[f"losses_{phase}"] = out[f"losses_{phase}"].fillna(0).astype(int)
    out["losses_per_90"] = np.where(
        out["minutes"] > 0, out["losses_nearby"] / out["minutes"] * 90.0, np.nan
    )
    out = out.sort_values(["team", "losses_nearby"], ascending=[True, False], ignore_index=True)
    out.attrs["weighting"] = weighting
    out.attrs["alt_weighting"] = alternate
    return out


# --------------------------------------------------------------------------- #
# Team and phase aggregation
# --------------------------------------------------------------------------- #


#: The metrics every aggregate table reports, in order.
#:
#: ``mean_recovery`` is ALWAYS the headline -- the mean of the ``recovery``
#: column, whichever weighting that currently carries -- and
#: ``mean_recovery_alt`` is ALWAYS the other weighting. The two can never be
#: the same number under two headings, which is what a fixed
#: ``mean_recovery_area`` alongside an area-weighted headline used to be. Which
#: is which is stated on the frame: ``attrs["weighting"]`` names the headline
#: and ``attrs["alt_weighting"]`` the alternate, and
#: :data:`WEIGHTING_LABELS` turns either into words for a table heading.
_AGG_COLUMNS: Final[tuple[str, ...]] = (
    "losses",
    "scored_losses",
    "mean_recovery",
    "median_recovery",
    "mean_recovery_alt",
    "mean_epv_conceded",
    "total_epv_conceded",
    "regain_rate",
    "median_time_to_regain",
)


def _aggregate(block: pd.DataFrame, alt_column: str) -> dict[str, Any]:
    """Standard aggregate row over a set of losses.

    Args:
        block: The losses to aggregate.
        alt_column: Which per-loss column feeds ``mean_recovery_alt``.

    Returns:
        One row's worth of :data:`_AGG_COLUMNS`.
    """
    scored = block[block["scored"]]
    has_alt = len(scored) > 0 and alt_column in scored.columns
    return {
        "losses": int(len(block)),
        "scored_losses": int(len(scored)),
        "mean_recovery": float(scored["recovery"].mean()) if len(scored) else np.nan,
        "median_recovery": float(scored["recovery"].median()) if len(scored) else np.nan,
        "mean_recovery_alt": float(scored[alt_column].mean()) if has_alt else np.nan,
        "mean_epv_conceded": float(scored["epv_conceded"].mean()) if len(scored) else np.nan,
        "total_epv_conceded": float(scored["epv_conceded"].sum()) if len(scored) else np.nan,
        "regain_rate": float(block["regained_within_5s"].mean()) if len(block) else np.nan,
        "median_time_to_regain": float(block["time_to_regain"].median())
        if len(block)
        else np.nan,
    }


def _label_weighting(out: pd.DataFrame, losses: pd.DataFrame, weighting: Weighting) -> pd.DataFrame:
    """Carry ``losses``' attrs onto an aggregate and stamp the weighting names."""
    out.attrs.update(losses.attrs)
    out.attrs["weighting"] = weighting
    out.attrs["alt_weighting"] = alternate_weighting(weighting)
    return out


def per_team(losses: pd.DataFrame) -> pd.DataFrame:
    """Aggregate counter-press performance per team.

    ``mean_recovery`` is the headline weighting -- the one the ``recovery``
    column carries, ``"area"`` by default -- and ``mean_recovery_alt`` is the
    other weighting. They are never the same number, and ``attrs["weighting"]``
    / ``attrs["alt_weighting"]`` say which is which, so a table built straight
    off this frame can label both correctly. :func:`with_weighting` swaps them.

    Args:
        losses: Scored losses from :func:`score_losses`.

    Returns:
        One row per team, indexed by the team that LOST the ball, with loss
        counts, mean and median recovery, the alternate weighting, EPV conceded
        and the event-stream regain rate.
    """
    weighting = headline_weighting(losses)
    alt_column = f"recovery_{alternate_weighting(weighting)}"
    if losses.empty:
        return _label_weighting(
            pd.DataFrame(columns=["team", *_AGG_COLUMNS]), losses, weighting
        )
    rows = [
        {"team": str(team), **_aggregate(block, alt_column)}
        for team, block in losses.groupby("losing_team", observed=True, sort=True)
    ]
    return _label_weighting(pd.DataFrame(rows), losses, weighting)


def per_phase(losses: pd.DataFrame) -> pd.DataFrame:
    """Aggregate counter-press performance per team and pitch phase.

    Args:
        losses: Scored losses from :func:`score_losses`.

    Returns:
        One row per ``(team, phase)`` in a stable phase order, with loss counts
        and the same recovery statistics :func:`per_team` reports, including the
        ``mean_recovery`` / ``mean_recovery_alt`` pair. Phases with no losses
        are still emitted, with zero counts, so the report's bar chart always
        has three bars per team.
    """
    weighting = headline_weighting(losses)
    alt_column = f"recovery_{alternate_weighting(weighting)}"
    columns = ["team", "phase", *_AGG_COLUMNS]
    if losses.empty:
        return _label_weighting(pd.DataFrame(columns=columns), losses, weighting)
    rows: list[dict[str, Any]] = []
    for team in sorted(str(t) for t in losses["losing_team"].dropna().unique()):
        side = losses[losses["losing_team"] == team]
        for phase in PHASES:
            block = side[side["phase"] == phase]
            rows.append({"team": team, "phase": phase, **_aggregate(block, alt_column)})
    return _label_weighting(pd.DataFrame(rows).loc[:, columns], losses, weighting)


def team_comparison(losses: pd.DataFrame) -> pd.DataFrame:
    """The head-to-head table the report cover page needs.

    Each metric becomes a row with a column per team plus the home-minus-away
    difference, which is the shape a two-column cover panel wants.

    Args:
        losses: Scored losses from :func:`score_losses`.

    Returns:
        Columns ``metric``, one per team present, and ``difference``
        (home minus away when both are present, otherwise NaN).
    """
    teams = per_team(losses)
    if teams.empty:
        return pd.DataFrame(columns=["metric", "home", "away", "difference"])
    pivot = teams.set_index("team").T
    pivot.index.name = "metric"
    out = pivot.reset_index()
    if "home" in out.columns and "away" in out.columns:
        out["difference"] = out["home"] - out["away"]
    else:
        out["difference"] = np.nan
    out.attrs.update(teams.attrs)
    return out


def summarise_match(losses: pd.DataFrame) -> dict[str, Any]:
    """Headline numbers for the cover page and the CLI summary.

    Args:
        losses: Scored losses from :func:`score_losses`.

    Returns:
        A dict of scalars: loss and skip counts, the separately reported
        ``BALL OUT`` count, per-team mean recovery and regain rate, the
        skip-reason breakdown, and the ``weighting`` every recovery number in
        it was computed under.
    """
    empty = losses.empty or "scored" not in losses.columns
    scored = losses.head(0) if empty else losses[losses["scored"]]
    summary: dict[str, Any] = {
        "match_id": losses.attrs.get("match_id"),
        "weighting": headline_weighting(losses),
        "n_losses": int(len(losses)),
        "n_scored": int(len(scored)),
        "n_skipped": int(len(losses) - len(scored)),
        "skip_reasons": dict(losses.attrs.get("skip_reasons", {})),
        "ball_out_count": losses.attrs.get("ball_out_count"),
        "mean_recovery": float(scored["recovery"].mean()) if len(scored) else np.nan,
        "regain_rate": float(losses["regained_within_5s"].mean()) if len(losses) else np.nan,
    }
    for team, block in (
        losses.groupby("losing_team", observed=True) if not losses.empty else []
    ):
        good = block[block["scored"]]
        summary[f"{team}_losses"] = int(len(block))
        summary[f"{team}_mean_recovery"] = float(good["recovery"].mean()) if len(good) else np.nan
        summary[f"{team}_regain_rate"] = float(block["regained_within_5s"].mean())
    return summary


def with_weighting(losses: pd.DataFrame, weighting: Weighting) -> pd.DataFrame:
    """Re-point the headline ``recovery`` column at the other weighting.

    Both shares are computed for every loss, so switching is free and needs no
    recomputation.

    Args:
        losses: Scored losses from :func:`score_losses`.
        weighting: ``"epv"`` or ``"area"``.

    Returns:
        A copy with ``recovery`` set from the requested weighting.

    Raises:
        ValueError: If ``weighting`` is not a known weighting.
    """
    if weighting not in ("epv", "area"):
        raise ValueError(f"weighting must be 'epv' or 'area', got {weighting!r}")
    out = losses.copy()
    out.attrs.update(losses.attrs)
    out["recovery"] = losses["recovery_epv" if weighting == "epv" else "recovery_area"]
    params = out.attrs.get("counterpress_params")
    if isinstance(params, CounterPressParams):
        out.attrs["counterpress_params"] = replace(params, weighting=weighting)
    out.attrs["weighting"] = weighting
    out.attrs["alt_weighting"] = alternate_weighting(weighting)
    return out
