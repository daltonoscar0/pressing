"""Probabilistic pitch control surfaces.

Implements the Spearman-style model: every player is given a time-to-intercept
for each target location from their current position and velocity, those times
are turned into arrival probabilities with a logistic ramp, and control is
integrated forward along the ball flight to give each team's share of every
cell on the pitch.

Everything here is vectorised over ``(n_players, n_cells)``; the only Python
loop is over the ~250 integration timesteps. That is deliberate. A per-cell
Python loop of the identical algorithm costs 0.5 s for a single 50x32 surface,
and a match has hundreds of possession losses.

Coordinates
-----------
The public API speaks the canonical project frame: metres on a 105 x 68 pitch,
origin bottom-left, ``x`` in ``[0, 105]``, ``y`` in ``[0, 68]``, ``y``
increasing upward, the home team attacking ``+x`` for the whole match.

The Friends of Tracking reference implementation this model is validated
against uses a *centred* frame with ``field_dimen = (106.0, 68.0)``, i.e.
``x`` in ``[-53, 53]`` and ``y`` in ``[-34, 34]``. Converting between the two
is an explicit, tested pair of functions (:func:`to_reference_frame` and
:func:`from_reference_frame`) so the difference can never become an implicit
assumption buried in an index calculation.

Array orientation
-----------------
Surfaces are returned as ``(n_y, n_x)``: axis 0 is ``y``, axis 1 is ``x``, and
**row 0 is the lowest ``y``** (the bottom touchline), matching the reference.
Plotting code must ``np.flipud`` before ``imshow``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from pressing.schema import PITCH_LENGTH, PITCH_WIDTH

__all__ = [
    "MAX_STABLE_INT_DT",
    "REFERENCE_FIELD_DIMEN",
    "SIGMOID_CLIP",
    "PitchControlParams",
    "PitchControlResult",
    "PitchGrid",
    "PlayerSet",
    "check_offsides",
    "control_share",
    "control_surface",
    "from_reference_frame",
    "make_grid",
    "pitch_control",
    "pitch_control_full",
    "reference_grid",
    "to_reference_frame",
]

#: The forward-Euler integration below is only *conditionally* stable. Measured
#: on this model: ``int_dt <= 0.06`` agrees with ``0.04`` at Pearson r >=
#: 0.999998, ``0.08`` diverges to surface values of order 1e42 and ``0.10`` to
#: 1e44 -- with no exception raised, only easily-suppressed RuntimeWarnings.
#: Coarsening the timestep is therefore NOT a speed lever, it is a correctness
#: cliff, and :class:`PitchControlParams` refuses to be built past it.
MAX_STABLE_INT_DT: float = 0.06

#: The logistic argument is clipped to +/- this before ``np.exp``. Required:
#: time-to-intercept reaches ~21 s for far cells at 5 m/s, ``k = pi/sqrt(3)/
#: 0.45 = 4.03``, so the raw argument reaches ~88 -- exactly the float32
#: overflow limit for ``exp``. Without the clip a float32 surface silently
#: becomes all-NaN. exp(60) = 1.1e26, so the clip costs nothing numerically.
SIGMOID_CLIP: float = 60.0

#: Field dimensions assumed by the Friends of Tracking reference code.
REFERENCE_FIELD_DIMEN: tuple[float, float] = (106.0, 68.0)

ClampPolicy = Literal["renormalise", "clip", "raw"]


# --------------------------------------------------------------------------- #
# Frame conversion
# --------------------------------------------------------------------------- #


def to_reference_frame(
    xy: np.ndarray,
    pitch_length: float = PITCH_LENGTH,
    pitch_width: float = PITCH_WIDTH,
    field_dimen: tuple[float, float] = REFERENCE_FIELD_DIMEN,
) -> np.ndarray:
    """Convert canonical bottom-left metres to the reference's centred frame.

    The two frames disagree about the length of a pitch (105 m here, 106 m
    there), so the mapping rescales proportionally rather than merely shifting
    the origin: the goal line at ``x = pitch_length`` lands exactly on
    ``field_dimen[0] / 2``.

    Args:
        xy: ``(..., 2)`` array of canonical positions in metres.
        pitch_length: Canonical pitch length in metres.
        pitch_width: Canonical pitch width in metres.
        field_dimen: ``(length, width)`` assumed by the reference code.

    Returns:
        ``(..., 2)`` array in the reference's centred frame.
    """
    arr = np.asarray(xy, dtype=np.float64)
    out = np.empty_like(arr)
    out[..., 0] = arr[..., 0] / pitch_length * field_dimen[0] - field_dimen[0] / 2.0
    out[..., 1] = arr[..., 1] / pitch_width * field_dimen[1] - field_dimen[1] / 2.0
    return out


def from_reference_frame(
    xy: np.ndarray,
    pitch_length: float = PITCH_LENGTH,
    pitch_width: float = PITCH_WIDTH,
    field_dimen: tuple[float, float] = REFERENCE_FIELD_DIMEN,
) -> np.ndarray:
    """Convert the reference's centred frame to canonical bottom-left metres.

    Exact inverse of :func:`to_reference_frame`.

    Args:
        xy: ``(..., 2)`` array in the reference's centred frame.
        pitch_length: Canonical pitch length in metres.
        pitch_width: Canonical pitch width in metres.
        field_dimen: ``(length, width)`` assumed by the reference code.

    Returns:
        ``(..., 2)`` array of canonical positions in metres.
    """
    arr = np.asarray(xy, dtype=np.float64)
    out = np.empty_like(arr)
    out[..., 0] = (arr[..., 0] + field_dimen[0] / 2.0) / field_dimen[0] * pitch_length
    out[..., 1] = (arr[..., 1] + field_dimen[1] / 2.0) / field_dimen[1] * pitch_width
    return out


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PitchControlParams:
    """Model constants for the pitch control surface.

    Defaults are the Friends of Tracking reference defaults verbatim, so that a
    correlation against the reference tests our code rather than a parameter
    choice.

    Attributes:
        max_player_speed: Player top speed in m/s.
        reaction_time: Seconds a player keeps their current velocity before
            running straight at the target at ``max_player_speed``.
        max_player_accel: Player acceleration in m/s^2. Present for parity with
            the reference's parameter dict; the reaction-time-then-top-speed
            approximation never uses it.
        tti_sigma: Width of the logistic ramp on time-to-intercept, in seconds.
        kappa_def: Defensive advantage multiplier applied to ``lambda_def``.
        lambda_att: Ball-control rate for an attacking player, per second.
        lambda_def: Ball-control rate for an outfield defending player.
        lambda_gk: Ball-control rate for a goalkeeper on the *defending* team.
            The reference applies the goalkeeper boost only to the defending
            lambda; a goalkeeper on the attacking team uses ``lambda_att``.
        average_ball_speed: Assumed ball speed in m/s, used for travel time.
        int_dt: Integration timestep in seconds. Must be <=
            :data:`MAX_STABLE_INT_DT`.
        max_int_time: Integration horizon in seconds.
        model_converge_tol: A cell stops integrating once the two teams'
            probabilities sum to more than ``1 - model_converge_tol``.
        time_to_control_att: Arrival-time gap beyond which an attacker is
            irrelevant, and beyond which the attacking team wins the cell
            outright without integrating.
        time_to_control_def: Same for the defending team.
        dtype: Working floating point type. float32 is the default: measured
            1.2-1.46x faster than float64 at Pearson r = 1.0000000 and
            max abs difference 1.2e-06.
    """

    max_player_speed: float = 5.0
    reaction_time: float = 0.7
    max_player_accel: float = 7.0
    tti_sigma: float = 0.45
    kappa_def: float = 1.0
    lambda_att: float = 4.3
    lambda_def: float = 4.3
    lambda_gk: float = 12.9
    average_ball_speed: float = 15.0
    int_dt: float = 0.04
    max_int_time: float = 10.0
    model_converge_tol: float = 0.01
    time_to_control_att: float = 3.3202550055547313
    time_to_control_def: float = 3.3202550055547313
    dtype: np.dtype = field(default_factory=lambda: np.dtype(np.float32))

    def __post_init__(self) -> None:
        """Validate the timestep and normalise ``dtype``.

        Raises:
            ValueError: If ``int_dt`` exceeds :data:`MAX_STABLE_INT_DT` or is
                not positive, or if ``dtype`` is not a floating point type.
        """
        object.__setattr__(self, "dtype", np.dtype(self.dtype))
        if not np.issubdtype(self.dtype, np.floating):
            raise ValueError(f"dtype must be a floating point type, got {self.dtype!r}")
        if self.int_dt <= 0.0:
            raise ValueError(f"int_dt must be positive, got {self.int_dt!r}")
        # Deliberately a ValueError and not an `assert`: `python -O` strips
        # asserts, and the failure mode this guards is silent divergence to
        # 1e42 that still renders as a plausible-looking heatmap.
        if self.int_dt > MAX_STABLE_INT_DT:
            raise ValueError(
                f"int_dt={self.int_dt} exceeds the stability limit "
                f"{MAX_STABLE_INT_DT}. The forward-Euler scheme used here is "
                "conditionally stable: dt<=0.06 matches dt=0.04 to r>=0.999998, "
                "dt=0.08 diverges to ~1e42 and dt=0.10 to ~1e44 with no "
                "exception raised. int_dt is a stability boundary, not an "
                "accuracy/speed knob -- if you need this faster, coarsen the "
                "grid or subsample time, never dt."
            )

    @staticmethod
    def reference_time_to_control(tti_sigma: float, lambda_: float) -> float:
        """Reference formula ``3*ln(10)*(sqrt(3)*sigma/pi + 1/lambda)``.

        Args:
            tti_sigma: Logistic ramp width in seconds.
            lambda_: Ball-control rate per second.

        Returns:
            The arrival-time gap after which a team's control is settled.
        """
        return 3.0 * np.log(10.0) * (np.sqrt(3.0) * tti_sigma / np.pi + 1.0 / lambda_)

    @classmethod
    def from_reference_formula(cls, **overrides: float) -> PitchControlParams:
        """Build params with ``time_to_control_*`` derived rather than fixed.

        Args:
            **overrides: Any field of :class:`PitchControlParams`.

        Returns:
            Params whose ``time_to_control_att`` / ``time_to_control_def`` are
            recomputed from ``tti_sigma`` and the relevant lambda.
        """
        base = cls(**overrides)
        return cls(
            **{
                **{k: getattr(base, k) for k in base.__dataclass_fields__},
                "time_to_control_att": cls.reference_time_to_control(
                    base.tti_sigma, base.lambda_att
                ),
                "time_to_control_def": cls.reference_time_to_control(
                    base.tti_sigma, base.lambda_def
                ),
            }
        )


# --------------------------------------------------------------------------- #
# Grid
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PitchGrid:
    """A regular grid of cell centres covering the pitch.

    Attributes:
        x: ``(n_x,)`` cell-centre x coordinates in metres.
        y: ``(n_y,)`` cell-centre y coordinates in metres.
        pitch_length: Pitch length in metres.
        pitch_width: Pitch width in metres.
        targets: ``(n_y * n_x, 2)`` cell centres in C order, so that
            ``targets.reshape(n_y, n_x, 2)`` lines up with a surface.
    """

    x: np.ndarray
    y: np.ndarray
    pitch_length: float = PITCH_LENGTH
    pitch_width: float = PITCH_WIDTH
    targets: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Precompute the flattened target array once."""
        gx, gy = np.meshgrid(np.asarray(self.x, float), np.asarray(self.y, float))
        object.__setattr__(self, "targets", np.stack([gx.ravel(), gy.ravel()], axis=1))

    @property
    def shape(self) -> tuple[int, int]:
        """Surface shape as ``(n_y, n_x)``."""
        return (len(self.y), len(self.x))

    @property
    def cell_area(self) -> float:
        """Area of one cell in square metres."""
        return (self.pitch_length / len(self.x)) * (self.pitch_width / len(self.y))

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """``(x_min, x_max, y_min, y_max)`` for ``imshow``, in metres."""
        return (0.0, self.pitch_length, 0.0, self.pitch_width)


def make_grid(
    n_x: int = 50,
    n_y: int | None = None,
    pitch_length: float = PITCH_LENGTH,
    pitch_width: float = PITCH_WIDTH,
) -> PitchGrid:
    """Build a cell-centred grid in the canonical bottom-left metre frame.

    Args:
        n_x: Number of cells along the pitch length.
        n_y: Number of cells across the pitch width. Defaults to
            ``int(n_x * pitch_width / pitch_length)``, which is the reference's
            own rule and gives 32 for the default 50.
        pitch_length: Pitch length in metres.
        pitch_width: Pitch width in metres.

    Returns:
        A :class:`PitchGrid` whose ``shape`` is ``(n_y, n_x)``.

    Raises:
        ValueError: If either cell count is not positive.
    """
    if n_x <= 0:
        raise ValueError(f"n_x must be positive, got {n_x}")
    if n_y is None:
        n_y = int(n_x * pitch_width / pitch_length)
    if n_y <= 0:
        raise ValueError(f"n_y must be positive, got {n_y}")
    dx = pitch_length / n_x
    dy = pitch_width / n_y
    return PitchGrid(
        x=np.arange(n_x) * dx + dx / 2.0,
        y=np.arange(n_y) * dy + dy / 2.0,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
    )


def reference_grid(n_x: int = 50) -> PitchGrid:
    """The Friends of Tracking grid, expressed in the canonical frame.

    The reference evaluates ``xgrid = arange(50)*2.12 - 53 + 1.06`` and
    ``ygrid = arange(32)*2.125 - 34 + 1.0625`` on a 106 x 68 field. Building a
    grid here with ``pitch_length=106`` reproduces exactly those points shifted
    by ``(+53, +34)``, which is precisely :func:`from_reference_frame`. Use
    this, not :func:`make_grid`, when correlating against the reference.

    Args:
        n_x: Number of cells along the pitch length.

    Returns:
        A :class:`PitchGrid` on a 106 x 68 pitch.
    """
    return make_grid(
        n_x=n_x,
        pitch_length=REFERENCE_FIELD_DIMEN[0],
        pitch_width=REFERENCE_FIELD_DIMEN[1],
    )


# --------------------------------------------------------------------------- #
# Players
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PlayerSet:
    """Positions, velocities and goalkeeper flags for one team at one instant.

    Attributes:
        positions: ``(n, 2)`` positions in canonical metres.
        velocities: ``(n, 2)`` velocities in m/s.
        is_gk: ``(n,)`` boolean, True for the goalkeeper.
        ids: ``(n,)`` player identifiers, for diagnostics and offside logging.
    """

    positions: np.ndarray
    velocities: np.ndarray
    is_gk: np.ndarray
    ids: tuple[str, ...] = ()

    def __len__(self) -> int:
        """Number of players in the set."""
        return int(self.positions.shape[0])

    @classmethod
    def from_arrays(
        cls,
        positions: np.ndarray,
        velocities: np.ndarray | None = None,
        is_gk: np.ndarray | None = None,
        ids: object = None,
    ) -> PlayerSet:
        """Build a :class:`PlayerSet`, dropping players who are out of frame.

        Matches the reference's own hygiene: a player with any NaN coordinate
        is excluded entirely, and a player with NaN velocity is kept but
        treated as stationary.

        Args:
            positions: ``(n, 2)`` positions in canonical metres.
            velocities: ``(n, 2)`` velocities in m/s, or None for stationary.
            is_gk: ``(n,)`` boolean goalkeeper flags, or None for no keeper.
            ids: Optional iterable of ``n`` identifiers.

        Returns:
            A :class:`PlayerSet` containing only the in-frame players.

        Raises:
            ValueError: If the array shapes disagree.
        """
        pos = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
        n = pos.shape[0]
        vel = (
            np.zeros_like(pos)
            if velocities is None
            else np.asarray(velocities, dtype=np.float64).reshape(-1, 2)
        )
        gk = np.zeros(n, bool) if is_gk is None else np.asarray(is_gk, bool).reshape(-1)
        pid = tuple(str(i) for i in range(n)) if ids is None else tuple(str(i) for i in ids)
        if vel.shape[0] != n or gk.shape[0] != n or len(pid) != n:
            raise ValueError(
                f"inconsistent player arrays: positions {n}, velocities "
                f"{vel.shape[0]}, is_gk {gk.shape[0]}, ids {len(pid)}"
            )
        vel = np.where(np.isfinite(vel), vel, 0.0)
        keep = np.isfinite(pos).all(axis=1)
        return cls(
            positions=pos[keep],
            velocities=vel[keep],
            is_gk=gk[keep],
            ids=tuple(p for p, k in zip(pid, keep, strict=True) if k),
        )

    def take(self, mask: np.ndarray) -> PlayerSet:
        """Return the subset selected by a boolean mask.

        Args:
            mask: ``(n,)`` boolean array.

        Returns:
            A new :class:`PlayerSet` with the masked players.
        """
        m = np.asarray(mask, bool)
        return PlayerSet(
            positions=self.positions[m],
            velocities=self.velocities[m],
            is_gk=self.is_gk[m],
            ids=tuple(p for p, k in zip(self.ids, m, strict=True) if k),
        )


# --------------------------------------------------------------------------- #
# Offsides
# --------------------------------------------------------------------------- #


def check_offsides(
    players_att: PlayerSet,
    players_def: PlayerSet,
    ball_pos: np.ndarray | None,
    attack_direction: int = 1,
    pitch_length: float = PITCH_LENGTH,
    tol: float = 0.2,
) -> PlayerSet:
    """Remove offside attackers, following the reference's semantics.

    The offside line is ``max(second-deepest defender, ball, halfway line) +
    tol`` measured toward the goal being attacked, and attackers beyond it are
    dropped from the calculation entirely rather than down-weighted.

    This is OFF by default in :func:`pitch_control` -- see that function's
    docstring for why -- and unlike the reference it never raises. The
    reference indexes ``attacking_players[0]`` (IndexError when every attacker
    is out of frame) and asserts that the defending goalkeeper's jersey is in
    the defending list (AssertionError around substitutions, red cards, or a
    keeper who is briefly untracked). Over 250+ possession losses in a real
    match both of those happen; in the tutorial's single handpicked pass
    neither does. Here, any degenerate input simply returns the attackers
    unchanged.

    Args:
        players_att: Attacking (in-possession) team.
        players_def: Defending team.
        ball_pos: ``(2,)`` ball position in canonical metres, or None.
        attack_direction: ``+1`` if the attacking team attacks ``+x``, ``-1``
            otherwise. This replaces the reference's ``sign(defending GK x)``,
            which computes the same quantity less robustly.
        pitch_length: Pitch length in metres, used to locate the halfway line.
        tol: Metres of leniency before a player counts as offside.

    Returns:
        The attacking :class:`PlayerSet` with offside players removed, or the
        input unchanged when the situation is too degenerate to judge.
    """
    if len(players_att) == 0 or len(players_def) < 2:
        return players_att
    half = 1.0 if attack_direction >= 0 else -1.0
    centre = pitch_length / 2.0
    # Signed distance toward the goal being attacked, zero at the halfway line.
    def_x = half * (players_def.positions[:, 0] - centre)
    second_deepest = float(np.sort(def_x)[-2])
    line = max(second_deepest, 0.0)
    if ball_pos is not None:
        ball = np.asarray(ball_pos, dtype=np.float64).reshape(2)
        if np.isfinite(ball[0]):
            line = max(line, float(half * (ball[0] - centre)))
    line += tol
    att_x = half * (players_att.positions[:, 0] - centre)
    return players_att.take(att_x <= line)


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PitchControlResult:
    """Both teams' surfaces plus the raw, unclamped probability checksum.

    Attributes:
        attacking: ``(n_y, n_x)`` control probability for the team in
            possession, after the clamp policy has been applied.
        defending: ``(n_y, n_x)`` control probability for the other team,
            after the clamp policy has been applied.
        checksum: ``(n_y, n_x)`` of ``attacking + defending`` as the integrator
            produced it, *before* any clamping. Values slightly above 1.0 are
            normal -- forward Euler overshoots the convergence tolerance on the
            final step -- and 1.05 has been observed on a real event. Values
            far from 1 mean something is wrong.
        clamp: The policy that was applied.
        converged: ``(n_y, n_x)`` boolean, True where the cell reached the
            convergence tolerance rather than exhausting ``max_int_time``.
    """

    attacking: np.ndarray
    defending: np.ndarray
    checksum: np.ndarray
    clamp: ClampPolicy
    converged: np.ndarray

    @property
    def max_checksum(self) -> float:
        """Largest raw ``attacking + defending`` over the surface."""
        return float(self.checksum.max())

    @property
    def min_checksum(self) -> float:
        """Smallest raw ``attacking + defending`` over the surface."""
        return float(self.checksum.min())

    @property
    def converged_fraction(self) -> float:
        """Fraction of cells that reached the convergence tolerance."""
        return float(self.converged.mean())


def _time_to_intercept(
    players: PlayerSet, targets: np.ndarray, params: PitchControlParams
) -> np.ndarray:
    """Time for each player to reach each target.

    The reference model: hold current velocity for ``reaction_time`` seconds,
    then run in a straight line at ``max_player_speed``. Acceleration is never
    used.

    Args:
        players: One team at one instant.
        targets: ``(n_cells, 2)`` target positions in canonical metres.
        params: Model constants.

    Returns:
        ``(n_players, n_cells)`` intercept times in seconds.
    """
    reaction = players.positions + players.velocities * params.reaction_time
    delta = targets[None, :, :] - reaction[:, None, :]
    dist = np.sqrt(np.einsum("pcd,pcd->pc", delta, delta))
    return params.reaction_time + dist / params.max_player_speed


def pitch_control_full(
    players_att: PlayerSet,
    players_def: PlayerSet,
    ball_pos: np.ndarray | None,
    grid: PitchGrid | None = None,
    params: PitchControlParams | None = None,
    offsides: bool = False,
    attack_direction: int = 1,
    clamp: ClampPolicy = "renormalise",
) -> PitchControlResult:
    """Compute both teams' pitch control surfaces over ``grid``.

    Args:
        players_att: The team currently in possession.
        players_def: The other team.
        ball_pos: ``(2,)`` ball position in canonical metres, or None / NaN to
            assume the ball is already at each target (travel time zero), which
            is what the reference does.
        grid: Target grid. Defaults to ``make_grid()``, a 50 x 32 grid.
        params: Model constants. Defaults to :class:`PitchControlParams`.
        offsides: Whether to drop offside attackers first. Off by default:
            the surface is then a pure function of positions and velocities,
            which keeps it comparable across events and against a reference run
            with ``offsides=False``.
        attack_direction: ``+1`` if ``players_att`` attacks ``+x``. Only used
            when ``offsides`` is True.
        clamp: What to do about raw probabilities that overshoot 1. See
            :func:`pitch_control` for the policy descriptions.

    Returns:
        A :class:`PitchControlResult`.

    Raises:
        ValueError: If either team is empty.
    """
    grid = make_grid() if grid is None else grid
    params = PitchControlParams() if params is None else params
    if len(players_att) == 0 or len(players_def) == 0:
        raise ValueError(
            f"both teams need at least one in-frame player; got "
            f"{len(players_att)} attacking and {len(players_def)} defending"
        )
    if offsides:
        filtered = check_offsides(
            players_att,
            players_def,
            ball_pos,
            attack_direction=attack_direction,
            pitch_length=grid.pitch_length,
        )
        # An offside line ahead of every attacker is physically impossible
        # (the line is at least the halfway line and at least the ball), but
        # keep the original set rather than integrate an empty team if some
        # pathological frame manages it.
        players_att = filtered if len(filtered) else players_att

    dt_type = params.dtype
    targets = grid.targets
    n_cells = targets.shape[0]

    # --- ball travel time, per cell ---------------------------------------- #
    if ball_pos is None:
        btt = np.zeros(n_cells, dtype=dt_type)
    else:
        ball = np.asarray(ball_pos, dtype=np.float64).reshape(2)
        if not np.isfinite(ball).all():
            btt = np.zeros(n_cells, dtype=dt_type)
        else:
            btt = (
                np.linalg.norm(targets - ball, axis=1) / params.average_ball_speed
            ).astype(dt_type)

    # --- time to intercept, per player per cell ---------------------------- #
    tti_a = _time_to_intercept(players_att, targets, params).astype(dt_type)
    tti_d = _time_to_intercept(players_def, targets, params).astype(dt_type)
    tmin_a = tti_a.min(axis=0)
    tmin_d = tti_d.min(axis=0)

    # --- the reference's two pre-integration vetoes ------------------------ #
    # SUBTLETY 1a. These are per cell, and each compares one team's minimum
    # arrival against the *later* of the ball's arrival and the other team's
    # minimum arrival. Dropping the ball term (or using a global minimum
    # instead of a per-cell one) is one of the two ways this model silently
    # decorrelates from the reference.
    def_wins = (tmin_a - np.maximum(btt, tmin_d)) >= params.time_to_control_def
    att_wins = (~def_wins) & ((tmin_d - np.maximum(btt, tmin_a)) >= params.time_to_control_att)

    # SUBTLETY 1b. Each team's players are filtered against ITS OWN per-cell
    # minimum time to intercept, never the global minimum across both teams.
    # Using the global minimum drops most of the slower team and collapses the
    # correlation with the reference to roughly zero.
    keep_a = (tti_a - tmin_a) < params.time_to_control_att
    keep_d = (tti_d - tmin_d) < params.time_to_control_def

    # A dropped player is expressed by pushing their intercept time far into
    # the future: the clipped logistic then evaluates to exp(-60) ~ 1e-26,
    # which saves a multiply by a mask inside the hot loop. 1e6 s is large
    # enough to always saturate the clip and small enough that k * far cannot
    # overflow float32 (unlike, say, finfo.max).
    far = dt_type.type(1.0e6)
    tti_a = np.where(keep_a, tti_a, far)
    tti_d = np.where(keep_d, tti_d, far)

    # The goalkeeper boost applies only on the defending side; the reference
    # gives a goalkeeper in possession the ordinary attacking lambda.
    lam_d = np.where(
        players_def.is_gk, params.lambda_gk, params.lambda_def * params.kappa_def
    ).astype(dt_type)
    lam_a = np.full(len(players_att), params.lambda_att, dtype=dt_type)

    k = dt_type.type(np.pi / np.sqrt(3.0) / params.tti_sigma)
    dt = dt_type.type(params.int_dt)
    tol = dt_type.type(params.model_converge_tol)
    clip = dt_type.type(SIGMOID_CLIP)

    # The reference builds dT_array = arange(btt - dt, btt + max_int_time, dt)
    # and starts its loop at index 1, so the first time actually evaluated is
    # exactly btt and there are 250 steps at the default dt.
    offsets = np.arange(-params.int_dt, params.max_int_time, params.int_dt, dtype=dt_type)[1:]

    ppcf_a = att_wins.astype(dt_type)
    ppcf_d = def_wins.astype(dt_type)

    # Working set. Cells drop out as they converge, so the arrays inside the
    # loop shrink; `idx` maps a working slot back to its cell in the full grid.
    # Measured on a typical frame: 130 of the 250 steps run at all, and the
    # summed active-cell count is 71.6k against 208k for a fixed-width loop.
    idx = np.arange(n_cells)
    w_kta = k * tti_a
    w_ktd = k * tti_d
    w_btt = btt.copy()
    w_a = ppcf_a.copy()
    w_d = ppcf_d.copy()
    scratch_a = np.empty_like(w_kta)
    scratch_d = np.empty_like(w_ktd)

    for offset in offsets:
        remaining = 1.0 - w_a - w_d
        # SUBTLETY 2. Convergence is applied PER CELL. A converged cell must
        # stop accumulating -- either by being dropped from the working set or
        # by having its remaining probability zeroed. A global "break once
        # max(remaining) < tol" instead lets already-converged cells keep
        # integrating, and the explicit-Euler scheme then diverges.
        active = remaining > tol
        n_active = int(active.sum())
        if n_active == 0:
            break
        if n_active * 2 < active.size and active.size > 64:
            # Flush every current working cell (converged ones included) back
            # to the full grid, then shrink to just the survivors.
            ppcf_a[idx] = w_a
            ppcf_d[idx] = w_d
            idx = idx[active]
            w_kta = w_kta[:, active]
            w_ktd = w_ktd[:, active]
            w_btt = w_btt[active]
            w_a = w_a[active]
            w_d = w_d[active]
            remaining = remaining[active]
            scratch_a = np.empty_like(w_kta)
            scratch_d = np.empty_like(w_ktd)
        else:
            remaining = remaining * active

        kt = k * (w_btt + offset)
        np.subtract(w_kta, kt, out=scratch_a)
        np.clip(scratch_a, -clip, clip, out=scratch_a)
        np.exp(scratch_a, out=scratch_a)
        scratch_a += 1.0
        np.reciprocal(scratch_a, out=scratch_a)

        np.subtract(w_ktd, kt, out=scratch_d)
        np.clip(scratch_d, -clip, clip, out=scratch_d)
        np.exp(scratch_d, out=scratch_d)
        scratch_d += 1.0
        np.reciprocal(scratch_d, out=scratch_d)

        # lam @ scratch is a GEMV: the lambda-weighted sum over players with no
        # (n_players, n_cells) temporary.
        w_a += remaining * (lam_a @ scratch_a) * dt
        w_d += remaining * (lam_d @ scratch_d) * dt

    ppcf_a[idx] = w_a
    ppcf_d[idx] = w_d
    # Veto-decided cells sum to exactly 1, so this single test also marks them.
    converged = (1.0 - ppcf_a - ppcf_d) <= tol

    checksum = (ppcf_a + ppcf_d).reshape(grid.shape)
    att = ppcf_a.reshape(grid.shape)
    dfd = ppcf_d.reshape(grid.shape)
    att, dfd = _apply_clamp(att, dfd, clamp)
    return PitchControlResult(
        attacking=att,
        defending=dfd,
        checksum=checksum,
        clamp=clamp,
        converged=converged.reshape(grid.shape),
    )


def _apply_clamp(
    att: np.ndarray, dfd: np.ndarray, clamp: ClampPolicy
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the overshoot policy to a pair of raw surfaces.

    Args:
        att: Raw attacking surface.
        dfd: Raw defending surface.
        clamp: Policy name.

    Returns:
        ``(attacking, defending)`` after the policy.

    Raises:
        ValueError: If ``clamp`` is not a known policy.
    """
    if clamp == "raw":
        return att, dfd
    if clamp == "clip":
        return np.clip(att, 0.0, 1.0), np.clip(dfd, 0.0, 1.0)
    if clamp == "renormalise":
        total = att + dfd
        over = total > 1.0
        scale = np.where(over, 1.0 / np.where(over, total, 1.0), 1.0).astype(att.dtype)
        return np.clip(att * scale, 0.0, 1.0), np.clip(dfd * scale, 0.0, 1.0)
    raise ValueError(f"unknown clamp policy {clamp!r}; expected renormalise, clip or raw")


def pitch_control(
    players_att: PlayerSet,
    players_def: PlayerSet,
    ball_pos: np.ndarray | None = None,
    grid: PitchGrid | None = None,
    params: PitchControlParams | None = None,
    offsides: bool = False,
    attack_direction: int = 1,
    clamp: ClampPolicy = "renormalise",
) -> np.ndarray:
    """Probability that the team in possession controls each cell.

    This is the headline entry point. The returned array is ``(n_y, n_x)``
    with row 0 at the bottom touchline.

    Clamp policies. Forward Euler takes one whole step past the convergence
    tolerance, so the two teams' raw probabilities can sum to slightly more
    than 1 -- 1.0418 has been measured on a real possession loss. Downstream
    code must never be handed a "probability" above 1, so:

    * ``"renormalise"`` (default) divides both surfaces by their sum wherever
      that sum exceeds 1, preserving the ratio between the teams, then clips to
      ``[0, 1]``. Cells that under-shoot (did not converge within
      ``max_int_time``) are left alone, since there the missing mass genuinely
      is "nobody controls this".
    * ``"clip"`` clips each surface independently to ``[0, 1]``, which can
      leave the pair summing above 1.
    * ``"raw"`` returns the integrator's output untouched. Use it only for
      numerical diagnostics.

    Use :func:`pitch_control_full` if you need the defending surface or the
    raw checksum.

    Args:
        players_att: The team currently in possession.
        players_def: The other team.
        ball_pos: ``(2,)`` ball position in canonical metres, or None.
        grid: Target grid. Defaults to a 50 x 32 grid on 105 x 68 m.
        params: Model constants. Defaults to :class:`PitchControlParams`.
        offsides: Whether to drop offside attackers first. Off by default.
        attack_direction: ``+1`` if ``players_att`` attacks ``+x``.
        clamp: Overshoot policy, as described above.

    Returns:
        ``(n_y, n_x)`` array of control probabilities in ``[0, 1]``.
    """
    return pitch_control_full(
        players_att,
        players_def,
        ball_pos,
        grid=grid,
        params=params,
        offsides=offsides,
        attack_direction=attack_direction,
        clamp=clamp,
    ).attacking


# --------------------------------------------------------------------------- #
# Dataframe-facing convenience wrappers
# --------------------------------------------------------------------------- #


def players_from_tracking(
    tracking_frame: pd.DataFrame, team: str, gk_jersey: str | None = None
) -> PlayerSet:
    """Build a :class:`PlayerSet` from canonical tracking rows for one instant.

    Args:
        tracking_frame: Rows of a canonical tracking frame for a single
            instant, with ``team``, ``jersey``, ``x``, ``y``, ``vx``, ``vy``.
        team: ``"home"`` or ``"away"``.
        gk_jersey: Jersey number of that team's goalkeeper, or None if unknown.

    Returns:
        The team's in-frame players.
    """
    rows = tracking_frame[tracking_frame["team"] == team]
    is_gk = (
        np.zeros(len(rows), bool)
        if gk_jersey is None
        else (rows["jersey"].astype(str).to_numpy() == str(gk_jersey))
    )
    return PlayerSet.from_arrays(
        positions=rows[["x", "y"]].to_numpy(dtype=float),
        velocities=rows[["vx", "vy"]].to_numpy(dtype=float),
        is_gk=is_gk,
        ids=rows["jersey"].astype(str).tolist(),
    )


def control_surface(
    tracking_frame: pd.DataFrame,
    attacking_team: str,
    params: PitchControlParams | None = None,
    grid: PitchGrid | None = None,
    gk_jerseys: tuple[str | None, str | None] = (None, None),
    offsides: bool = False,
    attack_direction: int = 1,
) -> np.ndarray:
    """Compute the attacking team's pitch control from a canonical frame.

    A thin dataframe adapter over :func:`pitch_control`.

    Args:
        tracking_frame: Rows of a canonical tracking frame for a single
            instant, including the ``team == "ball"`` row.
        attacking_team: ``"home"`` or ``"away"``; the team whose control is
            returned.
        params: Model constants. Defaults to :class:`PitchControlParams`.
        grid: Target grid. Defaults to a 50 x 32 grid.
        gk_jerseys: ``(home_gk_jersey, away_gk_jersey)``.
        offsides: Whether to drop offside attackers first.
        attack_direction: ``+1`` if ``attacking_team`` attacks ``+x``.

    Returns:
        ``(n_y, n_x)`` array of control probabilities in ``[0, 1]`` for the
        attacking team.

    Raises:
        ValueError: If ``attacking_team`` is not ``"home"`` or ``"away"``.
    """
    if attacking_team not in ("home", "away"):
        raise ValueError(f"attacking_team must be 'home' or 'away', got {attacking_team!r}")
    defending_team = "away" if attacking_team == "home" else "home"
    gk = {"home": gk_jerseys[0], "away": gk_jerseys[1]}
    ball_rows = tracking_frame[tracking_frame["team"] == "ball"]
    ball = None if ball_rows.empty else ball_rows[["x", "y"]].to_numpy(dtype=float)[0]
    return pitch_control(
        players_from_tracking(tracking_frame, attacking_team, gk[attacking_team]),
        players_from_tracking(tracking_frame, defending_team, gk[defending_team]),
        ball_pos=ball,
        grid=grid,
        params=params,
        offsides=offsides,
        attack_direction=attack_direction,
    )


def control_share(
    surface: np.ndarray,
    region: tuple[float, float, float, float] | None = None,
    grid: PitchGrid | None = None,
) -> float:
    """Mean control across the pitch or a rectangular region.

    Args:
        surface: Control surface from :func:`pitch_control`.
        region: Optional ``(x_min, y_min, x_max, y_max)`` box in metres. When
            omitted the whole pitch is used.
        grid: The grid the surface was computed on. Defaults to a grid built to
            match ``surface.shape`` on the canonical 105 x 68 pitch.

    Returns:
        Mean control probability in ``[0, 1]``.

    Raises:
        ValueError: If the grid shape does not match the surface, or the region
            selects no cells.
    """
    n_y, n_x = surface.shape
    grid = make_grid(n_x=n_x, n_y=n_y) if grid is None else grid
    if grid.shape != surface.shape:
        raise ValueError(f"grid shape {grid.shape} does not match surface {surface.shape}")
    if region is None:
        return float(surface.mean())
    x0, y0, x1, y1 = region
    mask_x = (grid.x >= min(x0, x1)) & (grid.x <= max(x0, x1))
    mask_y = (grid.y >= min(y0, y1)) & (grid.y <= max(y0, y1))
    if not mask_x.any() or not mask_y.any():
        raise ValueError(f"region {region} selects no grid cells")
    return float(surface[np.ix_(mask_y, mask_x)].mean())
