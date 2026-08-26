"""Tests for the vectorised pitch control model.

The scenarios here are synthetic and have answers you can reason about without
running the code: a lone attacker owns the ground he is standing on, two
mirrored players split the midpoint exactly, and a goalkeeper's boosted lambda
has to show up as extra control in his own box.

The correlation against the Friends of Tracking reference lives in
``validation/`` rather than here, because it needs the Metrica sample data,
which is not committed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pressing.pitchcontrol import (
    MAX_STABLE_INT_DT,
    REFERENCE_FIELD_DIMEN,
    PitchControlParams,
    PitchGrid,
    PlayerSet,
    check_offsides,
    control_share,
    control_surface,
    from_reference_frame,
    make_grid,
    pitch_control,
    pitch_control_full,
    players_from_tracking,
    reference_grid,
    to_reference_frame,
)

PITCH_L = 105.0
PITCH_W = 68.0
CENTRE = np.array([PITCH_L / 2.0, PITCH_W / 2.0])


def _single(pos, vel=(0.0, 0.0), gk: bool = False) -> PlayerSet:
    """One stationary player, for scenarios where only geometry matters."""
    return PlayerSet.from_arrays([list(pos)], [list(vel)], [gk])


def _cell(grid: PitchGrid, x: float, y: float) -> tuple[int, int]:
    """Index of the grid cell whose centre is nearest ``(x, y)``."""
    return int(np.argmin(np.abs(grid.y - y))), int(np.argmin(np.abs(grid.x - x)))


# --------------------------------------------------------------------------- #
# Grid and frame conversion
# --------------------------------------------------------------------------- #


def test_default_grid_is_32_by_50_cell_centred() -> None:
    grid = make_grid()
    assert grid.shape == (32, 50)
    assert grid.x[0] == pytest.approx(PITCH_L / 50 / 2)
    assert grid.x[-1] == pytest.approx(PITCH_L - PITCH_L / 50 / 2)
    assert grid.y[0] == pytest.approx(PITCH_W / 32 / 2)
    assert grid.y[-1] == pytest.approx(PITCH_W - PITCH_W / 32 / 2)
    # targets are C-ordered so targets.reshape(n_y, n_x, 2) lines up with a surface
    assert grid.targets.shape == (32 * 50, 2)
    assert grid.targets.reshape(32, 50, 2)[3, 7].tolist() == [grid.x[7], grid.y[3]]


def test_grid_rejects_nonsense_cell_counts() -> None:
    with pytest.raises(ValueError, match="n_x must be positive"):
        make_grid(n_x=0)
    with pytest.raises(ValueError, match="n_y must be positive"):
        make_grid(n_x=10, n_y=-1)


def test_reference_grid_reproduces_the_reference_cell_centres() -> None:
    """The reference evaluates xgrid = arange(50)*2.12 - 53 + 1.06, ygrid likewise."""
    ref_x = np.arange(50) * 2.12 - 53.0 + 2.12 / 2
    ref_y = np.arange(32) * 2.125 - 34.0 + 2.125 / 2
    grid = reference_grid(50)
    ours = to_reference_frame(
        grid.targets, REFERENCE_FIELD_DIMEN[0], REFERENCE_FIELD_DIMEN[1]
    ).reshape(32, 50, 2)
    np.testing.assert_allclose(ours[0, :, 0], ref_x, atol=1e-12)
    np.testing.assert_allclose(ours[:, 0, 1], ref_y, atol=1e-12)


def test_frame_conversion_endpoints_and_roundtrip() -> None:
    corners = np.array([[0.0, 0.0], [PITCH_L, PITCH_W], [PITCH_L / 2, PITCH_W / 2]])
    ref = to_reference_frame(corners)
    np.testing.assert_allclose(ref, [[-53.0, -34.0], [53.0, 34.0], [0.0, 0.0]], atol=1e-12)
    np.testing.assert_allclose(from_reference_frame(ref), corners, atol=1e-12)


def test_frame_conversion_rescales_x_rather_than_only_shifting() -> None:
    """105 m here vs 106 m there: a proportional rescale, not a bare offset."""
    assert to_reference_frame(np.array([84.0, 34.0]))[0] == pytest.approx(
        84.0 / 105.0 * 106.0 - 53.0
    )


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #


def test_reference_defaults_match_the_reference_verbatim() -> None:
    p = PitchControlParams()
    assert (p.max_player_speed, p.reaction_time, p.tti_sigma) == (5.0, 0.7, 0.45)
    assert (p.lambda_att, p.lambda_def, p.lambda_gk) == (4.3, 4.3, 12.9)
    assert (p.average_ball_speed, p.int_dt, p.max_int_time) == (15.0, 0.04, 10.0)
    assert p.model_converge_tol == 0.01
    assert p.time_to_control_att == pytest.approx(3.3202550055547313, abs=0)
    assert p.time_to_control_def == pytest.approx(3.3202550055547313, abs=0)
    assert p.dtype == np.dtype(np.float32)


def test_time_to_control_formula_reproduces_the_constant() -> None:
    got = PitchControlParams.reference_time_to_control(0.45, 4.3)
    assert got == pytest.approx(3.3202550055547313, rel=1e-15)
    derived = PitchControlParams.from_reference_formula()
    assert derived.time_to_control_att == pytest.approx(3.3202550055547313, rel=1e-15)


def test_int_dt_above_the_stability_boundary_is_rejected() -> None:
    """dt is an accuracy guard, not a stability boundary: the integrator does
    not diverge at a coarse timestep, it overshoots by one Euler step and the
    renormalise clamp then divides the overshoot back out so nothing downstream
    can tell. See MAX_STABLE_INT_DT in pitchcontrol.py for the measured numbers.
    """
    PitchControlParams(int_dt=MAX_STABLE_INT_DT)  # exactly on the boundary is allowed
    for bad in (0.061, 0.08, 0.1, 0.2):
        with pytest.raises(ValueError, match="stability limit"):
            PitchControlParams(int_dt=bad)
    with pytest.raises(ValueError, match="must be positive"):
        PitchControlParams(int_dt=0.0)


def test_params_reject_a_non_float_dtype() -> None:
    with pytest.raises(ValueError, match="floating point"):
        PitchControlParams(dtype=np.int32)


# --------------------------------------------------------------------------- #
# PlayerSet hygiene
# --------------------------------------------------------------------------- #


def test_player_set_drops_out_of_frame_players_and_zeroes_nan_velocity() -> None:
    ps = PlayerSet.from_arrays(
        positions=[[1.0, 2.0], [np.nan, 5.0], [7.0, 8.0]],
        velocities=[[np.nan, 1.0], [0.0, 0.0], [2.0, 3.0]],
        is_gk=[True, False, False],
        ids=["1", "2", "3"],
    )
    assert len(ps) == 2
    assert ps.ids == ("1", "3")
    np.testing.assert_array_equal(ps.velocities[0], [0.0, 1.0])
    assert ps.is_gk.tolist() == [True, False]


def test_player_set_rejects_ragged_input() -> None:
    with pytest.raises(ValueError, match="inconsistent player arrays"):
        PlayerSet.from_arrays([[1.0, 2.0], [3.0, 4.0]], is_gk=[True])


# --------------------------------------------------------------------------- #
# Scenarios with known answers
# --------------------------------------------------------------------------- #


def test_lone_attacker_on_the_ball_controls_his_own_cell() -> None:
    grid = make_grid()
    att = _single(CENTRE)
    dfn = _single([2.0, PITCH_W / 2], gk=True)
    surface = pitch_control(att, dfn, ball_pos=CENTRE, grid=grid)
    iy, ix = _cell(grid, CENTRE[0], CENTRE[1])
    assert surface[iy, ix] > 0.99
    # and he does not control the cell in the keeper's lap
    iy_far, ix_far = _cell(grid, 2.0, PITCH_W / 2)
    assert surface[iy_far, ix_far] < 0.01


def test_two_mirrored_players_split_the_midpoint() -> None:
    """Identical players equidistant from a cell must give it 0.5 each.

    A 105-cell grid puts a cell centre exactly on the halfway line; the default
    50-cell grid does not, and asking for "the midpoint" there really asks
    about a cell 1.05 m off centre.
    """
    grid = make_grid(n_x=105, n_y=68)
    ix = int(np.argmin(np.abs(grid.x - PITCH_L / 2)))
    assert grid.x[ix] == pytest.approx(PITCH_L / 2)
    att = _single([PITCH_L / 2 - 10.0, PITCH_W / 2])
    dfn = _single([PITCH_L / 2 + 10.0, PITCH_W / 2])
    res = pitch_control_full(att, dfn, ball_pos=CENTRE, grid=grid, clamp="raw")
    iy = int(np.argmin(np.abs(grid.y - PITCH_W / 2)))
    assert res.attacking[iy, ix] == pytest.approx(0.5, abs=0.01)
    assert res.defending[iy, ix] == pytest.approx(0.5, abs=0.01)
    # the whole halfway line is a tie, not just the centre spot
    assert res.attacking[:, ix] == pytest.approx(0.5, abs=0.01)


def test_mirroring_the_scene_mirrors_the_surface() -> None:
    grid = make_grid()
    att_pos = np.array([[40.0, 20.0], [55.0, 50.0]])
    def_pos = np.array([[70.0, 30.0], [88.0, 44.0]])
    ball = np.array([48.0, 26.0])

    def mirror(p: np.ndarray) -> np.ndarray:
        return np.column_stack([PITCH_L - p[:, 0], PITCH_W - p[:, 1]])

    a = pitch_control(
        PlayerSet.from_arrays(att_pos), PlayerSet.from_arrays(def_pos), ball, grid=grid
    )
    b = pitch_control(
        PlayerSet.from_arrays(mirror(att_pos)),
        PlayerSet.from_arrays(mirror(def_pos)),
        np.array([PITCH_L - ball[0], PITCH_W - ball[1]]),
        grid=grid,
    )
    np.testing.assert_allclose(a, np.flipud(np.fliplr(b)), atol=2e-6)


def test_goalkeeper_lambda_boost_shifts_control_in_his_own_box() -> None:
    """Same geometry, keeper flag on or off: lambda 12.9 vs 4.3 must show up."""
    grid = make_grid()
    keeper_pos = [6.0, PITCH_W / 2]
    att = _single([22.0, PITCH_W / 2])
    with_gk = _single(keeper_pos, gk=True)
    without_gk = _single(keeper_pos, gk=False)
    ball = np.array([22.0, PITCH_W / 2])
    boosted = pitch_control(att, with_gk, ball, grid=grid)
    plain = pitch_control(att, without_gk, ball, grid=grid)
    # inside the six-yard area in front of the keeper
    box = np.ix_(
        (grid.y > PITCH_W / 2 - 9) & (grid.y < PITCH_W / 2 + 9), (grid.x > 5) & (grid.x < 16)
    )
    assert boosted[box].mean() < plain[box].mean() - 0.02
    assert (boosted <= plain + 1e-6).all()


def test_a_faster_arrival_from_velocity_beats_a_closer_standing_player() -> None:
    """tti uses position + velocity*reaction_time, so momentum matters."""
    grid = make_grid()
    target = np.array([70.0, 34.0])
    att = _single([58.0, 34.0], vel=(5.0, 0.0))
    dfn = _single([56.0, 34.0], vel=(-5.0, 0.0))
    surface = pitch_control(att, dfn, ball_pos=target, grid=grid)
    iy, ix = _cell(grid, target[0], target[1])
    assert surface[iy, ix] > 0.9


# --------------------------------------------------------------------------- #
# Numerical health
# --------------------------------------------------------------------------- #


def _busy_scene() -> tuple[PlayerSet, PlayerSet, np.ndarray]:
    rng = np.random.default_rng(7)
    ap = np.column_stack([rng.uniform(5, 100, 11), rng.uniform(3, 65, 11)])
    dp = np.column_stack([rng.uniform(5, 100, 11), rng.uniform(3, 65, 11)])
    gk = np.zeros(11, bool)
    gk[0] = True
    return (
        PlayerSet.from_arrays(ap, rng.normal(0, 2, (11, 2)), np.zeros(11, bool)),
        PlayerSet.from_arrays(dp, rng.normal(0, 2, (11, 2)), gk),
        np.array([52.0, 30.0]),
    )


def test_surface_is_a_probability_everywhere_and_sums_sensibly() -> None:
    grid = make_grid()
    att, dfn, ball = _busy_scene()
    res = pitch_control_full(att, dfn, ball, grid=grid)
    assert res.attacking.shape == grid.shape == res.defending.shape
    assert np.isfinite(res.attacking).all()
    assert (res.attacking >= 0.0).all() and (res.attacking <= 1.0).all()
    assert (res.defending >= 0.0).all() and (res.defending <= 1.0).all()
    assert (res.attacking + res.defending <= 1.0 + 1e-6).all()
    # with 22 players on the pitch every cell should converge
    assert res.converged_fraction == 1.0
    assert res.min_checksum >= 1.0 - PitchControlParams().model_converge_tol - 1e-6


def _overshoot_scene() -> tuple[PlayerSet, PlayerSet, np.ndarray]:
    """A scene whose raw probabilities genuinely blow past 1.

    Both teams packed around x = 95 with the ball back on x = 5. For cells
    inside the pack the ball takes ~6 s to arrive while every player is
    already standing there, so the very first Euler step sees every arrival
    probability at 1 and adds ``sum(lambda) * dt = 4.1`` in one go. This is not
    a bug -- the reference does exactly the same, and its own sanity check
    ``assert 1 - checksum < tol`` is one-sided and passes -- but it is why
    downstream code must never see the raw numbers.
    """
    rng = np.random.default_rng(11)
    gk = np.zeros(11, bool)
    gk[0] = True
    return (
        PlayerSet.from_arrays(np.array([95.0, 34.0]) + rng.normal(0, 3, (11, 2))),
        PlayerSet.from_arrays(np.array([94.0, 34.0]) + rng.normal(0, 3, (11, 2)), is_gk=gk),
        np.array([5.0, 34.0]),
    )


def test_clamp_policies_do_what_they_say() -> None:
    grid = make_grid()
    att, dfn, ball = _overshoot_scene()
    raw = pitch_control_full(att, dfn, ball, grid=grid, clamp="raw")
    # forward Euler takes one whole step past the tolerance, so the raw pair
    # genuinely overshoots 1; that is the reason a clamp policy exists at all
    assert raw.max_checksum > 1.5
    assert raw.attacking.max() > 1.0
    renorm = pitch_control_full(att, dfn, ball, grid=grid, clamp="renormalise")
    assert (renorm.attacking + renorm.defending <= 1.0 + 1e-6).all()
    # renormalising preserves the ratio between the teams
    over = raw.checksum > 1.0
    np.testing.assert_allclose(
        renorm.attacking[over] / (renorm.attacking + renorm.defending)[over],
        raw.attacking[over] / raw.checksum[over],
        atol=1e-5,
    )
    clipped = pitch_control_full(att, dfn, ball, grid=grid, clamp="clip")
    assert (clipped.attacking <= 1.0).all()
    # the raw checksum is reported unclamped whatever the policy
    np.testing.assert_allclose(renorm.checksum, raw.checksum)
    with pytest.raises(ValueError, match="unknown clamp policy"):
        pitch_control_full(att, dfn, ball, grid=grid, clamp="squish")  # type: ignore[arg-type]


def test_cells_that_never_converge_are_reported_as_such() -> None:
    """Nobody near a cell and a slow ball: 10 s of integration is not enough.

    The reference prints "Integration failed to converge" for these; we report
    them as a boolean surface instead so a caller can weight or mask them.
    """
    grid = make_grid()
    att, dfn, ball = _overshoot_scene()
    res = pitch_control_full(att, dfn, ball, grid=grid, clamp="raw")
    assert 0.0 < res.converged_fraction < 1.0
    assert res.min_checksum < 0.5
    # unconverged cells are the ones far from both packs
    assert res.converged[:, -5:].mean() > res.converged[:, :5].mean()


def test_float32_matches_float64() -> None:
    grid = make_grid()
    att, dfn, ball = _busy_scene()
    a32 = pitch_control(att, dfn, ball, grid=grid, params=PitchControlParams(dtype=np.float32))
    a64 = pitch_control(att, dfn, ball, grid=grid, params=PitchControlParams(dtype=np.float64))
    assert a32.dtype == np.float32
    assert a64.dtype == np.float64
    assert np.abs(a32 - a64).max() < 1e-4
    assert np.corrcoef(a32.ravel(), a64.ravel())[0, 1] > 1 - 1e-9


def test_float32_does_not_overflow_on_a_far_cell_configuration() -> None:
    """Two players in one corner: cells across the pitch have tti ~ 25 s.

    k = pi/sqrt(3)/0.45 = 4.03, so the unclipped logistic argument reaches ~90,
    past float32's exp limit of 88. Without the clip the whole surface is NaN.
    """
    grid = make_grid()
    att = _single([0.5, 0.5])
    dfn = _single([1.0, 0.5])
    res = pitch_control_full(att, dfn, np.array([0.5, 0.5]), grid=grid, clamp="raw")
    assert np.isfinite(res.attacking).all()
    assert np.isfinite(res.defending).all()
    assert not np.isnan(res.checksum).any()
    # the far corner is a genuine tie between two near-identical players
    iy, ix = _cell(grid, PITCH_L - 1, PITCH_W - 1)
    assert 0.0 <= res.attacking[iy, ix] <= 1.0


def test_ball_position_none_or_nan_means_the_ball_is_already_there() -> None:
    grid = make_grid(20)
    att, dfn, _ = _busy_scene()
    a = pitch_control(att, dfn, None, grid=grid)
    b = pitch_control(att, dfn, np.array([np.nan, np.nan]), grid=grid)
    np.testing.assert_array_equal(a, b)
    # and a real ball position changes the answer
    c = pitch_control(att, dfn, np.array([100.0, 5.0]), grid=grid)
    assert not np.allclose(a, c)


def test_empty_team_raises_rather_than_producing_a_surface() -> None:
    grid = make_grid(10)
    empty = PlayerSet.from_arrays(np.empty((0, 2)))
    with pytest.raises(ValueError, match="at least one in-frame player"):
        pitch_control(empty, _single(CENTRE), CENTRE, grid=grid)
    with pytest.raises(ValueError, match="at least one in-frame player"):
        pitch_control(_single(CENTRE), empty, CENTRE, grid=grid)


def test_non_square_grid_is_not_transposed() -> None:
    """A y-flip or transpose is the classic silent misalignment here."""
    grid = make_grid(n_x=40, n_y=9)
    att = _single([10.0, 5.0])
    dfn = _single([95.0, 63.0])
    surface = pitch_control(att, dfn, np.array([10.0, 5.0]), grid=grid)
    assert surface.shape == (9, 40)
    # row 0 is the BOTTOM of the pitch, so the attacker near y=5 owns low rows
    assert surface[0, 0] > 0.9
    assert surface[-1, -1] < 0.1


# --------------------------------------------------------------------------- #
# Offsides
# --------------------------------------------------------------------------- #


def test_check_offsides_removes_a_player_beyond_the_second_deepest_defender() -> None:
    att = PlayerSet.from_arrays([[60.0, 34.0], [95.0, 30.0]], ids=["on", "off"])
    dfn = PlayerSet.from_arrays([[80.0, 20.0], [78.0, 44.0], [4.0, 34.0]], ids=["a", "b", "gk"])
    kept = check_offsides(att, dfn, np.array([55.0, 34.0]), attack_direction=1)
    assert kept.ids == ("on",)


def test_check_offsides_honours_the_ball_and_the_halfway_line() -> None:
    # ball ahead of the defence: nobody level with the ball is offside
    att = PlayerSet.from_arrays([[95.0, 30.0]], ids=["off"])
    dfn = PlayerSet.from_arrays([[80.0, 20.0], [78.0, 44.0]], ids=["a", "b"])
    assert len(check_offsides(att, dfn, np.array([96.0, 34.0]), 1)) == 1
    # a defence pushed into the attacking half cannot play anyone in his own
    # half offside
    att2 = PlayerSet.from_arrays([[40.0, 30.0]], ids=["deep"])
    dfn2 = PlayerSet.from_arrays([[80.0, 20.0], [78.0, 44.0]], ids=["a", "b"])
    assert len(check_offsides(att2, dfn2, np.array([30.0, 34.0]), 1)) == 1


def test_check_offsides_mirrors_with_attack_direction() -> None:
    att = PlayerSet.from_arrays([[45.0, 34.0], [10.0, 30.0]], ids=["on", "off"])
    dfn = PlayerSet.from_arrays([[25.0, 20.0], [27.0, 44.0], [101.0, 34.0]], ids=["a", "b", "gk"])
    kept = check_offsides(att, dfn, np.array([50.0, 34.0]), attack_direction=-1)
    assert kept.ids == ("on",)


def test_check_offsides_never_crashes_on_degenerate_frames() -> None:
    """The reference IndexErrors on an empty attack and asserts a GK is present.

    Both happen across a match's worth of possession losses.
    """
    empty = PlayerSet.from_arrays(np.empty((0, 2)))
    dfn = PlayerSet.from_arrays([[80.0, 20.0], [78.0, 44.0]])
    assert len(check_offsides(empty, dfn, np.array([50.0, 34.0]), 1)) == 0
    # one defender only, so there is no "second deepest"
    att = PlayerSet.from_arrays([[95.0, 30.0]], ids=["x"])
    assert check_offsides(att, PlayerSet.from_arrays([[80.0, 20.0]]), None, 1).ids == ("x",)
    # NaN ball coordinate, and no goalkeeper flagged anywhere
    assert len(check_offsides(att, dfn, np.array([np.nan, np.nan]), 1)) in (0, 1)


def test_offsides_flag_changes_the_surface_and_is_off_by_default() -> None:
    grid = make_grid(25)
    att = PlayerSet.from_arrays([[60.0, 34.0], [99.0, 34.0]])
    dfn = PlayerSet.from_arrays([[80.0, 20.0], [78.0, 44.0], [4.0, 34.0]], is_gk=[0, 0, 1])
    ball = np.array([55.0, 34.0])
    default = pitch_control(att, dfn, ball, grid=grid)
    off = pitch_control(att, dfn, ball, grid=grid, offsides=True, attack_direction=1)
    plain = pitch_control(att, dfn, ball, grid=grid, offsides=False)
    np.testing.assert_array_equal(default, plain)
    assert not np.allclose(default, off)
    # removing the offside striker costs the attack control near the far goal
    assert off[:, -1].mean() < default[:, -1].mean()


# --------------------------------------------------------------------------- #
# Dataframe adapters
# --------------------------------------------------------------------------- #


def _tracking_frame() -> pd.DataFrame:
    rows = [
        ("home", "1", 5.0, 34.0, 0.0, 0.0),
        ("home", "9", 60.0, 34.0, 1.0, 0.0),
        ("away", "25", 100.0, 34.0, 0.0, 0.0),
        ("away", "4", 62.0, 30.0, -1.0, 0.0),
        ("ball", None, 60.0, 34.0, 0.0, 0.0),
    ]
    return pd.DataFrame(rows, columns=["team", "jersey", "x", "y", "vx", "vy"])


def test_players_from_tracking_flags_the_named_goalkeeper() -> None:
    ps = players_from_tracking(_tracking_frame(), "home", gk_jersey="1")
    assert ps.ids == ("1", "9")
    assert ps.is_gk.tolist() == [True, False]
    assert players_from_tracking(_tracking_frame(), "away").is_gk.tolist() == [False, False]


def test_control_surface_adapter_matches_the_array_api() -> None:
    frame = _tracking_frame()
    grid = make_grid(20)
    via_df = control_surface(frame, "home", grid=grid, gk_jerseys=("1", "25"))
    direct = pitch_control(
        players_from_tracking(frame, "home", "1"),
        players_from_tracking(frame, "away", "25"),
        np.array([60.0, 34.0]),
        grid=grid,
    )
    np.testing.assert_array_equal(via_df, direct)
    with pytest.raises(ValueError, match="must be 'home' or 'away'"):
        control_surface(frame, "Home", grid=grid)


def test_control_share_over_the_pitch_and_a_region() -> None:
    surface = np.zeros((32, 50))
    surface[:, 25:] = 1.0
    assert control_share(surface) == pytest.approx(0.5)
    assert control_share(surface, region=(60.0, 0.0, 105.0, 68.0)) == pytest.approx(1.0)
    assert control_share(surface, region=(0.0, 0.0, 40.0, 68.0)) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="selects no grid cells"):
        control_share(surface, region=(200.0, 0.0, 300.0, 68.0))
    with pytest.raises(ValueError, match="does not match surface"):
        control_share(surface, grid=make_grid(10))
