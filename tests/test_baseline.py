"""Tests for the conditional baseline.

Every scene here is built so its answer follows from arithmetic rather than
from having run the code once and frozen the output. A team laid out on a
known grid has a centroid you can compute on paper; a recovery column
generated as an exact function of position must be predicted exactly and leave
exactly zero residual; two matches built to disagree must produce a NEGATIVE
out-of-sample R2, which is the only proof that the cross-validation is really
holding a match out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pressing.baseline import (
    ALL_FEATURES,
    DEFAULT_SHAPE,
    FEATURE_COLUMNS,
    MIN_LOSSES_FOR_PLAYER,
    RESIDUAL_COLUMNS,
    SHAPE_FEATURES,
    BaselineSpec,
    add_expected_recovery,
    bootstrap_mean_ci,
    crossval_by_match,
    fit_baseline,
    loss_features,
    permutation_between_player,
    reportable_mask,
)
from pressing.schema import PITCH_LENGTH, PITCH_WIDTH

#: Position-only spec, for the "does shape matter" comparisons.
POSITION_ONLY = BaselineSpec(position_degree=3, use_width=True, shape=())


# --------------------------------------------------------------------------- #
# Scene builders
# --------------------------------------------------------------------------- #


def make_tracking(
    frames: dict[int, dict[str, np.ndarray]],
    *,
    match_id: str = "m1",
    goalkeepers: dict[str, str] | None = None,
) -> pd.DataFrame:
    """A minimal tracking frame carrying only what :func:`loss_features` reads.

    Args:
        frames: ``{frame: {"home": (n, 2) array, "away": (n, 2) array}}``.
        match_id: Value for the ``match_id`` column and attrs.
        goalkeepers: Optional ``attrs["goalkeeper"]`` payload.

    Returns:
        A long tracking frame with ``attrs`` populated.
    """
    rows: list[dict[str, object]] = []
    for frame, teams in frames.items():
        for team, positions in teams.items():
            for index, (x, y) in enumerate(np.asarray(positions, dtype=float)):
                rows.append(
                    {
                        "match_id": match_id,
                        "frame": int(frame),
                        "team": team,
                        "player_id": f"{team}_{index}",
                        "x": float(x),
                        "y": float(y),
                    }
                )
    out = pd.DataFrame(rows)
    out.attrs["match_id"] = match_id
    out.attrs["attack_direction"] = {"home": 1, "away": -1}
    out.attrs["goalkeeper"] = dict(goalkeepers or {})
    return out


def make_losses(records: list[dict[str, object]], *, match_id: str = "m1") -> pd.DataFrame:
    """A minimal scored-loss frame carrying only what the baseline reads.

    Args:
        records: One dict per loss. ``frame``, ``losing_team``, ``x``, ``y``
            and ``recovery`` are the interesting keys; ``phase`` and ``scored``
            default sensibly and ``x_norm`` is derived from ``x``.
        match_id: Value for the ``match_id`` column.

    Returns:
        A loss frame in the shape :func:`loss_features` expects.
    """
    rows: list[dict[str, object]] = []
    for index, record in enumerate(records):
        x = float(record.get("x", 52.5))  # type: ignore[arg-type]
        team = str(record.get("losing_team", "home"))
        fraction = x / PITCH_LENGTH
        rows.append(
            {
                "loss_id": int(record.get("loss_id", index)),  # type: ignore[call-overload]
                "match_id": match_id,
                "frame": int(record.get("frame", 1)),  # type: ignore[call-overload]
                "losing_team": team,
                "x": x,
                "y": float(record.get("y", PITCH_WIDTH / 2)),  # type: ignore[arg-type]
                "x_norm": float(
                    record.get("x_norm", fraction if team == "home" else 1.0 - fraction)  # type: ignore[arg-type]
                ),
                "phase": record.get("phase", "middle"),
                "recovery": float(record.get("recovery", 0.5)),  # type: ignore[arg-type]
                "scored": bool(record.get("scored", True)),
            }
        )
    return pd.DataFrame(rows)


def synthetic_features(
    n: int,
    *,
    match_id: str,
    slope: float = 1.0,
    intercept: float = 0.1,
    noise: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Features whose ``recovery`` is an exact linear function of ``x_norm``.

    Args:
        n: Rows.
        match_id: Match label.
        slope: Coefficient on ``x_norm``.
        intercept: Constant term.
        noise: Standard deviation of added Gaussian noise; zero for an exact
            relationship.
        seed: Generator seed.

    Returns:
        A frame with :data:`FEATURE_COLUMNS`.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0.05, 0.95, n)
    frame = pd.DataFrame(
        {
            "loss_id": np.arange(n),
            "match_id": match_id,
            "phase": "middle",
            "recovery": intercept + slope * x + rng.normal(0.0, noise, n),
            "scored": True,
            "x_norm": x,
            "width_norm": rng.uniform(0.0, 1.0, n),
        }
    )
    for name in SHAPE_FEATURES:
        frame[name] = rng.normal(0.0, 1.0, n)
    return frame.loc[:, list(FEATURE_COLUMNS)]


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #


def test_shape_features_are_the_geometry_you_can_compute_on_paper() -> None:
    """A team on a known line has a centroid, a spread and a line height you
    can work out by hand, and the extractor must agree with the arithmetic."""
    # Four home players at x = 10, 20, 30, 40, all on the centre line.
    home = np.array([[10.0, 34.0], [20.0, 34.0], [30.0, 34.0], [40.0, 34.0]])
    away = np.array([[60.0, 34.0], [70.0, 34.0]])
    tracking = make_tracking({1: {"home": home, "away": away}})
    losses = make_losses([{"frame": 1, "losing_team": "home", "x": 35.0, "y": 34.0}])

    features = loss_features(losses, tracking)
    row = features.iloc[0]

    # Home attacks +x, so goalside means a smaller x than the ball's 35 m.
    assert row["n_goalside"] == 3.0
    # Centroid is at x = 25, y = 34; the ball is at 35, so the distance is 10.
    assert row["ball_to_centroid_m"] == pytest.approx(10.0)
    # RMS distance from the centroid over offsets -15, -5, +5, +15.
    assert row["spread_m"] == pytest.approx(np.sqrt((225 + 25 + 25 + 225) / 4))
    # The ball is at 35 m: the 30 m and 40 m players are 5 m away and count,
    # the 20 m player is 15 m away and does not.
    assert row["n_within_10m"] == 2.0
    # Fourth deepest of four is the 40 m player: 40 / 105 of the way up.
    assert row["line_height"] == pytest.approx(40.0 / PITCH_LENGTH)


def test_shape_features_are_measured_from_the_losing_teams_own_goal() -> None:
    """The same physical scene, lost by the team attacking the other way, must
    give the mirrored line height and the complementary goalside count."""
    home = np.array([[10.0, 34.0], [20.0, 34.0], [30.0, 34.0], [40.0, 34.0]])
    tracking = make_tracking({1: {"home": home, "away": home.copy()}})
    losses = make_losses(
        [
            {"frame": 1, "losing_team": "home", "x": 35.0, "loss_id": 0},
            {"frame": 1, "losing_team": "away", "x": 35.0, "loss_id": 1},
        ]
    )
    features = loss_features(losses, tracking).set_index("loss_id")
    assert features.loc[0, "line_height"] == pytest.approx(40.0 / PITCH_LENGTH)
    assert features.loc[1, "line_height"] == pytest.approx(1.0 - 10.0 / PITCH_LENGTH)
    assert features.loc[0, "n_goalside"] == 3.0
    assert features.loc[1, "n_goalside"] == 1.0


def test_the_goalkeeper_is_left_out_of_every_shape_feature() -> None:
    """A keeper on his own line would drag the centroid twenty metres, so the
    features must be identical with and without him in the frame."""
    outfield = np.array([[40.0, 30.0], [45.0, 34.0], [50.0, 38.0], [55.0, 34.0]])
    keeper = np.array([[5.0, 34.0]])
    away = np.array([[70.0, 34.0]])
    losses = make_losses([{"frame": 1, "losing_team": "home", "x": 50.0, "y": 34.0}])

    without = loss_features(
        make_losses([{"frame": 1, "losing_team": "home", "x": 50.0, "y": 34.0}]),
        make_tracking({1: {"home": outfield, "away": away}}),
    )
    with_keeper = loss_features(
        losses,
        make_tracking(
            {1: {"home": np.vstack([outfield, keeper]), "away": away}},
            goalkeepers={"home": "home_4"},
        ),
    )
    for name in SHAPE_FEATURES:
        assert with_keeper[name].iloc[0] == pytest.approx(without[name].iloc[0])


def test_width_is_zero_down_the_middle_and_one_on_the_touchline() -> None:
    """``width_norm`` is normalised, so the two ends of its range are exact."""
    tracking = make_tracking({1: {"home": np.array([[50.0, 34.0]]), "away": np.zeros((1, 2))}})
    losses = make_losses(
        [
            {"frame": 1, "y": PITCH_WIDTH / 2, "loss_id": 0},
            {"frame": 1, "y": 0.0, "loss_id": 1},
            {"frame": 1, "y": PITCH_WIDTH, "loss_id": 2},
        ]
    )
    widths = loss_features(losses, tracking).set_index("loss_id")["width_norm"]
    assert widths.loc[0] == pytest.approx(0.0)
    assert widths.loc[1] == pytest.approx(1.0)
    assert widths.loc[2] == pytest.approx(1.0)


def test_a_loss_on_an_untracked_frame_keeps_its_row_with_nan_shape() -> None:
    """Nothing is silently dropped: the row survives with NaN shape features."""
    tracking = make_tracking({1: {"home": np.array([[50.0, 34.0]]), "away": np.zeros((1, 2))}})
    losses = make_losses([{"frame": 999, "losing_team": "home", "x": 50.0}])
    features = loss_features(losses, tracking)
    assert len(features) == 1
    assert np.isfinite(features["x_norm"].iloc[0])
    assert features[list(SHAPE_FEATURES)].isna().all(axis=None)


def test_loss_features_emits_exactly_the_documented_columns() -> None:
    """The column contract is pinned, because the model indexes it by name."""
    tracking = make_tracking({1: {"home": np.array([[50.0, 34.0]]), "away": np.zeros((1, 2))}})
    features = loss_features(make_losses([{"frame": 1}]), tracking)
    assert list(features.columns) == list(FEATURE_COLUMNS)
    assert set(ALL_FEATURES) <= set(features.columns)


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #


def test_residuals_on_the_fitting_set_are_zero_mean_by_construction() -> None:
    """The intercept is unpenalised, so this is exact arithmetic, not luck."""
    features = synthetic_features(120, match_id="m1", noise=0.05, seed=3)
    model = fit_baseline(features)
    residual = features["recovery"].to_numpy(float) - model.predict(features)
    assert residual.mean() == pytest.approx(0.0, abs=1e-12)
    assert model.intercept == pytest.approx(features["recovery"].mean())


def test_a_baseline_recovers_a_relationship_it_was_given_exactly() -> None:
    """Recovery generated as ``0.1 + 1.0 * x_norm`` must come back exactly."""
    features = synthetic_features(80, match_id="m1", slope=1.0, intercept=0.1, noise=0.0)
    exact = BaselineSpec(position_degree=1, use_width=False, shape=(), ridge=0.0)
    model = fit_baseline(features, exact)
    predicted = model.predict(features)
    assert predicted == pytest.approx(0.1 + 1.0 * features["x_norm"].to_numpy(float), abs=1e-6)
    assert (features["recovery"].to_numpy(float) - predicted) == pytest.approx(0.0, abs=1e-6)


def test_the_fit_ignores_unscored_losses() -> None:
    """A skipped loss has no recovery to learn from and must not shift the fit."""
    features = synthetic_features(60, match_id="m1", noise=0.02, seed=5)
    poisoned = pd.concat(
        [
            features,
            features.head(10).assign(recovery=99.0, scored=False, loss_id=range(100, 110)),
        ],
        ignore_index=True,
    )
    clean = fit_baseline(features)
    dirty = fit_baseline(poisoned)
    assert dirty.n_train == clean.n_train
    assert dirty.intercept == pytest.approx(clean.intercept)


def test_a_missing_feature_scores_at_the_training_mean_instead_of_vanishing() -> None:
    """A loss on an untracked frame still gets an expectation."""
    features = synthetic_features(60, match_id="m1", noise=0.02, seed=6)
    model = fit_baseline(features)
    blank = features.head(1).copy()
    for name in DEFAULT_SHAPE:
        blank[name] = np.nan
    assert np.isfinite(model.predict(blank)[0])


def test_fitting_on_nothing_is_an_error_not_a_silent_nan() -> None:
    """A frame with no scored loss cannot produce a baseline."""
    features = synthetic_features(10, match_id="m1").assign(scored=False)
    with pytest.raises(ValueError, match="zero scored losses"):
        fit_baseline(features)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"position_degree": 0}, "position_degree"),
        ({"ridge": -1.0}, "ridge"),
        ({"shape": ("not_a_feature",)}, "unknown shape features"),
    ],
)
def test_a_bad_spec_is_rejected_at_construction(kwargs: dict, message: str) -> None:
    """Validation happens once, in ``__post_init__``, not per prediction."""
    with pytest.raises(ValueError, match=message):
        BaselineSpec(**kwargs)


# --------------------------------------------------------------------------- #
# Cross-validation by match
# --------------------------------------------------------------------------- #


def two_matches(noise: float = 0.02, flip: bool = False) -> pd.DataFrame:
    """Two matches of synthetic losses, optionally with opposite slopes.

    Args:
        noise: Gaussian noise on ``recovery``.
        flip: When True the second match's relationship is the mirror of the
            first's, so a model fitted on one is actively wrong on the other.

    Returns:
        A feature frame covering ``m1`` and ``m2``.
    """
    first = synthetic_features(90, match_id="m1", slope=1.0, noise=noise, seed=1)
    second = synthetic_features(
        90, match_id="m2", slope=-1.0 if flip else 1.0, intercept=1.1 if flip else 0.1,
        noise=noise, seed=2,
    )
    return pd.concat([first, second], ignore_index=True)


def test_no_fold_is_ever_fitted_and_scored_on_the_same_match() -> None:
    """The leakage guard, asserted rather than assumed."""
    cv = crossval_by_match(two_matches())
    assert len(cv.folds) == 2
    for train_matches, held_out in cv.folds:
        assert held_out not in train_matches
        assert set(train_matches).isdisjoint({held_out})
    assert set(cv.fold_r2) == {"m1", "m2"}


def test_two_matches_that_disagree_produce_a_negative_out_of_sample_r2() -> None:
    """This is the test that would fail if the split leaked.

    The two matches are built with opposite slopes. Fitted together the model
    finds a compromise and reports a respectable in-sample number; fitted on
    one and scored on the other it must be worse than printing the grand mean,
    i.e. R2 below zero. A leaking split could not produce that.
    """
    cv = crossval_by_match(two_matches(flip=True), POSITION_ONLY)
    assert cv.r2_oos < 0.0
    assert cv.r2_in > cv.r2_oos


def test_two_matches_that_agree_generalise_across_the_split() -> None:
    """The other half of the same argument."""
    cv = crossval_by_match(two_matches(noise=0.02), POSITION_ONLY)
    assert cv.r2_oos > 0.9
    assert all(value > 0.9 for value in cv.fold_r2.values())


def test_out_of_fold_predictions_cover_every_scored_row_and_no_other() -> None:
    """Every scored loss is scored by a model that never saw its match."""
    features = two_matches()
    features.loc[0, "scored"] = False
    cv = crossval_by_match(features)
    predicted = np.isfinite(cv.prediction)
    assert predicted.tolist() == features["scored"].tolist()
    assert cv.n_scored == int(features["scored"].sum())


def test_cross_validating_one_match_is_an_error() -> None:
    """There is nothing to hold out, and pretending otherwise would be a lie."""
    with pytest.raises(ValueError, match="at least two matches"):
        crossval_by_match(synthetic_features(40, match_id="only"))


def test_shape_earns_its_place_or_it_does_not_and_the_number_says_which() -> None:
    """When recovery is built to depend on a shape feature the full spec must
    beat the position-only one out of sample; when it is not, it must not."""
    features = two_matches(noise=0.02)
    with_shape = features.copy()
    with_shape["recovery"] = with_shape["recovery"] + 0.3 * with_shape["line_height"]
    gain = (
        crossval_by_match(with_shape).r2_oos - crossval_by_match(with_shape, POSITION_ONLY).r2_oos
    )
    assert gain > 0.05

    noise_only = features.copy()
    idle = (
        crossval_by_match(noise_only).r2_oos
        - crossval_by_match(noise_only, POSITION_ONLY).r2_oos
    )
    assert idle < 0.01


# --------------------------------------------------------------------------- #
# Attaching the residual
# --------------------------------------------------------------------------- #


def scored_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A two-match loss frame and its features, with ids that collide.

    ``loss_id`` restarts at zero in every match, which is exactly the trap the
    join has to survive.
    """
    features = two_matches()
    losses = features.loc[:, ["loss_id", "match_id", "phase", "recovery", "scored"]].copy()
    return losses, features


def test_the_residual_is_the_observed_minus_the_expected_and_nothing_else() -> None:
    """No clipping, no rescaling, no quiet reindexing."""
    losses, features = scored_pair()
    out = add_expected_recovery(losses, features)
    assert set(RESIDUAL_COLUMNS) <= set(out.columns)
    difference = out["recovery"] - out["recovery_expected"]
    assert out["recovery_residual"].to_numpy() == pytest.approx(difference.to_numpy(), nan_ok=True)


def test_two_matches_give_an_out_of_fold_residual_and_say_so() -> None:
    """The flag the report has to print is set from the data, not by hand."""
    losses, features = scored_pair()
    out = add_expected_recovery(losses, features)
    assert out.attrs["residual_out_of_sample"] is True


def test_one_match_gives_an_in_sample_residual_and_admits_it() -> None:
    """With nothing to hold out the residual is in-sample; the flag says False
    and it is the report's job to repeat that to the reader."""
    features = synthetic_features(60, match_id="solo", noise=0.03, seed=8)
    losses = features.loc[:, ["loss_id", "match_id", "phase", "recovery", "scored"]].copy()
    out = add_expected_recovery(losses, features)
    assert out.attrs["residual_out_of_sample"] is False
    residual = out.loc[out["scored"], "recovery_residual"].to_numpy(float)
    assert residual.mean() == pytest.approx(0.0, abs=1e-12)


def test_losses_are_joined_on_match_and_id_not_on_id_alone() -> None:
    """Both matches number their losses from zero. Joining on ``loss_id`` would
    cross-product them; the row count is the proof that it does not."""
    losses, features = scored_pair()
    assert losses["loss_id"].duplicated().any()
    out = add_expected_recovery(losses, features)
    assert len(out) == len(losses)
    for match_id in ("m1", "m2"):
        block = out[out["match_id"] == match_id]
        source = losses[losses["match_id"] == match_id]
        assert block["recovery"].to_numpy() == pytest.approx(source["recovery"].to_numpy())


def test_an_unscored_loss_gets_no_expectation_and_no_residual() -> None:
    """A skipped loss keeps its row and stays NaN in both new columns."""
    losses, features = scored_pair()
    features.loc[0, "scored"] = False
    losses.loc[0, "scored"] = False
    out = add_expected_recovery(losses, features)
    assert np.isnan(out.loc[0, "recovery_expected"])
    assert np.isnan(out.loc[0, "recovery_residual"])
    assert out["recovery_expected"].notna().sum() == int(losses["scored"].sum())


def test_a_loss_with_no_features_is_an_error_not_a_quiet_nan() -> None:
    """Silently scoring half the table would be the worst possible outcome."""
    losses, features = scored_pair()
    with pytest.raises(ValueError, match="missing"):
        add_expected_recovery(losses, features.iloc[5:])


# --------------------------------------------------------------------------- #
# Bootstrap intervals and the print/suppress rule
# --------------------------------------------------------------------------- #


def test_the_bootstrap_interval_widens_as_the_count_falls() -> None:
    """This is the whole reason the threshold exists, so it is pinned.

    The same population, resampled at four sample sizes: the interval must
    shrink monotonically as n grows, and 5 losses must be several times wider
    than 200.
    """
    rng = np.random.default_rng(4)
    population = rng.normal(0.5, 0.18, 5000)
    widths = []
    for n in (5, 20, 80, 320):
        draw = population[:n]
        low, high = bootstrap_mean_ci(draw, n_boot=3000, rng=np.random.default_rng(9))
        widths.append(high - low)
    assert widths == sorted(widths, reverse=True)
    assert widths[0] > 4 * widths[-1]


def test_the_interval_brackets_the_mean_it_describes() -> None:
    """A percentile interval on a mean contains that mean."""
    rng = np.random.default_rng(12)
    values = rng.normal(0.2, 0.1, 60)
    low, high = bootstrap_mean_ci(values, n_boot=2000, rng=np.random.default_rng(1))
    assert low <= values.mean() <= high


def test_one_loss_gives_a_point_not_an_interval() -> None:
    """Resampling one value can only ever return that value, and the interval
    says so honestly instead of pretending to a width of zero it earned."""
    assert bootstrap_mean_ci(np.array([0.42]), rng=np.random.default_rng(0)) == (0.42, 0.42)


def test_an_empty_sample_gives_nan_rather_than_an_exception() -> None:
    """A player credited with nothing must not crash the table."""
    low, high = bootstrap_mean_ci(np.array([np.nan, np.nan]), rng=np.random.default_rng(0))
    assert np.isnan(low) and np.isnan(high)


def test_the_same_seed_gives_the_same_interval() -> None:
    """The demo PDF is required to be byte-identical between runs."""
    values = np.random.default_rng(2).normal(0.5, 0.1, 40)
    first = bootstrap_mean_ci(values, n_boot=500, rng=np.random.default_rng(77))
    second = bootstrap_mean_ci(values, n_boot=500, rng=np.random.default_rng(77))
    assert first == second


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"level": 1.0}, "level"), ({"n_boot": 0}, "n_boot")],
)
def test_a_bad_bootstrap_request_is_rejected(kwargs: dict, message: str) -> None:
    """Bounds are checked once, up front."""
    with pytest.raises(ValueError, match=message):
        bootstrap_mean_ci(np.array([1.0, 2.0]), **kwargs)


def test_the_threshold_is_inclusive_and_suppresses_below_it() -> None:
    """Exactly :data:`MIN_LOSSES_FOR_PLAYER` is printed; one fewer is not."""
    counts = np.array([MIN_LOSSES_FOR_PLAYER - 1, MIN_LOSSES_FOR_PLAYER, 0, np.nan])
    assert reportable_mask(counts).tolist() == [False, True, False, False]


def test_the_threshold_is_a_parameter_even_though_it_is_derived() -> None:
    """Someone with more matches should be able to raise it."""
    assert reportable_mask(np.array([12]), minimum=20).tolist() == [False]
    assert reportable_mask(np.array([12]), minimum=5).tolist() == [True]


# --------------------------------------------------------------------------- #
# The permutation test
# --------------------------------------------------------------------------- #


def grouped_values(
    effect: float,
    *,
    n_groups: int = 20,
    per_group: int = 25,
    seed: int = 0,
    offset: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Values with a controllable between-group effect and block offset.

    Args:
        effect: Standard deviation of the per-group means. Zero makes the
            groups genuinely exchangeable.
        n_groups: Groups per block.
        per_group: Values per group.
        seed: Generator seed.
        offset: Optional per-block additive offset, e.g.
            ``{"a": -0.05, "b": +0.05}``. Groups are split evenly between the
            blocks and no group ever spans two.

    Returns:
        ``(values, groups, strata)``.
    """
    rng = np.random.default_rng(seed)
    blocks = list((offset or {"only": 0.0}).items())
    values: list[float] = []
    groups: list[str] = []
    strata: list[str] = []
    for index in range(n_groups):
        block, shift = blocks[index % len(blocks)]
        centre = rng.normal(0.0, effect) if effect > 0 else 0.0
        values += list(rng.normal(centre + shift, 0.15, per_group))
        groups += [f"{block}_g{index}"] * per_group
        strata += [block] * per_group
    return np.asarray(values), np.asarray(groups, dtype=object), np.asarray(strata, dtype=object)


def test_a_real_between_player_effect_is_detected() -> None:
    """The test has to be able to find something, or its nulls mean nothing."""
    values, groups, _ = grouped_values(effect=0.12, seed=1)
    result = permutation_between_player(
        values, groups, n_shuffles=500, rng=np.random.default_rng(0)
    )
    assert result.p_value < 0.01
    assert result.observed_sd > result.null_p95
    assert result.n_groups == 20


def test_exchangeable_players_do_not_clear_the_test() -> None:
    """Groups drawn from one distribution must not look like a discovery."""
    values, groups, _ = grouped_values(effect=0.0, seed=2)
    result = permutation_between_player(
        values, groups, n_shuffles=500, rng=np.random.default_rng(0)
    )
    assert result.p_value > 0.05
    assert result.observed_sd == pytest.approx(result.null_median, abs=0.02)


def test_a_block_offset_masquerades_as_player_signal_unless_it_is_stratified() -> None:
    """The reason ``strata`` exists, on data where the truth is known.

    Two blocks, a large offset between them, and NO real difference between
    players inside a block. Shuffling across the blocks finds a "player
    effect" that is really a fixture; shuffling within them does not.
    """
    values, groups, strata = grouped_values(
        effect=0.0, n_groups=24, per_group=30, seed=3, offset={"a": -0.12, "b": 0.12}
    )
    naive = permutation_between_player(
        values, groups, n_shuffles=500, rng=np.random.default_rng(0)
    )
    stratified = permutation_between_player(
        values, groups, strata=strata, n_shuffles=500, rng=np.random.default_rng(0)
    )
    assert naive.p_value < 0.01
    assert stratified.p_value > 0.05


def test_a_player_may_not_span_two_strata() -> None:
    """A key that appears in two blocks would make the shuffle meaningless."""
    values = np.arange(20, dtype=float)
    groups = np.array(["p"] * 10 + ["q"] * 10, dtype=object)
    strata = np.array(["a", "b"] * 10, dtype=object)
    with pytest.raises(ValueError, match="span more than one stratum"):
        permutation_between_player(values, groups, strata=strata, n_shuffles=10, min_count=2)


def test_players_below_the_minimum_count_are_dropped_before_the_test() -> None:
    """A one-loss player's mean is that loss and would inflate the statistic."""
    values, groups, _ = grouped_values(effect=0.0, n_groups=6, per_group=20, seed=4)
    values = np.concatenate([values, [5.0, -5.0]])
    groups = np.concatenate([groups, np.array(["tiny_a", "tiny_b"], dtype=object)])
    result = permutation_between_player(values, groups, n_shuffles=200, min_count=5)
    assert result.n_groups == 6
    assert result.n_values == 120


def test_too_few_players_gives_nan_rather_than_a_p_value() -> None:
    """One group cannot be compared with anything."""
    result = permutation_between_player(
        np.arange(10, dtype=float),
        np.array(["only"] * 10, dtype=object),
        n_shuffles=50,
    )
    assert result.n_groups == 1
    assert np.isnan(result.p_value)


def test_the_p_value_can_never_be_reported_as_zero() -> None:
    """The plus-one correction is a floor, not a rounding artefact."""
    values, groups, _ = grouped_values(effect=1.0, n_groups=10, per_group=40, seed=5)
    result = permutation_between_player(values, groups, n_shuffles=99, rng=np.random.default_rng(0))
    assert result.p_value == pytest.approx(1.0 / 100.0)


def test_mismatched_inputs_are_rejected() -> None:
    """Length checks up front, so a silent misalignment cannot happen."""
    with pytest.raises(ValueError, match="differ in length"):
        permutation_between_player(np.arange(5.0), np.array(["a"] * 4, dtype=object))
    with pytest.raises(ValueError, match="n_shuffles"):
        permutation_between_player(np.arange(5.0), np.array(["a"] * 5, dtype=object), n_shuffles=0)


# --------------------------------------------------------------------------- #
# The residual is the point: it must remove position
# --------------------------------------------------------------------------- #


def test_residualising_removes_the_positional_ranking_it_is_meant_to_remove() -> None:
    """A table of players who differ ONLY in where they stand must rank
    strongly on the raw column and not at all on the residual."""
    rng = np.random.default_rng(6)
    rows: list[dict[str, object]] = []
    for player in range(12):
        centre = 0.1 + 0.07 * player
        for index in range(30):
            x = float(np.clip(rng.normal(centre, 0.03), 0.02, 0.98))
            rows.append(
                {
                    "loss_id": len(rows),
                    "match_id": "m1" if index % 2 else "m2",
                    "phase": "middle",
                    "recovery": 0.1 + 0.8 * x + float(rng.normal(0.0, 0.02)),
                    "scored": True,
                    "x_norm": x,
                    "width_norm": 0.5,
                    "player": f"p{player}",
                }
            )
    frame = pd.DataFrame(rows)
    for name in SHAPE_FEATURES:
        frame[name] = 0.0
    features = frame.loc[:, list(FEATURE_COLUMNS)]
    scored = add_expected_recovery(frame.loc[:, ["loss_id", "match_id", "recovery", "scored"]],
                                   features)
    scored["player"] = frame["player"]

    raw_spread = scored.groupby("player")["recovery"].mean().std(ddof=1)
    residual_spread = scored.groupby("player")["recovery_residual"].mean().std(ddof=1)
    assert raw_spread > 20 * residual_spread
