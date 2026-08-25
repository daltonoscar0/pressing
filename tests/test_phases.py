"""Tests for the build-up / middle / final-third split.

The boundary convention is the whole point of this module, so it is pinned
here: both cuts are LEFT-CLOSED, at exactly 1/3 and exactly 2/3 of the pitch
length measured from the team's own goal line.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pressing.phases import (
    BUILD_UP_MAX_FRACTION,
    BUILD_UP_MAX_X,
    MIDDLE_MAX_FRACTION,
    MIDDLE_MAX_X,
    PHASES,
    attacking_fraction,
    label_loss_phases,
    label_phases,
    phase_for_fraction,
    phase_for_x,
    phase_of_loss,
)
from pressing.schema import PITCH_LENGTH


def test_phase_order_is_stable() -> None:
    """The report iterates PHASES directly, so its order is part of the API."""
    assert PHASES == ("build_up", "middle", "final_third")


def test_thresholds_are_derived_from_the_fractions() -> None:
    """The metre cuts must be the fraction cuts, never independently typed."""
    assert BUILD_UP_MAX_X == pytest.approx(PITCH_LENGTH / 3.0)
    assert MIDDLE_MAX_X == pytest.approx(2.0 * PITCH_LENGTH / 3.0)


@pytest.mark.parametrize(
    ("fraction", "expected"),
    [
        (0.0, "build_up"),
        (0.3333, "build_up"),
        (0.5, "middle"),
        (0.6666, "middle"),
        (0.9, "final_third"),
        (1.0, "final_third"),
        # Players and the ball do go behind the goal lines.
        (-0.02, "build_up"),
        (1.05, "final_third"),
    ],
)
def test_phase_for_fraction(fraction: float, expected: str) -> None:
    """Ordinary fractions land in the phase a coach would name."""
    assert phase_for_fraction(fraction) == expected


def test_boundaries_are_left_closed() -> None:
    """Exactly 1/3 is middle and exactly 2/3 is final third, not the reverse."""
    assert phase_for_fraction(BUILD_UP_MAX_FRACTION) == "middle"
    assert phase_for_fraction(math.nextafter(BUILD_UP_MAX_FRACTION, 0.0)) == "build_up"
    assert phase_for_fraction(MIDDLE_MAX_FRACTION) == "final_third"
    assert phase_for_fraction(math.nextafter(MIDDLE_MAX_FRACTION, 0.0)) == "middle"


def test_boundaries_in_metres_agree_with_the_fractions() -> None:
    """The metre entry point must cut at the same two places."""
    assert phase_for_x(BUILD_UP_MAX_X) == "middle"
    assert phase_for_x(math.nextafter(BUILD_UP_MAX_X, 0.0)) == "build_up"
    assert phase_for_x(MIDDLE_MAX_X) == "final_third"
    assert phase_for_x(math.nextafter(MIDDLE_MAX_X, 0.0)) == "middle"


def test_non_finite_x_has_no_phase() -> None:
    """One Metrica game 1 loss has no location at all; it must not be forced."""
    assert phase_for_x(float("nan")) is None
    assert phase_for_fraction(float("nan")) is None
    assert phase_of_loss(float("nan"), 1) is None


def test_attacking_fraction_mirrors_for_a_team_attacking_minus_x() -> None:
    """A team attacking -x reads the pitch backwards."""
    assert attacking_fraction(10.0, 1) == pytest.approx(10.0 / 105.0)
    assert attacking_fraction(10.0, -1) == pytest.approx(1.0 - 10.0 / 105.0)
    assert attacking_fraction(52.5, -1) == pytest.approx(0.5)


def test_attacking_fraction_rejects_a_bad_direction() -> None:
    """A direction of 0 is a bug, not a neutral default."""
    with pytest.raises(ValueError, match="attack_direction"):
        attacking_fraction(10.0, 0)


def test_phase_of_loss_uses_the_losing_team_direction() -> None:
    """The same x is build-up for one team and final third for the other."""
    assert phase_of_loss(10.0, 1) == "build_up"
    assert phase_of_loss(10.0, -1) == "final_third"


def test_label_phases_defaults_to_already_normalised_x() -> None:
    """Without a direction the column is taken as attacking left to right."""
    frame = pd.DataFrame({"x": [1.0, 52.5, 100.0, np.nan]})
    out = label_phases(frame)
    assert list(out["phase"])[:3] == ["build_up", "middle", "final_third"]
    assert pd.isna(out["phase"].iloc[3])
    assert out["x_norm"].iloc[1] == pytest.approx(0.5)
    # The input is not mutated.
    assert "phase" not in frame.columns


def test_label_phases_honours_a_per_row_direction() -> None:
    """Two rows at the same x but opposite directions get opposite phases."""
    frame = pd.DataFrame({"x": [10.0, 10.0], "d": [1, -1]})
    out = label_phases(frame, direction_col="d")
    assert list(out["phase"]) == ["build_up", "final_third"]


def test_label_phases_rejects_a_missing_column() -> None:
    """A typo in the column name should say so rather than produce NaNs."""
    with pytest.raises(KeyError):
        label_phases(pd.DataFrame({"y": [1.0]}), "x")


def test_label_loss_phases_reads_the_direction_from_attrs() -> None:
    """A loss table carries its match's playing direction in attrs."""
    losses = pd.DataFrame(
        {"x": [10.0, 10.0, np.nan], "losing_team": ["home", "away", "home"]}
    )
    losses.attrs["attack_direction"] = {"home": 1, "away": -1}
    out = label_loss_phases(losses)
    assert list(out["phase"])[:2] == ["build_up", "final_third"]
    assert pd.isna(out["phase"].iloc[2])
    assert list(out["attack_direction"]) == [1, -1, 1]


def test_label_loss_phases_on_an_empty_table() -> None:
    """An empty match must still produce the columns downstream code reads."""
    losses = pd.DataFrame(columns=["x", "losing_team"])
    out = label_loss_phases(losses)
    assert {"attack_direction", "x_norm", "phase"} <= set(out.columns)
    assert out.empty
