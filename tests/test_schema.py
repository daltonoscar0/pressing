"""Unit tests for the canonical schema validators."""

from __future__ import annotations

import pandas as pd
import pytest

from pressing.schema import (
    EVENT_COLUMNS,
    TRACKING_COLUMNS,
    SchemaError,
    validate_events,
    validate_tracking,
)


def valid_events() -> pd.DataFrame:
    """Build a minimal events frame that satisfies the schema."""
    return pd.DataFrame(
        {
            "match_id": ["m1", "m1"],
            "competition": ["La Liga", "La Liga"],
            "season": ["2023/2024", "2023/2024"],
            "team_id": [1, 2],
            "team": ["Home FC", "Away FC"],
            "player_id": [10, 20],
            "player": ["A Player", "B Player"],
            "t": [12.5, 14.0],
            "period": [1, 1],
            "type": ["pass", "interception"],
            "subtype": ["ground", None],
            "x": [40.0, 62.0],
            "y": [30.0, 33.5],
            "end_x": [55.0, 60.0],
            "end_y": [31.0, 30.0],
            "outcome": ["incomplete", "complete"],
            "pressure": [True, False],
            "body_part": ["right_foot", "right_foot"],
            "extra": ["{}", "{}"],
        }
    )


def valid_tracking() -> pd.DataFrame:
    """Build a minimal tracking frame that satisfies the schema."""
    return pd.DataFrame(
        {
            "match_id": ["m1", "m1", "m1"],
            "t": [12.5, 12.5, 12.5],
            "period": [1, 1, 1],
            "frame": [125, 125, 125],
            "team": ["home", "away", "ball"],
            "player_id": [10, 20, None],
            "jersey": [7, 4, None],
            "x": [40.0, 44.0, 41.0],
            "y": [30.0, 31.0, 30.5],
            "vx": [3.1, -2.0, 5.5],
            "vy": [0.4, 0.1, 0.2],
            "confidence": [1.0, 1.0, 1.0],
        }
    )


class TestValidateEvents:
    def test_valid_frame_passes(self) -> None:
        out = validate_events(valid_events())
        assert list(out.columns) == list(EVENT_COLUMNS)
        assert len(out) == 2

    def test_reorders_columns_canonically(self) -> None:
        shuffled = valid_events()[list(reversed(EVENT_COLUMNS))]
        assert list(validate_events(shuffled).columns) == list(EVENT_COLUMNS)

    def test_missing_column_raises(self) -> None:
        df = valid_events().drop(columns=["pressure"])
        with pytest.raises(SchemaError, match="missing columns: pressure"):
            validate_events(df)

    def test_several_missing_columns_all_listed(self) -> None:
        df = valid_events().drop(columns=["x", "y"])
        with pytest.raises(SchemaError) as exc:
            validate_events(df)
        assert "x" in str(exc.value)
        assert "y" in str(exc.value)

    def test_extra_column_raises(self) -> None:
        df = valid_events()
        df["xg"] = 0.1
        with pytest.raises(SchemaError, match="unexpected columns: xg"):
            validate_events(df)

    def test_wrong_dtype_raises(self) -> None:
        df = valid_events()
        df["x"] = df["x"].astype(str)
        with pytest.raises(SchemaError, match="wrong dtypes"):
            validate_events(df)

    def test_non_bool_pressure_raises(self) -> None:
        df = valid_events()
        df["pressure"] = ["yes", "no"]
        with pytest.raises(SchemaError, match="pressure"):
            validate_events(df)

    def test_float_period_raises(self) -> None:
        df = valid_events()
        df["period"] = df["period"].astype(float)
        with pytest.raises(SchemaError, match="period"):
            validate_events(df)

    def test_not_a_dataframe_raises(self) -> None:
        with pytest.raises(SchemaError, match="expected a pandas DataFrame"):
            validate_events([1, 2, 3])  # type: ignore[arg-type]


class TestValidateTracking:
    def test_valid_frame_passes(self) -> None:
        out = validate_tracking(valid_tracking())
        assert list(out.columns) == list(TRACKING_COLUMNS)
        assert len(out) == 3

    def test_missing_column_raises(self) -> None:
        df = valid_tracking().drop(columns=["vx"])
        with pytest.raises(SchemaError, match="missing columns: vx"):
            validate_tracking(df)

    def test_extra_column_raises(self) -> None:
        df = valid_tracking()
        df["speed"] = 3.0
        with pytest.raises(SchemaError, match="unexpected columns: speed"):
            validate_tracking(df)

    def test_wrong_dtype_raises(self) -> None:
        df = valid_tracking()
        df["vy"] = df["vy"].astype(str)
        with pytest.raises(SchemaError, match="wrong dtypes"):
            validate_tracking(df)

    def test_non_integer_frame_raises(self) -> None:
        df = valid_tracking()
        df["frame"] = df["frame"].astype(float)
        with pytest.raises(SchemaError, match="frame"):
            validate_tracking(df)

    def test_unknown_team_value_raises(self) -> None:
        df = valid_tracking()
        df.loc[0, "team"] = "referee"
        with pytest.raises(SchemaError, match="referee"):
            validate_tracking(df)

    def test_error_message_names_the_frame_kind(self) -> None:
        df = valid_tracking().drop(columns=["confidence"])
        with pytest.raises(SchemaError, match="^tracking schema invalid"):
            validate_tracking(df)
