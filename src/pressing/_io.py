"""Loader shim over the shared data layer, plus the internal Metrica loaders.

The suite is meant to sit on top of ``sportio``, a shared package that loads
every open soccer data source and normalizes it to the canonical ``events`` and
``tracking`` frames described in :mod:`pressing.schema`. That package does not
exist on this machine yet, so this module follows the fallback the sibling
setpiece spec spells out: prefer ``sportio`` whenever it is importable, and
otherwise fall back to the loaders implemented here.

Canonical coordinate convention
-------------------------------
Everything this module emits is in METRES on a 105 x 68 pitch with the origin at
the BOTTOM-LEFT: ``x`` in ``[0, 105]``, ``y`` in ``[0, 68]``, ``y`` increasing
upward. Metrica ships normalized 0-1 coordinates with a TOP-LEFT origin and
``y`` increasing downward, so the conversion is::

    x_m = x_norm * 105
    y_m = (1 - y_norm) * 68

The whole match is then rotated to a SINGLE PLAYING DIRECTION: the home team
attacks ``+x`` for the entire match. Whichever period needs it is mirrored with
``x -> 105 - x``, ``y -> 68 - y``, decided empirically from goalkeeper mean x
per period (never hard-coded: Metrica games 1 and 2 genuinely disagree about
which way the home team attacks first). ``attack_direction`` is therefore ``+1``
for home and ``-1`` for away, constant across the match, and is recorded in
``DataFrame.attrs``.

The Friends of Tracking reference implementation uses a different frame: origin
at the centre circle, ``field_dimen = (106.0, 68.0)``, ``y`` already flipped.
:func:`to_reference_frame` and :func:`from_reference_frame` are the explicit,
tested bridge between the two. Never assume the conversion implicitly.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from pressing.schema import (
    PITCH_LENGTH,
    PITCH_WIDTH,
    validate_events,
    validate_tracking,
)

__all__ = [
    "HAVE_SPORTIO",
    "METRICA_FPS",
    "REFERENCE_FIELD_DIMEN",
    "MetricaMatch",
    "PlayingDirection",
    "backend_name",
    "compute_velocities",
    "detect_playing_direction",
    "event_extra",
    "from_reference_frame",
    "load_events",
    "load_metrica_events",
    "load_metrica_match",
    "load_metrica_tracking",
    "load_tracking",
    "read_tracking_parquet",
    "subtype_tokens",
    "to_reference_frame",
    "write_tracking_parquet",
]

try:  # pragma: no cover - depends on whether the sibling package is installed
    import sportio as _sportio
except ImportError:  # pragma: no cover
    _sportio = None

HAVE_SPORTIO: Final[bool] = _sportio is not None

#: Metrica's native tracking rate. Verified: every consecutive ``Time [s]``
#: difference in all four sample tracking files is exactly 0.04 s.
METRICA_FPS: Final[float] = 25.0

#: The Friends of Tracking reference pitch. Note it is 106 m long, not the 105 m
#: the Metrica README states; that discrepancy is real and is why the conversion
#: is an explicit function rather than an assumption.
REFERENCE_FIELD_DIMEN: Final[tuple[float, float]] = (106.0, 68.0)

#: The three non-positional columns every Metrica tracking file starts with.
_TRACKING_META_COLUMNS: Final[tuple[str, ...]] = ("Period", "Frame", "Time [s]")

_BALL_LABEL: Final[str] = "Ball"


def backend_name() -> str:
    """Return which loader backend is active.

    Returns:
        ``"sportio"`` when the shared data layer is importable, otherwise
        ``"internal"``.
    """
    return "sportio" if HAVE_SPORTIO else "internal"


def load_events(source: str, **kwargs: Any) -> pd.DataFrame:
    """Load events from ``source`` as a canonical ``events`` frame.

    Delegates to :func:`sportio.load_events` when the shared layer is available,
    otherwise falls back to :func:`_load_events_internal`.

    Args:
        source: Source key, e.g. ``"statsbomb"``, ``"metrica"``, ``"skillcorner"``.
        **kwargs: Source-specific selectors passed straight through, e.g.
            ``competition="La Liga"`` or ``match=1``.

    Returns:
        A dataframe with exactly :data:`pressing.schema.EVENT_COLUMNS`.
    """
    if _sportio is not None:  # pragma: no cover - exercised only with sportio present
        return _sportio.load_events(source, **kwargs)
    return _load_events_internal(source, **kwargs)


def load_tracking(source: str, **kwargs: Any) -> pd.DataFrame:
    """Load tracking from ``source`` as a canonical ``tracking`` frame.

    Delegates to :func:`sportio.load_tracking` when the shared layer is
    available, otherwise falls back to :func:`_load_tracking_internal`.

    Args:
        source: Source key, e.g. ``"metrica"`` or ``"skillcorner"``.
        **kwargs: Source-specific selectors passed straight through, e.g.
            ``match=1``.

    Returns:
        A dataframe with exactly :data:`pressing.schema.TRACKING_COLUMNS`.
    """
    if _sportio is not None:  # pragma: no cover - exercised only with sportio present
        return _sportio.load_tracking(source, **kwargs)
    return _load_tracking_internal(source, **kwargs)


def _load_events_internal(
    source: str,
    *,
    match: int | str | None = None,
    competition: str | None = None,
    season: str | None = None,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """In-package events loader used when ``sportio`` is absent.

    Args:
        source: Source key. Only ``"metrica"`` is implemented so far.
        match: Match selector within the source (the Metrica game number).
        competition: Competition name filter, unused for Metrica.
        season: Season label filter, unused for Metrica.
        cache_dir: Directory holding the raw source files. Defaults to
            ``data/raw/metrica`` relative to the repo root.

    Returns:
        A dataframe with exactly :data:`pressing.schema.EVENT_COLUMNS`.

    Raises:
        NotImplementedError: For any source other than ``"metrica"``.
    """
    del competition, season
    if source.lower() != "metrica":
        raise NotImplementedError(
            f"internal events loader for {source!r} is not implemented yet; "
            "install sportio or implement the fallback loader"
        )
    return load_metrica_events(int(match or 1), _default_metrica_dir(cache_dir))


def _load_tracking_internal(
    source: str,
    *,
    match: int | str | None = None,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """In-package tracking loader used when ``sportio`` is absent.

    Args:
        source: Source key. Only ``"metrica"`` is implemented so far.
        match: Match selector within the source (the Metrica game number).
        cache_dir: Directory holding the raw source files. Defaults to
            ``data/raw/metrica`` relative to the repo root.

    Returns:
        A dataframe with exactly :data:`pressing.schema.TRACKING_COLUMNS`.

    Raises:
        NotImplementedError: For any source other than ``"metrica"``.
    """
    if source.lower() != "metrica":
        raise NotImplementedError(
            f"internal tracking loader for {source!r} is not implemented yet; "
            "install sportio or implement the fallback loader"
        )
    return load_metrica_tracking(int(match or 1), _default_metrica_dir(cache_dir))


def _default_metrica_dir(cache_dir: Path | None) -> Path:
    """Return the directory the raw Metrica CSVs live in."""
    if cache_dir is not None:
        return Path(cache_dir)
    return Path(__file__).resolve().parents[2] / "data" / "raw" / "metrica"


# --------------------------------------------------------------------------- #
# Raw Metrica parsing
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PlayingDirection:
    """Which way each team attacked, per period, before the match is rotated.

    Attributes:
        home: Period number to ``+1`` (home attacks increasing x) or ``-1``.
        away: Same for the away team. Always the mirror of ``home``.
        goalkeeper: Team name to the detected goalkeeper's ``player_id``.
        gk_mean_x: ``(team, period)`` to the goalkeeper's mean canonical x.
        flip_period: The period that must be mirrored so home attacks ``+x``
            for the whole match, or ``None`` when no period needs it.
    """

    home: dict[int, int]
    away: dict[int, int]
    goalkeeper: dict[str, str]
    gk_mean_x: dict[tuple[str, int], float]
    flip_period: int | None


@dataclass(frozen=True)
class MetricaMatch:
    """A loaded Metrica match: canonical tracking, canonical events, metadata."""

    match_id: str
    tracking: pd.DataFrame
    events: pd.DataFrame
    meta: dict[str, Any] = field(default_factory=dict)


def _metrica_path(data_dir: Path, game: int, kind: str) -> Path:
    """Resolve the path of one raw Metrica CSV.

    Supports the upstream nested layout, the flat double-prefixed layout the
    recon cache also wrote, and a plain flat layout, so fixtures and the real
    cache both work.

    Args:
        data_dir: Directory holding the Metrica sample data.
        game: Game number, 1 or 2.
        kind: One of ``"home"``, ``"away"``, ``"events"``.

    Returns:
        The first candidate path that exists.

    Raises:
        FileNotFoundError: When no candidate exists.
    """
    stems = {
        "home": f"Sample_Game_{game}_RawTrackingData_Home_Team.csv",
        "away": f"Sample_Game_{game}_RawTrackingData_Away_Team.csv",
        "events": f"Sample_Game_{game}_RawEventsData.csv",
    }
    if kind not in stems:
        raise ValueError(f"unknown Metrica file kind: {kind!r}")
    stem = stems[kind]
    data_dir = Path(data_dir)
    candidates = [
        data_dir / f"Sample_Game_{game}" / stem,
        data_dir / f"Sample_Game_{game}__{stem}",
        data_dir / stem,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"no Metrica {kind} file for game {game} under {data_dir}; tried: "
        + ", ".join(str(c) for c in candidates)
    )


def _normalise_player_label(label: str) -> str:
    """Strip internal whitespace from a Metrica player label.

    Game 2 writes ``"Player 26"`` WITH A SPACE in both the tracking header row
    and the events ``From``/``To`` columns. Without this normalization the two
    sides fail to join and game 2 silently loses a player.

    Args:
        label: Raw label, e.g. ``"Player 26"``.

    Returns:
        The label with all whitespace removed, e.g. ``"Player26"``.
    """
    return "".join(str(label).split())


def _read_tracking_header(path: Path) -> tuple[list[str], list[str], list[str]]:
    """Read the THREE metadata rows that precede Metrica tracking data.

    Row 0 repeats the team name over each player's x/y pair, row 1 carries the
    jersey numbers and row 2 carries the real column names, where the x column
    holds ``PlayerNN`` and the y column is blank. The ball is the last named
    column.

    Args:
        path: Path to a raw tracking CSV.

    Returns:
        ``(team_row, jersey_row, name_row)``, each a list of raw cells.

    Raises:
        ValueError: When the file has fewer than three header rows.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows: list[list[str]] = []
        for row in reader:
            rows.append(row)
            if len(rows) == 3:
                break
    if len(rows) < 3:
        raise ValueError(f"{path} has fewer than the three expected header rows")
    return rows[0], rows[1], rows[2]


def _tracking_column_names(name_row: Sequence[str]) -> list[str]:
    """Derive real column names by walking the third header row.

    A named cell becomes ``"<name>_x"`` and the following blank cell becomes
    ``"<prev>_y"``. The first three cells are the metadata columns and keep
    their own names.

    Args:
        name_row: Raw third header row.

    Returns:
        One name per physical column.

    Raises:
        ValueError: When a blank cell appears before any named cell.
    """
    names: list[str] = []
    for index, cell in enumerate(name_row):
        label = cell.strip()
        if index < len(_TRACKING_META_COLUMNS):
            names.append(label or _TRACKING_META_COLUMNS[index])
            continue
        if label:
            names.append(f"{_normalise_player_label(label)}_x")
        else:
            if not names or not names[-1].endswith("_x"):
                raise ValueError(f"unpaired blank column at index {index} in tracking header")
            names.append(f"{names[-1][:-2]}_y")
    return names


@dataclass(frozen=True)
class _Entity:
    """One tracked object: a player or the ball."""

    label: str
    team: str
    jersey: int | None
    x_column: str
    y_column: str


def _tracking_entities(
    team_row: Sequence[str],
    jersey_row: Sequence[str],
    names: Sequence[str],
) -> list[_Entity]:
    """Pair up the header rows into one :class:`_Entity` per tracked object."""
    entities: list[_Entity] = []
    for index, name in enumerate(names):
        if not name.endswith("_x") or index < len(_TRACKING_META_COLUMNS):
            continue
        label = name[:-2]
        if label == _BALL_LABEL:
            entities.append(_Entity(label, "ball", None, name, f"{label}_y"))
            continue
        raw_team = team_row[index].strip().lower() if index < len(team_row) else ""
        raw_jersey = jersey_row[index].strip() if index < len(jersey_row) else ""
        entities.append(
            _Entity(
                label=label,
                team=raw_team or "home",
                jersey=int(raw_jersey) if raw_jersey else None,
                x_column=name,
                y_column=f"{label}_y",
            )
        )
    return entities


def _read_tracking_file(path: Path) -> tuple[pd.DataFrame, list[_Entity]]:
    """Read one raw tracking CSV into a wide frame plus its entity list."""
    team_row, jersey_row, name_row = _read_tracking_header(path)
    names = _tracking_column_names(name_row)
    wide = pd.read_csv(path, skiprows=3, names=names, na_values=["NaN"])
    entities = _tracking_entities(team_row, jersey_row, names)
    return wide, entities


# --------------------------------------------------------------------------- #
# Tracking
# --------------------------------------------------------------------------- #


def load_metrica_tracking(
    game: int,
    data_dir: Path | str,
    *,
    with_velocities: bool = True,
) -> pd.DataFrame:
    """Load one Metrica game's tracking data in the canonical schema.

    Both team files are parsed, converted to canonical metres, rotated to a
    single playing direction and melted to LONG format with one row per
    ``(frame, entity)``. Rows whose position is missing are dropped: verified
    against the raw files, a player's valid span is perfectly contiguous, so a
    missing value means "not on the pitch", never "off camera". The ball is
    missing for 39-41% of frames, which means dead ball.

    Args:
        game: Metrica sample game number, 1 or 2.
        data_dir: Directory holding the Metrica sample CSVs.
        with_velocities: Whether to fill ``vx``/``vy`` via
            :func:`compute_velocities`. When ``False`` they are left as NaN.

    Returns:
        A dataframe with exactly :data:`pressing.schema.TRACKING_COLUMNS`,
        sorted by ``(frame, team, player_id)``. ``DataFrame.attrs`` carries
        ``match_id``, ``source``, ``fps``, ``pitch_length``, ``pitch_width``,
        ``attack_direction`` (``{"home": 1, "away": -1}``), ``flip_period``,
        ``raw_direction`` and ``goalkeeper``.
    """
    data_dir = Path(data_dir)
    match_id = f"metrica_{game}"

    home_wide, home_entities = _read_tracking_file(_metrica_path(data_dir, game, "home"))
    away_wide, away_entities = _read_tracking_file(_metrica_path(data_dir, game, "away"))

    if not home_wide["Frame"].equals(away_wide["Frame"]):
        raise ValueError(f"{match_id}: home and away tracking files disagree on Frame")

    # The ball columns are byte-identical in both files; keep the home copy only.
    away_entities = [e for e in away_entities if e.team != "ball"]

    long = pd.concat(
        [
            _melt_tracking(home_wide, home_entities, match_id),
            _melt_tracking(away_wide, away_entities, match_id),
        ],
        ignore_index=True,
    )
    long = long.sort_values(["frame", "team", "player_id"], kind="stable", ignore_index=True)

    direction = detect_playing_direction(long)
    long = _apply_single_playing_direction(long, direction.flip_period)

    long.attrs.update(
        {
            "match_id": match_id,
            "source": "metrica",
            "game": game,
            "fps": METRICA_FPS,
            "pitch_length": PITCH_LENGTH,
            "pitch_width": PITCH_WIDTH,
            "attack_direction": {"home": 1, "away": -1},
            "flip_period": direction.flip_period,
            "raw_direction": {"home": dict(direction.home), "away": dict(direction.away)},
            "goalkeeper": dict(direction.goalkeeper),
        }
    )

    if with_velocities:
        long = compute_velocities(long)

    return validate_tracking(long)


def _melt_tracking(wide: pd.DataFrame, entities: Sequence[_Entity], match_id: str) -> pd.DataFrame:
    """Melt a wide Metrica tracking frame to the canonical long layout.

    Coordinates are converted from Metrica's normalized top-left frame to
    canonical metres here; the single-playing-direction rotation happens later.

    Args:
        wide: Parsed wide frame with ``Period``/``Frame``/``Time [s]`` and one
            ``_x``/``_y`` pair per entity.
        entities: Entities to emit rows for.
        match_id: Value for the ``match_id`` column.

    Returns:
        A long frame with the canonical tracking columns, positions in metres,
        ``vx``/``vy`` NaN, and rows with a missing position dropped.
    """
    n_frames = len(wide)
    n_entities = len(entities)

    xs = np.empty((n_frames, n_entities), dtype=np.float64)
    ys = np.empty((n_frames, n_entities), dtype=np.float64)
    for column, entity in enumerate(entities):
        xs[:, column] = wide[entity.x_column].to_numpy(dtype=np.float64)
        ys[:, column] = wide[entity.y_column].to_numpy(dtype=np.float64)

    # Metrica: normalized 0-1, origin TOP-LEFT, y increasing DOWNWARD.
    xs *= PITCH_LENGTH
    ys = (1.0 - ys) * PITCH_WIDTH

    keep = np.isfinite(xs.ravel()) & np.isfinite(ys.ravel())

    frames = np.repeat(wide["Frame"].to_numpy(dtype=np.int32), n_entities)[keep]
    periods = np.repeat(wide["Period"].to_numpy(dtype=np.int8), n_entities)[keep]
    times = np.repeat(wide["Time [s]"].to_numpy(dtype=np.float64), n_entities)[keep]

    labels = np.array([None if e.team == "ball" else e.label for e in entities], dtype=object)
    teams = np.array([e.team for e in entities], dtype=object)
    jerseys = np.array(
        [np.nan if e.jersey is None else float(e.jersey) for e in entities], dtype=np.float64
    )

    return pd.DataFrame(
        {
            "match_id": match_id,
            "t": times,
            "period": periods,
            "frame": frames,
            "team": np.tile(teams, n_frames)[keep],
            "player_id": np.tile(labels, n_frames)[keep],
            "jersey": _to_nullable_int16(np.tile(jerseys, n_frames)[keep]),
            "x": xs.ravel()[keep],
            "y": ys.ravel()[keep],
            "vx": np.nan,
            "vy": np.nan,
            "confidence": np.float32(1.0),
        }
    )


def _to_nullable_int16(values: np.ndarray) -> pd.array:
    """Convert a float array with NaN holes to a pandas nullable ``Int16`` array."""
    missing = ~np.isfinite(values)
    filled = np.where(missing, 0.0, values).astype(np.int16)
    return pd.arrays.IntegerArray(filled, missing)


def detect_playing_direction(tracking: pd.DataFrame) -> PlayingDirection:
    """Detect which way each team attacked in each period.

    The goalkeeper is the outfield-or-not player whose mean x sits furthest from
    the halfway line; a team whose goalkeeper averages in its own half of the
    canonical frame is attacking ``+x``. Both the sign consistency between the
    two teams and the half-time flip are asserted, so a silently wrong rotation
    surfaces as an error rather than a mirrored heatmap.

    Args:
        tracking: Canonical long tracking, in metres, BEFORE any rotation.

    Returns:
        A :class:`PlayingDirection` describing both teams in both periods.

    Raises:
        ValueError: When the two teams are not opposite in some period, or when
            a team does not flip at half time.
    """
    players = tracking[tracking["team"] != "ball"]
    if players.empty:
        raise ValueError("cannot detect playing direction: no player rows")

    centre = PITCH_LENGTH / 2.0
    home: dict[int, int] = {}
    away: dict[int, int] = {}
    goalkeeper: dict[str, str] = {}
    gk_mean_x: dict[tuple[str, int], float] = {}

    periods = sorted(int(p) for p in players["period"].unique())
    for team in ("home", "away"):
        side = players[players["team"] == team]
        if side.empty:
            raise ValueError(f"cannot detect playing direction: no rows for team {team!r}")
        for period in periods:
            block = side[side["period"] == period]
            if block.empty:
                raise ValueError(f"no {team} rows in period {period}")
            mean_x = block.groupby("player_id", observed=True)["x"].mean()
            keeper = str((mean_x - centre).abs().idxmax())
            value = float(mean_x.loc[keeper])
            gk_mean_x[(team, period)] = value
            goalkeeper.setdefault(team, keeper)
            sign = 1 if value < centre else -1
            (home if team == "home" else away)[period] = sign

    for period in periods:
        if home[period] != -away[period]:
            raise ValueError(
                f"period {period}: home and away were detected attacking the same way "
                f"(home GK mean x {gk_mean_x[('home', period)]:.2f}, "
                f"away GK mean x {gk_mean_x[('away', period)]:.2f})"
            )
    if len(periods) > 1:
        for team_map, name in ((home, "home"), (away, "away")):
            first = team_map[periods[0]]
            for period in periods[1:]:
                expected = first if (period - periods[0]) % 2 == 0 else -first
                if team_map[period] != expected:
                    raise ValueError(
                        f"{name} did not flip playing direction at half time "
                        f"(period {periods[0]} -> {first}, period {period} -> {team_map[period]})"
                    )

    flip_period = next((p for p in periods if home[p] == -1), None)
    return PlayingDirection(
        home=home,
        away=away,
        goalkeeper=goalkeeper,
        gk_mean_x=gk_mean_x,
        flip_period=flip_period,
    )


def _apply_single_playing_direction(df: pd.DataFrame, flip_period: int | None) -> pd.DataFrame:
    """Mirror one period so the home team attacks ``+x`` for the whole match.

    Args:
        df: Canonical frame carrying ``period`` plus any of ``x``, ``y``,
            ``vx``, ``vy``, ``end_x``, ``end_y``.
        flip_period: Period to mirror, or ``None`` to leave the frame alone.

    Returns:
        A new frame with the mirrored period rotated 180 degrees about the
        centre spot.
    """
    if flip_period is None:
        return df
    out = df.copy()
    out.attrs.update(df.attrs)
    mask = out["period"] == flip_period
    extents = (
        ("x", PITCH_LENGTH),
        ("end_x", PITCH_LENGTH),
        ("y", PITCH_WIDTH),
        ("end_y", PITCH_WIDTH),
    )
    for column, extent in extents:
        if column in out.columns:
            out.loc[mask, column] = extent - out.loc[mask, column]
    for column in ("vx", "vy"):
        if column in out.columns:
            out.loc[mask, column] = -out.loc[mask, column]
    return out


# --------------------------------------------------------------------------- #
# Velocities
# --------------------------------------------------------------------------- #


def compute_velocities(
    tracking: pd.DataFrame,
    *,
    window: int = 7,
    polyorder: int = 1,
    maxspeed: float = 12.0,
    smoothing: bool = True,
) -> pd.DataFrame:
    """Fill ``vx``/``vy`` on a canonical long tracking frame.

    This replicates the Friends of Tracking reference so the pitch-control
    validation comparison stays meaningful: a raw backward difference divided by
    dt, speeds above ``maxspeed`` blanked as position errors, then a
    Savitzky-Golay filter with ``window=7``, ``polyorder=1`` applied SEPARATELY
    PER PERIOD.

    NaN handling is the one place this needs a decision. scipy's
    ``savgol_filter`` with its default ``mode="interp"`` raises
    ``"array must not contain infs or NaNs"`` on any player who was substituted,
    so the reference cannot be run as published at all. The two ways out are
    ``mode="nearest"`` (runs, but smears NaN across the seven frames either side
    of every span boundary and every dead-ball gap) and a NaN-safe wrapper.
    This uses the wrapper: scipy's default ``mode="interp"`` applied to each
    maximal run of finite values inside a period, leaving runs shorter than
    ``window`` unsmoothed. On any run the reference could itself have processed
    the two agree exactly, including the edge extrapolation ``mode="interp"``
    does at run ends, so the pitch-control validation stays a like-for-like
    comparison.

    Args:
        tracking: Canonical long tracking frame.
        window: Savitzky-Golay window length in frames.
        polyorder: Savitzky-Golay polynomial order.
        maxspeed: Raw speed in m/s above which the sample is treated as a
            position error and blanked. Pass ``0`` to disable.
        smoothing: Whether to apply the filter at all.

    Returns:
        A copy of ``tracking`` with ``vx``/``vy`` populated and ``attrs``
        preserved.
    """
    out = tracking.sort_values(
        ["team", "player_id", "frame"], kind="stable", ignore_index=True
    ).copy()
    out.attrs.update(tracking.attrs)

    entity = out["team"].astype(str) + "/" + out["player_id"].astype(str)
    codes = pd.factorize(entity)[0]
    frames = out["frame"].to_numpy(dtype=np.int64)
    periods = out["period"].to_numpy(dtype=np.int64)

    boundary = np.empty(len(out), dtype=bool)
    boundary[0] = True
    boundary[1:] = (
        (codes[1:] != codes[:-1]) | (periods[1:] != periods[:-1]) | (np.diff(frames) != 1)
    )

    dt = np.empty(len(out), dtype=np.float64)
    times = out["t"].to_numpy(dtype=np.float64)
    dt[0] = np.nan
    dt[1:] = np.diff(times)
    dt[boundary] = np.nan

    vx = np.full(len(out), np.nan)
    vy = np.full(len(out), np.nan)
    xs = out["x"].to_numpy(dtype=np.float64)
    ys = out["y"].to_numpy(dtype=np.float64)
    vx[1:] = np.diff(xs) / dt[1:]
    vy[1:] = np.diff(ys) / dt[1:]
    vx[boundary] = np.nan
    vy[boundary] = np.nan

    if maxspeed > 0:
        too_fast = np.hypot(vx, vy) > maxspeed
        vx[too_fast] = np.nan
        vy[too_fast] = np.nan

    if smoothing:
        segment = np.cumsum(boundary)
        for start, stop in _runs(segment):
            _smooth_inplace(vx, start, stop, window, polyorder)
            _smooth_inplace(vy, start, stop, window, polyorder)

    out["vx"] = vx
    out["vy"] = vy
    out = out.sort_values(["frame", "team", "player_id"], kind="stable", ignore_index=True)
    out.attrs.update(tracking.attrs)
    return out


def _runs(labels: np.ndarray) -> Iterator[tuple[int, int]]:
    """Yield ``(start, stop)`` half-open index ranges of equal consecutive labels."""
    if labels.size == 0:
        return
    edges = np.flatnonzero(np.diff(labels)) + 1
    bounds = np.concatenate(([0], edges, [labels.size]))
    for start, stop in zip(bounds[:-1], bounds[1:], strict=True):
        yield int(start), int(stop)


def _smooth_inplace(values: np.ndarray, start: int, stop: int, window: int, polyorder: int) -> None:
    """Savitzky-Golay smooth each maximal finite run of ``values[start:stop]``."""
    block = values[start:stop]
    finite = np.isfinite(block)
    if not finite.any():
        return
    for run_start, run_stop in _runs(finite.astype(np.int8)):
        if not finite[run_start]:
            continue
        length = run_stop - run_start
        if length < window:
            continue
        block[run_start:run_stop] = savgol_filter(
            block[run_start:run_stop],
            window_length=window,
            polyorder=polyorder,
            mode="interp",
        )


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #


def subtype_tokens(subtype: object) -> frozenset[str]:
    """Split a Metrica ``Subtype`` into its hyphen-separated tokens.

    ``Subtype`` is a compound tag, not a flat enum: there are 51 distinct values
    in game 1 and 52 in game 2 and the vocabularies differ, so exact-string
    matching is not robust but token membership is. ``"HEAD-ON TARGET-GOAL"``
    becomes ``{"HEAD", "ON TARGET", "GOAL"}``.

    Args:
        subtype: Raw subtype cell; NaN and empty strings give an empty set.

    Returns:
        The tokens as a frozenset of upper-case strings.
    """
    if subtype is None or pd.isna(subtype):
        return frozenset()
    text = str(subtype).strip()
    if not text:
        return frozenset()
    return frozenset(part.strip().upper() for part in text.split("-") if part.strip())


def event_extra(events: pd.DataFrame, key: str) -> pd.Series:
    """Pull one key out of the per-row ``extra`` dict of a canonical events frame.

    Args:
        events: Canonical events frame.
        key: Key to extract, e.g. ``"end_frame"``.

    Returns:
        A series aligned to ``events.index`` holding ``extra[key]`` per row.
    """
    values = [(payload or {}).get(key) for payload in events["extra"]]
    # An explicit object series: a plain ``map`` would silently turn ``None``
    # into NaN, which loses the difference between "no receiver" and "unknown".
    return pd.Series(values, index=events.index, dtype=object, name=key)


def load_metrica_events(
    game: int,
    data_dir: Path | str,
    *,
    flip_period: int | None | str = "auto",
) -> pd.DataFrame:
    """Load one Metrica game's events in the canonical schema.

    Event ``Start``/``End Time [s]`` equals ``Frame / 25`` exactly, so events and
    tracking share one frame clock and no synchronization is needed.

    Args:
        game: Metrica sample game number, 1 or 2.
        data_dir: Directory holding the Metrica sample CSVs.
        flip_period: Which period to mirror so the home team attacks ``+x``.
            The default ``"auto"`` detects it from the tracking files (cached
            per ``(game, data_dir)`` so a match load pays for it once); pass an
            explicit period number, or ``None`` for no rotation.

    Returns:
        A dataframe with exactly :data:`pressing.schema.EVENT_COLUMNS`.
        ``extra`` holds ``start_frame``, ``end_frame``, ``end_t``,
        ``to_player``, ``subtype_tokens`` and ``event_index`` per row.
    """
    data_dir = Path(data_dir)
    path = _metrica_path(data_dir, game, "events")
    raw = pd.read_csv(path)
    match_id = f"metrica_{game}"

    if flip_period == "auto":
        flip_period = _detect_flip_period(game, str(data_dir.resolve()))

    tokens = raw["Subtype"].map(subtype_tokens)
    team = raw["Team"].astype(str).str.strip().str.lower()
    # Build these as plain lists: a pandas ``map`` turns the ``None`` results
    # back into NaN, and ``extra`` should carry a real ``None``.
    from_player = [_normalise_optional_player(value) for value in raw["From"]]
    to_player = [_normalise_optional_player(value) for value in raw["To"]]

    events = pd.DataFrame(
        {
            "match_id": match_id,
            "competition": "Metrica Sample Data",
            "season": pd.NA,
            "team_id": team,
            "team": team,
            "player_id": from_player,
            "player": from_player,
            "t": raw["Start Time [s]"].astype(float),
            "period": raw["Period"].astype("int8"),
            "type": raw["Type"].astype(str).str.strip(),
            "subtype": raw["Subtype"].where(raw["Subtype"].notna(), None),
            "x": raw["Start X"].astype(float) * PITCH_LENGTH,
            "y": (1.0 - raw["Start Y"].astype(float)) * PITCH_WIDTH,
            "end_x": raw["End X"].astype(float) * PITCH_LENGTH,
            "end_y": (1.0 - raw["End Y"].astype(float)) * PITCH_WIDTH,
            "outcome": [_event_outcome(tok) for tok in tokens],
            "pressure": False,
            "body_part": ["head" if "HEAD" in tok else None for tok in tokens],
        }
    )
    events["extra"] = [
        {
            "start_frame": int(sf),
            "end_frame": int(ef),
            "end_t": float(et),
            "to_player": tp,
            "subtype_tokens": tok,
            "event_index": index,
        }
        for index, (sf, ef, et, tp, tok) in enumerate(
            zip(
                raw["Start Frame"],
                raw["End Frame"],
                raw["End Time [s]"],
                to_player,
                tokens,
                strict=True,
            )
        )
    ]

    events = _apply_single_playing_direction(events, flip_period)
    events.attrs.update(
        {
            "match_id": match_id,
            "source": "metrica",
            "game": game,
            "fps": METRICA_FPS,
            "pitch_length": PITCH_LENGTH,
            "pitch_width": PITCH_WIDTH,
            "attack_direction": {"home": 1, "away": -1},
            "flip_period": flip_period,
        }
    )
    return validate_events(events)


def _normalise_optional_player(label: object) -> str | None:
    """Normalise a ``From``/``To`` cell, mapping missing values to ``None``."""
    if label is None or pd.isna(label):
        return None
    text = _normalise_player_label(label)
    return text or None


def _event_outcome(tokens: frozenset[str]) -> str | None:
    """Derive a coarse outcome label from the subtype tokens."""
    for token in ("GOAL", "ON TARGET", "OFF TARGET", "BLOCKED", "SAVED", "WON", "LOST"):
        if token in tokens:
            return token.lower()
    return None


@lru_cache(maxsize=8)
def _detect_flip_period(game: int, data_dir: str) -> int | None:
    """Return the period that must be mirrored for game ``game``, cached."""
    tracking = load_metrica_tracking(game, Path(data_dir), with_velocities=False)
    return tracking.attrs["flip_period"]


def load_metrica_match(game: int, data_dir: Path | str) -> MetricaMatch:
    """Load one Metrica game's tracking and events with a consistent rotation.

    Args:
        game: Metrica sample game number, 1 or 2.
        data_dir: Directory holding the Metrica sample CSVs.

    Returns:
        A :class:`MetricaMatch` whose tracking and events share one playing
        direction and one frame clock.
    """
    data_dir = Path(data_dir)
    tracking = load_metrica_tracking(game, data_dir)
    events = load_metrica_events(game, data_dir, flip_period=tracking.attrs["flip_period"])
    return MetricaMatch(
        match_id=tracking.attrs["match_id"],
        tracking=tracking,
        events=events,
        meta=dict(tracking.attrs),
    )


# --------------------------------------------------------------------------- #
# Reference frame bridge
# --------------------------------------------------------------------------- #

_REFERENCE_COLUMN_AXIS: Final[dict[str, str]] = {
    "x": "x",
    "end_x": "x",
    "vx": "x",
    "y": "y",
    "end_y": "y",
    "vy": "y",
}


def to_reference_frame(
    df: pd.DataFrame,
    *,
    field_dimen: tuple[float, float] = REFERENCE_FIELD_DIMEN,
    flip: bool = False,
) -> pd.DataFrame:
    """Convert a canonical frame into the Friends of Tracking centred frame.

    The reference puts the origin at the centre circle on a ``field_dimen``
    pitch with ``y`` already increasing upward, so the map is a shift and a
    per-axis scale::

        x_ref = (x / 105 - 0.5) * field_dimen[0]
        y_ref = (y / 68  - 0.5) * field_dimen[1]

    Velocities take the same scale factors but no shift. This is a pure
    geometric transform: it does NOT undo the single-playing-direction rotation.
    The reference keeps whatever direction period 1 happened to have, so when
    the canonical load mirrored period 1 (``attrs["flip_period"] == 1``) pass
    ``flip=True`` to land on the reference's orientation.

    Args:
        df: Canonical frame holding any of ``x``, ``y``, ``vx``, ``vy``,
            ``end_x``, ``end_y``.
        field_dimen: Reference pitch dimensions, ``(length, width)`` in metres.
        flip: Whether to also rotate 180 degrees about the centre spot.

    Returns:
        A copy of ``df`` with the position and velocity columns transformed.
    """
    return _rescale(df, field_dimen, flip, inverse=False)


def from_reference_frame(
    df: pd.DataFrame,
    *,
    field_dimen: tuple[float, float] = REFERENCE_FIELD_DIMEN,
    flip: bool = False,
) -> pd.DataFrame:
    """Invert :func:`to_reference_frame`.

    Args:
        df: Frame in the Friends of Tracking centred coordinate system.
        field_dimen: Reference pitch dimensions used on the way out.
        flip: Must match the ``flip`` used on the way out.

    Returns:
        A copy of ``df`` back in the canonical 105x68 bottom-left-origin frame.
    """
    return _rescale(df, field_dimen, flip, inverse=True)


def _rescale(
    df: pd.DataFrame,
    field_dimen: tuple[float, float],
    flip: bool,
    *,
    inverse: bool,
) -> pd.DataFrame:
    """Shared implementation of the canonical <-> reference coordinate map."""
    length, width = float(field_dimen[0]), float(field_dimen[1])
    scale = {"x": length / PITCH_LENGTH, "y": width / PITCH_WIDTH}
    origin = {"x": length / 2.0, "y": width / 2.0}
    sign = -1.0 if flip else 1.0

    out = df.copy()
    out.attrs.update(df.attrs)
    for column, axis in _REFERENCE_COLUMN_AXIS.items():
        if column not in out.columns:
            continue
        values = out[column].to_numpy(dtype=np.float64)
        shifted = column in ("x", "y", "end_x", "end_y")
        if inverse:
            values = values / sign
            if shifted:
                values = values + origin[axis]
            values = values / scale[axis]
        else:
            values = values * scale[axis]
            if shifted:
                values = values - origin[axis]
            values = values * sign
        out[column] = values
    return out


# --------------------------------------------------------------------------- #
# Demo parquet asset
# --------------------------------------------------------------------------- #

_PARQUET_META_KEY: Final[bytes] = b"pressing.tracking"
#: Positions and velocities are stored as int16 in centimetres (and cm/s).
#: Measured on the real files, over all 3,278,389 game-1 tracking rows: max
#: per-coordinate error 0.5000 cm and RMS 0.2887 cm (as a 2D distance, 0.7071 cm
#: max and 0.4082 cm RMS), and the required int16 range is only -478..11130.
_CM: Final[float] = 100.0


def write_tracking_parquet(
    tracking: pd.DataFrame,
    path: Path | str,
    *,
    compression_level: int = 19,
    include_velocities: bool = False,
) -> Path:
    """Write a canonical tracking frame as the compact demo parquet asset.

    Coordinates are quantised to int16 centimetres and the file is written with
    ``compression="zstd"`` and ``use_dictionary=["team"]``. That last argument
    is the single largest sizing lever and it is counter-intuitive: pyarrow's
    DEFAULT dictionary encoding makes this data substantially LARGER (measured
    on game 1 at native 25 Hz: 12.11 MB with dictionaries everywhere, against
    7.75 MB with dictionaries on ``team`` only and 7.76 MB with none at all --
    so ``team`` is the only column a dictionary helps, and only barely).

    Velocities are omitted by default because they are pure smoothed derivatives
    of the positions already in the file, and storing them nearly doubles it
    (measured on game 1: 7,754,194 bytes without, 13,852,088 bytes with, i.e.
    7.75 MB against 13.85 MB).

    That trade is paid back on read, and it is not free: see
    :func:`read_tracking_parquet`.

    Args:
        tracking: Canonical long tracking frame.
        path: Destination ``.parquet`` path.
        compression_level: zstd level; 19 is what the sizing was measured at.
        include_velocities: Store ``vx``/``vy`` as int16 cm/s rather than
            recomputing them on read.

    Returns:
        The path written.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    columns = {
        "period": pa.array(tracking["period"].to_numpy(dtype=np.int8), pa.int8()),
        "frame": pa.array(tracking["frame"].to_numpy(dtype=np.int32), pa.int32()),
        "t_ms": pa.array(
            np.rint(tracking["t"].to_numpy(dtype=np.float64) * 1000.0).astype(np.int32),
            pa.int32(),
        ),
        "team": pa.array(tracking["team"].astype(str), pa.string()),
        "player_id": pa.array(
            tracking["player_id"].where(tracking["player_id"].notna(), None), pa.string()
        ),
        "jersey": pa.array(tracking["jersey"].astype("Int16"), pa.int16()),
        "x_cm": _quantise(tracking["x"]),
        "y_cm": _quantise(tracking["y"]),
        "confidence": pa.array(tracking["confidence"].to_numpy(dtype=np.float32), pa.float32()),
    }
    if include_velocities:
        columns["vx_cms"] = _quantise(tracking["vx"])
        columns["vy_cms"] = _quantise(tracking["vy"])

    meta = dict(tracking.attrs)
    table = pa.table(columns).replace_schema_metadata(
        {_PARQUET_META_KEY: json.dumps(meta, default=str).encode("utf-8")}
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=compression_level,
        use_dictionary=["team"],
    )
    return path


def read_tracking_parquet(
    path: Path | str,
    *,
    recompute_velocities: bool = True,
) -> pd.DataFrame:
    """Read back an asset written by :func:`write_tracking_parquet`.

    Cost, measured on the game-1 asset (3,278,389 rows), median of five runs:
    the default path takes 1.79 s, of which the parquet read itself is 0.05 s
    and assembling the canonical frame brings it to 0.20 s. The remaining ~1.6 s
    is :func:`compute_velocities` rerunning the Savitzky-Golay smoothing that
    ``write_tracking_parquet`` chose not to store. Reading a file written with
    ``include_velocities=True`` instead takes 0.21 s, so the 6.1 MB saved on
    disk costs roughly 1.6 s on every read. That is the right way round for a
    demo asset committed to git and read once per run, and the wrong way round
    for anything read repeatedly.

    Args:
        path: Path to the ``.parquet`` file.
        recompute_velocities: When the file has no stored velocities, rerun
            :func:`compute_velocities` so the frame comes back complete. Pass
            ``False`` to leave ``vx``/``vy`` NaN.

    Returns:
        A dataframe with exactly :data:`pressing.schema.TRACKING_COLUMNS`, with
        ``attrs`` restored from the file's key-value metadata.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    frame = table.to_pandas()
    stored_velocities = "vx_cms" in frame.columns

    out = pd.DataFrame(
        {
            "match_id": None,
            "t": frame["t_ms"].to_numpy(dtype=np.float64) / 1000.0,
            "period": frame["period"].to_numpy(dtype=np.int8),
            "frame": frame["frame"].to_numpy(dtype=np.int32),
            "team": frame["team"].astype(str),
            "player_id": frame["player_id"],
            "jersey": pd.array(frame["jersey"], dtype="Int16"),
            "x": _dequantise(frame["x_cm"]),
            "y": _dequantise(frame["y_cm"]),
            "vx": _dequantise(frame["vx_cms"]) if stored_velocities else np.nan,
            "vy": _dequantise(frame["vy_cms"]) if stored_velocities else np.nan,
            "confidence": frame["confidence"].to_numpy(dtype=np.float32),
        }
    )

    raw_meta = (table.schema.metadata or {}).get(_PARQUET_META_KEY)
    if raw_meta is not None:
        meta = _restore_int_keys(json.loads(raw_meta.decode("utf-8")))
        out["match_id"] = meta.get("match_id")
        out.attrs.update(meta)

    if not stored_velocities and recompute_velocities:
        out = compute_velocities(out)
    return validate_tracking(out)


def _restore_int_keys(payload: Any) -> Any:
    """Turn JSON's stringified integer dict keys back into ints, recursively."""
    if isinstance(payload, dict):
        restored = {}
        for key, value in payload.items():
            new_key = int(key) if isinstance(key, str) and key.lstrip("-").isdigit() else key
            restored[new_key] = _restore_int_keys(value)
        return restored
    return payload


def _quantise(values: pd.Series) -> Any:
    """Quantise a metre-valued column to int16 centimetres, NaN-aware."""
    import pyarrow as pa

    raw = values.to_numpy(dtype=np.float64)
    finite = np.isfinite(raw)
    scaled = np.zeros(raw.shape, dtype=np.int16)
    if finite.any():
        rounded = np.rint(raw[finite] * _CM)
        if np.abs(rounded).max() > np.iinfo(np.int16).max:
            raise ValueError("value out of int16 centimetre range; cannot quantise")
        scaled[finite] = rounded.astype(np.int16)
    return pa.array(scaled, pa.int16(), mask=~finite)


def _dequantise(values: pd.Series) -> np.ndarray:
    """Invert :func:`_quantise` back to metres."""
    return values.to_numpy(dtype="float64") / _CM


def iter_metrica_games(data_dir: Path | str) -> Iterable[int]:
    """Yield the Metrica game numbers present under ``data_dir``."""
    for game in (1, 2, 3):
        try:
            _metrica_path(Path(data_dir), game, "events")
        except FileNotFoundError:
            continue
        yield game
