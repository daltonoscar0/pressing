"""Season mode: league-wide player rankings joined to market values.

Runs the event proxy across every match of a competition, aggregates
counter-press value per 90 minutes per player, and joins the result to
Transfermarkt market values via a public mirror. When the mirror is unreachable
the join falls back to a cached snapshot under ``data/``, and the snapshot date
is reported alongside the output so nobody mistakes stale values for current
ones.

Downloads follow the suite politeness rules: one request per second per host,
exponential backoff, everything cached under ``data/raw/``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def season_table(source: str, competition: str, season: str | None = None) -> pd.DataFrame:
    """Rank players in a competition by counter-press value per 90.

    Args:
        source: Event data source key, e.g. ``"statsbomb"``.
        competition: Competition name, e.g. ``"La Liga"``.
        season: Optional season label. Defaults to the latest available.

    Returns:
        One row per player with minutes played, losses pressed, counter-press
        value per 90, and a league rank.
    """
    raise NotImplementedError


def load_market_values(cache_dir: Path | None = None) -> tuple[pd.DataFrame, str]:
    """Load Transfermarkt market values.

    Tries the public mirror first, then the cached snapshot bundled under
    ``data/``.

    Args:
        cache_dir: Override for the download cache. Defaults to ``data/raw/``.

    Returns:
        ``(values, as_of)`` where ``values`` has ``player``, ``team`` and
        ``market_value_eur``, and ``as_of`` is the ISO date the values were
        captured.
    """
    raise NotImplementedError


def join_market_values(table: pd.DataFrame, values: pd.DataFrame) -> pd.DataFrame:
    """Join a season table to market values by player and team.

    Args:
        table: Season table from :func:`season_table`.
        values: Market values from :func:`load_market_values`.

    Returns:
        The season table with a ``market_value_eur`` column. Players with no
        match keep a null value rather than being dropped.
    """
    raise NotImplementedError
