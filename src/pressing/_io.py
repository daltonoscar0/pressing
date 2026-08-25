"""Loader shim over the shared data layer.

The suite is meant to sit on top of ``sportio``, a shared package that loads
every open soccer data source and normalizes it to the canonical ``events`` and
``tracking`` frames described in :mod:`pressing.schema`. That package does not
exist on this machine yet, so this module follows the fallback the sibling
setpiece spec spells out: prefer ``sportio`` whenever it is importable, and
otherwise fall back to minimal loaders implemented here.

Nothing else in the package touches raw source schemas. When ``sportio`` lands,
the internal loaders below can be deleted and the delegation path will already
be wired up.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import pandas as pd

__all__ = [
    "HAVE_SPORTIO",
    "load_events",
    "load_tracking",
    "backend_name",
]

try:  # pragma: no cover - depends on whether the sibling package is installed
    import sportio as _sportio
except ImportError:  # pragma: no cover
    _sportio = None

HAVE_SPORTIO: Final[bool] = _sportio is not None


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
        A dataframe with exactly :data:`pressing.schema.TRACKING_COLUMNS`,
        sampled at 10Hz or better.
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
    """Minimal in-package events loader used when ``sportio`` is absent.

    Downloads obey the suite politeness rules: at most one request per second
    per host, exponential backoff on failure, and every fetched file cached
    under ``data/raw/`` so a file is never downloaded twice.

    Args:
        source: Source key, e.g. ``"statsbomb"`` or ``"metrica"``.
        match: Match selector within the source, if the source needs one.
        competition: Competition name filter, for sources organized by competition.
        season: Season label filter.
        cache_dir: Override for the download cache. Defaults to ``data/raw/``.

    Returns:
        A dataframe with exactly :data:`pressing.schema.EVENT_COLUMNS`.

    Raises:
        NotImplementedError: Always, for now.
    """
    raise NotImplementedError(
        f"internal events loader for {source!r} is not implemented yet; "
        "install sportio or implement the fallback loader"
    )


def _load_tracking_internal(
    source: str,
    *,
    match: int | str | None = None,
    cache_dir: Path | None = None,
    target_hz: float = 10.0,
) -> pd.DataFrame:
    """Minimal in-package tracking loader used when ``sportio`` is absent.

    Same politeness and caching rules as :func:`_load_events_internal`.

    Args:
        source: Source key, e.g. ``"metrica"`` or ``"skillcorner"``.
        match: Match selector within the source.
        cache_dir: Override for the download cache. Defaults to ``data/raw/``.
        target_hz: Sample rate to resample to when the source differs.

    Returns:
        A dataframe with exactly :data:`pressing.schema.TRACKING_COLUMNS`.

    Raises:
        NotImplementedError: Always, for now.
    """
    raise NotImplementedError(
        f"internal tracking loader for {source!r} is not implemented yet; "
        "install sportio or implement the fallback loader"
    )
