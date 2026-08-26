"""pressing — counter-press report generator.

Turns a tracking or event file into a match report a coach can hand out:
pitch-control heatmaps at every possession loss, a five-second recovery metric
per player and phase, and a team comparison.

Importing this package points ``MPLCONFIGDIR`` at a repo-local cache directory
before matplotlib is ever imported. See :func:`use_repo_matplotlib_cache` for
what that buys and how to turn it off, and :func:`warm_matplotlib_cache` for
how a long-running entry point can pay for it in the background.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "matplotlib_cache_dir",
    "use_repo_matplotlib_cache",
    "warm_matplotlib_cache",
]

#: Where the repo-local matplotlib cache lives, relative to the repo root.
#: ``out/`` is already gitignored, and the contents are a pure cache: deleting
#: it costs one font-cache rebuild and nothing else.
_CACHE_RELATIVE: tuple[str, ...] = ("out", ".mplcache")


def matplotlib_cache_dir() -> Path | None:
    """The repo-local matplotlib cache directory, if we are in a checkout.

    Returns:
        ``<repo>/out/.mplcache``, or None when the package is not being
        imported from a source checkout (an installed wheel has no repo root,
        and writing into ``site-packages/../..`` would be wrong).
    """
    root = Path(__file__).resolve().parents[2]
    if not (root / "pyproject.toml").is_file():
        return None
    return root.joinpath(*_CACHE_RELATIVE)


def use_repo_matplotlib_cache() -> Path | None:
    """Point ``MPLCONFIGDIR`` at the repo-local cache, unless told otherwise.

    On a fresh clone matplotlib spends its first import building a font cache.
    Measured on this machine, in a process that imports matplotlib, pandas and
    reportlab: 30.2-36.7 s cold against 2.4-2.6 s warm, out of a 120 s demo
    budget. That cost is paid once per cache directory, so the point of a
    repo-local one is that the whole repo -- test runs, notebooks, every demo
    after the first -- shares a single warm cache that survives even where
    ``HOME`` does not, as on a throwaway CI container.

    This runs at package import because it has to: it must happen before
    anything imports matplotlib, and ``pressing`` is the one module every entry
    point loads first. It is deliberately the smallest intrusion that works --
    one ``os.environ`` key, no directory created, no import triggered -- and it
    stands aside completely if:

    - ``MPLCONFIGDIR`` is already set, so an explicit choice always wins;
    - ``matplotlib`` is already imported, when setting it would do nothing;
    - the package is not running from a source checkout.

    It does NOT warm the cache. Warming is the entry point's job -- see
    :func:`warm_matplotlib_cache`.

    Returns:
        The directory now in ``MPLCONFIGDIR``, or None if nothing was changed.
    """
    if os.environ.get("MPLCONFIGDIR"):
        return None
    if "matplotlib" in sys.modules:
        return None
    cache = matplotlib_cache_dir()
    if cache is None:
        return None
    os.environ["MPLCONFIGDIR"] = str(cache)
    return cache


#: What a child process runs to build the font cache and nothing else.
_WARM_SOURCE = "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot"


def warm_matplotlib_cache() -> subprocess.Popen[bytes] | None:
    """Start building matplotlib's font cache in a background process.

    On a fresh clone the first matplotlib import scans the system's fonts and
    writes a cache. Measured here, in a process that imports matplotlib, pandas
    and reportlab: 15.7-16.3 s with an empty cache against 1.5 s with a warm
    one -- and an audit on a colder machine measured 57.7 s. That is dead time
    at the front of a run that has a two-minute budget.

    It does not have to be dead. The cache is a file, and any process can write
    it, so an entry point that has real work to do first -- loading tracking
    data, detecting losses, scoring them -- can start a child on the font cache
    and get on with that work. Both finish, and the parent's own matplotlib
    import then finds the cache already there. Measured over load + score +
    aggregate on Metrica game 1 with an empty cache, then importing matplotlib
    and reportlab: 27.32 s without this and 18.61 s with it. With a warm cache
    it does nothing at all and costs nothing: 12.47 s either way.

    Usage, and the one rule: WAIT for the handle before importing matplotlib in
    this process, or the two of them build the cache twice::

        warm = pressing.warm_matplotlib_cache()
        tracking, events = load(...)          # the expensive part
        scored = score_losses(tracking, events)
        if warm is not None:
            warm.wait()                       # by now, usually instant
        from pressing import plots            # imports matplotlib, warm

    Returns:
        A handle to wait on, or None if there is nothing to do -- matplotlib is
        already imported here, the cache directory already has a font list, or
        the child could not be started. None always means "just carry on".
    """
    if "matplotlib" in sys.modules:
        return None
    cache = os.environ.get("MPLCONFIGDIR")
    if cache and any(Path(cache).glob("fontlist-*.json")):
        return None
    try:
        return subprocess.Popen(
            [sys.executable, "-c", _WARM_SOURCE],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return None


use_repo_matplotlib_cache()
