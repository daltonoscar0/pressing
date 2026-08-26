"""pressing — counter-press report generator.

Turns a tracking or event file into a match report a coach can hand out:
pitch-control heatmaps at every possession loss, a five-second recovery metric
per player and phase, and a team comparison.

Importing this package points ``MPLCONFIGDIR`` at a repo-local cache directory
before matplotlib is ever imported. See :func:`use_repo_matplotlib_cache` for
what that buys and how to turn it off, and :func:`warm_matplotlib_cache` for
how a long-running entry point can pay for it in the background.

It also stamps a monotonic clock reading, :func:`elapsed_since_import`, because
every heavy import in this tool is lazy and therefore happens after this module
does. Together with :func:`cpu_seconds` and :func:`load_average` that is what
lets an entry point report a cost a reader cannot mistake for something
smaller than it is.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

__version__ = "0.1.0"

__all__ = [
    "SATURATED_CPU_SHARE",
    "__version__",
    "adaptive_jobs",
    "cpu_availability",
    "cpu_seconds",
    "elapsed_since_import",
    "load_average",
    "matplotlib_cache_dir",
    "use_repo_matplotlib_cache",
    "warm_matplotlib_cache",
]

#: Monotonic clock at the moment this module was first imported.
#:
#: This is as early as any Python in this package can observe, and it is early
#: enough to matter: ``pressing/__init__.py`` imports nothing but the standard
#: library, and every heavy dependency -- numpy, pandas, pyarrow, matplotlib,
#: reportlab -- is imported lazily inside the subcommand that needs it. So the
#: gap between this stamp and now covers all of them. What it cannot cover is
#: interpreter start-up and ``import typer``, which run before it.
_IMPORTED_AT: float = time.monotonic()

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


# --------------------------------------------------------------------------- #
# What a run actually cost, and how much of the machine it may take
# --------------------------------------------------------------------------- #


def elapsed_since_import() -> float:
    """Seconds of wall clock since this package was first imported.

    The number an entry point should quote next to its own stage clock. A
    stage clock starts when the work starts and so cannot see the imports that
    had to finish before the work could begin; on a cold clone those imports
    have been measured as the single largest cost of the whole command. This
    starts at ``import pressing``, which is the first line of this package to
    run and precedes every heavy import in it.

    Still not the full cost of the process: interpreter start-up and
    ``import typer`` happen before this module is reached. Say so wherever it
    is printed rather than letting it read as end to end.

    Returns:
        Seconds since import, from :func:`time.monotonic`.
    """
    return time.monotonic() - _IMPORTED_AT


def cpu_seconds() -> tuple[float, float] | None:
    """User and system CPU seconds burned by this process and its children.

    The load-independent counterpart to wall clock, and the honest way to
    report the cost of a run on a machine that is shared with anything else:
    CPU time is what the work costs, wall clock is what the queue costs. It
    covers the whole process from ``exec``, so unlike :func:`elapsed_since_import`
    it does include interpreter start-up and every import.

    ``RUSAGE_CHILDREN`` only accounts for children that have been reaped, which
    for this package means the font-cache warmer once waited on and the scoring
    pool once shut down. Both are reaped before a run reports anything.

    Returns:
        ``(user, system)`` in seconds, or None on a platform with no
        :mod:`resource` module.
    """
    try:
        import resource
    except ImportError:  # pragma: no cover - not reachable on macOS or Linux
        return None
    me = resource.getrusage(resource.RUSAGE_SELF)
    kids = resource.getrusage(resource.RUSAGE_CHILDREN)
    return (me.ru_utime + kids.ru_utime, me.ru_stime + kids.ru_stime)


def load_average() -> float | None:
    """The one-minute load average, where the OS publishes one.

    Reported next to every timing this tool prints, so that a wall clock taken
    on a busy machine carries the evidence that it was. It is deliberately NOT
    what :func:`adaptive_jobs` decides on -- see there for the measurement that
    ruled it out.

    Returns:
        The 1-minute figure from :func:`os.getloadavg`, or None on a platform
        without it (Windows). Not clamped and not scaled by the core count --
        callers that want "load per core" divide it themselves.
    """
    try:
        return float(os.getloadavg()[0])
    except (AttributeError, OSError):  # pragma: no cover - Windows only
        return None


#: One probe of :func:`cpu_availability`, in seconds of wall clock.
_PROBE_SECONDS: float = 0.08

#: How many probes :func:`cpu_availability` takes before answering. The answer
#: is the *best* of them, so a single unlucky 80 ms slice -- another process
#: waking up, a page fault -- cannot talk the tool into the serial path.
_PROBE_SAMPLES: int = 3

#: Below this share of one core, the machine is treated as saturated and
#: scoring runs serially.
#:
#: The threshold sits in a gap that was measured from both sides, on the
#: bundled match, one fresh process per run:
#:
#: * share 0.94-0.96 (seven other agents on the box, one-minute load average
#:   26.7-27.7): six workers scored the match in 8.0 s against 15.5-26.3 s
#:   serial. Six workers win by 2-3x.
#: * share 0.73 (the same box plus twenty-four CPU burners of mine): the whole
#:   ``pressing demo`` command took 215.7 s serial against 275.8 s on six
#:   workers -- 91.0 s of scoring against 149.1 s -- and 32.6 s of CPU against
#:   45.5 s. One worker wins by 1.6x on the scoring stage, and the load average
#:   was *higher* during the serial run (47 rising to 84) than during the
#:   six-worker one (72 falling to 56), so the comparison understates it.
#: * share 0.31-0.58 (the same box plus eight burners): six workers took 79.0 s
#:   and 108.1 s of scoring against 35.8 s and 44.6 s serial. One worker wins
#:   by 2.4x.
#:
#: So serial wins everywhere at or below 0.73 and the cap wins at 0.94 and
#: above. Nothing was measured in between, and this constant is the midpoint of
#: that gap: it claims nothing about the gap's interior beyond symmetry of
#: ignorance about it.
SATURATED_CPU_SHARE: float = 0.85


def cpu_availability(
    samples: int = _PROBE_SAMPLES, seconds: float = _PROBE_SECONDS
) -> float:
    """What fraction of one core this process is actually being given.

    Spins on the clock for a few short slices and compares the CPU time it was
    credited with against the wall clock that passed. On a machine with a core
    to spare that ratio is close to 1; on a saturated one it is the share the
    scheduler is willing to hand over.

    It is a direct measurement of the thing that matters, which the load
    average is not: at a one-minute load average of 27 this returned 0.94-0.96
    on the machine this was written on, and at a load average of 29-38 on the
    same machine minutes later, with eight CPU burners running, it returned
    0.31-0.58. The load averages overlap; the shares do not.

    The known limitation is that it times one thread, so it reports whether
    *a* core is free rather than whether ``cap`` of them are. Under mixed load
    it also moves: the same eight burners gave readings from 0.31 to 0.78
    within a few minutes, depending on what the rest of the machine happened to
    be doing. Both are reasons the answer is fed to a threshold rather than
    scaled into a worker count.

    Deliberately a busy-wait and not a sleep, and deliberately CPU with no I/O
    in it: a probe that touched a file would report a cold disk as a busy CPU,
    which on a fresh clone is exactly backwards.

    Cost is ``samples * seconds`` of one core -- 0.24 s at the defaults, against
    a run measured in tens of seconds.

    Args:
        samples: How many slices to time. The best is returned.
        seconds: Wall clock per slice.

    Returns:
        A share in ``(0.0, 1.0]``. A probe that measured nothing at all returns
        1.0 rather than 0.0, so that a clock this cannot read means "carry on
        as before" and not "run everything serially for ever".
    """
    best = 0.0
    for _ in range(max(1, samples)):
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        while time.perf_counter() - wall_start < seconds:
            pass
        wall = time.perf_counter() - wall_start
        used = time.process_time() - cpu_start
        if wall > 0.0:
            best = max(best, used / wall)
    return min(1.0, best) if best > 0.0 else 1.0


def adaptive_jobs(
    cap: int | None = None,
    *,
    availability: float | None = None,
    load: float | None = None,
) -> int:
    """How many scoring workers to start, given the machine we are actually on.

    The policy, in one line: **use the full cap when the machine will give us a
    core, and the serial path when it will not.**

    ::

        workers = cap  if cpu_availability() >= SATURATED_CPU_SHARE else 1

    ``cap`` defaults to :func:`pressing.counterpress.default_jobs`, which is
    ``min(MAX_USEFUL_JOBS, cpu_count)`` and is itself measured. On an idle
    machine this returns exactly what the old fixed default returned, so
    nothing changes where nothing needed to change.

    Why not the load average
    ------------------------
    Because it was tried first and it does not work on this machine. Scoring
    the bundled match at six workers, one fresh process per run:

    ::

        one-minute load average 33.99  ->    7.99 s     (ambient load only)
        one-minute load average 33.77  ->  108.14 s     (plus eight burners)

    Same command, same machine, the same load average to two decimal places,
    13.5x apart -- and the worker count that wins is six in the first case and
    one in the second. No threshold on the load average separates them, so a
    policy keyed on it is a coin flip with a 2-3x penalty on the wrong side.
    :func:`cpu_availability` separates them cleanly and is what this decides on.

    The load average is still consulted, but only for the one thing it can be
    trusted for: a machine reporting less than one runnable thread cannot be
    saturated, so the probe is skipped and the cap used. That is what keeps the
    probe off the common developer path.

    Three properties worth stating because they are what make it safe to
    default to:

    * **It never changes an answer.** Worker count is a scheduling decision;
      every column :func:`pressing.counterpress.score_losses` produces is
      bit-identical at any ``jobs``. Asserted on the real match in
      ``tests/test_demo.py``.
    * **It is bounded below by one**, so the serial path is the floor and a
      wildly loaded machine still finishes.
    * **An explicit ``--jobs`` always wins**, including ``--jobs 1`` and
      including a value larger than the cap. This function is consulted only
      when the caller passed nothing.

    Args:
        cap: Most workers to allow. Defaults to
            :func:`pressing.counterpress.default_jobs`.
        availability: CPU share to decide on. Defaults to measuring it with
            :func:`cpu_availability`; passing it is how the tests pin a machine
            state without needing one.
        load: One-minute load average, for the idle short-circuit only.
            Defaults to :func:`load_average`.

    Returns:
        A worker count of at least one, never above ``cap``.
    """
    if cap is None:
        from pressing.counterpress import default_jobs

        cap = default_jobs()
    cap = max(1, int(cap))
    if cap == 1:
        return 1
    if availability is None:
        if load is None:
            load = load_average()
        if load is not None and load < 1.0:
            return cap
        availability = cpu_availability()
    return cap if float(availability) >= SATURATED_CPU_SHARE else 1


use_repo_matplotlib_cache()
