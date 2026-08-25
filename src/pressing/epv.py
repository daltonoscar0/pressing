"""Expected possession value surface.

The value of possessing the ball at each location on the pitch, used to weight
pitch control so that regaining the ball in a dangerous area counts for more
than regaining it in front of your own keeper.

Provenance, stated up front because it matters for how the report labels these
numbers: the bundled grid is the Friends of Tracking reference
``EPV_grid.csv``, copied verbatim under its MIT licence from commit
``7ed3ff857940d3f4611be55f3c08c79525c32171``. It is **not** our own estimate,
and upstream ships it with no published provenance and no fitting code. See
``data/epv/PROVENANCE.md``.

Coordinates follow the canonical project frame: metres on a 105 x 68 pitch,
origin bottom-left, ``y`` increasing upward. The grid itself is ``(32, 50)``
with **row 0 at the lowest y** and column 49 at the goal being attacked when
``attack_direction == +1`` -- the same orientation as a pitch control surface,
so the two multiply elementwise with no flipping.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np

from pressing.pitchcontrol import REFERENCE_FIELD_DIMEN
from pressing.schema import PITCH_LENGTH, PITCH_WIDTH

__all__ = [
    "DEFAULT_EPV_PATH",
    "EPV_ATTRIBUTION",
    "EPV_GRID_FILENAME",
    "epv_at",
    "epv_at_reference_frame",
    "epv_surface",
    "expected_epv",
    "expected_epv_added",
    "load_epv",
    "load_epv_grid",
    "resample_epv",
]

#: Filename of the committed copy of the Friends of Tracking grid.
EPV_GRID_FILENAME: str = "fot_epv_grid_32x50.csv"

# Candidate roots, in priority order. The first is the repository checkout
# (``data/epv/`` sits outside the package, so it is not inside a wheel); the
# second supports a future build that ships the grid as package data.
_EPV_SEARCH_PATHS: tuple[Path, ...] = (
    Path(__file__).resolve().parents[2] / "data" / "epv" / EPV_GRID_FILENAME,
    Path(__file__).resolve().parent / "data" / "epv" / EPV_GRID_FILENAME,
)


def _default_epv_path() -> Path:
    """Locate the bundled EPV grid.

    Returns:
        The first candidate path that exists, or the first candidate as-is so
        that the caller raises a :class:`FileNotFoundError` naming it.
    """
    for candidate in _EPV_SEARCH_PATHS:
        if candidate.is_file():
            return candidate
    return _EPV_SEARCH_PATHS[0]


#: The committed copy of the Friends of Tracking grid.
DEFAULT_EPV_PATH: Path = _default_epv_path()

#: One-line credit for any report that prints an EPV number. The grid is not
#: ours and must never be presented as if it were.
EPV_ATTRIBUTION: str = (
    "EPV surface: Friends of Tracking reference grid (LaurieOnTracking, MIT, "
    "commit 7ed3ff8). Not fitted by this tool."
)

#: The reference's lookup subtracts this from a coordinate before flooring, so
#: that a point exactly on the far touchline lands in the last cell rather than
#: one past it. Reproduced exactly, rescaled with the frame.
_REFERENCE_EPS: float = 0.0001


@lru_cache(maxsize=4)
def _load_cached(path: str) -> np.ndarray:
    """Read and cache an EPV csv.

    Args:
        path: Filesystem path to a comma-separated grid.

    Returns:
        A read-only ``(n_y, n_x)`` float64 array.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a 2-D numeric grid.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"EPV grid not found at {p}. The bundled grid lives at {DEFAULT_EPV_PATH}."
        )
    grid = np.loadtxt(p, delimiter=",", dtype=np.float64)
    if grid.ndim != 2:
        raise ValueError(f"EPV grid at {p} must be 2-D, got shape {grid.shape}")
    grid.setflags(write=False)
    return grid


def load_epv(path: str | Path | None = None) -> np.ndarray:
    """Load the EPV grid.

    The bundled default is the Friends of Tracking reference grid: a ``(32,
    50)`` array in which **row 0 is the most negative y** (the bottom
    touchline) and the last column is the goal being attacked under
    ``attack_direction == +1``. That orientation is the same as a pitch control
    surface, which is why the two can be multiplied elementwise.

    This grid is a reference artefact, not an estimate produced by this
    package. Upstream publishes neither its provenance nor the code that fitted
    it, so anything user-facing that quotes an EPV number should carry
    :data:`EPV_ATTRIBUTION`.

    Args:
        path: Optional path to a grid csv. Defaults to the bundled grid.

    Returns:
        A read-only ``(n_y, n_x)`` float64 array. The returned array is cached
        and shared; copy it before modifying.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a 2-D numeric grid.
    """
    return _load_cached(str(_default_epv_path() if path is None else Path(path)))


def load_epv_grid(path: str | Path | None = None) -> np.ndarray:
    """Alias of :func:`load_epv`.

    Args:
        path: Optional path to a grid csv.

    Returns:
        A read-only ``(n_y, n_x)`` float64 array.
    """
    return load_epv(path)


def epv_surface(
    attack_direction: int = 1,
    shape: tuple[int, int] | None = None,
    grid: np.ndarray | None = None,
) -> np.ndarray:
    """The EPV grid oriented for a playing direction, optionally resampled.

    Args:
        attack_direction: ``+1`` if the team in possession attacks ``+x``,
            ``-1`` otherwise. ``-1`` mirrors the grid left-right.
        shape: Optional ``(n_y, n_x)`` to resample to. Defaults to the grid's
            native shape.
        grid: Optional EPV grid. Defaults to the bundled one.

    Returns:
        A fresh ``(n_y, n_x)`` float64 array.

    Raises:
        ValueError: If ``attack_direction`` is not ``+1`` or ``-1``.
    """
    if attack_direction not in (1, -1):
        raise ValueError(f"attack_direction must be +1 or -1, got {attack_direction!r}")
    base = load_epv() if grid is None else np.asarray(grid, dtype=np.float64)
    oriented = np.fliplr(base) if attack_direction == -1 else base
    oriented = np.array(oriented, dtype=np.float64)
    if shape is None or tuple(shape) == oriented.shape:
        return oriented
    return resample_epv(oriented, tuple(shape))


def resample_epv(grid: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-cell resample of an EPV grid onto a different resolution.

    Both grids are taken to cover the same rectangle, so cell centres are
    matched proportionally. The index arithmetic is exactly :func:`epv_at`
    evaluated at each target cell centre -- a floor, not an interpolation --
    so the two can never disagree.

    One consequence to be aware of: the floor convention puts a target centre
    that lands precisely on a source cell boundary into the *upper* source
    cell, which is not mirror-symmetric. Resampling the y-symmetric bundled
    grid to a shape whose cell centres hit boundaries (68 rows from 32, say)
    therefore gives a result that is symmetric only to within one source row.
    Target shapes that are integer multiples of the source (64 rows from 32)
    have no ties and stay exactly symmetric. This is inherited from the
    reference's own lookup rule and is deliberately not papered over.

    Args:
        grid: ``(n_y, n_x)`` source grid.
        shape: Target ``(n_y, n_x)``.

    Returns:
        A ``(n_y, n_x)`` float64 array at the target resolution.

    Raises:
        ValueError: If the target shape is not two positive integers.
    """
    src = np.asarray(grid, dtype=np.float64)
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError(f"shape must be two positive integers, got {shape!r}")
    n_y, n_x = int(shape[0]), int(shape[1])
    src_y, src_x = src.shape
    # (i + 0.5) / n is the target cell centre as a fraction of the pitch; the
    # epsilon is the reference's, expressed in the same fractional units.
    eps_x = _REFERENCE_EPS / REFERENCE_FIELD_DIMEN[0]
    eps_y = _REFERENCE_EPS / REFERENCE_FIELD_DIMEN[1]
    iy = np.clip((((np.arange(n_y) + 0.5) / n_y - eps_y) * src_y).astype(int), 0, src_y - 1)
    ix = np.clip((((np.arange(n_x) + 0.5) / n_x - eps_x) * src_x).astype(int), 0, src_x - 1)
    return src[np.ix_(iy, ix)]


def epv_at(
    x: float,
    y: float,
    attack_direction: int = 1,
    grid: np.ndarray | None = None,
    pitch_length: float = PITCH_LENGTH,
    pitch_width: float = PITCH_WIDTH,
) -> float:
    """Expected possession value at a pitch location.

    Nearest-cell floor with no interpolation, matching the reference exactly.
    Positions off the pitch return ``0.0``.

    The reference performs this lookup in its own centred ``(106, 68)`` frame.
    We do the arithmetic natively in the canonical bottom-left ``(105, 68)``
    frame with the epsilon rescaled by the same factor, which is algebraically
    identical -- ``x_ref + 53 == x * 106 / pitch_length`` -- and
    ``tests/test_epv.py`` checks that equivalence against the actual reference
    function over a grid of sample points.

    Args:
        x: Metres along the pitch, ``0`` at the defended goal line and
            ``pitch_length`` at the attacked one.
        y: Metres across the pitch, ``0`` at the bottom touchline.
        attack_direction: ``+1`` if the team in possession attacks ``+x``.
        grid: Optional EPV grid. Defaults to the bundled one.
        pitch_length: Canonical pitch length in metres.
        pitch_width: Canonical pitch width in metres.

    Returns:
        The EPV of that location, or ``0.0`` if it is off the pitch.

    Raises:
        ValueError: If ``attack_direction`` is not ``+1`` or ``-1``.
    """
    if attack_direction not in (1, -1):
        raise ValueError(f"attack_direction must be +1 or -1, got {attack_direction!r}")
    if not (np.isfinite(x) and np.isfinite(y)):
        return 0.0
    if x < 0.0 or x > pitch_length or y < 0.0 or y > pitch_width:
        return 0.0
    base = load_epv() if grid is None else np.asarray(grid, dtype=np.float64)
    oriented = np.fliplr(base) if attack_direction == -1 else base
    n_y, n_x = oriented.shape
    # Epsilon rescaled from the reference frame into ours, so the tie-breaking
    # at the far touchlines is bit-identical to the reference's.
    eps_x = _REFERENCE_EPS * pitch_length / REFERENCE_FIELD_DIMEN[0]
    eps_y = _REFERENCE_EPS * pitch_width / REFERENCE_FIELD_DIMEN[1]
    ix = int((x - eps_x) / (pitch_length / n_x))
    iy = int((y - eps_y) / (pitch_width / n_y))
    ix = min(max(ix, 0), n_x - 1)
    iy = min(max(iy, 0), n_y - 1)
    return float(oriented[iy, ix])


def epv_at_reference_frame(
    position: tuple[float, float],
    attack_direction: int = 1,
    grid: np.ndarray | None = None,
    field_dimen: tuple[float, float] = REFERENCE_FIELD_DIMEN,
) -> float:
    """Look up EPV using a point given in the reference's centred frame.

    Provided so validation code can feed reference-frame coordinates straight
    in without hand-rolling the conversion.

    Args:
        position: ``(x, y)`` in the reference's centred frame.
        attack_direction: ``+1`` if the team in possession attacks ``+x``.
        grid: Optional EPV grid. Defaults to the bundled one.
        field_dimen: ``(length, width)`` assumed by the reference.

    Returns:
        The EPV of that location, or ``0.0`` if it is off the pitch.
    """
    x_ref, y_ref = position
    return epv_at(
        x_ref + field_dimen[0] / 2.0,
        y_ref + field_dimen[1] / 2.0,
        attack_direction=attack_direction,
        grid=grid,
        pitch_length=field_dimen[0],
        pitch_width=field_dimen[1],
    )


def expected_epv_added(
    control_surface: np.ndarray,
    attack_direction: int = 1,
    grid: np.ndarray | None = None,
    reduce: Literal["sum", "mean"] = "sum",
) -> float:
    """Control-weighted EPV over the whole pitch.

    The elementwise ``EPV * PPCF`` integral. With the default 50 x 32 pitch
    control grid the two arrays are both ``(32, 50)`` and this is a bare
    elementwise product; when the control surface has a different resolution
    the EPV grid is nearest-cell resampled onto it first.

    Args:
        control_surface: ``(n_y, n_x)`` pitch control probabilities for the
            team of interest, row 0 at the bottom touchline.
        attack_direction: ``+1`` if that team attacks ``+x``.
        grid: Optional EPV grid. Defaults to the bundled one.
        reduce: ``"sum"`` for the grid sum (the default; note that it scales
            with the number of cells, so only compare sums computed at the same
            resolution) or ``"mean"`` for the resolution-independent average.

    Returns:
        The reduced ``EPV * control`` product.

    Raises:
        ValueError: If ``control_surface`` is not 2-D, or ``reduce`` is not a
            known reduction.
    """
    surf = np.asarray(control_surface, dtype=np.float64)
    if surf.ndim != 2:
        raise ValueError(f"control_surface must be 2-D, got shape {surf.shape}")
    epv = epv_surface(attack_direction=attack_direction, shape=surf.shape, grid=grid)
    product = epv * surf
    if reduce == "sum":
        return float(product.sum())
    if reduce == "mean":
        return float(product.mean())
    raise ValueError(f"unknown reduction {reduce!r}; expected 'sum' or 'mean'")


def expected_epv(
    control: np.ndarray,
    grid: np.ndarray | None = None,
    attack_direction: int = 1,
) -> float:
    """Alias of :func:`expected_epv_added` with the legacy argument order.

    Args:
        control: Pitch control surface for the team of interest.
        grid: EPV grid on the same geometry, or None for the bundled one.
        attack_direction: ``+1`` if that team attacks ``+x``.

    Returns:
        Sum of ``control * grid``, the EPV a team currently commands.
    """
    return expected_epv_added(control, attack_direction=attack_direction, grid=grid)
