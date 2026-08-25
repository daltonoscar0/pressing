"""Tests for the EPV surface.

The bundled grid is the Friends of Tracking reference ``EPV_grid.csv`` (MIT,
commit 7ed3ff8). The lookup reproduces that repository's
``Metrica_EPV.get_EPV_at_location`` exactly, so the interesting tests here are
about the frame conversion between the reference's centred 106 x 68 metre
coordinates and this project's canonical bottom-left 105 x 68 metre ones.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from pressing.epv import (
    DEFAULT_EPV_PATH,
    EPV_ATTRIBUTION,
    epv_at,
    epv_at_reference_frame,
    epv_surface,
    expected_epv,
    expected_epv_added,
    load_epv,
    load_epv_grid,
    resample_epv,
)
from pressing.pitchcontrol import REFERENCE_FIELD_DIMEN, to_reference_frame

PITCH_L = 105.0
PITCH_W = 68.0
REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# The reference lookup, transcribed
# --------------------------------------------------------------------------- #


def _reference_get_epv_at_location(
    position: tuple[float, float],
    epv: np.ndarray,
    attack_direction: int,
    field_dimen: tuple[float, float] = (106.0, 68.0),
) -> float:
    """Verbatim transcription of ``Metrica_EPV.get_EPV_at_location``.

    From https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking,
    commit 7ed3ff857940d3f4611be55f3c08c79525c32171, MIT licence, Copyright
    (c) 2021 Friends-of-Tracking-Data-FoTD. Transcribed rather than imported so
    this test runs on a fresh clone: the vendored copy under ``data/raw/`` is
    gitignored. ``test_transcription_matches_the_vendored_reference`` checks
    the transcription against the real file whenever that file is present.
    """
    x, y = position
    if abs(x) > field_dimen[0] / 2.0 or abs(y) > field_dimen[1] / 2.0:
        return 0.0
    if attack_direction == -1:
        epv = np.fliplr(epv)
    ny, nx = epv.shape
    dx = field_dimen[0] / float(nx)
    dy = field_dimen[1] / float(ny)
    ix = (x + field_dimen[0] / 2.0 - 0.0001) / dx
    iy = (y + field_dimen[1] / 2.0 - 0.0001) / dy
    return float(epv[int(iy), int(ix)])


def test_transcription_matches_the_vendored_reference() -> None:
    """Cross-check the transcription above against the cached upstream file."""
    vendored = REPO_ROOT / "data" / "raw" / "reference" / "Metrica_EPV.py"
    if not vendored.is_file():
        pytest.skip("vendored reference not present (data/raw/ is not committed)")
    source = vendored.read_text()
    start = source.index("def get_EPV_at_location")
    body = source[start : source.index("\ndef ", start + 1)]
    namespace: dict[str, object] = {"np": np}
    exec(compile(body, str(vendored), "exec"), namespace)  # noqa: S102
    upstream = namespace["get_EPV_at_location"]
    grid = load_epv()
    rng = np.random.default_rng(0)
    for _ in range(400):
        pos = (rng.uniform(-56, 56), rng.uniform(-36, 36))
        for direction in (1, -1):
            assert _reference_get_epv_at_location(pos, grid, direction) == pytest.approx(
                float(upstream(pos, grid, direction))
            )


# --------------------------------------------------------------------------- #
# Loading and provenance
# --------------------------------------------------------------------------- #


def test_bundled_grid_ships_with_its_licence_and_provenance() -> None:
    epv_dir = DEFAULT_EPV_PATH.parent
    assert DEFAULT_EPV_PATH.is_file()
    assert (epv_dir / "LICENSE.friends-of-tracking").is_file()
    provenance = (epv_dir / "PROVENANCE.md").read_text()
    assert "7ed3ff857940d3f4611be55f3c08c79525c32171" in provenance
    assert "LaurieOnTracking" in provenance
    licence = (epv_dir / "LICENSE.friends-of-tracking").read_text()
    assert "MIT License" in licence
    assert "Friends-of-Tracking-Data-FoTD" in licence
    # anything user-facing has to say whose grid this is
    assert "Friends of Tracking" in EPV_ATTRIBUTION
    assert "Not fitted by this tool" in EPV_ATTRIBUTION


def test_load_epv_shape_and_known_statistics() -> None:
    grid = load_epv()
    assert grid.shape == (32, 50)
    assert grid.dtype == np.float64
    assert grid.min() == pytest.approx(0.0041)
    assert grid.max() == pytest.approx(0.5714)
    assert grid.sum() == pytest.approx(42.2504, abs=1e-4)
    assert np.unravel_index(int(grid.argmax()), grid.shape) == (15, 49)
    assert load_epv_grid() is grid  # the alias and the cache


def test_load_epv_is_cached_and_read_only() -> None:
    a = load_epv()
    assert a is load_epv()
    assert not a.flags.writeable


def test_load_epv_rejects_a_missing_file() -> None:
    with pytest.raises(FileNotFoundError, match="EPV grid not found"):
        load_epv(REPO_ROOT / "data" / "epv" / "does_not_exist.csv")


# --------------------------------------------------------------------------- #
# Grid shape properties
# --------------------------------------------------------------------------- #


def test_grid_is_symmetric_in_y() -> None:
    grid = load_epv()
    np.testing.assert_array_equal(grid, np.flipud(grid))


def test_grid_is_monotone_toward_the_attacked_goal() -> None:
    column_means = load_epv().mean(axis=0)
    assert (np.diff(column_means) > 0).all()
    # and mirrored when attacking the other way
    assert (np.diff(epv_surface(-1).mean(axis=0)) < 0).all()


def test_epv_at_is_symmetric_in_y_and_monotone_in_x() -> None:
    for x in (10.0, 52.5, 90.0, 104.0):
        assert epv_at(x, 20.0) == pytest.approx(epv_at(x, PITCH_W - 20.0))
    # Averaged across the width, EPV rises strictly toward the attacked goal.
    # A single row does not: the grid dips slightly in places, e.g. along the
    # centre line at x = 40 m, so a per-row monotonicity assertion would be
    # asserting something the reference grid simply is not.
    ys = np.arange(1.0, PITCH_W, 2.0)
    means = [float(np.mean([epv_at(x, y) for y in ys])) for x in np.arange(1.0, 105.0, 3.0)]
    assert all(b > a for a, b in pairwise(means))
    assert epv_at(104.0, 34.0) > 10 * epv_at(1.0, 34.0)


def test_epv_at_mirrors_with_attack_direction() -> None:
    for x in (5.0, 30.0, 88.0):
        assert epv_at(x, 20.0, 1) == pytest.approx(epv_at(PITCH_L - x, 20.0, -1))
    assert epv_at(104.0, 34.0, 1) > epv_at(1.0, 34.0, 1)
    assert epv_at(104.0, 34.0, -1) < epv_at(1.0, 34.0, -1)


def test_epv_at_is_zero_off_the_pitch_and_at_nan() -> None:
    assert epv_at(-0.5, 34.0) == 0.0
    assert epv_at(105.5, 34.0) == 0.0
    assert epv_at(50.0, -1.0) == 0.0
    assert epv_at(50.0, 68.5) == 0.0
    assert epv_at(np.nan, 34.0) == 0.0
    # the touchlines themselves are on the pitch
    assert epv_at(0.0, 0.0) > 0.0
    assert epv_at(PITCH_L, PITCH_W) > 0.0


def test_epv_at_rejects_a_bad_direction() -> None:
    with pytest.raises(ValueError, match="attack_direction"):
        epv_at(50.0, 34.0, 0)
    with pytest.raises(ValueError, match="attack_direction"):
        epv_surface(2)


# --------------------------------------------------------------------------- #
# The frame conversion, against the reference
# --------------------------------------------------------------------------- #


def test_epv_at_agrees_with_the_reference_after_frame_conversion() -> None:
    """The whole point of the conversion: same physical point, same number."""
    grid = load_epv()
    xs = np.linspace(0.0, PITCH_L, 61)
    ys = np.linspace(0.0, PITCH_W, 37)
    checked = 0
    for x in xs:
        for y in ys:
            ref_xy = to_reference_frame(np.array([x, y]))
            for direction in (1, -1):
                expected = _reference_get_epv_at_location(
                    (float(ref_xy[0]), float(ref_xy[1])), grid, direction
                )
                assert epv_at(x, y, direction) == pytest.approx(expected), (
                    f"mismatch at canonical ({x:.3f}, {y:.3f}) -> reference "
                    f"({ref_xy[0]:.3f}, {ref_xy[1]:.3f}), direction {direction}"
                )
                checked += 1
    assert checked == 61 * 37 * 2


def test_epv_at_agrees_with_the_reference_on_random_points() -> None:
    grid = load_epv()
    rng = np.random.default_rng(20240825)
    for _ in range(2000):
        x = rng.uniform(-2.0, PITCH_L + 2.0)
        y = rng.uniform(-2.0, PITCH_W + 2.0)
        ref_xy = to_reference_frame(np.array([x, y]))
        for direction in (1, -1):
            expected = _reference_get_epv_at_location(
                (float(ref_xy[0]), float(ref_xy[1])), grid, direction
            )
            assert epv_at(x, y, direction) == pytest.approx(expected)


def test_reference_frame_helper_matches_the_reference_directly() -> None:
    grid = load_epv()
    rng = np.random.default_rng(5)
    for _ in range(500):
        pos = (rng.uniform(-54, 54), rng.uniform(-35, 35))
        for direction in (1, -1):
            assert epv_at_reference_frame(pos, direction) == pytest.approx(
                _reference_get_epv_at_location(pos, grid, direction)
            )


def test_the_x_centre_is_genuinely_ambiguous_at_the_halfway_line() -> None:
    """50 columns is even, so there is no centre column; document the noise."""
    left = epv_at(PITCH_L / 2, PITCH_W / 2, 1)
    right = epv_at(PITCH_L / 2, PITCH_W / 2, -1)
    assert left != right
    assert abs(left - right) < 0.001


# --------------------------------------------------------------------------- #
# Control-weighted EPV
# --------------------------------------------------------------------------- #


def test_expected_epv_added_is_the_elementwise_product_at_native_resolution() -> None:
    rng = np.random.default_rng(3)
    control = rng.uniform(0.0, 1.0, (32, 50))
    grid = load_epv()
    assert expected_epv_added(control, 1) == pytest.approx((grid * control).sum())
    assert expected_epv_added(control, -1) == pytest.approx((np.fliplr(grid) * control).sum())
    assert expected_epv_added(control, 1, reduce="mean") == pytest.approx(
        (grid * control).mean()
    )
    assert expected_epv(control) == pytest.approx(expected_epv_added(control, 1))


def test_expected_epv_added_bounds() -> None:
    ones = np.ones((32, 50))
    zeros = np.zeros((32, 50))
    assert expected_epv_added(zeros, 1) == 0.0
    assert expected_epv_added(ones, 1) == pytest.approx(load_epv().sum())
    # controlling the attacking half is worth more than the defending half
    half_far = np.zeros((32, 50))
    half_far[:, 25:] = 1.0
    half_near = np.zeros((32, 50))
    half_near[:, :25] = 1.0
    assert expected_epv_added(half_far, 1) > 5 * expected_epv_added(half_near, 1)


def test_expected_epv_added_resamples_a_non_native_control_grid() -> None:
    control = np.ones((68, 105))
    coarse = np.ones((16, 25))
    native_mean = load_epv().mean()
    assert expected_epv_added(control, 1, reduce="mean") == pytest.approx(
        native_mean, rel=0.05
    )
    assert expected_epv_added(coarse, 1, reduce="mean") == pytest.approx(native_mean, rel=0.10)
    # the sum scales with cell count, which is why "mean" exists
    assert expected_epv_added(control, 1) > expected_epv_added(np.ones((32, 50)), 1)
    # direction still flips after resampling
    tilted = np.zeros((68, 105))
    tilted[:, 70:] = 1.0
    assert expected_epv_added(tilted, 1) > expected_epv_added(tilted, -1)


def test_expected_epv_added_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="must be 2-D"):
        expected_epv_added(np.ones(50), 1)
    with pytest.raises(ValueError, match="unknown reduction"):
        expected_epv_added(np.ones((32, 50)), 1, reduce="median")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Resampling
# --------------------------------------------------------------------------- #


def test_resample_is_a_no_op_at_the_native_shape() -> None:
    grid = load_epv()
    np.testing.assert_array_equal(resample_epv(grid, (32, 50)), grid)
    np.testing.assert_array_equal(epv_surface(1, shape=(32, 50)), grid)


def test_resample_preserves_orientation_and_range() -> None:
    fine = epv_surface(1, shape=(64, 100))
    assert fine.shape == (64, 100)
    # 64 rows from 32 has no boundary ties, so symmetry survives exactly
    np.testing.assert_array_equal(fine, np.flipud(fine))
    assert (np.diff(fine.mean(axis=0)) >= 0).all()
    assert fine.min() >= load_epv().min()
    assert fine.max() <= load_epv().max()
    coarse = epv_surface(1, shape=(4, 5))
    assert coarse.shape == (4, 5)
    assert coarse[0, -1] > coarse[0, 0]


def test_resample_symmetry_degrades_by_at_most_one_source_row_on_ties() -> None:
    """68 rows from 32 puts some target centres exactly on a source boundary.

    The reference's floor rule sends those into the upper cell, which is not
    mirror-symmetric. Documented rather than hidden: the damage is bounded by
    the gap between adjacent source rows.
    """
    src = load_epv()
    fine = epv_surface(1, shape=(68, 105))
    worst = np.abs(fine - np.flipud(fine)).max()
    assert worst > 0.0
    assert worst <= np.abs(np.diff(src, axis=0)).max() + 1e-12


def test_resample_cells_agree_with_pointwise_lookup() -> None:
    """A resampled cell must hold the same value as epv_at at its centre."""
    n_y, n_x = 17, 23
    fine = epv_surface(1, shape=(n_y, n_x))
    for iy in range(n_y):
        for ix in range(n_x):
            x = (ix + 0.5) * PITCH_L / n_x
            y = (iy + 0.5) * PITCH_W / n_y
            assert fine[iy, ix] == pytest.approx(epv_at(x, y, 1))


def test_resample_rejects_a_bad_shape() -> None:
    with pytest.raises(ValueError, match="two positive integers"):
        resample_epv(load_epv(), (0, 5))
    with pytest.raises(ValueError, match="two positive integers"):
        resample_epv(load_epv(), (5, -2))


def test_epv_surface_returns_a_writeable_copy() -> None:
    surface = epv_surface(1)
    surface[0, 0] = 99.0
    assert load_epv()[0, 0] != 99.0


def test_reference_field_dimen_is_what_the_reference_uses() -> None:
    assert REFERENCE_FIELD_DIMEN == (106.0, 68.0)
