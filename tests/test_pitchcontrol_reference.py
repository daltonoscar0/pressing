"""Regression test: our pitch control against the Friends of Tracking reference.

This is the project's headline claim -- that ``pressing.pitchcontrol`` reproduces
the published Friends of Tracking implementation of the Spearman model -- defended
by something that runs on a fresh clone with no data at all.

Until now that claim lived only in ``validation/pitch_control_validation.ipynb``,
which needs the Metrica sample CSVs under the gitignored ``data/raw/``. A refactor
could have broken the agreement and the suite would have stayed green. The scenes
below are synthetic, so nothing here reads a data file, downloads anything, or
skips.

What is compared
----------------
The reference is the vendored copy at ``validation/reference/``. Its own public
entry point ``generate_pitch_control_for_event`` is called, on synthetic one-row
tracking and event DataFrames, so the comparison exercises the reference's own
grid construction, player initialisation and per-cell loop rather than a
paraphrase of them. The vendored bytes are never modified: the whole directory is
copied to a pytest tmp dir, ``modern_pandas.patch`` is applied to the copy, and
``MANIFEST.sha256`` is verified before and after. The patch touches only
``Metrica_IO.py`` and ``Metrica_Velocities.py``; ``Metrica_PitchControl.py``, the
model actually under comparison, is byte-identical to upstream and is asserted to
be so.

Ours is run at ``clamp="raw"`` on :func:`pressing.pitchcontrol.reference_grid`, in
float64, which is the configuration the validation notebook compares in: the
reference has no clamp, so clamping ours would be comparing two different things.
A second, looser check covers the shipped float32 dtype.

Measured agreement
------------------
Every number below is reproducible with::

    uv run python tests/test_pitchcontrol_reference.py

which runs this file's own scenes through both implementations and prints the
table. Measured on 2026-08-28 with numpy 2.4.6 on CPython 3.11.15 (arm64)::

    scenes compared                         : 10
    cells compared                          : 16,000
    pooled Pearson r                        : 0.999999972693
    worst per-scene Pearson r               : 0.999999288256
    max |difference| over converged cells   : 3.2196e-15  (15,984 of 16,000 cells)
    max |difference| over every cell        : 1.131579e-02
    mean |difference| pooled                : 1.6494e-06
    worst per-scene mean |difference|       : 1.2463e-05
    cells disagreeing by more than 1e-12    : 8 of 16,000
      of those, on a converged cell         : 0
    float32: worst per-scene Pearson r      : 0.999999288319
    float32: max |diff| over converged cells: 1.1291e-06

Eight of the ten scenes agree to 2.9e-15 or better and correlate at
1.000000000000 to twelve decimal places. The two that do not,
``ball_far_from_everyone`` (r = 0.999999288256) and ``veto_contour_sweep``
(r = 0.999999980923), are explained exactly in the next section.

The same command was run in a throwaway clone of this repository with no
``data/raw/`` at all, whose ``uv sync`` resolved numpy 2.5.2 rather than 2.4.6,
and every figure above reproduced to the last digit printed. So the bounds below
are not tuned to one numpy build.

Correlations, cell counts and differences are load-independent, so they are
quoted without a load-average caveat. No wall-clock number is quoted anywhere in
this file: the 1-minute load average on this machine was 9.07 while the fresh
clone was measured, and nothing timed under that is worth writing down.

Where the 1.1e-2 comes from, and why it is not a bug
----------------------------------------------------
All 8 disagreeing cells are cells our own ``PitchControlResult.converged`` flag
reports as *not* converged, and on which the reference prints its own
"Integration failed to converge" message. The cause is exact and was measured:
the reference builds ``dT_array = np.arange(btt - dt, btt + max_int_time, dt)``
per cell and integrates ``len(dT_array) - 1`` steps, and that length is 251 or
252 depending purely on the floating point value of the ball travel time ``btt``.
Ours integrates a fixed 250. Splitting the 16,000 cells on that alone::

    reference took 250 steps (as we do): 10,158 cells, worst |diff| 3.2196e-15,
                                         0 cells above 1e-12
    reference took 251 steps           :  5,842 cells, worst |diff| 1.1316e-02,
                                         8 cells above 1e-12, all unconverged

That is the entire residual. On a cell that converges the extra step contributes
nothing, which is why 5,834 of those 5,842 cells still agree to 3.2e-15. Only on
a cell that never converges does it remain visible, and there it is worth at most
one forward-Euler step, ``remaining * lambda * f * dt`` <= 0.99 * 4.3 * 1 * 0.04
= 0.17. The 1.1e-2 observed is well inside that.

Does this test actually catch anything?
---------------------------------------
Checked by mutation, on a scratch copy of ``src/`` -- the repository was not
modified. Fourteen single-line mutations of ``pitchcontrol.py`` were introduced
one at a time and this file (34 tests) was run against each:

====================================================  ============
mutation                                              tests failed
====================================================  ============
``from_reference_frame`` rescaled to a 105 m pitch    27
surface returned ``np.flipud``-ed                     26
``model_converge_tol`` 0.01 -> 0.02                   25
reaction time dropped from the intercept model        24
per-team tti filter -> global minimum (SUBTLETY 1b)   14
goalkeeper boost also applied to the attacker          9
veto drops the ball travel term (SUBTLETY 1a)          8
NaN velocity drops the player instead of zeroing it    7
integrator runs 251 steps instead of 250               6
offside uses the deepest, not second-deepest, defender 2
integrator runs 249 steps instead of 250               1
offside line ignores the ball                          1
offside ``tol`` 0.2 -> 0.5                             1
``SIGMOID_CLIP`` 60 -> 600                             0
====================================================  ============

Thirteen of the fourteen are caught. The survivor is honest: widening the logistic
clip from 60 to 600 does not change a single cell value here, because ``exp``
saturates either way -- what it changes is that float32 starts emitting overflow
warnings, which ``test_pitchcontrol.py`` already owns.

Two of the mutations only became detectable because of the design above. The
249-step mutation is invisible on any cell that converges, and was caught only
once :func:`test_disagreement_is_exactly_the_reference_taking_one_extra_euler_step`
started holding the 10,158 same-step-count cells to the converged-cell bound
rather than to the loose whole-surface one. "Offside line ignores the ball" is
invisible at any single ball position where the second-deepest defender is
already ahead of the ball, and was caught only once
:func:`test_offside_line_tracks_the_ball_exactly_as_the_reference_does` began
sweeping the ball the length of the pitch.

Tolerances, and where they come from
------------------------------------
Every bound below is the measured figure with a stated margin, not a taste
judgement:

* :data:`MAX_ABS_DIFF_CONVERGED` = 1e-12, against a measured 3.2196e-15: 310x.
* :data:`MIN_PER_SCENE_R` = 0.99999, against a measured worst 0.999999288256.
  In the quantity that actually varies, ``1 - r``, that is 1e-5 against a
  measured 7.12e-7: 14x.
* :data:`MAX_ABS_DIFF_ANY` = 0.05, against a measured 1.131579e-02: 4.4x. This
  one is deliberately the loosest, because it is the only bound the unconverged
  cells are tested by and one whole Euler step is 0.17.
* :data:`MAX_MEAN_ABS_DIFF` = 1e-4, against a measured worst per-scene 1.2463e-05
  and a pooled 1.6494e-06: 8x on the worst scene.
* :data:`MIN_PER_SCENE_R_F32` = 0.99999 and :data:`MAX_ABS_DIFF_CONVERGED_F32` =
  1e-5, against measured 0.999999288319 and 1.1291e-06.

Alignment guards
----------------
A regression test that silently compares a transposed or flipped surface is worse
than no test, so the agreement assertions are accompanied by
:func:`test_alignment_guards_rule_out_a_flipped_or_inverted_surface`, which checks
the shape, checks that a y-flipped and an x-flipped reference both correlate
*worse* than the aligned one on every scene, checks that the mean absolute
difference against a y-flipped reference is at least 100x the aligned one
(measured minimum ratio 1.96e4), and checks the sign in both directions:
``corr(ours, ref)`` is positive and ``corr(ours, 1 - ref)`` is negative. Eight of
the ten scenes are deliberately asymmetric in y for exactly this reason.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pressing.pitchcontrol import (
    REFERENCE_FIELD_DIMEN,
    PitchControlParams,
    PitchControlResult,
    PlayerSet,
    check_offsides,
    from_reference_frame,
    pitch_control_full,
    reference_grid,
)

# --------------------------------------------------------------------------- #
# Tolerances. Each is the measured figure with the margin stated in the module
# docstring; none is a round number chosen for looking tidy.
# --------------------------------------------------------------------------- #

#: Max ``|ours - reference|`` on a cell both implementations converged. Measured
#: 3.2196e-15 over 15,984 converged cells.
MAX_ABS_DIFF_CONVERGED: float = 1e-12

#: Max ``|ours - reference|`` on any cell at all, including the handful where the
#: reference's own integration never converges. Measured 1.131579e-02.
MAX_ABS_DIFF_ANY: float = 0.05

#: Max mean ``|ours - reference|`` for a single scene. Measured worst 1.2463e-05.
MAX_MEAN_ABS_DIFF: float = 1e-4

#: Min per-scene Pearson r in float64. Measured worst 0.999999288256.
MIN_PER_SCENE_R: float = 0.99999

#: Min pooled Pearson r over every cell of every scene. Measured 0.999999972693.
MIN_POOLED_R: float = 0.99999

#: Min per-scene Pearson r at the shipped float32 dtype. Measured worst
#: 0.999999288319 -- float32 is not the thing that limits agreement here.
MIN_PER_SCENE_R_F32: float = 0.99999

#: Max ``|ours_float32 - reference|`` on a converged cell. Measured 1.1291e-06.
MAX_ABS_DIFF_CONVERGED_F32: float = 1e-5

#: Cells whose difference exceeds this are required to be cells our own model
#: reports as unconverged. Chosen 300x above the converged-cell maximum so that
#: ordinary float64 noise can never trip it.
DISAGREEMENT_FLOOR: float = 1e-12

#: The reference grid, and therefore every surface in this file, is (n_y, n_x).
SURFACE_SHAPE: tuple[int, int] = (32, 50)

REFERENCE_DIR: Path = Path(__file__).resolve().parents[1] / "validation" / "reference"
PATCH_PATH: Path = REFERENCE_DIR / "modern_pandas.patch"
MANIFEST_PATH: Path = REFERENCE_DIR / "MANIFEST.sha256"

#: The model file is *not* touched by the compatibility patch; only the two
#: loaders are. Asserted, not assumed.
MODEL_FILE: str = "Metrica_PitchControl.py"
PATCHED_FILES: tuple[str, ...] = ("Metrica_IO.py", "Metrica_Velocities.py")


# --------------------------------------------------------------------------- #
# A pure-python unified-diff applier.
#
# The validation notebook shells out to patch(1). A test may not: if patch(1) is
# absent the test would skip, and a skipping test is precisely the hole this file
# exists to fill. This applier is ~40 lines and additionally checks that every
# hunk's context still matches, which patch(1) would fuzz over.
# --------------------------------------------------------------------------- #

_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

Hunk = tuple[int, list[str], list[str]]


def parse_unified_diff(patch_text: str) -> dict[str, list[Hunk]]:
    """Parse a unified diff into ``{filename: [(start_line, old, new), ...]}``.

    Args:
        patch_text: The contents of a unified diff.

    Returns:
        Per-file hunks. ``start_line`` is 1-based into the original file, and
        ``old`` / ``new`` are the lines that hunk replaces and installs.

    Raises:
        ValueError: If the diff is malformed.
    """
    files: dict[str, list[Hunk]] = {}
    current: list[Hunk] | None = None
    lines = patch_text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("--- "):
            if i + 1 >= len(lines) or not lines[i + 1].startswith("+++ "):
                raise ValueError(f"'---' header at line {i + 1} has no '+++' partner")
            name = lines[i + 1][4:].split("\t")[0].strip()
            files.setdefault(name.split("/", 1)[1] if "/" in name else name, [])
            current = files[name.split("/", 1)[1] if "/" in name else name]
            i += 2
            continue
        header = _HUNK_HEADER.match(line)
        if header is None:
            i += 1
            continue
        if current is None:
            raise ValueError(f"hunk at line {i + 1} precedes any file header")
        start = int(header.group(1))
        n_old = int(header.group(2) or "1")
        n_new = int(header.group(4) or "1")
        old: list[str] = []
        new: list[str] = []
        i += 1
        while len(old) < n_old or len(new) < n_new:
            if i >= len(lines):
                raise ValueError(f"hunk at line {start} is truncated")
            body = lines[i]
            i += 1
            if body.startswith("\\"):  # "\ No newline at end of file"
                continue
            tag, text = body[:1], body[1:]
            if tag in (" ", ""):
                old.append(text)
                new.append(text)
            elif tag == "-":
                old.append(text)
            elif tag == "+":
                new.append(text)
            else:
                raise ValueError(f"unexpected patch line {body!r}")
        current.append((start, old, new))
    return files


def apply_unified_diff(original: str, hunks: list[Hunk]) -> str:
    """Apply parsed hunks to a file's text, verifying every context line.

    Args:
        original: The unpatched file contents.
        hunks: Hunks from :func:`parse_unified_diff`.

    Returns:
        The patched text.

    Raises:
        ValueError: If a hunk's context does not match the original, which means
            the vendored file and the patch have drifted apart.
    """
    src = original.split("\n")
    out: list[str] = []
    pos = 0
    for start, old, new in sorted(hunks):
        at = start - 1
        if at < pos:
            raise ValueError(f"hunk at line {start} overlaps the previous hunk")
        found = src[at : at + len(old)]
        if found != old:
            raise ValueError(
                f"patch context mismatch at line {start}:\n  expected {old}\n  found    {found}"
            )
        out.extend(src[pos:at])
        out.extend(new)
        pos = at + len(old)
    out.extend(src[pos:])
    return "\n".join(out)


def sha256(path: Path) -> str:
    """Hex sha256 of a file.

    Args:
        path: File to digest.

    Returns:
        The lowercase hex digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_manifest() -> dict[str, str]:
    """Read ``validation/reference/MANIFEST.sha256`` into ``{filename: digest}``.

    Returns:
        The expected digest of every vendored file.
    """
    manifest: dict[str, str] = {}
    for line in MANIFEST_PATH.read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(None, 1)
        manifest[name.strip()] = digest
    return manifest


# --------------------------------------------------------------------------- #
# Scenes.
#
# Coordinates are in the REFERENCE's centred frame -- x in [-53, 53], y in
# [-34, 34] on a 106 x 68 pitch -- because that is the frame the reference
# insists on and writing the scene in it means the only conversion in the whole
# file is the one being tested, from_reference_frame.
#
# The attacking team is "Home" and attacks +x throughout.
# --------------------------------------------------------------------------- #

#: One player: ``(x, y, vx, vy, is_gk)``.
Player = tuple[float, float, float, float, bool]


@dataclass(frozen=True)
class Scene:
    """A synthetic instant: two teams, a ball, and why the scene is here.

    Attributes:
        name: Identifier used in pytest ids and failure messages.
        why: The property of the model this geometry is chosen to stress.
        att: Attacking ("Home") players.
        dfd: Defending ("Away") players.
        ball: ``(x, y)`` ball position, possibly NaN.
    """

    name: str
    why: str
    att: tuple[Player, ...]
    dfd: tuple[Player, ...]
    ball: tuple[float, float]


NAN = float("nan")

SCENES: tuple[Scene, ...] = (
    Scene(
        name="pitch_edge_and_off_pitch",
        why=(
            "Players pinned to the touchlines and the corner flags, two of them "
            "outside the playing area entirely, so the nearest-player field is "
            "degenerate along three edges of the grid at once."
        ),
        att=(
            (-53.0, 34.0, 0.0, 0.0, False),
            (53.0, -34.0, 1.0, -1.0, False),
            (0.0, 34.0, 0.0, 5.0, False),
            (-52.9, -33.9, -3.0, -3.0, False),
            (55.0, 36.0, 2.0, 2.0, False),
        ),
        dfd=(
            (-53.0, -34.0, 0.0, 0.0, False),
            (53.0, 34.0, -1.0, 1.0, False),
            (0.0, -34.0, 0.0, -5.0, False),
            (-40.0, 34.0, 0.0, 0.0, True),
            (-56.0, 0.0, 0.0, 0.0, False),
        ),
        ball=(-52.0, 33.0),
    ),
    Scene(
        name="deep_goalkeeper",
        why=(
            "A defending goalkeeper on his own six-yard line with every outfield "
            "defender pushed thirty metres up the pitch. This is the only scene "
            "in which lambda_gk = 12.9 rather than lambda_def = 4.3 dominates a "
            "large region, and the reference applies that boost to the defending "
            "keeper only -- the attacking keeper, also present here, must get the "
            "ordinary lambda_att."
        ),
        att=(
            (10.0, -9.0, 4.0, 0.0, False),
            (14.0, 3.0, 3.0, -1.0, False),
            (18.0, -18.0, 2.0, 1.0, False),
            (30.0, 6.0, 5.0, 0.0, False),
            (-45.0, 1.0, 0.0, 0.0, True),
        ),
        dfd=(
            (48.0, -2.0, 0.0, 0.0, True),
            (40.0, 7.0, -1.0, 0.0, False),
            (41.0, -2.0, -1.0, 0.0, False),
            (35.0, 16.0, -2.0, -1.0, False),
            (36.0, -21.0, -2.0, 1.0, False),
        ),
        ball=(25.0, -7.0),
    ),
    Scene(
        name="coincident_players",
        why=(
            "Three attackers on the same point, two defenders on the same point, "
            "and one attacker exactly on top of one defender. Every pair produces "
            "bit-identical intercept times, so any tie-break that differs between "
            "the two implementations shows up immediately."
        ),
        att=(
            (0.0, 0.0, 1.0, 1.0, False),
            (0.0, 0.0, 1.0, 1.0, False),
            (0.0, 0.0, -2.0, 0.0, False),
            (-20.0, 10.0, 0.0, 0.0, False),
            (12.0, -12.0, 0.0, 0.0, False),
        ),
        dfd=(
            (0.0, 0.0, 1.0, 1.0, False),
            (5.0, 5.0, 0.0, 0.0, False),
            (5.0, 5.0, 0.0, 0.0, False),
            (30.0, 0.0, 0.0, 0.0, True),
            (-20.0, 10.0, 0.0, 0.0, False),
        ),
        ball=(0.0, 0.0),
    ),
    Scene(
        name="zero_velocity_except_one",
        why=(
            "Seven players with exactly (0, 0) velocity and one sprinting at the "
            "5 m/s cap. The reaction-time term r + v * 0.7 collapses to r for "
            "everyone but the sprinter, which is the branch a velocity-handling "
            "regression would break."
        ),
        att=(
            (-10.0, -4.0, 5.0, 0.0, False),
            (-30.0, 22.0, 0.0, 0.0, False),
            (-30.0, -13.0, 0.0, 0.0, False),
            (0.0, 25.0, 0.0, 0.0, False),
        ),
        dfd=(
            (10.0, 6.0, 0.0, 0.0, False),
            (20.0, 11.0, 0.0, 0.0, False),
            (20.0, -19.0, 0.0, 0.0, False),
            (46.0, 2.0, 0.0, 0.0, True),
        ),
        ball=(-10.0, -4.0),
    ),
    Scene(
        name="ball_far_from_everyone",
        why=(
            "Every player crammed into one corner with the ball in the diagonally "
            "opposite one, so the ball travel time reaches ~6 s and dominates the "
            "two pre-integration vetoes across the whole grid. This is also the "
            "only scene in which the reference's own integration fails to "
            "converge, which is where the two implementations genuinely differ."
        ),
        att=(
            (-50.0, -32.0, 0.0, 0.0, False),
            (-48.0, -30.0, 1.0, 0.0, False),
            (-46.0, -33.0, 0.0, 1.0, False),
        ),
        dfd=(
            (-49.0, -31.0, 0.0, 0.0, False),
            (-47.0, -28.0, -1.0, 0.0, False),
            (52.0, 33.0, 0.0, 0.0, True),
        ),
        ball=(52.0, 33.0),
    ),
    Scene(
        name="offside_attackers",
        why=(
            "Three attackers beyond the second-deepest defender, a back line and "
            "both goalkeepers. Used twice: with offsides off (the shipped default, "
            "and the notebook's configuration) and again with offsides on, where "
            "our attack_direction argument has to reproduce the reference's "
            "sign(defending goalkeeper x)."
        ),
        att=(
            (44.0, 0.0, 0.0, 0.0, False),
            (46.0, 8.0, 1.0, 0.0, False),
            (45.0, -8.0, 0.0, 1.0, False),
            (10.0, 0.0, 2.0, 0.0, False),
            (-5.0, 12.0, 1.0, 1.0, False),
            (-40.0, 0.0, 0.0, 0.0, True),
        ),
        dfd=(
            (50.0, 0.0, 0.0, 0.0, True),
            (30.0, 5.0, -1.0, 0.0, False),
            (28.0, -5.0, -1.0, 0.0, False),
            (20.0, 12.0, 0.0, 0.0, False),
            (18.0, -12.0, 0.0, 0.0, False),
            (5.0, 0.0, -2.0, 0.0, False),
        ),
        ball=(8.0, 0.0),
    ),
    Scene(
        name="exact_mirror_tie",
        why=(
            "The hardest geometry class for agreement. Every defender is the exact "
            "x-mirror of an attacker (x -> -x, vx -> -vx, y and vy kept), so on the "
            "x = 0 column of cells the two teams' minimum intercept times are "
            "bit-for-bit equal and every cell there sits precisely on the >= "
            "boundary of both pre-integration vetoes. One ulp of difference in how "
            "a distance is summed would flip a cell from an integrated ~0.5 to a "
            "vetoed 0.0 or 1.0."
        ),
        att=(
            (-20.0, 3.0, 3.0, 0.0, False),
            (-30.0, 17.0, 0.0, 2.0, False),
            (-30.0, -11.0, 0.0, -2.0, False),
            (-5.0, 26.0, 1.0, 0.0, False),
            (-45.0, 1.0, 0.0, 0.0, True),
        ),
        dfd=(
            (20.0, 3.0, -3.0, 0.0, False),
            (30.0, 17.0, 0.0, 2.0, False),
            (30.0, -11.0, 0.0, -2.0, False),
            (5.0, 26.0, -1.0, 0.0, False),
            (45.0, 1.0, 0.0, 0.0, True),
        ),
        ball=(0.0, 0.0),
    ),
    Scene(
        name="veto_contour_sweep",
        why=(
            "One attacker, one defender, at opposite ends. The difference "
            "tau_att - max(ball_travel, tau_def) then sweeps smoothly across the "
            "grid and necessarily crosses time_to_control_def = 3.32025 s along a "
            "whole contour of cells, so the veto boundary is not a corner case "
            "here, it is a line through the middle of the surface."
        ),
        att=((-40.0, -12.0, 0.0, 0.0, False),),
        dfd=((40.0, 9.0, 0.0, 0.0, False),),
        ball=(-3.0, 5.0),
    ),
    Scene(
        name="nan_positions_and_velocities",
        why=(
            "One attacker and one defender out of frame (NaN position), which both "
            "implementations must drop entirely, and three players with a NaN "
            "velocity component, which both must keep and treat as stationary. "
            "Getting either rule backwards changes the surface without raising."
        ),
        att=(
            (0.0, -6.0, NAN, NAN, False),
            (NAN, 5.0, 1.0, 1.0, False),
            (-15.0, 14.0, 2.0, 0.0, False),
            (-25.0, -21.0, 0.0, 0.0, False),
        ),
        dfd=(
            (10.0, 2.0, 0.0, NAN, False),
            (20.0, -3.0, NAN, 0.0, False),
            (5.0, NAN, 0.0, 0.0, False),
            (45.0, 8.0, 0.0, 0.0, True),
        ),
        ball=(0.0, 0.0),
    ),
    Scene(
        name="nan_ball_position",
        why=(
            "A NaN ball. Both implementations must fall back to a zero ball travel "
            "time -- 'the ball is already there' -- rather than propagating NaN "
            "through the vetoes and returning a surface of NaN."
        ),
        att=(
            (-12.0, 3.0, 1.0, 0.0, False),
            (-25.0, -8.0, 0.0, 2.0, False),
            (4.0, 20.0, -1.0, 0.0, False),
        ),
        dfd=(
            (6.0, -3.0, 0.0, 0.0, False),
            (16.0, 9.0, -2.0, 0.0, False),
            (44.0, 0.0, 0.0, 0.0, True),
        ),
        ball=(NAN, NAN),
    ),
)

SCENES_BY_NAME: dict[str, Scene] = {s.name: s for s in SCENES}


# --------------------------------------------------------------------------- #
# Adapters: one scene, two implementations
# --------------------------------------------------------------------------- #


def _columns(players: tuple[Player, ...], teamname: str) -> tuple[dict[str, float], str | None]:
    """Build the reference's wide tracking columns for one team.

    Args:
        players: The team's players.
        teamname: ``"Home"`` or ``"Away"``.

    Returns:
        ``(columns, gk_jersey)`` where ``columns`` is keyed
        ``f"{teamname}_{jersey}_{x|y|vx|vy}"`` exactly as Metrica's CSVs are, and
        ``gk_jersey`` is None when the team has no goalkeeper.
    """
    cols: dict[str, float] = {}
    gk: str | None = None
    for i, (x, y, vx, vy, is_gk) in enumerate(players):
        jersey = str(i + 1)
        cols[f"{teamname}_{jersey}_x"] = x
        cols[f"{teamname}_{jersey}_y"] = y
        cols[f"{teamname}_{jersey}_vx"] = vx
        cols[f"{teamname}_{jersey}_vy"] = vy
        if is_gk:
            gk = jersey
    return cols, gk


def reference_surface(module: ModuleType, scene: Scene, offsides: bool = False) -> np.ndarray:
    """Run the reference's own public entry point on a scene.

    ``generate_pitch_control_for_event`` is used rather than the per-cell
    function, so the reference's grid construction, player initialisation,
    offside filter and convergence assertion are all exercised as upstream wrote
    them. The synthetic tracking and event frames are the minimum shape it reads.

    Args:
        module: The patched ``Metrica_PitchControl`` module.
        scene: The scene to evaluate.
        offsides: Whether to let the reference drop offside attackers.

    Returns:
        ``(32, 50)`` attacking-team pitch control, row 0 at the lowest y.
    """
    frame = 7  # arbitrary; the reference only ever uses it as a .loc key
    home_cols, home_gk = _columns(scene.att, "Home")
    away_cols, away_gk = _columns(scene.dfd, "Away")
    tracking_home = pd.DataFrame([home_cols], index=[frame])
    tracking_away = pd.DataFrame([away_cols], index=[frame])
    events = pd.DataFrame(
        [
            {
                "Start Frame": frame,
                "Team": "Home",
                "Start X": scene.ball[0],
                "Start Y": scene.ball[1],
            }
        ]
    )
    surface, _xgrid, _ygrid = module.generate_pitch_control_for_event(
        0,
        events,
        tracking_home,
        tracking_away,
        module.default_model_params(),
        (home_gk, away_gk),
        field_dimen=REFERENCE_FIELD_DIMEN,
        n_grid_cells_x=SURFACE_SHAPE[1],
        offsides=offsides,
    )
    return np.asarray(surface, dtype=float)


def _player_set(players: tuple[Player, ...]) -> PlayerSet:
    """Convert a scene's team into our :class:`PlayerSet`, converting frames.

    Args:
        players: The team's players, in the reference's centred frame.

    Returns:
        The team in the canonical bottom-left frame of a 106 x 68 pitch, which is
        the frame :func:`pressing.pitchcontrol.reference_grid` lives in. Ids are
        the same 1-based jersey strings :func:`_columns` gives the reference, so
        the two implementations' surviving-player lists can be compared directly.
    """
    arr = np.array([[x, y] for x, y, _, _, _ in players], dtype=float)
    return PlayerSet.from_arrays(
        positions=from_reference_frame(
            arr, pitch_length=REFERENCE_FIELD_DIMEN[0], pitch_width=REFERENCE_FIELD_DIMEN[1]
        ),
        velocities=np.array([[vx, vy] for _, _, vx, vy, _ in players], dtype=float),
        is_gk=np.array([gk for *_, gk in players], dtype=bool),
        ids=[str(i + 1) for i in range(len(players))],
    )


def _ball(scene: Scene) -> np.ndarray:
    """The scene's ball position in our canonical frame.

    Args:
        scene: The scene.

    Returns:
        ``(2,)`` array, NaN preserved.
    """
    return from_reference_frame(
        np.array(scene.ball, dtype=float),
        pitch_length=REFERENCE_FIELD_DIMEN[0],
        pitch_width=REFERENCE_FIELD_DIMEN[1],
    )


def our_result(
    scene: Scene, dtype: Any = np.float64, offsides: bool = False
) -> PitchControlResult:
    """Run our model on a scene in the reference's configuration.

    ``clamp="raw"`` because the reference has no clamp; clamping one side and not
    the other would be comparing two different quantities.

    Args:
        scene: The scene to evaluate.
        dtype: Working floating point type.
        offsides: Whether to drop offside attackers first.

    Returns:
        The full result, including the ``converged`` mask this file relies on.
    """
    return pitch_control_full(
        _player_set(scene.att),
        _player_set(scene.dfd),
        ball_pos=_ball(scene),
        grid=reference_grid(n_x=SURFACE_SHAPE[1]),
        params=PitchControlParams(dtype=dtype),
        offsides=offsides,
        attack_direction=1,
        clamp="raw",
    )


def reference_step_counts(scene: Scene) -> np.ndarray:
    """How many integration steps the reference takes at each cell of a scene.

    The reference builds ``dT_array = np.arange(btt - dt, btt + max_int_time,
    dt)`` per cell and integrates indices 1 onward, so it takes
    ``len(dT_array) - 1`` steps. That length is 251 or 252 depending purely on the
    floating point value of the ball travel time ``btt``, so the reference is not
    even self-consistent about its own step count. Ours is a fixed 250.

    Args:
        scene: The scene, whose ball position sets the travel times.

    Returns:
        ``(32, 50)`` integer array of the reference's step count per cell.
    """
    params = PitchControlParams()
    grid = reference_grid(n_x=SURFACE_SHAPE[1])
    ball = _ball(scene)
    if np.isfinite(ball).all():
        btt = np.linalg.norm(grid.targets - ball, axis=1) / params.average_ball_speed
    else:
        btt = np.zeros(grid.targets.shape[0], dtype=float)
    counts = [
        len(np.arange(t - params.int_dt, t + params.max_int_time, params.int_dt)) - 1 for t in btt
    ]
    return np.asarray(counts, dtype=int).reshape(grid.shape)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two arrays flattened in C order.

    Args:
        a: First array.
        b: Second array.

    Returns:
        The correlation coefficient.
    """
    return float(np.corrcoef(np.asarray(a).ravel(), np.asarray(b).ravel())[0, 1])


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def patched_reference(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A patched copy of ``validation/reference/``, leaving the vendored bytes alone.

    Args:
        tmp_path_factory: pytest's session-scoped temporary directory factory.

    Returns:
        The directory holding the patched copy.
    """
    manifest = read_manifest()
    before = {name: sha256(REFERENCE_DIR / name) for name in manifest}
    work = tmp_path_factory.mktemp("fot_reference")
    for name in manifest:
        shutil.copy2(REFERENCE_DIR / name, work / name)
    for name, hunks in parse_unified_diff(PATCH_PATH.read_text()).items():
        target = work / name
        target.write_text(apply_unified_diff(target.read_text(), hunks))
    after = {name: sha256(REFERENCE_DIR / name) for name in manifest}
    assert after == before, "patching the copy modified the vendored originals"
    return work


@pytest.fixture(scope="session")
def reference_module(patched_reference: Path) -> ModuleType:
    """Import ``Metrica_PitchControl`` from the patched copy.

    Args:
        patched_reference: The patched directory.

    Returns:
        The imported module.
    """
    return _import_from(patched_reference / MODEL_FILE, "fot_pitchcontrol")


def _import_from(path: Path, name: str) -> ModuleType:
    """Import a module from an explicit file path.

    Args:
        path: The ``.py`` file.
        name: The name to register it under.

    Returns:
        The imported module.

    Raises:
        ImportError: If the file cannot be turned into a module spec.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class Comparison:
    """Both implementations' output for one scene, plus the derived differences.

    Attributes:
        scene: The scene that produced these surfaces.
        reference: The reference surface, ``(32, 50)`` float64.
        ours: Our surface in float64, ``clamp="raw"``.
        ours_f32: Our surface at the shipped float32 dtype, cast up for comparison.
        converged: Our per-cell convergence mask.
    """

    scene: Scene
    reference: np.ndarray
    ours: np.ndarray
    ours_f32: np.ndarray
    converged: np.ndarray

    @property
    def abs_diff(self) -> np.ndarray:
        """``|ours - reference|`` per cell."""
        return np.abs(self.ours - self.reference)

    @property
    def abs_diff_f32(self) -> np.ndarray:
        """``|ours_float32 - reference|`` per cell."""
        return np.abs(self.ours_f32 - self.reference)

    @property
    def r(self) -> float:
        """Pearson correlation of our float64 surface against the reference."""
        return pearson(self.ours, self.reference)

    @property
    def r_f32(self) -> float:
        """Pearson correlation of our float32 surface against the reference."""
        return pearson(self.ours_f32, self.reference)


@pytest.fixture(scope="session")
def comparisons(reference_module: ModuleType) -> dict[str, Comparison]:
    """Run both implementations over every scene, once per session.

    The reference is a per-cell Python loop, so this is the expensive part of the
    file; every test below reads these arrays rather than recomputing them.

    Args:
        reference_module: The patched reference model.

    Returns:
        One :class:`Comparison` per scene, keyed by scene name.
    """
    out: dict[str, Comparison] = {}
    for scene in SCENES:
        res = our_result(scene, np.float64)
        res32 = our_result(scene, np.float32)
        out[scene.name] = Comparison(
            scene=scene,
            reference=reference_surface(reference_module, scene),
            ours=np.asarray(res.attacking, dtype=float),
            ours_f32=np.asarray(res32.attacking, dtype=float),
            converged=np.asarray(res.converged, dtype=bool),
        )
    return out


# --------------------------------------------------------------------------- #
# The vendored reference itself
# --------------------------------------------------------------------------- #


def test_vendored_reference_is_byte_identical_to_its_manifest() -> None:
    """Every file in ``validation/reference/`` still hashes to its recorded digest.

    Without this, "we agree with the reference" could be satisfied by quietly
    editing the reference.
    """
    manifest = read_manifest()
    assert manifest, "MANIFEST.sha256 is empty"
    for name, expected in sorted(manifest.items()):
        path = REFERENCE_DIR / name
        assert path.is_file(), f"vendored file {name} is missing"
        assert sha256(path) == expected, f"vendored file {name} has been modified"


def test_compatibility_patch_applies_and_spares_the_model_file(patched_reference: Path) -> None:
    """The patch touches only the two loaders, and fixes what it claims to fix.

    The model under comparison must be upstream's bytes, or the agreement measured
    below would be partly an agreement with our own edits.
    """
    hunks = parse_unified_diff(PATCH_PATH.read_text())
    assert set(hunks) == set(PATCHED_FILES), f"patch touches {sorted(hunks)}"
    assert sha256(patched_reference / MODEL_FILE) == read_manifest()[MODEL_FILE]

    io_text = (patched_reference / "Metrica_IO.py").read_text()
    vel_text = (patched_reference / "Metrica_Velocities.py").read_text()
    assert "idxmax(2)" not in io_text
    assert "idxmax(axis=1)" not in io_text
    assert "idxmax(2)" not in vel_text
    assert vel_text.count("mode='nearest'") == 4


def test_patched_loaders_import_under_the_installed_pandas_and_scipy(
    patched_reference: Path,
) -> None:
    """The patched loaders import cleanly, which is the whole point of the patch.

    Not needed by the comparison below -- the scenes are synthetic -- but it is
    the claim ``PROVENANCE.md`` makes about ``modern_pandas.patch``, and it is
    free to check.
    """
    for i, name in enumerate(PATCHED_FILES):
        module = _import_from(patched_reference / name, f"fot_loader_{i}")
        assert module is not None


def test_our_defaults_equal_the_reference_parameter_dict(reference_module: ModuleType) -> None:
    """A correlation against the reference tests our code only if the constants match.

    ``test_pitchcontrol.py`` checks these against transcribed literals; this checks
    them against the dict the reference actually builds at run time, so the two
    cannot drift apart silently.

    One constant is not bit-identical and this test records rather than hides it.
    We write ``lambda_gk = 12.9``; the reference evaluates
    ``4.3 * kappa_def * 3.0 = 12.899999999999999``. The gap is 1.7763568394e-15,
    which is exactly ``np.spacing(12.9)`` -- one ulp, the last bit of a float64.
    It is asserted to be exactly that, so a real drift in the goalkeeper boost
    could not hide behind the tolerance. The measured surface agreement of
    3.2196e-15 above is the empirical answer to whether one ulp here matters.
    """
    ref = reference_module.default_model_params()
    ours = PitchControlParams()
    assert ours.max_player_speed == ref["max_player_speed"]
    assert ours.reaction_time == ref["reaction_time"]
    assert ours.max_player_accel == ref["max_player_accel"]
    assert ours.tti_sigma == ref["tti_sigma"]
    assert ours.kappa_def == ref["kappa_def"]
    assert ours.lambda_att == ref["lambda_att"]
    assert ours.lambda_def == ref["lambda_def"]
    assert ours.lambda_gk - ref["lambda_gk"] == float(np.spacing(12.9))
    assert ours.average_ball_speed == ref["average_ball_speed"]
    assert ours.int_dt == ref["int_dt"]
    assert ours.max_int_time == ref["max_int_time"]
    assert ours.model_converge_tol == ref["model_converge_tol"]
    assert ours.time_to_control_att == ref["time_to_control_att"]
    assert ours.time_to_control_def == ref["time_to_control_def"]


def test_reference_grid_matches_the_reference_cell_centres(reference_module: ModuleType) -> None:
    """Our ``reference_grid()`` is the reference's own grid, shifted into our frame.

    Comparing surfaces cell by cell is meaningless unless the cells are the same
    cells. The reference returns its ``xgrid`` and ``ygrid``; those are compared
    here against ours to the last bit.
    """
    scene = SCENES_BY_NAME["deep_goalkeeper"]
    frame = 7
    home_cols, home_gk = _columns(scene.att, "Home")
    away_cols, away_gk = _columns(scene.dfd, "Away")
    _surface, xgrid, ygrid = reference_module.generate_pitch_control_for_event(
        0,
        pd.DataFrame([{"Start Frame": frame, "Team": "Home", "Start X": 0.0, "Start Y": 0.0}]),
        pd.DataFrame([home_cols], index=[frame]),
        pd.DataFrame([away_cols], index=[frame]),
        reference_module.default_model_params(),
        (home_gk, away_gk),
        field_dimen=REFERENCE_FIELD_DIMEN,
        n_grid_cells_x=SURFACE_SHAPE[1],
        offsides=False,
    )
    grid = reference_grid(n_x=SURFACE_SHAPE[1])
    assert grid.shape == SURFACE_SHAPE
    centres = from_reference_frame(
        np.stack([np.asarray(xgrid), np.zeros_like(np.asarray(xgrid))], axis=1),
        pitch_length=REFERENCE_FIELD_DIMEN[0],
        pitch_width=REFERENCE_FIELD_DIMEN[1],
    )
    np.testing.assert_allclose(grid.x, centres[:, 0], rtol=0, atol=1e-12)
    centres_y = from_reference_frame(
        np.stack([np.zeros_like(np.asarray(ygrid)), np.asarray(ygrid)], axis=1),
        pitch_length=REFERENCE_FIELD_DIMEN[0],
        pitch_width=REFERENCE_FIELD_DIMEN[1],
    )
    np.testing.assert_allclose(grid.y, centres_y[:, 1], rtol=0, atol=1e-12)


# --------------------------------------------------------------------------- #
# Agreement
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", [s.name for s in SCENES])
def test_scene_agrees_with_the_reference(name: str, comparisons: dict[str, Comparison]) -> None:
    """Each scene, cell by cell, in float64.

    Measured over these ten scenes: worst per-scene r = 0.999999288256, worst
    per-scene mean |difference| = 1.2463e-05, worst |difference| on a converged
    cell = 3.2196e-15, worst |difference| anywhere = 1.131579e-02. The bounds are
    those figures with the margins recorded in the module docstring.
    """
    c = comparisons[name]
    assert c.reference.shape == SURFACE_SHAPE, f"reference produced {c.reference.shape}"
    assert c.ours.shape == SURFACE_SHAPE, f"we produced {c.ours.shape}"
    assert np.isfinite(c.ours).all(), "our surface contains a non-finite cell"
    assert np.isfinite(c.reference).all(), "the reference surface contains a non-finite cell"

    diff = c.abs_diff
    assert c.r >= MIN_PER_SCENE_R, f"{name}: r = {c.r:.12f}"
    assert float(diff.mean()) <= MAX_MEAN_ABS_DIFF, f"{name}: mean |diff| = {diff.mean():.4e}"
    assert float(diff.max()) <= MAX_ABS_DIFF_ANY, f"{name}: max |diff| = {diff.max():.4e}"
    if c.converged.any():
        worst = float(diff[c.converged].max())
        assert worst <= MAX_ABS_DIFF_CONVERGED, (
            f"{name}: max |diff| on a converged cell = {worst:.4e}, "
            f"bound {MAX_ABS_DIFF_CONVERGED:.0e}"
        )


def test_pooled_agreement_over_every_cell_of_every_scene(
    comparisons: dict[str, Comparison],
) -> None:
    """The headline number, pooled over all 16,000 cells.

    Measured: pooled Pearson r = 0.999999972693, mean |difference| = 1.6494e-06,
    16,000 cells across 10 scenes, of which 15,984 converged and the worst
    converged-cell difference was 3.2196e-15.
    """
    ours = np.concatenate([c.ours.ravel() for c in comparisons.values()])
    ref = np.concatenate([c.reference.ravel() for c in comparisons.values()])
    converged = np.concatenate([c.converged.ravel() for c in comparisons.values()])
    assert ours.size == len(SCENES) * SURFACE_SHAPE[0] * SURFACE_SHAPE[1] == 16000

    pooled = pearson(ours, ref)
    assert pooled >= MIN_POOLED_R, f"pooled r = {pooled:.12f}"
    assert float(np.abs(ours - ref).mean()) <= MAX_MEAN_ABS_DIFF
    assert float(np.abs(ours - ref)[converged].max()) <= MAX_ABS_DIFF_CONVERGED
    assert float(np.abs(ours - ref).max()) <= MAX_ABS_DIFF_ANY
    # The overwhelming majority of cells converge; if that stopped being true the
    # converged-cell bound above would be quietly testing almost nothing.
    assert converged.mean() >= 0.99, f"only {converged.mean():.4f} of cells converged"


def test_disagreement_is_confined_to_cells_that_never_converged(
    comparisons: dict[str, Comparison],
) -> None:
    """Every cell that differs at all is a cell the reference could not converge.

    This is the structural version of the 1.1e-2 bound and the reason that bound
    is not evidence of a modelling difference. The reference builds its time axis
    as ``arange(btt - dt, btt + max_int_time, dt)``, whose length is 251 or 252
    depending on the floating point value of the ball travel time, and integrates
    one step per element after the first; ours integrates a fixed 250. On a cell
    that converges the extra step is worth nothing, and 8 of 16,000 cells -- all
    of them unconverged -- are the entire consequence.
    """
    offenders: list[str] = []
    for name, c in comparisons.items():
        disagree = c.abs_diff > DISAGREEMENT_FLOOR
        leaked = int((disagree & c.converged).sum())
        if leaked:
            offenders.append(f"{name}: {leaked} converged cells differ by more than 1e-12")
    assert not offenders, "; ".join(offenders)


def test_disagreement_is_exactly_the_reference_taking_one_extra_euler_step(
    comparisons: dict[str, Comparison],
) -> None:
    """Split every cell by whether the reference took the same 250 steps we did.

    This is the sharp form of the previous test, and the one that would notice a
    change in *our* step count. Measured over the 16,000 cells:

    * 10,158 cells where the reference also took 250 steps: worst |difference|
      3.2196e-15, and not one cell above 1e-12.
    * 5,842 cells where the reference took 251: worst |difference| 1.1316e-02,
      concentrated in 8 cells, every one of them a cell neither implementation
      converged.

    So the step-count classes, not the geometry, explain the whole residual. The
    matched class is held to the converged-cell bound, which means a regression
    that changed our 250 to 249 or 251 would surface here even though it is
    invisible on cells that converge.
    """
    matched_total = 0
    extra_total = 0
    for name, c in comparisons.items():
        steps = reference_step_counts(c.scene)
        assert set(np.unique(steps).tolist()) <= {250, 251}, f"{name}: steps {np.unique(steps)}"
        matched = steps == 250
        extra = steps == 251
        matched_total += int(matched.sum())
        extra_total += int(extra.sum())
        if matched.any():
            worst = float(c.abs_diff[matched].max())
            assert worst <= MAX_ABS_DIFF_CONVERGED, (
                f"{name}: {int(matched.sum())} cells where the reference took the "
                f"same 250 steps we did disagree by up to {worst:.4e}"
            )
        if extra.any():
            assert float(c.abs_diff[extra].max()) <= MAX_ABS_DIFF_ANY
        # Cells the extra step does reach must be cells that never converged.
        leaked = int(((c.abs_diff > DISAGREEMENT_FLOOR) & c.converged).sum())
        assert leaked == 0, f"{name}: {leaked} converged cells disagree"
    assert matched_total > 0 and extra_total > 0, (
        f"the scenes no longer cover both step-count classes: {matched_total} matched, "
        f"{extra_total} with an extra step"
    )


def test_float32_agrees_with_the_reference_too(comparisons: dict[str, Comparison]) -> None:
    """The shipped dtype is float32, so it gets its own bound.

    Measured: worst per-scene r = 0.999999288319 (the same scene, and the same
    unconverged cells, that limit float64), worst |difference| on a converged
    cell = 1.1291e-06.
    """
    for name, c in comparisons.items():
        assert c.r_f32 >= MIN_PER_SCENE_R_F32, f"{name}: float32 r = {c.r_f32:.12f}"
        if c.converged.any():
            worst = float(c.abs_diff_f32[c.converged].max())
            assert worst <= MAX_ABS_DIFF_CONVERGED_F32, f"{name}: float32 max |diff| = {worst:.4e}"


# --------------------------------------------------------------------------- #
# Alignment guards
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", [s.name for s in SCENES])
def test_alignment_guards_rule_out_a_flipped_or_inverted_surface(
    name: str, comparisons: dict[str, Comparison]
) -> None:
    """A high correlation is worthless if it is a correlation with the wrong array.

    Guards, per scene: the surface is ``(n_y, n_x) = (32, 50)`` and not its
    transpose; a y-flipped reference correlates strictly worse than the aligned
    one, and differs from ours by at least 100x as much (measured minimum ratio
    1.96e4); an x-flipped reference likewise; the aligned correlation is positive;
    and ``1 - reference`` anti-correlates, because the surface has to be one
    team's share of a two-team probability rather than an arbitrary scalar field.
    """
    c = comparisons[name]
    assert c.ours.shape == c.reference.shape == SURFACE_SHAPE
    assert SURFACE_SHAPE[0] != SURFACE_SHAPE[1], "a square grid would hide a transpose"

    aligned = c.r
    flipud = pearson(c.ours, np.flipud(c.reference))
    fliplr = pearson(c.ours, np.fliplr(c.reference))
    inverted = pearson(c.ours, 1.0 - c.reference)

    assert flipud < aligned, f"{name}: y-flipped reference correlates at {flipud:+.6f}: upside down"
    assert fliplr < aligned, f"{name}: x-flipped reference correlates at {fliplr:+.6f}: mirrored"
    assert aligned > 0.0, f"{name}: aligned correlation {aligned:+.6f} is not positive"
    assert inverted < 0.0, f"{name}: 1 - reference correlates at {inverted:+.6f}, not anti"

    aligned_mad = float(c.abs_diff.mean())
    flipud_mad = float(np.abs(c.ours - np.flipud(c.reference)).mean())
    assert flipud_mad > 100.0 * aligned_mad, (
        f"{name}: mean |diff| against a y-flipped reference is {flipud_mad:.4e}, "
        f"only {flipud_mad / max(aligned_mad, 1e-300):.2e}x the aligned {aligned_mad:.4e}"
    )


def test_pooled_alignment_guards(comparisons: dict[str, Comparison]) -> None:
    """The same guards pooled, where the margins are wide and easy to read.

    Measured over all 16,000 cells: aligned r = 0.999999972693, y-flipped
    r = +0.663621, x-flipped r = -0.411080, ``1 - reference`` r = -1.000000, mean
    |difference| aligned 1.6494e-06 against 0.1858 y-flipped.
    """
    ours = np.concatenate([c.ours.ravel() for c in comparisons.values()])
    ref = np.concatenate([c.reference.ravel() for c in comparisons.values()])
    flipped_ud = np.concatenate([np.flipud(c.reference).ravel() for c in comparisons.values()])
    flipped_lr = np.concatenate([np.fliplr(c.reference).ravel() for c in comparisons.values()])

    assert pearson(ours, ref) >= MIN_POOLED_R
    assert pearson(ours, flipped_ud) <= 0.9
    assert pearson(ours, flipped_lr) <= 0.9
    assert pearson(ours, 1.0 - ref) <= -0.9
    assert float(np.abs(ours - flipped_ud).mean()) >= 0.05


# --------------------------------------------------------------------------- #
# Offsides
# --------------------------------------------------------------------------- #


def test_offside_filter_and_surface_agree_with_the_reference(
    reference_module: ModuleType,
) -> None:
    """With offsides on, both implementations drop the same players.

    Our :func:`~pressing.pitchcontrol.check_offsides` takes an explicit
    ``attack_direction`` where the reference infers the same quantity from
    ``sign(defending goalkeeper x)``. That substitution is only safe if it agrees,
    so it is checked here on the offside scene at the level of the surviving
    players and again at the level of the whole surface.

    Measured on this scene: 3 of 6 attackers survive in both, r = 1.000000000000,
    max |difference| = 2.220e-15.
    """
    scene = SCENES_BY_NAME["offside_attackers"]
    params = reference_module.default_model_params()
    home_cols, home_gk = _columns(scene.att, "Home")
    away_cols, away_gk = _columns(scene.dfd, "Away")
    ref_att = reference_module.initialise_players(home_cols, "Home", params, home_gk)
    ref_def = reference_module.initialise_players(away_cols, "Away", params, away_gk)
    kept_ref = reference_module.check_offsides(
        ref_att, ref_def, np.array(scene.ball, dtype=float), (home_gk, away_gk)
    )
    kept_ours = check_offsides(
        _player_set(scene.att),
        _player_set(scene.dfd),
        _ball(scene),
        attack_direction=1,
        pitch_length=REFERENCE_FIELD_DIMEN[0],
    )
    assert len(kept_ref) < len(ref_att), "the offside scene has nobody offside"
    assert sorted(str(p.id) for p in kept_ref) == sorted(kept_ours.ids) == ["4", "5", "6"]

    ref_surf = reference_surface(reference_module, scene, offsides=True)
    ours = np.asarray(our_result(scene, np.float64, offsides=True).attacking, dtype=float)
    assert ours.shape == ref_surf.shape == SURFACE_SHAPE
    assert pearson(ours, ref_surf) >= MIN_PER_SCENE_R
    assert float(np.abs(ours - ref_surf).max()) <= MAX_ABS_DIFF_CONVERGED


def test_offside_line_tracks_the_ball_exactly_as_the_reference_does(
    reference_module: ModuleType,
) -> None:
    """Sweep the ball up the pitch and check the offside line moves identically.

    The offside line is ``max(second-deepest defender, ball, halfway) + 0.2``. On
    the scene above the second-deepest defender sits at x = 30, so the ball term
    only binds once the ball goes past him -- and a single ball position therefore
    tests only one of the three arguments to that ``max``. Sweeping the ball from
    behind the halfway line to the byline exercises all three, and the two
    implementations must agree at every step.

    No surface is computed here, so the sweep is free.
    """
    scene = SCENES_BY_NAME["offside_attackers"]
    params = reference_module.default_model_params()
    home_cols, home_gk = _columns(scene.att, "Home")
    away_cols, away_gk = _columns(scene.dfd, "Away")
    ours_def = _player_set(scene.dfd)

    outcomes: set[tuple[str, ...]] = set()
    for ball_x in (-52.0, -30.0, -0.1, 0.1, 20.0, 29.9, 30.1, 33.0, 44.1, 45.5, 46.5, 52.0):
        ball = np.array([ball_x, 0.0], dtype=float)
        kept_ref = reference_module.check_offsides(
            reference_module.initialise_players(home_cols, "Home", params, home_gk),
            reference_module.initialise_players(away_cols, "Away", params, away_gk),
            ball,
            (home_gk, away_gk),
        )
        kept_ours = check_offsides(
            _player_set(scene.att),
            ours_def,
            from_reference_frame(
                ball,
                pitch_length=REFERENCE_FIELD_DIMEN[0],
                pitch_width=REFERENCE_FIELD_DIMEN[1],
            ),
            attack_direction=1,
            pitch_length=REFERENCE_FIELD_DIMEN[0],
        )
        got_ref = tuple(sorted(str(p.id) for p in kept_ref))
        got_ours = tuple(sorted(kept_ours.ids))
        assert got_ref == got_ours, (
            f"ball at x={ball_x}: reference kept {got_ref}, we kept {got_ours}"
        )
        outcomes.add(got_ref)
    assert len(outcomes) >= 3, (
        f"the sweep produced only {len(outcomes)} distinct offside lines, so it is "
        "not actually exercising the ball term of the max()"
    )


# --------------------------------------------------------------------------- #
# The scenes themselves
# --------------------------------------------------------------------------- #


def test_every_scene_is_documented_and_distinct() -> None:
    """Each scene states which property of the model it stresses, and is unique.

    Cheap, but it is what stops this file decaying into ten variations on the same
    convenient midfield geometry.
    """
    assert len(SCENES) == 10
    assert len({s.name for s in SCENES}) == len(SCENES)
    geometries = {(s.att, s.dfd, s.ball) for s in SCENES}
    assert len(geometries) == len(SCENES), "two scenes have identical geometry"
    for scene in SCENES:
        assert len(scene.why) > 80, f"{scene.name} does not say why it exists"
        assert len(scene.att) >= 1 and len(scene.dfd) >= 1


def test_the_scenes_actually_cover_the_adversarial_cases() -> None:
    """The adversarial properties are asserted, not just asserted in prose.

    If a future edit tidies a scene into something convenient -- moves the
    off-pitch player back inside the touchline, gives the stationary players a
    velocity, separates the coincident pair -- this fails rather than silently
    weakening the comparison.
    """
    half_l, half_w = REFERENCE_FIELD_DIMEN[0] / 2.0, REFERENCE_FIELD_DIMEN[1] / 2.0
    every = [p for s in SCENES for p in (*s.att, *s.dfd)]

    # players on and beyond the boundary
    assert any(abs(x) >= half_l for x, _y, *_ in every if not np.isnan(x))
    assert any(abs(y) >= half_w for _x, y, *_ in every if not np.isnan(y))
    assert any(abs(x) > half_l for x, _y, *_ in every if not np.isnan(x))

    # a deep goalkeeper, and a goalkeeper on each side of the same scene
    gk = SCENES_BY_NAME["deep_goalkeeper"]
    assert any(is_gk and x >= 45.0 for x, _y, _vx, _vy, is_gk in gk.dfd)
    assert any(is_gk for *_, is_gk in gk.att)

    # coincident players within a team and across teams
    co = SCENES_BY_NAME["coincident_players"]
    att_xy = [(x, y) for x, y, *_ in co.att]
    def_xy = [(x, y) for x, y, *_ in co.dfd]
    assert len(set(att_xy)) < len(att_xy)
    assert len(set(def_xy)) < len(def_xy)
    assert set(att_xy) & set(def_xy)

    # exactly-zero velocities
    assert sum(vx == 0.0 and vy == 0.0 for _x, _y, vx, vy, _gk in every) >= 8

    # a ball a long way from every player
    far = SCENES_BY_NAME["ball_far_from_everyone"]
    ball = np.array(far.ball, dtype=float)
    nearest = min(
        float(np.hypot(x - ball[0], y - ball[1]))
        for x, y, *_ in far.att
        if not np.isnan(x)
    )
    assert nearest > 90.0, f"nearest attacker is {nearest:.1f} m from the ball"

    # offside attackers: at least three beyond the second-deepest defender
    off = SCENES_BY_NAME["offside_attackers"]
    line = sorted((x for x, _y, *_ in off.dfd), reverse=True)[1]
    assert sum(x > line + 0.2 for x, _y, *_ in off.att) >= 3

    # the exact-tie scene really is an exact x-mirror
    tie = SCENES_BY_NAME["exact_mirror_tie"]
    mirrored = tuple((-x, y, -vx, vy, gk) for x, y, vx, vy, gk in tie.att)
    assert set(mirrored) == set(tie.dfd)

    # NaN positions and NaN velocities both present
    nan_scene = SCENES_BY_NAME["nan_positions_and_velocities"]
    members = (*nan_scene.att, *nan_scene.dfd)
    assert any(np.isnan(x) or np.isnan(y) for x, y, *_ in members)
    assert any(
        (np.isnan(vx) or np.isnan(vy)) and not (np.isnan(x) or np.isnan(y))
        for x, y, vx, vy, _gk in members
    )
    assert all(np.isnan(v) for v in SCENES_BY_NAME["nan_ball_position"].ball)


# --------------------------------------------------------------------------- #
# Reproducing the numbers in the module docstring.
#
#   uv run python tests/test_pitchcontrol_reference.py
#
# The tests above assert bounds; this prints the measurements those bounds were
# derived from, so nothing in the docstring is a figure nobody can regenerate.
# --------------------------------------------------------------------------- #


def _measure() -> None:  # pragma: no cover - developer entry point, not a test
    """Print the per-scene and pooled agreement table."""
    import contextlib
    import tempfile

    work = Path(tempfile.mkdtemp(prefix="fot_measure_"))
    try:
        for name in read_manifest():
            shutil.copy2(REFERENCE_DIR / name, work / name)
        for name, hunks in parse_unified_diff(PATCH_PATH.read_text()).items():
            (work / name).write_text(apply_unified_diff((work / name).read_text(), hunks))
        module = _import_from(work / MODEL_FILE, "fot_measure")

        rows: list[Comparison] = []
        for scene in SCENES:
            with contextlib.redirect_stdout(io.StringIO()):
                reference = reference_surface(module, scene)
            res = our_result(scene, np.float64)
            res32 = our_result(scene, np.float32)
            rows.append(
                Comparison(
                    scene=scene,
                    reference=reference,
                    ours=np.asarray(res.attacking, dtype=float),
                    ours_f32=np.asarray(res32.attacking, dtype=float),
                    converged=np.asarray(res.converged, dtype=bool),
                )
            )

        head = f"{'scene':32s} {'r':>14s} {'r_f32':>14s} {'max|d|':>10s} {'mean|d|':>10s}"
        print(f"{head} {'flipud':>8s} {'fliplr':>8s} {'1-ref':>8s}")
        for c in rows:
            print(
                f"{c.scene.name:32s} {c.r:14.12f} {c.r_f32:14.12f} "
                f"{c.abs_diff.max():10.3e} {c.abs_diff.mean():10.3e} "
                f"{pearson(c.ours, np.flipud(c.reference)):+8.4f} "
                f"{pearson(c.ours, np.fliplr(c.reference)):+8.4f} "
                f"{pearson(c.ours, 1.0 - c.reference):+8.4f}"
            )

        ours = np.concatenate([c.ours.ravel() for c in rows])
        ref = np.concatenate([c.reference.ravel() for c in rows])
        f32 = np.concatenate([c.abs_diff_f32.ravel() for c in rows])
        conv = np.concatenate([c.converged.ravel() for c in rows])
        steps = np.concatenate([reference_step_counts(c.scene).ravel() for c in rows])
        diff = np.abs(ours - ref)
        print()
        print(f"scenes compared                         : {len(rows)}")
        print(f"cells compared                          : {ours.size:,}")
        print(f"pooled Pearson r                        : {pearson(ours, ref):.12f}")
        print(f"worst per-scene Pearson r               : {min(c.r for c in rows):.12f}")
        print(
            f"max |difference| over converged cells   : {diff[conv].max():.4e}  "
            f"({int(conv.sum()):,} of {ours.size:,} cells)"
        )
        print(f"max |difference| over every cell        : {diff.max():.6e}")
        print(f"mean |difference| pooled                : {diff.mean():.4e}")
        print(
            "worst per-scene mean |difference|       : "
            f"{max(float(c.abs_diff.mean()) for c in rows):.4e}"
        )
        print(f"cells disagreeing by more than 1e-12    : {int((diff > 1e-12).sum())}")
        print(f"  of those, on a converged cell         : {int(((diff > 1e-12) & conv).sum())}")
        print(f"float32: worst per-scene Pearson r      : {min(c.r_f32 for c in rows):.12f}")
        print(f"float32: max |diff| over converged cells: {f32[conv].max():.4e}")
        print()
        for n in (250, 251):
            sel = steps == n
            print(
                f"reference took {n} steps                : {int(sel.sum()):,} cells, "
                f"worst |diff| {diff[sel].max():.4e}, above 1e-12: "
                f"{int((diff[sel] > 1e-12).sum())}"
            )
        print()
        flipped = np.concatenate([np.flipud(c.reference).ravel() for c in rows])
        mirrored = np.concatenate([np.fliplr(c.reference).ravel() for c in rows])
        print(f"pooled r vs y-flipped reference         : {pearson(ours, flipped):+.6f}")
        print(f"pooled r vs x-flipped reference         : {pearson(ours, mirrored):+.6f}")
        print(f"pooled r vs (1 - reference)             : {pearson(ours, 1.0 - ref):+.6f}")
        print(f"pooled mean |diff| vs y-flipped         : {np.abs(ours - flipped).mean():.4f}")
        ratio = min(
            float(np.abs(c.ours - np.flipud(c.reference)).mean())
            / max(float(c.abs_diff.mean()), 1e-300)
            for c in rows
        )
        print(f"minimum per-scene mad_flipud / mad      : {ratio:.2e}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":  # pragma: no cover
    _measure()
