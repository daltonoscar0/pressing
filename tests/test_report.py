"""Tests for the six-page PDF.

A report is not a dataframe, so the interesting failures are not the ones a
type check finds. Four of them are worth a test each:

1. **The page count is a contract.** Six pages, whatever the match is. Content
   that overflows a frame silently becomes a seventh page, and every other
   assertion in this file would still pass on it.
2. **The disclosure has to survive extraction.** A report built from
   event-proxy estimates says so in three places, and the one a machine can
   check is the flat opaque footer line -- the diagonal watermark is rotated,
   so a real PDF reader hands its characters back out of order, and a naive
   grep for it fails on a perfectly good report. The test therefore asserts on
   the footer string as one contiguous run, which the watermark never produces.
3. **Player names have to survive the font.** reportlab's built-in Helvetica
   replaces every character outside Latin-1 with a box and raises nothing, so a
   squad list can be quietly corrupted in exactly the column names are read
   from. A fixture squad with a c-acute, a g-breve and a slashed L is built
   into a real PDF and read back out of the text layer.
4. **Real data is degenerate.** A phase with no losses in it, a match with
   fewer scorable moments than the report prints panels for, a loss with no
   located ball, a player with no name, a squad of thirty. None of these may
   raise, and none may quietly change the page count.

The text-layer checks use a small PDF reader written for this file rather than
an external binary, so they run everywhere ``pytest`` does. It undoes the two
stream filters reportlab actually emits (ASCII85 then Flate), reads the
``/ToUnicode`` CMap of each embedded font subset, and decodes the text-showing
operators through it -- which is the same path a PDF viewer takes, and the only
way to prove a glyph maps back to the character it was asked to draw.
"""

from __future__ import annotations

import base64
import re
import time
import zlib
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import pytest

from pressing import report
from pressing.counterpress import LOSS_SCORE_COLUMNS, CounterPressParams
from pressing.pitchcontrol import make_grid
from pressing.schema import PITCH_LENGTH, PITCH_WIDTH, TRACKING_COLUMNS

matplotlib.use("Agg")

#: Small enough that a whole report builds in a couple of seconds, big enough
#: that the surfaces are real surfaces and the panels are real pictures.
TEST_GRID = make_grid(n_x=24)

#: Characters outside Latin-1. reportlab's Helvetica turns each of these into a
#: box glyph with no exception raised; see the module docstring.
AWKWARD_NAMES = ("Vuković", "Gündoğan", "Łukasz", "Şahin")


# --------------------------------------------------------------------------- #
# A synthetic match
# --------------------------------------------------------------------------- #


def make_tracking(
    *,
    n_players: int = 6,
    n_frames: int = 800,
    fps: float = 25.0,
    names: tuple[str, ...] | None = None,
    ball_gap: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Build a canonical tracking frame for a made-up match.

    Args:
        n_players: Players per team.
        n_frames: Frames in the (single-period) match.
        fps: Tracking rate.
        names: Player ids to use, cycled across both teams. Defaults to
            ``Player0``, ``Player1``, ...
        ball_gap: Half-open frame range where the ball is untracked, or None.

    Returns:
        A frame with :data:`pressing.schema.TRACKING_COLUMNS` and the ``attrs``
        the report reads.
    """
    rng = np.random.default_rng(17)
    frames = np.arange(1, n_frames + 1, dtype=np.int64)
    rows: list[pd.DataFrame] = []
    for side, sign in (("home", 1.0), ("away", -1.0)):
        for index in range(n_players):
            base_x = PITCH_LENGTH / 2.0 + sign * (6.0 + 4.0 * index)
            base_y = 6.0 + (PITCH_WIDTH - 12.0) * index / max(n_players - 1, 1)
            drift = np.linspace(0.0, sign * 9.0, frames.size)
            wobble = rng.normal(0.0, 0.35, frames.size).cumsum() * 0.05
            label = (
                names[(0 if side == "home" else n_players) + index]
                if names is not None
                else f"{'H' if side == 'home' else 'A'}{index}"
            )
            rows.append(
                pd.DataFrame(
                    {
                        "match_id": "synthetic",
                        "t": frames / fps,
                        "period": 1,
                        "frame": frames,
                        "team": side,
                        "player_id": label,
                        "jersey": index + 1 + (0 if side == "home" else 20),
                        "x": np.clip(base_x + drift + wobble, 1.0, PITCH_LENGTH - 1.0),
                        "y": np.clip(base_y + wobble, 1.0, PITCH_WIDTH - 1.0),
                        "vx": sign * 1.2,
                        "vy": 0.2,
                        "confidence": 1.0,
                    }
                )
            )
    ball_x = PITCH_LENGTH / 2.0 + 18.0 * np.sin(frames / 90.0)
    ball_y = PITCH_WIDTH / 2.0 + 12.0 * np.cos(frames / 70.0)
    if ball_gap is not None:
        lo, hi = ball_gap
        missing = (frames >= lo) & (frames < hi)
        ball_x = np.where(missing, np.nan, ball_x)
        ball_y = np.where(missing, np.nan, ball_y)
    rows.append(
        pd.DataFrame(
            {
                "match_id": "synthetic",
                "t": frames / fps,
                "period": 1,
                "frame": frames,
                "team": "ball",
                "player_id": "ball",
                "jersey": pd.NA,
                "x": ball_x,
                "y": ball_y,
                "vx": 0.0,
                "vy": 0.0,
                "confidence": 1.0,
            }
        )
    )
    tracking = pd.concat(rows, ignore_index=True)
    tracking = tracking.loc[:, list(TRACKING_COLUMNS)]
    tracking = tracking.sort_values(["frame", "team", "player_id"], ignore_index=True)
    tracking.attrs.update(
        {
            "match_id": "synthetic",
            "source": "metrica",
            "fps": fps,
            "attack_direction": {"home": 1, "away": -1},
            "goalkeeper": {"home": "H0", "away": "A0"} if names is None else {},
        }
    )
    return tracking


def make_losses(
    tracking: pd.DataFrame,
    *,
    n: int = 8,
    phases: tuple[str, ...] = ("build_up", "middle", "final_third"),
    unlocated: int = 0,
) -> pd.DataFrame:
    """Build a scored-loss table over a synthetic match.

    Args:
        tracking: The match the losses belong to.
        n: How many losses.
        phases: Phases to cycle through. Pass a shorter tuple to build a match
            where some phase has no losses at all.
        unlocated: How many of the losses have no located ball, and so cannot
            be scored.

    Returns:
        A frame with :data:`pressing.counterpress.LOSS_SCORE_COLUMNS`.
    """
    rng = np.random.default_rng(5)
    fps = float(tracking.attrs["fps"])
    last = int(tracking["frame"].max())
    # Leave five seconds of match after the last loss, so every window closes.
    frames = np.linspace(60, max(61, last - int(6 * fps)), n).astype(int)
    rows: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        losing = "home" if index % 2 == 0 else "away"
        located = index >= unlocated
        recovery_area = float(rng.uniform(0.2, 0.9))
        recovery_epv = float(rng.uniform(0.5, 0.95))
        rows.append(
            {
                "loss_id": index,
                "match_id": "synthetic",
                "t": frame / fps,
                "frame": int(frame),
                "period": 1,
                "losing_team": losing,
                "winning_team": "away" if losing == "home" else "home",
                "x": float(rng.uniform(5.0, PITCH_LENGTH - 5.0)) if located else np.nan,
                "y": float(rng.uniform(5.0, PITCH_WIDTH - 5.0)) if located else np.nan,
                "x_norm": float(rng.uniform(0.0, 1.0)),
                "phase": phases[index % len(phases)] if located else None,
                "location_source": "tracking_ball" if located else None,
                "losing_player": f"H{index % 3}" if losing == "home" else f"A{index % 3}",
                "subtype": "INTERCEPTION",
                "event_index": index * 2 + 1,
                "n_timepoints": 6 if located else 0,
                "n_ball_filled": 0,
                "recovery": recovery_area if located else np.nan,
                "recovery_area": recovery_area if located else np.nan,
                "recovery_epv": recovery_epv if located else np.nan,
                "share_t0_area": recovery_area if located else np.nan,
                "share_t0_epv": recovery_epv if located else np.nan,
                "share_end_area": recovery_area if located else np.nan,
                "share_end_epv": recovery_epv if located else np.nan,
                "epv_winner_t0": 0.004,
                "epv_winner_mean": 0.005,
                "epv_conceded": float(rng.normal(0.0, 0.002)) if located else np.nan,
                "regained_within_5s": bool(index % 3),
                "time_to_regain": 2.0 + index % 3 if index % 3 else np.nan,
                "scored": located,
                "skip_reason": None if located else "no_loss_location",
            }
        )
    losses = pd.DataFrame(rows).loc[:, list(LOSS_SCORE_COLUMNS)]
    losses.attrs.update(
        {
            "match_id": "synthetic",
            "fps": fps,
            "counterpress_params": CounterPressParams(),
            "weighting": "area",
            "alt_weighting": "epv",
            "ball_out_count": 12,
            "n_skipped": unlocated,
            "skip_reasons": {"no_loss_location": unlocated} if unlocated else {},
        }
    )
    return losses


def make_events(losses: pd.DataFrame, *, fps: float = 25.0) -> pd.DataFrame:
    """Build an events frame that regains the ball after some of the losses.

    Args:
        losses: The loss table the events must line up with.
        fps: Tracking rate.

    Returns:
        A canonical events frame in event order.
    """
    rows: list[dict[str, Any]] = []
    for row in losses.to_dict("records"):
        index = int(row["event_index"])
        while len(rows) < index:
            # Not a ball-carrying type, so it can never be mistaken for a
            # regain: this row exists only to line the event index up.
            rows.append(
                {
                    "type": "CHALLENGE",
                    "team": row["winning_team"],
                    "player": "filler",
                    "frame": int(row["frame"]) - 1,
                    "period": 1,
                }
            )
        rows.append(
            {
                "type": "BALL LOST",
                "team": row["losing_team"],
                "player": row["losing_player"],
                "frame": int(row["frame"]),
                "period": 1,
            }
        )
        if row["regained_within_5s"]:
            rows.append(
                {
                    "type": "RECOVERY",
                    "team": row["losing_team"],
                    "player": f"winner-{row['loss_id']}",
                    "frame": int(row["frame"]) + int(2 * fps),
                    "period": 1,
                }
            )
    events = pd.DataFrame(
        {
            "match_id": "synthetic",
            "competition": "synthetic",
            "season": pd.NA,
            "team_id": [row["team"] for row in rows],
            "team": [row["team"] for row in rows],
            "player_id": [row["player"] for row in rows],
            "player": [row["player"] for row in rows],
            "t": [row["frame"] / fps for row in rows],
            "period": [row["period"] for row in rows],
            "type": [row["type"] for row in rows],
            "subtype": pd.NA,
            "x": 50.0,
            "y": 30.0,
            "end_x": np.nan,
            "end_y": np.nan,
            "outcome": pd.NA,
            "pressure": False,
            "body_part": pd.NA,
            "extra": [{"start_frame": row["frame"]} for row in rows],
        }
    )
    events.attrs["fps"] = fps
    return events


@pytest.fixture(scope="module")
def match() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """A synthetic match with enough losses to fill every page."""
    tracking = make_tracking()
    losses = make_losses(tracking, n=9)
    return tracking, make_events(losses), losses


@pytest.fixture(scope="module")
def built(
    match: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """The plain report, built once and reused."""
    tracking, events, losses = match
    out = tmp_path_factory.mktemp("report") / "plain.pdf"
    return report.build_report(
        tracking, losses, "home", out, events=events, grid=TEST_GRID
    )


@pytest.fixture(scope="module")
def built_proxy(
    match: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """The same report, watermarked as event-proxy estimates."""
    tracking, events, losses = match
    out = tmp_path_factory.mktemp("report") / "proxy.pdf"
    return report.build_report(
        tracking,
        losses,
        "home",
        out,
        events=events,
        watermark=report.PROXY_WATERMARK,
        grid=TEST_GRID,
    )


# --------------------------------------------------------------------------- #
# A small PDF reader, so the text-layer tests need no external binary
# --------------------------------------------------------------------------- #

_STREAM = re.compile(rb"stream\r?\n(.*?)endstream", re.S)
_BFCHAR = re.compile(rb"beginbfchar(.*?)endbfchar", re.S)
_BFPAIR = re.compile(rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>")
_LITERAL = re.compile(rb"\((?:\\.|[^\\()])*\)", re.S)
_ESCAPE = re.compile(rb"\\(?:([0-7]{1,3})|(.))", re.S)
_SIMPLE = {b"n": b"\n", b"r": b"\r", b"t": b"\t", b"b": b"\b", b"f": b"\f"}


def _unescape(data: bytes) -> bytes:
    """Undo PDF string escaping, including octal character codes."""

    def replace(match: re.Match[bytes]) -> bytes:
        octal, char = match.group(1), match.group(2)
        if octal is not None:
            return bytes([int(octal, 8) & 0xFF])
        return _SIMPLE.get(char, char)

    return _ESCAPE.sub(replace, data)


def _decode_stream(blob: bytes) -> bytes:
    """Undo the stream filters reportlab writes: ASCII85, then Flate."""
    data = blob
    if b"~>" in data:
        try:
            data = base64.a85decode(data.split(b"~>")[0], adobe=False, ignorechars=b" \t\r\n")
        except ValueError:
            data = blob
    try:
        return zlib.decompress(data)
    except zlib.error:
        return data


def pdf_text(path: Path) -> str:
    """Extract the text layer of a PDF written by this module.

    Every embedded subset font carries a ``/ToUnicode`` CMap mapping the byte
    codes in the content stream back to characters. Decoding through it is what
    proves a glyph really is the character it was asked to be -- the whole
    point of the non-Latin-1 test, which a check on the *source* string would
    pass on a document full of box glyphs.

    Args:
        path: The PDF.

    Returns:
        Every text-showing operand, decoded through every font's CMap and
        joined. Not laid out and not deduplicated: this is for membership
        tests, not for reading.
    """
    streams = [_decode_stream(blob) for blob in _STREAM.findall(Path(path).read_bytes())]
    tables: list[dict[int, str]] = []
    for stream in streams:
        for block in _BFCHAR.findall(stream):
            table = {
                int(code, 16): bytes.fromhex(value.decode()).decode("utf-16-be", "ignore")
                for code, value in _BFPAIR.findall(block)
            }
            if table:
                tables.append(table)
    out: list[str] = []
    for stream in streams:
        if b"Tj" not in stream and b"TJ" not in stream:
            continue
        chunks = [_unescape(match.group(0)[1:-1]) for match in _LITERAL.finditer(stream)]
        for table in tables:
            out.append(
                "\n".join("".join(table.get(byte, "") for byte in chunk) for chunk in chunks)
            )
    return "\n".join(out)


def pdf_prose(path: Path) -> str:
    """:func:`pdf_text` with every run of whitespace collapsed to one space.

    A paragraph is drawn one line at a time, so a sentence that wraps arrives
    as several separate text operations. Collapsing whitespace lets a test
    assert on the sentence rather than on where the line happened to break --
    which is a layout detail and would make every prose test fragile.

    Args:
        path: The PDF.

    Returns:
        The flattened text.
    """
    return " ".join(pdf_text(path).split())


def page_count(path: Path) -> int:
    """Count pages by parsing the document's own page tree.

    Args:
        path: The PDF.

    Returns:
        The ``/Count`` on the page tree root, cross-checked against the number
        of ``/Type /Page`` objects. Zero when the two disagree, which fails
        every caller loudly rather than trusting one of them.

    Raises:
        AssertionError: If the two counts disagree.
    """
    raw = Path(path).read_bytes()
    counts = [int(value) for value in re.findall(rb"/Count\s+(\d+)", raw)]
    pages = len(re.findall(rb"/Type\s*/Page[^s]", raw))
    assert counts, "no page tree /Count in the document"
    assert max(counts) == pages, f"/Count {max(counts)} but {pages} page objects"
    return pages


# --------------------------------------------------------------------------- #
# The contract: six pages, and a file a reader can open
# --------------------------------------------------------------------------- #


def test_report_is_exactly_six_pages(built: Path) -> None:
    """The page count is fixed, and it is what the footer claims it is."""
    assert page_count(built) == report.N_PAGES == 6


def test_pdf_opens_and_page_tree_parses(built: Path) -> None:
    """The file is a well-formed PDF with a trailer and a resolvable root."""
    raw = built.read_bytes()
    assert raw.startswith(b"%PDF-")
    assert raw.rstrip().endswith(b"%%EOF")
    assert b"/Type /Catalog" in raw
    assert b"trailer" in raw
    assert b"/Root" in raw


def test_every_page_is_numbered_out_of_six(built: Path) -> None:
    """Each page's footer carries its own number, not just the first."""
    text = pdf_text(built)
    for number in range(1, report.N_PAGES + 1):
        assert f"page {number} of 6" in text


def test_source_is_credited_on_the_page(built: Path) -> None:
    """Metrica asks to be acknowledged; the footer does it."""
    assert report.DATA_CREDIT in pdf_text(built)


def test_epv_grid_is_labelled_as_borrowed_not_derived(built: Path) -> None:
    """The EPV surface is somebody else's, and the report says so."""
    text = pdf_prose(built)
    assert "Friends of Tracking reference grid" in text
    assert "not an estimate this tool derived" in text


def test_field_position_caveat_is_printed(built: Path) -> None:
    """Recovery is not comparable across phases, where a coach can see it."""
    text = pdf_prose(built)
    assert "Read every recovery figure inside its own phase" in text


# --------------------------------------------------------------------------- #
# Disclosure
# --------------------------------------------------------------------------- #


def test_watermarked_report_differs_from_the_plain_one(
    built: Path, built_proxy: Path
) -> None:
    """The two builds are not the same document."""
    assert built.read_bytes() != built_proxy.read_bytes()


def test_footer_disclosure_extracts_as_one_run(built_proxy: Path) -> None:
    """The disclosure a machine can find is the flat footer line.

    The diagonal watermark draws the same words, but as two separately
    positioned strings on a rotated matrix -- a real reader hands those back
    reordered. The footer draws the whole sentence in a single operation, so
    this contiguous match can only have come from the footer.
    """
    line = f"{report.PROXY_WATERMARK} — {report.PROXY_SUBTITLE}"
    assert line in pdf_text(built_proxy)


def test_plain_report_carries_no_disclosure(built: Path) -> None:
    """A report built from tracking must not claim to be a proxy."""
    assert report.PROXY_WATERMARK not in pdf_text(built)


def test_disclosure_is_in_the_pdf_metadata(built: Path, built_proxy: Path) -> None:
    """Third channel: the document's own subject field."""
    assert b"/Subject" in built_proxy.read_bytes()
    proxy_text = pdf_text(built_proxy)
    assert report.PROXY_SUBTITLE in proxy_text
    subjects = re.findall(rb"/Subject\s*\((.*?)\)", built_proxy.read_bytes(), re.S)
    assert subjects, "no /Subject in the watermarked document"
    assert any(report.PROXY_WATERMARK.encode("latin-1") in value for value in subjects)
    plain = re.findall(rb"/Subject\s*\((.*?)\)", built.read_bytes(), re.S)
    assert all(report.PROXY_WATERMARK.encode("latin-1") not in value for value in plain)


def test_watermark_is_drawn_after_the_flowables(built_proxy: Path) -> None:
    """The watermark must be on top of the panels, not under them.

    Drawn from ``onPage`` it renders first and a full-width heatmap covers it
    completely, while every text check in this file still passes. The only
    structural evidence is the order of the operations in the page's content
    stream: the watermark's rotation matrix has to come after the last image.
    """
    streams = [
        _decode_stream(blob) for blob in _STREAM.findall(built_proxy.read_bytes())
    ]
    checked = 0
    for stream in streams:
        if b"Do" not in stream or b"cm" not in stream:
            continue
        image_at = stream.rfind(b" Do")
        # The watermark is the only rotation on the page: every other transform
        # reportlab writes for a placed image is a pure scale-and-translate.
        rotations = [
            match.start()
            for match in re.finditer(
                rb"(-?\d*\.?\d+) (-?\d*\.?\d+) (-?\d*\.?\d+) (-?\d*\.?\d+) "
                rb"(-?\d*\.?\d+) (-?\d*\.?\d+) cm",
                stream,
            )
            if abs(float(match.group(2))) > 1e-6
        ]
        if not rotations:
            continue
        checked += 1
        assert max(rotations) > image_at, (
            "the watermark transform precedes the last image on the page, so it "
            "is drawn underneath it -- use onPageEnd, not onPage"
        )
    assert checked, "no page with both an image and a rotated watermark was found"


# --------------------------------------------------------------------------- #
# Fonts
# --------------------------------------------------------------------------- #


def test_registered_faces_are_dejavu_not_helvetica() -> None:
    """Helvetica silently mangles non-Latin-1 names; DejaVu does not."""
    regular, bold, italic = report.register_pdf_fonts()
    assert regular == "DejaVuSans"
    assert bold == "DejaVuSans-Bold"
    assert italic == "DejaVuSans-Oblique"


def test_registering_fonts_twice_is_harmless() -> None:
    """One report build per process is not the only case."""
    assert report.register_pdf_fonts() == report.register_pdf_fonts()


def test_non_latin1_player_names_survive_into_the_text_layer(tmp_path: Path) -> None:
    """The test that catches the Helvetica bug.

    Built with a squad whose names carry a c-acute, a g-breve, a slashed L and
    an s-cedilla, then read back out of the PDF's own text layer through the
    embedded font's ToUnicode map. With reportlab's default face every one of
    these comes back as a box, and nothing raises on the way there.
    """
    names = tuple(f"{name}" for name in AWKWARD_NAMES) + tuple(
        f"Player {index}" for index in range(8)
    )
    tracking = make_tracking(n_players=6, names=names)
    losses = make_losses(tracking, n=6)
    path = report.build_report(
        tracking, losses, "home", tmp_path / "unicode.pdf", grid=TEST_GRID
    )
    text = pdf_text(path)
    for name in AWKWARD_NAMES:
        assert name in text, f"{name!r} did not survive into the text layer"
        assert "�" not in text


# --------------------------------------------------------------------------- #
# Degeneracy
# --------------------------------------------------------------------------- #


def test_minimal_match_of_one_loss_and_two_players(tmp_path: Path) -> None:
    """One loss, one player a side. The template still produces six pages."""
    tracking = make_tracking(n_players=1, n_frames=300)
    losses = make_losses(tracking, n=1)
    path = report.build_report(
        tracking,
        losses,
        "home",
        tmp_path / "minimal.pdf",
        events=make_events(losses),
        grid=TEST_GRID,
    )
    assert page_count(path) == 6
    text = pdf_prose(path)
    assert "No further scorable moments" in text
    assert "there is no fifth moment to enlarge" in text
    assert "Nothing has been substituted for it" in text


def test_a_phase_with_no_losses_does_not_break_the_table(tmp_path: Path) -> None:
    """Every phase keeps a row even when nothing happened in it."""
    tracking = make_tracking(n_players=4, n_frames=400)
    losses = make_losses(tracking, n=6, phases=("middle",))
    path = report.build_report(
        tracking, losses, "home", tmp_path / "onephase.pdf", grid=TEST_GRID
    )
    assert page_count(path) == 6
    text = pdf_prose(path)
    for phase in ("build up", "middle", "final third"):
        assert phase in text


def test_unscorable_losses_are_kept_and_named(tmp_path: Path) -> None:
    """A loss with no located ball keeps its row and is reported, not dropped."""
    tracking = make_tracking(n_players=4, n_frames=400)
    losses = make_losses(tracking, n=6, unlocated=2)
    path = report.build_report(
        tracking, losses, "home", tmp_path / "skipped.pdf", grid=TEST_GRID
    )
    text = pdf_prose(path)
    assert page_count(path) == 6
    assert "could not be scored" in text
    assert "no_loss_location" in text


def test_a_loss_whose_ball_is_untracked_still_renders(tmp_path: Path) -> None:
    """The ball vanishes for whole stretches of real tracking data."""
    tracking = make_tracking(n_players=4, n_frames=500, ball_gap=(1, 500))
    losses = make_losses(tracking, n=5)
    moments = report.render_moments(tracking, losses, "home", grid=TEST_GRID)
    assert moments
    assert all(frame.ball is None for moment in moments for frame in moment.frames)
    path = report.build_report(
        tracking,
        losses,
        "home",
        tmp_path / "noball.pdf",
        moments=moments,
        grid=TEST_GRID,
    )
    assert page_count(path) == 6


def test_an_oversized_squad_is_capped_and_says_so(tmp_path: Path) -> None:
    """The player table is capped rather than allowed to spill onto page 7.

    Thirty-four a side is not a real squad, but a table that would need two
    pages is a real failure mode -- an injury-time-heavy match with a long
    substitute list gets there, and so does any source that lists unused subs.
    The report prints what fits, names the count it left out, and stays six
    pages.
    """
    squad = report.MAX_PLAYER_ROWS + 4
    tracking = make_tracking(n_players=squad, n_frames=400)
    losses = make_losses(tracking, n=6)
    path = report.build_report(
        tracking, losses, "home", tmp_path / "squad.pdf", grid=TEST_GRID
    )
    assert page_count(path) == 6
    text = pdf_prose(path)
    assert f"of {squad} who played" in text
    assert f"the {report.MAX_PLAYER_ROWS} players most often near a turnover" in text


def test_a_missing_player_name_does_not_raise(tmp_path: Path) -> None:
    """Names arrive missing; a caption prints what it has and no more."""
    tracking = make_tracking(n_players=4, n_frames=400)
    losses = make_losses(tracking, n=5)
    losses.loc[0, "losing_player"] = None
    path = report.build_report(
        tracking, losses, "home", tmp_path / "noname.pdf", grid=TEST_GRID
    )
    assert page_count(path) == 6


def test_no_scored_losses_at_all(tmp_path: Path) -> None:
    """A match where nothing could be scored still produces the document."""
    tracking = make_tracking(n_players=4, n_frames=400)
    losses = make_losses(tracking, n=4, unlocated=4)
    path = report.build_report(
        tracking, losses, "home", tmp_path / "nothing.pdf", grid=TEST_GRID
    )
    assert page_count(path) == 6
    assert "No further scorable moments" in pdf_prose(path)


def test_team_must_be_home_or_away(match: tuple[pd.DataFrame, ...], tmp_path: Path) -> None:
    """A typo in the team name fails immediately, not three pages in."""
    tracking, events, losses = match
    with pytest.raises(ValueError, match="home"):
        report.build_report(tracking, losses, "Rovers", tmp_path / "bad.pdf")


# --------------------------------------------------------------------------- #
# Moments
# --------------------------------------------------------------------------- #


def test_moments_are_ranked_by_the_column_they_claim(
    match: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """"Most costly" means largest EPV swing, in that order."""
    tracking, events, losses = match
    moments = report.render_moments(
        tracking, losses, "home", rank="costly", grid=TEST_GRID, events=events
    )
    swings = [moment.loss["epv_conceded"] for moment in moments]
    assert swings == sorted(swings, reverse=True)

    best = report.render_moments(
        tracking, losses, "home", rank="recovered", grid=TEST_GRID, events=events
    )
    shares = [moment.loss["recovery"] for moment in best]
    assert shares == sorted(shares, reverse=True)


def test_moments_only_come_from_the_subject_team(
    match: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """A report for one team does not print the other team's turnovers."""
    tracking, events, losses = match
    for side in ("home", "away"):
        moments = report.render_moments(tracking, losses, side, grid=TEST_GRID)
        assert {moment.losing_team for moment in moments} == {side}


def test_moment_surfaces_are_on_the_requested_grid(
    match: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """The printed panels are recomputed, not upsampled from the demo grid."""
    tracking, events, losses = match
    moments = report.render_moments(tracking, losses, "home", grid=TEST_GRID)
    for moment in moments:
        for frame in moment.frames:
            assert frame.control.shape == TEST_GRID.shape
            assert np.isfinite(frame.control).all()
            assert frame.control.min() >= 0.0
            assert frame.control.max() <= 1.0


def test_moment_offsets_cover_the_window(
    match: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """The strip needs t0, the middle and the end; the panel needs the middle."""
    tracking, events, losses = match
    moments = report.render_moments(tracking, losses, "home", grid=TEST_GRID)
    offsets = [frame.offset for frame in moments[0].frames]
    assert offsets == sorted(offsets)
    assert set(report.STRIP_OFFSETS) <= set(offsets)
    assert report.PANEL_OFFSET in offsets
    assert moments[0].at(report.PANEL_OFFSET) is not None


def test_an_unscored_loss_is_never_printed_as_a_moment() -> None:
    """A loss with no surface behind it is not given a picture anyway."""
    tracking = make_tracking(n_players=4, n_frames=400)
    losses = make_losses(tracking, n=4, unlocated=4)
    assert report.render_moments(tracking, losses, "home", grid=TEST_GRID) == []


def test_recovering_player_comes_from_the_event_stream(
    match: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """The caption names who won it back, and only when somebody did."""
    tracking, events, losses = match
    names = report.recovering_players(losses, events)
    for row in losses.to_dict("records"):
        if row["regained_within_5s"]:
            assert names[row["loss_id"]] == f"winner-{row['loss_id']}"
        else:
            assert row["loss_id"] not in names


def test_recovering_players_without_events_is_empty(
    match: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """Events are optional; the caption simply says less without them."""
    _, _, losses = match
    assert report.recovering_players(losses, None) == {}


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.5, "+0.50"), (-0.5, "-0.50"), (0.0, "0.00"), (-0.0, "0.00"), (np.nan, "—")],
)
def test_signed_never_prints_a_signed_zero(value: float, expected: str) -> None:
    """"+0.00" reads as a rounded-down positive. It is not one."""
    assert report._signed(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"), [(0.0, "0'00\""), (61.0, "1'01\""), (3599.0, "59'59\""), (None, "—")]
)
def test_clock_formats_match_time(value: float | None, expected: str) -> None:
    """Match time is minutes and seconds, or a dash when it is unknown."""
    assert report._clock(value) == expected


def test_missing_numbers_print_a_dash_not_a_zero() -> None:
    """A missing measurement and a measured zero are different claims."""
    assert report._num(np.nan) == "—"
    assert report._num(None) == "—"
    assert report._num(0.0) == "0.00"
    assert report._pct(np.nan) == "—"
    assert report._int(np.nan) == "—"


# --------------------------------------------------------------------------- #
# Cost
# --------------------------------------------------------------------------- #


def test_build_time_and_file_size_are_sane(
    match: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], tmp_path: Path
) -> None:
    """The whole document, from prepared inputs, is seconds and megabytes.

    The bounds are deliberately loose -- this is a smoke alarm for a renderer
    that has started rasterising something enormous or looping over every loss,
    not a benchmark. The measured numbers on a real Metrica match are far
    inside them.
    """
    tracking, events, losses = match
    start = time.perf_counter()
    path = report.build_report(
        tracking, losses, "home", tmp_path / "timed.pdf", events=events, grid=TEST_GRID
    )
    elapsed = time.perf_counter() - start
    size_mib = path.stat().st_size / 1024 / 1024
    assert elapsed < 60.0, f"the report took {elapsed:.1f} s to build"
    assert 0.05 < size_mib < 6.0, f"the report is {size_mib:.2f} MiB"
