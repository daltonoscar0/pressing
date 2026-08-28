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
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import pytest

from pressing import baseline, plots, report
from pressing.counterpress import LOSS_SCORE_COLUMNS, CounterPressParams, per_player
from pressing.phases import label_loss_phases
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
def busy() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A match with enough losses per phase to run a between-player test.

    The permutation test wants at least three players carrying at least five
    losses each inside a single phase column. Nine losses over twelve players
    never reaches that, which is itself a fact the report has to handle -- see
    the "too thin to test" case -- but it leaves the interesting branch
    untested, so this fixture is deliberately busy.
    """
    tracking = make_tracking(n_players=6, n_frames=2400)
    losses = make_losses(tracking, n=90)
    # make_losses scatters turnovers uniformly, which on this synthetic pitch
    # leaves most of them out of every player's 15 m radius -- 26 credits over
    # 90 losses. Putting each turnover on top of a real player, cycling through
    # the squad, is what makes the per-player table dense enough to test.
    positions = tracking[tracking["team"].isin(("home", "away"))].set_index(
        ["frame", "player_id"]
    )
    squads = {
        side: sorted(set(tracking.loc[tracking["team"] == side, "player_id"]))
        for side in ("home", "away")
    }
    x: list[float] = []
    y: list[float] = []
    for index, row in enumerate(losses.to_dict("records")):
        squad = squads[str(row["losing_team"])]
        spot = positions.loc[(int(row["frame"]), squad[index % len(squad)])]
        x.append(float(spot["x"]))
        y.append(float(spot["y"]))
    losses["x"] = x
    losses["y"] = y
    attrs = dict(losses.attrs)
    losses = label_loss_phases(losses, tracking.attrs["attack_direction"])
    losses.attrs.update(attrs)
    return tracking, losses


@pytest.fixture(scope="module")
def built_busy(
    busy: tuple[pd.DataFrame, pd.DataFrame], tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """The report for :func:`busy`, built once."""
    tracking, losses = busy
    return report.build_report(
        tracking,
        losses,
        "home",
        tmp_path_factory.mktemp("busy") / "busy.pdf",
        grid=TEST_GRID,
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
    return "\n".join(_page_texts(path))


def _page_texts(path: Path) -> list[str]:
    """Decoded text, one entry per content stream, in file order.

    Args:
        path: The PDF.

    Returns:
        The text of each stream that draws any, in the order reportlab wrote
        them -- which is page order.
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
        out.append(
            "\n".join(
                "\n".join("".join(table.get(byte, "") for byte in chunk) for chunk in chunks)
                for table in tables
            )
        )
    return out


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


def pdf_page_prose(path: Path) -> list[str]:
    """:func:`pdf_prose`, split one string per page, in page order.

    Several of the claims below are about *where* a sentence is printed, not
    merely that it exists somewhere in the document -- the whole finding that
    forced the page-6 warning was that the field-position caveat was on pages 1
    and 2 and nowhere near the table it applies to. A test that greps the whole
    PDF cannot tell those two states apart.

    Reportlab writes one content stream per page, in order, and the running
    footer stamps "page N of 6" into each. Both are asserted here, so a future
    reportlab that batches pages differently fails this helper loudly instead
    of silently making every caller vacuous.

    Args:
        path: The PDF.

    Returns:
        One flattened string per page.

    Raises:
        AssertionError: If the page streams cannot be told apart by footer.
    """
    pages = [
        " ".join(text.split())
        for text in _page_texts(path)
        if text.strip()
    ]
    numbered = [page for page in pages if re.search(r"page \d+ of \d+", page)]
    assert numbered, "no page carries a footer, so the split cannot be trusted"
    for index, page in enumerate(numbered, start=1):
        assert f"page {index} of " in page, f"stream {index} does not carry its own footer"
    return numbered


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
    """Recovery is not comparable across phases, where a coach can see it.

    The caveat now lives inside the cover's one boxed warning rather than in
    the fourth of four competing caveat registers, so this asserts on the page
    it is on as well as on the words.
    """
    cover = pdf_page_prose(built)[0]
    assert "Read every recovery figure inside its own phase" in cover
    assert report.PHASE_CAVEAT in cover
    assert report.COVER_WARNING_HEADING in cover


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
# What the report has to admit about itself
#
# Two measured findings drove every test in this section. On the bundled
# Metrica match, over the 27 players the report prints a Recovery for, a cubic
# in the attacking fraction of the turnover -- position and nothing else --
# reproduces the printed column at r = +0.9771 (R2 = 0.9547, Spearman +0.9402),
# and the player's own mean turnover position alone reaches R2 = 0.9583. At
# loss level a straight line in x_norm gives R2 = 0.8212 over the 255 scored
# losses. And the remedy the report used to prescribe does not work: a
# 5,000-shuffle permutation test on the three per-phase player columns,
# shuffled within a squad, gives p = 0.97 (build-up), 0.86 (middle) and 0.16
# (final third), with the build-up spread SMALLER than chance. Every command
# behind those numbers is in the module docstring of ``pressing.baseline``.
#
# So the tests here are not "a caveat exists". They are: the warning is on the
# page a coach photocopies, its numbers are measured on the match in hand
# rather than copied from this comment, and a cell with too little behind it
# does not print a number at all.
# --------------------------------------------------------------------------- #


def measure(
    tracking: pd.DataFrame, losses: pd.DataFrame, team: str = "home"
) -> tuple[report.Disclosure, pd.DataFrame]:
    """Run the report's own self-measurement over a synthetic match.

    Args:
        tracking: Canonical tracking frame.
        losses: Scored losses, with or without the baseline columns.
        team: Subject team.

    Returns:
        ``(disclosure, players)`` exactly as :func:`report.build_report` builds
        them, so a test asserting on the numbers is asserting on the numbers
        the pages print.
    """
    scored = report.with_expected_recovery(losses, tracking)
    players = per_player(scored, tracking)
    other = "away" if team == "home" else "home"
    printed = pd.concat(
        [report._player_rows(players, team), report._player_rows(players, other)]
    )
    disclosure = report.measure_disclosure(
        scored, players, tracking, team, printed=printed
    )
    return disclosure, players


def test_page_six_opens_with_the_field_position_warning(built: Path) -> None:
    """The warning is on page 6 itself, not only on the cover.

    This is the finding in its purest form: page 6 is a per-player ranking, it
    is the page that gets photocopied and pinned up, and it used to carry no
    field-position warning at all.
    """
    pages = pdf_page_prose(built)
    assert report.RANK_WARNING_HEADING in pages[5]
    assert "map of where each player plays" in pages[5]


def test_page_six_prints_the_measured_position_correlation(
    match: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], built: Path
) -> None:
    """The warning carries a number, and it is the number this match measures."""
    tracking, _, losses = match
    disclosure, _ = measure(tracking, losses)
    assert np.isfinite(disclosure.position_r)
    page = pdf_page_prose(built)[5]
    assert f"r = {report._signed(disclosure.position_r, 2)}" in page
    assert f"{disclosure.position_r ** 2 * 100.0:.0f}% of everything separating" in page
    assert f"over {disclosure.position_rows} players" in page


def test_the_position_correlation_is_measured_not_a_constant(
    tmp_path: Path,
) -> None:
    """Two matches, two different printed numbers.

    A warning that prints a figure an audit once measured on somebody else's
    match is worse than no warning: it is a false number in the place a reader
    trusts most. This drives the same code down two matches that differ only in
    how tightly recovery tracks the turnover's position, and requires the
    printed correlations to differ.
    """
    tracking = make_tracking(n_players=6, n_frames=900)
    geography = make_losses(tracking, n=24)
    # Recovery *is* position: the strongest possible version of the finding.
    geography["recovery"] = geography["x_norm"].to_numpy(dtype=float) * 0.8 + 0.1
    geography["recovery_area"] = geography["recovery"]

    noise = geography.copy()
    noise.attrs.update(geography.attrs)
    # The same values, dealt out in an order that has nothing to do with x.
    noise["recovery"] = np.random.default_rng(0).permutation(
        geography["recovery"].to_numpy(dtype=float)
    )
    noise["recovery_area"] = noise["recovery"]

    tight, _ = measure(tracking, geography)
    loose, _ = measure(tracking, noise)
    assert tight.position_r > loose.position_r
    printed = {
        report._signed(tight.position_r, 2),
        report._signed(loose.position_r, 2),
    }
    assert len(printed) == 2, f"both matches printed {printed}"


def test_a_thin_player_is_suppressed_rather_than_printed_as_a_number(
    match: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], tmp_path: Path
) -> None:
    """Below the minimum count the cell says so instead of showing a mean.

    The threshold is derived in ``pressing.baseline``: eleven is the count at
    which a 95% bootstrap interval on a player's mean first becomes narrower
    than the whole range of player means, i.e. the least a number must manage
    before it can order anybody against anybody.
    """
    tracking, events, losses = match
    disclosure, players = measure(tracking, losses)
    thin = players[~players["reportable"].to_numpy(dtype=bool)]
    assert not thin.empty, "the fixture must contain at least one thin player"
    assert (thin["losses_nearby"] < baseline.MIN_LOSSES_FOR_PLAYER).all()

    path = report.build_report(
        tracking,
        losses,
        "home",
        tmp_path / "suppressed.pdf",
        events=events,
        grid=TEST_GRID,
    )
    page = pdf_page_prose(path)[5]
    marker = report.SUPPRESSED.format(minimum=baseline.MIN_LOSSES_FOR_PLAYER)
    assert marker in page
    for value in thin["mean_recovery"].dropna():
        assert report._num(value, 2) not in page, "a suppressed mean reached the page"


def test_a_suppressed_row_suppresses_every_derived_cell(
    match: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
) -> None:
    """One decision, not four.

    Recovery, the residual, its interval and the EPV swing are all per-loss
    means over the same losses. A page that hides one of them at n = 7 and
    prints another says two different things about one sample -- and makes the
    warning's own "no cell prints unless 11 losses stand behind it" false on
    the page that prints it.
    """
    tracking, _, losses = match
    disclosure, players = measure(tracking, losses)
    fonts = report.register_pdf_fonts()
    styles = report._build_styles(*fonts)
    table, _, _ = report._player_table(players, "home", {}, styles, fonts, disclosure)
    marker = report.SUPPRESSED.format(minimum=baseline.MIN_LOSSES_FOR_PLAYER)
    rows = {
        str(row[1].text if hasattr(row[1], "text") else row[1]): row
        for row in table._cellvalues[1:]
    }
    shown = report._player_rows(players, "home")
    for record in shown.to_dict("records"):
        if bool(record.get("reportable", False)):
            continue
        row = rows[str(record["player_id"])]
        # Recovery, vs expected and EPV swing, with an empty interval between.
        assert row[4] == marker
        assert row[5] == marker
        assert row[6] == ""
        assert row[7] == marker


def test_phase_headings_carry_their_own_permutation_p_value(
    busy: tuple[pd.DataFrame, pd.DataFrame], built_busy: Path
) -> None:
    """The question "does this column separate anybody?" is answered on it."""
    tracking, losses = busy
    disclosure, _ = measure(tracking, losses)
    assert any(np.isfinite(value) for value in disclosure.phase_p.values())
    page = pdf_page_prose(built_busy)[5]
    for phase, p_value in disclosure.phase_p.items():
        assert report._p_header(p_value) in page, f"{phase} heading lost its p-value"
    assert f"{report.PERMUTATION_SHUFFLES:,}-shuffle permutation test" in page


def test_a_match_too_thin_to_test_prints_no_p_value_rather_than_a_fake_one(
    match: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], built: Path
) -> None:
    """Nine losses cannot support a between-player test, and the page says nothing.

    The permutation test needs at least three players with at least five losses
    each in the column. A page that printed "p = 1.00" from two players would
    be inventing a result; a page that printed nothing at all would leave the
    reader assuming one was found. It prints "p n/a" and omits the paragraph.
    """
    tracking, _, losses = match
    disclosure, _ = measure(tracking, losses)
    assert all(not np.isfinite(value) for value in disclosure.phase_p.values())
    page = pdf_page_prose(built)[5]
    assert "p n/a" in page
    assert "None of them separates players by more than chance" not in page
    assert "By the same test they give" not in page


def test_the_warning_states_the_phase_columns_carry_no_signal(
    busy: tuple[pd.DataFrame, pd.DataFrame], built_busy: Path
) -> None:
    """When no phase column clears the test, the page says so in words.

    This is the second critical finding put on the paper: the report tells the
    reader to compare within a phase, and on the bundled match a 5,000-shuffle
    test on those three columns returns p = 0.97, 0.86 and 0.16 -- with the
    build-up spread smaller than chance. The page has to say so where the
    columns are.
    """
    tracking, losses = busy
    disclosure, _ = measure(tracking, losses)
    measured = [value for value in disclosure.phase_p.values() if np.isfinite(value)]
    assert measured, "the busy fixture must produce at least one usable p-value"
    page = pdf_page_prose(built_busy)[5]
    assert "the comparison the rest of this report tells you to make" in page
    for phase, value in disclosure.phase_p.items():
        if np.isfinite(value):
            assert f"p = {value:.2f} in the {phase.replace('_', ' ')}" in page
    if all(value >= 0.05 for value in measured):
        assert "None of them separates players by more than chance" in page
    else:
        assert "Read them one column at a time" in page


def test_an_empty_phase_column_is_explained_rather_than_left_blank(
    busy: tuple[pd.DataFrame, pd.DataFrame], built_busy: Path
) -> None:
    """A column with nothing in it is a fact about the match, not a rendering bug.

    On the bundled match no player on either squad is near eleven build-up
    turnovers, so that column prints nothing for anybody. A reader scanning a
    blank column has no way to tell that from a broken one.
    """
    tracking, losses = busy
    disclosure, _ = measure(tracking, losses)
    blank = [
        phase for phase in disclosure.phase_printed if disclosure.phase_printed[phase] == 0
    ]
    assert blank, "the busy fixture must leave at least one phase column empty"
    page = pdf_page_prose(built_busy)[5]
    for phase in blank:
        assert phase.replace("_", " ") in page
    assert "empty for a reason, not by mistake" in page


def test_a_table_with_nothing_suppressed_says_so_too(
    busy: tuple[pd.DataFrame, pd.DataFrame], built_busy: Path
) -> None:
    """The suppression sentence describes this table, whichever way it fell."""
    tracking, losses = busy
    disclosure, _ = measure(tracking, losses)
    page = pdf_page_prose(built_busy)[5]
    if disclosure.suppressed_rows:
        assert f"silences every number on {disclosure.suppressed_rows} of these" in page
    else:
        assert "Every row here clears it for Recovery" in page


def test_the_cover_states_the_regain_correlation(
    match: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], built: Path
) -> None:
    """The two stacked headline rows are not one finding said twice.

    Measured on the bundled match, r(recovery, regained_within_5s) = +0.071
    over its 255 scored losses. The cover prints recovery immediately above
    regain, so the cover is where that has to be said.
    """
    tracking, _, losses = match
    disclosure, _ = measure(tracking, losses)
    assert np.isfinite(disclosure.regain_r)
    cover = pdf_page_prose(built)[0]
    assert "Almost unrelated" in cover
    assert f"r = {report._signed(disclosure.regain_r, 2)}" in cover
    assert f"{disclosure.n_scored} scored losses" in cover


def test_the_regain_note_is_left_out_when_it_cannot_be_measured() -> None:
    """An unmeasurable correlation removes the sentence; it never guesses."""
    fonts = report.register_pdf_fonts()
    styles = report._build_styles(*fonts)
    assert report._regain_note(None, styles) is None
    assert report._regain_note(report.Disclosure(), styles) is None
    said = report._regain_note(report.Disclosure(regain_r=0.07, n_scored=255), styles)
    assert said is not None
    assert "+0.07" in said.text


def test_the_cover_marks_which_direction_of_each_row_is_good(built: Path) -> None:
    """A row where higher is worse must not be styled like one where it is better."""
    cover = pdf_page_prose(built)[0]
    assert "Better is" in cover
    assert "Median time to regain" in cover
    assert report.POLARITY_LEGEND.format(shuffles=report.COVER_SHUFFLES) in cover
    # Both directions occur in the table, so neither label is dead code.
    assert report._POLARITY["recovery"] == "higher"
    assert report._POLARITY["time_to_regain"] == "lower"
    assert report._POLARITY["epv"] == "lower"


def test_the_cover_shows_the_phase_mix_check_it_demands(
    match: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], built: Path
) -> None:
    """A pooled difference printed above a caveat forbidding it shows its working."""
    tracking, _, losses = match
    disclosure, _ = measure(tracking, losses)
    cover = pdf_page_prose(built)[0]
    assert "On the match's own phase mix it reads" in cover
    assert report._signed(disclosure.standardised_difference, 3) in cover
    assert report._signed(disclosure.raw_difference, 3) in cover


# --------------------------------------------------------------------------- #
# Head-to-head gaps: pages 1 and 2 get a test, not a disclaimer
# --------------------------------------------------------------------------- #


def test_a_gap_that_is_only_noise_is_not_called_a_gap() -> None:
    """Two squads drawn from one distribution must not come out separated."""
    rng = np.random.default_rng(11)
    values = rng.normal(0.5, 0.2, 400)
    teams = np.where(np.arange(400) % 2 == 0, "home", "away")
    gap = report.permutation_team_gap(
        values,
        teams,
        subject="home",
        opponent="away",
        n_shuffles=2000,
        rng=np.random.default_rng(3),
    )
    assert gap.tested
    assert not gap.separates
    assert gap.p_value > 0.2
    assert gap.n_subject == 200
    assert gap.n_opponent == 200


def test_a_real_gap_is_found() -> None:
    """A planted half-sigma offset has to clear the same null the noise failed."""
    rng = np.random.default_rng(12)
    teams = np.where(np.arange(400) % 2 == 0, "home", "away")
    values = rng.normal(0.5, 0.2, 400) + np.where(teams == "home", 0.1, 0.0)
    gap = report.permutation_team_gap(
        values,
        teams,
        subject="home",
        opponent="away",
        n_shuffles=2000,
        rng=np.random.default_rng(4),
    )
    assert gap.separates
    assert gap.p_value < 0.01
    assert gap.difference > 0.05


def test_the_stratified_design_removes_a_gap_that_is_only_phase_mix() -> None:
    """The point of stratifying, stated as a test.

    Two squads that press identically inside every phase, but lose the ball in
    different places, differ substantially when the phases are pooled. The
    unstratified null -- the one behind the cover's Difference column -- is
    entitled to call that a difference, because it is one. The stratified null
    must not, because within a phase there is nothing there, and the cover
    prints the stratified p beside the standardised number for exactly this
    reason.
    """
    phases = np.array(["build_up"] * 100 + ["final_third"] * 100)
    # Home loses it high, away loses it low; the phase means are identical.
    teams = np.where(
        phases == "build_up",
        np.where(np.arange(200) % 5 == 0, "home", "away"),
        np.where(np.arange(200) % 5 == 0, "away", "home"),
    )
    rng = np.random.default_rng(13)
    values = np.where(phases == "build_up", 0.25, 0.70) + rng.normal(0, 0.02, 200)

    pooled = report.permutation_team_gap(
        values,
        teams,
        subject="home",
        opponent="away",
        n_shuffles=2000,
        rng=np.random.default_rng(5),
    )
    conditional = report.permutation_team_gap(
        values,
        teams,
        subject="home",
        opponent="away",
        strata=phases,
        n_shuffles=2000,
        rng=np.random.default_rng(5),
    )
    assert pooled.separates, "pooling two different phases must show the phase mix"
    assert abs(pooled.difference) > 0.2
    assert conditional.stratified and not pooled.stratified
    assert not conditional.separates, "inside a phase the squads are identical"
    assert abs(conditional.difference) < 0.02


def test_a_gap_with_too_few_losses_is_not_tested_rather_than_guessed() -> None:
    """A mean of two against a mean of one gets no p-value."""
    gap = report.permutation_team_gap(
        np.array([0.4, 0.6, 0.5]),
        np.array(["home", "home", "away"]),
        subject="home",
        opponent="away",
        n_shuffles=500,
        rng=np.random.default_rng(6),
    )
    assert not gap.tested
    assert np.isfinite(gap.difference)
    assert report._gap_cell(gap) == report.GAP_UNTESTED


@pytest.mark.parametrize("bad", [{"n_shuffles": 0}, {"statistic": "mode"}])
def test_permutation_team_gap_rejects_a_design_it_cannot_run(bad: dict[str, Any]) -> None:
    """A misspelled statistic must raise, not silently become the mean."""
    with pytest.raises(ValueError):
        report.permutation_team_gap(
            np.arange(10, dtype=float),
            np.array(["home"] * 5 + ["away"] * 5),
            subject="home",
            opponent="away",
            **bad,
        )


def test_every_cover_row_carries_a_p_value_or_says_it_was_not_tested(
    match: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], built: Path
) -> None:
    """The finding that forced this round: no row may go out unqualified.

    Page 1 used to print a bolded head-to-head whose only statement about
    uncertainty was the confession that no gap in it had been tested against
    chance, while the machinery to test it sat three modules away. Every row
    now carries either a measured p-value or the words "not tested", and this
    asserts on the rendered page rather than on the dictionary behind it.
    """
    tracking, _, losses = match
    disclosure, _ = measure(tracking, losses)
    cover = pdf_page_prose(built)[0]
    assert "vs chance" in cover
    assert report.GAP_UNTESTED in cover, "the untestable count rows must say so"
    tested = [gap for gap in disclosure.cover_gaps.values() if gap.tested]
    assert tested, "the fixture must produce at least one testable row"
    for key in ("recovery", "regain"):
        gap = disclosure.cover_gaps[key]
        assert gap.tested, f"the {key} row is the whole point and must be tested"
        assert report._gap_cell(gap) in cover


#: Column index of the cover table's Difference cell, which is the one styled
#: by the test result rather than by the metric.
DIFFERENCE_COLUMN = 3


def kpi_difference_style(
    losses: pd.DataFrame,
    tracking: pd.DataFrame,
    disclosure: report.Disclosure,
    gaps: dict[str, report.TeamGap],
) -> tuple[str, Any]:
    """Face and colour of the recovery row's Difference cell, as drawn.

    reportlab resolves ``FONT`` and ``TEXTCOLOR`` into per-cell styles rather
    than keeping the command list, and that resolved style is what the page
    actually draws, so it is what the test reads.

    Args:
        losses: Scored losses.
        tracking: Canonical tracking frame.
        disclosure: The measured disclosure to base the table on.
        gaps: Gap results to substitute for the measured ones.

    Returns:
        ``(fontname, colour)`` of the recovery row's Difference cell.
    """
    fonts = report.register_pdf_fonts()
    table = report._kpi_table(
        report.with_expected_recovery(losses, tracking),
        "home",
        None,
        report._build_styles(*fonts),
        fonts,
        replace(disclosure, cover_gaps=gaps),
    )
    cell = table._cellStyles[1][DIFFERENCE_COLUMN]
    return cell.fontname, cell.color


def test_a_gap_consistent_with_chance_is_rendered_differently(
    match: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """Grey and unbolded for noise; black and bold for a finding.

    A text-layer check cannot see a colour, and the whole point of the styling
    rule is that a gap the match cannot separate from chance must not look like
    a result. So this drives one real table twice, changing nothing but the
    p-value, and reads the resolved cell style off the flowable.
    """
    tracking, _, losses = match
    disclosure, _ = measure(tracking, losses)
    assert "recovery" in disclosure.cover_gaps

    def at(p_value: float) -> tuple[str, Any]:
        """The Difference cell's style when every gap lands at ``p_value``."""
        return kpi_difference_style(
            losses,
            tracking,
            disclosure,
            {
                key: replace(gap, p_value=p_value, difference=0.05, n_shuffles=100)
                for key, gap in disclosure.cover_gaps.items()
            },
        )

    noise_font, noise_colour = at(0.90)
    finding_font, finding_colour = at(0.001)
    assert noise_colour is report._INK_MUTED, "a gap consistent with chance must be greyed"
    assert noise_font != finding_font, "the two must not be drawn in the same face"
    assert finding_font == report.FONT_BOLD
    assert finding_colour is not report._INK_MUTED


def test_the_phase_table_tests_every_row(
    match: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], built: Path
) -> None:
    """Page 2's phase table prints a gap and a p-value for each phase."""
    tracking, _, losses = match
    disclosure, _ = measure(tracking, losses)
    page = pdf_page_prose(built)[1]
    assert disclosure.phase_gaps, "the fixture must produce at least one phase gap"
    for phase, gap in disclosure.phase_gaps.items():
        assert report._signed(gap.difference, 3) in page, f"{phase} lost its gap"
        assert report._gap_cell(gap) in page, f"{phase} lost its p-value"


def test_the_one_phase_that_clears_its_null_is_distinguishable(
    tmp_path: Path,
) -> None:
    """The finding this round exists to rescue, driven both ways.

    On the bundled match exactly one comparison in the whole document survives
    its own null -- the middle-third recovery gap -- and it was printed with the
    same weight as a final-third row at p = 0.85. Here a match is built where
    one phase carries a large planted gap and the others carry none, and the
    page is required to name that phase and no other.
    """
    tracking = make_tracking(n_players=6, n_frames=2400)
    losses = make_losses(tracking, n=90)
    phase = losses["phase"].to_numpy(dtype=object)
    side = losses["losing_team"].to_numpy(dtype=object)
    rng = np.random.default_rng(21)
    base = {"build_up": 0.25, "middle": 0.50, "final_third": 0.70}
    recovery = np.array([base.get(str(name), 0.5) for name in phase])
    recovery = recovery + rng.normal(0, 0.02, len(losses))
    # Only the middle third separates the squads, and it separates them hugely.
    recovery += np.where((phase == "middle") & (side == "home"), 0.20, 0.0)
    losses = losses.assign(recovery=recovery, recovery_area=recovery)

    path = report.build_report(
        tracking, losses, "home", tmp_path / "planted.pdf", grid=TEST_GRID
    )
    disclosure, _ = measure(tracking, losses)
    clear = [name for name, gap in disclosure.phase_gaps.items() if gap.separates]
    assert clear == ["middle"], f"only the middle third was planted, got {clear}"
    page = pdf_page_prose(path)[1]
    assert "One row here is not like the others" in page
    assert "in the middle, Home by" in page
    assert "after correcting for the 3 phases tested" in page
    for quiet in ("build up", "final third"):
        assert f"in the {quiet}, " not in page


def test_a_match_where_nothing_separates_says_nothing_separated(
    match: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], built: Path
) -> None:
    """The verdict line has to work in the direction the data actually goes."""
    tracking, _, losses = match
    disclosure, _ = measure(tracking, losses)
    tested = [gap for gap in disclosure.phase_gaps.values() if gap.tested]
    page = pdf_page_prose(built)[1]
    if tested and not any(gap.separates for gap in tested):
        assert "is bigger than reshuffling" in page
        assert "the two squads press alike" in page


def test_page_six_says_the_metric_is_not_a_regain(
    match: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], built: Path
) -> None:
    """Page 6 leaves the building alone, so it has to carry this itself.

    The column is headed "Recovery". The r(recovery, regain) note lived on the
    cover only, so a reader handed page 6 on its own could still read a high
    number as a claim that the ball came back.
    """
    tracking, _, losses = match
    disclosure, _ = measure(tracking, losses)
    page = pdf_page_prose(built)[5]
    assert np.isfinite(disclosure.regain_r)
    assert "It is also not a regain" in page
    assert f"r = {report._signed(disclosure.regain_r, 2)}" in page
    assert "not that the ball came back" in page


def test_page_six_p_values_describe_the_cells_the_reader_can_see(
    busy: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """A p-value in a column heading is computed over that column's own rows.

    The test's own default is five losses per player; the table prints at
    eleven. A heading that ran the test at five described a population the
    table refuses to show -- on the bundled match the Build up heading printed
    a p-value over fourteen players in a column where every cell read n<11.
    """
    tracking, losses = busy
    disclosure, players = measure(tracking, losses)
    assert disclosure.permutation_min_count == disclosure.min_losses
    assert disclosure.permutation_min_count > baseline.MIN_LOSSES_FOR_PLAYER - 1
    for phase, printed in disclosure.phase_printed.items():
        if printed == 0:
            assert not np.isfinite(disclosure.phase_p.get(phase, float("nan"))), (
                f"{phase} prints nothing, so it must carry no p-value"
            )


@pytest.mark.parametrize("fixture", ["thin", "busy"])
def test_page_six_names_both_populations_of_the_position_check(
    fixture: str,
    match: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    built: Path,
    busy: tuple[pd.DataFrame, pd.DataFrame],
    built_busy: Path,
) -> None:
    """The wider figure is the better estimate; the narrower one is checkable.

    Both fixtures matter. On the busy one some rows print and some do not, so
    the two populations differ and the page has to name both. On the thin one
    nothing prints at all, so the narrower figure does not exist and the clause
    has to be absent rather than computed over zero rows.
    """
    tracking, losses = (busy if fixture == "busy" else (match[0], match[2]))
    path = built_busy if fixture == "busy" else built
    disclosure, _ = measure(tracking, losses)
    page = pdf_page_prose(path)[5]
    assert f"over {disclosure.position_rows} players" in page
    if (
        np.isfinite(disclosure.position_r_printed)
        and disclosure.position_rows_printed
        and disclosure.position_rows_printed != disclosure.position_rows
    ):
        assert f"over the {disclosure.position_rows_printed} of them" in page
        assert f"r = {report._signed(disclosure.position_r_printed, 2)}" in page
    else:
        # Nothing prints a value, so the narrower figure does not exist and the
        # clause is left out rather than filled with a number over zero rows.
        assert "of them whose Recovery cell below" not in page


def test_gaps_survive_a_match_with_one_loss(tmp_path: Path) -> None:
    """A degenerate match prints six pages and no invented p-value."""
    tracking = make_tracking(n_players=2, n_frames=400)
    losses = make_losses(tracking, n=1)
    path = report.build_report(
        tracking, losses, "home", tmp_path / "one.pdf", grid=TEST_GRID
    )
    assert page_count(path) == report.N_PAGES
    disclosure, _ = measure(tracking, losses)
    assert not any(gap.tested for gap in disclosure.cover_gaps.values())
    assert not any(gap.tested for gap in disclosure.phase_gaps.values())
    pages = pdf_page_prose(path)
    assert report.GAP_UNTESTED in pages[0]
    assert "p = " not in pages[0]


def test_the_unscored_footnote_agrees_with_the_map(tmp_path: Path) -> None:
    """The page must not promise grey crosses the reader cannot see.

    ``plots.loss_map`` can only draw a loss it has coordinates for, so an
    unscored loss with no ball position reaches neither the cross layer nor the
    "not scored" legend entry. On the bundled match the single unscored loss is
    exactly that kind, and the footnote used to boast about drawing it anyway.
    This asserts the two agree, in both directions, by reading the legend of
    the figure the page actually embeds.
    """
    tracking = make_tracking(n_players=4, n_frames=500)

    undrawable = make_losses(tracking, n=6, unlocated=2)
    drawable = make_losses(tracking, n=6)
    # Unscored, but with a position: the metric failed on it, the map can draw
    # it, and the footnote is allowed to say so.
    drawable.loc[0, "scored"] = False
    drawable.loc[0, "skip_reason"] = "window_cut_short"
    drawable.attrs.update(drawable.attrs | {"n_skipped": 1})

    for losses, expect_crosses in ((undrawable, False), (drawable, True)):
        disclosure, _ = measure(tracking, losses)
        note = report._unscored_note(disclosure)
        figure = plots.loss_map(losses)
        labels = {
            text.get_text()
            for axis in figure.axes
            for legend in ([axis.get_legend()] if axis.get_legend() else [])
            for text in legend.get_texts()
        }
        # `plots` never imports pyplot -- see the invariant asserted in
        # tests/test_plots.py -- so there is no global registry to close
        # against, and clearing the figure is the whole teardown.
        figure.clear()
        drawn = "not scored" in labels
        assert drawn is expect_crosses, f"legend {labels}"
        if drawn:
            assert "drawn as grey crosses" in note
        else:
            assert "no grey crosses on this map" in note


def test_the_unscored_footnote_reconciles_the_phase_table(tmp_path: Path) -> None:
    """A loss with no position has no third either, and the pages add up.

    The cover counts every detected loss; page 2's phase table can only count
    the ones that have a pitch third. On the bundled match those two are 256
    and 255, and a reader who adds the Losses column and gets a different
    number from the headline deserves the sentence that explains it.
    """
    tracking = make_tracking(n_players=4, n_frames=500)
    losses = make_losses(tracking, n=6, unlocated=2)
    disclosure, _ = measure(tracking, losses)
    assert disclosure.total_losses == 6
    assert disclosure.phase_rows == 4
    assert disclosure.phase_table_short_by == 2

    path = report.build_report(
        tracking, losses, "home", tmp_path / "short.pdf", grid=TEST_GRID
    )
    pages = pdf_page_prose(path)
    assert "adds up to 4 and not the 6 the cover counts" in pages[1]
    assert "page 2's phase table works from 4" in pages[0]


def test_a_match_with_nothing_missing_makes_no_excuses(tmp_path: Path) -> None:
    """No unscored losses, no sentence about unscored losses."""
    tracking = make_tracking(n_players=4, n_frames=500)
    losses = make_losses(tracking, n=6)
    disclosure, _ = measure(tracking, losses)
    assert disclosure.unscored == 0
    note = report._unscored_note(disclosure)
    assert "Every loss on this map was scored" in note
    assert "grey cross" not in note.replace("grey crosses on it", "")


def test_the_residual_is_flagged_as_in_sample_on_a_single_match(
    match: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], built: Path
) -> None:
    """One match cannot hold anything out, and the page admits it."""
    tracking, _, losses = match
    disclosure, _ = measure(tracking, losses)
    assert disclosure.residual_out_of_sample is False
    assert "in-sample and if anything flatters itself" in pdf_page_prose(built)[5]


def test_disclosure_survives_a_match_it_cannot_measure() -> None:
    """Nothing scorable means no numbers, and no numbers means no sentences."""
    tracking = make_tracking(n_players=3, n_frames=300)
    losses = make_losses(tracking, n=3, unlocated=3)
    disclosure, _ = measure(tracking, losses)
    assert disclosure.n_scored == 0
    assert not disclosure.has_position_check
    fonts = report.register_pdf_fonts()
    styles = report._build_styles(*fonts)
    warning = report._rank_warning(disclosure, styles)
    assert warning is not None
    flat = " ".join(
        " ".join(str(getattr(item, "text", "")) for item in cell)
        for row in warning._cellvalues
        for cell in row
    )
    assert "r = —" not in flat
    assert "nan" not in flat.lower()


def page_six_story(
    tracking: pd.DataFrame, losses: pd.DataFrame
) -> tuple[list[Any], report.Disclosure]:
    """Page 6's flowables, without going through a PDF.

    Args:
        tracking: Canonical tracking frame.
        losses: Scored losses.

    Returns:
        ``(story, disclosure)``.
    """
    disclosure, players = measure(tracking, losses)
    fonts = report.register_pdf_fonts()
    styles = report._build_styles(*fonts)
    story = report.player_table_page(
        players,
        "home",
        goalkeepers=tracking.attrs.get("goalkeeper", {}) or {},
        labels=None,
        radius_m=report.DEFAULT_RADIUS_M,
        styles=styles,
        fonts=fonts,
        opponent="away",
        disclosure=disclosure,
    )
    return story, disclosure


def test_page_six_does_not_overflow_far_enough_to_shrink_the_warning(
    busy: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    """Page 6 has to nearly fit, because overflow is paid for in warning size.

    ``build_report`` wraps every page in ``KeepInFrame(mode="shrink")``, which
    holds the page count at six by scaling the whole page down. Measured on the
    bundled match when the warning was first added: 1,033 pt of content in a
    748 pt frame, a scale of 0.72, and squad tables 367 pt wide inside a 510 pt
    frame -- at which point the warning heading renders at 7.6 pt. The bound
    here is loose because two full squads is genuinely a lot of page; it exists
    to catch a return to 0.72, not to police 0.95.
    """
    tracking, losses = busy
    story, _ = page_six_story(tracking, losses)
    total = sum(
        item.wrap(report.FRAME_W, report.FRAME_H)[1]
        + item.getSpaceBefore()
        + item.getSpaceAfter()
        for item in story
    )
    assert total <= report.FRAME_H * 1.15, (
        f"page 6 is {total:.0f} pt of content in a {report.FRAME_H:.0f} pt frame, "
        f"a scale of {report.FRAME_H / total:.2f}"
    )


def test_whatever_page_six_shrink_remains_is_uniform(
    busy: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    """A shrunk page must read as a smaller page, not a broken one.

    ``KeepInFrame`` re-wraps an overflowing page at a wider measure and draws
    the result scaled down. A paragraph reflows into that wider measure and
    comes back full width; a table with fixed column widths cannot. Mixing the
    two puts 367 pt tables under 496 pt of footnote. Every block on this page
    is therefore pinned to the frame width, so all of it shrinks together.
    """
    tracking, losses = busy
    story, _ = page_six_story(tracking, losses)
    for item in story:
        if isinstance(item, report.Paragraph):
            # Headings are single lines: they scale, they do not reflow.
            assert "\n" not in item.text
            continue
        if isinstance(item, report.Spacer):
            continue
        width, _ = item.wrap(report.FRAME_W, report.FRAME_H)
        assert width == pytest.approx(report.FRAME_W), f"{type(item).__name__} is {width:.0f} pt"


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
