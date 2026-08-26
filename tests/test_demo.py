"""The bundled offline demo.

These tests are the acceptance criterion in executable form: the assets are
committed, they load without touching ``data/raw/`` or the network, and the
whole pipeline runs off them into a six-page PDF.

The end-to-end test really does run the whole thing on the real match. It is
the slowest test in the suite by a wide margin and that is the point -- a demo
that is only ever exercised through mocks is a demo nobody has run.
"""

from __future__ import annotations

import ast
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import pressing
from pressing import SATURATED_CPU_SHARE, adaptive_jobs, cpu_availability, load_average
from pressing.demo import (
    DEMO_DIR,
    DEMO_GAME,
    DEMO_RANK,
    EVENTS_ASSET,
    TRACKING_ASSET,
    _panel_name,
    asset_summary,
    assets_present,
    demo_data_path,
    demo_fps,
    dump_all_panels,
    load_demo_match,
    read_demo_events,
    run_demo,
    write_demo_events,
)
from pressing.schema import EVENT_COLUMNS, TRACKING_COLUMNS

REPO_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# --------------------------------------------------------------------------- #
# The committed assets
# --------------------------------------------------------------------------- #


def test_assets_are_present() -> None:
    assert assets_present()
    assert demo_data_path() == DEMO_DIR
    assert (DEMO_DIR / TRACKING_ASSET).is_file()
    assert (DEMO_DIR / EVENTS_ASSET).is_file()


def test_notice_credits_metrica_and_says_the_data_is_derived() -> None:
    """The notice is a licensing obligation, not decoration."""
    text = (DEMO_DIR / "NOTICE.md").read_text(encoding="utf-8")
    assert "Metrica Sports" in text
    assert "https://github.com/metrica-sports/sample-data" in text
    assert "derived" in text.lower()
    assert "no licence" in text.lower()
    assert "removed on request" in text.lower()


def test_assets_are_tracked_by_git() -> None:
    """``data/raw/`` is gitignored; ``data/demo/`` must not be.

    Checking the working tree is not enough. The demo has to work on a fresh
    clone, so what matters is whether git is carrying the files.
    """
    if shutil.which("git") is None or not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    listed = subprocess.run(
        ["git", "ls-files", "data/demo"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert f"data/demo/{TRACKING_ASSET}" in listed
    assert f"data/demo/{EVENTS_ASSET}" in listed
    assert "data/demo/NOTICE.md" in listed


def test_asset_summary_reports_sizes() -> None:
    sizes = dict(asset_summary())
    assert set(sizes) == {TRACKING_ASSET, EVENTS_ASSET}
    assert all(n > 0 for n in sizes.values())


def test_missing_asset_raises_a_useful_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as excinfo:
        load_demo_match(tmp_path)
    message = str(excinfo.value)
    assert TRACKING_ASSET in message
    assert "build_demo_assets" in message


# --------------------------------------------------------------------------- #
# The events asset round-trips
# --------------------------------------------------------------------------- #


def _tiny_events() -> pd.DataFrame:
    """A two-row canonical events frame exercising every awkward field."""
    frame = pd.DataFrame(
        {
            "match_id": ["m1", "m1"],
            "competition": ["Test", "Test"],
            "season": [pd.NA, pd.NA],
            "team_id": ["home", "away"],
            "team": ["home", "away"],
            # A non-Latin-1 name, because reportlab silently mangles those and
            # a round-trip that only ever sees ASCII proves nothing.
            "player_id": ["Vuković", None],
            "player": ["Vuković", None],
            "t": [12.5, 30.0],
            "period": pd.array([1, 2], dtype="int8"),
            "type": ["PASS", "BALL LOST"],
            "subtype": ["HEAD", None],
            "x": [10.0, 90.5],
            "y": [34.0, 12.25],
            "end_x": [20.0, 95.0],
            "end_y": [30.0, 10.0],
            "outcome": ["lost", None],
            "pressure": [False, True],
            "body_part": ["head", None],
            "extra": [
                {
                    "start_frame": 10,
                    "end_frame": 20,
                    "end_t": 0.8,
                    "to_player": "Player2",
                    "subtype_tokens": frozenset({"HEAD", "LOST"}),
                    "event_index": 0,
                },
                {
                    "start_frame": 21,
                    "end_frame": 25,
                    "end_t": 1.0,
                    "to_player": None,
                    "subtype_tokens": frozenset(),
                    "event_index": 1,
                },
            ],
        }
    )
    frame.attrs.update({"match_id": "m1", "fps": 25.0, "source": "test"})
    return frame


def test_events_asset_round_trips(tmp_path: Path) -> None:
    original = _tiny_events()
    path = write_demo_events(original, tmp_path / "events.parquet")
    back = read_demo_events(path)

    assert list(back.columns) == list(EVENT_COLUMNS)
    for column in ("t", "x", "y", "end_x", "end_y"):
        assert back[column].tolist() == original[column].tolist()
    for column in ("team", "type", "pressure", "period"):
        assert back[column].tolist() == original[column].tolist()
    # Missing strings come back as the dtype's own NA, exactly as
    # ``load_metrica_events`` produces them -- not as None.
    assert back["player"].iloc[0] == "Vuković"
    assert pd.isna(back["player"].iloc[1])
    assert back["subtype"].iloc[0] == "HEAD"
    assert pd.isna(back["subtype"].iloc[1])
    assert back["extra"].tolist() == original["extra"].tolist()
    # The frozenset is the part JSON cannot represent, so pin it explicitly.
    assert back["extra"][0]["subtype_tokens"] == frozenset({"HEAD", "LOST"})
    assert isinstance(back["extra"][1]["subtype_tokens"], frozenset)
    assert back.attrs["fps"] == 25.0


def test_events_asset_is_byte_reproducible(tmp_path: Path) -> None:
    """Two writes of one frame produce identical bytes.

    Only true because ``extra`` is serialised with sorted keys and its
    frozenset sorted into a list; an unsorted set iterates in whatever order
    its hashes give and the asset would churn on every rebuild.
    """
    frame = _tiny_events()
    first = (write_demo_events(frame, tmp_path / "a.parquet")).read_bytes()
    second = (write_demo_events(frame, tmp_path / "b.parquet")).read_bytes()
    assert first == second


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def match() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_demo_match()


def test_loaded_match_is_canonical(match: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    tracking, events = match
    assert list(tracking.columns) == list(TRACKING_COLUMNS)
    assert list(events.columns) == list(EVENT_COLUMNS)
    assert set(tracking["team"].unique()) == {"home", "away", "ball"}
    assert len(tracking) > 3_000_000
    assert len(events) > 1_000


def test_loaded_match_keeps_its_provenance(match: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    tracking, events = match
    for frame in (tracking, events):
        assert frame.attrs["source"] == "metrica"
        assert frame.attrs["game"] == DEMO_GAME
        assert frame.attrs["fps"] == demo_fps() == 25.0
        assert "metrica-sports/sample-data" in frame.attrs["derived_from"]
        assert "NOTICE.md" in frame.attrs["derived_note"]


def test_loaded_match_is_rotated_to_one_direction(
    match: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Home attacks +x in both halves, which is not how Metrica ships it."""
    tracking, _ = match
    assert tracking.attrs["attack_direction"] == {"home": 1, "away": -1}
    assert tracking.attrs["flip_period"] in (1, 2)


def test_velocities_are_recomputed_not_stored(
    match: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """The asset stores positions only; the read has to fill in the rest."""
    tracking, _ = match
    players = tracking[tracking["team"] != "ball"]
    assert players["vx"].notna().mean() > 0.9


def test_coordinates_are_inside_the_pitch(match: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    tracking, _ = match
    x = tracking["x"].dropna()
    y = tracking["y"].dropna()
    # Players run behind the goal line and over the touchline; a few metres of
    # slack is real, a hundred would mean the frame is wrong.
    assert -10.0 < x.min() and x.max() < 115.0
    assert -10.0 < y.min() and y.max() < 78.0


# --------------------------------------------------------------------------- #
# The whole thing
# --------------------------------------------------------------------------- #


def _page_count(pdf: Path) -> int:
    """Count pages by parsing the PDF's own catalogue."""
    blob = pdf.read_bytes()
    counts = [int(m) for m in re.findall(rb"/Count\s+(\d+)", blob)]
    return max(counts) if counts else 0


@pytest.fixture(scope="module")
def demo_run(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """One real end-to-end run, shared by the tests that inspect it."""
    out = tmp_path_factory.mktemp("demo") / "report.pdf"
    return run_demo(out)


def test_run_demo_writes_a_six_page_pdf(demo_run: Any) -> None:
    assert demo_run.pdf.is_file()
    assert demo_run.pdf.stat().st_size > 100_000
    assert _page_count(demo_run.pdf) == 6


def test_run_demo_reports_what_it_did(demo_run: Any) -> None:
    assert demo_run.n_losses > 100
    assert demo_run.n_scored <= demo_run.n_losses
    # A match where almost nothing scores means the metric quietly broke.
    assert demo_run.n_scored / demo_run.n_losses > 0.9
    assert demo_run.jobs >= 1
    assert demo_run.seconds > 0
    assert set(demo_run.timings) >= {"load", "score", "moments", "report"}
    assert sum(demo_run.timings.values()) <= demo_run.seconds + 1e-6


# --------------------------------------------------------------------------- #
# The worker count, and the measurement behind it
# --------------------------------------------------------------------------- #


def _stub_pipeline(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace everything expensive in ``run_demo`` and record what it was asked.

    The tests below are about the worker-count decision and the cost report,
    neither of which needs a real match, a real score or a real PDF. The one
    thing that must stay real is ``run_demo``'s own control flow.

    Args:
        monkeypatch: Fixture.

    Returns:
        A dict that gains a ``"jobs"`` key when scoring is called.
    """
    import pressing.report as report_module

    seen: dict[str, Any] = {}

    def spy_score(*args: Any, **kwargs: Any) -> pd.DataFrame:
        seen["jobs"] = kwargs.get("jobs")
        return _stub_losses()

    monkeypatch.setattr(
        "pressing.demo.load_demo_match", lambda *a, **k: (_stub_tracking(), _stub_events())
    )
    monkeypatch.setattr("pressing.counterpress.score_losses", spy_score)
    monkeypatch.setattr(report_module, "render_moments", lambda *a, **k: [])
    monkeypatch.setattr(report_module, "build_report", lambda *a, **k: Path(a[3]))
    return seen


def test_adaptive_jobs_never_leaves_its_bounds() -> None:
    """Whatever the probe returns, the answer is a usable worker count."""
    for share in (0.0, 0.01, 0.34, 0.58, SATURATED_CPU_SHARE, 0.95, 1.0):
        assert 1 <= adaptive_jobs(cap=6, availability=share) <= 6
    assert adaptive_jobs(cap=1, availability=1.0) == 1


def test_adaptive_jobs_is_the_old_fixed_default_when_a_core_is_free() -> None:
    """Nothing changes where nothing needed to change."""
    from pressing.counterpress import default_jobs

    assert adaptive_jobs(availability=1.0) == default_jobs()
    # 0.94 was measured on the box where six workers beat serial by 2-3x.
    assert adaptive_jobs(availability=0.94) == default_jobs()


def test_adaptive_jobs_falls_to_the_serial_path_on_a_saturated_machine() -> None:
    """Both shares here are measured, and serial won the run at each of them.

    0.31-0.58 came from eight CPU burners, where serial scored the match 2.4x
    faster; 0.73 came from twenty-four, where the whole command was 215.7 s
    serial against 275.8 s on six workers.
    """
    assert adaptive_jobs(cap=6, availability=0.34) == 1
    assert adaptive_jobs(cap=6, availability=0.58) == 1
    assert adaptive_jobs(cap=6, availability=0.73) == 1


def test_the_threshold_sits_between_the_two_measured_regimes() -> None:
    """It is a midpoint of ignorance, and it must stay inside what was measured.

    Serial won at 0.73 and the cap won at 0.94. A constant that drifted outside
    that bracket would be contradicting a measurement rather than interpolating
    between two.
    """
    assert 0.73 < SATURATED_CPU_SHARE < 0.94


def test_adaptive_jobs_does_not_probe_an_idle_machine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A machine with under one runnable thread cannot be saturated.

    Skipping the probe there is what keeps its 0.24 s off the common path, and
    it is only safe because the short-circuit can only ever answer "the cap".
    """
    monkeypatch.setattr(
        pressing, "cpu_availability", lambda *a, **k: pytest.fail("probed an idle box")
    )
    assert adaptive_jobs(cap=6, load=0.4) == 6


def test_adaptive_jobs_probes_when_the_load_average_says_it_might_matter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """And the probe, not the load average, is what decides.

    The load average is passed a value that under the policy this replaced
    would have forced the serial path outright.
    """
    monkeypatch.setattr(pressing, "cpu_availability", lambda *a, **k: 0.97)
    assert adaptive_jobs(cap=6, load=40.0) == 6

    monkeypatch.setattr(pressing, "cpu_availability", lambda *a, **k: 0.40)
    assert adaptive_jobs(cap=6, load=40.0) == 1


def test_the_probe_returns_a_share_of_one_core() -> None:
    """Run for real, on whatever machine the suite is on."""
    share = cpu_availability(samples=2, seconds=0.02)
    assert 0.0 < share <= 1.0


def test_explicit_jobs_beats_the_policy_however_loaded_the_machine_is(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``--jobs`` is an override, not a hint, and it skips the probe entirely."""
    seen = _stub_pipeline(monkeypatch)
    monkeypatch.setattr(
        "pressing.demo.cpu_availability", lambda *a, **k: pytest.fail("probed anyway")
    )

    result = run_demo(tmp_path / "r.pdf", jobs=4)
    assert seen["jobs"] == 4
    assert result.jobs == 4
    assert result.jobs_source == "--jobs"
    assert result.cpu_share is None


def test_jobs_one_is_never_second_guessed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The serial path is a supported choice; an idle machine must not widen it."""
    seen = _stub_pipeline(monkeypatch)
    monkeypatch.setattr("pressing.demo.cpu_availability", lambda *a, **k: 1.0)

    result = run_demo(tmp_path / "r.pdf", jobs=1)
    assert seen["jobs"] == 1
    assert result.jobs == 1
    assert result.jobs_source == "--jobs"


def test_the_default_worker_count_follows_the_measured_cpu_share(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Same call, same machine, two measured shares, two worker counts."""
    seen = _stub_pipeline(monkeypatch)

    monkeypatch.setattr("pressing.demo.cpu_availability", lambda *a, **k: 0.95)
    free = run_demo(tmp_path / "free.pdf")
    assert seen["jobs"] == free.jobs == adaptive_jobs(availability=0.95)
    assert free.cpu_share == 0.95

    monkeypatch.setattr("pressing.demo.cpu_availability", lambda *a, **k: 0.35)
    busy = run_demo(tmp_path / "busy.pdf")
    assert seen["jobs"] == busy.jobs == 1
    assert busy.cpu_share == 0.35

    assert free.jobs_source == busy.jobs_source == "adaptive"


def test_the_run_records_the_load_it_ran_under(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A wall clock without the load it was taken at is not evidence of anything."""
    _stub_pipeline(monkeypatch)
    result = run_demo(tmp_path / "r.pdf", jobs=1)

    # None only where the platform has no load average at all, and then both.
    assert (result.load_start is None) == (load_average() is None)
    assert (result.load_end is None) == (load_average() is None)
    if result.load_start is not None:
        assert result.load_start >= 0.0


# --------------------------------------------------------------------------- #
# The cost the tool reports for itself
# --------------------------------------------------------------------------- #


def test_the_pipeline_clock_is_never_reported_on_its_own(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The whole point of ``cost_line``: the small number cannot stand alone.

    ``DemoResult.seconds`` structurally excludes the imports, and on a cold
    clone those have been measured as the larger half of the command. A line
    that printed it by itself would read as an end-to-end cost.
    """
    _stub_pipeline(monkeypatch)
    result = run_demo(tmp_path / "r.pdf", jobs=1)
    line = result.cost_line()

    assert "wall clock since import" in line
    assert f"{result.seconds:.1f}s in the pipeline" in line
    assert "interpreter start-up is in neither" in line
    assert "CPU" in line
    assert "load average" in line


def test_the_reported_process_clock_is_bigger_than_the_pipeline_clock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """And the gap is exactly the thing the old number could not see."""
    _stub_pipeline(monkeypatch)
    result = run_demo(tmp_path / "r.pdf", jobs=1)

    assert result.process_seconds > result.seconds >= 0.0


def test_the_reported_cpu_covers_the_imports_the_stage_clock_missed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``getrusage`` counts from ``exec``, so numpy and pandas are already in it.

    With the pipeline stubbed out to nothing, any CPU at all can only have come
    from before ``run_demo`` was entered.
    """
    _stub_pipeline(monkeypatch)
    result = run_demo(tmp_path / "r.pdf", jobs=1)

    assert result.cpu_seconds is not None
    assert result.cpu_seconds > result.seconds


def test_run_demo_reports_the_epv_and_metrica_credit(demo_run: Any) -> None:
    """The credit Metrica asks for has to survive into the artefact."""
    if shutil.which("pdftotext") is None:
        pytest.skip("pdftotext not available")
    text = subprocess.run(
        ["pdftotext", str(demo_run.pdf), "-"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "Metrica Sports sample data" in text
    assert "Friends of Tracking" in text


def test_run_demo_touches_no_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Not a proxy for offline: sockets are made to raise, then the run happens.

    Scoped to the cheap part of the pipeline -- load plus detect -- because
    that is where every remaining loader lives; the rest is arithmetic and
    matplotlib.
    """

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the demo must not open a socket")

    for name in ("socket", "create_connection", "getaddrinfo", "gethostbyname"):
        monkeypatch.setattr(socket, name, refuse)

    tracking, events = load_demo_match()
    from pressing.counterpress import find_possession_losses

    assert len(find_possession_losses(tracking, events)) > 100


def test_run_demo_does_not_read_data_raw(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing in the demo path may open anything under ``data/raw/``."""
    raw = str((REPO_ROOT / "data" / "raw").resolve())
    real_open = Path.open

    def guarded(self: Path, *args: object, **kwargs: object) -> Any:
        if str(self.resolve()).startswith(raw):
            raise AssertionError(f"the demo read {self}, which a fresh clone lacks")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    tracking, events = load_demo_match()
    assert len(tracking) > 0 and len(events) > 0


def test_the_scores_do_not_depend_on_the_worker_count(
    match: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Load-bearing in a way it was not before the default became adaptive.

    Two runs of ``pressing demo`` on the same machine ten minutes apart can now
    legitimately choose different worker counts, so worker count has to be
    invisible in the output or the report stops being reproducible.

    Exact equality, not ``approx``: both paths do the same float64 arithmetic
    on the same windows, and a difference could only come from how the losses
    were divided between workers. A quarter of the real match, spread evenly
    over it, so the split is genuinely uneven across four workers.
    """
    from pressing.counterpress import find_possession_losses, score_losses

    tracking, events = match
    losses = find_possession_losses(tracking, events).iloc[::4].reset_index(drop=True)
    assert len(losses) > 50

    serial = score_losses(tracking, events, losses=losses, jobs=1)
    parallel = score_losses(tracking, events, losses=losses, jobs=4)

    assert serial["scored"].fillna(False).astype(bool).sum() > 50
    pd.testing.assert_frame_equal(serial, parallel, check_exact=True)


# --------------------------------------------------------------------------- #
# --all-panels
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def scored(match: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    """Every loss in the bundled match, scored once and shared."""
    from pressing.counterpress import score_losses

    tracking, events = match
    losses = score_losses(tracking, events)
    return losses[losses["scored"].fillna(False).astype(bool)]


def test_dump_all_panels_writes_one_png_per_loss(
    match: tuple[pd.DataFrame, pd.DataFrame], scored: pd.DataFrame, tmp_path: Path
) -> None:
    """Run on a slice: the behaviour under test is one file per row, not 255."""
    tracking, events = match
    subset = scored.head(3)

    written = dump_all_panels(tracking, subset, tmp_path, events=events, dpi=60)
    assert written == len(subset)
    files = sorted(tmp_path.glob("*.png"))
    assert len(files) == len(subset)
    assert all(f.stat().st_size > 1000 for f in files)
    # dump_all_panels is the one place in the package that builds figures in a
    # loop, and it used to import pyplot to close them. See the pyplot tests
    # below for why that is not allowed.
    assert "matplotlib.pyplot" not in sys.modules


# --------------------------------------------------------------------------- #
# pyplot must stay out of the process
# --------------------------------------------------------------------------- #


def _pyplot_importers() -> list[str]:
    """Every ``src/pressing`` module that imports pyplot, at any nesting depth.

    Walks the AST rather than grepping so that a function-local
    ``from matplotlib import pyplot`` -- which is exactly how this regression
    got in, invisible to any module-level import check -- is still caught.
    """
    offenders = []
    for path in sorted((REPO_ROOT / "src" / "pressing").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = [f"{module}.{alias.name}" for alias in node.names]
            else:
                continue
            if any(name == "matplotlib.pyplot" for name in names):
                offenders.append(path.name)
                break
    return offenders


def test_no_module_imports_pyplot_anywhere() -> None:
    """``pyplot`` is process-global state, and this package renders in a loop.

    ``plots`` builds bare ``Figure`` objects on purpose: pyplot keeps every
    figure it makes in a global registry, so a report that builds hundreds of
    panels through pyplot leaks all of them. ``demo.dump_all_panels`` once
    imported pyplot inside the function purely to call ``plt.close(fig)`` --
    which does nothing at all to a bare figure, because it was never in the
    registry to begin with -- and the cost was a poisoned ``sys.modules`` for
    the whole process and 25 unrelated failures in ``test_plots.py``.
    """
    assert _pyplot_importers() == []


def test_the_pyplot_scan_would_actually_catch_it(tmp_path: Path) -> None:
    """The guard above is only worth having if it can fail. Prove it can."""
    module = tmp_path / "offender.py"
    module.write_text("def f():\n    from matplotlib import pyplot as plt\n")
    tree = ast.parse(module.read_text(encoding="utf-8"))
    found = [
        f"{node.module}.{alias.name}"
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    ]
    assert "matplotlib.pyplot" in found


def test_a_full_render_leaves_pyplot_unimported() -> None:
    """Order-independent: a fresh process draws real figures, pyplot stays out.

    The in-session assertions above only hold if nothing earlier imported
    pyplot. This one cannot be affected by test order at all.
    """
    source = (
        "import sys; import numpy as np;"
        "from pressing import plots;"
        "from pressing.pitchcontrol import make_grid;"
        "g = make_grid();"
        "s = np.full(g.shape, 0.5);"
        "f = plots.control_heatmap(s); f.clear();"
        "print('matplotlib.pyplot' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=True
    ).stdout
    assert out.strip() == "False"


# --------------------------------------------------------------------------- #
# Which five moments get printed
# --------------------------------------------------------------------------- #


def test_recovery_ranking_is_really_a_phase_ranking(scored: pd.DataFrame) -> None:
    """The measured reason the demo does not rank by recovery share.

    The spec calls pages 3-5 the "top-5 recovery moments". Taken literally that
    sorts by recovery share -- and on the bundled match that returns 5 final
    third losses out of 5, for both teams, because recovery share is dominated
    by where the ball was lost. This test exists so that if the metric ever
    stops behaving that way, the decision gets revisited instead of inherited.
    """
    means = scored.groupby("phase", observed=True)["recovery"].mean()
    assert means["build_up"] < means["middle"] < means["final_third"]
    # A gradient this steep is the confound, not a detail.
    assert means["final_third"] - means["build_up"] > 0.3

    for team in ("home", "away"):
        table = scored[scored["losing_team"] == team]
        top = table.sort_values("recovery", ascending=False, kind="stable").head(5)
        assert set(top["phase"]) == {"final_third"}


def test_the_costly_ranking_is_not_a_restatement_of_recovery(
    scored: pd.DataFrame,
) -> None:
    """The two rankings answer different questions, so the choice is real."""
    assert abs(scored["recovery"].corr(scored["epv_conceded"])) < 0.3
    # And it reaches losses the recovery ranking never shows.
    costly = scored.sort_values("epv_conceded", ascending=False, kind="stable").head(10)
    assert set(costly["phase"]) != {"final_third"}


def test_demo_ranks_by_epv_swing_by_default() -> None:
    assert DEMO_RANK == "costly"


def test_the_report_is_told_the_same_ranking_it_is_given_moments_for(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The page heading names the ranking; the panels come from a separate call.

    ``build_report`` captions the moment pages from the ``rank`` it is passed,
    but accepts pre-computed ``moments``. Nothing makes the two agree except
    ``run_demo`` passing the same value to both, so a report captioned "ranked
    by recovery share" over panels chosen by EPV swing is one dropped keyword
    away. Pin it.
    """
    import pressing.report as report_module

    seen: dict[str, Any] = {}

    def spy_render(*args: Any, **kwargs: Any) -> Any:
        seen["moments_rank"] = kwargs.get("rank")
        return []

    def spy_build(*args: Any, **kwargs: Any) -> Path:
        seen["report_rank"] = kwargs.get("rank")
        return Path(args[3])

    monkeypatch.setattr(report_module, "render_moments", spy_render)
    monkeypatch.setattr(report_module, "build_report", spy_build)
    monkeypatch.setattr(
        "pressing.demo.load_demo_match", lambda *a, **k: (_stub_tracking(), _stub_events())
    )
    monkeypatch.setattr("pressing.counterpress.score_losses", lambda *a, **k: _stub_losses())

    run_demo(tmp_path / "r.pdf", rank="recovered", jobs=1)
    assert seen["moments_rank"] == "recovered"
    assert seen["report_rank"] == "recovered"


def _stub_tracking() -> pd.DataFrame:
    frame = pd.DataFrame({"frame": [0], "team": ["home"], "x": [1.0], "y": [1.0]})
    frame.attrs["fps"] = 25.0
    return frame


def _stub_events() -> pd.DataFrame:
    return pd.DataFrame({"type": ["BALL LOST"]})


def _stub_losses() -> pd.DataFrame:
    return pd.DataFrame({"scored": [True], "loss_id": [1], "t": [10.0]})


def test_panel_names_sort_into_match_order() -> None:
    early = _panel_name({"loss_id": 2, "t": 65.0, "losing_team": "home"}, 1)
    late = _panel_name({"loss_id": 40, "t": 3400.0, "losing_team": "away"}, 2)
    assert early == "loss_0002_001m05s_home.png"
    assert late == "loss_0040_056m40s_away.png"
    assert early < late


def test_panel_name_survives_a_row_with_nothing_in_it() -> None:
    assert _panel_name({}, 7) == "loss_0007_000m00s_unknown.png"
