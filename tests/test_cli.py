"""The command line interface.

Three things are being pinned here. First, that the flags a user types actually
reach the functions behind them -- ``--jobs``, ``--team``, ``--rank`` and
``--all-panels`` are wired through with a stub in place of the pipeline,
because the pipeline itself is tested in ``test_demo.py`` and running it again
per flag would cost minutes to prove routing.

Second, that a subcommand asked for something the tool does not do fails like a
tool and not like a bug: a clear sentence and a distinct exit code, never a
traceback, and never a plausible-looking output full of numbers nobody
computed. ``proxy`` and ``season`` are both built now, so the case that remains
is an event source that has not been wired up.

Third, that the exit codes stay meaningful. Success, "this run broke", "you
typed it wrong" and "that feature does not exist" are four different things a
script has to be able to tell apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import pressing
from pressing import cli
from pressing.cli import NOT_BUILT, USAGE_ERROR, app

runner = CliRunner()


def _run(*args: str) -> Any:
    return runner.invoke(app, list(args))


# --------------------------------------------------------------------------- #
# The shell of it
# --------------------------------------------------------------------------- #


def test_help_lists_every_subcommand() -> None:
    result = _run("--help")
    assert result.exit_code == 0
    for name in ("report", "proxy", "season", "demo"):
        assert name in result.output


@pytest.mark.parametrize("command", ["report", "proxy", "season", "demo"])
def test_no_subcommand_is_advertised_as_unbuilt_any_more(command: str) -> None:
    """All four do something now, so none of them may claim otherwise.

    This test used to assert the opposite: that ``proxy`` and ``season`` said
    "NOT BUILT YET" on their own help screens, because finding out only after
    typing the arguments is finding out late. The same reasoning now runs the
    other way -- a working command that still advertises itself as unbuilt is a
    command nobody will type.
    """
    assert "NOT BUILT YET" not in _run(command, "--help").output
    assert "NOT BUILT YET" not in _run("--help").output


def test_version() -> None:
    result = _run("--version")
    assert result.exit_code == 0
    assert pressing.__version__ in result.output


def test_help_does_not_import_matplotlib() -> None:
    """``--help`` must not pay for the analysis stack.

    Imports in ``cli`` are lazy for two reasons: a help screen that takes
    several seconds is a bad help screen, and
    :func:`pressing.warm_matplotlib_cache` can only overlap the font scan with
    real work if matplotlib has not already been imported.
    """
    import subprocess
    import sys

    source = (
        "import sys; from pressing import cli; "
        "print('matplotlib' in sys.modules, 'reportlab' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=True
    ).stdout
    assert out.strip() == "False False"


# --------------------------------------------------------------------------- #
# Unbuilt features fail honestly
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("source", ["skillcorner", "metrica", "opta"])
def test_an_unwired_event_source_says_so_and_exits_three(source: str) -> None:
    """StatsBomb is the only source with events and lineups both."""
    result = _run("season", "--source", source, "--competition", "La Liga")
    assert result.exit_code == NOT_BUILT
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert f"`pressing season --source {source}` is not built yet." in result.output
    assert "Traceback" not in result.output
    # It has to point somewhere that works, or the message is a dead end.
    assert "pressing demo" in result.output
    assert "pressing season --source statsbomb" in result.output


def test_unbuilt_exit_code_is_distinct_from_a_failed_run() -> None:
    """A tool that cannot do something is not the same as a run that broke."""
    assert NOT_BUILT != 1
    assert _run("season", "--source", "skillcorner").exit_code == NOT_BUILT
    assert _run("proxy", "no_such_events.json").exit_code == 1
    assert _run("report", "nope.csv", "--team", "home").exit_code == 1


def test_unbuilt_exit_code_is_also_distinct_from_a_typo() -> None:
    """The three failure modes have to be three codes, or the code says nothing.

    ``NOT_BUILT`` was 2, which is the code click already exits with for any
    usage error, so "season mode does not exist" and "you misspelled a flag"
    were indistinguishable to a caller -- the exact distinction the constant is
    for.
    """
    assert NOT_BUILT != USAGE_ERROR != 1
    assert _run("demo", "--jobs", "0").exit_code == USAGE_ERROR
    assert _run("demo", "--rank", "nonsense").exit_code == USAGE_ERROR
    assert _run("demo", "--no-such-flag").exit_code == USAGE_ERROR
    assert _run("season", "--limit", "0").exit_code == USAGE_ERROR
    assert _run("season", "--source", "skillcorner").exit_code == NOT_BUILT


# --------------------------------------------------------------------------- #
# report: input handling
# --------------------------------------------------------------------------- #


def test_report_rejects_a_team_that_is_not_home_or_away() -> None:
    result = _run("report", "whatever.parquet", "--team", "Liverpool")
    assert result.exit_code == 1
    assert "home" in result.output and "away" in result.output


def test_report_rejects_a_missing_path() -> None:
    result = _run("report", "definitely/not/here.parquet", "--team", "home")
    assert result.exit_code == 1
    assert "no such file" in result.output


def test_report_requires_events_alongside_a_parquet_asset(tmp_path: Path) -> None:
    asset = tmp_path / "tracking.parquet"
    asset.write_bytes(b"not really a parquet, and never opened")
    result = _run("report", str(asset), "--team", "home")
    assert result.exit_code == 1
    assert "--events" in result.output


def test_report_rejects_an_unreadable_events_file(tmp_path: Path) -> None:
    asset = tmp_path / "tracking.parquet"
    asset.write_bytes(b"")
    events = tmp_path / "events.json"
    events.write_text("{}")
    result = _run("report", str(asset), "--team", "home", "--events", str(events))
    assert result.exit_code == 1
    assert "events" in result.output


def test_report_rejects_a_directory_with_no_match_in_it(tmp_path: Path) -> None:
    result = _run("report", str(tmp_path), "--team", "home")
    assert result.exit_code == 1
    assert "Sample_Game_N" in result.output


def test_report_rejects_an_unrecognised_file(tmp_path: Path) -> None:
    path = tmp_path / "something.xlsx"
    path.write_text("x")
    result = _run("report", str(path), "--team", "home")
    assert result.exit_code == 1
    assert "do not know how to read" in result.output


def test_directory_glob_ignores_the_flattened_csv_copies(tmp_path: Path) -> None:
    """Metrica dumps often hold both ``Sample_Game_1/`` and flattened copies.

    Globbing ``Sample_Game_*`` picks up
    ``Sample_Game_1__Sample_Game_1_RawTrackingData_Home_Team.csv`` as well, and
    the game number is then parsed out of ``Team.csv``. That was a real
    traceback, so it gets a test.
    """
    (tmp_path / "Sample_Game_1__Sample_Game_1_RawTrackingData_Home_Team.csv").write_text("x")
    result = _run("report", str(tmp_path), "--team", "home")
    assert result.exit_code == 1
    assert "Sample_Game_N" in result.output
    assert "Traceback" not in result.output


def test_a_csv_with_no_game_number_fails_in_words_not_a_valueerror(tmp_path: Path) -> None:
    """``Sample_Game_Final.csv`` looks like a match file until int() sees it."""
    path = tmp_path / "Sample_Game_Final_RawTrackingData_Home_Team.csv"
    path.write_text("x")
    result = _run("report", str(path), "--team", "home")
    assert result.exit_code == 1
    assert "game number" in result.output
    assert "Traceback" not in result.output
    assert not isinstance(result.exception, ValueError)


# --------------------------------------------------------------------------- #
# demo: the flags reach the pipeline
# --------------------------------------------------------------------------- #


class _Recorder:
    """Stands in for :func:`pressing.demo.run_demo` and records its call."""

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}
        self.out: Any = None

    def __call__(self, out: Any, **kwargs: Any) -> Any:
        from pressing.demo import DemoResult

        self.out = out
        self.kwargs = kwargs
        return DemoResult(
            pdf=Path(out),
            n_losses=256,
            n_scored=255,
            jobs=6,
            seconds=11.9,
            process_seconds=17.4,
            cpu_seconds=48.6,
            load_start=1.25,
            load_end=3.5,
        )


@pytest.fixture()
def recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    stub = _Recorder()
    monkeypatch.setattr("pressing.demo.run_demo", stub)
    return stub


def test_demo_defaults(recorder: _Recorder) -> None:
    from pressing.demo import DEMO_RANK

    result = _run("demo")
    assert result.exit_code == 0
    assert Path(recorder.out) == Path("out/demo_report.pdf")
    assert recorder.kwargs["jobs"] is None
    assert recorder.kwargs["team"] == "home"
    assert recorder.kwargs["rank"] == DEMO_RANK
    assert recorder.kwargs["all_panels_dir"] is None
    assert callable(recorder.kwargs["progress"])


@pytest.mark.parametrize("rank", ["costly", "recovered", "conceded"])
def test_demo_passes_every_ranking_through(recorder: _Recorder, rank: str) -> None:
    """Including 'recovered', which is the spec's literal "top-5 recovery"."""
    result = _run("demo", "--rank", rank)
    assert result.exit_code == 0
    assert recorder.kwargs["rank"] == rank


def test_rank_choices_match_the_report_module() -> None:
    """``--help`` restates the rankings so it costs no import. Keep them equal."""
    from typing import get_args

    from pressing.cli import RankChoice
    from pressing.report import MomentRank

    assert set(get_args(RankChoice)) == set(get_args(MomentRank))


def test_demo_passes_jobs_team_and_all_panels(recorder: _Recorder, tmp_path: Path) -> None:
    panels = tmp_path / "panels"
    result = _run(
        "demo",
        "--out",
        str(tmp_path / "r.pdf"),
        "--team",
        "away",
        "--jobs",
        "3",
        "--all-panels",
        str(panels),
    )
    assert result.exit_code == 0
    assert recorder.kwargs["jobs"] == 3
    assert recorder.kwargs["team"] == "away"
    assert Path(recorder.kwargs["all_panels_dir"]) == panels


def test_demo_prints_the_pdf_path_on_stdout(recorder: _Recorder, tmp_path: Path) -> None:
    """The path goes to stdout; progress goes to stderr. ``$(pressing demo)``."""
    target = tmp_path / "r.pdf"
    result = _run("demo", "--out", str(target))
    assert result.exit_code == 0
    assert str(target) in result.output


def test_demo_rejects_a_bad_team_before_doing_any_work(recorder: _Recorder) -> None:
    result = _run("demo", "--team", "both")
    assert result.exit_code == 1
    assert recorder.kwargs == {}


def test_demo_rejects_zero_jobs(recorder: _Recorder) -> None:
    result = _run("demo", "--jobs", "0")
    assert result.exit_code != 0
    assert recorder.kwargs == {}


def test_demo_leaves_the_worker_count_to_run_demo(recorder: _Recorder) -> None:
    """One place decides it, and the CLI is not that place.

    ``--jobs`` unset arrives at :func:`pressing.demo.run_demo` as None, which is
    what lets the load reading and the count that came out of it be recorded on
    the same result object the run reports from.
    """
    assert _run("demo").exit_code == 0
    assert recorder.kwargs["jobs"] is None


def test_demo_never_prints_the_pipeline_clock_by_itself(recorder: _Recorder) -> None:
    """The reported cost must not be readable as an end-to-end number.

    The stub returns 11.9 s of pipeline clock against 17.4 s since import and
    48.6 s of CPU. Printing the first alone -- which is what this command used
    to do -- understates the command by more than the number it prints.
    """
    result = _run("demo")
    assert result.exit_code == 0
    assert "11.9s in the pipeline" in result.output
    assert "17.4s wall clock since import" in result.output
    assert "48.6s CPU including imports and workers" in result.output
    assert "1-min load average 1.25 -> 3.50" in result.output
    assert "interpreter start-up is in neither" in result.output


def test_demo_says_where_the_worker_count_came_from(recorder: _Recorder) -> None:
    """A count that moves between runs looks like a bug unless the run says why."""
    result = _run("demo")
    assert "6 worker(s) (adaptive)" in result.output


def test_report_falls_back_to_serial_on_a_saturated_machine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``report`` shares the policy, and shows the measurement behind it."""
    seen: dict[str, Any] = {}

    monkeypatch.setattr("pressing.cli._load_match", lambda t, e: ("TRACKING", "EVENTS"))
    monkeypatch.setattr("pressing.report.build_report", lambda *a, **k: Path(a[3]))
    monkeypatch.setattr(pressing, "cpu_availability", lambda *a, **k: 0.35)

    def fake_score(*args: Any, **kwargs: Any) -> Any:
        seen["jobs"] = kwargs.get("jobs")
        import pandas as pd

        return pd.DataFrame({"scored": [True]})

    monkeypatch.setattr("pressing.counterpress.score_losses", fake_score)

    result = _run("report", str(tmp_path), "--team", "home", "--out", str(tmp_path / "r.pdf"))
    assert result.exit_code == 0, result.output
    assert seen["jobs"] == 1
    assert "1 worker(s) (adaptive, CPU share 0.35)" in result.output


def test_report_uses_the_full_cap_when_a_core_is_free(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The other side of the same switch, so neither branch is untested."""
    seen: dict[str, Any] = {}

    monkeypatch.setattr("pressing.cli._load_match", lambda t, e: ("TRACKING", "EVENTS"))
    monkeypatch.setattr("pressing.report.build_report", lambda *a, **k: Path(a[3]))
    monkeypatch.setattr(pressing, "cpu_availability", lambda *a, **k: 0.96)

    def fake_score(*args: Any, **kwargs: Any) -> Any:
        seen["jobs"] = kwargs.get("jobs")
        import pandas as pd

        return pd.DataFrame({"scored": [True]})

    monkeypatch.setattr("pressing.counterpress.score_losses", fake_score)

    result = _run("report", str(tmp_path), "--team", "home", "--out", str(tmp_path / "r.pdf"))
    assert result.exit_code == 0, result.output
    assert seen["jobs"] == pressing.adaptive_jobs(availability=0.96)
    assert "CPU share 0.96" in result.output


# --------------------------------------------------------------------------- #
# report: the flags reach the pipeline too
# --------------------------------------------------------------------------- #


def test_report_wires_rank_jobs_and_all_panels_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``report`` has the same flags as ``demo`` and must route them the same.

    Stubbed at the loader, so this proves routing without paying for a match.
    """
    seen: dict[str, Any] = {}

    monkeypatch.setattr("pressing.cli._load_match", lambda t, e: ("TRACKING", "EVENTS"))

    def fake_score(*args: Any, **kwargs: Any) -> Any:
        seen["jobs"] = kwargs.get("jobs")
        import pandas as pd

        return pd.DataFrame({"scored": [True, True]})

    def fake_build(*args: Any, **kwargs: Any) -> Path:
        seen["rank"] = kwargs.get("rank")
        return Path(args[3])

    def fake_panels(*args: Any, **kwargs: Any) -> int:
        seen["panels_dir"] = args[2]
        return 4

    monkeypatch.setattr("pressing.counterpress.score_losses", fake_score)
    monkeypatch.setattr("pressing.report.build_report", fake_build)
    monkeypatch.setattr("pressing.demo.dump_all_panels", fake_panels)

    panels = tmp_path / "panels"
    result = _run(
        "report",
        str(tmp_path),
        "--team",
        "away",
        "--out",
        str(tmp_path / "r.pdf"),
        "--jobs",
        "2",
        "--rank",
        "recovered",
        "--all-panels",
        str(panels),
    )
    assert result.exit_code == 0, result.output
    assert seen["jobs"] == 2
    assert seen["rank"] == "recovered"
    assert Path(seen["panels_dir"]) == panels
    assert "wrote 4 per-loss panels" in result.output


def test_report_prints_the_pdf_path_last_so_it_can_be_captured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``$(pressing report ...)`` must be the path, after all the work is done.

    The panel dump can take minutes; announcing the PDF before starting it
    reports a finished run that is not finished.
    """
    import pandas as pd

    monkeypatch.setattr("pressing.cli._load_match", lambda t, e: ("TRACKING", "EVENTS"))
    monkeypatch.setattr(
        "pressing.counterpress.score_losses",
        lambda *a, **k: pd.DataFrame({"scored": [True]}),
    )
    monkeypatch.setattr("pressing.report.build_report", lambda *a, **k: Path(a[3]))

    # Record the real call sequence. Asserting on captured stdout cannot see
    # this: the panel line goes to stderr, so a path printed first is still the
    # only line on stdout and the test would pass either way.
    order: list[str] = []
    monkeypatch.setattr(
        "pressing.demo.dump_all_panels",
        lambda *a, **k: (order.append("panels"), 1)[1],
    )
    real_echo = cli.typer.echo
    monkeypatch.setattr(
        cli.typer,
        "echo",
        lambda message="", **k: (order.append(f"stdout:{message}"), real_echo(message, **k))[1],
    )

    target = tmp_path / "r.pdf"
    result = _run(
        "report",
        str(tmp_path),
        "--team",
        "home",
        "--out",
        str(target),
        "--all-panels",
        str(tmp_path / "p"),
    )
    assert result.exit_code == 0, result.output
    assert order == ["panels", f"stdout:{target}"]
    # And stdout carries the path and nothing else.
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert lines == [str(target)]


def test_demo_reports_a_missing_asset_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(*args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError("the bundled demo asset is missing: data/demo/x.parquet")

    monkeypatch.setattr("pressing.demo.run_demo", missing)
    result = _run("demo")
    assert result.exit_code == 1
    assert "bundled demo asset is missing" in result.output
    assert "Traceback" not in result.output
