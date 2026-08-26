"""The command line interface.

Three things are being pinned here. First, that the flags a user types actually
reach the functions behind them -- ``--jobs``, ``--team``, ``--rank`` and
``--all-panels`` are wired through with a stub in place of the pipeline,
because the pipeline itself is tested in ``test_demo.py`` and running it again
per flag would cost minutes to prove routing.

Second, that the two unbuilt subcommands fail like a tool and not like a bug.
``pressing proxy`` and ``pressing season`` have no model behind them, and the
requirement is a clear sentence and a distinct exit code -- never a traceback,
and never a plausible-looking PDF full of numbers nobody computed.

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


@pytest.mark.parametrize("command", ["proxy", "season"])
def test_help_says_up_front_which_subcommands_are_unbuilt(command: str) -> None:
    """Finding out only after you have typed the arguments is finding out late.

    Both of these have fully specified flags, so their help screens read like
    working commands unless the summary line says otherwise.
    """
    assert "NOT BUILT YET" in _run(command, "--help").output
    assert "NOT BUILT YET" in _run("--help").output


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


@pytest.mark.parametrize(
    ("args", "what"),
    [
        (("proxy", "events.json"), "proxy"),
        (("season", "--competition", "La Liga"), "season"),
    ],
)
def test_unbuilt_subcommands_say_so_and_exit_two(args: tuple[str, ...], what: str) -> None:
    result = _run(*args)
    assert result.exit_code == NOT_BUILT
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert f"`pressing {what}` is not built yet." in result.output
    assert "Traceback" not in result.output
    # It has to point somewhere that works, or the message is a dead end.
    assert "pressing demo" in result.output


def test_unbuilt_exit_code_is_distinct_from_a_failed_run() -> None:
    """A tool that cannot do something is not the same as a run that broke."""
    assert NOT_BUILT != 1
    assert _run("proxy", "e.json").exit_code == NOT_BUILT
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
    assert _run("season").exit_code == NOT_BUILT


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
        return DemoResult(pdf=Path(out), n_losses=256, n_scored=255, jobs=6, seconds=11.9)


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
