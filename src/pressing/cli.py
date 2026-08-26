"""Command line interface.

Four subcommands mirror the four ways the tool gets used::

    pressing report match_tracking.csv --team home
    pressing proxy events.json
    pressing season --source statsbomb --competition "La Liga"
    pressing demo

Two of them work. ``report`` builds the six-page PDF from a real match, and
``demo`` does the same for the match bundled under ``data/demo/`` with no
network and no ``data/raw/``. ``proxy`` and ``season`` have no model behind
them yet, and say so in one line and exit :data:`NOT_BUILT` rather than raising
through a traceback: an unbuilt feature is not a crash, and it should not read
like one.

Exit codes: 0 success, 1 the run could not proceed, 2 a usage error (click's,
not ours -- a bad flag), 3 the feature is not built.

Imports here are lazy on purpose. Every analysis module pulls in matplotlib,
pandas and reportlab -- several seconds of import even warm -- and ``pressing
--help`` and ``pressing --version`` have no business paying for any of it. The
heavy imports happen inside the command that needs them, which also lets
:func:`pressing.warm_matplotlib_cache` do its job: it can only overlap the font
scan with real work if matplotlib has not already been imported at module load.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, NoReturn

import typer

if TYPE_CHECKING:  # pragma: no cover - kept out of the import path on purpose
    import pandas as pd

from pressing import __version__

app = typer.Typer(
    name="pressing",
    help="Counter-press report generator: pitch control, recovery metrics, coach-ready PDFs.",
    no_args_is_help=True,
    add_completion=False,
)

#: Exit code for a subcommand whose feature is not built yet, so a script can
#: tell "this tool does not do that" apart from "this run went wrong".
#:
#: 3, not 2. 1 is every other failure path here, and 2 is what click already
#: exits with for a usage error -- an unknown flag, a bad ``--rank``, a
#: ``--jobs 0``. Reusing 2 would have made "the feature is unbuilt"
#: indistinguishable from "you typed the command wrong", which is exactly the
#: distinction this constant exists to draw.
NOT_BUILT: int = 3

#: What click exits with on a usage error. Not ours to choose; named so the
#: tests can assert the two codes stay apart.
USAGE_ERROR: int = 2


def _not_built(what: str, why: str) -> NoReturn:
    """Report an unbuilt feature honestly and exit.

    Args:
        what: The subcommand the user asked for.
        why: What is actually missing, in plain words.

    Raises:
        typer.Exit: Always, with :data:`NOT_BUILT`.
    """
    typer.secho(f"`pressing {what}` is not built yet.", fg=typer.colors.YELLOW, err=True)
    typer.secho(why, err=True)
    typer.secho(
        "Working today: `pressing demo` and `pressing report`. See TODO.md.", err=True
    )
    raise typer.Exit(code=NOT_BUILT)


def _fail(message: str) -> NoReturn:
    """Report a run that could not proceed and exit non-zero.

    Args:
        message: What went wrong, and where possible what to do about it.

    Raises:
        typer.Exit: Always, with code 1.
    """
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _echo(message: str) -> None:
    """Write one progress line to stderr, keeping stdout for the result."""
    typer.secho(message, err=True)


def _version_callback(value: bool) -> None:
    """Print the version and exit when ``--version`` is passed.

    Args:
        value: Whether the flag was given.

    Raises:
        typer.Exit: With code 0 when ``value`` is true.
    """
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show the version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Counter-press report generator."""


# --------------------------------------------------------------------------- #
# Shared options
# --------------------------------------------------------------------------- #

JobsOption = Annotated[
    int | None,
    typer.Option(
        "--jobs",
        "-j",
        min=1,
        help=(
            "Worker processes for scoring. Defaults to min(6, cpu count). "
            "Use --jobs 1 for the serial path; the scores are identical."
        ),
    ),
]

AllPanelsOption = Annotated[
    Path | None,
    typer.Option(
        "--all-panels",
        help=(
            "Also write one pitch-control heatmap PNG per possession loss into "
            "this directory. The PDF always prints the top five moments; the "
            "metric is computed at every loss, and this dumps the rest."
        ),
    ),
]

#: The three ways to pick the five losses the moment pages print. Mirrors
#: :data:`pressing.report.MomentRank`, restated here so that ``--help`` costs no
#: import.
RankChoice = Literal["costly", "recovered", "conceded"]

RankOption = Annotated[
    RankChoice,
    typer.Option(
        "--rank",
        help=(
            "Which five losses the moment pages show. 'costly' (default): "
            "largest EPV swing, i.e. where the opponent gained the most ground. "
            "'recovered': highest recovery share. 'conceded': lowest. Note that "
            "recovery share is driven by where the ball was lost, so 'recovered' "
            "returns almost only final-third losses; 'costly' is the one that "
            "surfaces presses that went wrong."
        ),
    ),
]


def _read(reader: Callable[[Path], Any], path: Path, what: str) -> Any:
    """Read a file, turning a malformed one into a sentence rather than a stack.

    pyarrow's own message for a truncated or non-parquet file is accurate and
    completely opaque to anyone who is not already debugging pyarrow.

    Args:
        reader: Function that turns the path into a frame.
        path: File to read.
        what: What the file is meant to be, for the error message.

    Returns:
        Whatever ``reader`` returned.

    Raises:
        typer.Exit: With code 1 if the read failed.
    """
    try:
        return reader(path)
    except Exception as exc:  # noqa: BLE001 - any read failure is the user's problem
        _fail(f"could not read {path} as a {what}: {exc}")


def _load_match(tracking: Path, events: Path | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a match for ``pressing report`` from whatever the user pointed at.

    Three shapes are accepted, because three shapes exist in practice:

    * a ``.parquet`` asset written by
      :func:`pressing._io.write_tracking_parquet`, which needs ``--events``;
    * a directory holding Metrica's ``Sample_Game_N/`` folders;
    * one of Metrica's raw tracking CSVs, from which the game number and the
      directory are read off the filename.

    Args:
        tracking: What the user passed as the tracking argument.
        events: What the user passed to ``--events``, if anything.

    Returns:
        ``(tracking, events)`` as canonical frames.

    Raises:
        typer.Exit: If the path is not one of the three shapes, or a needed
            companion file is missing.
    """
    from pressing._io import load_metrica_match, read_tracking_parquet
    from pressing.demo import read_demo_events

    if not tracking.exists():
        _fail(f"no such file or directory: {tracking}")

    if tracking.is_dir():
        # Match the directory only. The same folder often also holds flattened
        # copies named Sample_Game_1__Sample_Game_1_RawTrackingData_Home_Team.csv,
        # which the glob picks up and int() then chokes on.
        games = sorted(
            int(p.name.rsplit("_", 1)[-1])
            for p in tracking.glob("Sample_Game_*")
            if p.is_dir() and p.name.rsplit("_", 1)[-1].isdigit()
        )
        if not games:
            _fail(
                f"{tracking} holds no Sample_Game_N directory. Point --tracking at the "
                "parent of Metrica's Sample_Game_1/, or at a .parquet asset."
            )
        match = load_metrica_match(games[0], tracking)
        return match.tracking, match.events

    if tracking.suffix == ".parquet":
        # Both arguments are checked before either file is opened. Reading the
        # tracking asset is seconds of work, and discovering only afterwards
        # that --events was a .json is a slow way to deliver a typo.
        if events is None:
            _fail(
                "a .parquet tracking asset carries no events, and possession losses are "
                "detected from the event stream. Pass --events."
            )
        if events.suffix != ".parquet":
            _fail(f"unsupported events file: {events}. Expected a .parquet events asset.")
        if not events.is_file():
            _fail(f"no such events file: {events}")
        frame = _read(read_tracking_parquet, tracking, "tracking asset")
        return frame, _read(read_demo_events, events, "events asset")

    name = tracking.name
    parts = name.split("_")
    # The game number has to be a number. `Sample_Game_Final.csv` reaches here
    # looking exactly like a match file, and a bare int() on it raised a
    # ValueError straight through the CLI.
    if name.startswith("Sample_Game_") and name.endswith(".csv") and len(parts) > 2:
        if not parts[2].isdigit():
            _fail(
                f"cannot read a game number out of {name}. Metrica's files are named "
                "Sample_Game_<number>_RawTrackingData_<Home|Away>_Team.csv."
            )
        game = int(parts[2])
        data_dir = (
            tracking.parent.parent
            if tracking.parent.name.startswith("Sample_Game_")
            else tracking.parent
        )
        match = load_metrica_match(game, data_dir)
        return match.tracking, match.events

    _fail(
        f"do not know how to read {tracking}. Accepted: a .parquet asset (with "
        "--events), a directory holding Metrica's Sample_Game_N/, or a Metrica "
        "Sample_Game_N_RawTrackingData_*.csv."
    )


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


@app.command()
def report(
    tracking: Annotated[Path, typer.Argument(help="Tracking file, asset or match directory.")],
    team: Annotated[str, typer.Option("--team", help="Team the report is written for.")],
    out: Annotated[Path, typer.Option("--out", help="Destination PDF.")] = Path(
        "out/report.pdf"
    ),
    events: Annotated[
        Path | None, typer.Option("--events", help="Events file, when tracking is a .parquet.")
    ] = None,
    jobs: JobsOption = None,
    rank: RankOption = "costly",
    all_panels: AllPanelsOption = None,
) -> None:
    """Build the six-page counter-press report from tracking data."""
    if team not in ("home", "away"):
        _fail(f"--team must be 'home' or 'away', got {team!r}")

    import pressing

    warm = pressing.warm_matplotlib_cache()

    from pressing.counterpress import default_jobs, score_losses

    workers = default_jobs() if jobs is None else int(jobs)
    frames, event_frame = _load_match(tracking, events)
    _echo(f"loaded {len(frames):,} tracking rows and {len(event_frame):,} events")

    losses = score_losses(frames, event_frame, jobs=workers)
    scored = int(losses["scored"].fillna(False).astype(bool).sum())
    _echo(f"scored {scored} of {len(losses)} possession losses on {workers} worker(s)")

    if warm is not None:
        warm.wait()

    from pressing.demo import dump_all_panels
    from pressing.report import build_report

    build_report(frames, losses, team, out, events=event_frame, rank=rank)

    if all_panels is not None:
        written = dump_all_panels(frames, losses, all_panels, events=event_frame)
        _echo(f"wrote {written} per-loss panels to {all_panels}")

    # Last, and on stdout: everything above went to stderr, so `$(pressing
    # report ...)` is the path and nothing else. Printing it before --all-panels
    # would have announced a finished run with minutes of work still to go.
    typer.echo(str(out))


@app.command()
def proxy(
    events: Annotated[Path, typer.Argument(help="Events file for the match.")],
    team: Annotated[
        str | None, typer.Option("--team", help="Team the report is written for.")
    ] = None,
    out: Annotated[Path, typer.Option("--out", help="Destination PDF.")] = Path(
        "out/report_proxy.pdf"
    ),
) -> None:
    """[NOT BUILT YET] Build the report from events only, as proxy estimates."""
    _not_built(
        "proxy",
        "There is no trained event-only model in this repository. The report "
        "template can already watermark itself as an estimate, but nothing "
        "produces the estimates, and printing a report full of numbers that "
        "were never computed would be worse than printing nothing.",
    )


@app.command()
def season(
    source: Annotated[str, typer.Option("--source", help="Event data source.")] = "statsbomb",
    competition: Annotated[
        str, typer.Option("--competition", help="Competition name.")
    ] = "La Liga",
    season_label: Annotated[
        str | None, typer.Option("--season", help="Season label. Defaults to the latest.")
    ] = None,
    out_dir: Annotated[
        Path, typer.Option("--out-dir", help="Directory for the csv and scatter PNG.")
    ] = Path("out"),
) -> None:
    """[NOT BUILT YET] Rank players league-wide against market values."""
    _not_built(
        "season",
        "Season mode ranks a whole competition through the event proxy, which "
        "does not exist yet, and joins market values from a source this "
        "repository has not been wired to. Neither half is built.",
    )


@app.command()
def demo(
    out: Annotated[Path, typer.Option("--out", help="Destination PDF.")] = Path(
        "out/demo_report.pdf"
    ),
    team: Annotated[
        str, typer.Option("--team", help="Team the report is written for.")
    ] = "home",
    jobs: JobsOption = None,
    rank: RankOption = "costly",
    all_panels: AllPanelsOption = None,
) -> None:
    """Run the bundled Metrica match offline and write the full report."""
    if team not in ("home", "away"):
        _fail(f"--team must be 'home' or 'away', got {team!r}")

    from pressing.demo import run_demo

    try:
        result = run_demo(
            out,
            jobs=jobs,
            team=team,
            rank=rank,
            all_panels_dir=all_panels,
            progress=_echo,
        )
    except FileNotFoundError as exc:
        _fail(str(exc))

    _echo(
        f"done in {result.seconds:.1f}s wall clock — {result.n_scored} of "
        f"{result.n_losses} losses scored on {result.jobs} worker(s)"
    )
    typer.echo(str(result.pdf))


if __name__ == "__main__":  # pragma: no cover
    app()
