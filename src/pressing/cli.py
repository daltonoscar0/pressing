"""Command line interface.

Four subcommands mirror the four ways the tool gets used::

    pressing report match_tracking.csv --team home
    pressing proxy events.json
    pressing season --source statsbomb --competition "La Liga"
    pressing demo

``report`` builds the six-page PDF from a real match and ``demo`` does the same
for the match bundled under ``data/demo/`` with no network. ``proxy`` runs the
event-only model over a StatsBomb events file and writes per-turnover estimates.
``season`` runs that model across a whole competition and writes the ranking CSV
and the market-value scatter.

``proxy`` writes a CSV, not a PDF, and that is a limit rather than a shortcut.
Three of the report's six pages are pitch-control heatmaps and its player table
is built by attributing tracked positions at the moment of the loss; none of
that exists in an event stream. A PDF in the report's shape with those pages
blank would look like a report that had failed rather than a different, smaller
thing that succeeded.

A subcommand whose feature genuinely does not exist -- an event source that has
not been wired up -- says so in one line and exits :data:`NOT_BUILT` rather than
raising through a traceback: an unbuilt feature is not a crash, and it should
not read like one.

Every path a subcommand will eventually write to is checked before any work
starts, by :func:`_check_output_path`. ``pressing demo --out <a directory>`` used
to load the match, score 255 of its 256 possession losses and render five panels
and only then raise ``IsADirectoryError`` out of reportlab: 44.11 s of user CPU
spent to deliver a typo, as a 60-line traceback. A destination that cannot be
written is a usage mistake, and a usage mistake should cost one line and no
seconds -- the same command now fails in 0.10 s.

Exit codes: 0 success, 1 the run could not proceed, 2 a usage error (click's,
not ours -- a bad flag), 3 the feature is not built.

``--jobs`` is left unset by default and the tool measures the machine instead:
see :func:`pressing.adaptive_jobs` for the policy and the runs behind it. Every
line that reports a worker count also reports the measurement that produced it,
because a number that moves between two runs of the same command on the same
machine reads as a bug unless the command says what moved it. For the same
reason ``demo`` never prints its pipeline clock on its own -- that clock cannot
see the imports, which on a loaded machine have been measured at 70 s of a
150 s command -- so it prints wall clock, CPU and the load average together.

Imports here are lazy on purpose. Every analysis module pulls in matplotlib,
pandas and reportlab -- several seconds of import even warm -- and ``pressing
--help`` and ``pressing --version`` have no business paying for any of it. The
heavy imports happen inside the command that needs them, which also lets
:func:`pressing.warm_matplotlib_cache` do its job: it can only overlap the font
scan with real work if matplotlib has not already been imported at module load.
"""

from __future__ import annotations

import os
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
        "Working today: `pressing demo`, `pressing report`, `pressing proxy` and "
        "`pressing season --source statsbomb`. See TODO.md.",
        err=True,
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


def _nearest_existing(path: Path) -> Path:
    """The closest ancestor of ``path`` that exists, ``path`` itself included.

    Used to answer "could this be created?" without creating it. ``Path.parents``
    ends at ``.`` for a relative path and at ``/`` for an absolute one, both of
    which exist, so the loop always finds something.

    Args:
        path: The path being checked.

    Returns:
        The first existing path in ``[path, *path.parents]``.
    """
    for candidate in (path, *path.parents):
        if candidate.exists():
            return candidate
    return path  # pragma: no cover - only reachable if the root vanishes mid-run


def _check_output_path(
    path: Path,
    flag: str,
    kind: Literal["file", "dir"],
    example: str = "output",
) -> None:
    """Refuse a destination that cannot be written, before any work is done.

    Every write in this tool happens at the end of the run, so an unwritable
    destination is discovered after the whole pipeline has been paid for and
    surfaces as a library traceback -- ``IsADirectoryError`` out of reportlab,
    ``FileExistsError`` out of ``Path.mkdir``. None of those name the flag the
    user got wrong. The four failures checked here are the four that were
    reproducible from the command line: a directory where a file was wanted, a
    file where a directory was wanted, a parent that is a file so no directory
    can be made under it, and a destination inside somewhere unwritable.

    Existence is not required and nothing is created: every writer downstream
    does ``mkdir(parents=True, exist_ok=True)`` itself, and a command that
    creates directories and then fails for another reason leaves litter behind.

    Args:
        path: What the user passed.
        flag: The option it came from, for the message.
        kind: Whether ``path`` names a file to write or a directory to fill.
        example: A plausible filename, used only to show the shape wanted when
            a directory was passed where a file belongs. Ignored for ``"dir"``.

    Raises:
        typer.Exit: With code 1, and a sentence, if the path cannot be written.
    """
    if kind == "file" and path.is_dir():
        _fail(
            f"{flag} takes a file, and {path} is a directory. "
            f"Pass a filename, e.g. {path / example}."
        )
    if kind == "dir" and path.exists() and not path.is_dir():
        _fail(f"{flag} takes a directory, and {path} is a file.")

    container = path.parent if kind == "file" else path
    anchor = _nearest_existing(container)
    if not anchor.is_dir():
        _fail(
            f"{flag}: {anchor} is a file, not a directory, so nothing can be written "
            f"under it. Wanted to put {path} there."
        )
    if not os.access(anchor, os.W_OK | os.X_OK):
        _fail(f"{flag}: no permission to write inside {anchor}.")
    if kind == "file" and path.is_file() and not os.access(path, os.W_OK):
        _fail(f"{flag}: {path} exists and is not writable.")


def _check_input_dir(path: Path, flag: str) -> None:
    """Refuse an input directory that is really a file, before any work is done.

    A missing directory is not an error here: both callers treat it as "nothing
    cached yet" and either fetch or fall back. A directory that is a file is
    never that.

    Args:
        path: What the user passed.
        flag: The option it came from, for the message.

    Raises:
        typer.Exit: With code 1 if ``path`` exists and is not a directory.
    """
    if path.exists() and not path.is_dir():
        _fail(f"{flag} takes a directory, and {path} is a file.")


def _adaptive_note(share: float | None) -> str:
    """Say why the tool chose the worker count it chose.

    A worker count that moves between runs on the same machine looks like a bug
    unless the run says what moved it, so the measurement behind it is printed
    alongside it.

    Args:
        share: Measured share of one core, or None if nothing was measured.

    Returns:
        A short parenthetical.
    """
    if share is None:
        return "adaptive"
    return f"adaptive, CPU share {share:.2f}"


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
            "Worker processes for scoring. Left alone, the tool measures what "
            "share of a core this machine is giving it and uses min(6, cpu "
            "count) workers when there is a core to spare, or the serial path "
            "when there is not. An explicit value always wins, including "
            "--jobs 1. The scores are identical at any worker count."
        ),
    ),
]

AllPanelsOption = Annotated[
    Path | None,
    typer.Option(
        "--all-panels",
        help=(
            "Directory to also write one pitch-control heatmap PNG per "
            "possession loss into; created if missing. The PDF always prints "
            "the top five moments; the metric is computed at every loss, and "
            "this dumps the rest -- one PNG per loss, not five."
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
    tracking: Annotated[
        Path,
        typer.Argument(
            help=(
                "One of three shapes: a Metrica Sample_Game_N_RawTrackingData_*.csv, "
                "a directory holding Metrica's Sample_Game_N/, or a .parquet tracking "
                "asset (which needs --events)."
            )
        ),
    ],
    team: Annotated[
        str,
        typer.Option(
            "--team",
            help=(
                "Which side the report is written for: 'home' or 'away'. Not a team "
                "name -- tracking files do not carry one."
            ),
        ),
    ],
    out: Annotated[
        Path, typer.Option("--out", help="Destination PDF file, not a directory.")
    ] = Path("out/report.pdf"),
    events: Annotated[
        Path | None,
        typer.Option(
            "--events",
            help=(
                "Companion .parquet events asset. Required when TRACKING is a .parquet, "
                "ignored for the other two shapes, which carry their own events."
            ),
        ),
    ] = None,
    jobs: JobsOption = None,
    rank: RankOption = "costly",
    all_panels: AllPanelsOption = None,
) -> None:
    """Build the six-page counter-press report from tracking data."""
    if team not in ("home", "away"):
        _fail(f"--team must be 'home' or 'away', got {team!r}")

    # Before the match is loaded, not after the PDF is rendered.
    _check_output_path(out, "--out", "file", "report.pdf")
    if all_panels is not None:
        _check_output_path(all_panels, "--all-panels", "dir")

    import pressing

    # Probed before the font-cache child is started, not after: on a cold cache
    # that child is a busy CPU of our own making, and a probe that saw it would
    # read the machine as saturated and drop us to the serial path for it.
    share = pressing.cpu_availability() if jobs is None else None
    workers = int(jobs) if jobs is not None else pressing.adaptive_jobs(availability=share)

    warm = pressing.warm_matplotlib_cache()

    from pressing.counterpress import score_losses

    frames, event_frame = _load_match(tracking, events)
    _echo(f"loaded {len(frames):,} tracking rows and {len(event_frame):,} events")

    losses = score_losses(frames, event_frame, jobs=workers)
    scored = int(losses["scored"].fillna(False).astype(bool).sum())
    why = "--jobs" if jobs is not None else _adaptive_note(share)
    _echo(f"scored {scored} of {len(losses)} possession losses on {workers} worker(s) ({why})")

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
    events: Annotated[Path, typer.Argument(help="StatsBomb events JSON for one match.")],
    team: Annotated[
        str | None,
        typer.Option(
            "--team",
            help=(
                "Only keep turnovers this team lost. A team name spelt exactly as the "
                "events file spells it, e.g. 'Barcelona'; a name that is not in the "
                "match is rejected with the two that are."
            ),
        ),
    ] = None,
    out: Annotated[
        Path,
        typer.Option("--out", help="Destination CSV file, not a directory."),
    ] = Path("out/proxy_estimates.csv"),
    model_path: Annotated[
        Path | None,
        typer.Option(
            "--model",
            help=(
                "A .pkl.gz proxy artefact to score with. Defaults to the bundled "
                "models/proxy_gbr.pkl.gz."
            ),
        ),
    ] = None,
) -> None:
    """Estimate the counter-press metric from an events file, with no tracking.

    The output is an ESTIMATE, not a measurement: the header of the CSV states
    what the model is worth and on what scale its numbers live.
    """
    if not events.is_file():
        _fail(f"no such events file: {events}")
    if model_path is not None and model_path.is_dir():
        _fail(f"--model takes a model artefact, and {model_path} is a directory.")
    _check_output_path(out, "--out", "file", "proxy_estimates.csv")

    from pressing import proxy as proxy_model

    try:
        model = proxy_model.load_bundled_model(model_path)
    except (FileNotFoundError, TypeError) as exc:
        _fail(str(exc))

    frame = _read(proxy_model.load_statsbomb_events, events, "StatsBomb events file")
    estimates = proxy_model.predict(model, frame)
    if estimates.empty:
        _fail(
            f"{events} yielded no open-play turnovers. A file of only Starting XI and "
            "Half Start events, or a possession stream with no team changes, produces "
            "nothing to score."
        )

    if team is not None:
        teams = sorted(set(estimates["losing_team"].astype(str)))
        if team not in teams:
            _fail(f"--team {team!r} is not in this match. Teams: {', '.join(teams)}")
        estimates = estimates[estimates["losing_team"] == team]

    out.parent.mkdir(parents=True, exist_ok=True)
    # The phase caveat comes from proxy.recovery_phase_caveat, the same call
    # season.py's CSV header makes, so the two outputs cannot drift apart. Until
    # now this header hard-coded the TRACKING metric's phase means -- build-up
    # 0.26, middle 0.47, final third 0.70 -- above a column of proxy numbers
    # whose own gradient across the pitch is +0.078 against the tracking
    # metric's +0.444. A reader who used that as a scale read this model's own
    # build-up average of 0.464 as an extraordinary press.
    lines = [
        "pressing proxy estimates: counter-press control share per open-play turnover",
        _statsbomb_credit(),
        "recovery is an ESTIMATE from an event-only model, not a measurement. The model "
        "composes to r ~ 0.18 against the tracking-derived metric it stands in for; see "
        "models/MODEL_CARD.md before quoting any single value.",
        *proxy_model.recovery_phase_caveat(model),
        "epv_conceded is empty on purpose: EPV needs the opponent's control surface, "
        "which cannot be recovered from events.",
    ]
    header = [f"# {line}" for line in lines]
    out.write_text("\n".join(header) + "\n" + estimates.to_csv(index=False))

    _echo(f"{len(estimates)} open-play turnovers estimated")
    by_phase = estimates.groupby("phase", dropna=False)["recovery"].agg(["count", "mean"])
    for phase, row in by_phase.iterrows():
        _echo(f"  {str(phase):<12} n={int(row['count']):>4}  mean recovery {row['mean']:.3f}")
    _echo("recovery values are comparable only within a phase")
    typer.echo(str(out))


def _statsbomb_credit() -> str:
    """The StatsBomb attribution line stamped on every event-derived output.

    Returns:
        The credit text.
    """
    from pressing.season import STATSBOMB_ATTRIBUTION

    return STATSBOMB_ATTRIBUTION


@app.command()
def season(
    source: Annotated[
        str,
        typer.Option(
            "--source",
            help=(
                "Event data source. 'statsbomb' is the only one built and the only "
                "open one with both an event stream and the lineup files the "
                "market-value join needs; anything else exits 3 without working."
            ),
        ),
    ] = "statsbomb",
    competition: Annotated[
        str,
        typer.Option(
            "--competition",
            help=(
                "Competition name as StatsBomb's competitions.json spells it. A name "
                "that matches nothing is rejected with the full list of what is there."
            ),
        ),
    ] = "La Liga",
    season_label: Annotated[
        str | None,
        typer.Option(
            "--season",
            help=(
                "Season label, e.g. '2015/2016'. Defaults to the season with the MOST "
                "matches, not the most recent one: La Liga's latest open season has 35 "
                "matches and its largest has 380."
            ),
        ),
    ] = None,
    out_dir: Annotated[
        Path,
        typer.Option(
            "--out-dir",
            help="Directory for the csv and scatter PNG; created if missing.",
        ),
    ] = Path("out"),
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            "--matches",
            min=1,
            help=(
                "Process only this many matches, spread evenly across the season. A full "
                "La Liga 2015/16 pull is 380 events files plus 380 lineups at one request "
                "per second."
            ),
        ),
    ] = None,
    min_minutes: Annotated[
        float,
        typer.Option(
            "--min-minutes",
            min=0.0,
            help="Minutes a player needs before a per-90 rate is reported at all.",
        ),
    ] = 270.0,
    offline: Annotated[
        bool,
        typer.Option(
            "--offline",
            help=(
                "Never touch the network. Uses only what is already under "
                "--cache-dir, and fails naming what is missing rather than fetching it."
            ),
        ),
    ] = False,
    cache_dir: Annotated[
        Path,
        typer.Option(
            "--cache-dir",
            help=(
                "StatsBomb cache root, mirroring their data/ layout. Downloaded into "
                "unless --offline. No StatsBomb data is ever committed; see the licence "
                "note in the README."
            ),
        ),
    ] = Path("data/raw/statsbomb"),
    values_dir: Annotated[
        Path | None,
        typer.Option(
            "--values-dir",
            help=(
                "Directory holding players.csv.gz and player_valuations.csv.gz. Defaults to "
                "data/raw/transfermarkt, falling back to the committed snapshot."
            ),
        ),
    ] = None,
) -> None:
    """Rank a competition's players by counter-pressing and price them."""
    if source.lower() != "statsbomb":
        _not_built(
            f"season --source {source}",
            "StatsBomb is the only open source with both an event stream and the lineup "
            "files the market-value join needs. SkillCorner ships no events at all, and "
            "Metrica is three matches.",
        )

    # A season run is hundreds of HTTP requests at one per second; discovering
    # afterwards that --out-dir names a file is the most expensive typo here.
    _check_output_path(out_dir, "--out-dir", "dir")
    _check_input_dir(cache_dir, "--cache-dir")
    if not offline:
        _check_output_path(cache_dir, "--cache-dir", "dir")
    if values_dir is not None:
        _check_input_dir(values_dir, "--values-dir")

    from pressing.season import MIN_SEASON_MATCHES, run_season

    try:
        result = run_season(
            source,
            competition,
            season_label,
            out_dir,
            limit=limit,
            cache_dir=cache_dir,
            tm_cache_dir=values_dir,
            allow_download=not offline,
            min_minutes=min_minutes,
            progress=_echo,
        )
    except LookupError as exc:
        _fail(str(exc))
    except FileNotFoundError as exc:
        _fail(f"{exc}. Drop --offline to fetch what is missing.")

    _echo("")
    _echo(
        f"{result.spec.label}: {result.n_matches_used} of {result.spec.n_matches} matches, "
        f"{result.n_turnovers:,} open-play turnovers, {result.n_players} players "
        f"({result.n_ranked} with enough minutes to rank)"
    )
    _echo(
        f"market-value join: {result.n_matched} of {result.n_players} "
        f"({100.0 * result.join_rate:.1f}%) priced; {result.n_no_valuation} with no valuation "
        f"in the season window; {result.n_ambiguous} dropped as ambiguous rather than guessed; "
        f"{result.n_unmatched} not found"
    )
    _echo(
        f"values from the {result.values_source}, data current to {result.values_as_of}; "
        f"scatter has {result.n_plotted} points"
    )
    for warning in result.warnings:
        typer.secho(f"warning: {warning}", fg=typer.colors.YELLOW, err=True)
    if result.n_matches_used < MIN_SEASON_MATCHES:
        _echo("per-90 rates from a sample are noise; treat the ranking accordingly")

    typer.echo(str(result.csv_path))
    typer.echo(str(result.png_path))


@app.command()
def demo(
    out: Annotated[
        Path, typer.Option("--out", help="Destination PDF file, not a directory.")
    ] = Path("out/demo_report.pdf"),
    team: Annotated[
        str,
        typer.Option(
            "--team",
            help=(
                "Which side the report is written for: 'home' or 'away'. Not a team "
                "name -- the bundled Metrica match does not name its sides."
            ),
        ),
    ] = "home",
    jobs: JobsOption = None,
    rank: RankOption = "costly",
    all_panels: AllPanelsOption = None,
) -> None:
    """Run the bundled Metrica match offline and write the full report.

    No network and no arguments needed: the match ships under data/demo/.
    """
    if team not in ("home", "away"):
        _fail(f"--team must be 'home' or 'away', got {team!r}")

    # Before run_demo, not after. The whole point is that a bad --out costs
    # nothing rather than the entire pipeline.
    _check_output_path(out, "--out", "file", "demo_report.pdf")
    if all_panels is not None:
        _check_output_path(all_panels, "--all-panels", "dir")

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
        f"{result.n_scored} of {result.n_losses} losses scored on "
        f"{result.jobs} worker(s) ({result.jobs_source})"
    )
    # Never the pipeline clock on its own. See DemoResult.cost_line for why.
    _echo(f"cost: {result.cost_line()}")
    typer.echo(str(result.pdf))


if __name__ == "__main__":  # pragma: no cover
    app()
