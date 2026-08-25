"""Command line interface.

Four subcommands mirror the four ways the tool gets used::

    pressing report match_tracking.csv --team X
    pressing proxy events.json
    pressing season --source statsbomb --competition "La Liga"
    pressing demo

The analytics behind them are still stubs. Every subcommand is wired through to
the real entry point, so as each module lands the CLI starts working with no
changes here; until then a stubbed command reports what is missing and exits
non-zero rather than dumping a traceback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, NoReturn

import typer

from pressing import __version__

app = typer.Typer(
    name="pressing",
    help="Counter-press report generator: pitch control, recovery metrics, coach-ready PDFs.",
    no_args_is_help=True,
    add_completion=False,
)


def _not_ready(what: str) -> NoReturn:
    """Report an unimplemented pipeline stage and exit non-zero.

    Args:
        what: Name of the stage the user asked for.

    Raises:
        typer.Exit: Always, with code 1.
    """
    typer.secho(f"{what} is not implemented yet.", fg=typer.colors.YELLOW, err=True)
    raise typer.Exit(code=1)


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


@app.command()
def report(
    tracking: Annotated[Path, typer.Argument(help="Tracking file for the match.")],
    team: Annotated[str, typer.Option("--team", help="Team the report is written for.")],
    out: Annotated[
        Path, typer.Option("--out", help="Destination PDF.")
    ] = Path("out/report.pdf"),
    events: Annotated[
        Path | None, typer.Option("--events", help="Optional events file for turnover detection.")
    ] = None,
) -> None:
    """Build the six-page counter-press report from tracking data."""
    _not_ready("report")


@app.command()
def proxy(
    events: Annotated[Path, typer.Argument(help="Events file for the match.")],
    team: Annotated[
        str | None, typer.Option("--team", help="Team the report is written for.")
    ] = None,
    out: Annotated[
        Path, typer.Option("--out", help="Destination PDF.")
    ] = Path("out/report_proxy.pdf"),
) -> None:
    """Build the report from events only, watermarked as event-proxy estimates."""
    _not_ready("proxy")


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
    """Rank players league-wide and join Transfermarkt market values."""
    _not_ready("season")


@app.command()
def demo(
    out: Annotated[
        Path, typer.Option("--out", help="Destination PDF.")
    ] = Path("out/demo_report.pdf"),
) -> None:
    """Run the bundled Metrica match offline and write the full report."""
    _not_ready("demo")


if __name__ == "__main__":  # pragma: no cover
    app()
