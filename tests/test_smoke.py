"""Smoke tests: the package imports and the CLI runs."""

from __future__ import annotations

from typer.testing import CliRunner

import pressing
from pressing.cli import app

runner = CliRunner()


def test_package_imports() -> None:
    assert pressing.__version__ == "0.1.0"


def test_cli_help_exits_zero() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "pressing" in result.output


def test_cli_version_prints_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert pressing.__version__ in result.output


def test_subcommands_are_registered() -> None:
    result = runner.invoke(app, ["--help"])
    for name in ("report", "proxy", "season", "demo"):
        assert name in result.output
