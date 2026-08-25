"""Bundled offline demo.

Runs the whole pipeline end to end on the Metrica sample match shipped under
``data/demo/``, with no network access and under two minutes of CPU. This is
the first thing a new user should run and the thing CI checks.
"""

from __future__ import annotations

from pathlib import Path

DEMO_DIR: Path = Path(__file__).resolve().parents[2] / "data" / "demo"
DEMO_TEAM: str = "home"


def demo_data_path() -> Path:
    """Return the directory holding the bundled demo match.

    Returns:
        Path to ``data/demo/``.
    """
    return DEMO_DIR


def run_demo(out_path: Path | None = None) -> Path:
    """Run the full pipeline on the bundled match and write the report.

    Args:
        out_path: Destination PDF path. Defaults to ``out/demo_report.pdf``.

    Returns:
        The path written.
    """
    raise NotImplementedError
