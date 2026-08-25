"""Six-page PDF report builder.

The template is the product. Pages, in order:

1. Cover: match, teams, headline counter-press numbers.
2. Loss map: every possession loss, colored by recovery.
3. Phase breakdown: build-up / middle / final third, both teams.
4. Top-5 recovery moments, with pitch control heatmap panels.
5. Player table: losses pressed, mean recovery, EPV conceded.
6. Method note and data provenance.

Rendered with reportlab, which is pure Python and needs no system libraries, so
the bundled demo runs on a fresh clone with nothing but ``uv sync``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

N_PAGES: int = 6
TOP_MOMENTS: int = 5


def build_report(
    tracking: pd.DataFrame,
    losses: pd.DataFrame,
    team: str,
    out_path: Path,
    events: pd.DataFrame | None = None,
    watermark: str | None = None,
) -> Path:
    """Render the full six-page report.

    Args:
        tracking: Canonical tracking frame for the match.
        losses: Scored losses from :func:`pressing.counterpress.score_match`.
        team: Team the report is written for.
        out_path: Destination PDF path.
        events: Optional canonical events frame, used for the provenance page.
        watermark: Optional watermark text stamped on every page, e.g.
            ``"event-proxy estimates"``.

    Returns:
        The path written.
    """
    raise NotImplementedError


def cover_page(canvas: object, team: str, summary: dict[str, float]) -> None:
    """Draw the cover page.

    Args:
        canvas: Active reportlab canvas.
        team: Team the report is written for.
        summary: Headline metrics keyed by label.
    """
    raise NotImplementedError


def player_table_page(canvas: object, players: pd.DataFrame) -> None:
    """Draw the player table page.

    Args:
        canvas: Active reportlab canvas.
        players: Per-player aggregates.
    """
    raise NotImplementedError
