"""Season mode: league-wide player rankings joined to market values.

Runs the event proxy across every match of a competition, aggregates
counter-pressing per 90 minutes per player, joins the result to Transfermarkt
market values, and writes a CSV plus a scatter PNG.

What the numbers are
--------------------
Two different quantities live in the output table and they must not be confused.

``counterpress_per_90``
    A **count**. StatsBomb tag a defensive action ``counterpress`` when it
    happens "within 5 seconds of an open play turnover" -- verbatim the same
    window this package's tracking metric uses -- so the flag is a direct,
    measured, per-player volume statistic. This is the scatter's y axis and it
    involves no model at all.

``mean_recovery_pressed`` / ``press_value_per_90``
    **Estimates**, from the event-only proxy in :mod:`pressing.proxy`. The proxy
    predicts a control share per turnover; a player is credited with the
    turnovers he actually pressed. Per the model card the proxy composes to
    ``r ~ 0.10 to 0.18`` against the tracking-derived quantity it stands in for
    -- the range, not the flattering end, because the label-quality term it
    multiplies is itself only known to lie between 0.26 and 0.47. These columns
    are weak signals and are labelled as such in the CSV header.

**Recovery is dominated by WHERE the ball was lost.** A player who presses
mostly in the final third will show a higher mean recovery than one who presses
mostly in his own build-up phase, and that difference says more about his team's
shape than about him. The table therefore carries the per-phase counts
(``pressed_build_up`` / ``pressed_middle`` / ``pressed_final_third``) next to the
mean, the CSV header repeats the caveat, and recovery values are only comparable
within a phase.

The reference scale in that header is the **proxy's own**, from
:func:`pressing.proxy.phase_reference_means` -- build-up 0.465, middle 0.489,
final third 0.541 for the shipped artefact. Up to v0.1.0 the header printed the
*tracking* metric's phase means instead (0.26 / 0.47 / 0.70 on Metrica game 1),
which these numbers do not live on: the proxy's gradient across pitch thirds is
+0.076 against the tracking metric's +0.444, so a proxy build-up value of 0.47 --
its own average -- read against a stated 0.26 looks like an extraordinary press
and is not one.

Minutes
-------
Minutes come from the lineup files' position intervals intersected with the real
per-period clock, never from an assumed 90. They are **elapsed** minutes
including stoppage time, so a player who is on for the whole match reads about
94, not 90. That is a uniform scaling and does not reorder anybody, but it is
stated here so nobody reads 93.9 as a bug.

Sources and licences
--------------------
Event data is StatsBomb open data, cached under the gitignored ``data/raw/`` and
never committed. Its licence is materially stricter than the repository README
suggests: use here is non-commercial personal analysis and no data is
redistributed. Every season output carries the attribution -- the CSV header,
the scatter caption.

Market values come from Transfermarkt through the ``dcaribou/transfermarkt-datasets``
repackaging, which is CC0-1.0. CC0 covers that repackaging, **not** the
underlying scraped rights. The primary source is the public no-auth R2 mirror;
a reduced snapshot is committed under ``data/transfermarkt/`` so the join works
offline on a fresh clone, with its date recorded in ``data/transfermarkt/NOTICE.md``.

Downloads follow the suite politeness rules: one request per second per host,
exponential backoff, everything cached and never re-fetched. An interrupted run
resumes from the cache.
"""

from __future__ import annotations

import re
import time
import unicodedata
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from . import proxy
from .phases import PHASES

__all__ = [
    "DEFAULT_MIN_MINUTES",
    "GOALKEEPER_POSITION",
    "MATCH_PLAYER_COLUMNS",
    "MIN_SCATTER_POINTS",
    "MIN_SEASON_MATCHES",
    "MatchStats",
    "SEASON_TABLE_COLUMNS",
    "SNAPSHOT_AS_OF",
    "SNAPSHOT_DIR",
    "STATSBOMB_ATTRIBUTION",
    "TRANSFERMARKT_ATTRIBUTION",
    "TRANSFERMARKT_BASE_URL",
    "TRANSFERMARKT_COMPETITIONS",
    "SeasonResult",
    "SeasonSpec",
    "aggregate_players",
    "fetch_season",
    "is_goalkeeper",
    "join_market_values",
    "last_two_tokens",
    "load_lineup",
    "load_market_values",
    "match_player_stats",
    "normalise_name",
    "period_clock_bounds",
    "resolve_season",
    "run_season",
    "season_table",
    "season_window",
    "select_matches",
    "write_scatter",
    "write_season_csv",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Below this many matches a per-90 ranking is noise rather than a table, and the
#: run warns loudly. Measured motivation: of StatsBomb's 18 La Liga seasons only
#: 2015/16 has a full 380-match league programme; the rest are 1-38 match
#: samples in which a player can appear once.
MIN_SEASON_MATCHES: Final[int] = 200

#: Below this many plotted points the scatter is an anecdote. It is still
#: written -- refusing to draw would be worse -- but the caller is warned and the
#: caption says how few points it has.
MIN_SCATTER_POINTS: Final[int] = 30

#: StatsBomb's position name for a goalkeeper. Keepers are kept in the table --
#: they are players and they have minutes -- but they are left out of the
#: scatter: a goalkeeper cannot counter-press, so a dozen of them pinned at zero
#: per 90 across the full range of market values is a cluster the eye reads as a
#: relationship when it is only a position.
GOALKEEPER_POSITION: Final[str] = "Goalkeeper"

#: Fewest minutes a player needs before a per-90 is reported at all. 270 is
#: three matches' worth. Below that the denominator dominates: one counter-press
#: in twelve minutes is 7.5 per 90, which would top any league table.
DEFAULT_MIN_MINUTES: Final[float] = 270.0

#: Public, no-auth, CC0-1.0 Cloudflare R2 mirror of ``dcaribou/transfermarkt-datasets``.
TRANSFERMARKT_BASE_URL: Final[str] = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"

#: The two tables the join needs, reduced to the columns actually used.
TRANSFERMARKT_FILES: Final[tuple[str, ...]] = ("players.csv.gz", "player_valuations.csv.gz")

#: Download cache for Transfermarkt. Gitignored, like every other raw source.
DEFAULT_TM_CACHE: Final[Path] = Path("data/raw/transfermarkt")

#: Committed offline snapshot, used when neither the cache nor the network has
#: the data. Reduced to the needed columns; see its NOTICE.md.
SNAPSHOT_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "data" / "transfermarkt"

#: Date the committed snapshot was taken from the R2 mirror.
SNAPSHOT_AS_OF: Final[str] = "2026-08-25"

#: Transfermarkt domestic-competition codes for the StatsBomb competitions this
#: mode can serve. Used only to disambiguate a name that matches more than one
#: Transfermarkt player; an unmapped competition simply gets no disambiguation
#: and its ambiguous names are dropped rather than guessed.
TRANSFERMARKT_COMPETITIONS: Final[dict[str, str]] = {
    "la liga": "ES1",
    "premier league": "GB1",
    "serie a": "IT1",
    "1. bundesliga": "L1",
    "bundesliga": "L1",
    "ligue 1": "FR1",
}

STATSBOMB_ATTRIBUTION: Final[str] = (
    "Event data: StatsBomb open data (github.com/statsbomb/open-data), used under the "
    "StatsBomb Public Data User Agreement. Non-commercial personal analysis; no data "
    "redistributed."
)

TRANSFERMARKT_ATTRIBUTION: Final[str] = (
    "Market values: Transfermarkt via dcaribou/transfermarkt-datasets. CC0-1.0 covers that "
    "repackaging, not the underlying scraped rights."
)

#: One request per second per host. Transfermarkt's mirror is a different host
#: from StatsBomb's, so it keeps its own clock.
_MIN_REQUEST_INTERVAL_S: Final[float] = 1.0
_USER_AGENT: Final[str] = "pressing/0.1 (research; contact via repository)"
_last_request_at: float = 0.0

#: Columns of the season table, in output order.
SEASON_TABLE_COLUMNS: Final[tuple[str, ...]] = (
    "rank",
    "player",
    "team",
    "position",
    "player_id",
    "matches",
    "minutes",
    "counterpress_actions",
    "counterpress_per_90",
    "pressures",
    "pressures_per_90",
    "losses_pressed",
    "losses_pressed_per_90",
    "mean_recovery_pressed",
    "press_value_per_90",
    "pressed_build_up",
    "pressed_middle",
    "pressed_final_third",
    "market_value_eur",
    "valuation_date",
    "tm_player_id",
    "tm_name",
    "join_tier",
    "join_status",
)


# --------------------------------------------------------------------------- #
# Names
# --------------------------------------------------------------------------- #

_PUNCTUATION = re.compile(r"[^a-z0-9]+")

#: Latin letters that NFKD does **not** decompose, because they are letters in
#: their own right rather than a base plus a combining mark. Without this table
#: "Odegaard" normalises to ``degaard``: the stroke-o survives decomposition,
#: fails the ASCII filter, and is deleted along with the sound it carries. That
#: still self-consistently matches a provider who spells it the same way, and
#: silently fails against one who has already ASCII-ised the name -- which is
#: exactly the case the join exists to handle.
_TRANSLITERATE: Final[dict[str, str]] = {
    "ø": "o", "đ": "d", "ð": "d", "ł": "l", "ß": "ss", "æ": "ae", "œ": "oe",
    "þ": "th", "ı": "i", "ħ": "h", "ŋ": "n", "ĸ": "k", "ſ": "s",
}


def normalise_name(text: str | None) -> str:
    """Normalise a player name for matching across two providers.

    StatsBomb stores legal names ("Francisco Alcacer Garcia") and Transfermarkt
    stores common names ("Paco Alcacer"), and the two disagree about accents,
    hyphens and apostrophes as well. Normalisation is: lowercase, transliterate
    the Latin letters NFKD cannot decompose (see :data:`_TRANSLITERATE`), NFKD
    decomposition with the combining marks dropped, every run of
    non-alphanumerics collapsed to a single space, and the ends stripped.

    Args:
        text: A name, or None.

    Returns:
        The normalised form; the empty string for None or for a name with no
        alphanumeric content.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    lowered = str(text).lower()
    if any(ch in _TRANSLITERATE for ch in lowered):
        lowered = "".join(_TRANSLITERATE.get(ch, ch) for ch in lowered)
    decomposed = unicodedata.normalize("NFKD", lowered)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _PUNCTUATION.sub(" ", stripped).strip()


def last_two_tokens(text: str | None) -> str:
    """Last two tokens of a normalised name, the third and weakest join tier.

    "Francisco Alcacer Garcia" and "Alcacer Garcia" agree here where the full
    strings do not. It is deliberately last: two tokens are much likelier to
    collide than a full name, and a collision becomes an ambiguity that gets
    dropped rather than a wrong value that gets published.

    Args:
        text: A name, or None.

    Returns:
        The last two tokens joined by a space, or the empty string when the name
        has fewer than two tokens.
    """
    tokens = normalise_name(text).split()
    return " ".join(tokens[-2:]) if len(tokens) >= 2 else ""


# --------------------------------------------------------------------------- #
# Resolving a competition and season
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SeasonSpec:
    """One resolved competition-season and the matches in it.

    Attributes:
        competition_id: StatsBomb competition id.
        season_id: StatsBomb season id.
        competition: Competition name as StatsBomb spells it.
        season: Season label, e.g. ``"2015/2016"``.
        match_ids: Every match id listed in the season manifest, ordered by
            date then id.
        matches: The manifest rows, for dates and team names.
    """

    competition_id: int
    season_id: int
    competition: str
    season: str
    match_ids: tuple[int, ...]
    matches: pd.DataFrame = field(compare=False, repr=False)

    @property
    def n_matches(self) -> int:
        """Number of matches the season manifest lists."""
        return len(self.match_ids)

    @property
    def label(self) -> str:
        """Human-readable ``"Competition Season"``."""
        return f"{self.competition} {self.season}"

    @property
    def is_thin(self) -> bool:
        """Whether this season is too small for a per-90 ranking."""
        return self.n_matches < MIN_SEASON_MATCHES


def _season_manifest(
    competition_id: int,
    season_id: int,
    cache_dir: Path | str,
    *,
    allow_download: bool,
) -> list[dict[str, Any]]:
    """Read one ``matches/{cid}/{sid}.json`` manifest.

    Args:
        competition_id: StatsBomb competition id.
        season_id: StatsBomb season id.
        cache_dir: StatsBomb cache root.
        allow_download: Whether the file may be fetched if absent.

    Returns:
        The manifest's list of match records.
    """
    import json

    path = proxy.fetch_statsbomb(
        "matches", f"{competition_id}/{season_id}.json", cache_dir,
        allow_download=allow_download,
    )
    return list(json.loads(path.read_text()))


def resolve_season(
    competition: str,
    season: str | None = None,
    cache_dir: Path | str = proxy.DEFAULT_CACHE_DIR,
    *,
    allow_download: bool = True,
) -> SeasonSpec:
    """Resolve a competition name (and optional season) to a match list.

    The match list is driven from ``competitions.json`` to
    ``matches/{cid}/{sid}.json`` and never from a listing of ``data/events/``.
    That is not a stylistic preference: 274 event files in the open-data
    repository (831.9 MB) belong to no season manifest at all, and a directory
    scan would silently fold matches from other competitions into the aggregate.

    When ``season`` is omitted the season with the **most matches** is chosen,
    not the most recent one. La Liga's most recent open season is 2020/21 with
    35 matches; its largest is 2015/16 with 380. Defaulting to "latest" would
    hand back a sample in which a per-90 ranking is noise.

    Args:
        competition: Competition name, matched case-insensitively. A unique
            substring works, e.g. ``"la liga"``.
        season: Season label such as ``"2015/2016"``. Also matched
            case-insensitively, and ``"2015/16"`` is accepted.
        cache_dir: StatsBomb cache root.
        allow_download: When False, work only from what is already cached.

    Returns:
        The resolved :class:`SeasonSpec`.

    Raises:
        LookupError: If the competition or season cannot be matched, with the
            available options listed.
    """
    import json

    path = proxy.fetch_statsbomb(
        "", "competitions.json", cache_dir, allow_download=allow_download
    )
    rows = list(json.loads(path.read_text()))

    wanted = competition.strip().lower()
    hits = [r for r in rows if str(r["competition_name"]).lower() == wanted]
    if not hits:
        hits = [r for r in rows if wanted in str(r["competition_name"]).lower()]
    if not hits:
        names = sorted({str(r["competition_name"]) for r in rows})
        raise LookupError(
            f"no competition matching {competition!r}. Available: {', '.join(names)}"
        )
    if len({int(r["competition_id"]) for r in hits}) > 1:
        names = sorted({str(r["competition_name"]) for r in hits})
        raise LookupError(
            f"{competition!r} is ambiguous: {', '.join(names)}. Pass the full name."
        )

    if season is not None:
        key = season.strip().lower().replace(" ", "")
        chosen = [r for r in hits if str(r["season_name"]).lower().replace(" ", "") == key]
        if not chosen:
            # Accept the "2015/16" short form for "2015/2016".
            chosen = [
                r for r in hits
                if _short_season(str(r["season_name"])) == _short_season(key)
            ]
        if not chosen:
            labels = sorted(str(r["season_name"]) for r in hits)
            raise LookupError(
                f"no season {season!r} in {hits[0]['competition_name']}. "
                f"Available: {', '.join(labels)}"
            )
        candidates = chosen
    else:
        candidates = hits

    best: tuple[int, dict[str, Any], list[dict[str, Any]]] | None = None
    for row in candidates:
        manifest = _season_manifest(
            int(row["competition_id"]), int(row["season_id"]), cache_dir,
            allow_download=allow_download,
        )
        if best is None or len(manifest) > best[0]:
            best = (len(manifest), row, manifest)
    if best is None:  # pragma: no cover - `candidates` is never empty here
        raise LookupError(f"no seasons found for {competition!r}")

    _, row, manifest = best
    frame = pd.DataFrame(
        [
            {
                "match_id": int(m["match_id"]),
                "match_date": str(m.get("match_date") or ""),
                "home_team": str(m["home_team"]["home_team_name"]),
                "away_team": str(m["away_team"]["away_team_name"]),
            }
            for m in manifest
        ]
    ).sort_values(["match_date", "match_id"]).reset_index(drop=True)

    return SeasonSpec(
        competition_id=int(row["competition_id"]),
        season_id=int(row["season_id"]),
        competition=str(row["competition_name"]),
        season=str(row["season_name"]),
        match_ids=tuple(int(v) for v in frame["match_id"]),
        matches=frame,
    )


def _short_season(label: str) -> str:
    """Collapse ``"2015/2016"`` and ``"2015/16"`` to one comparable form.

    Args:
        label: A season label.

    Returns:
        ``"start/last-two-digits"``, or the input lowercased when it does not
        look like a two-part season label.
    """
    text = label.strip().lower().replace(" ", "")
    if "/" not in text:
        return text
    head, tail = text.split("/", 1)
    return f"{head}/{tail[-2:]}"


def select_matches(match_ids: Sequence[int], limit: int | None) -> tuple[int, ...]:
    """Take an evenly spread subset of a season's matches.

    A bounded run exists so the command is testable and demoable without a
    six-and-a-half-minute fetch. Taking the *first* ``limit`` matches would
    sample one end of the calendar -- an August-only table, with squads that had
    not settled and half the January signings absent. Striding evenly across the
    season is still a sample, and the run says so, but it is not a seasonal one.

    Args:
        match_ids: Every match id in the season, in calendar order.
        limit: How many to keep, or None for all of them.

    Returns:
        The selected ids, in the input order.
    """
    ids = tuple(int(v) for v in match_ids)
    if limit is None or limit >= len(ids) or limit <= 0:
        return ids
    picks = np.linspace(0, len(ids) - 1, num=int(limit))
    return tuple(ids[int(round(float(p)))] for p in picks)


def fetch_season(
    match_ids: Sequence[int],
    cache_dir: Path | str = proxy.DEFAULT_CACHE_DIR,
    *,
    allow_download: bool = True,
    progress: Callable[[str], None] | None = None,
) -> list[tuple[int, Path, Path]]:
    """Cache the events and lineups files for a list of matches.

    Both files are needed: events carry the counter-pressing actions, lineups
    carry the minutes and the ``player_nickname`` on which the whole
    Transfermarkt join turns. 360 files are deliberately **not** fetched -- the
    proxy runs on events alone, and 360 files average 7.5 MB.

    Every file is cached by match id and never re-fetched, so an interrupted run
    resumes for free. A match that fails is logged and skipped rather than
    aborting the pull.

    Args:
        match_ids: Matches to cache.
        cache_dir: StatsBomb cache root.
        allow_download: When False, use only what is already cached.
        progress: Optional line-oriented progress sink.

    Returns:
        ``(match_id, events_path, lineups_path)`` for every match that ended up
        fully available.
    """
    say = progress or (lambda _message: None)
    out: list[tuple[int, Path, Path]] = []
    total = len(match_ids)
    for i, mid in enumerate(match_ids, start=1):
        name = f"{int(mid)}.json"
        try:
            events = proxy.fetch_statsbomb(
                "events", name, cache_dir, allow_download=allow_download
            )
            lineups = proxy.fetch_statsbomb(
                "lineups", name, cache_dir, allow_download=allow_download
            )
        except Exception as exc:  # noqa: BLE001 - one bad match must not stop a season
            say(f"[{i}/{total}] skipped {mid}: {exc}")
            continue
        out.append((int(mid), events, lineups))
        if i % 20 == 0 or i == total:
            say(f"[{i}/{total}] cached")
    return out


# --------------------------------------------------------------------------- #
# Minutes
# --------------------------------------------------------------------------- #


def _clock_seconds(text: str) -> float:
    """Parse a lineup ``MM:SS`` (or ``HH:MM:SS``) clock reading into seconds.

    Args:
        text: The clock string as the lineup file stores it.

    Returns:
        Seconds on the match clock.

    Raises:
        ValueError: If the string is not two or three colon-separated numbers.
    """
    parts = str(text).split(":")
    if len(parts) == 2:
        return float(parts[0]) * 60.0 + float(parts[1])
    if len(parts) == 3:
        return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
    raise ValueError(f"malformed lineup clock: {text!r}")


def period_clock_bounds(raw_events: Iterable[dict[str, Any]]) -> dict[int, tuple[float, float]]:
    """Per-period start and end on the MATCH CLOCK, in seconds.

    The lineup files timestamp substitutions on the match clock -- a half-time
    substitution reads ``"45:00"`` with ``from_period`` 2 -- which is the same
    clock as the events' ``minute`` / ``second`` and *not* the period-relative
    ``timestamp``. So minutes are computed on the clock, and this function reads
    the clock bounds off ``Half Start`` / ``Half End`` rather than assuming a
    period is 45 minutes: the observed second period of a real match runs from
    45:00 to about 92:59.

    The two clocks overlap (period 1 can end at 45:57 while period 2 starts at
    45:00), so an interval must be intersected with its own period's bounds and
    the pieces summed, never differenced end-to-end.

    Args:
        raw_events: Raw StatsBomb event dicts.

    Returns:
        ``{period: (start_s, end_s)}``. Periods with no ``Half Start`` or
        ``Half End`` fall back to the smallest and largest clock reading seen in
        that period.
    """
    seen: dict[int, list[float]] = {}
    starts: dict[int, float] = {}
    ends: dict[int, float] = {}
    for event in raw_events:
        period = int(event["period"])
        clock = float(event.get("minute", 0)) * 60.0 + float(event.get("second", 0))
        seen.setdefault(period, []).append(clock)
        name = str(event["type"]["name"])
        if name == "Half Start":
            starts[period] = min(starts.get(period, clock), clock)
        elif name == "Half End":
            ends[period] = max(ends.get(period, clock), clock)
    return {
        period: (starts.get(period, min(clocks)), ends.get(period, max(clocks)))
        for period, clocks in sorted(seen.items())
    }


def _interval_minutes(
    positions: Sequence[dict[str, Any]],
    bounds: dict[int, tuple[float, float]],
) -> float:
    """Minutes played, from a lineup player's position intervals.

    Each interval is clipped to the bounds of every period it spans and the
    pieces are summed, which handles the three shapes that occur: a starter
    (``to`` null), a substitute (``from`` on the clock in a later period), and a
    player whose position changed at a tactical shift (two contiguous intervals
    that must not be double counted -- they share an endpoint, so summing
    clipped lengths is exact).

    Args:
        positions: The lineup entry's ``positions`` list.
        bounds: Per-period clock bounds from :func:`period_clock_bounds`.

    Returns:
        Minutes on the pitch, elapsed and including stoppage time.
    """
    if not bounds:
        return 0.0
    last_period = max(bounds)
    total = 0.0
    for spell in positions:
        if spell.get("from") is None:
            continue
        from_period = int(spell.get("from_period") or 1)
        to_period = int(spell["to_period"]) if spell.get("to_period") else last_period
        start = _clock_seconds(spell["from"])
        end = _clock_seconds(spell["to"]) if spell.get("to") else None
        for period in range(from_period, to_period + 1):
            if period not in bounds:
                continue
            lo, hi = bounds[period]
            piece_lo = max(lo, start) if period == from_period else lo
            piece_hi = hi if end is None or period != to_period else min(hi, end)
            total += max(0.0, piece_hi - piece_lo)
    return total / 60.0


def load_lineup(
    lineups_path: Path | str,
    bounds: dict[int, tuple[float, float]],
) -> pd.DataFrame:
    """Read one lineups file into per-player rows with minutes.

    Args:
        lineups_path: Path to ``data/lineups/{match_id}.json``.
        bounds: Per-period clock bounds from :func:`period_clock_bounds` for the
            same match.

    Returns:
        One row per player named in the lineup, with ``player_id``, ``player``,
        ``nickname``, ``team``, ``team_id`` and ``minutes``. Unused substitutes
        appear with zero minutes.
    """
    import json

    payload = json.loads(Path(lineups_path).read_text())
    rows: list[dict[str, Any]] = []
    for team in payload:
        for entry in team.get("lineup", []):
            spells = entry.get("positions") or []
            rows.append(
                {
                    "player_id": int(entry["player_id"]),
                    "player": str(entry["player_name"]),
                    "nickname": entry.get("player_nickname"),
                    "team": str(team["team_name"]),
                    "team_id": int(team["team_id"]),
                    "position": str(spells[0]["position"]) if spells else None,
                    "minutes": _interval_minutes(spells, bounds),
                }
            )
    return pd.DataFrame(
        rows,
        columns=["player_id", "player", "nickname", "team", "team_id", "position", "minutes"],
    )


def is_goalkeeper(position: Any) -> bool:
    """Whether a lineup position is the goalkeeper's.

    Args:
        position: A StatsBomb position name, or anything else.

    Returns:
        True for ``"Goalkeeper"``.
    """
    return str(position) == GOALKEEPER_POSITION


# --------------------------------------------------------------------------- #
# One match
# --------------------------------------------------------------------------- #

#: Per-player columns produced for one match, before the season aggregate.
MATCH_PLAYER_COLUMNS: Final[tuple[str, ...]] = (
    "match_id",
    "player_id",
    "player",
    "nickname",
    "team",
    "team_id",
    "position",
    "minutes",
    "counterpress_actions",
    "pressures",
    "losses_pressed",
    "recovery_sum",
    "press_value",
    "pressed_build_up",
    "pressed_middle",
    "pressed_final_third",
)


@dataclass(frozen=True)
class MatchStats:
    """One match's contribution to the season table.

    Attributes:
        players: Per-player rows, :data:`MATCH_PLAYER_COLUMNS`.
        n_turnovers: Open-play turnovers detected.
        n_counterpress: Counter-pressing actions tagged by StatsBomb.
        n_unmatched_actions: Actions whose player is not in the lineup file, and
            which are therefore credited to nobody. Reported rather than hidden.
    """

    players: pd.DataFrame
    n_turnovers: int
    n_counterpress: int
    n_unmatched_actions: int


def match_player_stats(
    events_path: Path | str,
    lineups_path: Path | str,
    model: proxy.ProxyModel | None = None,
    *,
    window_s: float = 5.0,
) -> MatchStats:
    """Per-player counter-pressing for one match.

    Three things happen here.

    1. **Volume.** Every event the provider flagged ``counterpress`` is counted
       against the player who made it. StatsBomb define that flag as a pressing
       action within five seconds of an open-play turnover, which is exactly
       this package's window, so the count needs no model and no interpretation.
    2. **Attribution to turnovers.** Each flagged action is assigned to the most
       recent open-play turnover whose five-second window contains it and whose
       losing team is the acting player's team. Actions that fall in no such
       window are still counted as volume but earn no value -- the two counts
       are reported separately so the gap is visible.
    3. **Value.** The proxy predicts a control share for every turnover; a
       player's ``press_value`` is the sum over the turnovers he pressed of
       ``recovery - 0.5``, i.e. control won above dead even. This is an
       estimate, and a weak one: see the model card.

    Args:
        events_path: Path to the match's events file.
        lineups_path: Path to the match's lineups file.
        model: Fitted proxy model, or None to skip the value columns.
        window_s: Counter-press window in seconds.

    Returns:
        The match's :class:`MatchStats`.
    """
    import json

    raw = json.loads(Path(events_path).read_text())
    bounds = period_clock_bounds(raw)
    lineup = load_lineup(lineups_path, bounds)

    events = proxy.load_statsbomb_events(events_path)
    turnovers = proxy.find_turnovers(events)

    recovery: dict[str, float] = {}
    if model is not None and not turnovers.empty:
        features = proxy.build_features(events, turnovers, window_s=window_s)
        if not features.empty:
            predicted = model.predict(features)
            recovery = dict(zip(features["turnover_id"], predicted, strict=True))

    phase_of: dict[str, str | None] = {}
    if not turnovers.empty:
        phase_of = dict(zip(turnovers["turnover_id"], turnovers["phase"], strict=True))

    # Volume: pressures and counter-pressing actions, per player.
    acted = events[events["player"].notna()]
    counts = (
        acted.assign(
            is_cp=acted["counterpress"].astype(bool),
            is_pressure=acted["type"].eq("Pressure"),
        )
        .groupby(["team", "player"], sort=False)[["is_cp", "is_pressure"]]
        .sum()
        .reset_index()
        .rename(columns={"is_cp": "counterpress_actions", "is_pressure": "pressures"})
    )

    # Attribution: each flagged action to the turnover whose window holds it.
    per_player: dict[tuple[str, str], dict[str, Any]] = {}
    flagged = acted[acted["counterpress"].astype(bool)]
    unmatched_actions = 0
    if not flagged.empty and not turnovers.empty:
        for period, block in turnovers.groupby("period", sort=False):
            starts = block["t"].to_numpy(dtype=float)
            order = np.argsort(starts)
            starts = starts[order]
            rows = block.iloc[order]
            same_period = flagged[flagged["period"] == period]
            for action in same_period.itertuples(index=False):
                j = int(np.searchsorted(starts, float(action.t), side="right")) - 1
                if j < 0:
                    continue
                turn = rows.iloc[j]
                if float(action.t) > float(turn["t"]) + window_s:
                    continue
                if str(action.team) != str(turn["losing_team"]):
                    continue
                key = (str(action.team), str(action.player))
                slot = per_player.setdefault(
                    key,
                    {"turnovers": set(), "phases": {p: set() for p in PHASES}},
                )
                slot["turnovers"].add(str(turn["turnover_id"]))
                phase = phase_of.get(str(turn["turnover_id"]))
                if phase in slot["phases"]:
                    slot["phases"][phase].add(str(turn["turnover_id"]))

    attributed: list[dict[str, Any]] = []
    for (team, player), slot in per_player.items():
        ids = slot["turnovers"]
        values = [recovery[t] for t in ids if t in recovery]
        attributed.append(
            {
                "team": team,
                "player": player,
                "losses_pressed": len(ids),
                "recovery_sum": float(np.sum(values)) if values else np.nan,
                "press_value": float(np.sum([v - 0.5 for v in values])) if values else np.nan,
                "n_valued": len(values),
                **{f"pressed_{p}": len(slot["phases"][p]) for p in PHASES},
            }
        )
    attributed_frame = pd.DataFrame(
        attributed,
        columns=["team", "player", "losses_pressed", "recovery_sum", "press_value", "n_valued"]
        + [f"pressed_{p}" for p in PHASES],
    )

    table = lineup.merge(counts, on=["team", "player"], how="left")
    table = table.merge(attributed_frame, on=["team", "player"], how="left")
    table["match_id"] = int(Path(events_path).stem)
    for column in (
        "counterpress_actions", "pressures", "losses_pressed", "n_valued",
        *[f"pressed_{p}" for p in PHASES],
    ):
        table[column] = pd.to_numeric(table[column], errors="coerce").fillna(0.0)

    # Actions credited to a player the lineup file does not list. StatsBomb's
    # own files are consistent here, but a silent zero would be indistinguishable
    # from a player who simply did not press.
    known = set(zip(lineup["team"], lineup["player"], strict=True))
    if not flagged.empty:
        unmatched_actions = int(
            sum(1 for row in flagged.itertuples(index=False)
                if (str(row.team), str(row.player)) not in known)
        )

    return MatchStats(
        players=table.reindex(columns=list(MATCH_PLAYER_COLUMNS) + ["n_valued"]),
        n_turnovers=int(len(turnovers)),
        n_counterpress=int(events["counterpress"].astype(bool).sum()),
        n_unmatched_actions=unmatched_actions,
    )


def aggregate_players(
    per_match: Iterable[pd.DataFrame],
    *,
    min_minutes: float = DEFAULT_MIN_MINUTES,
) -> pd.DataFrame:
    """Aggregate per-match player rows into the season table.

    Per-90 rates use the summed real minutes from the lineup files. A player
    below ``min_minutes`` is kept in the table but his rate columns are NaN: one
    counter-press in twelve minutes is 7.5 per 90 and would head any ranking.

    Args:
        per_match: Per-match frames from :func:`match_player_stats`.
        min_minutes: Minutes needed before a per-90 is reported.

    Returns:
        One row per player, sorted by ``counterpress_per_90`` descending.
    """
    frames = [f for f in per_match if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(columns=list(SEASON_TABLE_COLUMNS))

    stacked = pd.concat(frames, ignore_index=True)
    sums = (
        stacked.groupby("player_id", sort=False)
        .agg(
            matches=("match_id", "nunique"),
            minutes=("minutes", "sum"),
            counterpress_actions=("counterpress_actions", "sum"),
            pressures=("pressures", "sum"),
            losses_pressed=("losses_pressed", "sum"),
            recovery_sum=("recovery_sum", "sum"),
            press_value=("press_value", "sum"),
            n_valued=("n_valued", "sum"),
            pressed_build_up=("pressed_build_up", "sum"),
            pressed_middle=("pressed_middle", "sum"),
            pressed_final_third=("pressed_final_third", "sum"),
        )
        .reset_index()
    )

    # Identity: the name and team the player appeared under most often. A loan in
    # January genuinely produces two teams for one player; the modal one is the
    # only defensible single answer and the table says nothing more than that.
    identity = (
        stacked.groupby(["player_id", "player", "team"], sort=False)
        .agg(spells=("match_id", "nunique"), nickname=("nickname", "first"))
        .reset_index()
        .sort_values(["player_id", "spells"], ascending=[True, False])
        .drop_duplicates("player_id")
    )
    table = sums.merge(
        identity[["player_id", "player", "team", "nickname"]], on="player_id", how="left"
    )

    # Position: the one he played the most minutes in. A defender who finished a
    # match up front is a defender.
    positions = (
        stacked.groupby(["player_id", "position"], dropna=False)["minutes"]
        .sum()
        .reset_index()
        .sort_values(["player_id", "minutes"], ascending=[True, False])
        .drop_duplicates("player_id")
    )
    table = table.merge(positions[["player_id", "position"]], on="player_id", how="left")

    minutes = table["minutes"].to_numpy(dtype=float)
    enough = (minutes >= float(min_minutes)) & (minutes > 0.0)
    per_90 = np.where(enough, minutes / 90.0, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        table["counterpress_per_90"] = table["counterpress_actions"].to_numpy(float) / per_90
        table["pressures_per_90"] = table["pressures"].to_numpy(float) / per_90
        table["losses_pressed_per_90"] = table["losses_pressed"].to_numpy(float) / per_90
        table["press_value_per_90"] = table["press_value"].to_numpy(float) / per_90
        valued = table["n_valued"].to_numpy(dtype=float)
        table["mean_recovery_pressed"] = np.where(
            valued > 0, table["recovery_sum"].to_numpy(float) / np.where(valued > 0, valued, 1),
            np.nan,
        )

    table = table.sort_values(
        ["counterpress_per_90", "counterpress_actions"], ascending=False, na_position="last"
    ).reset_index(drop=True)
    table["rank"] = np.where(
        table["counterpress_per_90"].notna(), np.arange(1, len(table) + 1), np.nan
    )
    # Counts are counts. Written as floats they read as "4.0 counter-presses",
    # which invites a reader to wonder what the fraction means.
    for column in (
        "rank", "matches", "counterpress_actions", "pressures", "losses_pressed",
        *[f"pressed_{p}" for p in PHASES],
    ):
        table[column] = table[column].astype("Int64")
    return table


# --------------------------------------------------------------------------- #
# Transfermarkt
# --------------------------------------------------------------------------- #


def season_window(season: str) -> tuple[str, str]:
    """ISO date window a season's market values should be read from.

    Args:
        season: Season label, ``"2015/2016"``, ``"2015/16"`` or ``"2018"``.

    Returns:
        ``(start, end)`` as ISO dates. A split season runs 1 July to 30 June; a
        single-year season is the calendar year.

    Raises:
        ValueError: If no four-digit year can be read from the label.
    """
    years = re.findall(r"\d{4}", str(season))
    if not years:
        raise ValueError(f"cannot read a year from season label {season!r}")
    start_year = int(years[0])
    if "/" in str(season):
        return f"{start_year}-07-01", f"{start_year + 1}-06-30"
    return f"{start_year}-01-01", f"{start_year}-12-31"


def _polite_sleep() -> None:
    """Block until at least one second has passed since the last download."""
    global _last_request_at
    wait = _MIN_REQUEST_INTERVAL_S - (time.monotonic() - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.monotonic()


def _download(url: str, dest: Path, *, retries: int = 5) -> Path:
    """Download one file to ``dest``, politely and atomically.

    Args:
        url: Source URL.
        dest: Destination path; parents are created.
        retries: Attempts before giving up.

    Returns:
        ``dest``.

    Raises:
        OSError: If every attempt fails.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    last: Exception | None = None
    for attempt in range(retries):
        try:
            _polite_sleep()
            with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310
                payload = response.read()
            if not payload:
                raise OSError(f"empty response for {url}")
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(payload)
            tmp.replace(dest)
            return dest
        except (urllib.error.URLError, OSError, TimeoutError) as exc:  # pragma: no cover - net
            last = exc
            if attempt == retries - 1:
                break
            time.sleep(2.0**attempt)
    raise OSError(f"failed to fetch {url} after {retries} attempts: {last}")


def _transfermarkt_paths(
    cache_dir: Path | str | None,
    *,
    allow_download: bool,
    snapshot_dir: Path | str | None = None,
) -> tuple[Path, Path, str]:
    """Locate the two Transfermarkt tables, preferring what is already local.

    The order is cache, then network, then the committed snapshot. The snapshot
    is last because it is by construction the stalest of the three, and a run
    that silently used a months-old file when fresher data was already on disk
    would be reporting an old market value as a current one.

    Args:
        cache_dir: Download cache. Defaults to ``data/raw/transfermarkt``.
        allow_download: Whether the network may be used.
        snapshot_dir: Committed snapshot directory. Defaults to
            :data:`SNAPSHOT_DIR`.

    Returns:
        ``(players_path, valuations_path, source)`` where ``source`` is
        ``"cache"``, ``"download"`` or ``"snapshot"``.

    Raises:
        FileNotFoundError: If none of the three routes yields both files.
    """
    root = Path(cache_dir) if cache_dir is not None else DEFAULT_TM_CACHE
    cached = [root / name for name in TRANSFERMARKT_FILES]
    if all(p.exists() and p.stat().st_size > 0 for p in cached):
        return cached[0], cached[1], "cache"

    if allow_download:
        try:
            for name in TRANSFERMARKT_FILES:
                _download(f"{TRANSFERMARKT_BASE_URL}/{name}", root / name)
            return cached[0], cached[1], "download"
        except OSError:  # pragma: no cover - network
            pass

    snapshot_root = Path(snapshot_dir) if snapshot_dir is not None else SNAPSHOT_DIR
    shipped = [snapshot_root / name for name in TRANSFERMARKT_FILES]
    if all(p.exists() and p.stat().st_size > 0 for p in shipped):
        return shipped[0], shipped[1], "snapshot"

    raise FileNotFoundError(
        "no Transfermarkt data: nothing cached under "
        f"{root}, no network, and no committed snapshot under {snapshot_root}"
    )


def load_market_values(
    cache_dir: Path | None = None,
    *,
    window: tuple[str, str] | None = None,
    allow_download: bool = True,
    snapshot_dir: Path | str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Load Transfermarkt market values for a season window.

    Tries the download cache first, then the public mirror, then the snapshot
    committed under ``data/transfermarkt/``.

    Within the window a player can carry several valuations -- Transfermarkt
    re-values roughly twice a season -- and the **latest** one in the window is
    taken, i.e. what the player was worth by the end of that season. Players
    with no valuation inside the window are kept with a null value so that "we
    could not price him" stays distinguishable from "he is not in the data".

    Args:
        cache_dir: Override for the download cache. Defaults to
            ``data/raw/transfermarkt``.
        window: ``(start, end)`` ISO dates from :func:`season_window`. None
            takes each player's most recent valuation of all time.
        allow_download: Whether the mirror may be contacted.
        snapshot_dir: Override for the committed snapshot directory.

    Returns:
        ``(values, as_of)``. ``values`` has one row per Transfermarkt player
        with ``tm_player_id``, ``player``, ``team``, ``market_value_eur``,
        ``valuation_date``, ``valuation_competition`` (the player's competition
        at the time of that valuation) and ``current_competition``. ``as_of`` is
        the most recent valuation date anywhere in the file, which is the real
        currency of the data rather than the date it was downloaded.
    """
    players_path, valuations_path, source = _transfermarkt_paths(
        cache_dir, allow_download=allow_download, snapshot_dir=snapshot_dir
    )

    players = pd.read_csv(
        players_path,
        usecols=lambda c: c
        in {
            "player_id", "name", "first_name", "last_name", "current_club_name",
            "current_club_domestic_competition_id", "last_season",
        },
    )
    valuations = pd.read_csv(
        valuations_path,
        usecols=lambda c: c
        in {"player_id", "date", "market_value_in_eur", "player_club_domestic_competition_id"},
    )
    as_of = str(pd.to_datetime(valuations["date"], errors="coerce").max().date())

    picked = valuations
    if window is not None:
        start, end = window
        dates = valuations["date"].astype(str)
        picked = valuations[(dates >= start) & (dates <= end)]
    picked = (
        picked.sort_values(["player_id", "date"])
        .drop_duplicates("player_id", keep="last")
        .rename(
            columns={
                "market_value_in_eur": "market_value_eur",
                "date": "valuation_date",
                "player_club_domestic_competition_id": "valuation_competition",
            }
        )
    )

    values = players.merge(
        picked[["player_id", "market_value_eur", "valuation_date", "valuation_competition"]],
        on="player_id",
        how="left",
    ).rename(
        columns={
            "player_id": "tm_player_id",
            "name": "player",
            "current_club_name": "team",
            "current_club_domestic_competition_id": "current_competition",
        }
    )
    values.attrs["source"] = source
    values.attrs["as_of"] = as_of
    return values, as_of


def _name_index(values: pd.DataFrame) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """Build the full-name and last-two-token lookups over Transfermarkt rows.

    Args:
        values: Frame from :func:`load_market_values`.

    Returns:
        ``(full, short)`` mapping a normalised key to positional row indices.
        Both the display name and ``first last`` are indexed under ``full``,
        because Transfermarkt's ``name`` is sometimes a mononym.
    """
    full: dict[str, list[int]] = {}
    short: dict[str, list[int]] = {}
    names = values["player"].tolist()
    firsts = values["first_name"].tolist() if "first_name" in values.columns else [None] * len(
        values
    )
    lasts = values["last_name"].tolist() if "last_name" in values.columns else [None] * len(values)
    for i, (name, first, last) in enumerate(zip(names, firsts, lasts, strict=True)):
        keys = {normalise_name(name)}
        joined = normalise_name(f"{first or ''} {last or ''}")
        if joined:
            keys.add(joined)
        for key in keys:
            if key:
                full.setdefault(key, []).append(i)
        two = last_two_tokens(name)
        if two:
            short.setdefault(two, []).append(i)
    return full, short


def join_market_values(
    table: pd.DataFrame,
    values: pd.DataFrame,
    *,
    competition_code: str | None = None,
) -> pd.DataFrame:
    """Join a season table to market values by name, in three tiers.

    Measured on 527 real La Liga 2015/16 players, exact normalised full-name
    matching alone joins **16.3%**: StatsBomb store legal names and Transfermarkt
    store common ones. Adding StatsBomb's ``player_nickname`` lifts it to
    **94.5%**, and a last-two-tokens fallback to **94.9%**. The nickname field is
    the whole ballgame, which is why the lineup files are fetched at all.

    Ambiguity is resolved, never guessed. A name matching more than one
    Transfermarkt player is first filtered by the competition the candidate was
    playing in **at the time of the season's valuation**, then by his current
    domestic competition. Anything still ambiguous is dropped with
    ``join_status='ambiguous'`` and no market value: attaching the wrong price
    to a real player puts a wrong point in a published scatter, and a missing
    point is a smaller lie than a wrong one.

    Args:
        table: Season table from :func:`aggregate_players`.
        values: Market values from :func:`load_market_values`.
        competition_code: Transfermarkt competition code used for
            disambiguation, e.g. ``"ES1"``. None disables that step, so
            ambiguous names are simply dropped.

    Returns:
        ``table`` with ``tm_player_id``, ``tm_name``, ``market_value_eur``,
        ``valuation_date``, ``join_tier`` and ``join_status``. Players with no
        match keep a null value rather than being dropped.
    """
    out = table.copy()
    extra = {
        "tm_player_id": pd.NA,
        "tm_name": pd.NA,
        "market_value_eur": np.nan,
        "valuation_date": pd.NA,
        "join_tier": "none",
        "join_status": "unmatched",
    }
    for column, default in extra.items():
        out[column] = default
    if out.empty or values.empty:
        return out

    full, short = _name_index(values)
    tm_ids = values["tm_player_id"].tolist()
    tm_names = values["player"].tolist()
    tm_value = values["market_value_eur"].tolist()
    tm_date = values["valuation_date"].tolist()
    tm_val_comp = values["valuation_competition"].tolist()
    tm_cur_comp = (
        values["current_competition"].tolist()
        if "current_competition" in values.columns
        else [None] * len(values)
    )

    nicknames = out["nickname"].tolist() if "nickname" in out.columns else [None] * len(out)
    for position, (name, nickname) in enumerate(
        zip(out["player"].tolist(), nicknames, strict=True)
    ):
        tiers: list[tuple[str, list[int]]] = []
        for tier, key, index in (
            ("name", normalise_name(name), full),
            ("nickname", normalise_name(nickname), full),
            ("last_two", last_two_tokens(name), short),
        ):
            if key and key in index:
                tiers.append((tier, list(index[key])))
        if not tiers:
            continue

        tier, candidates = tiers[0]
        status = "matched"
        if len(candidates) > 1 and competition_code is not None:
            by_valuation = [
                i for i in candidates
                if str(tm_val_comp[i]).upper() == competition_code.upper()
            ]
            if len(by_valuation) == 1:
                candidates, tier = by_valuation, f"{tier}+season_club"
            else:
                by_current = [
                    i for i in candidates
                    if str(tm_cur_comp[i]).upper() == competition_code.upper()
                ]
                if len(by_current) == 1:
                    candidates, tier = by_current, f"{tier}+current_club"
        if len(candidates) != 1:
            out.iloc[position, out.columns.get_loc("join_tier")] = tier
            out.iloc[position, out.columns.get_loc("join_status")] = "ambiguous"
            continue

        i = candidates[0]
        out.iloc[position, out.columns.get_loc("tm_player_id")] = tm_ids[i]
        out.iloc[position, out.columns.get_loc("tm_name")] = tm_names[i]
        out.iloc[position, out.columns.get_loc("market_value_eur")] = tm_value[i]
        out.iloc[position, out.columns.get_loc("valuation_date")] = tm_date[i]
        out.iloc[position, out.columns.get_loc("join_tier")] = tier
        value_missing = tm_value[i] is None or (
            isinstance(tm_value[i], float) and np.isnan(tm_value[i])
        )
        out.iloc[position, out.columns.get_loc("join_status")] = (
            "no_valuation" if value_missing else status
        )
    out["tm_player_id"] = pd.to_numeric(out["tm_player_id"], errors="coerce").astype("Int64")
    out["market_value_eur"] = pd.to_numeric(out["market_value_eur"], errors="coerce")
    return out


# --------------------------------------------------------------------------- #
# The pipeline
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SeasonResult:
    """Everything one season run produced, including what it could not do.

    Attributes:
        spec: The resolved competition-season.
        table: The joined season table.
        csv_path: CSV written, or None.
        png_path: Scatter written, or None.
        n_matches_used: Matches that actually contributed events.
        n_turnovers: Open-play turnovers found across those matches.
        n_players: Players in the table.
        n_ranked: Players with enough minutes for a per-90.
        n_matched: Players joined to a Transfermarkt row.
        n_ambiguous: Players whose name matched several Transfermarkt rows and
            were dropped rather than guessed.
        n_unmatched: Players whose name matched nothing.
        n_no_valuation: Players joined but with no valuation inside the season
            window.
        n_plotted: Points the scatter actually shows.
        values_as_of: Most recent valuation date in the Transfermarkt file.
        values_source: ``"cache"``, ``"download"`` or ``"snapshot"``.
        warnings: Everything the run wants the reader to know.
    """

    spec: SeasonSpec
    table: pd.DataFrame = field(compare=False, repr=False)
    csv_path: Path | None
    png_path: Path | None
    n_matches_used: int
    n_turnovers: int
    n_players: int
    n_ranked: int
    n_matched: int
    n_ambiguous: int
    n_unmatched: int
    n_no_valuation: int
    n_plotted: int
    values_as_of: str
    values_source: str
    warnings: tuple[str, ...] = ()

    @property
    def join_rate(self) -> float:
        """Fraction of players carrying a usable market value, end to end."""
        return (self.n_matched / self.n_players) if self.n_players else 0.0


def season_table(
    source: str = "statsbomb",
    competition: str = "La Liga",
    season: str | None = None,
    *,
    limit: int | None = None,
    cache_dir: Path | str = proxy.DEFAULT_CACHE_DIR,
    tm_cache_dir: Path | None = None,
    snapshot_dir: Path | str | None = None,
    allow_download: bool = True,
    min_minutes: float = DEFAULT_MIN_MINUTES,
    model: proxy.ProxyModel | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, SeasonSpec, dict[str, Any]]:
    """Rank players in a competition by counter-pressing volume, priced.

    Args:
        source: Event data source key. Only ``"statsbomb"`` exists.
        competition: Competition name, e.g. ``"La Liga"``.
        season: Season label. Defaults to the season with the most matches.
        limit: Cap on matches processed; see :func:`select_matches`.
        cache_dir: StatsBomb cache root.
        tm_cache_dir: Transfermarkt cache root.
        snapshot_dir: Committed Transfermarkt snapshot directory.
        allow_download: Whether the network may be used.
        min_minutes: Minutes needed before a per-90 is reported.
        model: Fitted proxy model. Defaults to the bundled one, or to no model
            at all when the artefact is missing, in which case the estimate
            columns are null and the counts are unaffected.
        progress: Optional line-oriented progress sink.

    Returns:
        ``(table, spec, stats)``. ``stats`` carries the counts the caller has to
        print: matches used, turnovers, join outcomes and the values' date.

    Raises:
        ValueError: If ``source`` is not ``"statsbomb"``.
    """
    if source.lower() != "statsbomb":
        raise ValueError(
            f"unknown source {source!r}. Only 'statsbomb' is wired up: it is the only open "
            "source with both event data and the lineups the market-value join needs."
        )
    say = progress or (lambda _message: None)

    spec = resolve_season(competition, season, cache_dir, allow_download=allow_download)
    say(f"{spec.label}: {spec.n_matches} matches in the season manifest")

    wanted = select_matches(spec.match_ids, limit)
    if len(wanted) < spec.n_matches:
        say(f"processing {len(wanted)} of {spec.n_matches} matches, spread across the season")
    available = fetch_season(
        wanted, cache_dir, allow_download=allow_download, progress=say
    )

    if model is None:
        try:
            model = proxy.load_bundled_model()
        except FileNotFoundError:
            say("no proxy model found: the estimate columns will be empty")

    frames: list[pd.DataFrame] = []
    n_turnovers = 0
    n_counterpress = 0
    n_unmatched_actions = 0
    for i, (match_id, events_path, lineups_path) in enumerate(available, start=1):
        try:
            stats = match_player_stats(events_path, lineups_path, model)
        except Exception as exc:  # noqa: BLE001 - a malformed match must not stop a season
            say(f"skipped match {match_id}: {exc}")
            continue
        frames.append(stats.players)
        n_turnovers += stats.n_turnovers
        n_counterpress += stats.n_counterpress
        n_unmatched_actions += stats.n_unmatched_actions
        if i % 20 == 0 or i == len(available):
            say(f"[{i}/{len(available)}] scored")

    table = aggregate_players(frames, min_minutes=min_minutes)

    window = season_window(spec.season)
    values, as_of = load_market_values(
        tm_cache_dir,
        window=window,
        allow_download=allow_download,
        snapshot_dir=snapshot_dir,
    )
    code = TRANSFERMARKT_COMPETITIONS.get(spec.competition.strip().lower())
    joined = join_market_values(table, values, competition_code=code)

    status = joined["join_status"].astype(str)
    stats_out: dict[str, Any] = {
        "n_matches_used": len(frames),
        "n_matches_listed": spec.n_matches,
        "n_turnovers": n_turnovers,
        "n_counterpress_events": n_counterpress,
        "n_unmatched_actions": n_unmatched_actions,
        "n_players": int(len(joined)),
        "n_ranked": int(joined["counterpress_per_90"].notna().sum()) if not joined.empty else 0,
        "n_matched": int((status == "matched").sum()),
        "n_ambiguous": int((status == "ambiguous").sum()),
        "n_unmatched": int((status == "unmatched").sum()),
        "n_no_valuation": int((status == "no_valuation").sum()),
        "values_as_of": as_of,
        "values_source": str(values.attrs.get("source", "unknown")),
        "value_window": window,
        "tm_competition": code,
    }
    return joined, spec, stats_out


def _csv_header(
    spec: SeasonSpec,
    stats: dict[str, Any],
    model: proxy.ProxyModel | None = None,
) -> list[str]:
    """Comment lines stamped on top of the season CSV.

    Attribution is required on every season output, and the two caveats that
    matter most -- the proxy's measured weakness and the phase dependence of
    recovery -- travel with the file rather than living only in a README that
    the file will be separated from.

    The phase reference means come from :func:`pressing.proxy.phase_reference_means`,
    which reports what the *proxy* predicts. Until v0.1.0 this header printed the
    tracking metric's phase means instead, which are a different scale entirely:
    the proxy's own gradient across pitch thirds is about +0.08 where the tracking
    metric's is +0.44, so a reader who scaled a proxy value against the tracking
    numbers would read the proxy's own average as a remarkable press.

    Args:
        spec: The resolved season.
        stats: Counts from :func:`season_table`.
        model: The proxy model whose estimates the table carries, so the header
            quotes that model's own phase means rather than a recorded default.

    Returns:
        Lines, each already prefixed with ``#``.
    """
    lines = [
        f"pressing season table: {spec.label}",
        f"{stats['n_matches_used']} of {stats['n_matches_listed']} matches; "
        f"{stats['n_turnovers']} open-play turnovers",
        STATSBOMB_ATTRIBUTION,
        f"{TRANSFERMARKT_ATTRIBUTION} Values dated in {stats['value_window'][0]}.."
        f"{stats['value_window'][1]}; data current to {stats['values_as_of']} "
        f"(source: {stats['values_source']}).",
        "counterpress_actions is StatsBomb's own counterpress flag: a defensive action within "
        "5 seconds of an open-play turnover. It is a count, not a model output.",
        "mean_recovery_pressed and press_value_per_90 are event-proxy ESTIMATES. The proxy "
        "composes to r ~ 0.10 to 0.18 against the tracking-derived metric it stands in for. "
        "That is a range, not a rounding: the label's own quality is only known to lie between "
        "r = 0.26 and r = 0.47 (models/MODEL_CARD.md).",
        *proxy.recovery_phase_caveat(model),
        "the pressed_* columns give each player's phase mix, which is what makes the "
        "within-phase comparison possible.",
        "minutes are elapsed clock minutes from the lineup files, including stoppage time, so "
        "a full match reads about 94 rather than 90.",
    ]
    if stats["n_matches_used"] < MIN_SEASON_MATCHES:
        lines.append(
            f"WARNING: {stats['n_matches_used']} matches is a sample, not a season. "
            "Per-90 rates from this few matches are noise."
        )
    return [f"# {line}" for line in lines]


def write_season_csv(
    table: pd.DataFrame,
    path: Path | str,
    spec: SeasonSpec,
    stats: dict[str, Any],
    model: proxy.ProxyModel | None = None,
) -> Path:
    """Write the season table with its attribution and caveats on top.

    Args:
        table: The joined season table.
        path: Destination CSV.
        spec: The resolved season.
        stats: Counts from :func:`season_table`.
        model: The proxy model behind the estimate columns, so the header can
            quote that model's own phase means. ``None`` falls back to the means
            recorded for the shipped artefact.

    Returns:
        The path written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = [c for c in SEASON_TABLE_COLUMNS if c in table.columns]
    body = table.reindex(columns=columns).to_csv(index=False)
    destination.write_text("\n".join(_csv_header(spec, stats, model)) + "\n" + body)
    return destination


def write_scatter(
    table: pd.DataFrame,
    path: Path | str,
    spec: SeasonSpec,
    stats: dict[str, Any],
) -> tuple[Path, int]:
    """Write the counter-press per 90 against market value scatter.

    Goalkeepers are excluded and the caption says so. A goalkeeper cannot
    counter-press: keeping them in puts a row of zeros along the whole
    market-value axis, and a reader's eye turns that into a downward
    relationship that is really just a position.

    Args:
        table: The joined season table.
        path: Destination PNG.
        spec: The resolved season.
        stats: Counts from :func:`season_table`.

    Returns:
        ``(path, n_plotted)``. ``n_plotted`` is how many points the chart shows,
        which is what the minimum-points warning is judged on.
    """
    # No pyplot. `season_scatter` returns a bare Figure that was never in
    # pyplot's global registry, so `plt.close` on it would be a no-op -- while
    # the import alone poisons `sys.modules` for the whole process. The figure
    # is garbage collected when this function returns.
    from .plots import season_scatter

    plottable = table
    n_keepers = 0
    if not table.empty:
        value = pd.to_numeric(table["market_value_eur"], errors="coerce")
        metric = pd.to_numeric(table["counterpress_per_90"], errors="coerce")
        usable = value.notna() & (value > 0) & metric.notna()
        keeper = (
            table["position"].map(is_goalkeeper)
            if "position" in table.columns
            else pd.Series(False, index=table.index)
        )
        n_keepers = int((usable & keeper).sum())
        plottable = table[usable & ~keeper]
    n_plotted = int(len(plottable))

    caption = (
        f"{n_plotted} outfield players · {stats['n_matches_used']} of "
        f"{stats['n_matches_listed']} matches · counter-press = StatsBomb's 5-second flag · "
        f"{n_keepers} goalkeepers excluded (a keeper cannot counter-press) · values dated "
        f"{stats['value_window'][0]}..{stats['value_window'][1]}, data to "
        f"{stats['values_as_of']} · StatsBomb open data; Transfermarkt via "
        "dcaribou/transfermarkt-datasets (CC0-1.0 repackaging)"
    )
    if n_plotted < MIN_SCATTER_POINTS:
        caption = f"SAMPLE ONLY — {caption}"

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = season_scatter(
        plottable,
        title=f"Counter-press per 90 against market value — {spec.label}",
        caption=caption,
    )
    figure.savefig(destination, bbox_inches="tight")
    return destination, n_plotted


def run_season(
    source: str = "statsbomb",
    competition: str = "La Liga",
    season: str | None = None,
    out_dir: Path | str = Path("out"),
    *,
    limit: int | None = None,
    cache_dir: Path | str = proxy.DEFAULT_CACHE_DIR,
    tm_cache_dir: Path | None = None,
    snapshot_dir: Path | str | None = None,
    allow_download: bool = True,
    min_minutes: float = DEFAULT_MIN_MINUTES,
    model: proxy.ProxyModel | None = None,
    progress: Callable[[str], None] | None = None,
) -> SeasonResult:
    """Run season mode end to end and write both outputs.

    Args:
        source: Event data source key.
        competition: Competition name.
        season: Season label, or None for the season with the most matches.
        out_dir: Directory for the CSV and the PNG.
        limit: Cap on matches processed.
        cache_dir: StatsBomb cache root.
        tm_cache_dir: Transfermarkt cache root.
        snapshot_dir: Committed Transfermarkt snapshot directory.
        allow_download: Whether the network may be used.
        min_minutes: Minutes needed before a per-90 is reported.
        model: Fitted proxy model, or None for the bundled one.
        progress: Optional line-oriented progress sink.

    Returns:
        The :class:`SeasonResult`, including the warnings the caller should
        print.
    """
    if model is None:
        # Resolved here as well as inside season_table so that the CSV header can
        # quote THIS model's own phase means rather than a recorded default. A
        # failure is not fatal and is reported by season_table exactly as before.
        try:
            model = proxy.load_bundled_model()
        except FileNotFoundError:
            model = None

    table, spec, stats = season_table(
        source,
        competition,
        season,
        limit=limit,
        cache_dir=cache_dir,
        tm_cache_dir=tm_cache_dir,
        snapshot_dir=snapshot_dir,
        allow_download=allow_download,
        min_minutes=min_minutes,
        model=model,
        progress=progress,
    )

    stem = f"{_slug(spec.competition)}_{_slug(spec.season)}"
    csv_path = write_season_csv(
        table, Path(out_dir) / f"{stem}_counterpress.csv", spec, stats, model
    )
    png_path, n_plotted = write_scatter(
        table, Path(out_dir) / f"{stem}_counterpress_vs_value.png", spec, stats
    )

    warnings: list[str] = []
    if spec.is_thin:
        warnings.append(
            f"{spec.label} has only {spec.n_matches} matches in the open data. "
            f"A per-90 ranking needs at least {MIN_SEASON_MATCHES} matches to be worth "
            "reading; treat this table as a sample."
        )
    if stats["n_matches_used"] < spec.n_matches:
        warnings.append(
            f"only {stats['n_matches_used']} of {spec.n_matches} matches were processed, so "
            "minutes and every per-90 rate cover a fraction of the season."
        )
    if n_plotted < MIN_SCATTER_POINTS:
        warnings.append(
            f"the scatter has {n_plotted} points, below the {MIN_SCATTER_POINTS} this tool "
            "considers the minimum worth reading. It was written anyway, captioned as a sample."
        )
    if stats["n_unmatched_actions"]:
        warnings.append(
            f"{stats['n_unmatched_actions']} counter-pressing actions were credited to players "
            "absent from their own lineup file and are not in the table."
        )

    return SeasonResult(
        spec=spec,
        table=table,
        csv_path=csv_path,
        png_path=png_path,
        n_matches_used=int(stats["n_matches_used"]),
        n_turnovers=int(stats["n_turnovers"]),
        n_players=int(stats["n_players"]),
        n_ranked=int(stats["n_ranked"]),
        n_matched=int(stats["n_matched"]),
        n_ambiguous=int(stats["n_ambiguous"]),
        n_unmatched=int(stats["n_unmatched"]),
        n_no_valuation=int(stats["n_no_valuation"]),
        n_plotted=n_plotted,
        values_as_of=str(stats["values_as_of"]),
        values_source=str(stats["values_source"]),
        warnings=tuple(warnings),
    )


def _slug(text: str) -> str:
    """Filesystem-safe lowercase slug.

    Args:
        text: Any label.

    Returns:
        Lowercased, non-alphanumerics collapsed to single underscores.
    """
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
