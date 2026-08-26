"""Generate the synthetic season fixtures under ``tests/fixtures/season/``.

Run with::

    uv run python tests/fixtures/make_season_fixtures.py

**Nothing here is StatsBomb data.** The StatsBomb licence forbids redistributing
their files, so the season tests run against a fake competition -- "Fake Liga",
four matches, four teams, sixteen invented players -- written in StatsBomb's
*shape*: the same event keys, the same period-relative ``timestamp`` next to a
match-absolute ``minute``, the same lineup ``positions`` intervals, the same
``matches/{cid}/{sid}.json`` manifest layout. The cache root is called
``event_cache`` rather than ``statsbomb`` so that nobody auditing ``git
ls-files`` has to open a file to establish that no provider data was committed.

The Transfermarkt side is invented too, and is built to exercise every branch of
the join: an exact name match, a nickname-only match, a last-two-tokens match, a
name that maps to two players and is resolvable by competition, a name that maps
to two players and is not, a player with no valuation inside the season window,
and a player who is not in the market-value data at all.

One file is deliberately a trap: ``events/999999.json`` belongs to no season
manifest. The real open-data repository contains 274 such orphans (831.9 MB),
and any code that enumerates ``data/events/`` instead of following the manifests
would silently fold them into a league aggregate.

The output is committed so the tests need neither this script nor a network.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
from pathlib import Path
from typing import Any, Final

ROOT: Final[Path] = Path(__file__).resolve().parent / "season"
SB_ROOT: Final[Path] = ROOT / "event_cache"
TM_ROOT: Final[Path] = ROOT / "transfermarkt"

COMPETITION_ID: Final[int] = 99
COMPETITION: Final[str] = "Fake Liga"
BIG_SEASON_ID: Final[int] = 1
BIG_SEASON: Final[str] = "2019/2020"
THIN_SEASON_ID: Final[int] = 2
THIN_SEASON: Final[str] = "2020/2021"

#: Clock readings, in seconds, of the two periods. Period 1 ends at 45:57 and
#: period 2 starts again at 45:00, exactly as StatsBomb's real files do -- the
#: two periods overlap on the clock, which is what makes minutes a per-period
#: intersection rather than a subtraction.
PERIOD_CLOCK: Final[dict[int, tuple[int, int]]] = {1: (0, 2757), 2: (2700, 5579)}

#: Possessions per period. Alternating teams, so each boundary is a turnover.
POSSESSIONS_PER_PERIOD: Final[int] = 6

#: Turnover x on the 120x80 pitch, in the WINNING team's frame. Cycled so the
#: fixture contains all three phases once mirrored into the losing team's frame:
#: 30 -> 90 final third, 60 -> 60 middle, 90 -> 30 build-up.
TURNOVER_X: Final[tuple[float, ...]] = (30.0, 60.0, 90.0)

TEAMS: Final[dict[str, int]] = {
    "Alpha FC": 101,
    "Beta CF": 102,
    "Gamma SC": 103,
    "Delta AC": 104,
}

#: ``team -> [(player_id, player_name, player_nickname)]``. The names encode
#: which join tier each player is meant to exercise; see the module docstring.
SQUADS: Final[dict[str, list[tuple[int, str, str | None]]]] = {
    "Alpha FC": [
        (9001, "Francisco Alcacero Garcia", "Paco Alcacero"),
        (9002, "Juan Manuel Iturbo Arevalos", None),
        (9003, "Rafael Alcantaro do Nascimiento", None),
        (9004, "Carlos Ambiguo Perez", None),
    ],
    "Beta CF": [
        (9005, "David Ambiguo Lopez", None),
        (9006, "Ninguno Sinvalor Ruiz", None),
        (9007, "Desconocido Nadie Gomez", None),
        (9008, "Ana Suplente Diaz", None),
    ],
    "Gamma SC": [
        (9009, "Gonzalo Primero Martin", None),
        (9010, "Hector Segundo Blanco", None),
        (9011, "Ivan Tercero Navas", None),
        (9012, "Jorge Cuarto Molina", None),
    ],
    "Delta AC": [
        (9013, "Kevin Quinto Ortega", None),
        (9014, "Luis Sexto Cabrera", None),
        (9015, "Mario Septimo Vidal", None),
        (9016, "Nestor Octavo Rey", None),
    ],
}

#: ``(match_id, date, home, away)``. Each team plays twice.
FIXTURES: Final[tuple[tuple[int, str, str, str], ...]] = (
    (700001, "2019-08-17", "Alpha FC", "Beta CF"),
    (700002, "2019-08-18", "Gamma SC", "Delta AC"),
    (700003, "2020-01-11", "Beta CF", "Gamma SC"),
    (700004, "2020-05-09", "Delta AC", "Alpha FC"),
)

#: The one match of the thin season, there to prove the too-few-matches warning.
THIN_FIXTURE: Final[tuple[int, str, str, str]] = (700005, "2020-09-12", "Alpha FC", "Gamma SC")

#: Alpha FC substitute the player at index 0 for the one at index 3 on the hour
#: of match 700001, so at least one player in the fixture has minutes that are
#: neither zero nor a full match.
SUB_MATCH: Final[int] = 700001
SUB_CLOCK: Final[str] = "60:00"


def _clock(period: int, offset_s: float) -> tuple[str, int, int]:
    """Timestamps for an event ``offset_s`` into ``period``.

    Args:
        period: Match period.
        offset_s: Seconds since that period kicked off.

    Returns:
        ``(timestamp, minute, second)``: a period-relative ``HH:MM:SS.mmm``
        string next to the match-absolute minute and second, which is precisely
        the inconsistency StatsBomb's real files carry.
    """
    total = float(offset_s)
    stamp = f"00:{int(total // 60):02d}:{total % 60:06.3f}"
    clock = PERIOD_CLOCK[period][0] + total
    return stamp, int(clock // 60), int(clock % 60)


def _event(
    index: int,
    period: int,
    offset_s: float,
    type_name: str,
    team: str,
    possession: int,
    possession_team: str,
    *,
    player: tuple[int, str] | None = None,
    location: list[float] | None = None,
    play_pattern: str = "Regular Play",
    counterpress: bool = False,
    under_pressure: bool = False,
    duration: float | None = 0.6,
) -> dict[str, Any]:
    """Build one StatsBomb-shaped event dict.

    Args:
        index: Event ordering key.
        period: Match period.
        offset_s: Seconds since the period kicked off.
        type_name: ``type.name``.
        team: Team the event belongs to.
        possession: Possession number.
        possession_team: Team credited with the possession.
        player: ``(player_id, player_name)`` of the actor, if any.
        location: ``[x, y]`` on the 120x80 pitch.
        play_pattern: ``play_pattern.name``.
        counterpress: The ``counterpress`` flag.
        under_pressure: The ``under_pressure`` flag.
        duration: Event duration in seconds.

    Returns:
        The event dict.
    """
    stamp, minute, second = _clock(period, offset_s)
    payload: dict[str, Any] = {
        "id": f"{team[:1].lower()}{period}-{index:04d}",
        "index": index,
        "period": period,
        "timestamp": stamp,
        "minute": minute,
        "second": second,
        "type": {"id": 1, "name": type_name},
        "team": {"id": TEAMS[team], "name": team},
        "possession": possession,
        "possession_team": {"id": TEAMS[possession_team], "name": possession_team},
        "play_pattern": {"id": 1, "name": play_pattern},
    }
    if player is not None:
        payload["player"] = {"id": player[0], "name": player[1]}
    if location is not None:
        payload["location"] = location
    if duration is not None:
        payload["duration"] = duration
    if counterpress:
        payload["counterpress"] = True
    if under_pressure:
        payload["under_pressure"] = True
    return payload


def build_events(match_id: int, home: str, away: str) -> list[dict[str, Any]]:
    """Build one match's event stream.

    Six possessions per period alternate between the teams, so every possession
    boundary is an open-play turnover: five per period, ten per match. Each
    turnover is followed 1.5 s later by a ``Pressure`` event flagged
    ``counterpress``, credited to one of the losing team's first three players in
    rotation, so per-player counts differ and are checkable by hand.

    Args:
        match_id: Match id, used to seed nothing -- the stream is deterministic.
        home: Home team name.
        away: Away team name.

    Returns:
        Events in file order.
    """
    events: list[dict[str, Any]] = []
    index = 1
    possession = 1
    turnover = 0

    for period in (1, 2):
        for team in (home, away):
            events.append(
                _event(index, period, 0.0, "Half Start", team, possession, home, duration=None)
            )
            index += 1

        for slot in range(POSSESSIONS_PER_PERIOD):
            team = home if slot % 2 == 0 else away
            squad = SQUADS[team]
            start = 10.0 + slot * 20.0

            if slot > 0:
                # This possession begins by winning the ball: that IS the
                # turnover, and its location is read in the winner's frame.
                loser = away if team == home else home
                x = TURNOVER_X[turnover % len(TURNOVER_X)]
                events.append(
                    _event(
                        index, period, start, "Ball Recovery", team, possession, team,
                        player=(squad[1][0], squad[1][1]), location=[x, 40.0],
                    )
                )
                index += 1
                presser = SQUADS[loser][turnover % 3]
                events.append(
                    _event(
                        index, period, start + 1.5, "Pressure", loser, possession, team,
                        player=(presser[0], presser[1]),
                        location=[120.0 - x, 40.0], counterpress=True,
                    )
                )
                index += 1
                turnover += 1

            events.append(
                _event(
                    index, period, start + 3.0, "Pass", team, possession, team,
                    player=(squad[0][0], squad[0][1]), location=[55.0, 40.0],
                    under_pressure=True,
                )
            )
            index += 1
            events.append(
                _event(
                    index, period, start + 5.0, "Ball Receipt*", team, possession, team,
                    player=(squad[2][0], squad[2][1]), location=[70.0, 40.0],
                )
            )
            index += 1
            possession += 1

        if period == 2 and match_id == SUB_MATCH:
            events.append(
                _event(
                    index, period, 900.0, "Substitution", home, possession - 1, home,
                    player=(SQUADS[home][0][0], SQUADS[home][0][1]), duration=None,
                )
            )
            index += 1

        end = PERIOD_CLOCK[period][1] - PERIOD_CLOCK[period][0]
        for team in (home, away):
            events.append(
                _event(index, period, end, "Half End", team, possession - 1, home, duration=None)
            )
            index += 1

    return events


def build_lineup(match_id: int, home: str, away: str) -> list[dict[str, Any]]:
    """Build one match's lineups file.

    Args:
        match_id: Match id. Match :data:`SUB_MATCH` carries a substitution so
            that the fixture contains a partial appearance.
        home: Home team name.
        away: Away team name.

    Returns:
        The two team records, in StatsBomb's lineup shape.
    """
    out: list[dict[str, Any]] = []
    for team in (home, away):
        entries: list[dict[str, Any]] = []
        for slot, (player_id, name, nickname) in enumerate(SQUADS[team]):
            if slot < 3:
                spell: dict[str, Any] = {
                    "position_id": 10 + slot,
                    "position": "Center Midfield",
                    "from": "00:00",
                    "to": None,
                    "from_period": 1,
                    "to_period": None,
                    "start_reason": "Starting XI",
                    "end_reason": "Final Whistle",
                }
                positions = [spell]
                if match_id == SUB_MATCH and team == home and slot == 0:
                    spell["to"] = SUB_CLOCK
                    spell["to_period"] = 2
                    spell["end_reason"] = "Substitution - Off (Tactical)"
            else:
                positions = []
                if match_id == SUB_MATCH and team == home:
                    positions = [
                        {
                            "position_id": 13,
                            "position": "Center Midfield",
                            "from": SUB_CLOCK,
                            "to": None,
                            "from_period": 2,
                            "to_period": None,
                            "start_reason": "Substitution - On (Tactical)",
                            "end_reason": "Final Whistle",
                        }
                    ]
            entries.append(
                {
                    "player_id": player_id,
                    "player_name": name,
                    "player_nickname": nickname,
                    "jersey_number": 2 + slot,
                    "country": {"id": 214, "name": "Nowhere"},
                    "cards": [],
                    "positions": positions,
                }
            )
        out.append({"team_id": TEAMS[team], "team_name": team, "lineup": entries})
    return out


def _match_record(match_id: int, date: str, home: str, away: str, season_id: int,
                  season: str) -> dict[str, Any]:
    """Build one row of a season manifest.

    Args:
        match_id: Match id.
        date: ISO match date.
        home: Home team name.
        away: Away team name.
        season_id: StatsBomb season id.
        season: Season label.

    Returns:
        The manifest record.
    """
    return {
        "match_id": match_id,
        "match_date": date,
        "competition": {
            "competition_id": COMPETITION_ID,
            "country_name": "Nowhere",
            "competition_name": COMPETITION,
        },
        "season": {"season_id": season_id, "season_name": season},
        "home_team": {"home_team_id": TEAMS[home], "home_team_name": home},
        "away_team": {"away_team_id": TEAMS[away], "away_team_name": away},
    }


#: ``(tm_player_id, name, first, last, current_competition, current_club, last_season)``.
TM_PLAYERS: Final[tuple[tuple[int, str, str, str, str, str, int], ...]] = (
    # Nickname tier: StatsBomb's legal name never matches this.
    (5001, "Paco Alcacero", "Paco", "Alcacero", "ES1", "Real Nowhere", 2020),
    # Exact full-name tier.
    (5002, "Juan Manuel Iturbo Arevalos", "Juan Manuel", "Iturbo Arevalos", "ES1",
     "Real Nowhere", 2020),
    # Last-two-tokens tier: "Rafael Alcantaro do Nascimiento" vs this.
    (5003, "Rafinha Alcantaro do Nascimiento", "Rafinha", "Alcantaro do Nascimiento", "ES1",
     "Nowhere City", 2020),
    # Two players share this name; only one was in ES1 during the season.
    (5004, "Carlos Ambiguo Perez", "Carlos", "Ambiguo Perez", "ES1", "Nowhere City", 2020),
    (5005, "Carlos Ambiguo Perez", "Carlos", "Ambiguo Perez", "GB1", "Elsewhere Town", 2021),
    # Two players share this name and neither is in ES1: genuinely ambiguous.
    (5006, "David Ambiguo Lopez", "David", "Ambiguo Lopez", "GB1", "Elsewhere Town", 2021),
    (5007, "David Ambiguo Lopez", "David", "Ambiguo Lopez", "IT1", "Altrove Calcio", 2021),
    # Known to Transfermarkt, but with no valuation inside the season window.
    (5008, "Ninguno Sinvalor Ruiz", "Ninguno", "Sinvalor Ruiz", "ES1", "Real Nowhere", 2020),
    # Plain matches for the rest of the league.
    (5009, "Ana Suplente Diaz", "Ana", "Suplente Diaz", "ES1", "Real Nowhere", 2020),
    (5010, "Gonzalo Primero Martin", "Gonzalo", "Primero Martin", "ES1", "Nowhere City", 2020),
    (5011, "Hector Segundo Blanco", "Hector", "Segundo Blanco", "ES1", "Nowhere City", 2020),
    (5012, "Ivan Tercero Navas", "Ivan", "Tercero Navas", "ES1", "Nowhere City", 2020),
    (5013, "Jorge Cuarto Molina", "Jorge", "Cuarto Molina", "ES1", "Nowhere City", 2020),
    (5014, "Kevin Quinto Ortega", "Kevin", "Quinto Ortega", "ES1", "Nowhere United", 2020),
    (5015, "Luis Sexto Cabrera", "Luis", "Sexto Cabrera", "ES1", "Nowhere United", 2020),
    (5016, "Mario Septimo Vidal", "Mario", "Septimo Vidal", "ES1", "Nowhere United", 2020),
    (5017, "Nestor Octavo Rey", "Nestor", "Octavo Rey", "ES1", "Nowhere United", 2020),
)

#: ``(tm_player_id, date, value_eur, competition)``. 5008 is valued only outside
#: the 2019/2020 window; 5007 is deliberately left with no valuation at all.
TM_VALUATIONS: Final[tuple[tuple[int, str, int, str], ...]] = (
    (5001, "2019-06-01", 20_000_000, "ES1"),
    (5001, "2019-12-15", 25_000_000, "ES1"),
    (5001, "2020-06-01", 30_000_000, "ES1"),
    (5002, "2019-12-15", 12_000_000, "ES1"),
    (5003, "2019-12-15", 8_000_000, "ES1"),
    (5004, "2019-12-15", 5_000_000, "ES1"),
    (5005, "2019-12-15", 4_000_000, "GB1"),
    (5006, "2019-12-15", 3_000_000, "GB1"),
    (5008, "2014-12-15", 1_000_000, "ES1"),
    (5009, "2019-12-15", 900_000, "ES1"),
    (5010, "2019-12-15", 2_000_000, "ES1"),
    (5011, "2019-12-15", 3_500_000, "ES1"),
    (5012, "2019-12-15", 6_000_000, "ES1"),
    (5013, "2019-12-15", 1_500_000, "ES1"),
    (5014, "2019-12-15", 11_000_000, "ES1"),
    (5015, "2019-12-15", 750_000, "ES1"),
    (5016, "2019-12-15", 40_000_000, "ES1"),
    (5017, "2019-12-15", 2_750_000, "ES1"),
)


def _write_gzip_csv(path: Path, header: list[str], rows: list[tuple[Any, ...]]) -> None:
    """Write a gzipped CSV reproducibly.

    ``mtime=0`` keeps the bytes identical between runs, so regenerating the
    fixtures does not show up as a diff when nothing changed.

    Args:
        path: Destination ``.csv.gz``.
        header: Column names.
        rows: Row tuples.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as raw, gzip.GzipFile(
        fileobj=raw, mode="wb", compresslevel=9, mtime=0
    ) as gz:
        text = io.TextIOWrapper(gz, encoding="utf-8", newline="")
        writer = csv.writer(text)
        writer.writerow(header)
        writer.writerows(rows)
        text.flush()


def write_fixtures(root: Path = ROOT) -> Path:
    """Write the whole fixture tree.

    Args:
        root: Directory to write under.

    Returns:
        ``root``.
    """
    sb = root / "event_cache"
    (sb / "events").mkdir(parents=True, exist_ok=True)
    (sb / "lineups").mkdir(parents=True, exist_ok=True)
    (sb / "matches" / str(COMPETITION_ID)).mkdir(parents=True, exist_ok=True)

    competitions = [
        {
            "competition_id": COMPETITION_ID,
            "season_id": season_id,
            "country_name": "Nowhere",
            "competition_name": COMPETITION,
            "season_name": season,
            "competition_gender": "male",
            "match_available": "2026-01-01T00:00:00.000000",
        }
        for season_id, season in ((BIG_SEASON_ID, BIG_SEASON), (THIN_SEASON_ID, THIN_SEASON))
    ]
    (sb / "competitions.json").write_text(json.dumps(competitions, indent=1))

    manifest = [
        _match_record(mid, date, home, away, BIG_SEASON_ID, BIG_SEASON)
        for mid, date, home, away in FIXTURES
    ]
    (sb / "matches" / str(COMPETITION_ID) / f"{BIG_SEASON_ID}.json").write_text(
        json.dumps(manifest, indent=1)
    )
    thin_id, thin_date, thin_home, thin_away = THIN_FIXTURE
    (sb / "matches" / str(COMPETITION_ID) / f"{THIN_SEASON_ID}.json").write_text(
        json.dumps(
            [_match_record(thin_id, thin_date, thin_home, thin_away, THIN_SEASON_ID, THIN_SEASON)],
            indent=1,
        )
    )

    for mid, _date, home, away in (*FIXTURES, THIN_FIXTURE):
        (sb / "events" / f"{mid}.json").write_text(json.dumps(build_events(mid, home, away)))
        (sb / "lineups" / f"{mid}.json").write_text(json.dumps(build_lineup(mid, home, away)))

    # The trap: a match in no manifest, with players who exist nowhere else.
    (sb / "events" / "999999.json").write_text(
        json.dumps(build_events(999999, "Gamma SC", "Delta AC"))
    )
    (sb / "lineups" / "999999.json").write_text(
        json.dumps(build_lineup(999999, "Gamma SC", "Delta AC"))
    )

    _write_gzip_csv(
        root / "transfermarkt" / "players.csv.gz",
        [
            "player_id", "name", "first_name", "last_name",
            "current_club_domestic_competition_id", "current_club_name", "last_season",
        ],
        [tuple(row) for row in TM_PLAYERS],
    )
    _write_gzip_csv(
        root / "transfermarkt" / "player_valuations.csv.gz",
        ["player_id", "date", "market_value_in_eur", "player_club_domestic_competition_id"],
        [tuple(row) for row in TM_VALUATIONS],
    )
    return root


if __name__ == "__main__":  # pragma: no cover
    print(f"wrote {write_fixtures()}")
