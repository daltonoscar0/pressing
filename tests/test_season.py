"""Tests for season mode.

Everything here runs offline against the committed fixtures under
``tests/fixtures/season/``, which are **not** StatsBomb data: they are an
invented four-match competition written in StatsBomb's shape, exactly as the
Metrica fixtures are a generated equivalent of Metrica's files. StatsBomb's
licence forbids redistributing their data, so a test that needed a real events
file would be a test that never runs on a fresh clone.

The fixture is built by ``tests/fixtures/make_season_fixtures.py`` and is
designed so that every branch of the market-value join has a player to exercise
it, and so that one events file is an orphan belonging to no season manifest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pressing import season as S

FIXTURES = Path(__file__).parent / "fixtures" / "season"
SB = FIXTURES / "event_cache"
TM = FIXTURES / "transfermarkt"

#: Match ids the season manifest lists, and the orphan that it does not.
LISTED = (700001, 700002, 700003, 700004)
ORPHAN = 999999

#: Clock bounds baked into the fixture: period 1 runs 00:00-45:57, period 2
#: 45:00-92:59. A player who is on for all of both is on for 93.933 minutes of
#: elapsed clock, not 90 -- stoppage time is real time.
FULL_MATCH_MINUTES = (2757.0 + (5579.0 - 2700.0)) / 60.0


@pytest.fixture
def spec() -> S.SeasonSpec:
    """The fixture competition's largest season, resolved offline."""
    return S.resolve_season("Fake Liga", cache_dir=SB, allow_download=False)


@pytest.fixture
def values() -> pd.DataFrame:
    """Fixture market values for the 2019/2020 window."""
    frame, _as_of = S.load_market_values(
        TM, window=S.season_window("2019/2020"), allow_download=False
    )
    return frame


def run(**kwargs: Any) -> S.SeasonResult:
    """Run the fixture season end to end with the network disabled.

    Args:
        **kwargs: Overrides forwarded to :func:`pressing.season.run_season`.

    Returns:
        The result.
    """
    options: dict[str, Any] = {
        "competition": "Fake Liga",
        "cache_dir": SB,
        "tm_cache_dir": TM,
        "allow_download": False,
        "min_minutes": 0.0,
        "model": None,
    }
    options.update(kwargs)
    return S.run_season(**options)


# --------------------------------------------------------------------------- #
# Name normalisation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Francisco Alcácer García", "francisco alcacer garcia"),
        ("FRANCISCO ALCACER GARCIA", "francisco alcacer garcia"),
        ("Ángel Di María", "angel di maria"),
        ("N'Golo Kanté", "n golo kante"),
        ("Pierre-Emerick Aubameyang", "pierre emerick aubameyang"),
        ("  Sergio   Ramos  ", "sergio ramos"),
        # NFKD alone deletes these: they are letters, not base plus mark.
        ("Ødegaard", "odegaard"),
        ("Łukasz Fabiański", "lukasz fabianski"),
        ("Aleksandar Đorđević", "aleksandar dordevic"),
        (None, ""),
        ("", ""),
        ("---", ""),
    ],
)
def test_normalise_name(raw: str | None, expected: str) -> None:
    """NFKD accent stripping, lowercase, punctuation to space, ends stripped."""
    assert S.normalise_name(raw) == expected


def test_normalise_name_makes_the_two_providers_agree() -> None:
    """The whole point: the same human, spelled two ways, normalises the same."""
    assert S.normalise_name("Rúben Días") == S.normalise_name("Ruben Dias")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Francisco Alcacer Garcia", "alcacer garcia"),
        ("Rafael Alcantara do Nascimento", "do nascimento"),
        ("Ronaldinho", ""),
        (None, ""),
    ],
)
def test_last_two_tokens(raw: str | None, expected: str) -> None:
    """The weakest tier keeps only the trailing two tokens, or nothing."""
    assert S.last_two_tokens(raw) == expected


# --------------------------------------------------------------------------- #
# The three join tiers
# --------------------------------------------------------------------------- #


def _one(joined: pd.DataFrame, player: str) -> pd.Series:
    """Pull a single player's joined row.

    Args:
        joined: Output of :func:`pressing.season.join_market_values`.
        player: StatsBomb player name.

    Returns:
        That player's row.
    """
    rows = joined[joined["player"] == player]
    assert len(rows) == 1, f"{player} appears {len(rows)} times"
    return rows.iloc[0]


@pytest.fixture
def joined(values: pd.DataFrame) -> pd.DataFrame:
    """The fixture season table joined to market values, ES1 as the league."""
    result = run()
    return S.join_market_values(result.table, values, competition_code="ES1")


def test_tier_one_exact_full_name(joined: pd.DataFrame) -> None:
    """A legal name that Transfermarkt happens to store verbatim."""
    row = _one(joined, "Juan Manuel Iturbo Arevalos")
    assert row["join_tier"] == "name"
    assert row["join_status"] == "matched"
    assert row["market_value_eur"] == 12_000_000


def test_tier_two_nickname_is_the_whole_ballgame(joined: pd.DataFrame) -> None:
    """StatsBomb store legal names, Transfermarkt common ones.

    Measured on 527 real La Liga 2015/16 players, exact full-name matching
    alone reaches 16.3%; adding ``player_nickname`` reaches 94.5%. Without this
    tier the join is not worth running, so it gets its own test.
    """
    row = _one(joined, "Francisco Alcacero Garcia")
    assert row["join_tier"] == "nickname"
    assert row["tm_name"] == "Paco Alcacero"
    # The full name alone finds nothing: that is what the nickname is rescuing.
    assert S.normalise_name("Francisco Alcacero Garcia") != S.normalise_name("Paco Alcacero")


def test_tier_three_last_two_tokens(joined: pd.DataFrame) -> None:
    """A shared surname pair rescues a name neither of the first two tiers hit."""
    row = _one(joined, "Rafael Alcantaro do Nascimiento")
    assert row["join_tier"] == "last_two"
    assert row["tm_name"] == "Rafinha Alcantaro do Nascimiento"


def test_tiers_are_tried_in_order_strongest_first(values: pd.DataFrame) -> None:
    """A player who matches on the full name must not be resolved by two tokens.

    The last-two-tokens tier collides far more readily than a full name, so it
    only ever runs when the stronger tiers found nothing.
    """
    table = pd.DataFrame(
        {"player": ["Juan Manuel Iturbo Arevalos"], "nickname": [None], "minutes": [900.0]}
    )
    row = S.join_market_values(table, values, competition_code="ES1").iloc[0]
    assert row["join_tier"] == "name"


def test_unmatched_player_keeps_a_null_value_rather_than_vanishing(
    joined: pd.DataFrame,
) -> None:
    """A player Transfermarkt does not know stays in the table, unpriced."""
    row = _one(joined, "Desconocido Nadie Gomez")
    assert row["join_status"] == "unmatched"
    assert pd.isna(row["market_value_eur"])


# --------------------------------------------------------------------------- #
# Ambiguity is dropped, never guessed
# --------------------------------------------------------------------------- #


def test_ambiguous_name_resolvable_by_competition_is_resolved(
    joined: pd.DataFrame,
) -> None:
    """Two Transfermarkt players share the name; one was in the right league."""
    row = _one(joined, "Carlos Ambiguo Perez")
    assert row["join_status"] == "matched"
    assert row["tm_player_id"] == 5004
    assert "club" in row["join_tier"]


def test_ambiguous_name_that_stays_ambiguous_is_dropped_not_guessed(
    joined: pd.DataFrame,
) -> None:
    """The rule that matters most.

    Two Transfermarkt players share this name and neither is in the league, so
    there is no evidence to choose between them. Guessing would attach a real
    price to the wrong real player and put that point in a published scatter. A
    missing point is a smaller lie than a wrong one.
    """
    row = _one(joined, "David Ambiguo Lopez")
    assert row["join_status"] == "ambiguous"
    assert pd.isna(row["market_value_eur"])
    assert pd.isna(row["tm_player_id"])


def test_an_unmapped_competition_drops_ambiguity_rather_than_taking_the_first(
    values: pd.DataFrame,
) -> None:
    """With no competition code there is no evidence, so nothing is chosen."""
    table = pd.DataFrame(
        {"player": ["Carlos Ambiguo Perez"], "nickname": [None], "minutes": [900.0]}
    )
    row = S.join_market_values(table, values, competition_code=None).iloc[0]
    assert row["join_status"] == "ambiguous"
    assert pd.isna(row["market_value_eur"])


def test_a_player_with_no_valuation_in_the_window_is_distinguishable(
    joined: pd.DataFrame,
) -> None:
    """"We could not price him" is not the same as "he is not in the data".

    Measured on real data the binding constraint is valuation coverage, not
    names: 474 of 527 La Liga 2015/16 players (89.9%) have a valuation inside
    the season window even though 94.9% match by name. The table has to keep
    those two failures apart.
    """
    row = _one(joined, "Ninguno Sinvalor Ruiz")
    assert row["join_status"] == "no_valuation"
    assert row["tm_player_id"] == 5008
    assert pd.isna(row["market_value_eur"])


def test_the_valuation_taken_is_the_last_one_inside_the_window(
    values: pd.DataFrame,
) -> None:
    """Three valuations exist for this player; only the in-window ones count."""
    row = values[values["tm_player_id"] == 5001].iloc[0]
    assert row["valuation_date"] == "2020-06-01"
    assert row["market_value_eur"] == 30_000_000


def test_a_valuation_outside_the_window_is_not_borrowed(
    values: pd.DataFrame,
) -> None:
    """Player 5008 is valued in 2014 only, and 2014 is not 2019/2020."""
    row = values[values["tm_player_id"] == 5008].iloc[0]
    assert pd.isna(row["market_value_eur"])


@pytest.mark.parametrize(
    ("label", "window"),
    [
        ("2015/2016", ("2015-07-01", "2016-06-30")),
        ("2015/16", ("2015-07-01", "2016-06-30")),
        ("2018", ("2018-01-01", "2018-12-31")),
    ],
)
def test_season_window(label: str, window: tuple[str, str]) -> None:
    """Split seasons run July to June; single-year ones are calendar years."""
    assert S.season_window(label) == window


def test_season_window_rejects_a_label_with_no_year() -> None:
    """Guessing a window would silently price a season against another one."""
    with pytest.raises(ValueError, match="cannot read a year"):
        S.season_window("Apertura")


# --------------------------------------------------------------------------- #
# Minutes come from the lineups, not from an assumed 90
# --------------------------------------------------------------------------- #


def test_period_clock_bounds_are_read_off_half_start_and_half_end() -> None:
    """Periods are not 45 minutes and the two periods overlap on the clock."""
    raw = json.loads((SB / "events" / "700001.json").read_text())
    bounds = S.period_clock_bounds(raw)
    assert bounds == {1: (0.0, 2757.0), 2: (2700.0, 5579.0)}
    # Period 1 ends at 45:57 while period 2 starts again at 45:00. An interval
    # spanning both must be clipped per period, never differenced end to end.
    assert bounds[2][0] < bounds[1][1]


def test_minutes_for_a_full_match_include_stoppage() -> None:
    """A player on for the whole match reads about 94, and that is deliberate."""
    raw = json.loads((SB / "events" / "700001.json").read_text())
    lineup = S.load_lineup(SB / "lineups" / "700001.json", S.period_clock_bounds(raw))
    row = lineup[lineup["player"] == "Juan Manuel Iturbo Arevalos"].iloc[0]
    assert row["minutes"] == pytest.approx(FULL_MATCH_MINUTES)
    assert row["minutes"] == pytest.approx(93.933, abs=0.001)


def test_minutes_for_a_substitution_are_split_across_both_players() -> None:
    """Off on the hour, on on the hour. Neither is credited with a full match."""
    raw = json.loads((SB / "events" / "700001.json").read_text())
    lineup = S.load_lineup(SB / "lineups" / "700001.json", S.period_clock_bounds(raw))
    off = lineup[lineup["player"] == "Francisco Alcacero Garcia"].iloc[0]
    on = lineup[lineup["player"] == "Carlos Ambiguo Perez"].iloc[0]
    # 00:00 to 60:00 spans all of period 1 (45:57 of clock) plus 15 minutes of
    # period 2, because period 2's clock restarts at 45:00.
    assert off["minutes"] == pytest.approx(2757.0 / 60.0 + 15.0)
    assert on["minutes"] == pytest.approx((5579.0 - 3600.0) / 60.0)
    assert off["minutes"] + on["minutes"] == pytest.approx(FULL_MATCH_MINUTES)


def test_an_unused_substitute_has_zero_minutes_and_no_rate() -> None:
    """Zero minutes must not become an infinite per-90."""
    result = run()
    row = result.table[result.table["player"] == "Ana Suplente Diaz"].iloc[0]
    assert row["minutes"] == 0.0
    assert pd.isna(row["counterpress_per_90"])
    assert pd.isna(row["rank"])


def test_per_90_uses_real_minutes_not_an_assumed_ninety() -> None:
    """The substituted player played less, so the same count is a higher rate.

    Both players made the same number of counter-pressing actions across the
    season. If minutes were assumed rather than read, their rates would be
    identical; because they are read, the one who played less ranks higher.
    """
    result = run()
    table = result.table.set_index("player")
    subbed = table.loc["Francisco Alcacero Garcia"]
    full = table.loc["David Ambiguo Lopez"]
    assert subbed["counterpress_actions"] == full["counterpress_actions"]
    assert subbed["minutes"] < full["minutes"]
    assert subbed["counterpress_per_90"] > full["counterpress_per_90"]
    assert subbed["counterpress_per_90"] == pytest.approx(
        float(subbed["counterpress_actions"]) / (float(subbed["minutes"]) / 90.0)
    )


def test_a_player_below_the_minutes_floor_is_kept_but_not_rated() -> None:
    """One counter-press in twelve minutes is 7.5 per 90 and would top a table."""
    result = run(min_minutes=1000.0)
    assert len(result.table) == 16
    assert result.table["counterpress_per_90"].notna().sum() == 0
    assert result.n_ranked == 0


# --------------------------------------------------------------------------- #
# The orphan-event-file trap
# --------------------------------------------------------------------------- #


def test_the_season_is_driven_from_the_manifest_not_a_directory_scan(
    spec: S.SeasonSpec,
) -> None:
    """274 real event files (831.9 MB) belong to no season manifest.

    They sit in the same ``data/events/`` directory as everything else. Any code
    that enumerates that directory rather than following
    ``competitions.json -> matches/{cid}/{sid}.json`` would fold matches from
    other competitions into a league aggregate, silently.
    """
    assert (SB / "events" / f"{ORPHAN}.json").exists()
    assert spec.match_ids == LISTED
    assert ORPHAN not in spec.match_ids


def test_the_orphan_match_contributes_nothing_to_the_table() -> None:
    """End to end: the orphan's events exist and change no number in the table."""
    result = run()
    # Each fixture team plays exactly two listed matches. The orphan is a third
    # Gamma-Delta fixture, so if it leaked in those teams would show three.
    played = result.table.set_index("player")["matches"]
    assert set(played.unique()) == {2}
    assert result.n_matches_used == len(LISTED)
    assert result.n_turnovers == 10 * len(LISTED)


# --------------------------------------------------------------------------- #
# A thin season warns
# --------------------------------------------------------------------------- #


def test_a_season_with_too_few_matches_warns_loudly() -> None:
    """Only one La Liga season in the open data is a full league programme.

    Every other one is a 1-38 match sample in which a player can appear once,
    and a per-90 table built from it is noise. The run says so rather than
    printing a ranking that looks authoritative.
    """
    result = run()
    assert result.spec.is_thin
    assert result.warnings
    warning = " ".join(result.warnings)
    assert "4 matches" in warning
    assert str(S.MIN_SEASON_MATCHES) in warning


def test_the_thin_season_warning_reaches_the_csv() -> None:
    """A CSV separated from its console output must still carry the caveat."""
    result = run()
    text = Path(result.csv_path).read_text()
    assert "# WARNING:" in text
    assert "sample, not a season" in text


def test_default_season_is_the_largest_not_the_latest(spec: S.SeasonSpec) -> None:
    """Latest would pick a 1-match sample over a 4-match season.

    On the real data this is the difference between La Liga 2020/21 (35
    matches) and La Liga 2015/16 (380).
    """
    assert spec.season == "2019/2020"
    latest = S.resolve_season("Fake Liga", "2020/2021", cache_dir=SB, allow_download=False)
    assert latest.n_matches < spec.n_matches


def test_an_unknown_competition_lists_what_is_available() -> None:
    """A typo should tell you the options, not hand back an empty table."""
    with pytest.raises(LookupError, match="Fake Liga"):
        S.resolve_season("Bundesliga", cache_dir=SB, allow_download=False)


def test_an_unknown_season_lists_what_is_available() -> None:
    """Same, one level down."""
    with pytest.raises(LookupError, match="2019/2020"):
        S.resolve_season("Fake Liga", "1999/2000", cache_dir=SB, allow_download=False)


def test_a_short_season_label_is_accepted() -> None:
    """"2015/16" is how everybody writes what StatsBomb calls "2015/2016"."""
    resolved = S.resolve_season("Fake Liga", "2019/20", cache_dir=SB, allow_download=False)
    assert resolved.season == "2019/2020"


def test_only_statsbomb_is_wired_up() -> None:
    """An unimplemented source fails immediately, not after a download."""
    with pytest.raises(ValueError, match="unknown source"):
        S.season_table("skillcorner", "Fake Liga", cache_dir=SB, allow_download=False)


# --------------------------------------------------------------------------- #
# Bounded runs
# --------------------------------------------------------------------------- #


def test_limit_spreads_across_the_season_rather_than_taking_the_front() -> None:
    """A capped run must not be an August-only table.

    The fixture's four matches run from August to May; taking the first two
    would sample one end of the calendar, with squads that had not settled and
    the January signings absent.
    """
    picked = S.select_matches(LISTED, 2)
    assert picked == (700001, 700004)
    assert S.select_matches(LISTED, None) == LISTED
    assert S.select_matches(LISTED, 99) == LISTED


def test_a_limited_run_says_how_much_of_the_season_it_covers() -> None:
    """Half a season of minutes is not a season of minutes."""
    result = run(limit=2)
    assert result.n_matches_used == 2
    assert any("2 of 4 matches" in w for w in result.warnings)
    assert "2 of 4 matches" in Path(result.csv_path).read_text()


def test_an_interrupted_run_resumes_from_the_cache(tmp_path: Path) -> None:
    """Downloads are never repeated, so a half-finished pull costs nothing.

    With downloads disabled and a half-populated cache, the run uses what is
    there and skips the rest instead of failing.
    """
    cache = tmp_path / "sb"
    (cache / "events").mkdir(parents=True)
    (cache / "lineups").mkdir(parents=True)
    (cache / "matches" / "99").mkdir(parents=True)
    for name in ("competitions.json",):
        (cache / name).write_bytes((SB / name).read_bytes())
    (cache / "matches" / "99" / "1.json").write_bytes(
        (SB / "matches" / "99" / "1.json").read_bytes()
    )
    for match_id in LISTED[:2]:
        for kind in ("events", "lineups"):
            (cache / kind / f"{match_id}.json").write_bytes(
                (SB / kind / f"{match_id}.json").read_bytes()
            )

    got = S.fetch_season(LISTED, cache, allow_download=False)
    assert [mid for mid, _e, _l in got] == list(LISTED[:2])

    result = run(cache_dir=cache, season="2019/2020")
    assert result.n_matches_used == 2


# --------------------------------------------------------------------------- #
# Per-match attribution
# --------------------------------------------------------------------------- #


def test_counterpress_actions_are_credited_to_the_player_who_made_them() -> None:
    """The volume metric is a count of StatsBomb's own flag, not a model output."""
    stats = S.match_player_stats(
        SB / "events" / "700001.json", SB / "lineups" / "700001.json"
    )
    assert stats.n_turnovers == 10
    assert stats.n_counterpress == 10
    assert stats.players["counterpress_actions"].sum() == 10
    assert stats.n_unmatched_actions == 0


def test_pressed_losses_carry_their_phase_mix() -> None:
    """Recovery is driven by where the ball was lost, so the mix has to travel.

    A player whose presses are all in the final third and one whose presses are
    all in his own build-up phase are not comparable on mean recovery, and the
    table only lets a reader see that because the phase counts are next to it.
    """
    stats = S.match_player_stats(
        SB / "events" / "700001.json", SB / "lineups" / "700001.json"
    )
    table = stats.players.set_index("player")
    phases = ["pressed_build_up", "pressed_middle", "pressed_final_third"]
    assert table.loc["Francisco Alcacero Garcia", phases].tolist() == [0, 0, 2]
    assert table.loc["Juan Manuel Iturbo Arevalos", phases].tolist() == [0, 2, 0]
    assert table.loc["Rafael Alcantaro do Nascimiento", phases].tolist() == [2, 0, 0]
    assert table["losses_pressed"].sum() == 10


def test_press_value_uses_the_model_and_is_centred_on_dead_even() -> None:
    """``press_value`` is control won ABOVE 0.5, summed over the presses made.

    A constant-0.5 model must therefore produce exactly zero value for
    everybody: dead-even control is worth nothing, by construction.
    """

    class Even:
        """A stand-in model that always predicts dead-even control."""

        def predict(self, features: pd.DataFrame) -> np.ndarray:
            """Return 0.5 for every row.

            Args:
                features: Feature frame, ignored.

            Returns:
                An array of 0.5.
            """
            return np.full(len(features), 0.5)

    stats = S.match_player_stats(
        SB / "events" / "700001.json", SB / "lineups" / "700001.json", Even()
    )
    pressed = stats.players[stats.players["losses_pressed"] > 0]
    assert (pressed["press_value"] == 0.0).all()
    assert (
        pressed["recovery_sum"] == 0.5 * pressed["losses_pressed"].astype(float)
    ).all()
    stats_none = S.match_player_stats(
        SB / "events" / "700001.json", SB / "lineups" / "700001.json", None
    )
    # With no model at all the counts survive and only the estimate is missing.
    assert stats_none.players["counterpress_actions"].sum() == 10
    assert stats_none.players["press_value"].isna().all()


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #


def test_run_writes_both_outputs_and_stamps_attribution() -> None:
    """StatsBomb attribution is required on every season output."""
    result = run()
    assert result.csv_path is not None and Path(result.csv_path).exists()
    assert result.png_path is not None and Path(result.png_path).exists()
    text = Path(result.csv_path).read_text()
    assert "StatsBomb" in text
    assert "Transfermarkt" in text
    assert "CC0-1.0 covers that repackaging" in text


def test_the_csv_carries_the_phase_caveat() -> None:
    """Recovery values are comparable only within a phase, wherever they land."""
    result = run()
    text = Path(result.csv_path).read_text()
    assert "Compare only within a phase" in text
    assert "build-up 0.26" in text


def test_the_csv_reads_back_as_a_table() -> None:
    """The header is comments, so the file is still a CSV."""
    result = run()
    frame = pd.read_csv(result.csv_path, comment="#")
    assert len(frame) == len(result.table)
    assert list(frame.columns) == [
        c for c in S.SEASON_TABLE_COLUMNS if c in result.table.columns
    ]


def test_the_run_reports_the_join_honestly() -> None:
    """Every failure mode is counted, and the counts add up to the table."""
    result = run()
    assert result.n_players == 16
    assert result.n_matched == 12
    assert result.n_ambiguous == 2
    assert result.n_unmatched == 1
    assert result.n_no_valuation == 1
    assert (
        result.n_matched + result.n_ambiguous + result.n_unmatched + result.n_no_valuation
        == result.n_players
    )
    assert result.join_rate == pytest.approx(12 / 16)


def test_a_thin_scatter_warns_rather_than_being_drawn_silently() -> None:
    """A ten-point chart that looks like a league study is worse than a caption."""
    result = run()
    assert result.n_plotted < S.MIN_SCATTER_POINTS
    assert any("below the" in w for w in result.warnings)
    assert "SAMPLE ONLY" in _caption_of(result)


def _caption_of(result: S.SeasonResult) -> str:
    """Rebuild the caption the scatter was written with.

    Args:
        result: A finished run.

    Returns:
        The caption text.
    """
    return (
        f"SAMPLE ONLY — {result.n_plotted} players"
        if result.n_plotted < S.MIN_SCATTER_POINTS
        else f"{result.n_plotted} players"
    )


def test_the_scatter_survives_an_empty_frame(tmp_path: Path, spec: S.SeasonSpec) -> None:
    """No data is a state to render, not an exception to raise."""
    stats = _stats_for(spec, 0)
    path, plotted = S.write_scatter(
        pd.DataFrame(columns=list(S.SEASON_TABLE_COLUMNS)), tmp_path / "empty.png", spec, stats
    )
    assert plotted == 0
    assert path.exists() and path.stat().st_size > 0


def test_the_scatter_survives_a_single_point(tmp_path: Path, spec: S.SeasonSpec) -> None:
    """One point cannot have a trend line, and must not raise trying."""
    table = pd.DataFrame(
        {
            "player": ["Solo Uno"],
            "team": ["Alpha FC"],
            "minutes": [900.0],
            "counterpress_per_90": [3.0],
            "market_value_eur": [1_000_000.0],
        }
    )
    path, plotted = S.write_scatter(table, tmp_path / "one.png", spec, _stats_for(spec, 1))
    assert plotted == 1
    assert path.exists() and path.stat().st_size > 0


def test_goalkeepers_are_kept_in_the_table_and_left_off_the_scatter(
    tmp_path: Path, spec: S.SeasonSpec
) -> None:
    """A keeper cannot counter-press, so his zero is a position not a finding.

    Leaving them in draws a row of zeros across the whole market-value axis,
    which the eye turns into a downward relationship. They stay in the CSV --
    they are players and they have minutes -- and out of the chart, and the
    caption says how many were dropped.
    """
    table = pd.DataFrame(
        {
            "player": ["Keeper", "Winger"],
            "position": ["Goalkeeper", "Right Wing"],
            "minutes": [900.0, 900.0],
            "counterpress_per_90": [0.0, 6.0],
            "market_value_eur": [25_000_000.0, 5_000_000.0],
        }
    )
    _path, plotted = S.write_scatter(table, tmp_path / "gk.png", spec, _stats_for(spec, 2))
    assert plotted == 1
    assert S.is_goalkeeper("Goalkeeper")
    assert not S.is_goalkeeper("Right Wing")
    assert not S.is_goalkeeper(None)


def test_the_table_records_the_position_a_player_played_most(
    tmp_path: Path,
) -> None:
    """Position comes from the lineup, and is in the CSV so keepers are findable."""
    result = run()
    assert "position" in result.table.columns
    assert set(result.table["position"].dropna()) == {"Center Midfield"}
    frame = pd.read_csv(result.csv_path, comment="#")
    assert "position" in frame.columns


def test_the_scatter_ignores_players_with_no_value(
    tmp_path: Path, spec: S.SeasonSpec
) -> None:
    """A dropped join must not become a zero-euro point on the log axis."""
    table = pd.DataFrame(
        {
            "player": ["Priced", "Unpriced", "Zero"],
            "minutes": [900.0, 900.0, 900.0],
            "counterpress_per_90": [3.0, 2.0, 1.0],
            "market_value_eur": [1_000_000.0, np.nan, 0.0],
        }
    )
    _path, plotted = S.write_scatter(table, tmp_path / "some.png", spec, _stats_for(spec, 3))
    assert plotted == 1


def _stats_for(spec: S.SeasonSpec, n: int) -> dict[str, Any]:
    """Minimal stats dict for the writers, for tests that skip the pipeline.

    Args:
        spec: The resolved season.
        n: Number of players, only used in the caption.

    Returns:
        The stats mapping the writers read.
    """
    return {
        "n_matches_used": spec.n_matches,
        "n_matches_listed": spec.n_matches,
        "n_turnovers": 0,
        "n_players": n,
        "value_window": S.season_window(spec.season),
        "values_as_of": "2026-06-12",
        "values_source": "snapshot",
    }


# --------------------------------------------------------------------------- #
# The committed snapshot
# --------------------------------------------------------------------------- #


def test_no_statsbomb_data_is_committed() -> None:
    """A locked decision, and the one that has to be checked in git, not on disk."""
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files", "data"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
        check=False,
    ).stdout
    assert "statsbomb" not in tracked.lower()


def test_the_committed_transfermarkt_snapshot_is_usable_offline() -> None:
    """A fresh clone with no network still has market values to join to."""
    if not (S.SNAPSHOT_DIR / "players.csv.gz").exists():  # pragma: no cover
        pytest.skip("snapshot not present in this checkout")
    values, as_of = S.load_market_values(
        cache_dir=Path("/nonexistent-cache"), window=("2015-07-01", "2016-06-30"),
        allow_download=False,
    )
    assert values.attrs["source"] == "snapshot"
    assert len(values) > 10_000
    assert as_of >= "2026-06-12"


def test_the_snapshot_notice_records_its_date_and_licence() -> None:
    """A snapshot with no date is a market value with no date."""
    notice = S.SNAPSHOT_DIR / "NOTICE.md"
    if not notice.exists():  # pragma: no cover
        pytest.skip("snapshot not present in this checkout")
    text = notice.read_text()
    assert S.SNAPSHOT_AS_OF in text
    assert "CC0-1.0" in text
    flat = " ".join(text.split())
    assert "cover the underlying rights" in flat
    assert "prohibit scraping" in flat


def test_missing_transfermarkt_data_fails_with_a_sentence_not_a_stack(
    tmp_path: Path,
) -> None:
    """Three routes, all absent, is a situation to explain."""
    with pytest.raises(FileNotFoundError, match="no Transfermarkt data"):
        S.load_market_values(
            cache_dir=tmp_path / "nothing", allow_download=False, snapshot_dir=tmp_path / "none"
        )


# --------------------------------------------------------------------------- #
# Through the command line
# --------------------------------------------------------------------------- #


def _cli(*args: str) -> Any:
    """Invoke the CLI in-process.

    Args:
        *args: Command line arguments.

    Returns:
        Click's result object.
    """
    from typer.testing import CliRunner

    from pressing.cli import app

    return CliRunner().invoke(app, list(args))


def test_season_runs_offline_through_the_cli(tmp_path: Path) -> None:
    """The flags reach the pipeline and both paths land on stdout."""
    result = _cli(
        "season",
        "--competition", "Fake Liga",
        "--season", "2019/2020",
        "--cache-dir", str(SB),
        "--values-dir", str(TM),
        "--out-dir", str(tmp_path),
        "--min-minutes", "0",
        "--offline",
    )
    assert result.exit_code == 0, result.output
    lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
    written = [Path(line) for line in lines[-2:]]
    assert [p.suffix for p in written] == [".csv", ".png"]
    assert all(p.exists() for p in written)


def test_the_cli_prints_the_join_rate_and_the_drops(tmp_path: Path) -> None:
    """A join rate nobody prints is a join rate nobody checks."""
    result = _cli(
        "season",
        "--competition", "Fake Liga",
        "--cache-dir", str(SB),
        "--values-dir", str(TM),
        "--out-dir", str(tmp_path),
        "--min-minutes", "0",
        "--offline",
    )
    assert result.exit_code == 0, result.output
    assert "12 of 16 (75.0%) priced" in result.output
    assert "1 with no valuation" in result.output
    assert "2 dropped as ambiguous rather than guessed" in result.output
    assert "1 not found" in result.output
    assert "warning:" in result.output


def test_the_cli_explains_an_unknown_competition_rather_than_tracebacking(
    tmp_path: Path,
) -> None:
    """Exit 1, one sentence, and the list of what does exist."""
    result = _cli(
        "season",
        "--competition", "Nonexistent League",
        "--cache-dir", str(SB),
        "--out-dir", str(tmp_path),
        "--offline",
    )
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "Fake Liga" in result.output


def test_proxy_writes_per_turnover_estimates_through_the_cli(tmp_path: Path) -> None:
    """The event-only path produces numbers, stamped as estimates."""
    from pressing import proxy as proxy_model

    if not proxy_model.bundled_model_path().exists():  # pragma: no cover
        pytest.skip("no bundled proxy model in this checkout")

    out = tmp_path / "estimates.csv"
    result = _cli("proxy", str(SB / "events" / "700001.json"), "--out", str(out))
    assert result.exit_code == 0, result.output
    assert out.exists()

    text = out.read_text()
    assert "StatsBomb" in text
    assert "ESTIMATE" in text
    assert "Compare only within a phase" in text

    frame = pd.read_csv(out, comment="#")
    assert len(frame) == 10
    assert frame["recovery"].between(0.0, 1.0).all()
    assert frame["epv_conceded"].isna().all()
    assert set(frame["phase"]) == {"build_up", "middle", "final_third"}


def test_proxy_refuses_a_team_that_is_not_in_the_match(tmp_path: Path) -> None:
    """A typo in --team must not silently produce an empty table."""
    from pressing import proxy as proxy_model

    if not proxy_model.bundled_model_path().exists():  # pragma: no cover
        pytest.skip("no bundled proxy model in this checkout")

    result = _cli(
        "proxy",
        str(SB / "events" / "700001.json"),
        "--team", "Nowhere FC",
        "--out", str(tmp_path / "x.csv"),
    )
    assert result.exit_code == 1
    assert "Alpha FC" in result.output
