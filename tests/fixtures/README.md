# Test fixtures

Everything here is committed so the test suite runs completely offline.

## `metrica/`

**These are not Metrica's data.** They are a wholly invented two-game match
pair written by `make_metrica_fixtures.py`, which is committed alongside them.
Every coordinate is the output of the closed-form motion model in that script
plus a fixed PRNG seed; the frame numbers, the substitution moment, the roster
and the event sequence are all chosen in that file.

What the fixtures *do* copy is the container: the three header rows, the blank
`_y` columns, the `NaN` spelling, the `PlayerNN` / `Player NN` labels and the
`Sample_Game_N_...csv` filenames that `pressing._io` parses. A file format is
not match data.

Two tests keep that claim honest:

- `tests/test_io.py::test_fixtures_are_exactly_what_the_generator_writes`
  regenerates all six CSVs into a temporary directory and compares them byte
  for byte with the committed ones, so nothing can be smuggled in by hand.
- `tests/test_io.py::test_every_documented_quirk_is_still_in_the_fixtures`
  runs the generator's own `verify()` over the committed files.

### Why synthetic rather than derived

The Metrica sample-data repository publishes no licence file, GitHub reports
`license: null`, which is all rights reserved by default, and its README asks
only that users be responsible with the data and acknowledge the source. This
project's data decision is therefore to ship derived or invented assets and
never the raw CSVs. A quantised or downsampled slice of the real match would
still be Metrica's data, just at lower precision; a generated match is not
their data at any precision, and that is a claim that can be checked by running
one script.

### What overlap with the upstream files remains, measured

Compared line by line against all twelve upstream CSVs (576,014 distinct
non-empty lines) on 2026-08-25:

- **13 whole lines match verbatim, and every one of them is a header row** -
  the two events-file header rows, and the team-name, shirt-number and
  column-name rows of the four tracking files. **Zero data rows match.**
  Those header rows are the container: an events header identical for every
  Metrica export, `Home` repeated over each pair, and anonymous `PlayerNN`
  placeholders with their shirt numbers. There are no names, no positions and
  no coordinates in them.
- **11 of 13,663 adjacent `(x, y)` coordinate pairs collide**, 0.08%.
  Regenerating the same fixtures from a different PRNG seed gives 9 of 13,663,
  0.07%, the same rate, so the collisions are chance, not provenance. The
  upstream files hold 6,396,548 distinct pairs.
- Per-cell overlap is high (about 89%) and means nothing: upstream holds
  246,842 distinct 5-decimal values in a space of roughly 100,000 slots per
  unit interval, so it saturates the value space and almost any plausible
  5-decimal number appears in it somewhere. The pair test above is the one
  that discriminates.

### The quirks these fixtures exist to catch

A generator that emitted clean, well-behaved data would silently retire the
regression tests below, so each quirk is asserted against the written bytes by
`verify()` in the generator, and the full list lives in its `QUIRKS` tuple.

| quirk | where |
| ----- | ----- |
| three header rows, data from the fourth line | both games, both sides |
| team name on row 0, jersey numbers on row 1 | both games, both sides |
| every named column followed by a blank `_y`, ball named last | both games |
| `Player 26` written with a space | game 2 away, tracking header and events |
| bench columns that are `NaN` for the whole file | `Player13`, `Player14` |
| substitution with a one-frame overlap | game 1 home, `Player1` -> `Player12` |
| contiguous player presence spans | game 1 home |
| dead-ball `NaN` runs in the ball columns | both games |
| ball columns identical in the home and away files | both games |
| non-contiguous frame blocks | game 1 has three, game 2 has two |
| both periods inside one file | both games |
| the two games attack opposite ways in the first half | game 1 vs game 2 |
| coordinates outside `[0, 1]`, i.e. off the pitch | both games |
| an event row with `End Frame` 0 and `NaN` coordinates | game 1 kick-off |
| a `BALL LOST` with `NaN` `End X`/`End Y` | game 2, frame 55071 |
| hyphenated subtypes needing token splitting | game 2 `KICK OFF-RETAKEN` |

### Layout

| game | frame blocks | periods |
| ---- | ------------ | ------- |
| 1    | 1-120, 30001-30120, 62501-62620 | 1, 1, 2 |
| 2    | 1-120, 55001-55120 | 1, 2 |

### Regenerating

```
uv run python tests/fixtures/make_metrica_fixtures.py
```

The script rewrites all six CSVs and then prints one line of evidence per
quirk. It is deterministic: the same seeds give the same bytes.

## Acknowledgement

The CSV layout these fixtures imitate is Metrica Sports'. See
<https://github.com/metrica-sports/sample-data>, which asks that the source be
acknowledged wherever the data is used. No Metrica match data is committed to
this repository.
