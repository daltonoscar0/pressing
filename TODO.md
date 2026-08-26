# TODO

## Blockers / manual steps

- **`sportio` is not available on this machine.** The spec has this package depend on `sportio`, the
  shared data layer that normalizes every open soccer source into the canonical `events` and `tracking`
  frames. That package does not exist here yet, so following the fallback the sibling setpiece spec
  spells out, `src/pressing/_io.py` implements the Metrica loaders internally. It tries `import sportio`
  first and delegates when the import succeeds, so nothing needs to change here once the shared layer
  lands — the internal loaders can simply be deleted.
- **Demo match not yet bundled.** `data/demo/` is empty. The format is settled (see below) and
  `write_tracking_parquet` works; the asset itself has never been generated, so `pressing demo` has
  nothing to run on even once `demo.py` exists.
- **Transfermarkt market values.** Season mode joins market values via a public mirror. If the mirror
  is unreachable offline, a cached snapshot goes under `data/` and the capture date must be recorded
  in the README next to the numbers.

## Next up

- Six-page reportlab PDF template (`report.py`) and the bundled offline demo (`demo.py`).
- Event-only proxy model (`proxy.py`), split by match, with the held-out correlation reported honestly.
  The label source is StatsBomb 360 freeze-frames; the spec's Metrica+SkillCorner overlap cannot be
  built and the README records why.
- Season mode and the market-value join (`season.py`).

Done and validated: the Metrica loaders and canonical frame, possession segmentation and loss detection,
the vectorised pitch-control model (validated against the Friends of Tracking reference — see
`validation/`), the EPV surface, the five-second counter-press metric and its aggregations, thirds of
the pitch, and the matplotlib renderers.

## Notes

- reportlab is the chosen PDF engine: pure Python, no system libraries, so the demo works on a fresh
  clone. Do not add weasyprint.
- Downloads are polite: at most one request per second per host, exponential backoff, everything
  cached under `data/raw/` (gitignored). Never re-download a file that is already cached.

## Data layer notes (loaders, possession, losses)

- `src/pressing/_io.py` implements the Metrica loaders behind the `sportio` shim. Canonical frame:
  metres on 105x68, origin bottom-left, y up, home attacking `+x` for the whole match. The period that
  needs mirroring is detected from goalkeeper mean x per period and recorded as `attrs["flip_period"]`;
  it is period 2 in game 1 and period 1 in game 2, so nothing about direction may be hard-coded.
- Velocities follow the Friends of Tracking recipe (backward diff / dt, 12 m/s clip, Savitzky-Golay
  window 7 / polyorder 1, per period). scipy's default `mode="interp"` raises on the NaN a substituted
  player leaves behind, so the filter is applied per maximal finite run instead of switching to
  `mode="nearest"`. That is bit-identical to the reference on every run the reference could itself have
  processed; it differs only within three frames of a span boundary, where the reference gives NaN.
- Possession runs break on a change of team **and** at half time. Game 1's home team both ends period 1
  in possession and kicks period 2 off, so a pure team-change cut splices a single 1,600 s "possession"
  across the interval. With the half-time break the counts are 358 (game 1) and 336 (game 2).
- Demo asset format is settled but **not yet built**: `write_tracking_parquet` / `read_tracking_parquet`,
  int16 centimetres, zstd level 19, `use_dictionary=["team"]`. Measured on game 1 at native 25 Hz
  (3,278,389 rows): 7,754,194 bytes = 7.75 MB, max per-coordinate position error 0.005 m.
  Velocities are omitted by default and recomputed on read, which is what keeps it under 8 MB (storing
  them costs 13,852,088 bytes = 13.85 MB).
- **That read path is not cheap, and the earlier 0.35 s figure in this file was wrong.** Re-timed on the
  game-1 asset, median of five runs: `read_tracking_parquet()` takes **1.79 s**. The parquet read itself
  is 0.05 s and assembling the canonical frame brings it to 0.20 s; the remaining ~1.6 s is
  `compute_velocities` rerunning the Savitzky-Golay smoothing the writer chose not to store. Reading a
  file written with `include_velocities=True` takes 0.21 s instead. So the 6.10 MB saved on disk costs
  about 1.6 s on every read — worth it for an asset committed to git and read once per demo run, not
  worth it for anything read repeatedly. If the 120 s budget gets tight, storing velocities is the
  cheapest second to buy back.
- **Metrica licensing decision (settled).** The sample-data repo publishes no LICENSE file — GitHub
  reports the licence as null, i.e. all rights reserved by default — and its README asks only that the
  source be acknowledged. Decision: never commit the raw CSVs; ship a *derived* asset (quantised,
  downsampled parquet) under `data/demo/` with a prominent `NOTICE.md` crediting Metrica Sports by name,
  linking the upstream repo, recording the derivation date and upstream commit, stating plainly that
  the upstream repo publishes no licence, and stating that the file will be removed on request.
  `tests/fixtures/metrica/` no longer contains Metrica data at all: the fixtures are synthetic files
  from a committed generator that copy the file *format* only, and two tests keep that claim honest.
- One open-play loss in game 1 cannot be located at all: the tracked ball is missing at its End Frame,
  the event carries no End X/Y and the following event has no Start X/Y. It is reported with
  `location_source == "none"` rather than guessed. Game 2 has none.

## Documentation-honesty notes

- Numbers in docstrings, comments, this file and the README must be reproducible by a command someone
  actually ran. Several were not and have been corrected against fresh measurements:
  `pitchcontrol.MAX_STABLE_INT_DT` (the integrator does not diverge at a coarse timestep — the guard
  protects accuracy that the renormalise clamp hides, not stability), the probability-overshoot
  magnitude (2.800681 measured, not 1.05 or 1.0418), the working-set saving (5.5x, not 2.9x), the
  per-cell reference loop cost (0.687-0.699 s per surface, not 0.5 s), the parquet read time above, and
  the dictionary-encoding sizing (12.11 MB / 7.75 MB / 7.76 MB).
- Still outstanding: `tests/test_pitchcontrol.py::test_int_dt_above_the_stability_boundary_is_rejected`
  carries a docstring repeating the old "dt>=0.08 diverges to ~1e42" claim. The guard it tests is
  correct and the test passes; only the docstring needs replacing.
