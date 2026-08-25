# TODO

## Blockers / manual steps

- **`sportio` is not available on this machine.** The spec has this package depend on `sportio`, the
  shared data layer that normalizes every open soccer source into the canonical `events` and `tracking`
  frames. That package does not exist here yet, so following the fallback the sibling setpiece spec
  spells out, `src/pressing/_io.py` implements minimal loaders internally. It tries `import sportio`
  first and delegates when the import succeeds, so nothing needs to change here once the shared layer
  lands — the internal loaders can simply be deleted. The internal loaders are currently stubs that
  raise `NotImplementedError`.
- **Demo match not yet bundled.** `data/demo/` is empty. The Metrica sample match needs to be fetched
  and truncated to something small enough to commit, so `pressing demo` runs offline on a fresh clone.
- **Transfermarkt market values.** Season mode joins market values via a public mirror. If the mirror
  is unreachable offline, a cached snapshot goes under `data/` and the capture date must be recorded
  in the README next to the numbers.

## Next up

- Pitch control model and validation against the Friends of Tracking reference implementation on
  Metrica match 1; record the correlation actually achieved under `validation/`.
- Counter-press metric plus per-player, per-team and per-phase aggregation.
- Event-only proxy model, split by match, with the held-out correlation reported honestly.
- Season mode and the market-value join.
- Six-page reportlab PDF template and the bundled offline demo.

## Notes

- reportlab is the chosen PDF engine: pure Python, no system libraries, so the demo works on a fresh
  clone. Do not add weasyprint.
- Downloads are polite: at most one request per second per host, exponential backoff, everything
  cached under `data/raw/` (gitignored). Never re-download a file that is already cached.

## Data layer notes (loaders, possession, losses)

- `src/pressing/_io.py` now implements the Metrica loaders behind the `sportio` shim. Canonical frame:
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
  int16 centimetres, zstd level 19, `use_dictionary=["team"]`. Measured on game 1 at native 25 Hz:
  7.75 MB, read back in 0.35 s, max position error 0.005 m. Velocities are omitted by default and
  recomputed on read, which is what keeps it under 8 MB (storing them costs 13.85 MB). Creating the
  committed asset is still blocked on the Metrica licensing decision below.
- **Metrica licensing is still open.** The sample-data repo has no LICENSE file and GitHub reports the
  licence as null; the README only asks that the source be acknowledged. `tests/fixtures/metrica/`
  currently commits truncated slices (360 and 240 frames) with that acknowledgement in
  `tests/fixtures/README.md`. Shipping a full match in `data/demo/` is a larger ask and needs a decision.
- One open-play loss in game 1 cannot be located at all: the tracked ball is missing at its End Frame,
  the event carries no End X/Y and the following event has no Start X/Y. It is reported with
  `location_source == "none"` rather than guessed. Game 2 has none.
