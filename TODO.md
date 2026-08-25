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
