# TODO

State as of v0.1.0. Everything the specification asked for is built and measured: loaders, possession
segmentation, pitch control, EPV, the five-second counter-press metric, phases, plots, the six-page
report, the offline demo, the event-only proxy and season mode. What follows is what is genuinely
outstanding, what needs a human to do something, and what was decided against on purpose.

## Manual steps (nothing here is a bug)

- **`sportio` does not exist on this machine.** The spec has this package sit on the shared loader
  layer. `src/pressing/_io.py` implements the Metrica loaders internally behind a shim that prefers
  `import sportio` and delegates when the import succeeds (`pressing._io.backend_name()` reports which
  path is live). When the shared layer lands, delete the internal loaders; nothing above them changes.
- **Raw data is not committed and has to be fetched.** `data/raw/` is gitignored. Re-running the
  pitch-control notebook needs Metrica's sample CSVs; retraining the proxy needs about 1.2 GB of
  StatsBomb events and 360 files at one request per second (`uv run python -m pressing.proxy
  --download`); a full-season `pressing season` run without `--offline` is 380 events files plus 380
  lineup files, roughly 13 minutes of polite downloading before any work starts. The demo needs none of
  this: it runs on the committed derived asset with no network.
- **The Transfermarkt snapshot ages.** `data/transfermarkt/` is a column-reduced copy taken 2026-08-25,
  most recent valuation 2026-06-12, and it is the last of three routes the join tries (full tables in
  `data/raw/transfermarkt/`, then the public R2 mirror, then this). Refresh it when the values matter,
  and update the date in `data/transfermarkt/NOTICE.md` with it.
- **There is no top-level `LICENSE` file**, although `pyproject.toml` declares
  `license = { text = "MIT" }`. `git ls-files | grep -i licen` returns only the two vendored notices,
  `data/epv/LICENSE.friends-of-tracking` and `validation/reference/LICENSE`. Either add the MIT text or
  change the declaration — left alone here because it is a statement the owner makes, not a code change.
- **Metrica may ask for the demo asset to be removed.** The upstream repository publishes no licence.
  If that request comes, delete `data/demo/` and the demo falls back to needing a local Metrica
  checkout. `data/demo/NOTICE.md` states the undertaking.

## Genuinely outstanding

- **A phase- and formation-conditional baseline.** This is the real fix for the field-position problem
  and it is not built. Recovery is dominated by where the ball was lost (measured means 0.257 build-up,
  0.468 middle, 0.701 final third on the bundled match), so the honest per-player number is a residual
  against what that team's shape would be expected to produce from that position, not a level. Fitting
  it needs far more than two tracked matches. Until then every table prints the per-phase split and
  every output repeats the caveat.
- **The two-minute demo budget has no margin on a contended machine.** Six cold-cache runs measured
  57.4 s, 58.9 s, 60.3 s, 81.0 s, 92.7 s and 152.5 s end to end on the same laptop the same afternoon;
  the last one missed the budget because six worker processes were competing with unrelated builds.
  The CPU cost is stable and the code is not the problem, but nothing in the tool adapts `--jobs` to
  the load it actually finds, and it could.
- **No uncertainty on the per-player numbers.** The player table prints three decimals over roughly
  twenty losses per player. It should print an interval, or refuse to print at all below a minimum
  count, the way season mode already refuses a per-90 below 270 minutes.
- **`tests/test_pitchcontrol.py::test_int_dt_above_the_stability_boundary_is_rejected` has a stale
  docstring.** It says "dt>=0.08 diverges to ~1e42". The integrator does not diverge at a coarse
  timestep; the guard protects accuracy that the renormalise clamp would otherwise hide. The test and
  the guard are correct — only the sentence is wrong.
- **Proxy predictions are heavily shrunk.** Held-out prediction spread is 0.435 of the label's, which
  is what a squared-error fit does when most of the target is unexplained. Ranks are usable; levels are
  not. A quantile or rank-based objective would be the thing to try.
- **The proxy's numbers are not bit-reproducible across processes.** Refits inside one process are
  identical; a refit in a fresh process reproduced the artefact's recorded held-out metrics to within
  0.001 on the full model (0.3899 against 0.3907) and 0.008 on the x-only baseline (0.1990 against
  0.2070). That is scikit-learn's own cross-process variation, and it is the honest precision of every
  proxy figure quoted anywhere in this repo. Reporting three decimals on them would be false.
- **The shipped proxy artefact is refitted on every labelled row**, so scoring it on the "held-out"
  matches is in-sample and returns r = 0.612 rather than 0.390. That is documented in the README and
  the model card and the artefact carries the split model's metrics in its metadata, but a reader who
  reproduces it naively will get the wrong number. A `held_out_match_ids` field on the artefact would
  make the mistake impossible instead of merely documented.
- **Season mode's market-value join is name-based** and drops ambiguous names rather than guessing
  (16 of 462 on the measured run). A StatsBomb-to-Transfermarkt id mapping would remove the guesswork
  entirely, and does not exist in open data.

## Deliberately not done

- **No per-player attribution from the proxy.** 360 freeze-frames carry no player identities. Anything
  that looked like a per-player event-only table would be invented.
- **No EPV from the proxy.** `predict()` emits `epv_conceded` as NaN on purpose: EPV needs the
  opponent's control surface, which an event stream cannot supply. A report built on proxy output must
  not print an EPV number.
- **SkillCorner is not used.** It ships no event stream, its ten open matches overlap no StatsBomb
  competition, and its derived events come from the same tracking that would generate our labels.
- **The PDF prints five moment panels, not one per loss.** The metric is computed at every loss;
  `--all-panels DIR` dumps the rest as PNGs. Three hundred panels in the PDF would be tens of seconds
  of plotting and a multi-megabyte file, and would contradict the six-page layout.
- **reportlab, not weasyprint.** Pure Python, no cairo/pango system libraries, so the demo survives a
  fresh clone. Do not add weasyprint.
- **Metrica's native 25 Hz is kept, not decimated to 10 Hz.** 25 Hz already clears the schema minimum
  and 25/10 is not an integer, so decimating would trade a smaller file for timestamp jitter.
- **No commercial framing anywhere.** The StatsBomb agreement forbids commercially exploiting the data
  or any analysis derived from it, and this is non-commercial personal analysis.

## House rules that outlived their tickets

- Every number in a docstring, comment, README or commit message must be reproducible by a command
  someone actually ran. The README ends with the list of commands that regenerate its numbers.
- Downloads are polite: at most one request per second per host, exponential backoff, everything cached
  under gitignored `data/raw/`, and never re-fetched once cached.
- StatsBomb data is never committed in any form. Check with `git ls-files data`, not the working tree.
- Wall-clock timings move with machine load; correlations do not. Say which machine state a timing came
  from, or do not quote it. The demo was timed seven times for the v0.1.0 README and the cold-cache
  end-to-end spread was 57.4 s to 152.5 s on one machine on one afternoon, entirely from contention.
  Quote the spread, never the best run.
