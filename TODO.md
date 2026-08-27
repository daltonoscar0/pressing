# TODO

State as of v0.1.1, which is a measurement round rather than a feature round. Everything the
specification asked for was already built at v0.1.0; what changed since is that two adversarial audits
measured the flagship output, both central findings reproduced, and the repository now says so on the
page a coach reads. What follows is what is genuinely outstanding, what needs a human, and what was
decided against on purpose. Numbers here are reproducible by the commands in the README.

## Manual steps (nothing here is a bug)

- **The package version string still reads `0.1.0` while the tag is `v0.1.1`.** `pyproject.toml` and
  `pressing.__version__` were not bumped in this round, so `uv run pressing --version` prints `0.1.0`
  and the README quotes it that way. Bumping a version is a statement the owner makes; do it, or move
  the tag, but do not leave a third answer lying around.
- **`sportio` does not exist on this machine.** The spec has this package sit on the shared loader
  layer. `src/pressing/_io.py` implements the Metrica loaders internally behind a shim that prefers
  `import sportio` and delegates when the import succeeds (`pressing._io.backend_name()` reports which
  path is live). When the shared layer lands, delete the internal loaders; nothing above them changes.
- **Raw data is not committed and has to be fetched.** `data/raw/` is gitignored. Re-running the
  pitch-control notebook needs Metrica's sample CSVs; retraining the proxy needs 1.3 GB of
  StatsBomb events and 360 files at one request per second (`uv run python -m pressing.proxy
  --download`); a full-season `pressing season` run without `--offline` is 380 events files plus 380
  lineup files, roughly 13 minutes of polite downloading before any work starts. The demo needs none of
  this: it runs on the committed derived asset with no network.
- **The Transfermarkt snapshot ages.** `data/transfermarkt/` is a column-reduced copy taken 2026-08-25,
  most recent valuation 2026-06-12, and it is the last of three routes the join tries (full tables in
  `data/raw/transfermarkt/`, then the public R2 mirror, then this). Refresh it when the values matter,
  and update the date in `data/transfermarkt/NOTICE.md` with it.
- **Metrica may ask for the demo asset to be removed.** The upstream repository publishes no licence.
  If that request comes, delete `data/demo/` and the demo falls back to needing a local Metrica
  checkout. `data/demo/NOTICE.md` states the undertaking, and now also records the upstream commit
  `e706dd506b360d69d9d123d5b8026e7294b13996` with the git blob id and sha256 of each source file.
- **The committed proxy artefacts contain StatsBomb-derived content that is not data.** No rows, no
  identities: a fitted estimator plus StatsBomb's event-type vocabulary and three competition-season
  names frozen as ordinal-encoder categories. `models/MODEL_CARD.md` §6.1 lists it in full. DECISIONS
  §3 is worded absolutely ("never commit StatsBomb data of any kind"), so this is the owner's call to
  make explicitly rather than a thing to leave implicit.

## Genuinely outstanding

- **`pressing proxy`'s CSV header is still the wrong one.** `cli.py` writes the v0.1.0 lines: it quotes
  the *tracking* metric's phase means (0.26 / 0.47 / 0.70) above proxy values whose own gradient is
  +0.078, and it collapses the label-composition range to a single "r ~ 0.18". `pressing.proxy`
  already has the corrected text in `recovery_phase_caveat()` and `pressing season` already uses it.
  This is a small change in one file and it is the last place in the tool that tells a reader
  something measurably false. While in there: with the model now at r = 0.369 the composed range is
  0.10 to 0.17, so the season header's "0.10 to 0.18" wants the same edit.
- **A per-player counter-press number is still not defensible, and no amount of code fixes that.**
  The conditional baseline landed; the residual it produces separates players at p = 0.67 on the
  bundled match and every printed interval covers zero. `pressing.baseline` puts the requirement at
  roughly 233 nearby losses per player for the level and 194 for the residual against a median of 24
  in a match — eight to ten full matches each. The only real fix is more tracked matches, and there
  are two on earth I can get.
- **The conditional baseline is in-sample when there is only one match.** With two or more matches the
  residual is out-of-fold by construction; with one it is fitted on the losses it scores.
  `scored.attrs["residual_out_of_sample"]` carries the flag and page 6 prints it, but an in-sample
  residual is a weaker object and the report is honest about it rather than fixed.
- **The label calibration table in the README was not re-measured this round.** It needs both the
  StatsBomb cache and Metrica's CSVs and takes hours. It is the one table in the README carrying
  figures from an earlier run, and the README says so where it prints it. Re-run it before the next
  tag; the two independent runs recorded so far disagree between r = 0.26 and r = 0.47 on the camera
  variant, which is the widest uncertainty anywhere in this project.
- **Proxy predictions are heavily shrunk.** Held-out prediction spread is 0.451 of the label's, which
  is what a squared-error fit does when most of the target is unexplained. Ranks are usable; levels are
  not. An absolute-error objective was tried: +0.0098 Spearman with a match-clustered bootstrap CI of
  [−0.0087, +0.0273], i.e. nothing, and rank-transforming the target made it worse. The negative
  result is recorded in `models/MODEL_CARD.md` §7.1; a quantile objective is the next thing to try.
- **Season mode's market-value join is name-based** and drops ambiguous names rather than guessing
  (16 of 462 on the measured run). A StatsBomb-to-Transfermarkt id mapping would remove the guesswork
  entirely, and does not exist in open data.
- **Nothing adapts the plotting stage to machine load.** `pressing.adaptive_jobs` now measures the CPU
  share this process is actually getting and picks the worker count from it, which is the scoring
  stage. The font scan and the panel rendering are unmanaged, and on a loaded machine they are what is
  left: with twelve CPU burners running, the demo measured 135.26 s of wall clock against 46.2 s of
  CPU, a 9.0x inflation on 1.1x the work.
- **`--all-panels` is expensive and unoptimised.** It is opt-in and outside the six-page contract, so
  nothing has ever been done about its cost: on the bundled match it adds 255 PNGs and takes the demo
  process from 39.4 s to 97.9 s of CPU, a factor of 2.5.

## Fixed in v0.1.1 — recorded so the ledger is honest about what moved

- The player table's field-position problem is now **measured and printed on page 6**, not just
  described in a README: r = +0.98 over 27 players against a position-only model, the per-phase
  permutation p-values in the column headings, a residual column with a bootstrap interval, and cells
  suppressed below 11 losses.
- The cover prints r(recovery, regain) = +0.07 between the two rows that invite the confusion, and a
  `Better is` column so that "+0.48 s slower to win the ball back" is no longer styled like a gain.
- The proxy refit is deterministic across processes; both artefacts carry `train_match_ids` and
  `held_out_match_ids`; `score_held_out` refuses to score the refit model on its own training matches.
  Every proxy figure moved with the fix (held-out r 0.3907 → 0.3692) and the old three-decimal
  "reproducible to within 0.001" claim is gone, because it was not.
- The season scatter's correlation note is legible, and every colourbar in the project carries numeric
  ticks.
- There is a top-level `LICENSE` (MIT, Copyright (c) 2026 Oscar Dalton), which `pyproject.toml` had
  been declaring without shipping. It states explicitly that it does not cover the two vendored
  Friends of Tracking notices, the Metrica-derived demo asset or the Transfermarkt snapshot.
- The README no longer duplicates the model card's leave-one-competition-out and permutation-importance
  tables. Duplicating them is what let the two documents drift 36% apart on `dist_opp_goal`.
- `tests/test_pitchcontrol.py`'s stale "diverges to ~1e42" docstring is gone; the integrator does not
  diverge at a coarse timestep and the guard protects accuracy the renormalise clamp would hide.

## Deliberately not done

- **No per-player attribution from the proxy.** 360 freeze-frames carry no player identities. Anything
  that looked like a per-player event-only table would be invented.
- **No EPV from the proxy.** `predict()` emits `epv_conceded` as NaN on purpose: EPV needs the
  opponent's control surface, which an event stream cannot supply. A report built on proxy output must
  not print an EPV number.
- **No claim that this tool ranks players by pressing.** It ranks them by where they play. That is the
  finding, it is printed on the artefact, and no amount of presentation is allowed to soften it.
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
  someone actually ran, or attributed to the document that ran it. The README ends with the list of
  commands that regenerate its numbers, and states that rule in those words.
- **Wall-clock timings are not evidence unless the load average is beside them.** Measured this round
  on one laptop, minutes apart: the same demo command took 14.95 s of wall clock on 41.5 s of CPU at a
  load average of 4.9, and 135.26 s on 46.2 s of CPU at a load average of 41 rising to 72. Quote CPU
  seconds, deterministic counts and correlations freely; quote wall clock only with the load it was
  taken under, and never quote the best run.
- Downloads are polite: at most one request per second per host, exponential backoff, everything cached
  under gitignored `data/raw/`, and never re-fetched once cached.
- StatsBomb data is never committed in any form. Check with `git ls-files data`, not the working tree.
- One document owns each measurement. `models/MODEL_CARD.md` owns the proxy tables, `validation/`
  owns the pitch-control comparison, `pressing.baseline`'s docstring owns the baseline's own
  cross-validation, and the README quotes them with attribution rather than keeping a second copy.
