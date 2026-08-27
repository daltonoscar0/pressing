# pressing

**What it is.** When a team loses the ball, the next five seconds decide whether they get it straight
back or spend the next minute defending. `pressing` measures exactly that. Give it a match's tracking
data and it finds every moment possession changed hands in open play, works out how much of the pitch
the team that just lost the ball controlled over the following five seconds, and prints a six-page PDF:
where the losses happened, which five hurt most, how the two teams compare, and a per-player table. It
also ships a weaker event-only estimate for matches that have no tracking, and a season mode that ranks
a whole competition and puts the ranking next to what the players cost.

**Who it is for.** A coach or an analyst who wants the counter-press described by a number that comes
from where the players actually were rather than from someone's eye — and anyone who wants to read how
such a number is built, and where it stops being trustworthy. Everything below that looks like a claim
is a measurement, and the command that produces it is written next to it.

## Read this before you read a single number

Three things are measured, they point the same way, and they are the most important sentences in this
repository. The match-level ones are re-measured on every run of the tool and printed on the PDF
itself, so the paper a coach holds says what this section says rather than trusting this file to have
been kept up to date.

### 1. The per-player recovery column is 95% a map of where each player plays

Recovery is a share of modelled pitch control, and control of ground is mostly geography. On the
bundled match the mean recovery share is **0.257** for losses in a team's own build-up third and
**0.701** for losses in the opponent's third — a near-threefold gradient that has nothing to do with
pressing intensity.

That gradient survives all the way up into the player table, which is the output a coach will actually
circulate:

| what was fitted | on what | result |
|---|---|---|
| a straight line in the attacking fraction `x_norm` of the turnover | 255 scored losses | r = +0.9062, **R² = 0.8212** |
| a **cubic in `x_norm` and nothing else**, fitted at loss level, averaged over each player's nearby losses, against the printed Recovery column | 27 players | r = **+0.9771**, **R² = 0.9547**, Spearman +0.9402 |
| the player's own mean turnover position, on its own | 27 players | r = +0.9789, **R² = 0.9583** |

**Squared, that is 95% of everything that separates one player from another in that column.** Sorting
it sorts the squad by where it plays. What is left after position is removed has sd 0.0186 against the
column's own sd of 0.0813 — under a quarter of the spread.

### 2. "Compare within a phase" — the remedy this report used to prescribe — does not rescue it

Each per-phase player column gets a 5,000-shuffle permutation test: hold every player's loss count
fixed, reshuffle the values among players, and ask how often chance spreads the per-player means as
far apart as reality did.

| column | players | observed sd | null median | p |
|---|---:|---:|---:|---:|
| build-up | 14 | 0.0174 | 0.0221 | **0.88** |
| middle | 21 | 0.0355 | 0.0289 | **0.12** |
| final third | 19 | 0.0336 | 0.0279 | **0.17** |
| pooled recovery | 26 | 0.0581 | 0.0415 | 0.013 |

Shuffling **inside each squad**, so that a team-level offset cannot masquerade as a player one — which
is what the report itself does — those three p-values are 0.97, 0.86 and 0.16. Either way: **in
build-up the real spread between players is smaller than chance.** The only column that clears the test
is the pooled one, and the pooled one is the 95%-field-position column above.

### 3. The conditional baseline is now built, and it says there is nothing there

`pressing.baseline` fits what a team losing the ball *in that place, in that shape* was already going
to get — a cubic in `x_norm`, the width, and four shape features read off the tracking frame at the
turnover — and the player table prints `recovery − expected` beside the level. Cross-validated by
match it reaches R² = 0.9240 out of sample against 0.8368 for position and width alone
(`pressing.baseline`, measured over the 487 scored losses of Metrica games 1 and 2).

Then the residual is tested the same way as everything else, and:

- it separates players at **p = 0.67** on the bundled match;
- **every one of the 22 printed players has a 95% interval on it that includes zero** — 0 of 22 clear;
- the expectation was fitted on the one match it scores, so it is in-sample and if anything flatters
  itself.

### So, in this project's own voice

**A per-player counter-press number is not defensible on two tracked matches, and this repository will
not pretend otherwise.** The counts are solid — who was near the ball when it was lost, how often, in
which third — and everything to the right of them is context for those counts, not a ranking of
players. The median player in the bundled match has 24 nearby losses; `pressing.baseline` measures the
requirement at roughly 233 to separate two typical players on the level and 194 on the residual, which
is eight to ten full matches each. No cell prints at all below 11 losses, which silences every number
on 6 of the 28 rows the bundled match produces, and where a value does print its typical 95% interval
is 0.12 wide against a column spanning 0.24 top to bottom.

**And the metric is not ball recovery.** Over the same 255 losses, the modelled recovery share
correlates with the event stream's plain fact of whether the ball came back inside five seconds at
**r = 0.071**. The cover of the PDF prints those two numbers one above the other and now prints that
correlation between them.

Two matches on earth (that I can get) have both true 25 Hz tracking and hand-tagged events. Read every
statement above as being about those two matches, because that is all it is.

## What you get

`pressing demo` writes a six-page A4 PDF:

1. **Cover** — fixture, provider, frame rate, a head-to-head table with a `Better is` column and the
   recovery-versus-regain correlation between the two rows that invite confusion, the field-position
   caveat, and a rolling-recovery sparkline.
2. **Loss map** — every possession loss on one pitch in team colours, plus the build-up / middle /
   final-third table.
3–4. **Top moments 1–4**, two per page: a pitch-control heatmap each, captioned with the minute, who
   lost the ball, who won it, the recovery share and the EPV swing.
5. **Moment 5 in detail** — one full-width panel plus a t0 / t0+2s / t0+5s thumbnail strip.
6. **Player table**, under a boxed warning that states this match's own measured version of §1, §2 and
   §3 above: minutes, losses nearby, recovery, the residual against the conditional baseline with its
   bootstrap interval, EPV swing and the three phase splits, each phase heading carrying its own
   permutation p-value.

Every number in that warning is measured on the match in hand by `pressing.report.measure_disclosure`,
so a report built on a different match prints that match's numbers rather than the ones above.

`--all-panels DIR` additionally dumps one heatmap per scored possession loss — measured, **255 PNGs**
on the bundled match. It took the process from 39.4 s to 97.9 s of CPU in the runs timed below. The
PDF itself is always six pages.

## Install

Requires Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv venv --python 3.11
uv sync --all-extras
uv run pressing --version      # 0.1.0
```

## Quickstart

```sh
uv run pressing demo
```

No network, no data to fetch: the demo runs on a derived Metrica match committed under `data/demo/`
(see [Data sources and licences](#data-sources-and-licences)). Real output:

```
[   2.5s] load (2.5s)  3,278,389 tracking rows, 1,745 events
[   8.8s] score (6.2s)  256 losses, 255 scored, jobs=6 (adaptive, CPU share 1.00)
[   9.8s] moments (1.0s)  5 panels at print resolution, rank=costly
[  13.4s] report (3.5s)  out/demo_report.pdf
255 of 256 losses scored on 6 worker(s) (adaptive)
cost: 14.8s wall clock since import (of which 13.4s in the pipeline; interpreter start-up is in
neither), 39.4s CPU including imports and workers, 1-min load average 5.16 -> 6.09
out/demo_report.pdf
```

A six-page PDF of **1,398,722 bytes**, and the 256 possession losses of the bundled match scored on the
way. The worker count is not a constant: `pressing.adaptive_jobs` measures what share of a core the
machine is actually handing this process and drops to the serial path when there is none to spare.
`--jobs` overrides it and changes no number.

### That wall clock, against the two-minute budget

**Read the cost line, not the stage clock, and do not expect these wall-clock figures to reproduce.**
The acceptance criterion is a full PDF in under two minutes on a fresh clone. Five runs, timed with
`/usr/bin/time -p` on a ten-core laptop that was shared throughout with other heavy jobs, with the
one-minute load average recorded immediately before and after each:

| run | font cache | load average, before → after | `real` | CPU (`user` + `sys`) | PDF |
|---|---|---|---:|---:|---:|
| 1 | cold (empty `MPLCONFIGDIR`) | 3.90 → 4.61 | 24.00 s | 57.3 s | 1,398,722 B |
| 2 | warm | 4.88 → 5.68 | 14.95 s | 41.5 s | 1,398,722 B |
| 3 | warm | 5.68 → 5.48 | 15.01 s | 41.6 s | 1,398,722 B |
| 4 | warm, `--all-panels` | 6.75 → 5.95 | 72.58 s | 98.1 s | 1,398,722 B + 255 PNGs |
| 5 | warm, **twelve CPU burners started deliberately** | 19.68 → 72.42 | **135.26 s** | 46.2 s | 1,398,722 B |

**Run 5 is the point of this table.** It is the same command, on the same laptop, minutes after run 2,
with twelve busy-wait processes of my own added: **9.0x the wall clock on 1.1x the CPU**, and it misses
the two-minute budget while doing exactly the same amount of work. Nothing about the code changed
between rows 2 and 5. The `real` column is a property of an afternoon's machine load and should not be
expected to reproduce anywhere; the CPU column moves by a few percent and is the number that travels.
That is why the tool now prints its own CPU time and the load average at both ends of every run, and
why `pressing.adaptive_jobs` measures the share of a core it is actually being given rather than
trusting a constant. If a run of yours takes four minutes, read your load average before concluding
anything about this code.

What is deterministic and does reproduce: `/Count 6` pages, 1,398,722 bytes on every one of those five
runs, 256 losses detected, 255 scored, 255 panels from `--all-panels`. Two of those PDFs differ in
exactly 62 bytes, and all 62 are the creation timestamp, the modification timestamp and the
clock-derived `/ID`.

## The commands

Four subcommands. `report` and `demo` need tracking data; `proxy` and `season` work from an event
stream alone and are considerably weaker for it.

### `pressing demo`

```sh
uv run pressing demo --out out/demo_report.pdf --team home --rank costly
```

Runs the bundled match end to end. `--rank` chooses which five losses the moment pages print:
`costly` (default, largest EPV swing), `recovered` (highest recovery share) or `conceded` (lowest).
`recovered` is the spec's literal reading and is deliberately not the default: because recovery is
driven by field position, it returns final-third losses and essentially nothing else — measured, 5 of
5 for each team on the bundled match. `costly` is the ranking that can show a press that went wrong,
and on this match it returns nine middle-third losses and one build-up loss across the two teams.

### `pressing report`

```sh
uv run pressing report data/demo/metrica_game_1_tracking.parquet \
    --events data/demo/metrica_game_1_events.parquet \
    --team home --out out/report.pdf
```

```
loaded 3,278,389 tracking rows and 1,745 events
scored 255 of 256 possession losses on 6 worker(s) (adaptive, CPU share 1.00)
out/report.pdf
```

Progress goes to stderr and the path to stdout, so `$(uv run pressing report ...)` is the path and
nothing else. That run wrote a 1,398,707-byte PDF (15.76 s `real`, 42.4 s CPU, warm font cache, at a
one-minute load average of 6.16 → 6.99; see the caveat above about quoting either).

Accepts three shapes of input: a `.parquet` tracking asset written by this package (with `--events`),
a directory holding Metrica's `Sample_Game_N/` folders, or one of Metrica's raw
`Sample_Game_N_RawTrackingData_*.csv` files. `--jobs` sets the worker count; the scores are identical
serially.

### `pressing proxy`

Estimates the same quantity from a StatsBomb events file when there is no tracking. It writes a CSV,
not a PDF: three of the report's six pages are pitch-control heatmaps and its player table is built
from tracked positions, and none of that exists in an event stream.

```sh
uv run pressing proxy data/raw/statsbomb/events/3825739.json --out out/proxy_estimates.csv
```

```
66 open-play turnovers estimated
  build_up     n=   1  mean recovery 0.497
  final_third  n=  33  mean recovery 0.539
  middle       n=  32  mean recovery 0.484
recovery values are comparable only within a phase
out/proxy_estimates.csv
```

The CSV carries its own caveats above the header row:

```
# pressing proxy estimates: counter-press control share per open-play turnover
# Event data: StatsBomb open data (github.com/statsbomb/open-data), used under the StatsBomb Public Data User Agreement. Non-commercial personal analysis; no data redistributed.
# recovery is an ESTIMATE from an event-only model, not a measurement. The model composes to r ~ 0.18 against the tracking-derived metric it stands in for; see models/MODEL_CARD.md before quoting any single value.
# recovery is driven by WHERE the ball was lost (Metrica game 1 phase means: build-up 0.26, middle 0.47, final third 0.70). Compare only within a phase.
# epv_conceded is empty on purpose: EPV needs the opponent's control surface, which cannot be recovered from events.
match_id,turnover_id,t,losing_team,winning_team,phase,recovery,epv_conceded,estimated
3825739,3825739:4,41.75,Sporting Gijón,Real Madrid,final_third,0.585145264908211,,True
3825739,3825739:6,102.727,Sporting Gijón,Real Madrid,middle,0.4101178894219955,,True
```

**Two lines of that header are known to be wrong and are printed here as they actually come out.** The
fourth line quotes the *tracking* metric's phase means (0.26 / 0.47 / 0.70) above a column whose own
gradient is +0.078, so a proxy build-up value of 0.47 — the proxy's own average — reads as an
extraordinary press. The third rounds the top of a range to a single number. `pressing.proxy` has a
corrected header (`recovery_phase_caveat`, which prints the model's own means and then forbids scaling
against the tracking ones) and `pressing season` already uses it; the `pressing proxy` subcommand in
`cli.py` does not yet. It is the first item in [TODO.md](TODO.md).

Read [the metrics table below](#the-event-only-proxy-honest-numbers) before you quote any single value:
the proxy is a rough ranking device, and every output built from it — the CSV header, the stdout
summary, and the footer and metadata of any PDF rendered from proxy numbers — says so.

### `pressing season`

Runs the proxy across a whole competition, aggregates per player per 90 minutes, joins Transfermarkt
market values, and writes a ranking CSV plus a scatter PNG.

```sh
uv run pressing season --source statsbomb --competition "La Liga" --season "2015/2016" \
    --limit 40 --offline --out-dir out/season
```

```
La Liga 2015/2016: 380 matches in the season manifest
processing 40 of 380 matches, spread across the season
[20/40] cached
[40/40] cached
[20/40] scored
[40/40] scored

La Liga 2015/2016: 40 of 380 matches, 2,944 open-play turnovers, 462 players (133 with enough minutes to rank)
market-value join: 406 of 462 (87.9%) priced; 15 with no valuation in the season window; 16 dropped as ambiguous rather than guessed; 25 not found
values from the cache, data current to 2026-06-12; scatter has 105 points
warning: only 40 of 380 matches were processed, so minutes and every per-90 rate cover a fraction of the season.
per-90 rates from a sample are noise; treat the ranking accordingly
out/season/la_liga_2015_2016_counterpress.csv
out/season/la_liga_2015_2016_counterpress_vs_value.png
```

`--offline` uses only what is already cached; without it, missing matches are fetched at one request
per second. A full La Liga 2015/16 pull is 380 events files plus 380 lineups, so about 13 minutes of
polite downloading before any work starts. `--season` defaults to the season with the *most* matches,
not the most recent one: La Liga's latest open season has 35 matches and its largest has 380.

## The metric

At every open-play possession loss:

1. Take the frame where the ball changed hands. The team that **lost** it is the counter-pressing team.
2. At six instants — the moment of the loss and each second after it out to five seconds — compute a
   pitch-control surface on a 50 x 32 grid: for every cell, the probability that the counter-pressing
   team would win the ball if it arrived there. The model is the Spearman one — time to intercept from
   each player's position and velocity, a logistic ramp on arrival, integrated forward along the ball
   flight.
3. Average each surface over the pitch to get that team's share of control at that instant.
4. **Recovery** is the mean of the six shares. Read it as a probability: 0.5 is dead even, above 0.5
   means the pressing team owns more of the pitch than the team that just won the ball.

A second weighting, `recovery_epv`, weights each cell by how dangerous it is instead of treating all
grass equally. It is carried alongside but is *not* the headline, because it measures worse. Over the
255 scored losses of the bundled match:

| weighting | min | median | mean | max | sd |
|---|---|---|---|---|---|
| `recovery_area` (headline) | 0.1698 | 0.5629 | 0.5559 | 0.9151 | 0.1767 |
| `recovery_epv` | 0.4351 | 0.8387 | 0.8025 | 0.9761 | 0.1233 |

The EPV weighting never goes below 0.43 on this match, sits three quarters of the way up its own
range, and buys nothing for it: the two rank losses almost identically (Pearson r = 0.9572). A number
a coach reads against a stated neutral of 0.5 is worth more than one that lives between 0.75 and 0.95.

### The caveat, with the numbers attached

Mean `recovery_area` on the bundled Metrica match, by the third of the pitch the ball was lost in,
counted from the losing team's own goal:

| phase | away | n | home | n | pooled | n |
|---|---|---|---|---|---|---|
| build-up | 0.238 | 13 | 0.279 | 11 | **0.257** | 24 |
| middle | 0.439 | 53 | 0.494 | 60 | **0.468** | 113 |
| final third | 0.703 | 61 | 0.699 | 57 | **0.701** | 118 |

That is a near-threefold gradient from one end of the pitch to the other, and it dwarfs the difference
between the two teams (0.238 against 0.279 in build-up). Compare within a phase, mind the sample size
— build-up rests on 24 losses out of 255 in this match — and see §1–§3 above for how far that
instruction actually gets you, which is not far.

### What the metric is not

**It measures spatial dominance, not ball recovery.** The event stream also records the plain,
checkable fact of whether the losing team had the ball again inside five seconds. On the bundled match
that happens 95 times in the 256 detected losses (37.1%; 37.3% of the 255 scored ones, and 37.5% /
36.7% split by team, which is what the cover prints), and its correlation with the modelled recovery
share is **r = 0.071** (0.083 for the EPV weighting). Those two things are almost unrelated.
Controlling space after a turnover is a reasonable description of what a counter-press is *trying* to
do, and this tool says nothing stronger than that.

Two further limits, both reported per loss rather than hidden:

* The ball has no tracking row in **39.1%** of the bundled match's 145,006 frames, so inside a
  five-second window the ball position is sometimes carried forward from the last known frame. Every
  loss reports how often that happened in `n_ball_filled`.
* One open-play loss in the bundled match cannot be located at all — no tracked ball at its end frame
  and no event coordinates either side. It is reported with `location_source == "none"` and skipped,
  not guessed: 256 losses detected, 255 scored. Having no position it has no pitch third either, which
  is why the cover counts 256 and page 2's phase table adds up to 255; both pages say so.

## Validation: pitch control against a published reference

`pressing.pitchcontrol` is checked against the
[Friends of Tracking](https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking)
implementation of the same model, at commit `7ed3ff85`, vendored byte-for-byte under
`validation/reference/` with its MIT licence. `validation/README.md` documents the method; the executed
notebook with its outputs is at `validation/pitch_control_validation.ipynb`.

Re-executed from a clean kernel for this file, over **all 257 `BALL LOST` events** of Metrica sample
game 1, on the reference's own grid, 0 errors across all 20 code cells:

| configuration | pooled Pearson r | mean per-event r | min per-event r | events below 0.95 | max abs difference |
|---|---|---|---|---|---|
| reference's own convention (`clamp="raw"`, float64) | **1.0000000000** | 1.0000000000 | 1.0000000000 | 0 of 257 | 3.941e-15 |
| the same in float32 | 1.000000 | 1.000000 | 1.000000 | 0 of 257 | 2.645e-03 |
| **shipped default** (`clamp="renormalise"`, float32) | **0.999563** | 0.999193 | 0.943349 | **1 of 257** | 6.457e-01 |

411,200 cells compared per configuration; mean absolute difference 1.737e-16 and 7.183e-04 in the first
and last rows. Guard: correlation against a y-flipped reference falls to 0.618073, so the surfaces
really are aligned. Those are the digits the notebook prints; earlier runs recorded the first row to
ten decimals as 1.0000000000 and the last to 0.9995631111, and two independent re-measurements agreed
to nine.

**Both rows matter, and the second is not hidden.** At the reference's own convention we reproduce it
to floating-point rounding. The shipped default deliberately diverges: forward Euler takes its final
integration step whole, so the two teams' raw probabilities can sum above 1 — measured maximum
2.800681, and the reference's own surface reaches 1.005366 at one cell. We renormalise those cells
rather than print a probability above 1 in a coach-facing report, and that one choice accounts for the
whole gap, including the single event that falls to r = 0.943.

**Speed.** The reference computes one surface with a Python loop over cells and this package computes
it with array arithmetic. In the rerun above, on the same 257 events and the same grid, the reference
took 151.61 s in total (0.5899 s per surface) against 2.48 s (0.0096 s per surface) — a **61.2x**
ratio, at a one-minute load average of 4.35 to 6.12. **Only the ratio is worth anything**: the
committed notebook's own timing cell, from a different run on the same laptop, reads 248.28 s against
4.04 s — the absolute totals differ by a factor of 1.6 and the ratio moves from 61.5x to 61.2x. The
correlations do not move at all. `validation/README.md` records both runs with what is known about the
machine they were taken on. The margin is why the
demo fits its runtime budget: a straight port of the reference loop would spend minutes on pitch
control in a single match.

**What this is not.** It shows we agree with one published implementation of one model. It says nothing
about whether that model predicts football, and it does not test the loader, the offside rule or the
product's choice of ball position — those have their own tests.

## The event-only proxy: honest numbers

Most matches have events and no tracking. `pressing proxy` predicts the counter-press control share
from event features alone. `models/MODEL_CARD.md` is the full account and is the authority on every
figure in this section; this is the short honest version, and where the two ever disagree the card
wins because it is the document that owns the measurement.

**The one-line summary: this model predicts an approximation of the metric, not the metric.** Against
its own label it reaches **r = 0.369** on held-out matches. The label itself reproduces the true
tracking-derived value at r = 0.26 to 0.47 (measured below). If the prediction reaches the truth only
through the label, those compose to **r ≈ 0.10 to 0.17** against the quantity the tracking report
prints. If you need a number a coach can act on, use tracking; this is a rough ranking device and it is
watermarked as an estimate wherever it appears.

**The spec's training plan could not be built.** It asked for a Metrica + SkillCorner overlap.
SkillCorner ships no event stream ("We are not an event data provider ourself"), its ten open matches
are Australian A-League and StatsBomb open data has no Australian competition, and its
`dynamic_events.csv` is derived from the same tracking that would produce our labels — training on it
would be circular. The labels come from **StatsBomb 360 freeze-frames** instead, split by match.

**The data.** 101 matches downloaded across three competition-seasons (La Liga 2020/21, 1. Bundesliga
2023/24, Ligue 1 2022/23); **6,896** open-play turnovers detected from `possession_team` transitions;
**6,195** labelled (89.8%) over the 97 matches that contribute a label. The four that contribute none
ship a well-formed 360 file whose `event_uuid`s intersect their own events file at exactly zero, which
is an upstream inconsistency the pipeline degrades on rather than papering over. The counts, the
competitions, the split and the metrics below are all recorded inside the committed artefacts and can
be read out in a second — see the reproduce section.

The split is **by match, never by turnover**: 68 training matches (4,394 rows) against 29 held-out
matches (1,801 rows), with an empty match-id intersection asserted in the tests. Splitting by turnover
would put two losses from the same match five seconds apart on opposite sides of the split.

The field-position gradient that dominates the true metric is almost gone from the label — mean
`recovery_360` is 0.4636 in build-up, 0.4894 in the middle and 0.5420 in the final third, a spread of
0.078 where the tracking truth has 0.444. The model inherits that compression, which is why **no
output may print the tracking metric's phase means as a scale for a proxy number** (see the note on
the CSV header above). `models/MODEL_CARD.md` §4 explains why: it is the broadcast camera.

### Held-out performance, against the baselines that matter

29 held-out matches, 1,801 turnovers, target `recovery_360`, as recorded in the split artefact's own
metadata:

| model | r | Spearman | MAE | RMSE | residual r after removing turnover `x` |
|---|---:|---:|---:|---:|---:|
| predict the training mean | — | — | 0.1029 | 0.1299 | 0.0806 |
| predict the mean of the pitch third | 0.1690 | 0.1813 | 0.1006 | 0.1283 | 0.0581 |
| **predict from turnover `x` alone** | **0.1974** | 0.2107 | 0.1002 | 0.1277 | — |
| **full model** (29 features) | **0.3692** | 0.3884 | 0.0927 | 0.1212 | **0.3260** |

`r` for the mean predictor is undefined, not zero: its predictions are constant. `residual r` is the
correlation between prediction and label after a smooth function of turnover `x`, fitted on training
rows only, has been subtracted from both.

**Those figures are quoted to four decimals because the refit is now deterministic.** Up to v0.1.0 it
was not — `build_dataset` appended each match as its worker finished, so the training set's row order
was a race and the early-stopping slice moved with it. v0.1.0 printed r = 0.390 from that regime and
claimed cross-process agreement to within 0.001; two independent auditors measured 0.3803 and 0.3820,
and the honest description of the old number is the top of a 0.368–0.391 band. The row order is now
canonical and three fresh processes agree bit for bit; `models/MODEL_CARD.md` §5.1 records how that was
established and what it cost.

**Two artefacts ship, and scoring the wrong one lies to you.** `models/proxy_gbr.pkl.gz` is refitted on
all 6,195 labelled rows, because that is the model you would want to run — so the held-out matches are
*inside* its training set and scoring it on them returns an in-sample number that means nothing.
`models/proxy_gbr_split.pkl.gz` is the model the table above describes. Both now carry `fitted_on`,
`train_match_ids` and `held_out_match_ids`, and `score_held_out` raises rather than returning the
flattering number.

**Read the table honestly.** The full model beats every baseline, and it knows something beyond field
position — the x-only baseline reaches 0.197 and the residual correlation is still 0.326. But r = 0.37
against a label is not a good number in absolute terms, and the predictions are badly shrunk: on the
held-out matches the label has sd 0.1299 while the predictions have sd 0.0586, a spread ratio of 0.451.
That is what a squared-error fit does when most of the target is unexplained. **Report ranks, never
levels.** A rank-appropriate objective was tried and the result was negative; the card records it.

The model card also carries, and this README deliberately no longer duplicates, the leave-one-
competition-out table (three competition-seasons, a spread of 0.004 across them), the permutation
importance with its seed and repeat count, the StatsBomb `counterpress`-flag companion (60.8% of all
6,896 turnovers carry one, 64.2% of the 6,195 labelled ones) and the full feature list. v0.1.0 printed
its own copies of those tables and they drifted out of agreement with the card's; one document owns
them now.

### How good is the label itself?

This is the number that decides how much the proxy can ever be worth, and it can be measured directly:
the Metrica matches have both real 25 Hz tracking and real events, so the true metric and a simulated
freeze-frame version of it can be computed on the same 487 possession losses. The simulation degrades
the truth one step at a time, driven by **real** StatsBomb 360 sampling geometry — the offsets frames
actually landed on and the actual `visible_area` polygons, anchored to the ball — transplanted onto
Metrica, so the camera model is measured rather than invented. The run below drew 1,381 real 360
windows from 20 matches; it was measured for the v0.1.0 tag and has not been re-run since, because it
needs both caches and hours. The command that produced it is in the reproduce section.

| variant | n | Pearson r | Spearman | MAE | mean | sd | phase gradient |
|---|---:|---:|---:|---:|---:|---:|---:|
| `recovery_area` (the shipped metric) | 487 | 1.0000 | 1.0000 | 0.0000 | 0.5530 | 0.1843 | **+0.4332** |
| `true_uniform` (harness self-check) | 487 | 1.0000 | 1.0000 | 0.0000 | 0.5530 | 0.1843 | +0.4332 |
| `sparse_offsets` (event-driven sampling) | 460 | **0.9934** | 0.9930 | 0.0145 | 0.5547 | 0.1890 | +0.4353 |
| `sparse_zero_velocity` (+ no velocities) | 460 | **0.9896** | 0.9891 | 0.0197 | 0.5549 | 0.1915 | +0.4416 |
| `sparse_zero_velocity_camera` (+ broadcast truncation) | 441 | **0.2609** | 0.2490 | 0.1660 | 0.5028 | 0.1659 | +0.0755 |
| `..._camera_visible` (+ integrate inside the polygon only) | 441 | **0.0366** | 0.0381 | 0.2031 | 0.4795 | 0.1413 | −0.0473 |

The harness self-check is exact: `true_uniform` reproduces `counterpress.score_match`'s `recovery_area`
with a maximum absolute difference of **0.0** over all 487 losses, so the degraded rows are produced by
code known to agree with the shipped metric.

What it says:

1. **Event-driven sampling costs almost nothing** (r = 0.993). The median window holds seven
   freeze-frames (`models/MODEL_CARD.md` §2), which is plenty.
2. **Assuming every player is stationary costs almost nothing** (r = 0.990). This is the assumption
   everyone worries about and it is not the problem.
3. **The broadcast camera is the whole problem.** Only the players inside the polygon exist, and the
   ones outside it are the ones far from the ball — exactly the asymmetry that creates the
   field-position gradient. The gradient collapses from +0.433 to +0.076.
4. **Restricting the integral to the visible polygon is worse, not better.** It looked like the more
   defensible choice; it was measured and rejected. `recovery_360` is the shipped target.

**How much worse the camera makes it depends on which windows you transplant, and the honest answer is
a range.** The run above put the camera variant at r = 0.261; the independent run recorded in
`models/MODEL_CARD.md`, which drew its windows from the full 97-match set, put it at r = 0.469, and a
sweep there that dilated the polygons until the retained player count matched a real freeze-frame's
still only reached 0.50. So: **the 360-derived label reproduces the true tracking-derived metric
somewhere around r = 0.26 to 0.47**, and no amount of modelling lifts that ceiling, because it is a
property of what the camera saw. Neither run is more authoritative than the other, and the card no
longer states the flattering end of that range as though it were the answer.

## Season mode

The run above (La Liga 2015/16, 40 of the season's 380 matches, everything from cache, offline)
measured:

| | |
|---|---|
| matches processed | 40 of 380 |
| open-play turnovers | 2,944 |
| players with any minutes | 462 |
| players with >= 270 minutes, so ranked | 133 |
| market values joined | **406 of 462 (87.9%)** |
| dropped as **ambiguous** rather than guessed | **16** |
| no valuation inside the 2015-07-01..2016-06-30 window | 15 |
| not found at all | 25 |
| points on the scatter | **105** (of the 133 ranked, after excluding goalkeepers and players with no value) |

The join is by normalised name, in tiers — full name, nickname, nickname plus the club the player
appeared for that season, last two tokens. Of the 406 priced players, 297 were found by a Transfermarkt
nickname, 74 by the full name, 34 needed the club to disambiguate and 1 fell through to the last-two-
tokens rule. A name that still matched several Transfermarkt players is dropped with
`join_status='ambiguous'` and no value: attaching the wrong price to a real player is worse than
leaving the cell empty.

**What the scatter shows: nothing.** Counter-presses per 90 against market value, over the 105 plotted
outfield players:

| statistic | value |
|---|---|
| Pearson r (per 90 vs value in EUR) | −0.2129 |
| Pearson r (per 90 vs log10 value) | −0.0927 |
| Spearman rho | −0.0704 |
| 95% interval on the log-value r | [−0.279, 0.101] |

The interval straddles zero, and the chart now prints it: v0.1.0's scatter annotated a bare `r = -0.09`
and let a player label print straight through it, so the one number the chart exists to carry was
illegible. There is no visible relationship between how often a player counter-presses and what he is
worth in this sample, and the mild negative tilt in the raw-value column is what you would expect from
a handful of expensive forwards who press less, not evidence of anything. It is also 40 matches, not a
season, and the run says so on stdout and again in the CSV header.

Two things to keep straight in the output table. `counterpress_per_90` is a **count** — StatsBomb's own
`counterpress` flag, defined verbatim as a defensive action "within 5 seconds of an open play
turnover", the same window this project's metric uses — and involves no model at all; it is the
scatter's y axis and the ranking key. `mean_recovery_pressed` and `press_value_per_90` are **proxy
estimates** and inherit every caveat in the section above. Minutes come from the lineup files'
position intervals intersected with the real period clock, so a full match reads about 94 rather
than 90.

## Licence

**This project's own code is MIT** — see [`LICENSE`](LICENSE), Copyright (c) 2026 Oscar Dalton. That
covers `src/`, `tests/`, `validation/pitch_control_validation.ipynb` and the documentation, and
nothing else.

It deliberately does **not** cover the third-party material vendored beside it, which keeps its own
notices and must not be relabelled with this project's:

- `validation/reference/LICENSE` and `data/epv/LICENSE.friends-of-tracking` are the *Friends of
  Tracking* MIT grant, Copyright (c) 2021 Friends-of-Tracking-Data-FoTD. Two copies because the
  material is vendored in two places — the reference implementation under `validation/reference/` and
  the EPV grid under `data/epv/`. Both stay exactly as upstream wrote them.
- `data/demo/` is derived from Metrica sample data, which is published under **no licence at all**. No
  grant is offered or implied for those two parquet files; `data/demo/NOTICE.md` says so, records the
  upstream commit and the digests, and undertakes to remove them on request.
- `data/transfermarkt/` is CC0-1.0 over dcaribou's repackaging only, not over the underlying scraped
  rights.

## Data sources and licences

Nothing in this table is committed to git except where the row says so. `data/raw/` is gitignored in
full; `git ls-files data` is the check, not the working tree.

| Source | Used for | Licence | How this repo treats it |
|---|---|---|---|
| [Metrica Sports sample data](https://github.com/metrica-sports/sample-data) | The bundled demo match, the metric's measured behaviour, the pitch-control validation | **None published.** GitHub reports `license: null`, which is all rights reserved by default. The README asks only that users be responsible with the data and acknowledge the source. | Credited by name here, in the PDF footer of every page, and wherever its numbers appear. The raw CSVs are **never committed**. What is committed is a *derived* asset — restructured, re-coordinated, rotated, quantised to int16 centimetres, velocities dropped — solely so the demo runs offline on a fresh clone, with [`data/demo/NOTICE.md`](data/demo/NOTICE.md) recording exactly how it differs, the upstream commit `e706dd50`, the sha256 and git blob id of every upstream file it came from, and an undertaking to remove it on request. `tests/fixtures/metrica/` holds **synthetic** files that copy the format only. |
| [Friends of Tracking / LaurieOnTracking](https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking) | Pitch-control reference implementation; the EPV grid | **MIT**, Copyright (c) 2021 Friends-of-Tracking-Data-FoTD | Vendored byte-for-byte with the MIT notice beside it, under `validation/reference/` and `data/epv/`, with `PROVENANCE.md` files recording the commit and the sha256 of every file. **The EPV grid is theirs and has no published provenance** — no fitting code, no source dataset, no documentation of what it was fitted on. It is used because it is redistributable and already the right shape, not because it is authoritative, and the report labels it as the Friends of Tracking grid rather than as our estimate. |
| [StatsBomb open data](https://github.com/statsbomb/open-data) | Proxy training labels; season mode | **Restrictive.** The `LICENSE.pdf` "StatsBomb Public Data User Agreement" governs and is stricter than the repo README's attribution-only wording: clause 1.2.1 forbids you to "edit, distort, distribute, reproduce, sell or in any way provide the data to any external or third party"; clause 1.2.2 forbids commercially exploiting "the data or any analysis derived from the use of the Service"; clause 1.4 requires accrediting published analysis with the StatsBomb brand logo. | **Never committed, in any form** — it sits in gitignored `data/raw/statsbomb/`, fetched at one request per second. Attribution is stamped on every event-derived output: the proxy CSV header, the season CSV header, the scatter caption, the PDF footer. The committed model artefacts contain no rows, but they do embed StatsBomb's event-type vocabulary and three competition-season names as frozen encoder categories; `models/MODEL_CARD.md` §6.1 lists exactly what is in there. This tool is non-commercial personal analysis and carries no commercial framing. Note that handing a coach a PDF of StatsBomb-derived numbers sits close to clause 1.2.1 — read the agreement first. |
| [dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) | Market values for season mode | **CC0-1.0 over the repackaging only.** | CC0 covers dcaribou's repackaging, **not** the underlying rights: the data is scraped from Transfermarkt, whose own terms prohibit scraping. Values are Transfermarkt's community estimates — not fees, not wages, not audited. A column-reduced snapshot is committed under `data/transfermarkt/` so the join works offline, with its capture date and row counts in [`data/transfermarkt/NOTICE.md`](data/transfermarkt/NOTICE.md); the full tables stay in gitignored `data/raw/`. |
| [SkillCorner open data](https://github.com/SkillCorner/opendata) | Evaluated and **not used** | MIT | Rejected on the facts, not the licence: no event stream, no competition overlap with StatsBomb, and its derived events come from the same tracking that would generate our labels. |

## Known limitations, and what I would do next

Written for someone deciding whether to trust this, not to sell it.

1. **A per-player counter-press number is not defensible on this data**, and the three measurements at
   the top of this README are the reason. The conditional baseline that was supposed to be the fix is
   built and its residual is noise. What the player table is good for is counts.
2. **The metric measures spatial dominance, and only barely relates to actually getting the ball back.**
   r = 0.071 against the event stream's regain flag on the bundled match. Anyone who reads "recovery
   0.70" as "wins it back 70% of the time" is wrong.
3. **Only two matches on earth (that I can get) have both true tracking and hand-tagged events.**
   Metrica games 1 and 2. Every measured statement about the metric's behaviour rests on them, so read
   "on the bundled match" literally: it is one game, not a population. Almost every limitation here is
   really a sample-size problem, and more tracked matches is the single thing that would move all of
   them.
4. **The conditional baseline is fitted on the match it scores when there is only one match.**
   `scored.attrs["residual_out_of_sample"]` says so and page 6 prints it, but an in-sample residual is
   a weaker thing than an out-of-fold one, and on one match that is all there is.
5. **The 360 labels have no velocities and no player identities.** A freeze-frame is positions inside
   the broadcast polygon. The proxy therefore cannot do per-player attribution at all, and its label is
   an approximation whose size is measured above rather than assumed.
6. **The EPV grid is unauditable.** It ships with no provenance. `recovery_epv` and `epv_conceded`
   inherit that, which is one reason the headline weighting is plain area.
7. **Season mode's market-value join is name-based** and drops ambiguous names rather than guessing;
   the join rate is reported on every run. The ranking itself is StatsBomb's `counterpress` count per
   90, which is a hand-tagged volume statistic, not this project's model.
8. **No cross-provider tracking support.** The loaders read Metrica and the canonical parquet asset.
   SkillCorner's format is understood but unused (no events to pair it with).

## `sportio`

The specification has this package sit on top of `sportio`, a shared loader layer that normalises every
open soccer source into the canonical `events` and `tracking` frames. **That package does not exist on
this machine**, so `src/pressing/_io.py` implements the Metrica loaders internally behind a shim: it
tries `import sportio` first and delegates when the import succeeds, and `pressing._io.backend_name()`
reports which path is live. When the shared layer lands, the internal loaders can be deleted with no
change above them.

## Layout and tests

```
src/pressing/     _io  possession  pitchcontrol  epv  counterpress  phases  baseline
                  plots  report  demo  proxy  season  cli  schema
tests/            one module per source module; fully offline
validation/       the pitch-control notebook, its README and the vendored reference
models/           the proxy artefacts and their model card
data/demo/        the derived Metrica match the demo runs on, and its notice
data/epv/         the Friends of Tracking EPV grid, its licence and provenance
data/transfermarkt/  the reduced market-value snapshot and its notice
```

```sh
uv run ruff check .
uv run pytest -q
```

**629 tests collected, 629 passing**, and `ruff check` clean, as of the v0.1.1 tag: the run printed 629
progress dots and not one `F`, `E`, `s` or `x`, so nothing is failing, skipped or expected-to-fail. It
cost 299.6 s of CPU (226.9 user + 72.7 sys) on a contended machine; the wall clock it took there is not
worth quoting and is shorter than the CPU total because parts of the suite score losses on a worker
pool. The suite is slow on purpose — it scores real possession losses and builds real PDFs rather than
mocking them — so budget minutes, not seconds, and do not assume it has hung. It is offline: no test
reads anything under gitignored `data/raw/` or opens a socket, and `tests/test_demo.py` asserts that
the demo path never opens a `data/raw/` file.

Open items are in [TODO.md](TODO.md).

## Reproducing every number in this README

The house rule is that every figure here is either (a) measured by one of the commands below, run on
this machine while writing this file, or (b) explicitly attributed to the document that owns the
measurement — `models/MODEL_CARD.md`, `pressing.baseline`'s module docstring, or the committed
notebook. Nothing is quoted from memory, and where a run disagreed with an older note the newer
measurement is what is printed. Wall-clock timings are the exception the rule cannot fix: they are
printed with the load average they were taken under and should not be expected to reproduce.

**The two findings at the top of this README** — the field-position dominance of the player column and
the permutation tests. This is the same computation the PDF's own page-6 warning is built from, so the
paper and this file cannot drift apart:

```sh
uv run python - <<'PY'
import numpy as np, pandas as pd
from pressing import baseline
from pressing._io import read_tracking_parquet
from pressing.counterpress import per_player, score_losses
from pressing.demo import read_demo_events
from pressing.report import (DEFAULT_RADIUS_M, _credited_pressers, _player_rows,
                             _position_only_prediction, measure_disclosure, with_expected_recovery)
tracking = read_tracking_parquet("data/demo/metrica_game_1_tracking.parquet")
events = read_demo_events("data/demo/metrica_game_1_events.parquet")
losses = with_expected_recovery(score_losses(tracking, events, jobs=1), tracking)
players = per_player(losses, tracking, DEFAULT_RADIUS_M)
printed = pd.concat([_player_rows(players, "home"), _player_rows(players, "away")])
d = measure_disclosure(losses, players, tracking, "home", printed=printed)
print(f"players compared {d.position_rows}  r {d.position_r:+.4f}  R2 {d.position_r ** 2:.4f} "
      f"rho {d.position_rho:+.4f}")
print("phase p (shuffled within squad)", {k: round(v, 3) for k, v in d.phase_p.items()})
print(f"residual p {d.residual_p:.4f}  intervals clear of zero {d.residual_clear_of_zero}")
print(f"printed rows {d.printed_rows}  suppressed {d.suppressed_rows} at n<{d.min_losses}  "
      f"median interval {d.level_interval:.4f} against a span of {d.level_span:.4f}")
print(f"r(recovery, regain) {d.regain_r:.4f} over {d.n_scored} scored losses")

# the same permutation test without the squad strata, which is how the audit ran it
credited = _credited_pressers(losses, tracking, DEFAULT_RADIUS_M)
joined = credited.merge(losses[["loss_id", "phase", "recovery", "scored"]], on="loss_id", how="left")
joined = joined[joined["scored"].fillna(False).to_numpy(dtype=bool)]
joined = joined.assign(key=joined["team"].astype(str) + "/" + joined["player_id"].astype(str))
for name, block in [("pooled", joined)] + [(p, joined[joined["phase"] == p])
                                           for p in ("build_up", "middle", "final_third")]:
    res = baseline.permutation_between_player(block["recovery"], block["key"],
                                              rng=np.random.default_rng(0))
    print(f"{name:<12} players {res.n_groups:2d}  observed sd {res.observed_sd:.4f}  "
          f"null median {res.null_median:.4f}  p {res.p_value:.4f}")

# position alone, at loss level and per player
features = baseline.loss_features(losses, tracking)
prediction = pd.Series(_position_only_prediction(losses, tracking),
                       index=losses["loss_id"].to_numpy())
x_norm = pd.Series(features["x_norm"].to_numpy(), index=losses["loss_id"].to_numpy())
by_player = joined.assign(pos=joined["loss_id"].map(prediction).to_numpy(float),
                          xn=joined["loss_id"].map(x_norm).to_numpy(float)).groupby(
    ["team", "player_id"]).agg(obs=("recovery", "mean"), pos=("pos", "mean"), xn=("xn", "mean"))
by_player = by_player.loc[by_player.index.intersection(printed.set_index(["team", "player_id"]).index)]
for column in ("pos", "xn"):
    r = float(np.corrcoef(by_player["obs"], by_player[column])[0, 1])
    print(f"player level, {column}: n {len(by_player)} r {r:+.4f} R2 {r * r:.4f}")
print("residual sd", round(float(np.std(by_player["obs"] - by_player["pos"], ddof=1)), 4),
      "against column sd", round(float(np.std(by_player["obs"], ddof=1)), 4))
scored = losses[losses["scored"].fillna(False).astype(bool)]
r = float(np.corrcoef(features.loc[scored.index, "x_norm"], scored["recovery"])[0, 1])
print(f"loss level, straight line in x_norm: n {len(scored)} r {r:+.4f} R2 {r * r:.4f}")
print("median nearby losses per printed player", float(printed["losses_nearby"].median()))
PY
```

**The metric, the phase table, the distributions and the regain correlation** — the bundled match, no
network:

```sh
uv run python - <<'PY'
import numpy as np
from pressing._io import read_tracking_parquet
from pressing.counterpress import per_phase, per_team, score_losses, summarise_match
from pressing.demo import read_demo_events
tracking = read_tracking_parquet("data/demo/metrica_game_1_tracking.parquet")
events = read_demo_events("data/demo/metrica_game_1_events.parquet")
losses = score_losses(tracking, events, jobs=1)
scored = losses[losses["scored"].fillna(False).astype(bool)]
print("detected", len(losses), "scored", len(scored))
print(losses["location_source"].value_counts(dropna=False).to_string())
print(per_phase(losses).round(4).to_string())
print(per_team(losses).round(4).to_string())
for col in ("recovery_area", "recovery_epv"):
    s = scored[col]
    print(col, f"min={s.min():.4f} median={s.median():.4f} mean={s.mean():.4f} "
               f"max={s.max():.4f} sd={s.std():.4f}")
reg = scored["regained_within_5s"].astype(float).to_numpy()
for col in ("recovery_area", "recovery_epv"):
    print("r(", col, ", regain) =", round(float(np.corrcoef(scored[col], reg)[0, 1]), 4))
print("r(area, epv) =", round(float(np.corrcoef(scored["recovery_area"],
                                                scored["recovery_epv"])[0, 1]), 4))
print("regain over the scored losses", round(float(reg.mean()), 4))
print(summarise_match(losses))
PY
```

**How often the ball is untracked** — the same asset:

```sh
uv run python -c "
from pressing._io import read_tracking_parquet
t = read_tracking_parquet('data/demo/metrica_game_1_tracking.parquet')
f = t['frame'].nunique(); b = t[t['team'] == 'ball']['frame'].nunique()
print(f, b, round(100 * (1 - b / f), 2))"
```

**The demo, its PDF and its cost**, with a deliberately cold font cache. Run `uptime` either side and
read the load average, or the `real` column means nothing:

```sh
uptime
MPLCONFIGDIR=$(mktemp -d) /usr/bin/time -p uv run pressing demo --out out/demo_report.pdf
uptime
wc -c out/demo_report.pdf
```

**The pitch-control validation** — needs Metrica's CSVs under `data/raw/`. The committed notebook is
the evidence and carries its outputs; `validation/README.md` documents the method. Re-executing it
into `out/` leaves the committed notebook untouched:

```sh
uv run jupyter nbconvert --execute --to notebook --output-dir out \
    --output pitch_control_rerun.ipynb \
    --ExecutePreprocessor.timeout=3600 validation/pitch_control_validation.ipynb
```

**The proxy dataset, split and held-out table.** The table in this README is read straight out of the
committed artefact, which takes a second and needs no cache:

```sh
uv run python -c "
import pandas as pd
from pressing.proxy import load_bundled_model
md = load_bundled_model(variant='split').metadata
print({k: md[k] for k in ('competitions', 'n_matches', 'n_turnovers', 'n_labelled',
                          'n_held_out_rows', 'test_fraction', 'seed', 'fitted_on')})
print('train matches', len(md['train_match_ids']), 'held out', len(md['held_out_match_ids']))
print('label phase means', md['label_phase_means'])
print(pd.DataFrame(md['held_out_metrics']).round(4).to_string(index=False))"
```

Retraining it from the StatsBomb cache — 1.3 GB as it stands here, fetched at one request per second
if the cache is empty —
writes both artefacts, so point `--out` somewhere temporary unless you mean to replace the shipped
ones:

```sh
uv run python -m pressing.proxy --out /tmp/proxy_check.pkl.gz
```

**The label calibration against true tracking** — needs both the StatsBomb cache and Metrica's CSVs,
and takes hours:

```sh
uv run python - <<'PY'
from pathlib import Path
from pressing.proxy import (DEFAULT_TRAINING_COMPETITIONS, calibrate_against_metrica,
                            calibration_report, collect_camera_samples, load_freeze_frames,
                            load_statsbomb_events, matches_with_360)
cache = Path("data/raw/statsbomb")
samples, used = [], 0
for match_id in matches_with_360(cache, DEFAULT_TRAINING_COMPETITIONS)["match_id"]:
    ev = cache / "events" / f"{match_id}.json"
    fr = cache / "three-sixty" / f"{match_id}.json"
    if not (ev.exists() and fr.exists()):
        continue
    got = collect_camera_samples(load_statsbomb_events(ev), load_freeze_frames(fr))
    if got:
        samples += got
        used += 1
    if used >= 20:
        break
calib = calibrate_against_metrica("data/raw/metrica", games=(1, 2), camera_samples=samples, jobs=2)
print(calibration_report(calib).round(4).to_string(index=False))
print("self-check:", (calib["true_uniform"] - calib["recovery_area"]).abs().max())
PY
```

**Season mode and the scatter statistics**:

```sh
uv run pressing season --source statsbomb --competition "La Liga" --season "2015/2016" \
    --limit 40 --offline --out-dir out/season
uv run python - <<'PY'
import numpy as np, pandas as pd
from pressing.season import is_goalkeeper
t = pd.read_csv("out/season/la_liga_2015_2016_counterpress.csv", comment="#")
v = pd.to_numeric(t["market_value_eur"], errors="coerce")
m = pd.to_numeric(t["counterpress_per_90"], errors="coerce")
p = t[v.notna() & (v > 0) & m.notna() & ~t["position"].map(is_goalkeeper)]
x = p["counterpress_per_90"].to_numpy(float); y = p["market_value_eur"].to_numpy(float)
r = float(np.corrcoef(x, np.log10(y))[0, 1])
lo, hi = np.tanh(np.arctanh(r) + np.array([-1.96, 1.96]) / np.sqrt(len(p) - 3))
print(len(p), np.corrcoef(x, y)[0, 1], r, pd.Series(x).corr(pd.Series(y), method="spearman"))
print("95% CI on the log-value r:", round(lo, 3), round(hi, 3))
print(t["join_status"].value_counts().to_string())
print(t[t["join_status"] == "matched"]["join_tier"].value_counts().to_string())
PY
```

**The upstream Metrica commit behind the bundled demo asset** — the check `data/demo/NOTICE.md`
records, which needs network:

```sh
git hash-object data/raw/metrica/Sample_Game_1__Sample_Game_1_Raw*.csv
curl -s 'https://api.github.com/repos/metrica-sports/sample-data/git/trees/e706dd506b360d69d9d123d5b8026e7294b13996?recursive=1' \
  | grep -A1 Sample_Game_1
```

**The suite and the linter**:

```sh
uv run ruff check .
uv run pytest -q
```
