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

**Recovery is dominated by where on the pitch the ball was lost.** Measured on the bundled match, the
mean recovery share is **0.257** for losses in a team's own build-up third and **0.701** for losses in
the opponent's third. That is not pressing intensity; it is geometry. Lose the ball on your own
eighteen-yard line and the opponents are already behind you facing your goal; lose it on theirs and you
have ten players goalside of the ball.

So: **a pooled per-player average of this metric ranks players by where they play.** Compare recovery
within a phase or not at all. Every output this tool writes — the PDF cover, the phase table, the CSV
headers — repeats that sentence, and it is the single most important honest statement in the project.

## What you get

`pressing demo` writes a six-page A4 PDF:

1. **Cover** — fixture, provider, frame rate, a head-to-head table with the subject team picked out,
   the field-position caveat, and a rolling-recovery sparkline.
2. **Loss map** — every possession loss on one pitch in team colours, plus the build-up / middle /
   final-third table.
3–4. **Top moments 1–4**, two per page: a pitch-control heatmap each, captioned with the minute, who
   lost the ball, who won it, the recovery share and the EPV swing.
5. **Moment 5 in detail** — one full-width panel plus a t0 / t0+2s / t0+5s thumbnail strip.
6. **Player table** — minutes, losses nearby, recovery, EPV swing and the three phase splits.

`--all-panels DIR` additionally dumps one heatmap per scored possession loss — measured, 255 PNGs on
the bundled match, for 99.6 s of extra plotting. The PDF itself is always six pages.

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
[   4.6s] load (4.6s)  3,278,389 tracking rows, 1,745 events
[  27.3s] score (22.6s)  256 losses, 255 scored, jobs=6
[  45.0s] font cache (17.7s)  matplotlib
[  47.7s] moments (2.7s)  5 panels at print resolution, rank=costly
[  53.0s] report (5.3s)  out/demo_report.pdf
done in 53.0s wall clock — 255 of 256 losses scored on 6 worker(s)
out/demo_report.pdf
```

A six-page PDF of 1,403,430 bytes, and the 256 possession losses of the bundled match scored on the
way. That run took **60.3 s end to end** measured by `/usr/bin/time`, on a **cold** matplotlib font
cache, with no network.

### That wall clock, against the two-minute budget

The acceptance criterion is a full PDF in under two minutes on a fresh clone. Seven runs were timed
for this README on a ten-core laptop that was shared throughout with unrelated heavy jobs.
`MPLCONFIGDIR` was pointed at an empty directory for the six cold runs; every run produced the same
six-page, 1,403,430-byte PDF.

| run | font cache | `run_demo` clock | end to end (`real`) | CPU (`user` + `sys`) |
|---|---|---:|---:|---:|
| 1 | cold | 48.2 s | 57.4 s | 69.5 s |
| 2 | cold | 45.9 s | 58.9 s | 78.4 s |
| 3 | **warm** | **15.3 s** | **18.5 s** | 43.8 s |
| 4 | cold | 141.1 s | 152.5 s | 121.0 s |
| 5 | cold | 80.8 s | 92.7 s | 119.2 s |
| 6 | cold | 72.8 s | 81.0 s | 117.9 s |
| 7 | cold | 53.0 s | 60.3 s | 78.3 s |

Run 7 is the transcript above.

**Honest summary: cold, the demo ran in 57.4 s at best, 152.5 s at worst, and a median of 70.7 s over
the six cold runs. Five of the six were inside the two-minute budget; run 4 was not.** The spread is
contention, not the code — in the fastest run the scoring stage took 8.7 s and in run 4 it took 65.9 s
for exactly the same 255 losses, because the six worker processes were sharing the machine with
somebody else's build. Warm, with the font cache already built, the same work takes 18.5 s.

Two knobs matter if a run has to be quick on a busy box: `--jobs` sets the worker count, and a warm
`MPLCONFIGDIR` removes the font scan, which cost between 13.6 s and 36.5 s of the cold runs above and
is pure first-run overhead.

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
driven by field position, it returns final-third losses and essentially nothing else — 5 of 5 for each
team on the bundled match. `costly` is the ranking that can show a press that went wrong.

### `pressing report`

```sh
uv run pressing report data/demo/metrica_game_1_tracking.parquet \
    --events data/demo/metrica_game_1_events.parquet \
    --team home --out out/report.pdf
```

```
loaded 3,278,389 tracking rows and 1,745 events
scored 255 of 256 possession losses on 6 worker(s)
out/report.pdf
```

Progress goes to stderr and the path to stdout, so `$(uv run pressing report ...)` is the path and
nothing else. That run took 48.9 s end to end with a warm font cache (51.5 s user + 5.0 s sys) and
wrote a 1,403,419-byte PDF.

Accepts three shapes of input: a `.parquet` tracking asset written by this package (with `--events`),
a directory holding Metrica's `Sample_Game_N/` folders, or one of Metrica's raw
`Sample_Game_N_RawTrackingData_*.csv` files. `--jobs` sets the worker count (default `min(6, cpus)`);
the scores are identical serially.

### `pressing proxy`

Estimates the same quantity from a StatsBomb events file when there is no tracking. It writes a CSV,
not a PDF: three of the report's six pages are pitch-control heatmaps and its player table is built
from tracked positions, and none of that exists in an event stream.

```sh
uv run pressing proxy data/raw/statsbomb/events/3825739.json --out out/proxy_estimates.csv
```

```
66 open-play turnovers estimated
  build_up     n=   1  mean recovery 0.495
  final_third  n=  33  mean recovery 0.537
  middle       n=  32  mean recovery 0.486
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
3825739,3825739:4,41.75,Sporting Gijón,Real Madrid,final_third,0.5825577983721228,,True
3825739,3825739:6,102.727,Sporting Gijón,Real Madrid,middle,0.4286503467310845,,True
```

Read [the metrics table below](#the-event-only-proxy-honest-numbers) before you quote any single value:
the proxy is a rough ranking device and the report watermarks it as an estimate in three separate
places.

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

The EPV weighting never goes below 0.44, sits three quarters of the way up its own range, and buys
nothing for it: the two rank losses almost identically (Pearson r = 0.9572). A number a coach reads
against a stated neutral of 0.5 is worth more than one that lives between 0.75 and 0.95.

### The caveat, with the numbers attached

Mean `recovery_area` on the bundled Metrica match, by the third of the pitch the ball was lost in,
counted from the losing team's own goal:

| phase | away | n | home | n | pooled | n |
|---|---|---|---|---|---|---|
| build-up | 0.238 | 13 | 0.279 | 11 | **0.257** | 24 |
| middle | 0.439 | 53 | 0.494 | 60 | **0.468** | 113 |
| final third | 0.703 | 61 | 0.699 | 57 | **0.701** | 118 |

That is a near-threefold gradient from one end of the pitch to the other, and it dwarfs the difference
between the two teams (0.238 against 0.279 in build-up). Compare within a phase, and mind the sample
size: build-up rests on 24 losses out of 255 in this match.

### What the metric is not

**It measures spatial dominance, not ball recovery.** The event stream also records the plain,
checkable fact of whether the losing team had the ball again inside five seconds. On the bundled match
that happens in 37.3% of the 255 scored losses, and its correlation with the modelled recovery share is
**r = 0.071** (0.083 for the EPV weighting). Those two things are almost unrelated. Controlling space
after a turnover is a reasonable description of what a counter-press is *trying* to do, and this tool
says nothing stronger than that.

Two further limits, both reported per loss rather than hidden:

* The ball has no tracking row in **39.1%** of the bundled match's 145,006 frames, so inside a
  five-second window the ball position is sometimes carried forward from the last known frame. Every
  loss reports how often that happened in `n_ball_filled`.
* One open-play loss in the bundled match cannot be located at all — no tracked ball at its end frame
  and no event coordinates either side. It is reported with `location_source == "none"` and skipped,
  not guessed: 256 losses detected, 255 scored.

## Validation: pitch control against a published reference

`pressing.pitchcontrol` is checked against the
[Friends of Tracking](https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking)
implementation of the same model, at commit `7ed3ff85`, vendored byte-for-byte under
`validation/reference/` with its MIT licence. `validation/README.md` documents the method; the executed
notebook with its outputs is at `validation/pitch_control_validation.ipynb`.

Re-measured for this README over **all 257 `BALL LOST` events** of Metrica sample game 1, on the
reference's own grid:

| configuration | pooled Pearson r | mean per-event r | min per-event r | events below 0.95 | max abs difference |
|---|---|---|---|---|---|
| reference's own convention (`clamp="raw"`, float64) | **1.0000000000** | 1.0000000000 | 1.0000000000 | 0 of 257 | 3.941e-15 |
| **shipped default** (`clamp="renormalise"`, float32) | **0.9995631111** | 0.9991927210 | 0.9433488807 | **1 of 257** | 6.457e-01 |

411,200 cells compared per configuration. Guard: correlation against a y-flipped reference falls to
0.618 in both rows, so the surfaces really are aligned.

**Both rows matter, and the second is not hidden.** At the reference's own convention we reproduce it
to floating-point rounding. The shipped default deliberately diverges: forward Euler takes its final
integration step whole, so the two teams' raw probabilities can sum above 1 — measured maximum
2.800681 in this run, and the reference's own surface reaches 1.005366 at one cell. We renormalise
those cells rather than print a probability above 1 in a coach-facing report, and that one choice
accounts for the whole gap, including the single event that falls to r = 0.943.

**Speed.** In the same run, on the same 257 events and the same grid: the reference's per-cell Python
loop took 315.8 s in total, 1228.6 ms per surface. Ours took 20.7 ms per surface in the shipped
float32 configuration and 27.5 ms in the reference's own float64 convention — a factor of **59** and
**45**. Both halves were timed on a machine shared with unrelated work, so the absolute times are
inflated and only the ratio is worth much; the correlations do not move at all. The margin is why the
demo fits its runtime budget — a straight port of the reference loop would spend five minutes on pitch
control in a single match.

**What this is not.** It shows we agree with one published implementation of one model. It says nothing
about whether that model predicts football, and it does not test the loader, the offside rule or the
product's choice of ball position — those have their own tests.

## The event-only proxy: honest numbers

Most matches have events and no tracking. `pressing proxy` predicts the counter-press control share
from event features alone. `models/MODEL_CARD.md` is the full account; this is the short honest version.

**The one-line summary: this model predicts an approximation of the metric, not the metric.** Against
its own label it reaches r = 0.390 on held-out matches. The label itself reproduces the true
tracking-derived value at r = 0.26 to 0.47 (measured below). If the prediction reaches the truth only
through the label, those compose to **r ≈ 0.10 to 0.18** against the quantity the tracking report
prints. The CSV header quotes the top of that range. If you need a number a coach can act on, use
tracking; this is a rough ranking device and it is watermarked as an estimate wherever it appears.

**The spec's training plan could not be built.** It asked for a Metrica + SkillCorner overlap.
SkillCorner ships no event stream ("We are not an event data provider ourself"), its ten open matches
are Australian A-League and StatsBomb open data has no Australian competition, and its
`dynamic_events.csv` is derived from the same tracking that would produce our labels — training on it
would be circular. The labels come from **StatsBomb 360 freeze-frames** instead, split by match.

**The data**, re-derived from the cache for this README:

| competition | season | matches | open-play turnovers | labelled |
|---|---|---:|---:|---:|
| La Liga | 2020/2021 | 35 | 2,383 | 2,256 |
| 1. Bundesliga | 2023/2024 | 34 | 2,341 | 2,008 |
| Ligue 1 | 2022/2023 | 32 | 2,172 | 1,931 |
| **total** | | **101** (97 contribute a label) | **6,896** | **6,195 (89.8%)** |

68.3 open-play turnovers per match, detected from `possession_team` transitions rather than a whitelist
of event types. Of the unlabelled rows, 392 had no freeze-frame in the window and 309 had only one.
Four matches ship a well-formed 360 file whose `event_uuid`s intersect their own events file at exactly
zero, which is an upstream inconsistency; the pipeline degrades to `skip_reason='no_360'` there rather
than inventing something.

The split is **by match, never by turnover**: 68 training matches (4,394 rows) against 29 held-out
matches (1,801 rows), with an empty match-id intersection asserted in the tests. Splitting by turnover
would put two losses from the same match five seconds apart on opposite sides of the split.

Label distribution, n = 6,195: mean 0.5171, sd 0.1294, range 0.0699–0.9060. And here is the first
warning sign, before any model is fitted — the field-position gradient that dominates the true metric
is almost gone from the label:

| phase | n | mean label (`recovery_360`) | true Metrica `recovery_area` |
|---|---:|---:|---:|
| build_up | 309 | 0.4636 | 0.257 |
| middle | 2,474 | 0.4894 | 0.468 |
| final_third | 3,412 | 0.5420 | 0.701 |

A spread of 0.078 where the truth has 0.444. The next section explains why. Note also the turnover
population differs: 4.99% of labelled StatsBomb turnovers fall in build-up against 13.14% of the 487
Metrica losses (games 1 and 2) that the tracking metric scores.

### Held-out performance, against the baselines that matter

29 held-out matches, 1,801 turnovers, target `recovery_360`, re-derived from the cache for this
README:

| model | r | Spearman | MAE | RMSE | residual r after removing turnover `x` |
|---|---:|---:|---:|---:|---:|
| predict the training mean | — | — | 0.1029 | 0.1299 | 0.080 |
| predict the mean of the pitch third | 0.169 | 0.181 | 0.1006 | 0.1283 | 0.053 |
| **predict from turnover `x` alone** | **0.199** | 0.212 | 0.1001 | 0.1277 | — |
| **full model** (29 features) | **0.390** | 0.406 | 0.0916 | 0.1197 | **0.349** |

`r` for the mean predictor is undefined, not zero: its predictions are constant. `residual r` is the
correlation between prediction and label after a smooth function of turnover `x`, fitted on training
rows only, has been subtracted from both.

**That table describes a model fitted on the 68 training matches, not the artefact in `models/`.**
`models/proxy_gbr.pkl.gz` is deliberately refitted on all 6,195 labelled rows, because that is the
model you would want to run — so the held-out matches are *inside* its training set, and scoring it
on them gives r = 0.612, which is an in-sample number and means nothing. It is quoted here only so
that anyone who reproduces it recognises it for what it is. The artefact carries the split model's
metrics in its own metadata (`load_bundled_model().metadata["held_out_metrics"]`); a refit in a
fresh process reproduced them to within 0.001 on the full model and 0.008 on the x-only baseline,
which is the scale of scikit-learn's cross-process non-determinism here and the honest precision of
every figure in this section.

**Read the table honestly.** The full model beats every baseline, and it knows something beyond field
position — the x-only baseline reaches 0.199 and the residual correlation is still 0.349. But r = 0.39
against a label is not a good number in absolute terms, and the predictions are badly shrunk: on the
held-out matches the label has sd 0.1299 and spans 0.096–0.906 while the predictions have sd 0.0565 and
span 0.370–0.688, a spread ratio of 0.435. That is what a squared-error fit does when most of the
target is unexplained. **Report ranks, never levels.**

Stable across leagues and seasons, so the random-match split is not flattering itself. Each row holds
out a whole competition-season and refits on the other two:

| held out | r | MAE | n | matches |
|---|---:|---:|---:|---:|
| 1. Bundesliga 2023/24 | 0.395 | 0.094 | 2,008 | 31 |
| La Liga 2020/21 | 0.391 | 0.089 | 2,256 | 35 |
| Ligue 1 2022/23 | 0.372 | 0.094 | 1,931 | 31 |

Permutation importance on the held-out matches (drop in R²): `ball_forward_5s` 0.056,
`losing_possession_events` 0.024, `dist_opp_goal` 0.020, `loss_event_type` 0.020, `loss_duration`
0.012, `x` 0.011, `counterpress_count` 0.007. How far the ball travelled forward for the team that won
it dominates — a sensible thing for a counter-press model to key on, available from events alone, and
partly a *consequence* of the press rather than a description of it.

A sanity companion, not a target: StatsBomb tag defensive actions with their own `counterpress` flag,
defined verbatim as "pressing actions within 5 seconds of an open play turnover" — our window exactly.
64.2% of open-play turnovers have at least one. Its correlation with the label is r = 0.174 (0.162 for
the yes/no version); mean label 0.533 with a counterpress against 0.489 without. Directionally right,
and weak — about as weak as the tracking metric's own correlation with regaining the ball (0.071).

### How good is the label itself?

This is the number that decides how much the proxy can ever be worth, and it can be measured directly:
the Metrica matches have both real 25 Hz tracking and real events, so the true metric and a simulated
freeze-frame version of it can be computed on the same 487 possession losses. The simulation degrades
the truth one step at a time, driven by **real** StatsBomb 360 sampling geometry — the offsets frames
actually landed on and the actual `visible_area` polygons, anchored to the ball — transplanted onto
Metrica, so the camera model is measured rather than invented. Re-run for this README over 1,381 real
360 windows drawn from 20 matches:

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

1. **Event-driven sampling costs almost nothing** (r = 0.993). A median of six frames in the window is
   plenty.
2. **Assuming every player is stationary costs almost nothing** (r = 0.990). This is the assumption
   everyone worries about and it is not the problem.
3. **The broadcast camera is the whole problem.** Only the players inside the polygon exist, and the
   ones outside it are the ones far from the ball — exactly the asymmetry that creates the
   field-position gradient. The gradient collapses from +0.433 to +0.076.
4. **Restricting the integral to the visible polygon is worse, not better.** It looked like the more
   defensible choice; it was measured and rejected. `recovery_360` is the shipped target.

**How much worse the camera makes it depends on which windows you transplant, and the honest answer is
a range.** The run above used windows from 20 matches and put the camera variant at r = 0.261; the
independent run recorded in `models/MODEL_CARD.md`, which drew its windows from the full 97-match set,
put it at r = 0.469, and a sweep there that dilated the polygons until the retained player count matched
a real freeze-frame's still only reached 0.50. So: **the 360-derived label reproduces the true
tracking-derived metric somewhere around r = 0.26 to 0.47**, and no amount of modelling lifts that
ceiling, because it is a property of what the camera saw.

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
| points on the scatter | **105** (of the 133 ranked, after excluding 13 goalkeepers and players with no value) |

The join is by normalised name, in tiers — full name, nickname, nickname plus the club the player
appeared for that season, last two tokens. Of the 406 priced players, 297 were found by a Transfermarkt
nickname, 74 by the full name, 34 needed the club to disambiguate and 1 fell through to the last-two-
tokens rule. A name that still matched several
Transfermarkt players is dropped with `join_status='ambiguous'` and no value: attaching the wrong price
to a real player is worse than leaving the cell empty.

**What the scatter shows: nothing.** Counter-presses per 90 against market value, over the 105 plotted
outfield players:

| statistic | value |
|---|---|
| Pearson r (per 90 vs value in EUR) | −0.213 |
| Pearson r (per 90 vs log10 value) | −0.093 |
| Spearman rho | −0.070 |
| 95% interval on the log-value r | [−0.279, 0.101] |

The interval straddles zero. There is no visible relationship between how often a player counter-presses
and what he is worth in this sample, and the mild negative tilt in the raw-value column is what you
would expect from a handful of expensive forwards who press less, not evidence of anything. It is also
40 matches, not a season, and the run says so on stdout and again in the CSV header.

Two things to keep straight in the output table. `counterpress_per_90` is a **count** — StatsBomb's own
`counterpress` flag, defined verbatim as a defensive action "within 5 seconds of an open play
turnover", the same window this project's metric uses — and involves no model at all; it is the
scatter's y axis and the ranking key. `mean_recovery_pressed` and `press_value_per_90` are **proxy
estimates** and inherit every caveat in the section above. Minutes come from the lineup files'
position intervals intersected with the real period clock, so a full match reads about 94 rather
than 90.

## Data sources and licences

Nothing in this table is committed to git except where the row says so. `data/raw/` is gitignored in
full; `git ls-files data` is the check, not the working tree.

| Source | Used for | Licence | How this repo treats it |
|---|---|---|---|
| [Metrica Sports sample data](https://github.com/metrica-sports/sample-data) | The bundled demo match, the metric's measured behaviour, the pitch-control validation | **None published.** GitHub reports `license: null`, which is all rights reserved by default. The README asks only that users be responsible with the data and acknowledge the source. | Credited by name here, in the PDF footer of every page, and wherever its numbers appear. The raw CSVs are **never committed**. What is committed is a *derived* asset — restructured, re-coordinated, rotated, quantised to int16 centimetres, velocities dropped — solely so the demo runs offline on a fresh clone, with [`data/demo/NOTICE.md`](data/demo/NOTICE.md) recording exactly how it differs, the sha256 of every upstream file it came from, and an undertaking to remove it on request. `tests/fixtures/metrica/` holds **synthetic** files that copy the format only. |
| [Friends of Tracking / LaurieOnTracking](https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking) | Pitch-control reference implementation; the EPV grid | **MIT**, Copyright (c) 2021 Friends-of-Tracking-Data-FoTD | Vendored byte-for-byte with the MIT notice beside it, under `validation/reference/` and `data/epv/`, with `PROVENANCE.md` files recording the commit and the sha256 of every file. **The EPV grid is theirs and has no published provenance** — no fitting code, no source dataset, no documentation of what it was fitted on. It is used because it is redistributable and already the right shape, not because it is authoritative, and the report labels it as the Friends of Tracking grid rather than as our estimate. |
| [StatsBomb open data](https://github.com/statsbomb/open-data) | Proxy training labels; season mode | **Restrictive.** The `LICENSE.pdf` "StatsBomb Public Data User Agreement" governs and is stricter than the repo README's attribution-only wording: clause 1.2.1 forbids you to "edit, distort, distribute, reproduce, sell or in any way provide the data to any external or third party"; clause 1.2.2 forbids commercially exploiting "the data or any analysis derived from the use of the Service"; clause 1.4 requires accrediting published analysis with the StatsBomb brand logo. | **Never committed, in any form** — 1.3 GB of it sits in gitignored `data/raw/statsbomb/`, fetched at one request per second. Attribution is stamped on every event-derived output: the proxy CSV header, the season CSV header, the scatter caption, the PDF footer. This tool is non-commercial personal analysis and carries no commercial framing. Note that handing a coach a PDF of StatsBomb-derived numbers sits close to clause 1.2.1 — read the agreement first. |
| [dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) | Market values for season mode | **CC0-1.0 over the repackaging only.** | CC0 covers dcaribou's repackaging, **not** the underlying rights: the data is scraped from Transfermarkt, whose own terms prohibit scraping. Values are Transfermarkt's community estimates — not fees, not wages, not audited. A column-reduced snapshot is committed under `data/transfermarkt/` so the join works offline, with its capture date and row counts in [`data/transfermarkt/NOTICE.md`](data/transfermarkt/NOTICE.md); the full tables stay in gitignored `data/raw/`. |
| [SkillCorner open data](https://github.com/SkillCorner/opendata) | Evaluated and **not used** | MIT | Rejected on the facts, not the licence: no event stream, no competition overlap with StatsBomb, and its derived events come from the same tracking that would generate our labels. |

## Known limitations, and what I would do next

Written for someone deciding whether to trust this, not to sell it.

1. **The metric measures spatial dominance, and only barely relates to actually getting the ball back.**
   r = 0.071 against the event stream's regain flag on the bundled match. Anyone who
   reads "recovery 0.70" as "wins it back 70% of the time" is wrong, and the number is a share of
   modelled pitch control, nothing else.
2. **The right fix is a phase- and formation-conditional baseline, and it is not built.** Because
   field position dominates, the honest per-player statistic is recovery *against what this team's
   shape would be expected to produce from that pitch position* — a residual, not a level. That needs
   more than two tracked matches to fit, so what ships instead is a caveat repeated everywhere and a
   per-phase breakdown in every table.
3. **Only two matches on earth (that I can get) have both true tracking and hand-tagged events.**
   Metrica games 1 and 2. Every measured statement about the metric's behaviour rests on them, so read
   "on the bundled match" literally: it is one game, not a population.
4. **The 360 labels have no velocities and no player identities.** A freeze-frame is positions inside
   the broadcast polygon. The proxy therefore cannot do per-player attribution at all, and its label is
   an approximation whose size is measured above rather than assumed.
5. **The EPV grid is unauditable.** It ships with no provenance. `recovery_epv` and `epv_conceded`
   inherit that, which is one reason the headline weighting is plain area.
6. **Season mode's market-value join is name-based** and drops ambiguous names rather than guessing;
   the join rate is reported on every run. The ranking itself is StatsBomb's `counterpress` count per
   90, which is a hand-tagged volume statistic, not this project's model.
7. **No cross-provider tracking support.** The loaders read Metrica and the canonical parquet asset.
   SkillCorner's format is understood but unused (no events to pair it with).

If I picked this up again, in order: the conditional baseline in (2); then a proper uncertainty band
on the per-player numbers, because 20 losses per player is noise and the tables currently print three
decimal places over it; then more tracked matches, since almost every limitation above is really a
sample-size problem.

## `sportio`

The specification has this package sit on top of `sportio`, a shared loader layer that normalises every
open soccer source into the canonical `events` and `tracking` frames. **That package does not exist on
this machine**, so `src/pressing/_io.py` implements the Metrica loaders internally behind a shim: it
tries `import sportio` first and delegates when the import succeeds, and `pressing._io.backend_name()`
reports which path is live. When the shared layer lands, the internal loaders can be deleted with no
change above them.

## Layout and tests

```
src/pressing/     _io  possession  pitchcontrol  epv  counterpress  phases
                  plots  report  demo  proxy  season  cli  schema
tests/            one module per source module; fully offline
validation/       the pitch-control notebook, its README and the vendored reference
models/           the proxy artefact and its model card
data/demo/        the derived Metrica match the demo runs on, and its notice
data/epv/         the Friends of Tracking EPV grid, its licence and provenance
data/transfermarkt/  the reduced market-value snapshot and its notice
```

```sh
uv run ruff check .
uv run pytest -q
```

**511 tests, all passing**, and `ruff check` clean, as of the v0.1.0 tag. The suite is slow on purpose
— it scores real possession losses and builds real PDFs rather than mocking them — and took 800 s of
wall clock (214 s user + 33 s sys) on the same contended machine as the timings above. Budget a
quarter of an hour and do not assume it has hung. The suite is offline: no test reads anything under
gitignored `data/raw/` or opens a socket, and `tests/test_demo.py` asserts that the demo path never
opens a `data/raw/` file.

Open items are in [TODO.md](TODO.md).

## Reproducing every number in this README

Every figure above came out of one of these, run on this machine while writing it. Nothing is quoted
from memory, and where a run disagreed with an older note the newer measurement is what is printed.

**The metric, the phase table, the distributions and the regain correlation** — the bundled match, no
network:

```sh
uv run python - <<'PY'
import numpy as np
from pressing._io import read_tracking_parquet
from pressing.counterpress import per_phase, score_losses, summarise_match
from pressing.demo import read_demo_events
tracking = read_tracking_parquet("data/demo/metrica_game_1_tracking.parquet")
events = read_demo_events("data/demo/metrica_game_1_events.parquet")
losses = score_losses(tracking, events, jobs=1)
scored = losses[losses["scored"].fillna(False).astype(bool)]
print("detected", len(losses), "scored", len(scored))
print(losses["location_source"].value_counts(dropna=False).to_string())
print(per_phase(losses).to_string())
print(scored.groupby("phase")["recovery_area"].agg(["count", "mean", "std", "min", "max"]).round(4))
for col in ("recovery_area", "recovery_epv"):
    s = scored[col]
    print(col, f"min={s.min():.4f} median={s.median():.4f} mean={s.mean():.4f} "
               f"max={s.max():.4f} sd={s.std():.4f}")
reg = scored["regained_within_5s"].astype(float).to_numpy()
for col in ("recovery_area", "recovery_epv"):
    print("r(", col, ", regain) =", round(float(np.corrcoef(scored[col], reg)[0, 1]), 4))
print("r(area, epv) =", round(float(np.corrcoef(scored["recovery_area"],
                                                scored["recovery_epv"])[0, 1]), 4))
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

**The demo wall clock**, with a deliberately cold font cache:

```sh
MPLCONFIGDIR=$(mktemp -d) /usr/bin/time -p uv run pressing demo --out out/demo_report.pdf
```

**The pitch-control validation** — needs Metrica's CSVs under `data/raw/`. The committed notebook is
the evidence and carries its outputs; `validation/README.md` documents the method, and the table above
was additionally confirmed by an independent out-of-notebook rerun over the same 257 events:

```sh
uv run jupyter nbconvert --execute --to notebook --inplace \
    --ExecutePreprocessor.timeout=3600 validation/pitch_control_validation.ipynb
```

**The proxy dataset, split and held-out table** — from the StatsBomb cache; add `--download` if it is
empty (about 1.2 GB at one request per second). This command retrains and writes an artefact, so point
`--out` somewhere temporary unless you mean to replace the shipped one:

```sh
uv run python -m pressing.proxy --out /tmp/proxy_check.pkl.gz
```

The table above came from running that pipeline's own steps, so the shipped artefact was never
touched and the numbers describe the split model rather than the refitted one:

```sh
uv run python - <<'PY'
from pressing.proxy import (DEFAULT_TARGET, DEFAULT_TRAINING_COMPETITIONS, build_dataset,
                            compare_models, matches_with_360, split_by_match, train)
cache = "data/raw/statsbomb"
listing = matches_with_360(cache, DEFAULT_TRAINING_COMPETITIONS)
dataset = build_dataset(listing["match_id"].tolist(), cache, jobs=5, verbose=False)
dataset = dataset.merge(listing[["match_id", "competition", "season"]], on="match_id", how="left")
labelled = dataset[dataset["labelled"].fillna(False)].reset_index(drop=True)
tr, te = split_by_match(labelled, 0.3, 0)
model = train(labelled[tr], labelled.loc[tr, DEFAULT_TARGET], target_name=DEFAULT_TARGET)
print(compare_models(labelled, labelled[DEFAULT_TARGET], tr, te, model)
      .round(4).to_string(index=False))
PY
```

**The label calibration against true tracking** — needs both the StatsBomb cache and Metrica's CSVs:

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

**The suite and the linter**:

```sh
uv run ruff check .
uv run pytest -q
```
