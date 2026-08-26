# Model card: `proxy_gbr.pkl.gz`

Event-only proxy for the tracking-derived counter-press metric.

- **Artefact**: `models/proxy_gbr.pkl.gz`, 134,961 bytes (0.135 MB), gzipped pickle of a
  `pressing.proxy.ProxyModel`.
- **Estimator**: `sklearn.ensemble.HistGradientBoostingRegressor`, squared error, learning rate
  0.05, 31 leaves, `min_samples_leaf=40`, `l2_regularization=1.0`, early stopping on a 15%
  internal validation slice. Stopped at **85** boosting iterations.
- **scikit-learn**: 1.9.0. **Trained**: 2026-08-26.
- **Reproduce**: `uv run python -m pressing.proxy --download --jobs 5`
  (about 200 requests at one per second, 1.2 GB of cache, then ~7 minutes of labelling).

---

## Read this first

**This model predicts an approximation of the metric, not the metric.** The labels come from
StatsBomb 360 freeze-frames. Measured against real 25 Hz tracking on the Metrica sample matches,
that label reproduces the true value at **Pearson r = 0.469**. The model reaches **r = 0.391**
against the label on held-out matches. If the prediction relates to the truth only through the
label, the two compose to **r ≈ 0.18** against the quantity the tracking report actually prints.

Everything below is measured. Nothing is projected.

---

## 1. Why the spec's training plan was replaced

The spec asked for training on a "Metrica + SkillCorner overlap". That set cannot be built:

- SkillCorner ships no event stream. Their README says so outright: *"We are not an event data
  provider ourself"*.
- SkillCorner's ten open matches are Australian A-League. StatsBomb open data contains no
  Australian competition, so there is zero overlap to join on.
- SkillCorner's `dynamic_events.csv` is derived from the same tracking that would produce our
  labels, so training on it would be circular.

The trainable universe with paired 25 Hz tracking and hand-tagged events is **three Metrica
matches**, which cannot support an honest split-by-match correlation. StatsBomb 360 was used
instead, per the repository owner's recorded decision.

## 2. Data

| Competition | Season | Matches | Turnovers | Labelled |
|---|---|---:|---:|---:|
| La Liga | 2020/2021 | 35 | 2,383 | — |
| 1. Bundesliga | 2023/2024 | 34 | 2,341 | — |
| Ligue 1 | 2022/2023 | 32 | 2,172 | — |
| **Total** | | **101** (97 usable) | **6,896** | **6,195 (89.8%)** |

Three domestic club leagues in three countries — club football is what the report is for, and
three competitions make the leave-one-competition-out check below meaningful. The international
tournaments carry more 360 matches but are a different population.

101 matches were downloaded; **97** contribute at least one labelled turnover. The other four —
`3837706`, `3895158`, `3895266`, `3895309` — ship a perfectly well-formed 360 file (3,280–3,615
freeze-frames each) whose `event_uuid`s intersect the paired events file at **exactly zero**. That
is an upstream inconsistency, not a bug here; the pipeline degrades to `skip_reason='no_360'` on
every turnover in those matches rather than silently producing garbage.

Unlabelled rows: 392 with no freeze-frame in the window (`no_360`), 309 with only one
(`too_few_frames`).

**StatsBomb data is never committed.** It is cached under the gitignored `data/raw/statsbomb/`
(1.2 GB for this set) and re-fetched at one request per second. Verified with
`git ls-files data`, not just by looking at the working tree. Use of this data is governed by the
StatsBomb Public Data User Agreement; see the repository README.

### Measured properties of the 360 windows

- Freeze-frames per five-second window: p10 **3**, median **7**, p90 **11**.
- Players captured per freeze-frame: mean **15.99** of 22.
- Fraction of the pitch inside the broadcast `visible_area` polygon: mean **30.5%**.
- Open-play turnovers per match: **68.3**.

## 3. The label

For each open-play turnover the freeze-frames of the events inside `[t0, t0+5s]` are taken, a
pitch-control surface is computed at each, and the losing (counter-pressing) team's share of
control is averaged across them. That is the same quantity as the tracking metric's
`recovery_area`, evaluated on whatever instants the event stream provides.

Decisions, all of them approximations, all documented in `pressing.proxy.label_turnovers`:

| Question | Decision | Why |
|---|---|---|
| Velocity | **Zero for everyone** | A freeze-frame carries none. Inventing one layers a second model under the first. Measured cost: `r` falls from 0.9925 to 0.9875 — negligible. |
| Team identity | `teammate` resolved through `event.team` | There are no player ids. A frame from a losing-team event has `teammate=True` on the *pressers*, so the mapping must be inverted or the label comes back as one minus the answer. |
| Orientation | Mirror any frame whose event belongs to the loser | StatsBomb re-orients per event so `event.team` attacks `+x`. Without the mirror a Pressure event by the counter-pressing team arrives 180° out — silently. |
| Ball | The event's own `location` | Where the ball is at a tagged on-ball event, and the only ball information a 360 file has. |
| Visible area | Two labels computed, one chosen by measurement | `recovery_360` integrates the whole pitch; `recovery_360_visible` integrates only inside the polygon. The Metrica calibration picked `recovery_360` (see §4). The two correlate at only **r = 0.439** with each other. |
| One-frame windows | Unlabelled, `skip_reason='too_few_frames'` | Averaging one instant measures the geometry at the turnover and says nothing about the press. 309 rows dropped. |

Turnovers are detected from **`possession_team` transitions**, not from a whitelist of turnover
event types: the last on-ball event of a possession is `Ball Receipt*` about a third of the time,
which no whitelist catches. A transition counts as open play when the new possession does not
start from a dead ball (`From Corner`, `From Free Kick`, `From Goal Kick`, `From Keeper`,
`From Kick Off`, `From Throw In`).

### Label distribution (n = 6,195)

mean **0.5171**, sd **0.1294**, min **0.0699**, max **0.9060**.

| Phase | n | mean `recovery_360` | true Metrica `recovery_area` |
|---|---:|---:|---:|
| build_up | 309 | 0.4636 | 0.273 |
| middle | 2,474 | 0.4894 | 0.466 |
| final_third | 3,412 | 0.5420 | 0.706 |

The field-position gradient that dominates the true metric — a spread of **0.433** from build-up
to final third — is compressed to **0.078** in the label. §4 shows why.

## 4. Calibration against true 25 Hz tracking

Both labels were computed on the same 487 Metrica possession losses (games 1 and 2), degrading
the truth one step at a time. Real StatsBomb 360 sampling geometry — the offsets frames actually
landed on and the actual `visible_area` polygons, anchored relative to the ball — was transplanted
onto Metrica, so the camera model is measured rather than invented.

| Variant | n | Pearson r | Spearman | MAE | bias | mean | sd | phase gradient |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `recovery_area` (shipped metric) | 487 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.5530 | 0.1843 | **+0.4332** |
| `true_uniform` (harness self-check) | 487 | **1.0000** | 1.0000 | 0.0000 | 0.0000 | 0.5530 | 0.1843 | +0.4332 |
| `sparse_offsets` (event-driven sampling) | 464 | **0.9925** | 0.9916 | 0.0153 | +0.0044 | 0.5565 | 0.1890 | +0.4462 |
| `sparse_zero_velocity` (+ no velocities) | 464 | **0.9875** | 0.9863 | 0.0217 | +0.0037 | 0.5558 | 0.1914 | +0.4520 |
| `sparse_zero_velocity_camera` (+ broadcast truncation) | 439 | **0.4687** | 0.4708 | 0.1386 | −0.0481 | 0.5080 | 0.1506 | +0.1422 |
| `..._camera_visible` (+ integrate inside polygon only) | 439 | **0.1339** | 0.1314 | 0.1812 | −0.0645 | 0.4915 | 0.1278 | −0.0168 |

**The harness self-check is exact.** `true_uniform` reproduces
`counterpress.score_match`'s `recovery_area` at r = 1.0000000000 with a maximum absolute
difference of **0.0** over all 487 losses, so the degraded variants are computed by code known to
agree with the shipped metric.

### What this says

1. **Sparse event-driven sampling costs almost nothing** (r = 0.993). Median 7 frames in the
   window is plenty.
2. **Assuming every player is stationary costs almost nothing** (r = 0.988). This is the
   assumption everyone worries about, and it is not the problem.
3. **The broadcast camera is the entire problem** (r drops to 0.469). Only ~30% of the pitch and
   ~16 of 22 players are captured, and the players who fall outside the frame are the ones far
   from the ball — which is exactly the asymmetry that creates the field-position gradient. The
   gradient collapses from +0.433 to +0.142.
4. **Restricting the integral to the visible polygon is worse, not better** (r = 0.134, gradient
   −0.017). It looked like the more defensible choice; it was measured and rejected.
   `recovery_360` is the shipped target.

### Is the simulated camera too harsh?

The transplanted polygons retain **11.93** Metrica players on average against the **15.99**
a real 360 frame contains, so the simulation is harsher than reality. Dilating the polygon about
the ball until the counts match brackets the effect (Metrica game 1, 82–83 losses each):

| polygon scale | players kept | camera r | phase gradient | visible-only r |
|---:|---:|---:|---:|---:|
| 1.00 | 11.85 | 0.407 | +0.126 | −0.003 |
| 1.15 | 13.14 | 0.415 | +0.092 | +0.050 |
| 1.30 | 14.24 | 0.446 | +0.058 | +0.068 |
| 1.50 | 15.24 | **0.498** | +0.153 | +0.165 |

Even with the simulated camera tuned to the real player count, r reaches only **0.50** and the
gradient stays below a third of the true 0.433. **The finding is not an artefact of the
simulation.**

## 5. Split

**By match, never by row.** 6,195 labelled turnovers over 97 matches, `test_fraction=0.3`,
`seed=0`:

- train: **68 matches**, 4,394 turnovers
- test: **29 matches**, 1,801 turnovers
- match-id intersection: **empty** (asserted in `tests/test_proxy.py`)

Splitting by turnover would put two losses from the same match five seconds apart on opposite
sides of the split. The spec calls that disqualifying.

## 6. Features

Every feature is derived from the event stream. **None touches a freeze-frame** — the model has
to run on matches that have events and nothing else. `tests/test_proxy.py` asserts this against
the parsed source of `build_features`.

The four groups the spec names, plus justified extras:

- **Turnover location**: `x`, `y` in canonical metres in the *losing* team's attacking frame,
  `dist_own_goal`, `dist_opp_goal`, `dist_from_centre`.
- **Time to next event**: `time_to_next_event`, computed from a period offset plus the parsed
  timestamp (see §8).
- **Next 3 actions**: `next_1_type`, `next_2_type`, `next_3_type` and whether each belongs to the
  counter-pressing team (`next_k_by_loser`).
- **Pressure flags**: `under_pressure_count`, `counterpress_count`, `pressure_event_count`,
  `loss_under_pressure`, `loss_counterpress`.
- **Extras**: `ball_forward_5s`, `ball_distance_5s`, `losing_possession_events`,
  `losing_possession_duration`, `loss_duration`, `loss_event_type`, `next_play_pattern`,
  `regained_within_5s`, `n_events_5s`, `n_loser_events_5s`, `n_winner_events_5s`, `period`.

29 features, 5 of them categorical. Categorical vocabularies are frozen at training time and
stored in the artefact; an unseen category encodes to `-1`.

**`regained_within_5s` is near-degenerate on StatsBomb**: only **1.7%** of open-play turnovers see
the losing team credited with a new possession inside five seconds, against a 37% regain rate
measured on Metrica. StatsBomb keeps contested phases inside one possession where Metrica splits
them. The feature is kept because it is honest and cheap, and the booster ignores it.

## 7. Held-out results

29 held-out matches, 1,801 turnovers, target `recovery_360`.

| model | r | Spearman | MAE | RMSE | residual r |
|---|---:|---:|---:|---:|---:|
| mean (predict the training mean) | — | — | 0.1029 | 0.1299 | 0.0706 |
| phase (mean of the pitch third) | 0.1690 | 0.1813 | 0.1006 | 0.1283 | 0.0353 |
| turnover `x` alone | 0.2070 | 0.2221 | 0.0998 | 0.1274 | — |
| **full model** | **0.3907** | **0.4075** | **0.0917** | **0.1199** | **0.3438** |

**The predictions are heavily shrunk.** On the held-out matches the label has sd **0.1299** and
spans 0.096–0.906, while the predictions have sd **0.0595** and span 0.351–0.690 — a spread ratio
of **0.46**. That is what a squared-error fit does when most of the target is unexplained, and it
means the proxy separates turnovers only weakly in absolute terms even where it ranks them
correctly. Report ranks, not levels.

`r` for the mean predictor is reported as undefined, not as 0: the predictions are constant.
`residual_r` is the correlation between prediction and label after a smooth function of turnover
`x` — fitted on training rows only — has been subtracted from both.

**The model is not just field position.** The x-only baseline reaches r = 0.207; the full model
reaches 0.391, and its residual correlation after removing x is 0.344. Roughly speaking, most of
what the model knows is not where the ball was lost.

### Leave-one-competition-out

| held out | r | MAE | n | matches |
|---|---:|---:|---:|---:|
| 1. Bundesliga 2023/24 | 0.3928 | 0.0938 | 2,008 | 31 |
| La Liga 2020/21 | 0.4011 | 0.0886 | 2,256 | 35 |
| Ligue 1 2022/23 | 0.3848 | 0.0933 | 1,931 | 31 |

Stable across leagues and seasons; the random-match split is not flattering itself.

### Permutation importance (held-out matches, drop in R²)

| feature | ΔR² |
|---|---:|
| `ball_forward_5s` | +0.0597 |
| `losing_possession_events` | +0.0228 |
| `loss_event_type` | +0.0197 |
| `dist_opp_goal` | +0.0147 |
| `loss_duration` | +0.0130 |
| `x` | +0.0125 |
| `time_to_next_event` | +0.0071 |
| `counterpress_count` | +0.0049 |
| `pressure_event_count` | +0.0040 |
| `ball_distance_5s` | +0.0036 |

How far the ball travelled forward for the team that won it dominates. That is a sensible thing
for a counter-press model to key on and it is available from events alone — but note it is partly
a *consequence* of the press rather than a description of it.

### Sanity companion: StatsBomb's own `counterpress` flag

StatsBomb define `counterpress` verbatim as *"pressing actions within 5 seconds of an open play
turnover"* — the identical window to ours. It is a per-event boolean, not a control share, so it
is a companion and not a target.

- **64.2%** of open-play turnovers have at least one `counterpress` event in the window.
- r(`recovery_360`, count of `counterpress` events) = **0.174**
- r(`recovery_360`, any `counterpress` in window) = **0.162**
- mean `recovery_360` with a counterpress: **0.5327**; without: **0.4890**

Directionally right — hand-tagged counter-pressing goes with more modelled control — and weak, at
about the strength the tracking metric's own correlation with the event-stream regain flag showed
(0.07–0.08 on Metrica).

## 8. The period-boundary trap

StatsBomb `timestamp` is period-relative (period 2's first event reads `00:00:00.000`) while
`minute` / `second` are match-absolute. Mixing them produces a spurious gap of roughly 2758 s at
every half boundary, with no exception raised, which would corrupt both the five-second window
and the time-to-next-event feature. `pressing.proxy.absolute_seconds` accumulates a per-period
offset from the observed length of each preceding period and never reads `minute`. It is
regression-tested in `tests/test_proxy.py`, including an assertion that the naive computation
really would jump by −2700 s on the same fixture.

## 9. Intended use, and what this must not be used for

**Intended**: a rough, watermarked estimate of counter-press control on a match that has event
data and no tracking, ranked *within* a pitch phase.

**Not intended**:

- **Never compare a build-up recovery with a final-third one and call the difference pressing.**
  The tracking metric spreads 0.27 → 0.71 across pitch thirds; the label spreads 0.46 → 0.54. The
  model is trained on the compressed version and does not know the real gradient.
- **Never present a proxy number next to a tracking number.** They are on different scales
  (label mean 0.517 sd 0.129; true metric mean 0.553 sd 0.184) and correlate at r ≈ 0.47.
- **No per-player attribution.** 360 freeze-frames carry no player identities, so nothing here
  supports the report's per-player table.
- **No EPV.** `predict()` emits `epv_conceded` as NaN because no EPV model exists; a report built
  on this frame must not print an EPV number.
- **Turnover population differs.** Detection from `possession_team` transitions finds 68.3
  open-play turnovers per match; Metrica's `BALL LOST` definition finds 255 and 232 in the two
  sample games. The StatsBomb set is the subset of decisive, cleanly-resolved turnovers, and
  skews toward losses that end in the losing team's attacking half — **4.99%** of labelled
  turnovers fall in build-up against **13.14%** of the 487 scored Metrica losses.

## 10. Honest summary

The model beats every baseline on the task it was given, generalises across three leagues, and
knows things beyond field position. But it is trained against a label that itself only reproduces
the truth at r ≈ 0.47, and that ceiling is set by the broadcast camera, which no amount of
modelling can lift. **If you need a number a coach can act on, use tracking. This is a rough
ranking device, and the report must watermark it as an estimate.**

## 11. Attribution

Data from StatsBomb open data (<https://github.com/statsbomb/open-data>), used under the StatsBomb
Public Data User Agreement, and Metrica Sports sample data
(<https://github.com/metrica-sports/sample-data>). Pitch control follows the Friends of Tracking
reference vendored under `validation/reference/` with its MIT licence.
