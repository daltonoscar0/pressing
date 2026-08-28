# Model card: `proxy_gbr.pkl.gz`

Event-only proxy for the tracking-derived counter-press metric.

- **Artefacts**: `models/proxy_gbr.pkl.gz`, 132,321 bytes, refitted on every labelled row — the
  deployable model; and `models/proxy_gbr_split.pkl.gz`, 124,333 bytes, fitted on the training
  matches only — the model the metrics below describe. Both are gzipped pickles of a
  `pressing.proxy.ProxyModel`. See §5.2 for which to use when, and §6.1 for what is inside them.
- **Estimator**: `sklearn.ensemble.HistGradientBoostingRegressor`, squared error, learning rate
  0.05, 31 leaves, `min_samples_leaf=40`, `l2_regularization=1.0`, early stopping on a 15%
  internal validation slice. Stopped at **83** boosting iterations (refit) and **79** (split).
- **scikit-learn**: 1.9.0.
- **Identity**: SHA-256 `94f6af0f22ffa5b52b208efb8756969b0a3b33a3dead91b87eed2bcb12d0a3ed` (refit)
  and `1364b8d9a831313b28c9f1c43ce61085f33ad7da0d88bdd47eedd82bdab5d22e` (split). Neither file
  records when it was built — a training timestamp is a byte that changes every run and nothing
  read it. `git log models/` answers "when"; the digests above answer "which". §5.1 has the proof
  that a rebuild lands on them.
- **Reproduce**: `uv run python -m pressing.proxy --download --jobs 5`, which writes both
  artefacts (about 200 requests at one per second, then 1.2 GB of cache to label). The result no
  longer depends on `--jobs`, and no longer depends on the clock; §5.1 says why both of those
  sentences had to be earned.

---

## Read this first

**This model predicts an approximation of the metric, not the metric.** The labels come from
StatsBomb 360 freeze-frames. Measured against real 25 Hz tracking on the Metrica sample matches,
that label reproduces the true value at **Pearson r somewhere between 0.26 and 0.47** — the two
independent runs recorded in this repository, which drew their transplanted camera windows from
different match sets, disagree that widely and neither is more authoritative than the other (§4).
The model reaches **r = 0.369** against the label on held-out matches. If the prediction relates to
the truth only through the label, the two compose to **r ≈ 0.10 to 0.17** against the quantity the
tracking report actually prints. Where a single number is unavoidable this repository writes
"r ≈ 0.18"; treat that as the optimistic end, not the estimate.

Everything below is measured. Nothing is projected.

### On precision

Held-out proxy figures are quoted to **three decimals only where a fixed row order makes them
exactly reproducible**, which it now is: see §5.1. Figures carried over from runs whose row order
was not controlled are quoted to two decimals, because that is the precision the measured
across-run variation supports.

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

**And the model inherits that compression, which is why no output may print the tracking metric's
phase means as a scale for a proxy number.** The shipped refit model's own predicted phase means
over the same 6,195 rows are build-up **0.4643**, middle **0.4904**, final third **0.5421** — a
gradient of **+0.0779**, against the tracking metric's **+0.444** on Metrica game 1 (0.257 / 0.468 /
0.701). Up to v0.1.0 the proxy and season CSV headers printed those tracking means above proxy
columns and told the reader to compare against them, so a proxy build-up value of 0.47 — the
proxy's own average — read as an extraordinary press. The headers now print
`pressing.proxy.phase_reference_means()`, which is the model's own recorded scale, and name the
tracking figures only inside the sentence telling you not to scale against them.

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
3. **The broadcast camera is the entire problem** (r drops to 0.469 in this run, and to 0.261 in
   the independent 20-match run recorded in the README — see the range note below). Only ~30% of
   the pitch and ~16 of 22 players are captured, and the players who fall outside the frame are the
   ones far from the ball — which is exactly the asymmetry that creates the field-position gradient.
   The gradient collapses from +0.433 to +0.142 here, and to +0.076 in that other run.
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

### The label-quality figure is a range, not a number

This card's camera row (r = 0.469, n = 439, windows drawn from the full 97-match set) and the
README's parallel table (r = 0.261, n = 441, windows drawn from 1,381 real 360 windows over 20
matches) are two runs of the same experiment over different transplanted windows. They disagree far
beyond rounding, and **neither is more authoritative than the other**. The honest statement is the
README's: *the 360-derived label reproduces the true tracking-derived metric somewhere around
r = 0.26 to 0.47.* Up to v0.1.0 this card stated 0.469 flatly, which is the flattering end of that
range presented as the answer; if you read only the card you got the best case. The composed
proxy-to-truth figure moves with it: 0.37 × 0.26 ≈ **0.10** at the pessimistic end against
0.37 × 0.47 ≈ **0.17** at the optimistic one.

## 5. Split

**By match, never by row.** 6,195 labelled turnovers over 97 matches, `test_fraction=0.3`,
`seed=0`:

- train: **68 matches**, 4,394 turnovers
- test: **29 matches**, 1,801 turnovers
- match-id intersection: **empty** (asserted in `tests/test_proxy.py`)

Splitting by turnover would put two losses from the same match five seconds apart on opposite
sides of the split. The spec calls that disqualifying.

**Both artefacts carry the split in their metadata** — `train_match_ids`, `held_out_match_ids`,
`n_train_rows`, `n_held_out_rows` and `fitted_on`. You do not have to trust this document for it.

### 5.1 Reproducibility, and what broke it

Up to v0.1.0 a refit in a fresh process did not reproduce this card's figures. Three independent
rebuilds landed at held-out r = 0.3803, 0.3820 and 0.3907 against a stated tolerance of 0.001 —
roughly ten times outside it — while the card still printed three decimals.

**Cause, measured.** `build_dataset` assembled the parallel path with
`concurrent.futures.as_completed` and appended each match as its worker *finished*, so the row order
of the training set was a race between five processes. The content was identical every time; the
order was not. `HistGradientBoostingRegressor` draws its early-stopping validation slice out of the
design matrix by position, so a different row order is a different validation slice, a different
stopping iteration and a different fitted tree.

**Demonstration.** Three fresh processes refitting the split model on the canonical frame of the
6,195 labelled rows produced byte-identical held-out prediction vectors — SHA-256
`c12d212d7d7aade99b7b93e8478bdc45d4581747f00fc0f31b51c9f4e30d7ccd`, which is
`sha256(split.predict(held_out).astype("float64").tobytes())` and which the shipped artefact still
reproduces — full-model r = 0.369162 and x-only r = 0.197444 to six decimals
each time, stopping at iteration 79. Shuffling nothing but the row order of that same frame, at
three seeds, gave full-model r = 0.369387, 0.374414 and 0.368491 (stopping at 85, 69 and 85) and
x-only r = 0.203469, 0.210770 and 0.202058. The `mean` and `phase` baselines, which fit no booster,
were identical throughout at r = 0.168970 — exactly the signature of an order-dependent fit.

The same experiment on the frame the v0.1.0 artefact was actually built from reproduced that
artefact's recorded metrics to every digit (r = 0.390659, three fresh processes, SHA-256
`1a1453c5f31e23a4…`) and moved to 0.386581 and 0.370107 under a row shuffle. So v0.1.0's 0.3907 was
not wrong; it was one draw from a band that the whole set of measured orders puts at roughly
**0.368 to 0.391**. Printed to four decimals, with a claimed tolerance of 0.001, it read as far more
settled than it was.

**Fix.** `build_dataset` now collects worker results into their submitted slots and returns a
canonical row order — ascending `match_id`, and within a match the order the per-match builder
produced. Verified on the first 16 training matches (1,124 turnovers): `--jobs 1`, `--jobs 4`,
`--jobs 7` and `--jobs 4` with the match ids handed over in reverse all produce a frame with the
identical digest `5bf21169f50119fd3a4f1d8d16833dcb80e0cd4b586747c2043adf2ca5cf8876`, where a
frame's digest means
`hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()).hexdigest()`.
`tests/test_proxy.py` pins the ordering property so it cannot regress.

> Up to v0.1.1 this paragraph gave `0a5cc7ec2f6539f9253f671e3f2a5bd5` and called it a SHA-256. It
> is 32 hex characters, which no SHA-256 is, it named no hashing recipe, and no recipe tried here
> reproduces it. It has been replaced by the digest above and by the command that produces it.

### 5.1.1 The artefacts are byte-reproducible

A deterministic fit is not a reproducible file. Up to v0.1.1 two training runs that fitted
identical trees still wrote different bytes, for two reasons that both came down to reading a
clock:

- **The gzip header.** `gzip.open` writes `time.time()` into header bytes 4:8. The v0.1.1 split
  artefact carried `1f8b 0808 2e41 8f6a` — mtime `0x6a8f412e`, 1787773230 — and every rewrite
  moved those four bytes. Demonstrated by saving one unchanged model twice, 1.2 s apart, through
  the old code path: the two files differed at byte offset 4 and nowhere else.
- **`trained_utc`.** `train_from_cache` stamped `pd.Timestamp.now(tz="UTC").isoformat()` into the
  metadata, so a fresh string went *inside* the pickle on every run. Because it is compressed, its
  length changed too: the same model saved with three different timestamps came out at 124,366,
  124,366 and 124,356 bytes. That is the size wobble the v0.1.1 audit observed.

`save_model` now writes `gzip.GzipFile(..., mtime=proxy.ARTEFACT_GZIP_MTIME)`, which is 0 — the
value the format reserves for "no timestamp" — and no metadata field reads a clock.

**Proof.** Two fresh Python processes, each running `train_from_cache` end to end over the whole
1.2 GB cache and writing into its own directory:

```
sha256  1364b8d9a831313b28c9f1c43ce61085f33ad7da0d88bdd47eedd82bdab5d22e  runA/proxy_gbr_split.pkl.gz
sha256  94f6af0f22ffa5b52b208efb8756969b0a3b33a3dead91b87eed2bcb12d0a3ed  runA/proxy_gbr.pkl.gz
sha256  1364b8d9a831313b28c9f1c43ce61085f33ad7da0d88bdd47eedd82bdab5d22e  runB/proxy_gbr_split.pkl.gz
sha256  94f6af0f22ffa5b52b208efb8756969b0a3b33a3dead91b87eed2bcb12d0a3ed  runB/proxy_gbr.pkl.gz
```

`cmp` agrees on both pairs, and both runs printed the identical §7 metrics table and stopped at
the same iteration (83 refit, 79 split). The committed files are runA's.

**The gzip was not masking a difference underneath.** Unpickled, the new estimators and the v0.1.1
ones they replace predict identically — maximum absolute difference **0.0** over 5,000 probe rows,
for both variants, same `n_iter_`, same number of predictors — and the only metadata that changed
is the removal of `trained_utc`. Nothing in this card moved because of the fix.

**What is claimed, exactly.** Two training runs write the same file. The file is not a fixed point
of load-and-resave: `save_model(load_bundled_model(p), p)` returns 124,328 bytes for the split
artefact against the committed 124,333, and 132,320 against 132,321 for the refit. That is
`pickle`'s framing, not the model — decompressed, the round trip is 336,411 bytes against 336,401
and the first differing byte is a `FRAME` length. `pickle` sizes its frames by accumulated bytes,
and a re-pickled object graph shares its strings differently after a load, so the boundaries move.
Two fresh processes doing a load-and-resave agree with each other to the byte; they simply do not
agree with a training run, and the estimator is identical across all of them.

**What is still not guaranteed.** Determinism here means *this machine, this scikit-learn, this
NumPy*. `HistGradientBoostingRegressor` reduces its histograms across OpenMP threads, so a
different thread count or a different BLAS build can still move the low-order digits. The claim
made is the one that was tested: with the row order fixed, repeated fresh processes on one machine
agree exactly.

**Consequence for this card.** The v0.1.0 artefact was trained on one particular completion order,
which the fixed code cannot and should not reproduce. Both artefacts were retrained from the
canonical order, so every proxy figure below moved. The figures below are the new ones and are
reproducible to the digit printed; the old ones are not recoverable and should not be quoted.

### 5.2 The two artefacts, and why scoring the wrong one lies to you

| file | `fitted_on` | fitted on | what it is for |
|---|---|---|---|
| `models/proxy_gbr.pkl.gz` | `all_labelled` | every labelled row, all 97 matches | the model you deploy; what `load_bundled_model()` returns |
| `models/proxy_gbr_split.pkl.gz` | `train_matches` | the 68 training matches only | the model §7's metrics describe |

The refit model is the one worth shipping — throwing away a third of the data to keep a clean test
set is the wrong trade for a deployed estimator. But **it has seen every match, so its own held-out
metrics cannot be reproduced with it.** Scoring it on the recorded held-out match ids is in-sample
and returns r = 0.6151 against a true 0.3692, with nothing to indicate anything went wrong. v0.1.0
warned about that in three documents, which is the weakest available fix.

It now raises:

```python
from pressing import proxy

refit = proxy.load_bundled_model()                    # fitted_on = "all_labelled"
proxy.score_held_out(refit, features, labels)         # ValueError: ... is in-sample

split = proxy.load_bundled_model(variant="split")     # fitted_on = "train_matches"
proxy.score_held_out(split, features, labels)         # the honest number
```

`score_held_out` filters the frame you give it down to the model's own recorded
`held_out_match_ids`, so passing the whole labelled dataset is the expected use, and refuses
outright if the model was fitted on those rows or if it does not record its split at all.

**Why ship both rather than only the split model.** Only the refit model should be used to estimate
anything, and only the split model can be honestly scored. Shipping one forces either a deployed
model that is worse than it needs to be, or a metrics table nobody can check without a 1.2 GB
download and a full refit. Two files of 132,321 and 124,333 bytes are a cheaper answer than
either. See §6.1 for what is inside them.

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

### 6.1 What is inside the committed artefacts, in full

Raised here for the repository owner's decision rather than left to be discovered. `DECISIONS.md`
§3 reads *"never commit StatsBomb data of any kind to git"* and is worded absolutely, and these two
committed files contain StatsBomb-derived **content**, though not StatsBomb **data**:

- The five categorical vocabularies, which are StatsBomb's own event-type taxonomy: `Ball Recovery`,
  `Dispossessed`, `Miscontrol`, `Pressure`, `Interception`, `Dribbled Past`, `Foul Committed`,
  `Regular Play`, `From Counter`, `Referee Ball-Drop`, `Tactical Shift` and the rest.
- The three competition-season labels: `La Liga 2020/2021`, `1. Bundesliga 2023/2024`,
  `Ligue 1 2022/2023`.
- In the new metadata, the 97 StatsBomb **match ids** split into `train_match_ids` and
  `held_out_match_ids`. These are integers identifying publicly listed fixtures; they carry no event
  content, and recording them is the whole point of §5.2.

What is **not** in them, checked rather than assumed: no rows, no event records, no `event_uuid`s,
no player names, no club names, no coordinates, no freeze-frames. Verify with

```
python -c "import gzip,pickle,sys; m=pickle.load(gzip.open('models/proxy_gbr.pkl.gz')); print(m.vocabularies, m.metadata.keys())"
```

The reading taken here is that a fitted estimator plus a category vocabulary is a model, not data,
and that a fixture id list is a citation rather than a reproduction. That reading is a judgement
call on an absolutely worded decision, so it is stated openly instead of relied on quietly. If the
owner reads §3 more strictly, the fix is to gitignore `models/*.pkl.gz` and have the demo path fall
back to the tracking metric; nothing else in the repository depends on the artefact being committed.

**`regained_within_5s` is near-degenerate on StatsBomb**: **1.96%** of the 6,896 detected open-play
turnovers (135 of them), and **2.11%** of the 6,195 labelled ones (131 of them), see the losing
team credited with a new possession inside five seconds — against a 37.3% regain rate measured on
Metrica. StatsBomb keeps contested phases inside one possession where Metrica splits them. The
feature is kept because it is honest and cheap, and the booster ignores it.

> This card gave 1.7% up to v0.1.0. That figure was not reproducible in either population and has
> been replaced by both measurements. Reproduce with
> `dataset["regained_within_5s"].mean()` over the frame `build_dataset` returns.

## 7. Held-out results

29 held-out matches, 1,801 turnovers, target `recovery_360`, scored with
`score_held_out(load_bundled_model(variant="split"), features, labels)`.

| model | r | Spearman | MAE | RMSE | residual r |
|---|---:|---:|---:|---:|---:|
| mean (predict the training mean) | — | — | 0.1029 | 0.1299 | 0.0806 |
| phase (mean of the pitch third) | 0.1690 | 0.1813 | 0.1006 | 0.1283 | 0.0581 |
| turnover `x` alone | 0.1974 | 0.2107 | 0.1002 | 0.1277 | — |
| **full model** | **0.3692** | **0.3884** | **0.0927** | **0.1212** | **0.3260** |

> These moved down from v0.1.0's 0.3907 / 0.2070 / 0.1690 / 0.4075. Nothing about the model
> changed: v0.1.0's training set happened to be assembled in a row order that fitted a slightly
> luckier tree, and that order was a race between worker processes rather than a choice. §5.1 has
> the demonstration. The figures here come from the canonical order and are reproducible to the
> digit printed; the old ones were the top of a ±0.02 band and were printed to four decimals.

**The predictions are heavily shrunk.** On the held-out matches the label has sd **0.1299** and
spans 0.096–0.906, while the predictions have sd **0.0586** and span 0.368–0.673 — a spread ratio
of **0.451**. That is what a squared-error fit does when most of the target is unexplained, and it
means the proxy separates turnovers only weakly in absolute terms even where it ranks them
correctly. Report ranks, not levels. §7.1 records what happened when the objective was changed to
one that cares about ranks.

`r` for the mean predictor is reported as undefined, not as 0: the predictions are constant.
`residual_r` is the correlation between prediction and label after a smooth function of turnover
`x` — fitted on training rows only — has been subtracted from both.

**The model is not just field position.** The x-only baseline reaches r = 0.197; the full model
reaches 0.369, and its residual correlation after removing x is 0.326. Roughly speaking, most of
what the model knows is not where the ball was lost.

### 7.1 A rank objective was tried and did not help

The shrinkage above is the signature of squared error on a mostly-unexplained target, so a
rank-appropriate objective is the obvious thing to reach for. Four were fitted on the identical
training rows and scored on the identical held-out matches. Spearman is the number that matters —
it is invariant to any monotone transform of the prediction, so a rank-transformed target can be
compared with a raw one directly.

| objective | Pearson r | Spearman | Kendall | spread ratio |
|---|---:|---:|---:|---:|
| squared error (shipped) | 0.3692 | 0.3884 | 0.2681 | 0.451 |
| absolute error | 0.3755 | **0.3982** | 0.2731 | 0.443 |
| quantile, α = 0.5 | 0.3755 | 0.3982 | 0.2731 | 0.443 |
| rank-gauss target (normal scores, fitted on train) | 0.3674 | 0.3852 | 0.2658 | — |
| within-match percentile rank target | 0.3703 | 0.3938 | 0.2696 | — |

**Negative result, reported as one.** The best alternative — absolute error, which the quantile
loss at α = 0.5 reproduces exactly, as it should — gains **+0.0098 Spearman**, from 0.3884 to
0.3982. A match-clustered bootstrap over 2,000 resamples of the 29 held-out matches puts that
difference's 95% interval at **[−0.0087, +0.0273]** — it straddles zero, and absolute error wins in
84.8% of resamples, not the 97.5% that would make the gain worth a claim. Rank-transforming the target
outright made things *worse*, in both the global and the within-match version. And none of it fixes
the complaint that motivated the experiment: absolute error's spread ratio is 0.443 against squared
error's 0.451, so the predictions are, if anything, marginally more shrunk, not less.

Shrinkage here is not an artefact of the loss function. It is the correct behaviour of a
conditional-mean estimate when most of the target's variance is genuinely unexplained, and no
choice of objective can manufacture spread that the features do not support. **The shipped model
keeps squared error.** LambdaRank proper was not tried: it needs LightGBM or XGBoost, neither of
which is a dependency of this project, and on the evidence above a listwise objective would be
chasing a hundredth of a Spearman point.

### Leave-one-competition-out

| held out | r | Spearman | MAE | n | matches |
|---|---:|---:|---:|---:|---:|
| 1. Bundesliga 2023/24 | 0.3809 | 0.3998 | 0.0942 | 2,008 | 31 |
| La Liga 2020/21 | 0.3812 | 0.3960 | 0.0893 | 2,256 | 35 |
| Ligue 1 2022/23 | 0.3848 | 0.4105 | 0.0932 | 1,931 | 31 |

Stable across leagues and seasons — a spread of 0.004 across three countries — and the random-match
split at 0.369 is if anything slightly *pessimistic* against them, so it is not flattering itself.

### Permutation importance (held-out matches, drop in R²)

`sklearn.inspection.permutation_importance` on the split model over the 29 held-out matches,
`n_repeats=20`, `random_state=0`, `scoring="r2"`. The standard deviation over repeats is given
because several of these rows are smaller than it.

| feature | ΔR² | sd over 20 repeats |
|---|---:|---:|
| `ball_forward_5s` | +0.0589 | 0.0062 |
| `dist_opp_goal` | +0.0431 | 0.0106 |
| `losing_possession_events` | +0.0243 | 0.0041 |
| `loss_event_type` | +0.0204 | 0.0076 |
| `loss_duration` | +0.0133 | 0.0032 |
| `pressure_event_count` | +0.0041 | 0.0020 |
| `next_3_type` | +0.0035 | 0.0025 |
| `n_loser_events_5s` | +0.0032 | 0.0020 |
| `counterpress_count` | +0.0021 | 0.0013 |
| `dist_own_goal` | +0.0021 | 0.0023 |

How far the ball travelled forward for the team that won it dominates. That is a sensible thing
for a counter-press model to key on and it is available from events alone — but note it is partly
a *consequence* of the press rather than a description of it.

Two cautions on reading this table. First, `dist_opp_goal` is second at +0.0431 while `x` — the
same information in a different parameterisation — has fallen out of the top ten at +0.0020; the
booster picked one of two collinear encodings of field position and the other now permutes for
free, so **neither row can be read as "how much field position matters"**. Second, everything below
about +0.004 is within two standard deviations of nothing.

> v0.1.0's version of this table gave `dist_opp_goal` at +0.0147 and the README gave 0.020 for the
> same quantity, a 36% disagreement that also reordered the two documents' rankings. Both were
> computed on the old artefact, whose row order was a race (§5.1); the number above is from the
> canonical order, with the repeat count and seed stated so it can be reproduced.

### Sanity companion: StatsBomb's own `counterpress` flag

StatsBomb define `counterpress` verbatim as *"pressing actions within 5 seconds of an open play
turnover"* — the identical window to ours. It is a per-event boolean, not a control share, so it
is a companion and not a target.

- **60.8%** of all 6,896 detected open-play turnovers have at least one `counterpress` event in the
  window (4,196 of them); **64.2%** of the 6,195 *labelled* turnovers do (3,975 of them). The two
  populations differ because a turnover is labelled only when the 360 file covers its window, and
  the same paragraph elsewhere uses "open-play turnovers" to mean the full 6,896 set — so both are
  given rather than one being presented as "the" rate. Up to v0.1.0 this card gave 64.2% against
  the word "open-play turnovers", which is the labelled subset's figure under the full set's name.
- The three correlations below are over the 6,195 labelled rows, the only rows that have a label:
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
  The tracking metric spreads 0.257 → 0.701 across pitch thirds; the label spreads 0.464 → 0.542
  and the model's own predictions 0.464 → 0.542. The model is trained on the compressed version and
  does not know the real gradient.
- **Never scale a proxy number against the tracking metric's phase means.** Use
  `pressing.proxy.phase_reference_means()`, which returns the model's own. See §3.
- **Never present a proxy number next to a tracking number.** They are on different scales
  (label mean 0.517 sd 0.129; true metric mean 0.553 sd 0.184) and correlate at r = 0.26 to 0.47.
- **Never score `models/proxy_gbr.pkl.gz` on the held-out matches.** It was fitted on them. That
  returns r = 0.615 against the honest 0.369. Use `score_held_out`, which refuses (§5.2).
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

The model beats every baseline on the task it was given (0.369 against 0.197 for turnover `x` and
0.169 for the pitch third), generalises across three leagues to within 0.004, and knows things
beyond field position (residual r = 0.326 after removing a smooth function of `x`). Everything
else is bad news, and it belongs here rather than in a footnote:

- It is trained against a label that itself reproduces the truth at only **r = 0.26 to 0.47**, a
  ceiling set by the broadcast camera that no amount of modelling can lift. Composed, the proxy
  stands at about **r = 0.10 to 0.17** against the quantity the tracking report prints.
- Its predictions are **shrunk to 0.451 of the label's spread**, and §7.1 establishes that this is
  not a fixable property of the loss function. It ranks; it does not measure.
- It **does not know the field-position gradient that dominates the real metric** (+0.078 against
  +0.444), so any comparison across pitch thirds is meaningless.
- The v0.1.0 figures were not reproducible across processes and were printed to four decimals
  anyway. They are now reproducible, and they are lower than what was printed.

**If you need a number a coach can act on, use tracking. This is a rough ranking device, and the
report must watermark it as an estimate.**

## 11. Attribution

Data from StatsBomb open data (<https://github.com/statsbomb/open-data>), used under the StatsBomb
Public Data User Agreement, and Metrica Sports sample data
(<https://github.com/metrica-sports/sample-data>). Pitch control follows the Friends of Tracking
reference vendored under `validation/reference/` with its MIT licence.
