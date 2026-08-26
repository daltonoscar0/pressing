# pressing

**What it is.** When a team loses the ball, the next five seconds decide whether they get it straight
back or spend the next minute defending. `pressing` measures exactly that. It takes tracking data for a
match, finds every moment possession changed hands in open play, and computes how much of the pitch the
losing team had won back one, two, three, four and five seconds later — using a pitch-control model that
accounts for where all twenty-two players were, how fast they were already moving, and how long the ball
would take to reach each blade of grass. The output is a per-loss score between 0 and 1, aggregated per
player, per team, and per third of the pitch, plus heatmaps of the individual moments.

In plain terms: it answers "when we lose it, do we swarm or do we jog back?", and it answers it with a
number that comes from the players' actual positions rather than from someone's eye.

**Status.** The analytics engine is built, tested and validated. The product wrapped around it — the PDF
report, the offline demo, the event-only proxy model and season mode — is not. See
[What is not built yet](#what-is-not-built-yet). All four CLI subcommands currently exit 1.

## Install

Requires Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv venv --python 3.11
uv sync --all-extras
uv run pytest -q          # the suite runs fully offline
uv run pressing --help
```

## Quickstart

`uv run pressing report ...` does not work yet — it prints `report is not implemented yet.` and exits 1.
Until the report module lands, the pipeline is driven from Python:

```python
from pressing import counterpress as cp
from pressing._io import load_metrica_match

match = load_metrica_match(1, "data/raw/metrica")     # Metrica sample game 1
losses = cp.find_possession_losses(match.tracking, match.events)
scored = cp.score_losses(match.tracking, match.events, losses)

print(cp.per_phase(scored))      # mean recovery by team and by third of the pitch
print(cp.per_player(scored))     # per-player attribution
print(cp.summarise_match(scored))
```

Metrica's sample data is not committed (see [Data sources and licences](#data-sources-and-licences)); it
lives under the gitignored `data/raw/`. The test suite does not need it — `tests/fixtures/metrica/`
contains synthetic files that copy the *format* only.

## The metric

At every open-play possession loss:

1. Take the frame where the ball changed hands.
2. At six timepoints — the moment of the loss and each second after it, out to five seconds — compute a
   pitch-control surface: for every cell of a 50 x 32 grid, the probability that the team that *lost* the
   ball would win it if it arrived there.
3. Average that surface over the pitch. That is the losing team's share at that instant.
4. The loss's **recovery score** is the mean of the six shares. 0.5 means the two teams are even; above
   0.5 means the pressing team owns more of the pitch than the team that just won the ball.

A second weighting (`recovery_epv`) weights each cell by how dangerous it is rather than treating all
grass equally, using the Friends of Tracking expected-possession-value grid.

### The caveat that matters most

**Recovery is dominated by where on the pitch you lost the ball, so the numbers are only comparable
within a phase.** Measured on Metrica sample game 1, mean recovery by the third of the pitch the ball was
lost in, counted from the losing team's own goal:

| phase       | away | n  | home | n  | pooled | n   |
|-------------|------|----|------|----|--------|-----|
| build-up    | 0.238 | 13 | 0.280 | 11 | 0.257 | 24  |
| middle      | 0.439 | 53 | 0.494 | 60 | 0.468 | 113 |
| final third | 0.703 | 61 | 0.699 | 57 | 0.701 | 118 |

That is a near-3x gradient from one end of the pitch to the other, and it dwarfs the difference between
the two teams (0.238 vs 0.280 in build-up). It is not a modelling artefact: losing the ball in your own
build-up means your opponents are already behind you and facing your goal, and losing it in their box
means you have ten players goalside of the ball. Comparing a defender's build-up losses against a
winger's final-third losses is meaningless. Compare within a phase, and mind the sample size — build-up
rests on 24 losses out of 255 in this match.

Two further honest limits:

* The ball has no tracking row in 39-41% of Metrica frames, so the ball position inside a five-second
  window is sometimes carried forward from the last known frame. That is recorded per loss
  (`n_ball_filled`) rather than hidden.
* One open-play loss in game 1 cannot be located at all — no tracked ball at its end frame, no event
  coordinates either side. It is reported with `location_source == "none"` and skipped, not guessed:
  256 losses detected, 255 scored.

## Validation

`pressing.pitchcontrol` is validated against the published
[Friends of Tracking](https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking) reference
implementation of the same Spearman model, at commit `7ed3ff85`, vendored byte-for-byte under
`validation/reference/` with its MIT licence. The executed notebook is committed with its outputs at
`validation/pitch_control_validation.ipynb`; `validation/README.md` documents the method in full.

Re-measured independently for this README over **all 257 `BALL LOST` events** of Metrica sample game 1 on
the reference's own grid (411,200 cells):

| configuration | pooled Pearson r | mean per-event r | min per-event r | events below 0.95 |
|---|---|---|---|---|
| model at the reference's own convention (`clamp="raw"`, float64) | **1.0000000000** | 1.0000000000 | 1.0000000000 | 0 |
| **shipped default** (`clamp="renormalise"`, float32) | **0.999563** | 0.999193 | 0.943349 | **1 of 257** |

Both rows matter and the second is not hidden. At the reference's own no-clamp convention we reproduce it
to floating-point rounding — maximum absolute difference over every cell of every event, 3.941e-15. The
shipped default deliberately diverges: forward Euler takes its final integration step whole, so the two
teams' raw probabilities can sum above 1 (largest raw sum measured, 2.800681; the reference's own surface
reaches 1.005366 at one cell). We renormalise those cells rather than emit a probability above 1 into a
coach-facing report, and that single choice accounts for the entire gap, including the one event that
falls to r = 0.943.

**Speed.** The reference is a per-cell Python loop and measures 0.687-0.699 s per 50 x 32 surface on this
laptop, against 9.96-10.27 ms here for the shipped configuration — 68-69x, over three runs of 40 events.
The committed notebook's full 257-event run recorded 61.5x. Wall-clock timings move with machine load;
the correlations do not move at all.

**What this is not.** It shows we agree with one specific published implementation of one specific model.
It says nothing about whether that model predicts football, and it does not test the loader, the offside
rule, or the product's choice of ball position — those have their own tests.

## Data sources and licences

Nothing in this table is committed to git except where the row says so. `data/raw/` is gitignored.

| Source | Used for | Licence | How this repo treats it |
|---|---|---|---|
| [Metrica Sports sample data](https://github.com/metrica-sports/sample-data) | The two sample matches everything is measured on | **None published.** GitHub reports `license: null`, which is all rights reserved by default. The README asks only that users be responsible with the data and acknowledge the source. | Credited by name here and wherever its numbers appear. Raw CSVs are **never committed** — they live in gitignored `data/raw/`. The offline demo asset, when it is built, will be a *derived* representation (quantised, downsampled parquet) shipped with a `NOTICE.md` recording the upstream commit and stating it will be removed on request. `tests/fixtures/metrica/` contains **synthetic** files that copy the file format only, generated by a committed script. |
| [Friends of Tracking / LaurieOnTracking](https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking) | Pitch-control reference implementation; the EPV grid | **MIT**, Copyright (c) 2021 Friends-of-Tracking-Data-FoTD | Vendored byte-for-byte with the MIT notice alongside it, under `validation/reference/` and `data/epv/`, plus `PROVENANCE.md` files recording the commit and the sha256 of every file. The EPV grid is labelled as theirs, not as our own estimate — it ships with no fitting code and no documented source dataset. |
| [StatsBomb open data](https://github.com/statsbomb/open-data) (now `hudl/open-data`) | Planned label source for the event-only proxy model, and for season mode | **Restrictive.** The `LICENSE.pdf` "StatsBomb Public Data User Agreement" governs, and is stricter than the repo README's attribution-only wording. Clause 1.2.1 forbids the user to "edit, distort, distribute, reproduce, sell or in any way provide the data to any external or third party"; clause 1.2.2 forbids commercially exploiting "the data or any analysis derived from the use of the Service"; clause 1.4 requires accrediting any published analysis with the StatsBomb brand logo. | **Never committed, in any form.** Downloads stay in gitignored `data/raw/`. Any season-mode output built from it must carry StatsBomb attribution (CSV header, chart caption, PDF footer). This tool is non-commercial personal analysis; do not put commercial framing on it. Note that a PDF handed to a coach containing StatsBomb-derived numbers sits close to clause 1.2.1 — read the agreement before distributing one. |
| [dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) | Planned market-value join for season mode | **CC0-1.0** on the repackaging | CC0 covers dcaribou's repackaging, **not** the underlying rights: the data is scraped from Transfermarkt, whose own terms prohibit scraping. Cached snapshots stay in gitignored `data/raw/`, and any published number must record its capture date. |
| [SkillCorner open data](https://github.com/SkillCorner/opendata) | Evaluated and **not used** | MIT | Rejected on the facts, not the licence: it ships no event stream ("We are not an event data provider ourself"), its 10 open matches are Australian A-League with zero overlap against StatsBomb's competitions, and its `dynamic_events.csv` is derived from the same tracking that would generate our labels — training on it would be circular. |

## What is not built yet

Honest inventory, so a reader can tell a stale note from a real gap.

**Built, tested and used:**

* `_io.py` — Metrica loaders, the canonical frame (metres on 105 x 68, origin bottom-left, y up, home
  attacking `+x` for the whole match), velocity computation, the demo parquet writer/reader.
* `possession.py` — possession segmentation and open-play loss detection.
* `pitchcontrol.py` — the vectorised Spearman model, validated above.
* `epv.py` — the expected-possession-value surface and lookup.
* `counterpress.py` — the five-second recovery metric and its aggregations.
* `phases.py` — thirds of the pitch, measured from the losing team's own goal.
* `plots.py` — the matplotlib renderers.

**Stubs that raise `NotImplementedError`:**

* `report.py` — the six-page reportlab PDF template.
* `demo.py` — the bundled offline demo. `data/demo/` is empty; no demo asset has been built, so
  `pressing demo` has nothing to run on even once the code lands.
* `proxy.py` — the event-only proxy model. Its held-out correlation is **not** reported anywhere yet,
  because it has not been trained. The spec's original plan (train on a Metrica + SkillCorner overlap)
  is impossible — see the SkillCorner row above — so the intended label source is StatsBomb 360
  freeze-frames instead, split by match. Those give positions but no velocities and no player
  identities, and cover only the broadcast `visible_area` polygon, so any label built from them is an
  approximation of the true tracking-derived value, not ground truth. That caveat will stay attached to
  whatever number the model eventually produces.
* `season.py` — league-wide ranking and the market-value join.

**Consequently:** all four CLI subcommands (`report`, `proxy`, `season`, `demo`) print a message and exit
1. `pressing --help` and `pressing --version` work.

Open items are tracked in [TODO.md](TODO.md).
