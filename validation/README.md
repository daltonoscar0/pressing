# Validation: pitch control against the Friends of Tracking reference

`pitch_control_validation.ipynb` is the evidence for the project's pitch-control acceptance
criterion — that `pressing.pitchcontrol` reproduces the published Friends of Tracking reference
implementation with a Pearson correlation above 0.95. It is committed **with its outputs**, so the
number below can be read without running anything.

## Headline

```
==============================================================================
HEADLINE: pressing.pitchcontrol vs the Friends of Tracking reference implementation
          Metrica sample match 1, every possession loss, reference grid and defaults
==============================================================================
  POOLED PEARSON r (all cells, all events) : 1.0000000000
  mean per-event Pearson r                 : 1.0000000000
  worst per-event Pearson r                : 1.0000000000
  events compared                          : 257 of 257 BALL LOST events
  failures (either implementation)         : 0
  cells compared                           : 411,200
  mean |difference|                        : 1.737e-16
  max  |difference|                        : 3.941e-15
  mean r against a y-flipped reference     : 0.618073  (guard: must be lower)
==============================================================================
  ACCEPTANCE CRITERION r > 0.95: PASS  (257 of 257 events clear it)
==============================================================================
```

At the reference's own no-clamp convention and in float64, the agreement is exact to floating-point
rounding on every possession loss in the match — not "above the threshold", but the same numbers.
The shipped configuration (float32, `clamp="renormalise"`) scores a pooled r of 0.999563, with one
event of 257 below 0.95; section 9 of the notebook shows that gap is entirely the deliberate
probability clamp described under **Caveats**, not a modelling difference.

## Re-measured for the v0.1.0 tag

The comparison was re-run from scratch outside the notebook before tagging, over the same 257 events
and the same reference grid, to confirm nothing had drifted while the rest of the package was built.
It was run twice on the tagging day, hours apart. Nothing moved:

| configuration | pooled r | mean per-event r | min per-event r | events < 0.95 | max abs difference |
|---|---|---|---|---|---|
| `clamp="raw"`, float64 | 1.0000000000 | 1.0000000000 | 1.0000000000 | 0 of 257 | 3.941e-15 |
| shipped default (`clamp="renormalise"`, float32) | 0.9995631111 | 0.9991927210 | 0.9433488807 | 1 of 257 | 6.457e-01 |

Same run: 411,200 cells compared per configuration, mean absolute difference 1.737e-16 and 7.183e-04
respectively, reference maximum cell value 1.005366, raw `PA+PD` checksum range 0.990000 to 2.800681,
y-flip guard 0.618073 and 0.618334 (both far below the aligned correlation).

Timings from the second of those runs: the reference took 315.75 s for the 257 surfaces, 1228.6 ms
each, against 20.66 ms each in the shipped float32 configuration and 27.53 ms in the reference's own
float64 convention — **59x** and **45x**. Both halves were timed on a machine shared with unrelated
work, so they are slower than the notebook's committed numbers and only the ratio means anything. The
correlations, as always, did not move: they were identical to ten decimal places across both reruns
and the notebook.

Timing on the same grid and the same 257 events: reference 248.28 s (0.9661 s per surface), ours
4.04 s (0.0157 s per surface) — **61.5x**. Wall-clock timings move with machine load; three full
runs on the same laptop gave 59.1x, 59.3x and the 61.5x recorded in the committed outputs. The
correlations do not move at all. That margin is the reason the demo fits its runtime budget: a
straight port of the reference's per-cell Python loop would spend four to five minutes on pitch
control alone in a single match.

## Method

* **Reference.** [LaurieOnTracking](https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking)
  at commit `7ed3ff857940d3f4611be55f3c08c79525c32171`, MIT, vendored byte-for-byte under
  `reference/` with its `LICENSE` and a `PROVENANCE.md` recording the sha256 of every file. The
  notebook re-checks those hashes before importing anything.
* **Patching.** The 2021 code does not run on the current numpy/pandas/scipy. It is **not** edited in
  place: `reference/modern_pandas.patch` is applied to a throwaway copy in a temporary directory, so
  the committed bytes stay identical to upstream and a reviewer can diff exactly what changed. Four
  changes, all printed in the notebook: three `Series.idxmax` API fixes, and `mode='nearest'` on the
  four `savgol_filter` calls (the default `mode='interp'` fits an edge polynomial and raises on the
  NaNs that `position.diff()` legitimately produces). The upstream tutorials' suggested workaround,
  `filter_='moving_average'`, is deliberately **not** used: the code tests for `'moving average'`
  with a space, so that string matches no branch and silently disables smoothing altogether, which
  would change every velocity and therefore every surface.
* **Data.** Metrica sample match 1, loaded by the reference's own `Metrica_IO` and
  `Metrica_Velocities` and fed to both implementations. That is deliberate: this notebook tests our
  pitch-control mathematics, not our loader.
* **Event set.** All 257 `BALL LOST` events — the project's counter-press trigger — rather than a
  hand-picked sample. `BALL OUT` is reported separately by the product and is not compared here.
* **Held fixed for an apples-to-apples comparison.** The reference's own grid
  (`n_grid_cells_x=50`, `field_dimen=(106., 68.)` → shape `(32, 50)`), the reference's model
  constants (checked equal to ours cell by cell in the notebook), the reference's frame and ball
  semantics (tracking snapshot from the event row's `Start Frame`, ball from the event row's
  `Start X`/`Start Y`, **not** the tracked ball columns), and **offsides disabled on both sides**.
* **Alignment guards, all asserted.** Shape is `(32, 50)`; `corr(ours, flipud(ref))` must be lower
  than `corr(ours, ref)` — a y flip is the one misalignment that neither a shape check nor a sign
  check catches, and it costs about 0.4 of correlation without looking obviously wrong; `corr` must
  be positive, since an attacking/defending swap gives exactly −1.
* **Failures are counted, not dropped.** Both implementations are wrapped per event and any
  exception is recorded with its side and message. There were none.

## Caveats

* **The reference's pitch is 106 x 68; the product's is 105 x 68.** The comparison runs on the
  reference's grid, because correlating surfaces evaluated at different physical points would be
  meaningless. The canonical project frame is bottom-left-origin metres on 105 x 68, and the
  conversion between the two (`to_reference_frame` / `from_reference_frame`) is an explicit,
  unit-tested function. The surfaces the PDF renders are on the 105 x 68 grid and are therefore not
  the arrays compared here.
* **One deliberate divergence: probability clamping.** Forward Euler takes its final step whole, so
  the two teams' probabilities can end up summing above 1 — on 0.675% of cells in this match, and up
  to 2.80 on the worst cell. The reference returns those untouched (its own sanity check
  `assert 1 - checksum < model_converge_tol` is one-sided and never fires on an overshoot), and its
  surface does reach 1.005366 at one cell. Our default renormalises them. Every disagreement above
  0.01 between the shipped default and the reference sits on such a cell — the notebook asserts
  exactly that. We would rather diverge from the reference than emit a probability above 1 into a
  coach-facing report.
* **The EPV grid has no published provenance.** `reference/EPV_grid.csv` ships in the same MIT
  repository with no fitting code, no source dataset and no documentation of what it was fitted on.
  It is used because it is redistributable and already `(32, 50)`, not because it is authoritative,
  and any report surfacing its numbers should label it as the Friends of Tracking reference grid
  rather than as our own estimate.
* **This is not a test of the loader, the offside rule, or the product's ball position.** Velocities
  and positions come from the reference's loader; offsides are off on both sides (our
  `check_offsides` deliberately does not raise where the reference does, around substitutions and
  untracked keepers); and the report takes the ball from tracking at the loss frame, not from the
  event row. Those paths have their own tests under `tests/`.
* **It is not a validation of the model against reality.** It shows we agree with a specific
  published implementation of a specific model, and nothing about whether that model predicts
  football.

## Reproducing

Metrica sample data lives under `data/raw/`, which is gitignored; fetch it with the project loader
first. Then, from the repository root:

```sh
uv run jupyter nbconvert --execute --to notebook --inplace \
    --ExecutePreprocessor.timeout=3600 validation/pitch_control_validation.ipynb
```

It takes about five minutes, almost all of it the reference's per-cell Python loop. The notebook
fails loudly rather than degrading quietly: it asserts the vendored hashes, the model constants, the
grid alignment, the four orientation guards, the acceptance threshold, and the claim that every
clamp disagreement lands on an overshooting cell.
