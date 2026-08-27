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

## Re-measured for the v0.1.1 tag

The notebook was re-executed from a clean kernel into `out/` — leaving the committed file untouched —
over the same 257 events and the same reference grid, to confirm nothing had drifted:

```sh
uv run jupyter nbconvert --execute --to notebook --output-dir out \
    --output pitch_control_rerun.ipynb \
    --ExecutePreprocessor.timeout=3600 validation/pitch_control_validation.ipynb
```

0 errors across all 20 code cells, and the comparison table came back identical to the committed one:

| configuration | pooled r | mean per-event r | min per-event r | events < 0.95 | max abs difference |
|---|---|---|---|---|---|
| `clamp="raw"`, float64 | 1.000000 | 1.000000 | 1.000000 | 0 of 257 | 3.941e-15 |
| `clamp="raw"`, float32 | 1.000000 | 1.000000 | 1.000000 | 0 of 257 | 2.645e-03 |
| shipped default (`clamp="renormalise"`, float32) | 0.999563 | 0.999193 | 0.943349 | 1 of 257 | 6.457e-01 |

Same run: 411,200 cells compared per configuration, mean absolute difference 1.737e-16 and 7.183e-04
in the first and last rows, reference maximum cell value 1.005366, largest raw `PA+PD` checksum
2.800681 on 2,776 cells of 411,200 (0.675%), every one of the 2,462 disagreements above 0.01 sitting
on such a cell, y-flip guard 0.618073. The headline `POOLED PEARSON r` line prints ten decimals for
the reference convention and reads `1.0000000000`.

### Timings, and why only one number here is worth reading

**The ratio between the two implementations is the measurement; the seconds are a property of whatever
else the laptop was doing.** Two runs are recorded in this repository:

| run | reference total | per surface | ours (shipped config) | per surface | ratio |
|---|---:|---:|---:|---:|---:|
| the committed notebook's own timing cell | 248.28 s | 0.9661 s | 4.04 s | 0.0157 s | **61.5x** |
| the v0.1.1 rerun above, at a 1-minute load average of 4.35 → 6.12 | 151.61 s | 0.5899 s | 2.48 s | 0.0096 s | **61.2x** |

The absolute totals differ by a factor of 1.6 between those two runs of identical code on identical
data. The ratio moves by 0.3x and the correlations do not move at all. Earlier versions of this file
carried two separate timing paragraphs quoting a third and fourth pair of figures with different
ratios; they disagreed with each other, neither could be tied to a run recorded here, and they have
been removed rather than reconciled. That margin — sixty-something to one — is the reason the demo
fits its runtime budget: a straight port of the reference's per-cell Python loop would spend minutes
on pitch control alone in a single match.

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
uv run jupyter nbconvert --execute --to notebook --output-dir out \
    --output pitch_control_rerun.ipynb \
    --ExecutePreprocessor.timeout=3600 validation/pitch_control_validation.ipynb
```

That writes the executed copy into gitignored `out/` and leaves the committed notebook alone, so a
rerun cannot quietly overwrite the evidence with itself. Use `--inplace` instead only when you mean to
replace the committed outputs.

On the run recorded above it took about three minutes of wall clock, almost all of it the reference's
per-cell Python loop, and it will take considerably longer on a busy machine. The notebook
fails loudly rather than degrading quietly: it asserts the vendored hashes, the model constants, the
grid alignment, the four orientation guards, the acceptance threshold, and the claim that every
clamp disagreement lands on an overshooting cell.
