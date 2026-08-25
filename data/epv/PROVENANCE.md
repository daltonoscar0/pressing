# EPV grid provenance

## What this is

`fot_epv_grid_32x50.csv` is a byte-for-byte copy of `EPV_grid.csv` from the
Friends of Tracking reference repository. It is **not** our own estimate.

| | |
|---|---|
| Source repository | https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking |
| File in that repository | `EPV_grid.csv` |
| Commit | `7ed3ff857940d3f4611be55f3c08c79525c32171` (2021-06-15, branch `master`) |
| Licence | MIT, Copyright (c) 2021 Friends-of-Tracking-Data-FoTD |
| Licence text | `LICENSE.friends-of-tracking`, in this directory |
| sha256 | `5733e1d99f311affd4ac39ebd48b245562e1d8086339281d5b0e7822a2b53274` |
| Author of the original code | Laurie Shaw (@EightyFivePoint) |

The MIT licence permits copying and redistribution provided the copyright
notice and permission notice ship with the copy, which is why
`LICENSE.friends-of-tracking` sits next to the grid rather than only in the
repository root.

## Shape and orientation

32 rows by 50 columns of comma-separated floats, i.e. `(n_y, n_x)`.

- Row 0 is the **most negative y**, the bottom touchline. Same convention as a
  pitch control surface.
- Column 49 is the **goal being attacked**, for `attack_direction == +1`.
- Cell size on the reference's own 106 x 68 m field is 2.12 m by 2.125 m.
  Re-expressed on this project's canonical 105 x 68 m pitch it is 2.10 m by
  2.125 m; `pressing.epv` rescales proportionally.

Measured properties (`numpy.loadtxt` on this exact file): min 0.0041, max
0.5714, mean 0.0264065, sum 42.2504, argmax at `[15, 49]`, argmin at `[2, 0]`.
It is exactly symmetric in y (`allclose(e, flipud(e))` is True with a maximum
absolute difference of 0.0) and its column means increase strictly with x.

## Honest caveat

The upstream repository ships this grid with **no stated provenance, no
fitting code and no description of the dataset it was fitted on**. We use it
because it is redistributable, because it is already exactly the same shape as
our default pitch control grid, and because sharing an EPV surface with the
reference implementation removes EPV as a confounder when we validate pitch
control against that reference.

Any coach-facing output must therefore label these numbers as the *Friends of
Tracking reference EPV grid*, never as a model of ours. Deriving our own grid
is a separate, opt-in job whose output belongs under `data/raw/`, not here.
