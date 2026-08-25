# Vendored third-party code: Friends of Tracking reference implementation

The files in this directory are **not ours**. They are copied verbatim from a third-party
repository and are redistributed here under their own licence, reproduced in full in `LICENSE`.

| | |
| --- | --- |
| Upstream | https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking |
| Branch | `master` |
| Commit | `7ed3ff857940d3f4611be55f3c08c79525c32171` (2021-06-15T08:40:03Z) |
| Licence | MIT, Copyright (c) 2021 Friends-of-Tracking-Data-FoTD (see `LICENSE`) |
| Author | Laurie Shaw |
| Retrieved | 2026-08-25 |

The MIT licence permits use, copying, modification and redistribution provided the copyright
notice and permission notice ship with the copies. `LICENSE` is that notice, unmodified.

## Why these files are vendored rather than downloaded

`validation/pitch_control_validation.ipynb` is the evidence for the project's pitch-control
acceptance criterion. Evidence that depends on a network fetch from a dormant 2021 repository is
not evidence. Vendoring pins the exact bytes that produced the committed correlation number.

## Files

Only the subset needed to reproduce a pitch control surface is vendored; `Metrica_Viz.py`, the
tutorial scripts and the upstream `README.md` are not, because nothing here uses them.

| File | sha256 | Role |
| --- | --- | --- |
| `Metrica_IO.py` | `1aea1039b0d4eff918db2d01fecbde2acdf679aca6ea4735a93ce80b107d77de` | reads Metrica CSVs, metric coordinates, single playing direction |
| `Metrica_Velocities.py` | `b17b2e3c54a8e88814f78b3d978df038b790e34d82f300e99c92a813b18ed541` | Savitzky-Golay smoothed player velocities |
| `Metrica_PitchControl.py` | `d0b6644fff32fd31c00ddbf23a37762dae77894b7d9126a34e191b8e015f0cb2` | the pitch control model under comparison |
| `Metrica_EPV.py` | `b1186221cb8b66166ad3acb37b6b9bf5746f10212f0506cfb1525c426c1cedb4` | EPV grid lookup, kept alongside the grid it reads |
| `EPV_grid.csv` | `5733e1d99f311affd4ac39ebd48b245562e1d8086339281d5b0e7822a2b53274` | the (32, 50) EPV surface |
| `LICENSE` | `9172ed42b991e0f36fd9220cf7c7f9ea88970513b0e6ec72ba25d0f733d0bf39` | MIT notice, verbatim |

The notebook re-checks every hash in this table before it imports anything, so a locally edited
"reference" cannot silently become the thing we validate against. The same digests are repeated in
`MANIFEST.sha256` in the usual format, so they can also be checked without Python:

```sh
shasum -a 256 -c MANIFEST.sha256
```

`modern_pandas.patch` is **ours**, not upstream. It is not applied to the files in this directory:
the notebook copies them to a temporary directory and patches the copy, so the vendored bytes stay
identical to upstream and a reviewer can see exactly what we changed and nothing else. The patch
carries four fixes: three are pure `pandas` API compatibility with identical semantics, and the
fourth passes `mode='nearest'` to `savgol_filter`, without which the reference raises on the NaNs
its own velocity calculation produces and cannot run at all. That fourth change alters edge
handling, but the smoothed velocities it produces are fed to *both* implementations, so it cannot
bias the comparison. See the notebook for the reasoning behind each.

## Verifying this directory against upstream

```sh
git clone https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking /tmp/lot
git -C /tmp/lot checkout 7ed3ff857940d3f4611be55f3c08c79525c32171
for f in Metrica_IO.py Metrica_Velocities.py Metrica_PitchControl.py Metrica_EPV.py \
         EPV_grid.csv LICENSE; do
  diff -q "/tmp/lot/$f" "$f" || echo "DIFFERS: $f"
done
```

## Known caveats in the upstream code

Recorded here so that nobody rediscovers them the hard way.

- `generate_pitch_control_for_event` returns an array of shape `(n_y, n_x) = (32, 50)`. Its own
  docstring says `(n_grid_cells_x, n_grid_cells_y)`, which is wrong.
- Row 0 of every surface is the **lowest** y (the bottom touchline). `Metrica_Viz.plot_pitchcontrol`
  applies `np.flipud` before `imshow` for exactly this reason.
- `Metrica_Viz.plot_EPV` does *not* apply that same `flipud`. The inconsistency is invisible only
  because `EPV_grid.csv` happens to be exactly symmetric in y.
- The surface is the probability that the **team in possession at that event** controls a cell, not
  the home team.
- `default_model_params()` defines `max_player_accel` and never uses it.
- The convergence check `assert 1 - checksum < model_converge_tol` is one-sided: it passes trivially
  whenever the two probabilities sum to more than 1, so it does not detect over-accumulation.
- `EPV_grid.csv` has no stated provenance: no fitting code, no source dataset, no documentation of
  what it was fitted on.
- The upstream tutorials suggest `filter_='moving_average'` to work around the Savitzky-Golay
  failure on modern scipy. `Metrica_Velocities.py` tests for `'moving average'` with a space, so
  that string matches no branch and silently disables smoothing altogether.
