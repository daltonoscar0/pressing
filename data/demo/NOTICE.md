# Data notice — the bundled demo match

**The match in this directory is derived from sample data published by
[Metrica Sports](https://github.com/metrica-sports/sample-data).**

Metrica Sports released two full anonymised matches of tracking and event data
publicly, and asks in the repository's README only that users "be responsible
with the use of this data" and acknowledge the source. This file is that
acknowledgement, and it is deliberately the first thing in the directory.

Upstream: <https://github.com/metrica-sports/sample-data>

## These are not Metrica's files

Nothing here is a copy of anything Metrica published. Metrica's sample data is
six CSVs; this directory holds two parquet files in this package's own
canonical schema, reconstructed from Sample Game 1 by
`pressing.demo.build_demo_assets`. Specifically, relative to the upstream CSVs
the assets are:

- **restructured** — the two wide, one-column-per-player tracking CSVs are
  melted into one long frame, one row per entity per frame, with the ball as a
  row rather than a pair of trailing columns;
- **re-coordinated** — Metrica's unit square with `y` increasing downward
  becomes metres on a 105 x 68 pitch with the origin at the bottom-left corner
  and `y` increasing upward;
- **rotated** — the second half is mirrored so that the home team attacks `+x`
  for the whole match, which is not how the upstream data is laid out;
- **quantised** — positions are stored as int16 centimetres, so every
  coordinate is rounded. Measured over all 3,278,389 rows against a load of the
  upstream CSVs, the largest position error introduced is 0.005 m;
- **stripped of derived columns** — velocities are not stored at all. They are
  recomputed on read by `pressing._io.read_tracking_parquet`, from the
  quantised positions, so they do not exactly reproduce velocities computed
  from the upstream CSVs: measured over 3,258,312 finite samples, the median
  absolute difference is 0.010 m/s, the 99.99th percentile is 0.080 m/s, and 61
  of the `vx` samples and 85 of the `vy` samples differ by more than 0.5 m/s,
  with a worst case of 5.54 m/s (`vx`) and 7.06 m/s (`vy`) at the boundaries of
  short tracked runs where the smoothing window is truncated;
- **re-encoded** — zstd level 19 parquet, not CSV.

Events keep their coordinates exactly: measured against a load of the upstream
events CSV, the maximum absolute difference in `t`, `x`, `y`, `end_x` and
`end_y` is 0.0, and every categorical field matches row for row. Their
structure is still this package's schema, not Metrica's columns.

The upshot: this is a lossy, restructured representation built for one purpose,
which is that `pressing demo` runs on a fresh clone. **Anyone who wants
Metrica's data should get it from Metrica**, at the link above. Do not treat
these files as a substitute for the originals, and do not cite numbers from
them as Metrica's numbers.

## Why it is committed at all

The `metrica-sports/sample-data` repository **publishes no licence file**;
GitHub reports its licence as null, which by default means all rights reserved.
There is therefore no licence grant to rely on here, and the honest position is
to say so rather than to imply one.

These derived files are included for exactly one reason: the tool's headline
acceptance criterion is that the demo produces the full PDF **offline on a
fresh clone**, and that is not achievable if the demo has to download 65 MB of
CSVs first. They are not published as a dataset, they are not offered as a
mirror, and no part of this project's value depends on redistributing them.

**If Metrica Sports would prefer these files not be distributed, they will be
removed on request, without argument.** In the meantime this repository has no
git remote, so nothing here is published anywhere.

## Provenance

Derived on **2026-08-26T03:51:41+00:00** (UTC) from Metrica sample Game 1, by
`pressing.demo.build_demo_assets("<dir holding Sample_Game_1/>")`.

### The upstream commit

**Upstream commit: `e706dd506b360d69d9d123d5b8026e7294b13996`** — the tip of
`master` in `metrica-sports/sample-data`, committed 2021-04-15, verified
2026-08-27.

The commit id was not written down when the CSVs were fetched, so it was
recovered afterwards by content rather than asserted from memory, and the three
digests below are what made that possible. Each of the three source files was
hashed as a git blob and matched against the repository's own tree, then
downloaded again from that commit and compared byte for byte:

```sh
# the blob ids of the local copies, and the tree GitHub serves for that commit
git hash-object data/raw/metrica/Sample_Game_1__Sample_Game_1_Raw*.csv
curl -s 'https://api.github.com/repos/metrica-sports/sample-data/git/trees/\
e706dd506b360d69d9d123d5b8026e7294b13996?recursive=1'
```

| Upstream file | Bytes | git blob | SHA-256 |
|---|---:|---|---|
| `Sample_Game_1_RawEventsData.csv` | 139,446 | `3c426dd4f442dd433bccf89e3634f91aa310cd8f` | `097dda6345e17390dca095331458fd00e47960c480c76f4356fd4edf2575ef05` |
| `Sample_Game_1_RawTrackingData_Away_Team.csv` | 32,782,073 | `ff409a5680fd75acc7cebcf786827214c687478b` | `0a3df68a92af0290ede472e62de3b6489ddee29223da09bdf0cf76c8e0ab0618` |
| `Sample_Game_1_RawTrackingData_Home_Team.csv` | 32,782,961 | `ce2e91c0e633426cec1146f4e728b72417454e3d` | `0a3d0eff7785950379c587d3597b86a2d91c2492281b4652e1abb97c99b6b364` |

All three blob ids appear verbatim in that commit's tree under
`data/Sample_Game_1/`, and re-downloading all three files from
`raw.githubusercontent.com` at that commit and running `cmp` against the local
copies reported no difference on any of them.

**What that does and does not establish.** It establishes that the bytes these
assets were derived from are the bytes that commit publishes. It does not
establish that `e706dd50` was the checkout that happened to be fetched: these
three files entered the repository in commit
`5f825443fa855896701b1460f193cca2899c61e6` (2020-04-01) and have never been
modified since — `GET /repos/metrica-sports/sample-data/commits?path=…` returns
exactly that one commit for each of them — so every commit from then to the tip
carries identical content. Any of them would be a true answer; the tip is the
one a reader who clones today will get.

What was written:

| Asset | Bytes | Rows | Contents |
|---|---:|---:|---|
| `metrica_game_1_tracking.parquet` | 7,754,775 | 3,278,389 | 25 Hz tracking, 28 players plus the ball, both halves |
| `metrica_game_1_events.parquet` | 48,324 | 1,745 | canonical events, `extra` flattened to JSON |

The tracking asset keeps Metrica's native 25 Hz. It is not decimated: 25 Hz
already clears this package's 10 Hz schema minimum, and 25/10 is not an
integer, so decimating would trade a smaller file for timestamp jitter.

## Other data in this repository

- The **EPV grid** under `data/epv/` is the Friends of Tracking reference grid,
  MIT-licensed and vendored with its own notice.
- **No StatsBomb data of any kind is committed**, and none ever should be —
  their public data agreement forbids redistribution. It stays in the
  gitignored `data/raw/`.
- `data/raw/` is gitignored in full. `data/demo/` is not, and this file exists
  because of that difference.
