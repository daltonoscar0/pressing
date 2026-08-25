# Test fixtures

Everything here is committed so the test suite runs completely offline.

## `metrica/`

Truncated copies of the Metrica Sports sample data, kept byte-for-byte in the
upstream CSV format (three header rows and all) so the parser is exercised on
real bytes rather than on a hand-written approximation of them. Only whole
frames were removed; no cell was edited.

Frame blocks kept, chosen so the slices still contain both periods, a
substitution and a real possession loss:

| game | blocks (inclusive frame ranges)          | why                                                       |
| ---- | ---------------------------------------- | --------------------------------------------------------- |
| 1    | 1-120, 46647-46766, 71269-71388          | kick-off; home substitution at frame 46706; period 2 start |
| 2    | 1-120, 67942-68061                       | kick-off; period 2 start, including a `BALL LOST` at 68031 |

Events rows are kept when either their Start Frame or their End Frame falls in
one of the blocks.

The two games are both needed: they attack in opposite directions in the first
half, and game 2 writes `Player 26` with a space in the tracking header.

Source: <https://github.com/metrica-sports/sample-data>, which asks that the
source be acknowledged wherever the data is used.
