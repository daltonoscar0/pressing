# Transfermarkt market-value snapshot

These two files are the offline fallback for `pressing season`. They exist so the market-value
join works on a fresh clone with no network. When `data/raw/transfermarkt/` holds the full
tables, or the mirror is reachable, those are used instead — this snapshot is tried last,
because it is by construction the stalest of the three routes.

## What it is

| File | Rows | Bytes | sha256 (first 16) |
|---|---:|---:|---|
| `players.csv.gz` | 50,149 | 1,118,421 | `1f1758ffc3787eef` |
| `player_valuations.csv.gz` | 656,301 | 3,789,354 | `933761f8b7eb36a9` |

- **Snapshot taken:** 2026-08-25.
- **Most recent market value in the file:** 2026-06-12. Value dates run 2000-01-20 to 2026-06-12,
  covering 41,528 distinct players.
- **Derived, not original.** Both files are column-reduced copies of the upstream tables, keeping
  only what the join reads:
  - `players.csv.gz`: `player_id`, `name`, `first_name`, `last_name`,
    `current_club_domestic_competition_id`, `current_club_name`, `last_season`
    (upstream has 27 columns).
  - `player_valuations.csv.gz`: `player_id`, `date`, `market_value_in_eur`,
    `player_club_domestic_competition_id` (upstream has 6).
  No row was dropped, edited or recomputed.

## Where it came from

- **Upstream project:** [`dcaribou/transfermarkt-datasets`](https://github.com/dcaribou/transfermarkt-datasets),
  licensed **CC0-1.0**.
- **Exact source:** the project's public, no-authentication Cloudflare R2 mirror,
  `https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/players.csv.gz` and
  `.../player_valuations.csv.gz`, downloaded 2026-08-25 (4,160,452 and 6,869,624 bytes gzipped
  as served).
- **Verified fallbacks**, both reachable without credentials on the same date: the Kaggle bundle
  `davidcariboo/player-scores`, and the HuggingFace mirror
  `ngeorgea/transfermarkt-player-scores` (cc0-1.0).

## Licence, stated plainly

CC0-1.0 covers **dcaribou's repackaging** of the data. It does not, and cannot, cover the
underlying rights in the material Transfermarkt publishes, and Transfermarkt's own terms
prohibit scraping. This snapshot is included for non-commercial personal analysis only. It will
be removed on request.

Market values are Transfermarkt's own community-driven estimates. They are not transfer fees,
not wages, and not audited.
