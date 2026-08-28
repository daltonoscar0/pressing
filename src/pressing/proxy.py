"""Event-only proxy for the tracking-derived counter-press metric.

Most matches have event data and no tracking. This module trains a gradient
boosted model that predicts a counter-press control share from event features
alone -- turnover location, time to the next event, the next three actions and
the pressure flags -- so the same report can be produced from an event file,
watermarked as an estimate.

What the labels actually are, and what they are not
---------------------------------------------------
The spec asked for training on a "Metrica + SkillCorner overlap". That set does
not exist: SkillCorner ships no event stream at all (their README says so
outright), its ten open matches are Australian A-League while StatsBomb open
data contains no Australian competition, and its ``dynamic_events.csv`` is
derived from the same tracking that would produce our labels, so training on it
would be circular. The trainable universe with paired 25 Hz tracking and
hand-tagged events is three Metrica matches, which cannot support an honest
split-by-match correlation.

The labels here therefore come from **StatsBomb 360 freeze-frames**. That is an
approximation of the tracking-derived value, not ground truth, and the size of
the approximation is measured rather than asserted -- see
:func:`calibrate_against_metrica` and the numbers recorded in the model card.
Four things differ from the real metric:

1. **Sampling.** The true metric evaluates pitch control at six evenly spaced
   instants, ``t0 .. t0+5s``. A 360 file carries one freeze-frame per *event*,
   so the window is sampled wherever the events happen to fall -- typically 3 or
   4 irregular instants, sometimes 2.
2. **Velocities.** A freeze-frame is positions only. Every player is assumed
   stationary. The pitch-control model's reaction-time term means a stationary
   assumption shrinks everyone's reachable area symmetrically, so it does not
   obviously bias either team, but it is an assumption.
3. **Identities.** A freeze-frame has no player ids, only ``teammate`` /
   ``actor`` / ``keeper`` booleans, all relative to the team of the event the
   frame belongs to. Teams are resolved through ``event.team``; per-player
   attribution is impossible, so the proxy is a per-turnover model only.
4. **Visible area.** A freeze-frame only contains the players inside the
   broadcast camera polygon, so a typical frame holds 10-16 of the 22 players.

Coordinates
-----------
StatsBomb events live on a 120 x 80 pitch with the origin at the TOP-left and
``y`` increasing downward, and the frame is re-oriented per event so that
``event.team`` always attacks ``+x``. This module converts to the package's
canonical frame -- 105 x 68 metres, origin bottom-left, ``y`` up -- and mirrors
any frame whose event belongs to the losing team, so that within one
counter-press window the team that WON the ball always attacks ``+x``. Without
that mirror a Pressure event by the counter-pressing team would silently arrive
180 degrees out.

The period-boundary trap
------------------------
StatsBomb ``timestamp`` is period-relative -- period 2's first event reads
``00:00:00.000`` -- while ``minute`` / ``second`` are match-absolute. Mixing
them yields a spurious gap of roughly 2758 s at every half boundary, with no
exception raised, which would poison both the five-second window and the
time-to-next-event feature. :func:`absolute_seconds` computes time from a period
offset plus the parsed timestamp and never touches ``minute``.
:func:`~tests.test_proxy` regression-tests it.

Two artefacts, and why
----------------------
``models/proxy_gbr.pkl.gz`` is refitted on every labelled row and is the model
you deploy; ``load_bundled_model()`` returns it. ``models/proxy_gbr_split.pkl.gz``
is fitted on the training matches only and is the model the model card's metrics
describe. Scoring the first one on the documented held-out matches is in-sample
and returns roughly r = 0.61 where the honest answer is about 0.39, with nothing
to say anything went wrong -- so both artefacts record ``fitted_on``,
``train_match_ids`` and ``held_out_match_ids`` in their metadata, and
:func:`score_held_out` raises rather than quietly returning the flattering
number. The mistake is an exception now, not a warning in a document.

Determinism
-----------
A refit is reproducible across processes **provided the row order of the
training set is fixed**, and it is not optional. ``HistGradientBoostingRegressor``
draws its early-stopping validation slice out of the design matrix by position,
so two frames with identical content in different row orders fit to different
trees. :func:`build_dataset` therefore returns a canonical row order that does
not depend on ``jobs`` or on which worker finished first. Measured on the 6,195
real labelled rows: three fresh processes on the canonical order gave
byte-identical predictions and held-out r = 0.369162 each time, while shuffling
nothing but the row order gave 0.369387, 0.374414 and 0.368491, and the frame the
v0.1.0 artefact came from gave 0.390659 -- a band of roughly 0.368 to 0.391 on
one machine with one dataset.

The **files** are reproducible too, which is a stricter claim and was not true
through v0.1.1. Two things in the write path read a clock. The gzip header
stamps ``time.time()`` into bytes 4:8 -- the shipped split artefact carried
mtime 1787773230 -- and :func:`train_from_cache` stamped a wall-clock
``trained_utc`` into the metadata *inside* the pickle, where it changed the
compressed length as well as the content. Both are gone: :func:`save_model`
writes :data:`ARTEFACT_GZIP_MTIME`, and no metadata field reads a clock.
Measured, two fresh processes each running :func:`train_from_cache` end to end
over the whole cache and writing into its own directory, both runs agreeing:

- ``proxy_gbr.pkl.gz``, 132,321 bytes, SHA-256
  ``94f6af0f22ffa5b52b208efb8756969b0a3b33a3dead91b87eed2bcb12d0a3ed``
- ``proxy_gbr_split.pkl.gz``, 124,333 bytes, SHA-256
  ``1364b8d9a831313b28c9f1c43ce61085f33ad7da0d88bdd47eedd82bdab5d22e``

The gzip was not hiding a difference underneath. Unpickled, the trees of both
new artefacts predict identically to the v0.1.1 artefacts they replace --
maximum absolute difference 0.0 over 5,000 probe rows, identical ``n_iter_``
(83 refit, 79 split) -- and the only metadata that moved is the removal of
``trained_utc``. ``tests/test_proxy.py`` asserts the header mtime and the
save-twice identity.

**The claim is "train twice, get the same file", not "the file is a fixed
point".** ``save_model(load_bundled_model(path), path)`` does *not* rewrite the
committed bytes: measured, the split artefact comes back 124,328 bytes against
the committed 124,333, and the refit 132,320 against 132,321. The cause is
pickle's own framing, not the model. Decompressed, the round-tripped payload is
336,411 bytes against the original's 336,401 and the first byte that differs is
a ``FRAME`` length -- ``pickle`` places frame boundaries by accumulated size, so
a re-pickled object graph, whose string objects are shared differently after a
load, frames differently. The two payloads describe the same estimator. Two
fresh processes each doing a load-and-resave agree with each other exactly; they
just do not agree with a training run.

Licence
-------
StatsBomb data is never committed. It is cached under the gitignored
``data/raw/`` and re-fetched on demand at one request per second. See the README
for the terms. The committed artefacts do contain StatsBomb's event-type
vocabulary, the three competition-season labels and the 97 match ids of the
training set; see ``models/MODEL_CARD.md`` §6.1, which surfaces that against the
absolutely-worded "never commit StatsBomb data in any form" decision.
"""

from __future__ import annotations

import gzip
import json
import pickle
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from .pitchcontrol import (
    PITCH_LENGTH,
    PITCH_WIDTH,
    PitchControlParams,
    PitchGrid,
    PlayerSet,
    make_grid,
    pitch_control_full,
)

__all__ = [
    "SB_PITCH_LENGTH",
    "SB_PITCH_WIDTH",
    "STATSBOMB_BASE_URL",
    "FEATURE_COLUMNS",
    "CATEGORICAL_FEATURES",
    "ProxyMetrics",
    "LabelParams",
    "FreezeFrame",
    "absolute_seconds",
    "statsbomb_url",
    "fetch_statsbomb",
    "matches_with_360",
    "download_match",
    "load_statsbomb_events",
    "load_freeze_frames",
    "find_turnovers",
    "sb_to_canonical",
    "freeze_frame_players",
    "label_turnovers",
    "build_features",
    "split_by_match",
    "train",
    "evaluate",
    "predict",
    "save_model",
    "load_bundled_model",
    "bundled_model_path",
    "ProxyModel",
    "LABEL_COLUMNS",
    "TURNOVER_COLUMNS",
    "EVENT_FIELDS",
    "CALIBRATION_VARIANTS",
    "CameraSample",
    "collect_camera_samples",
    "calibrate_against_metrica",
    "calibration_report",
    "compare_models",
    "score_predictions",
    "build_match_dataset",
    "build_dataset",
    "train_from_cache",
    "orient_to",
    "points_in_polygon",
    "download_training_set",
    "DEFAULT_TRAINING_COMPETITIONS",
    "MODEL_VARIANTS",
    "score_held_out",
    "held_out_match_ids",
    "training_match_ids",
    "phase_reference_means",
    "recovery_phase_caveat",
    "PROXY_PHASE_MEANS",
    "TRACKING_PHASE_MEANS",
]

# --------------------------------------------------------------------------- #
# StatsBomb constants
# --------------------------------------------------------------------------- #

#: StatsBomb pitch dimensions. Their coordinates are always on this pitch
#: regardless of the real stadium, so the conversion factor is a constant.
SB_PITCH_LENGTH: Final[float] = 120.0
SB_PITCH_WIDTH: Final[float] = 80.0

#: The old ``statsbomb/open-data`` path still resolves 200 after the repository
#: was renamed to ``hudl/open-data``; the rename is handled by a redirect we do
#: not depend on because raw.githubusercontent serves the old path directly.
STATSBOMB_BASE_URL: Final[str] = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

DEFAULT_CACHE_DIR: Final[Path] = Path("data/raw/statsbomb")

#: One request per second per host, as required. Measured mean file sizes are
#: 3.0 MB for an events file and 7.5 MB for a 360 file, so the transfer itself
#: usually dominates this floor anyway.
MIN_REQUEST_INTERVAL_S: Final[float] = 1.0

_USER_AGENT: Final[str] = "pressing/0.1 (research; contact via repository)"

_last_request_at: float = 0.0

#: Play patterns that mean the ball went dead. A possession that STARTS with one
#: of these did not start from an open-play turnover, so the preceding loss is
#: excluded. Driving the open-play test off the *next* possession's restart type
#: is more reliable than trying to classify the losing event.
DEAD_BALL_PATTERNS: Final[frozenset[str]] = frozenset(
    {"From Corner", "From Free Kick", "From Goal Kick", "From Keeper", "From Kick Off",
     "From Throw In"}
)

#: Event types that never carry a meaningful pitch location for our purposes.
_NON_LOCATED_TYPES: Final[frozenset[str]] = frozenset(
    {"Starting XI", "Half Start", "Half End", "Substitution", "Tactical Shift",
     "Player On", "Player Off", "Injury Stoppage", "Referee Ball-Drop"}
)


# --------------------------------------------------------------------------- #
# Time
# --------------------------------------------------------------------------- #


def _parse_timestamp(text: str) -> float:
    """Parse a StatsBomb ``HH:MM:SS.mmm`` timestamp into seconds.

    Args:
        text: Period-relative timestamp string.

    Returns:
        Seconds since the start of that period.

    Raises:
        ValueError: If the string is not three colon-separated fields.
    """
    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError(f"malformed StatsBomb timestamp: {text!r}")
    return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])


def absolute_seconds(events: Sequence[dict[str, Any]]) -> np.ndarray:
    """Match-absolute event time in seconds, immune to the period reset.

    StatsBomb ``timestamp`` restarts at ``00:00:00.000`` in every period while
    ``minute`` keeps counting, so any code that mixes the two sees a jump of
    roughly 2758 s at half time and computes a five-second window spanning the
    interval. This function accumulates a per-period offset from the *observed*
    length of each preceding period and adds the parsed timestamp, so it never
    reads ``minute`` at all.

    Periods are assumed to be presented in order, which the open-data files are.

    Args:
        events: Raw StatsBomb event dicts, in file order.

    Returns:
        ``(n,)`` float array of seconds since kick-off. Times are strictly
        comparable within a period and monotonic across periods; the gap across
        a period boundary is the length of the preceding period, not the real
        half-time interval, which is exactly what a five-second window wants.
    """
    if not events:
        return np.zeros(0, dtype=float)

    periods = np.array([int(e["period"]) for e in events], dtype=np.int64)
    within = np.array([_parse_timestamp(str(e["timestamp"])) for e in events], dtype=float)

    offsets: dict[int, float] = {}
    running = 0.0
    for period in sorted(set(periods.tolist())):
        offsets[period] = running
        running += float(within[periods == period].max())

    return within + np.array([offsets[p] for p in periods.tolist()], dtype=float)


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #


def statsbomb_url(kind: str, name: str) -> str:
    """Build a raw.githubusercontent URL for one StatsBomb open-data file.

    Args:
        kind: ``"events"``, ``"three-sixty"``, ``"lineups"``, ``"matches"`` or
            ``""`` for a top-level file such as ``competitions.json``.
        name: File name including the ``.json`` suffix, e.g. ``"3825739.json"``
            or ``"11/90.json"`` for a matches file.

    Returns:
        The full URL.
    """
    return f"{STATSBOMB_BASE_URL}/{kind}/{name}" if kind else f"{STATSBOMB_BASE_URL}/{name}"


def _polite_sleep() -> None:
    """Block until at least :data:`MIN_REQUEST_INTERVAL_S` since the last request."""
    global _last_request_at
    wait = MIN_REQUEST_INTERVAL_S - (time.monotonic() - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.monotonic()


def fetch_statsbomb(
    kind: str,
    name: str,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    *,
    retries: int = 5,
    allow_download: bool = True,
) -> Path:
    """Return a cached StatsBomb file, downloading it once if needed.

    A file that already exists in the cache is never re-fetched. Downloads are
    paced to one request per second and retried with exponential backoff. The
    write is atomic -- a partial download cannot poison the cache.

    Args:
        kind: Directory under ``data/``; see :func:`statsbomb_url`.
        name: File name, e.g. ``"3825739.json"``.
        cache_dir: Local root mirroring the ``data/`` layout.
        retries: Attempts before giving up.
        allow_download: When False, raise instead of touching the network.

    Returns:
        Path to the cached file.

    Raises:
        FileNotFoundError: If the file is absent and ``allow_download`` is False.
        OSError: If every attempt fails.
    """
    dest = Path(cache_dir) / kind / name if kind else Path(cache_dir) / name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    if not allow_download:
        raise FileNotFoundError(
            f"{dest} is not cached and downloads are disabled. Run the fetch step first."
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = statsbomb_url(kind, name)
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})

    last: Exception | None = None
    for attempt in range(retries):
        try:
            _polite_sleep()
            with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
                payload = response.read()
            if not payload:
                raise OSError(f"empty response for {url}")
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(payload)
            tmp.replace(dest)
            return dest
        except (urllib.error.URLError, OSError, TimeoutError) as exc:  # pragma: no cover - net
            last = exc
            if attempt == retries - 1:
                break
            time.sleep(2.0**attempt)
    raise OSError(f"failed to fetch {url} after {retries} attempts: {last}")


def matches_with_360(
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    competitions: Iterable[tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """List every open-data match that has a 360 freeze-frame file.

    Driven off the cached ``matches/{cid}/{sid}.json`` manifests and the cached
    repository tree listing, never off a directory scan: 274 orphan event files
    belong to no season and would corrupt any competition aggregate.

    Args:
        cache_dir: StatsBomb cache root. Must already hold ``tree.json`` and the
            ``matches/`` manifests.
        competitions: Optional ``(competition_id, season_id)`` allow-list.

    Returns:
        One row per match with ``match_id``, ``competition_id``, ``season_id``,
        ``competition``, ``season``, ``home_team``, ``away_team``, ``match_date``
        and ``bytes_360``, sorted by competition then match id.

    Raises:
        FileNotFoundError: If ``tree.json`` has not been cached.
    """
    root = Path(cache_dir)
    tree_path = root / "tree.json"
    if not tree_path.exists():
        raise FileNotFoundError(
            f"{tree_path} missing; fetch the repository tree listing before enumerating matches"
        )
    tree = json.loads(tree_path.read_text())
    sizes: dict[int, int] = {}
    for blob in tree["tree"]:
        path = str(blob["path"])
        if path.startswith("data/three-sixty/") and path.endswith(".json"):
            sizes[int(path.rsplit("/", 1)[1][:-5])] = int(blob.get("size", 0))

    wanted = set(competitions) if competitions is not None else None
    rows: list[dict[str, Any]] = []
    for manifest in sorted((root / "matches").rglob("*.json")):
        cid, sid = int(manifest.parent.name), int(manifest.stem)
        if wanted is not None and (cid, sid) not in wanted:
            continue
        for match in json.loads(manifest.read_text()):
            mid = int(match["match_id"])
            if mid not in sizes:
                continue
            rows.append(
                {
                    "match_id": mid,
                    "competition_id": cid,
                    "season_id": sid,
                    "competition": match["competition"]["competition_name"],
                    "season": match["season"]["season_name"],
                    "home_team": match["home_team"]["home_team_name"],
                    "away_team": match["away_team"]["away_team_name"],
                    "match_date": match.get("match_date"),
                    "bytes_360": sizes[mid],
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["competition_id", "season_id", "match_id"]).reset_index(drop=True)


def download_match(
    match_id: int,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    *,
    allow_download: bool = True,
) -> tuple[Path, Path]:
    """Cache the events and 360 files for one match.

    Args:
        match_id: StatsBomb match id.
        cache_dir: Cache root.
        allow_download: When False, only accept files already cached.

    Returns:
        ``(events_path, three_sixty_path)``.
    """
    name = f"{match_id}.json"
    events = fetch_statsbomb("events", name, cache_dir, allow_download=allow_download)
    frames = fetch_statsbomb("three-sixty", name, cache_dir, allow_download=allow_download)
    return events, frames


def download_training_set(
    match_ids: Sequence[int],
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    *,
    verbose: bool = True,
) -> list[int]:
    """Cache every events + 360 pair in ``match_ids``, skipping failures.

    A source that fails is logged and skipped rather than aborting the pull, so
    an interrupted or partly unavailable competition still yields a usable set.

    Args:
        match_ids: Match ids to cache.
        cache_dir: Cache root.
        verbose: Print progress to stdout.

    Returns:
        The match ids that ended up fully cached.
    """
    ok: list[int] = []
    for i, mid in enumerate(match_ids, start=1):
        try:
            download_match(mid, cache_dir)
            ok.append(mid)
            if verbose:
                print(f"[{i}/{len(match_ids)}] cached {mid}", flush=True)
        except Exception as exc:  # noqa: BLE001 - a bad match must not stop the pull
            if verbose:
                print(f"[{i}/{len(match_ids)}] FAILED {mid}: {exc}", flush=True)
    return ok


# --------------------------------------------------------------------------- #
# Event parsing
# --------------------------------------------------------------------------- #

#: The columns :func:`load_statsbomb_events` produces. Deliberately a flat,
#: fixed set: the raw event dicts nest arbitrarily deep and half the keys are
#: absent on most rows, which makes a naive ``json_normalize`` both huge and
#: unstable between competitions.
EVENT_FIELDS: Final[tuple[str, ...]] = (
    "match_id",
    "id",
    "index",
    "period",
    "t",
    "type",
    "team",
    "team_id",
    "player",
    "possession",
    "possession_team",
    "possession_team_id",
    "play_pattern",
    "duration",
    "under_pressure",
    "counterpress",
    "out",
    "x",
    "y",
    "outcome",
    "has_360",
)


def _outcome_of(event: dict[str, Any]) -> str | None:
    """First outcome name found in a StatsBomb event's type sub-object.

    Args:
        event: Raw event dict.

    Returns:
        The outcome name, or None when the event type carries no outcome.
    """
    for key in ("pass", "duel", "interception", "dribble", "shot", "ball_receipt",
                "50_50", "substitution", "goalkeeper"):
        sub = event.get(key)
        if isinstance(sub, dict):
            outcome = sub.get("outcome")
            if isinstance(outcome, dict) and "name" in outcome:
                return str(outcome["name"])
    return None


def load_statsbomb_events(
    path: Path | str,
    match_id: int | None = None,
    frame_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Flatten one StatsBomb events file into the fields the proxy needs.

    Args:
        path: Path to ``data/events/{match_id}.json``.
        match_id: Match id to stamp on every row. Defaults to the file stem.
        frame_ids: Optional set of event uuids that have a freeze-frame, used to
            fill the ``has_360`` column.

    Returns:
        A frame with :data:`EVENT_FIELDS`, sorted by ``index``, with ``t`` the
        match-absolute time from :func:`absolute_seconds`.
    """
    path = Path(path)
    raw = json.loads(path.read_text())
    raw.sort(key=lambda e: int(e["index"]))
    mid = int(path.stem) if match_id is None else int(match_id)
    times = absolute_seconds(raw)

    rows: list[dict[str, Any]] = []
    for event, t in zip(raw, times, strict=True):
        location = event.get("location")
        rows.append(
            {
                "match_id": mid,
                "id": str(event["id"]),
                "index": int(event["index"]),
                "period": int(event["period"]),
                "t": float(t),
                "type": str(event["type"]["name"]),
                "team": str(event["team"]["name"]),
                "team_id": int(event["team"]["id"]),
                "player": (event.get("player") or {}).get("name"),
                "possession": int(event["possession"]),
                "possession_team": str(event["possession_team"]["name"]),
                "possession_team_id": int(event["possession_team"]["id"]),
                "play_pattern": str(event["play_pattern"]["name"]),
                "duration": float(event["duration"]) if event.get("duration") is not None
                else np.nan,
                "under_pressure": bool(event.get("under_pressure", False)),
                "counterpress": bool(event.get("counterpress", False)),
                "out": bool(event.get("out", False)),
                "x": float(location[0]) if location else np.nan,
                "y": float(location[1]) if location else np.nan,
                "outcome": _outcome_of(event),
                "has_360": bool(frame_ids is not None and str(event["id"]) in frame_ids),
            }
        )
    return pd.DataFrame(rows, columns=list(EVENT_FIELDS))


# --------------------------------------------------------------------------- #
# 360 freeze-frames
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FreezeFrame:
    """One StatsBomb 360 freeze-frame.

    Everything here is relative to the team of the event the frame belongs to:
    ``teammate`` is True for that team's players, and the coordinates are
    oriented so that team attacks ``+x`` on the 120 x 80 pitch. There are no
    player identities and no velocities.

    Attributes:
        event_uuid: Id of the event this frame accompanies.
        positions: ``(n, 2)`` player positions on the 120 x 80 StatsBomb pitch.
        teammate: ``(n,)`` True for a player on the event's own team.
        actor: ``(n,)`` True for the player performing the event.
        keeper: ``(n,)`` True for a goalkeeper.
        visible_area: ``(m, 2)`` broadcast polygon vertices, or an empty array.
    """

    event_uuid: str
    positions: np.ndarray
    teammate: np.ndarray
    actor: np.ndarray
    keeper: np.ndarray
    visible_area: np.ndarray

    def __len__(self) -> int:
        """Number of players captured in the frame."""
        return int(self.positions.shape[0])


def load_freeze_frames(path: Path | str) -> dict[str, FreezeFrame]:
    """Read a StatsBomb 360 file into freeze-frames keyed by event uuid.

    Args:
        path: Path to ``data/three-sixty/{match_id}.json``.

    Returns:
        Mapping of event uuid to :class:`FreezeFrame`. Frames with no players
        are dropped.
    """
    payload = json.loads(Path(path).read_text())
    out: dict[str, FreezeFrame] = {}
    for record in payload:
        players = record.get("freeze_frame") or []
        if not players:
            continue
        area = record.get("visible_area") or []
        out[str(record["event_uuid"])] = FreezeFrame(
            event_uuid=str(record["event_uuid"]),
            positions=np.array([p["location"] for p in players], dtype=float).reshape(-1, 2),
            teammate=np.array([bool(p.get("teammate")) for p in players], dtype=bool),
            actor=np.array([bool(p.get("actor")) for p in players], dtype=bool),
            keeper=np.array([bool(p.get("keeper")) for p in players], dtype=bool),
            visible_area=np.asarray(area, dtype=float).reshape(-1, 2),
        )
    return out


# --------------------------------------------------------------------------- #
# Coordinates
# --------------------------------------------------------------------------- #


def sb_to_canonical(xy: np.ndarray) -> np.ndarray:
    """Convert StatsBomb 120 x 80 coordinates to the canonical metre frame.

    StatsBomb puts the origin at the top-left with ``y`` increasing downward.
    The canonical frame is 105 x 68 m with the origin bottom-left and ``y`` up,
    so the conversion scales both axes and flips ``y``.

    Args:
        xy: ``(..., 2)`` StatsBomb coordinates.

    Returns:
        ``(..., 2)`` canonical coordinates in metres.
    """
    arr = np.asarray(xy, dtype=float)
    out = np.empty_like(arr)
    out[..., 0] = arr[..., 0] * (PITCH_LENGTH / SB_PITCH_LENGTH)
    out[..., 1] = (SB_PITCH_WIDTH - arr[..., 1]) * (PITCH_WIDTH / SB_PITCH_WIDTH)
    return out


def _mirror_sb(xy: np.ndarray) -> np.ndarray:
    """Rotate StatsBomb coordinates 180 degrees, i.e. swap which end is attacked.

    Args:
        xy: ``(..., 2)`` StatsBomb coordinates.

    Returns:
        ``(..., 2)`` mirrored coordinates on the same pitch.
    """
    arr = np.asarray(xy, dtype=float)
    out = np.empty_like(arr)
    out[..., 0] = SB_PITCH_LENGTH - arr[..., 0]
    out[..., 1] = SB_PITCH_WIDTH - arr[..., 1]
    return out


def orient_to(xy: np.ndarray, event_team: str, target_team: str) -> np.ndarray:
    """Re-orient StatsBomb coordinates so ``target_team`` attacks ``+x``.

    StatsBomb flips the pitch per event so that the event's own team always
    attacks ``+x``. Inside a counter-press window events belong to both teams,
    so every frame has to be brought into one common orientation before the two
    can be compared. Getting this wrong is silent: the surface still renders and
    the share still lands in ``[0, 1]``, it is just 180 degrees out.

    Args:
        xy: ``(..., 2)`` StatsBomb coordinates as stored on the event.
        event_team: Name of the team the event belongs to.
        target_team: Team that should attack ``+x`` in the result.

    Returns:
        ``(..., 2)`` StatsBomb coordinates in the target orientation.
    """
    arr = np.asarray(xy, dtype=float)
    return arr if event_team == target_team else _mirror_sb(arr)


def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Even-odd ray-cast point-in-polygon test.

    Written out rather than pulled from matplotlib so that the label pipeline
    does not import a plotting stack, and so the boundary convention is pinned
    by this package's own tests.

    Args:
        points: ``(n, 2)`` query points.
        polygon: ``(m, 2)`` polygon vertices. The ring is closed implicitly; a
            repeated final vertex is harmless.

    Returns:
        ``(n,)`` boolean array. An empty or degenerate polygon returns all True,
        which is the right default: no polygon means no truncation.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    poly = np.asarray(polygon, dtype=float).reshape(-1, 2)
    if poly.shape[0] < 3:
        return np.ones(pts.shape[0], dtype=bool)

    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(pts.shape[0], dtype=bool)
    x1, y1 = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    for ax, ay, bx, by in zip(x1, y1, x2, y2, strict=True):
        if ay == by:
            continue
        straddles = (ay > y) != (by > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = ax + (y - ay) * (bx - ax) / (by - ay)
        inside ^= straddles & (x < x_cross)
    return inside


# --------------------------------------------------------------------------- #
# Turnovers
# --------------------------------------------------------------------------- #

TURNOVER_COLUMNS: Final[tuple[str, ...]] = (
    "match_id",
    "turnover_id",
    "possession",
    "period",
    "t",
    "losing_team",
    "losing_team_id",
    "winning_team",
    "winning_team_id",
    "loss_index",
    "start_index",
    "sb_x",
    "sb_y",
    "x",
    "y",
    "phase",
    "next_play_pattern",
)


def _phase_of(x_metres: float) -> str | None:
    """Phase label for a turnover already expressed in the losing team's frame.

    Args:
        x_metres: Metres from the losing team's own goal line, 0 to 105.

    Returns:
        ``"build_up"``, ``"middle"``, ``"final_third"``, or None if not finite.
    """
    from .phases import phase_for_x

    return phase_for_x(x_metres)


def find_turnovers(events: pd.DataFrame) -> pd.DataFrame:
    """Locate open-play possession losses from ``possession_team`` transitions.

    Detection is by possession-team change, not by a whitelist of turnover event
    types. In a real match the last on-ball event of a possession is
    ``Ball Receipt*`` roughly a third of the time, so a type whitelist
    misclassifies a third of the losses.

    A transition counts as an open-play turnover when the new possession belongs
    to the other team, lies in the same period, and does **not** start from a
    dead ball -- that is, its ``play_pattern`` is not one of
    :data:`DEAD_BALL_PATTERNS`. Driving the open-play test off how the next
    possession restarted is more reliable than trying to read intent off the
    losing event.

    ``t0`` is the time of the first located event of the new possession. The
    turnover location is that same event, re-oriented into the LOSING team's
    attacking frame so that ``x = 0`` is the losing team's own goal line, which
    is what the phase split and the report both mean by ``x``.

    Args:
        events: Frame from :func:`load_statsbomb_events`.

    Returns:
        One row per open-play turnover with :data:`TURNOVER_COLUMNS`.
    """
    if events.empty:
        return pd.DataFrame(columns=list(TURNOVER_COLUMNS))

    ev = events.sort_values("index").reset_index(drop=True)
    rows: list[dict[str, Any]] = []

    groups = list(ev.groupby("possession", sort=True))
    by_possession = {int(k): g for k, g in groups}
    order = sorted(by_possession)

    for prev_id, this_id in zip(order, order[1:], strict=False):
        prev, this = by_possession[prev_id], by_possession[this_id]
        losing_team = str(prev["possession_team"].iloc[0])
        winning_team = str(this["possession_team"].iloc[0])
        if losing_team == winning_team:
            continue
        if int(prev["period"].iloc[-1]) != int(this["period"].iloc[0]):
            continue
        pattern = str(this["play_pattern"].iloc[0])
        if pattern in DEAD_BALL_PATTERNS:
            continue

        located = this[this["x"].notna() & ~this["type"].isin(_NON_LOCATED_TYPES)]
        if located.empty:
            continue
        first = located.iloc[0]

        sb_xy = orient_to(
            np.array([float(first["x"]), float(first["y"])]),
            str(first["team"]),
            losing_team,
        )
        canonical = sb_to_canonical(sb_xy)
        rows.append(
            {
                "match_id": int(first["match_id"]),
                "turnover_id": f"{int(first['match_id'])}:{this_id}",
                "possession": this_id,
                "period": int(first["period"]),
                "t": float(first["t"]),
                "losing_team": losing_team,
                "losing_team_id": int(prev["possession_team_id"].iloc[0]),
                "winning_team": winning_team,
                "winning_team_id": int(this["possession_team_id"].iloc[0]),
                "loss_index": int(prev["index"].iloc[-1]),
                "start_index": int(first["index"]),
                "sb_x": float(sb_xy[0]),
                "sb_y": float(sb_xy[1]),
                "x": float(canonical[0]),
                "y": float(canonical[1]),
                "phase": _phase_of(float(canonical[0])),
                "next_play_pattern": pattern,
            }
        )
    return pd.DataFrame(rows, columns=list(TURNOVER_COLUMNS))


# --------------------------------------------------------------------------- #
# The 360 label
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LabelParams:
    """Knobs of the freeze-frame label.

    Attributes:
        window_s: Length of the counter-press window in seconds, closed at both
            ends. Matches :data:`pressing.counterpress.RECOVERY_WINDOW_S`.
        min_frames: Fewest freeze-frames a window needs to be labelled at all.
            Two is the minimum that says anything about a five-second window --
            one frame describes only the geometry at the instant of the loss and
            contains no information about the press. Windows below this are
            emitted with a null label and a ``skip_reason``.
        min_players_per_team: Fewest players a team must have in a frame for
            that frame to be usable. Pitch control with one visible defender is
            not a defence.
        velocity: Assumed player velocity. ``"zero"`` is the only honest option
            for a freeze-frame; the field exists so the assumption is named in
            the model card rather than buried in the code.
        weighting: Cell weighting for the control share. ``"area"`` mirrors the
            tracking metric's headline ``recovery_area``. The EPV weighting is
            deliberately not offered here: it needs an oriented EPV grid, and
            the tracking metric already found it less discriminating.
    """

    window_s: float = 5.0
    min_frames: int = 2
    min_players_per_team: int = 2
    velocity: str = "zero"
    weighting: str = "area"

    def __post_init__(self) -> None:
        """Validate the knobs.

        Raises:
            ValueError: On an unsupported velocity model or weighting, a
                non-positive window, or ``min_frames`` below one.
        """
        if self.velocity != "zero":
            raise ValueError(
                f"velocity must be 'zero'; a freeze-frame carries no velocity "
                f"information, got {self.velocity!r}"
            )
        if self.weighting != "area":
            raise ValueError(f"weighting must be 'area', got {self.weighting!r}")
        if self.window_s <= 0:
            raise ValueError(f"window_s must be positive, got {self.window_s}")
        if self.min_frames < 1:
            raise ValueError(f"min_frames must be at least 1, got {self.min_frames}")


LABEL_COLUMNS: Final[tuple[str, ...]] = (
    "turnover_id",
    "recovery_360",
    "recovery_360_t0",
    "recovery_360_end",
    "n_frames",
    "frame_span_s",
    "mean_players_per_frame",
    "mean_visible_fraction",
    "labelled",
    "skip_reason",
)


def freeze_frame_players(
    frame: FreezeFrame,
    event_team: str,
    winning_team: str,
    losing_team: str,
) -> tuple[PlayerSet, PlayerSet]:
    """Split one freeze-frame into the two teams, oriented to the winner.

    The ``teammate`` flag is relative to ``event_team``, and the coordinates are
    relative to ``event_team`` attacking ``+x``. Both are resolved here so the
    caller always gets ``(winner, presser)`` in the canonical metre frame with
    the winner attacking ``+x``.

    Velocities are zero: a freeze-frame has none. Goalkeepers come from the
    ``keeper`` flag.

    Args:
        frame: The freeze-frame.
        event_team: Team name of the event this frame belongs to.
        winning_team: Team that won the ball at the turnover.
        losing_team: Team that lost it, i.e. the counter-pressing team.

    Returns:
        ``(winner_players, presser_players)``.

    Raises:
        ValueError: If ``event_team`` is neither of the two named teams.
    """
    if event_team not in (winning_team, losing_team):
        raise ValueError(
            f"event team {event_team!r} is neither {winning_team!r} nor {losing_team!r}"
        )
    # `teammate` is True for `event_team`; map that onto the winner.
    is_winner = frame.teammate if event_team == winning_team else ~frame.teammate
    positions = sb_to_canonical(orient_to(frame.positions, event_team, winning_team))
    zero = np.zeros_like(positions)

    winner = PlayerSet.from_arrays(
        positions=positions[is_winner],
        velocities=zero[is_winner],
        is_gk=frame.keeper[is_winner],
    )
    presser = PlayerSet.from_arrays(
        positions=positions[~is_winner],
        velocities=zero[~is_winner],
        is_gk=frame.keeper[~is_winner],
    )
    return winner, presser


def _visible_mask(frame: FreezeFrame, event_team: str, winning_team: str,
                  grid: PitchGrid) -> np.ndarray:
    """Boolean grid mask of the cells inside the broadcast polygon.

    Args:
        frame: The freeze-frame carrying the polygon.
        event_team: Team the event belongs to.
        winning_team: Team the frame is being oriented to.
        grid: Grid the mask is built for.

    Returns:
        ``(n_y, n_x)`` boolean array; all True when no polygon is present.
    """
    if frame.visible_area.shape[0] < 3:
        return np.ones(grid.shape, dtype=bool)
    poly = sb_to_canonical(orient_to(frame.visible_area, event_team, winning_team))
    inside = points_in_polygon(grid.targets, poly)
    return inside.reshape(grid.shape)


def _share_from_surfaces(
    winner_surface: np.ndarray,
    presser_surface: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Counter-pressing team's share of control, optionally over a sub-region.

    Mirrors ``counterpress._evaluate_frame``'s area share exactly: the presser's
    integrated control over the two teams' summed control, so cells nobody
    controls do not count for either side.

    Args:
        winner_surface: Control of the team that won the ball.
        presser_surface: Control of the counter-pressing team.
        mask: Optional boolean mask selecting cells to integrate over.

    Returns:
        The share in ``[0, 1]``, or NaN when the region is empty.
    """
    won = winner_surface.astype(np.float64)
    pressed = presser_surface.astype(np.float64)
    if mask is not None:
        won, pressed = won[mask], pressed[mask]
    total = float((won + pressed).sum())
    return float(pressed.sum()) / total if total > 0 else float("nan")


def label_turnovers(
    events: pd.DataFrame,
    frames: dict[str, FreezeFrame],
    turnovers: pd.DataFrame | None = None,
    params: LabelParams | None = None,
    grid: PitchGrid | None = None,
    pc_params: PitchControlParams | None = None,
) -> pd.DataFrame:
    """Build the freeze-frame counter-press label for every turnover.

    For each open-play turnover this takes the freeze-frames of the events
    falling inside ``[t0, t0 + window_s]``, computes a pitch-control surface at
    each with every player stationary, and averages the losing team's share of
    control across them. That is the same quantity as the tracking metric's
    ``recovery_area``, evaluated on whatever instants the event stream happens
    to provide instead of on six evenly spaced ones.

    Decisions this function makes, all of them approximations:

    * **Velocity** is zero everywhere. A freeze-frame has none, and inventing
      one from the surrounding events would be a second model layered under the
      first.
    * **Teams** come from ``teammate`` resolved through ``event.team``. Frames
      belonging to the losing team are mirrored so the winner always attacks
      ``+x``.
    * **The ball** is placed at the event's own location. That is exactly where
      the ball is at a tagged on-ball event, and it is the only ball information
      a 360 file carries.
    * **Visible area** is handled two ways and both are reported.
      ``recovery_360`` integrates over the whole pitch and therefore lets the
      pitch-control model extrapolate into parts of the pitch where no player
      was captured. ``recovery_360_visible`` integrates only over the broadcast
      polygon, where the player set is at least complete. Which one tracks the
      true value better is measured in :func:`calibrate_against_metrica`, not
      guessed at here.
    * **One frame** is not enough. A single-frame window is emitted unlabelled
      with ``skip_reason='too_few_frames'`` rather than being scored off the
      loss instant alone, which would measure the geometry of the turnover and
      nothing about the press.

    Args:
        events: Frame from :func:`load_statsbomb_events`.
        frames: Mapping from :func:`load_freeze_frames`.
        turnovers: Precomputed turnovers, or None to call :func:`find_turnovers`.
        params: Label knobs. Defaults to :class:`LabelParams`.
        grid: Pitch-control grid. Defaults to the package's 50 x 32 demo grid,
            the same one the tracking metric uses.
        pc_params: Pitch-control constants. Defaults to the package defaults.

    Returns:
        ``turnovers`` with :data:`LABEL_COLUMNS` and ``recovery_360_visible``
        joined on, one row per turnover, unlabelled rows kept.
    """
    params = params or LabelParams()
    grid = grid or make_grid(n_x=50)
    pc_params = pc_params or PitchControlParams()
    turns = find_turnovers(events) if turnovers is None else turnovers

    if turns.empty:
        empty = pd.DataFrame(columns=list(TURNOVER_COLUMNS) + list(LABEL_COLUMNS)
                             + ["recovery_360_visible"])
        return empty

    ev = events.sort_values("index").reset_index(drop=True)
    times = ev["t"].to_numpy(dtype=float)
    periods = ev["period"].to_numpy(dtype=np.int64)
    ids = ev["id"].to_numpy()
    teams = ev["team"].to_numpy()
    xs = ev["x"].to_numpy(dtype=float)
    ys = ev["y"].to_numpy(dtype=float)

    out: list[dict[str, Any]] = []
    for turn in turns.itertuples(index=False):
        lo = float(turn.t)
        hi = lo + params.window_s
        sel = np.flatnonzero(
            (periods == int(turn.period)) & (times >= lo) & (times <= hi) & np.isfinite(xs)
        )

        shares: list[float] = []
        shares_visible: list[float] = []
        offsets: list[float] = []
        counts: list[int] = []
        visible: list[float] = []

        for i in sel:
            frame = frames.get(str(ids[i]))
            if frame is None:
                continue
            event_team = str(teams[i])
            if event_team not in (turn.winning_team, turn.losing_team):
                continue
            winner, presser = freeze_frame_players(
                frame, event_team, str(turn.winning_team), str(turn.losing_team)
            )
            if (len(winner) < params.min_players_per_team
                    or len(presser) < params.min_players_per_team):
                continue
            ball = sb_to_canonical(
                orient_to(np.array([xs[i], ys[i]]), event_team, str(turn.winning_team))
            )
            result = pitch_control_full(
                winner, presser, ball_pos=ball, grid=grid, params=pc_params, offsides=False
            )
            mask = _visible_mask(frame, event_team, str(turn.winning_team), grid)
            shares.append(_share_from_surfaces(result.attacking, result.defending))
            shares_visible.append(
                _share_from_surfaces(result.attacking, result.defending, mask)
            )
            offsets.append(float(times[i]) - lo)
            counts.append(len(frame))
            visible.append(float(mask.mean()))

        row: dict[str, Any] = {"turnover_id": turn.turnover_id}
        if len(shares) < params.min_frames:
            row.update(
                {
                    "recovery_360": np.nan,
                    "recovery_360_visible": np.nan,
                    "recovery_360_t0": np.nan,
                    "recovery_360_end": np.nan,
                    "n_frames": len(shares),
                    "frame_span_s": np.nan,
                    "mean_players_per_frame": np.nan,
                    "mean_visible_fraction": np.nan,
                    "labelled": False,
                    "skip_reason": "no_360" if not shares else "too_few_frames",
                }
            )
        else:
            row.update(
                {
                    "recovery_360": float(np.mean(shares)),
                    "recovery_360_visible": float(np.mean(shares_visible)),
                    "recovery_360_t0": float(shares[0]),
                    "recovery_360_end": float(shares[-1]),
                    "n_frames": len(shares),
                    "frame_span_s": float(offsets[-1] - offsets[0]),
                    "mean_players_per_frame": float(np.mean(counts)),
                    "mean_visible_fraction": float(np.mean(visible)),
                    "labelled": True,
                    "skip_reason": None,
                }
            )
        out.append(row)

    labels = pd.DataFrame(out)
    return turns.merge(labels, on="turnover_id", how="left")


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #

#: Categorical features. Encoded to integer codes against a vocabulary frozen at
#: training time and stored in the artefact, so a category never seen in
#: training maps to -1 rather than silently shifting every other code.
CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    "loss_event_type",
    "next_1_type",
    "next_2_type",
    "next_3_type",
    "next_play_pattern",
)

#: Numeric features, in the order the model sees them.
NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    "x",
    "y",
    "dist_own_goal",
    "dist_opp_goal",
    "dist_from_centre",
    "time_to_next_event",
    "n_events_5s",
    "n_loser_events_5s",
    "n_winner_events_5s",
    "under_pressure_count",
    "counterpress_count",
    "pressure_event_count",
    "loss_under_pressure",
    "loss_counterpress",
    "next_1_by_loser",
    "next_2_by_loser",
    "next_3_by_loser",
    "ball_forward_5s",
    "ball_distance_5s",
    "losing_possession_events",
    "losing_possession_duration",
    "loss_duration",
    "regained_within_5s",
    "period",
)

#: Every feature the model consumes, categoricals last.
FEATURE_COLUMNS: Final[tuple[str, ...]] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

#: Columns carried alongside the features for splitting, joining and reporting,
#: never fed to the model.
FEATURE_META_COLUMNS: Final[tuple[str, ...]] = (
    "match_id",
    "turnover_id",
    "t",
    "losing_team",
    "winning_team",
    "phase",
)


def build_features(
    events: pd.DataFrame,
    turnovers: pd.DataFrame | None = None,
    window_s: float = 5.0,
) -> pd.DataFrame:
    """Derive the event-only feature vector for every open-play turnover.

    Everything here comes from the event stream. **No feature touches a
    freeze-frame**, which is the whole point: the trained model has to run on a
    match that has events and no tracking and no 360.

    The spec names four feature groups and all four are present:

    ``turnover location``
        ``x`` / ``y`` in canonical metres, in the LOSING team's attacking frame
        so ``x = 0`` is its own goal, plus distance to each goal and distance
        from the centre line.
    ``time-to-next-event``
        Seconds from ``t0`` to the next event, computed from
        :func:`absolute_seconds` so a window at a half boundary is not handed a
        2758 s gap.
    ``the next 3 actions``
        ``type.name`` of the next three events by ``index``, plus whether each
        belongs to the counter-pressing team -- a pass completed by the team
        that just won the ball reads very differently from one by the team that
        just lost it.
    ``pressure flags``
        Counts of ``under_pressure`` and ``counterpress`` inside the window, the
        number of explicit ``Pressure`` events, and both flags on the losing
        event itself.

    Beyond the spec, and each earning its place: how far the ball moved and how
    far it moved forward for the winner during the window; how long the losing
    possession was in events and in seconds; the ``play_pattern`` the new
    possession started under; whether the ball came back inside five seconds
    (the plain, checkable event-stream companion to the continuous metric); and
    the period.

    Args:
        events: Frame from :func:`load_statsbomb_events`.
        turnovers: Precomputed turnovers, or None to call :func:`find_turnovers`.
        window_s: Window length in seconds, closed at both ends.

    Returns:
        One row per turnover with :data:`FEATURE_META_COLUMNS` and
        :data:`FEATURE_COLUMNS`. ``match_id`` is retained so splits can be made
        by match.
    """
    turns = find_turnovers(events) if turnovers is None else turnovers
    columns = list(FEATURE_META_COLUMNS) + list(FEATURE_COLUMNS)
    if turns.empty:
        return pd.DataFrame(columns=columns)

    ev = events.sort_values("index").reset_index(drop=True)
    idx = ev["index"].to_numpy(dtype=np.int64)
    times = ev["t"].to_numpy(dtype=float)
    periods = ev["period"].to_numpy(dtype=np.int64)
    types = ev["type"].to_numpy()
    teams = ev["team"].to_numpy()
    poss = ev["possession"].to_numpy(dtype=np.int64)
    poss_team = ev["possession_team"].to_numpy()
    xs = ev["x"].to_numpy(dtype=float)
    ys = ev["y"].to_numpy(dtype=float)
    pressed = ev["under_pressure"].to_numpy(dtype=bool)
    cpress = ev["counterpress"].to_numpy(dtype=bool)
    durations = ev["duration"].to_numpy(dtype=float)
    by_index = {int(v): i for i, v in enumerate(idx)}

    rows: list[dict[str, Any]] = []
    for turn in turns.itertuples(index=False):
        lo = float(turn.t)
        hi = lo + window_s
        loser, winner = str(turn.losing_team), str(turn.winning_team)

        in_window = np.flatnonzero(
            (periods == int(turn.period)) & (times >= lo) & (times <= hi)
        )
        # The next three actions strictly after the turnover event, kept inside
        # the same period so a half boundary cannot splice in the second half.
        after = np.flatnonzero((periods == int(turn.period)) & (idx > int(turn.start_index)))
        nxt = after[:3]

        loss_row = int(by_index[int(turn.loss_index)])

        located = in_window[np.isfinite(xs[in_window])]
        if located.size >= 2:
            first_xy = orient_to(
                np.array([xs[located[0]], ys[located[0]]]), str(teams[located[0]]), winner
            )
            last_xy = orient_to(
                np.array([xs[located[-1]], ys[located[-1]]]), str(teams[located[-1]]), winner
            )
            forward = float(last_xy[0] - first_xy[0]) * (PITCH_LENGTH / SB_PITCH_LENGTH)
            travelled = float(np.hypot(*(last_xy - first_xy))) * (
                PITCH_LENGTH / SB_PITCH_LENGTH
            )
        else:
            forward, travelled = np.nan, np.nan

        losing_poss = np.flatnonzero(poss_team == loser)
        losing_poss = losing_poss[poss[losing_poss] == poss[loss_row]]
        poss_len = float(times[losing_poss[-1]] - times[losing_poss[0]]) if (
            losing_poss.size >= 2
        ) else 0.0

        regained = bool(np.any((poss_team[in_window] == loser) & (idx[in_window] > int(
            turn.start_index))))

        feature: dict[str, Any] = {
            "match_id": int(turn.match_id),
            "turnover_id": str(turn.turnover_id),
            "period": int(turn.period),
            "t": lo,
            "losing_team": loser,
            "winning_team": winner,
            "phase": turn.phase,
            "x": float(turn.x),
            "y": float(turn.y),
            "dist_own_goal": float(np.hypot(turn.x, turn.y - PITCH_WIDTH / 2.0)),
            "dist_opp_goal": float(
                np.hypot(PITCH_LENGTH - turn.x, turn.y - PITCH_WIDTH / 2.0)
            ),
            "dist_from_centre": float(abs(turn.y - PITCH_WIDTH / 2.0)),
            "time_to_next_event": float(times[after[0]] - lo) if after.size else np.nan,
            "n_events_5s": int(in_window.size),
            "n_loser_events_5s": int(np.count_nonzero(teams[in_window] == loser)),
            "n_winner_events_5s": int(np.count_nonzero(teams[in_window] == winner)),
            "under_pressure_count": int(np.count_nonzero(pressed[in_window])),
            "counterpress_count": int(np.count_nonzero(cpress[in_window])),
            "pressure_event_count": int(np.count_nonzero(types[in_window] == "Pressure")),
            "loss_under_pressure": float(pressed[loss_row]),
            "loss_counterpress": float(cpress[loss_row]),
            "ball_forward_5s": forward,
            "ball_distance_5s": travelled,
            "losing_possession_events": int(losing_poss.size),
            "losing_possession_duration": poss_len,
            "loss_duration": float(durations[loss_row]),
            "regained_within_5s": float(regained),
            "loss_event_type": str(types[loss_row]),
            "next_play_pattern": str(turn.next_play_pattern),
        }
        for k in range(3):
            if k < nxt.size:
                j = int(nxt[k])
                feature[f"next_{k + 1}_type"] = str(types[j])
                feature[f"next_{k + 1}_by_loser"] = float(teams[j] == loser)
            else:
                feature[f"next_{k + 1}_type"] = None
                feature[f"next_{k + 1}_by_loser"] = np.nan
        rows.append(feature)

    out = pd.DataFrame(rows, columns=columns)
    for column in NUMERIC_FEATURES:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype(float)
    return out


# --------------------------------------------------------------------------- #
# Calibration against true 25 Hz tracking
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CameraSample:
    """One real 360 window's sampling geometry, reusable on another match.

    Captured from StatsBomb so that the Metrica simulation is driven by measured
    broadcast behaviour rather than by an invented camera model.

    Attributes:
        offsets: Seconds after the turnover at which frames were available.
        polygons: One ``(m, 2)`` broadcast polygon per offset, in canonical
            metres and expressed RELATIVE TO THE BALL, in the orientation where
            the team that won the ball attacks ``+x``. Storing it relative to
            the ball is what lets it be transplanted onto a different match: the
            camera follows the ball, so the framing is a function of where play
            is, not of absolute pitch position.
        n_players: Players actually captured in each frame.
    """

    offsets: tuple[float, ...]
    polygons: tuple[np.ndarray, ...]
    n_players: tuple[int, ...]


def collect_camera_samples(
    events: pd.DataFrame,
    frames: dict[str, FreezeFrame],
    turnovers: pd.DataFrame | None = None,
    window_s: float = 5.0,
) -> list[CameraSample]:
    """Extract the sampling geometry of every 360 counter-press window.

    Args:
        events: Frame from :func:`load_statsbomb_events`.
        frames: Mapping from :func:`load_freeze_frames`.
        turnovers: Precomputed turnovers, or None to detect them.
        window_s: Window length in seconds.

    Returns:
        One :class:`CameraSample` per turnover that had at least one frame.
    """
    turns = find_turnovers(events) if turnovers is None else turnovers
    if turns.empty:
        return []
    ev = events.sort_values("index").reset_index(drop=True)
    times = ev["t"].to_numpy(dtype=float)
    periods = ev["period"].to_numpy(dtype=np.int64)
    ids, teams = ev["id"].to_numpy(), ev["team"].to_numpy()
    xs, ys = ev["x"].to_numpy(dtype=float), ev["y"].to_numpy(dtype=float)

    out: list[CameraSample] = []
    for turn in turns.itertuples(index=False):
        lo, hi = float(turn.t), float(turn.t) + window_s
        sel = np.flatnonzero(
            (periods == int(turn.period)) & (times >= lo) & (times <= hi) & np.isfinite(xs)
        )
        offsets: list[float] = []
        polys: list[np.ndarray] = []
        counts: list[int] = []
        for i in sel:
            frame = frames.get(str(ids[i]))
            if frame is None or frame.visible_area.shape[0] < 3:
                continue
            team = str(teams[i])
            if team not in (turn.winning_team, turn.losing_team):
                continue
            ball = sb_to_canonical(
                orient_to(np.array([xs[i], ys[i]]), team, str(turn.winning_team))
            )
            poly = sb_to_canonical(
                orient_to(frame.visible_area, team, str(turn.winning_team))
            )
            offsets.append(float(times[i]) - lo)
            polys.append(poly - ball)
            counts.append(len(frame))
        if offsets:
            out.append(
                CameraSample(tuple(offsets), tuple(polys), tuple(counts))
            )
    return out


#: The variants :func:`calibrate_against_metrica` computes, in the order they
#: degrade away from the truth. Each adds exactly one of the ways a 360
#: freeze-frame differs from 25 Hz tracking, so the correlation table attributes
#: the loss of fidelity to a specific cause instead of reporting one lump.
CALIBRATION_VARIANTS: Final[tuple[str, ...]] = (
    "true_uniform",
    "sparse_offsets",
    "sparse_zero_velocity",
    "sparse_zero_velocity_camera",
    "sparse_zero_velocity_camera_visible",
)


def _metrica_variants(
    index: Any,
    loss: dict[str, Any],
    camera: CameraSample,
    grid: PitchGrid,
    pc_params: PitchControlParams,
    uniform_offsets: tuple[float, ...],
    min_players_per_team: int,
    polygon_scale: float = 1.0,
) -> dict[str, Any]:
    """Score one Metrica loss under every calibration variant.

    Args:
        index: A ``counterpress._TrackingIndex`` for the match.
        loss: One loss row as a dict, with ``frame``, ``losing_team`` and
            ``winning_team``.
        camera: The 360 window geometry transplanted onto this loss.
        grid: Pitch-control grid.
        pc_params: Pitch-control constants.
        polygon_scale: Dilation applied to the broadcast polygon about the ball.
            The polygon is transplanted onto a different match, so the framing
            will not sit over Metrica's play exactly as it sat over StatsBomb's;
            this knob lets the SEVERITY of the simulated truncation be tuned to
            reproduce the player count real 360 frames actually contain, instead
            of being left as an unquantified caveat.
        uniform_offsets: The true metric's timepoints.
        min_players_per_team: Minimum visible players per team for a frame to
            count under the camera variants.

    Returns:
        A dict with one entry per name in :data:`CALIBRATION_VARIANTS`, plus
        ``n_sparse_frames`` and ``sim_players_visible``.
    """
    from .counterpress import _ball_at  # local: private helper, same package

    winning, losing = str(loss["winning_team"]), str(loss["losing_team"])
    f0, period0 = int(loss["frame"]), index.period_of(int(loss["frame"]))
    direction = int(index.attack_direction.get(winning, 1))
    max_gap = int(round(5.0 * index.fps))

    def frames_for(offsets: Sequence[float]) -> list[tuple[float, np.ndarray, PlayerSet,
                                                           PlayerSet]]:
        got: list[tuple[float, np.ndarray, PlayerSet, PlayerSet]] = []
        for offset in offsets:
            frame = f0 + int(round(float(offset) * index.fps))
            if not index.has_frame(frame) or index.period_of(frame) != period0:
                continue
            ball = _ball_at(index, frame, "hold", max_gap)
            if not isinstance(ball, tuple):
                continue
            winner = index.player_set(frame, winning)
            presser = index.player_set(frame, losing)
            if len(winner) == 0 or len(presser) == 0:
                continue
            got.append((float(offset), ball[0], winner, presser))
        return got

    def share(winner: PlayerSet, presser: PlayerSet, ball: np.ndarray,
              mask: np.ndarray | None = None) -> float:
        result = pitch_control_full(
            winner, presser, ball_pos=ball, grid=grid, params=pc_params, offsides=False
        )
        return _share_from_surfaces(result.attacking, result.defending, mask)

    def stationary(players: PlayerSet) -> PlayerSet:
        return PlayerSet(
            positions=players.positions,
            velocities=np.zeros_like(players.velocities),
            is_gk=players.is_gk,
            ids=players.ids,
        )

    out: dict[str, Any] = dict.fromkeys(CALIBRATION_VARIANTS, np.nan)

    uniform = frames_for(uniform_offsets)
    if uniform and uniform[0][0] == 0.0 and len(uniform) >= 2:
        out["true_uniform"] = float(np.mean([share(w, p, b) for _, b, w, p in uniform]))

    sparse = frames_for(camera.offsets)
    out["n_sparse_frames"] = len(sparse)
    if len(sparse) >= 2:
        out["sparse_offsets"] = float(np.mean([share(w, p, b) for _, b, w, p in sparse]))
        out["sparse_zero_velocity"] = float(
            np.mean([share(stationary(w), stationary(p), b) for _, b, w, p in sparse])
        )

        full: list[float] = []
        vis: list[float] = []
        kept: list[int] = []
        for (_offset, ball, winner, presser), poly_rel in zip(
            sparse, camera.polygons[: len(sparse)], strict=False
        ):
            oriented = poly_rel if direction > 0 else -poly_rel
            poly = ball + oriented * float(polygon_scale)
            in_w = points_in_polygon(winner.positions, poly)
            in_p = points_in_polygon(presser.positions, poly)
            if in_w.sum() < min_players_per_team or in_p.sum() < min_players_per_team:
                continue
            seen_w, seen_p = stationary(winner.take(in_w)), stationary(presser.take(in_p))
            mask = points_in_polygon(grid.targets, poly).reshape(grid.shape)
            result = pitch_control_full(
                seen_w, seen_p, ball_pos=ball, grid=grid, params=pc_params, offsides=False
            )
            full.append(_share_from_surfaces(result.attacking, result.defending))
            vis.append(_share_from_surfaces(result.attacking, result.defending, mask))
            kept.append(int(in_w.sum() + in_p.sum()))
        if len(full) >= 2:
            out["sparse_zero_velocity_camera"] = float(np.mean(full))
            out["sparse_zero_velocity_camera_visible"] = float(np.mean(vis))
            out["sim_players_visible"] = float(np.mean(kept))
    return out


def calibrate_against_metrica(
    metrica_dir: Path | str = "data/raw/metrica",
    games: Sequence[int] = (1, 2),
    camera_samples: Sequence[CameraSample] | None = None,
    grid: PitchGrid | None = None,
    pc_params: PitchControlParams | None = None,
    seed: int = 0,
    jobs: int | None = None,
    polygon_scale: float = 1.0,
    losses_per_game: int | None = None,
) -> pd.DataFrame:
    """Measure how well a 360-style label reproduces the true tracking value.

    This is the single most valuable honesty check available to the proxy, and
    it is cheap. The Metrica sample matches have real 25 Hz tracking AND real
    events, so the tracking-derived counter-press value and a simulated
    freeze-frame version of it can both be computed on the same turnovers and
    correlated directly.

    The simulation degrades the truth one step at a time, so the table
    attributes the loss of fidelity rather than reporting one lump:

    ``true_uniform``
        The real metric: six evenly spaced instants, real velocities, all 22
        players. Computed here rather than taken from
        :func:`pressing.counterpress.score_match` so that every variant runs
        through identical code -- but it is checked against ``score_match``'s
        ``recovery_area``, and that check is the harness's own validation.
    ``sparse_offsets``
        The same, sampled at the irregular offsets a REAL StatsBomb 360 window
        provided instead of at 0..5 s. Isolates the cost of event-driven
        sampling.
    ``sparse_zero_velocity``
        Adds the freeze-frame's missing velocities: every player stationary.
    ``sparse_zero_velocity_camera``
        Adds the broadcast truncation: only the players inside a real
        ``visible_area`` polygon, transplanted onto this match relative to the
        ball. Integrated over the whole pitch, as ``recovery_360`` is.
    ``sparse_zero_velocity_camera_visible``
        The same players, but integrating only inside the polygon, as
        ``recovery_360_visible`` is.

    Args:
        metrica_dir: Directory holding the Metrica sample matches.
        games: Which Metrica games to use.
        camera_samples: Real 360 window geometries from
            :func:`collect_camera_samples`. Required -- an invented camera would
            defeat the point of the measurement.
        grid: Pitch-control grid. Defaults to the 50 x 32 demo grid.
        pc_params: Pitch-control constants.
        seed: Seed for the deterministic assignment of camera samples to losses.
        jobs: Worker processes for the reference ``score_match`` call.
        polygon_scale: Dilation of the transplanted broadcast polygon about the
            ball. ``1.0`` uses the polygon exactly as StatsBomb recorded it,
            which on Metrica retains fewer players than a real 360 frame
            contains; raising it lets the simulated truncation be tuned to
            match the measured real player count, so the severity of the
            camera effect is bracketed rather than assumed.
        losses_per_game: Cap on losses scored per game, for a quick sweep.

    Returns:
        One row per Metrica loss with ``match_id``, ``phase``, the shipped
        metric's ``recovery_area``, and every variant in
        :data:`CALIBRATION_VARIANTS`.

    Raises:
        ValueError: If ``camera_samples`` is empty or None.
    """
    if not camera_samples:
        raise ValueError(
            "camera_samples is required: the point of this measurement is to use REAL "
            "StatsBomb 360 sampling geometry, not an invented camera model"
        )
    from ._io import load_metrica_match
    from .counterpress import CounterPressParams, _build_index, score_match

    grid = grid or make_grid(n_x=50)
    pc_params = pc_params or PitchControlParams()
    defaults = CounterPressParams()
    rng = np.random.default_rng(seed)

    rows: list[dict[str, Any]] = []
    for game in games:
        match = load_metrica_match(int(game), metrica_dir)
        truth = score_match(match.tracking, match.events, jobs=jobs)
        index = _build_index(match.tracking, float(match.tracking.attrs.get("fps", 25.0)))
        picks = rng.integers(0, len(camera_samples), size=len(truth))
        used = 0
        for (_, loss), pick in zip(truth.iterrows(), picks, strict=True):
            if not bool(loss.get("scored", False)):
                continue
            if losses_per_game is not None and used >= losses_per_game:
                break
            used += 1
            variants = _metrica_variants(
                index,
                loss.to_dict(),
                camera_samples[int(pick)],
                grid,
                pc_params,
                defaults.timepoints,
                min_players_per_team=LabelParams().min_players_per_team,
                polygon_scale=polygon_scale,
            )
            rows.append(
                {
                    "match_id": str(loss["match_id"]),
                    "phase": loss.get("phase"),
                    "losing_team": loss["losing_team"],
                    "recovery_area": float(loss["recovery_area"]),
                    "polygon_scale": float(polygon_scale),
                    **variants,
                }
            )
    return pd.DataFrame(rows)


def calibration_report(calibration: pd.DataFrame) -> pd.DataFrame:
    """Summarise a :func:`calibrate_against_metrica` frame into one table.

    Args:
        calibration: Output of :func:`calibrate_against_metrica`.

    Returns:
        One row per variant with ``n``, ``pearson_r`` and ``spearman_r``
        against the shipped ``recovery_area``, plus ``mae``, ``bias``, the
        variant's own ``mean`` and ``sd``, and the phase gradient
        ``final_third_minus_build_up`` -- the last being the property the
        tracking metric is most strongly driven by and therefore the one a
        substitute label most needs to preserve.
    """
    rows: list[dict[str, Any]] = []
    truth = calibration["recovery_area"]
    for variant in CALIBRATION_VARIANTS:
        both = calibration[[variant, "recovery_area", "phase"]].dropna(
            subset=[variant, "recovery_area"]
        )
        if both.empty:
            rows.append({"variant": variant, "n": 0})
            continue
        values = both[variant]
        by_phase = both.groupby("phase")[variant].mean()
        rows.append(
            {
                "variant": variant,
                "n": int(len(both)),
                "pearson_r": float(values.corr(both["recovery_area"])),
                "spearman_r": float(values.corr(both["recovery_area"], method="spearman")),
                "mae": float((values - both["recovery_area"]).abs().mean()),
                "bias": float((values - both["recovery_area"]).mean()),
                "mean": float(values.mean()),
                "sd": float(values.std()),
                "final_third_minus_build_up": float(
                    by_phase.get("final_third", np.nan) - by_phase.get("build_up", np.nan)
                ),
            }
        )
    truth_phase = calibration.groupby("phase")["recovery_area"].mean()
    rows.insert(
        0,
        {
            "variant": "recovery_area (shipped metric)",
            "n": int(truth.notna().sum()),
            "pearson_r": 1.0,
            "spearman_r": 1.0,
            "mae": 0.0,
            "bias": 0.0,
            "mean": float(truth.mean()),
            "sd": float(truth.std()),
            "final_third_minus_build_up": float(
                truth_phase.get("final_third", np.nan) - truth_phase.get("build_up", np.nan)
            ),
        },
    )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Reporting caveats
# --------------------------------------------------------------------------- #

#: Phase means of the TRACKING metric on Metrica game 1 -- the quantity the proxy
#: stands in for. Reproduce with
#: ``uv run pressing report --rank costly`` or, directly,
#: ``counterpress.score_match(...).groupby("phase")["recovery_area"].mean()``.
#: These are here as a CONTRAST only. They must never be printed as a reference
#: scale for a proxy number; see :data:`PROXY_PHASE_MEANS`.
TRACKING_PHASE_MEANS: Final[dict[str, float]] = {
    "build_up": 0.257,
    "middle": 0.468,
    "final_third": 0.701,
}

#: Phase means of what the SHIPPED proxy artefact actually predicts, over the
#: 6,195 labelled training turnovers. Measured with::
#:
#:     m = proxy.load_bundled_model()
#:     lab["pred"] = m.predict(lab)
#:     lab.groupby("phase")["pred"].mean()
#:
#: Recorded here as the fallback for :func:`phase_reference_means` when no model
#: is to hand; a trained artefact carries its own copy in
#: ``metadata["predicted_phase_means"]``, which is what callers get in practice.
PROXY_PHASE_MEANS: Final[dict[str, float]] = {
    "build_up": 0.464,
    "middle": 0.490,
    "final_third": 0.542,
}

_PHASE_ORDER: Final[tuple[str, ...]] = ("build_up", "middle", "final_third")
_PHASE_LABELS: Final[dict[str, str]] = {
    "build_up": "build-up",
    "middle": "middle",
    "final_third": "final third",
}


def phase_reference_means(model: ProxyModel | None = None) -> dict[str, float]:
    """The proxy's OWN mean prediction per pitch third.

    A CSV of proxy estimates needs a reference scale, or "compare only within a
    phase" gives the reader nothing to compare against. Until v0.1.0 the scale
    printed in the header was the *tracking* metric's -- build-up 0.26, middle
    0.47, final third 0.70 -- which is not a scale these numbers live on. The
    proxy's own gradient across the pitch is about +0.08, roughly a sixth of the
    tracking metric's +0.44, so a reader who took a proxy build-up value of 0.47
    against a stated 0.26 would read the proxy's own average as an extraordinary
    press.

    Args:
        model: A fitted model whose metadata carries ``predicted_phase_means``.
            ``None``, or a model trained before that key existed, falls back to
            the recorded :data:`PROXY_PHASE_MEANS`.

    Returns:
        ``{phase: mean predicted recovery}`` for build-up, middle, final third.
    """
    recorded = dict(model.metadata).get("predicted_phase_means") if model else None
    source = recorded if isinstance(recorded, dict) and recorded else PROXY_PHASE_MEANS
    return {p: float(source[p]) for p in _PHASE_ORDER if p in source}


def recovery_phase_caveat(model: ProxyModel | None = None) -> list[str]:
    """The phase caveat for a proxy CSV header, on the proxy's own scale.

    Args:
        model: Fitted model, for its recorded predicted phase means.

    Returns:
        Two lines, without the leading ``#``. The first gives the proxy's own
        phase means and the instruction; the second states the compression
        against the tracking metric explicitly, so nobody transplants the
        tracking scale back on.
    """
    means = phase_reference_means(model)
    printed = ", ".join(f"{_PHASE_LABELS[p]} {means[p]:.3f}" for p in _PHASE_ORDER if p in means)
    proxy_gradient = means.get("final_third", float("nan")) - means.get("build_up", float("nan"))
    truth_gradient = TRACKING_PHASE_MEANS["final_third"] - TRACKING_PHASE_MEANS["build_up"]
    return [
        f"recovery is driven by WHERE the ball was lost. THIS MODEL's own phase means are "
        f"{printed}. Compare only within a phase.",
        f"do NOT scale these against the tracking metric's phase means "
        f"({TRACKING_PHASE_MEANS['build_up']:.2f} / {TRACKING_PHASE_MEANS['middle']:.2f} / "
        f"{TRACKING_PHASE_MEANS['final_third']:.2f} on Metrica game 1): the proxy's gradient "
        f"across the pitch is {proxy_gradient:+.3f} against the tracking metric's "
        f"{truth_gradient:+.3f}, so the two are not on the same scale.",
    ]


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProxyMetrics:
    """Held-out performance of the proxy model.

    Attributes:
        r: Pearson correlation between predicted and label values.
        spearman: Rank correlation, which a heavily skewed target makes worth
            reading alongside ``r``.
        mae: Mean absolute error.
        rmse: Root mean squared error.
        n_losses: Number of labelled turnovers scored.
        n_matches: Number of distinct matches contributing.
    """

    r: float
    spearman: float
    mae: float
    rmse: float
    n_losses: int
    n_matches: int


@dataclass
class ProxyModel:
    """A fitted proxy model plus everything needed to apply it.

    The categorical vocabularies travel with the estimator. Re-deriving them
    from whatever frame happens to be at hand would silently shift every code
    the moment a match contained an event type the training set did not, which
    is a bug that produces plausible predictions rather than an error.

    Attributes:
        estimator: The fitted scikit-learn regressor.
        feature_names: Feature order the estimator was fitted on.
        categorical_names: Which of those are categorical.
        vocabularies: Category values per categorical feature. An unseen value
            encodes to ``-1``.
        target: Name of the label column the model was trained against.
        metadata: Free-form provenance recorded in the model card.
    """

    estimator: Any
    feature_names: tuple[str, ...]
    categorical_names: tuple[str, ...]
    vocabularies: dict[str, tuple[str, ...]]
    target: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def encode(self, features: pd.DataFrame) -> np.ndarray:
        """Turn a feature frame into the model's numeric design matrix.

        Args:
            features: Frame containing at least :attr:`feature_names`.

        Returns:
            ``(n, len(feature_names))`` float array. Categoricals become their
            vocabulary index, or ``-1`` when unseen; missing numerics stay NaN,
            which the histogram gradient booster handles natively.

        Raises:
            KeyError: If a required feature column is absent.
        """
        missing = [c for c in self.feature_names if c not in features.columns]
        if missing:
            raise KeyError(f"features missing columns: {', '.join(missing)}")
        columns: list[np.ndarray] = []
        for name in self.feature_names:
            if name in self.categorical_names:
                lookup = {v: i for i, v in enumerate(self.vocabularies[name])}
                values = features[name].astype("object")
                columns.append(
                    np.array([lookup.get(v, -1) for v in values], dtype=float)
                )
            else:
                columns.append(pd.to_numeric(features[name], errors="coerce").to_numpy(float))
        return np.column_stack(columns)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict the label for each row of a feature frame.

        Args:
            features: Frame from :func:`build_features`.

        Returns:
            ``(n,)`` predictions clipped to ``[0, 1]``. The target is a control
            share, so a value outside that interval is not a number a report may
            print; the clip is applied here rather than left to a caller who
            might forget.
        """
        return np.clip(self.estimator.predict(self.encode(features)), 0.0, 1.0)


def split_by_match(
    frame: pd.DataFrame,
    test_fraction: float = 0.3,
    seed: int = 0,
    group_col: str = "match_id",
) -> tuple[np.ndarray, np.ndarray]:
    """Split rows into train and test by MATCH, never by row.

    Splitting by turnover would put two turnovers from the same match, five
    seconds apart, on opposite sides of the split; the spec is explicit that
    such leakage is disqualifying. Whole matches move together.

    Args:
        frame: Any frame carrying ``group_col``.
        test_fraction: Share of MATCHES held out, not of rows.
        seed: Seed for the shuffle, so the split is reproducible.
        group_col: Column identifying the match.

    Returns:
        ``(train_mask, test_mask)``, both boolean arrays over ``frame``'s rows.

    Raises:
        ValueError: If ``test_fraction`` is not strictly inside ``(0, 1)`` or
            there are too few matches to hold any out.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    matches = np.array(sorted(frame[group_col].unique()))
    if matches.size < 2:
        raise ValueError(f"need at least 2 matches to split, got {matches.size}")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(matches)
    n_test = max(1, int(round(matches.size * test_fraction)))
    n_test = min(n_test, matches.size - 1)
    test_matches = set(shuffled[:n_test].tolist())
    values = frame[group_col].to_numpy()
    test_mask = np.array([v in test_matches for v in values], dtype=bool)
    return ~test_mask, test_mask


def train(
    features: pd.DataFrame,
    targets: pd.Series,
    group_col: str = "match_id",
    *,
    feature_names: Sequence[str] = FEATURE_COLUMNS,
    categorical_names: Sequence[str] = CATEGORICAL_FEATURES,
    target_name: str = "recovery_360",
    metadata: dict[str, Any] | None = None,
    **estimator_kwargs: Any,
) -> ProxyModel:
    """Fit the gradient boosted proxy model.

    ``group_col`` is accepted for signature compatibility and is deliberately
    NOT used to fit anything -- the match id must never reach the estimator, or
    the model would learn a per-match offset that cannot exist at prediction
    time. Splitting is :func:`split_by_match`'s job and is done by the caller.

    Args:
        features: Feature frame from :func:`build_features`.
        targets: Label values aligned to ``features``.
        group_col: Column identifying the match. Excluded from the fit.
        feature_names: Features to fit on.
        categorical_names: Which of those are categorical.
        target_name: Name recorded on the artefact.
        metadata: Provenance to store on the artefact.
        **estimator_kwargs: Overrides for ``HistGradientBoostingRegressor``.

    Returns:
        The fitted :class:`ProxyModel`.

    Raises:
        ValueError: If ``features`` and ``targets`` disagree in length, if any
            target is missing, or if ``group_col`` was requested as a feature.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    if len(features) != len(targets):
        raise ValueError(f"features has {len(features)} rows, targets has {len(targets)}")
    if group_col in feature_names:
        raise ValueError(
            f"{group_col!r} must not be a feature: a per-match offset cannot exist at "
            "prediction time and would leak the split"
        )
    y = np.asarray(targets, dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("targets contain non-finite values; drop unlabelled rows first")

    names = tuple(feature_names)
    cats = tuple(categorical_names)
    vocabularies = {
        name: tuple(sorted({str(v) for v in features[name].dropna().unique()}))
        for name in cats
    }
    model = ProxyModel(
        estimator=None,
        feature_names=names,
        categorical_names=cats,
        vocabularies=vocabularies,
        target=target_name,
        metadata=dict(metadata or {}),
    )
    kwargs: dict[str, Any] = {
        "loss": "squared_error",
        "max_iter": 400,
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 40,
        "l2_regularization": 1.0,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 25,
        "random_state": 0,
    }
    kwargs.update(estimator_kwargs)
    estimator = HistGradientBoostingRegressor(
        categorical_features=[names.index(c) for c in cats], **kwargs
    )
    estimator.fit(model.encode(features), y)
    model.estimator = estimator
    return model


def evaluate(model: ProxyModel, features: pd.DataFrame, targets: pd.Series) -> ProxyMetrics:
    """Score a fitted model on held-out matches.

    Args:
        model: Fitted model from :func:`train`.
        features: Held-out feature frame, including ``match_id``.
        targets: Held-out labels aligned to ``features``.

    Returns:
        Measured :class:`ProxyMetrics`.
    """
    return score_predictions(np.asarray(model.predict(features)), targets, features)


def score_predictions(
    predictions: np.ndarray,
    targets: pd.Series,
    features: pd.DataFrame,
    group_col: str = "match_id",
) -> ProxyMetrics:
    """Metrics for any set of predictions, including baselines.

    Args:
        predictions: ``(n,)`` predicted values.
        targets: Observed values aligned to ``predictions``.
        features: Frame carrying ``group_col``, used only for the match count.
        group_col: Column identifying the match.

    Returns:
        Measured :class:`ProxyMetrics`. ``r`` is NaN when the predictions are
        constant, which is exactly what a mean-predictor baseline is: reporting
        r = 0 there would be an invented number.
    """
    y = pd.Series(np.asarray(targets, dtype=float)).reset_index(drop=True)
    p = pd.Series(np.asarray(predictions, dtype=float)).reset_index(drop=True)
    constant = float(p.std()) == 0.0
    return ProxyMetrics(
        r=float("nan") if constant else float(p.corr(y)),
        spearman=float("nan") if constant else float(p.corr(y, method="spearman")),
        mae=float((p - y).abs().mean()),
        rmse=float(np.sqrt(((p - y) ** 2).mean())),
        n_losses=int(len(y)),
        n_matches=int(features[group_col].nunique()),
    )


def compare_models(
    features: pd.DataFrame,
    labels: pd.Series,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    model: ProxyModel,
) -> pd.DataFrame:
    """Score the full model against the baselines that matter, on the same split.

    The baselines are not decoration. The tracking metric is dominated by where
    the ball was lost -- the shipped implementation measures a three-fold spread
    across pitch thirds -- so a model that merely rediscovers field position
    would post a respectable ``r`` while adding nothing. Reporting these
    side by side is the only way the reader can tell which happened.

    Baselines, in increasing strength:

    ``mean``
        Predict the training mean for everything. Its ``r`` is NaN by
        construction, not zero.
    ``phase``
        Predict the training mean of the turnover's pitch third.
    ``turnover_x``
        A gradient booster on the turnover's ``x`` and nothing else.
    ``full``
        Every feature.

    Args:
        features: The whole feature frame.
        labels: Labels aligned to ``features``.
        train_mask: Boolean training rows.
        test_mask: Boolean held-out rows.
        model: The already-fitted full model.

    Returns:
        One row per model with the fields of :class:`ProxyMetrics`, plus
        ``residual_r``: the correlation between prediction and label AFTER both
        have had a smooth function of turnover ``x`` removed. That is the number
        that answers "does this know anything beyond field position".
    """
    y = pd.Series(np.asarray(labels, dtype=float)).reset_index(drop=True)
    train_x, test_x = features[train_mask], features[test_mask]
    train_y, test_y = y[train_mask], y[test_mask]

    phase_means = train_y.groupby(train_x["phase"].to_numpy()).mean()
    x_only = train(
        train_x,
        train_y,
        feature_names=("x",),
        categorical_names=(),
        target_name=model.target,
        max_iter=200,
    )

    predictions: dict[str, np.ndarray] = {
        "mean": np.full(int(test_mask.sum()), float(train_y.mean())),
        "phase": test_x["phase"].map(phase_means).fillna(float(train_y.mean())).to_numpy(float),
        "turnover_x": x_only.predict(test_x),
        "full": model.predict(test_x),
    }

    # Residualise against a smooth function of x fitted on TRAINING rows only,
    # so the control never sees held-out labels.
    x_fit = x_only.predict(test_x)
    residual_y = test_y.to_numpy(float) - x_fit

    rows: list[dict[str, Any]] = []
    for name, values in predictions.items():
        metrics = score_predictions(values, test_y, test_x)
        residual_p = values - x_fit
        rows.append(
            {
                "model": name,
                "r": metrics.r,
                "spearman": metrics.spearman,
                "mae": metrics.mae,
                "rmse": metrics.rmse,
                "n_losses": metrics.n_losses,
                "n_matches": metrics.n_matches,
                "residual_r": (
                    float("nan")
                    if float(np.std(residual_p)) == 0.0
                    else float(np.corrcoef(residual_p, residual_y)[0, 1])
                ),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Dataset assembly
# --------------------------------------------------------------------------- #


def build_match_dataset(
    match_id: int,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    params: LabelParams | None = None,
    grid: PitchGrid | None = None,
    pc_params: PitchControlParams | None = None,
) -> pd.DataFrame:
    """Features and freeze-frame labels for one cached StatsBomb match.

    Args:
        match_id: StatsBomb match id, already cached by :func:`download_match`.
        cache_dir: Cache root.
        params: Label knobs.
        grid: Pitch-control grid.
        pc_params: Pitch-control constants.

    Returns:
        One row per open-play turnover, features joined to labels. Rows the
        label could not be built for are kept with a null label and a
        ``skip_reason``, so the coverage rate is measurable rather than
        invisible.

    Raises:
        FileNotFoundError: If either file is missing from the cache.
    """
    root = Path(cache_dir)
    events_path = root / "events" / f"{match_id}.json"
    frames_path = root / "three-sixty" / f"{match_id}.json"
    for path in (events_path, frames_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} is not cached; run download_match({match_id}) first")

    frames = load_freeze_frames(frames_path)
    events = load_statsbomb_events(events_path, match_id=match_id, frame_ids=set(frames))
    turnovers = find_turnovers(events)
    labels = label_turnovers(events, frames, turnovers, params=params, grid=grid,
                             pc_params=pc_params)
    features = build_features(events, turnovers)
    label_only = [c for c in labels.columns if c not in features.columns or c == "turnover_id"]
    return features.merge(labels[label_only], on="turnover_id", how="left")


def build_dataset(
    match_ids: Sequence[int],
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    params: LabelParams | None = None,
    jobs: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Assemble the training set from a list of cached matches.

    Args:
        match_ids: Cached StatsBomb match ids.
        cache_dir: Cache root.
        params: Label knobs.
        jobs: Worker processes. ``None`` or 1 runs serially. Each worker handles
            whole matches, so nothing large is pickled per task.
        verbose: Print per-match progress.

    Returns:
        The concatenated per-match datasets, unlabelled rows included, in a
        CANONICAL row order: ascending ``match_id``, and within a match the
        order ``build_match_dataset`` produced (ascending ``loss_index``). The
        frame does not depend on ``jobs``, on which worker finished first, or on
        the order ``match_ids`` arrived in.

    Note:
        That canonical order is correctness, not tidiness. This function used to
        append each part as its future *completed*, so with ``--jobs 5`` the row
        order was a race. The content was identical either way, but
        ``HistGradientBoostingRegressor`` draws its early-stopping validation
        slice out of the design matrix by position, so a different row order is
        a different validation slice, a different stopping iteration and a
        different fitted tree. That is the whole of the "the refit is not
        reproducible across processes" problem the v0.1.0 audit measured.
        Measured on the identical 6,195 labelled rows, refitting the split model
        after nothing but a row shuffle moved the held-out ``r`` to 0.369387,
        0.374414 and 0.368491 and the stopping iteration to 85, 69 and 85, where
        three fresh processes on the canonical order all gave 0.369162 stopping
        at 79 and agreed bit for bit (identical SHA-256 of the held-out
        prediction vector). Across every row order measured, including the one
        the v0.1.0 artefact happened to be built from, ``r`` spans roughly 0.368
        to 0.391. Fix the order and the refit is deterministic; leave it to
        chance and no printed digit past the second is meaningful.
    """
    if jobs is None or jobs <= 1:
        parts = []
        for i, mid in enumerate(match_ids, start=1):
            parts.append(build_match_dataset(mid, cache_dir, params=params))
            if verbose:
                print(f"[{i}/{len(match_ids)}] {mid}: {len(parts[-1])} turnovers", flush=True)
        return _canonical_row_order(parts)

    from concurrent.futures import ProcessPoolExecutor, as_completed

    slots: list[pd.DataFrame | None] = [None] * len(match_ids)
    with ProcessPoolExecutor(max_workers=int(jobs)) as pool:
        futures = {
            pool.submit(build_match_dataset, int(mid), str(cache_dir), params): (i, int(mid))
            for i, mid in enumerate(match_ids)
        }
        for done, future in enumerate(as_completed(futures), start=1):
            index, mid = futures[future]
            slots[index] = future.result()
            if verbose:
                print(f"[{done}/{len(match_ids)}] {mid}: {len(slots[index])} turnovers",
                      flush=True)
    return _canonical_row_order([p for p in slots if p is not None])


def _canonical_row_order(parts: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-match frames into one canonically ordered dataset.

    Args:
        parts: Per-match frames, in any order.

    Returns:
        The concatenation sorted by ``match_id`` with a *stable* sort, so each
        match's own row order survives, and re-indexed from zero. An empty list
        gives an empty frame.
    """
    if not parts:
        return pd.DataFrame()
    frame = pd.concat(parts, ignore_index=True)
    if "match_id" not in frame.columns:
        return frame
    return frame.sort_values("match_id", kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Prediction and persistence
# --------------------------------------------------------------------------- #


def predict(model: ProxyModel, events: pd.DataFrame) -> pd.DataFrame:
    """Estimate counter-press values for an event-only match.

    Args:
        model: Fitted model from :func:`train`, or :func:`load_bundled_model`.
        events: StatsBomb events frame from :func:`load_statsbomb_events`.

    Returns:
        One row per open-play turnover with :data:`FEATURE_META_COLUMNS`, the
        estimated ``recovery``, and ``epv_conceded`` as NaN. The EPV column is
        present so a report template can take the same frame from either path,
        and it is NaN rather than zero because **no EPV model exists**: EPV
        needs the opponent's control surface, which cannot be recovered from
        events. A report built on this frame must not print an EPV number.
    """
    features = build_features(events)
    if features.empty:
        return pd.DataFrame(
            columns=list(FEATURE_META_COLUMNS) + ["recovery", "epv_conceded", "estimated"]
        )
    out = features[list(FEATURE_META_COLUMNS)].copy()
    out["recovery"] = model.predict(features)
    out["epv_conceded"] = np.nan
    out["estimated"] = True
    return out


#: The two artefacts the repository ships, and what each is fitted on.
#:
#: ``"refit"``
#:     ``models/proxy_gbr.pkl.gz``. Fitted on EVERY labelled row. This is the one
#:     you would deploy, and the one :func:`load_bundled_model` returns by
#:     default. **Its held-out metrics cannot be reproduced with it**: it has
#:     seen every match, so scoring it on the recorded held-out matches is
#:     in-sample and flatters itself (measured: r = 0.6151 against a true
#:     0.3692).
#: ``"split"``
#:     ``models/proxy_gbr_split.pkl.gz``. Fitted on the 68 training matches only.
#:     This is the model the model card's metrics table describes, and the only
#:     one :func:`score_held_out` will score.
MODEL_VARIANTS: Final[dict[str, str]] = {
    "refit": "proxy_gbr.pkl.gz",
    "split": "proxy_gbr_split.pkl.gz",
}


def bundled_model_path(variant: str = "refit") -> Path:
    """Location of a model artefact shipped with the repository.

    Args:
        variant: ``"refit"`` (all labelled rows, the deployable model) or
            ``"split"`` (training matches only, the model the metrics describe).

    Returns:
        Path to the artefact, whether or not it exists.

    Raises:
        ValueError: If ``variant`` is not one of :data:`MODEL_VARIANTS`.
    """
    if variant not in MODEL_VARIANTS:
        raise ValueError(
            f"unknown model variant {variant!r}; expected one of {sorted(MODEL_VARIANTS)}"
        )
    return Path(__file__).resolve().parents[2] / "models" / MODEL_VARIANTS[variant]


#: gzip header mtime for every artefact this module writes.
#:
#: ``gzip.open`` stamps ``time.time()`` into header bytes 4:8, so two training
#: runs that fit bit-identical trees still produced different files. Measured on
#: the shipped split artefact before this was fixed: the header read
#: ``1f8b 0808 2e41 8f6a`` -- flags ``0x08`` (a filename follows) and mtime
#: ``0x6a8f412e`` = 1787773230 -- and every rewrite moved those four bytes, which
#: is the whole of the three-byte size wobble the v0.1.1 audit saw. Zero is the
#: value the gzip format reserves for "no timestamp available", so a reader loses
#: nothing: *when* an artefact was built is a git question, not a file-header one.
ARTEFACT_GZIP_MTIME: Final[int] = 0


def save_model(model: ProxyModel, path: Path | str | None = None) -> Path:
    """Serialise a fitted model to a gzipped pickle, byte-reproducibly.

    The output is a pure function of the model, the destination *basename* (gzip
    stores it in the header) and the sklearn version. Nothing here reads a clock:
    see :data:`ARTEFACT_GZIP_MTIME` for the header field that used to, and
    :func:`train_from_cache` for the metadata field that used to.

    Args:
        model: The fitted model.
        path: Destination. Defaults to :func:`bundled_model_path`.

    Returns:
        The path written.
    """
    from sklearn import __version__ as sklearn_version

    destination = Path(path) if path is not None else bundled_model_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.metadata = {**model.metadata, "sklearn_version": sklearn_version}
    payload = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    with destination.open("wb") as raw:
        with gzip.GzipFile(
            filename=destination.name,
            mode="wb",
            compresslevel=9,
            fileobj=raw,
            mtime=ARTEFACT_GZIP_MTIME,
        ) as handle:
            handle.write(payload)
    return destination


def load_bundled_model(path: Path | None = None, *, variant: str = "refit") -> ProxyModel:
    """Load a model shipped with the package.

    The artefact is a pickle, so it executes code on load. It is produced by
    this repository's own training run and committed alongside its model card;
    do not point this at a file from anywhere else.

    Args:
        path: Optional override path. Wins over ``variant``.
        variant: Which shipped artefact to load; see :data:`MODEL_VARIANTS`.
            The default ``"refit"`` model has seen every labelled match, so do
            not score it on the held-out matches -- use ``variant="split"`` and
            :func:`score_held_out`, which enforces this.

    Returns:
        The fitted :class:`ProxyModel`.

    Raises:
        FileNotFoundError: If no artefact is present.
    """
    source = Path(path) if path is not None else bundled_model_path(variant)
    if not source.exists():
        raise FileNotFoundError(
            f"no proxy model at {source}. Train one with "
            "`python -m pressing.proxy` (see models/MODEL_CARD.md for the exact command)."
        )
    with gzip.open(source, "rb") as handle:
        model = pickle.load(handle)  # noqa: S301 - our own committed artefact
    if not isinstance(model, ProxyModel):
        raise TypeError(f"{source} does not contain a ProxyModel, got {type(model).__name__}")
    return model


# --------------------------------------------------------------------------- #
# Honest scoring
# --------------------------------------------------------------------------- #


def training_match_ids(model: ProxyModel) -> tuple[int, ...]:
    """Match ids the model was fitted on, from its own metadata.

    Args:
        model: A fitted model.

    Returns:
        Sorted match ids.

    Raises:
        KeyError: If the artefact predates the metadata and cannot say.
    """
    return _recorded_ids(model, "train_match_ids")


def held_out_match_ids(model: ProxyModel) -> tuple[int, ...]:
    """Match ids held out of the fit that produced the model's metrics.

    Args:
        model: A fitted model.

    Returns:
        Sorted match ids.

    Raises:
        KeyError: If the artefact predates the metadata and cannot say.
    """
    return _recorded_ids(model, "held_out_match_ids")


def _recorded_ids(model: ProxyModel, key: str) -> tuple[int, ...]:
    """Read a match-id list out of model metadata.

    Args:
        model: A fitted model.
        key: Metadata key.

    Returns:
        Sorted match ids.

    Raises:
        KeyError: If the key is absent.
    """
    values = dict(model.metadata).get(key)
    if values is None:
        raise KeyError(
            f"this artefact records no {key!r}; it predates the split metadata. "
            "Retrain with `python -m pressing.proxy` to get an artefact that knows "
            "which matches it saw."
        )
    return tuple(sorted(int(v) for v in values))


def score_held_out(
    model: ProxyModel,
    features: pd.DataFrame,
    targets: pd.Series,
    group_col: str = "match_id",
) -> ProxyMetrics:
    """Score a model on the matches it was actually held out of -- or refuse.

    This exists because the obvious thing to do is wrong. The deployable
    artefact is refitted on every labelled row; feeding it the documented
    held-out match ids and reading the correlation gives an in-sample number
    (measured: r = 0.6151 against the honest 0.3692) with nothing to signal
    that anything went wrong. v0.1.0 warned about that in three places, which
    is the weakest possible fix. This function makes it raise instead.

    Args:
        model: A fitted model whose metadata records ``fitted_on`` and
            ``held_out_match_ids``.
        features: Any feature frame carrying ``group_col``; it is filtered down
            to the model's recorded held-out matches, so passing the whole
            labelled dataset is fine and is the expected use.
        targets: Labels aligned to ``features``.
        group_col: Column identifying the match.

    Returns:
        :class:`ProxyMetrics` over the held-out matches only.

    Raises:
        KeyError: If the artefact does not record its split.
        ValueError: If the model was fitted on the rows it is being asked to
            score, or if none of the held-out matches are present.
    """
    fitted_on = dict(model.metadata).get("fitted_on")
    held_out = held_out_match_ids(model)
    if fitted_on != "train_matches":
        raise ValueError(
            f"this model was fitted on {fitted_on!r}, so its held-out matches are not held out "
            "of it and scoring it on them is in-sample. Load the split artefact with "
            "`load_bundled_model(variant='split')`, which is fitted on the training matches "
            "only, or refit on the training rows yourself."
        )
    if len(features) != len(targets):
        raise ValueError(f"features has {len(features)} rows, targets has {len(targets)}")
    keep = np.isin(features[group_col].to_numpy(), np.asarray(held_out))
    if not keep.any():
        raise ValueError(
            f"none of the model's {len(held_out)} held-out matches appear in {group_col}"
        )
    subset = features[keep]
    y = pd.Series(np.asarray(targets, dtype=float)).reset_index(drop=True)[keep]
    return score_predictions(np.asarray(model.predict(subset)), y, subset, group_col)


# --------------------------------------------------------------------------- #
# Training entry point
# --------------------------------------------------------------------------- #

#: The competition-seasons the shipped artefact was trained on: the three
#: domestic club leagues that carry 360 data. Club football is what the report
#: is for, and three leagues in three countries make the leave-one-competition-
#: out check in the model card meaningful. The international tournaments (World
#: Cup 2022, the Euros, the women's tournaments) carry more 360 matches but are
#: a different population of teams and fitness levels.
DEFAULT_TRAINING_COMPETITIONS: Final[tuple[tuple[int, int], ...]] = (
    (11, 90),   # La Liga 2020/2021, 35 matches
    (9, 281),   # 1. Bundesliga 2023/2024, 34 matches
    (7, 235),   # Ligue 1 2022/2023, 32 matches
)

DEFAULT_TARGET: Final[str] = "recovery_360"


def train_from_cache(
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    competitions: Sequence[tuple[int, int]] = DEFAULT_TRAINING_COMPETITIONS,
    *,
    target: str = DEFAULT_TARGET,
    test_fraction: float = 0.3,
    seed: int = 0,
    jobs: int | None = None,
    download: bool = False,
    verbose: bool = True,
) -> tuple[ProxyModel, ProxyModel, pd.DataFrame, pd.DataFrame]:
    """Build the dataset, split by match, train, and measure -- in one call.

    **Two models come back, and the difference matters.** The refit model is
    fitted on EVERY labelled row: it is the one you would deploy, and shipping
    only the split model would throw a third of the data away. But it has seen
    every match, so its own held-out metrics cannot be reproduced with it --
    scoring it on the recorded held-out matches is in-sample and returns about
    r = 0.61 against a true 0.39, with nothing to signal the mistake. The split
    model is fitted on the training matches only and is the one the metrics
    table describes. Both carry ``fitted_on``, ``train_match_ids`` and
    ``held_out_match_ids`` in their metadata, and :func:`score_held_out` refuses
    to score the refit one, so the confusion is now an exception rather than a
    warning in a document.

    Neither model's metadata carries a training timestamp. It used to carry
    ``trained_utc``, read off the wall clock, which put a fresh string inside the
    pickle on every run and made the artefacts impossible to reproduce byte for
    byte however deterministic the fit was. Nothing in the repository read it.
    Everything that matters -- which matches, which target, which split, which
    sklearn -- is recorded in fields that are functions of the inputs, and *when*
    the file was built is a question for ``git log``.

    Args:
        cache_dir: StatsBomb cache root.
        competitions: ``(competition_id, season_id)`` pairs to train on.
        target: Label column, ``"recovery_360"`` or ``"recovery_360_visible"``.
        test_fraction: Share of MATCHES held out.
        seed: Split seed.
        jobs: Worker processes for dataset assembly. Does not affect the result:
            :func:`build_dataset` returns a canonical row order.
        download: Fetch anything not already cached, at one request per second.
        verbose: Print progress and the comparison table.

    Returns:
        ``(refit_model, split_model, comparison_table, dataset)``.

    Raises:
        ValueError: If no labelled rows survive.
    """
    listing = matches_with_360(cache_dir, competitions)
    match_ids = listing["match_id"].tolist()
    if download:
        match_ids = download_training_set(match_ids, cache_dir, verbose=verbose)

    dataset = build_dataset(match_ids, cache_dir, jobs=jobs, verbose=verbose)
    dataset = dataset.merge(
        listing[["match_id", "competition", "season"]], on="match_id", how="left"
    )
    labelled = dataset[dataset["labelled"].fillna(False)].reset_index(drop=True)
    if labelled.empty:
        raise ValueError("no labelled turnovers; is the 360 cache populated?")

    train_mask, test_mask = split_by_match(labelled, test_fraction, seed)
    train_ids = sorted(int(v) for v in labelled.loc[train_mask, "match_id"].unique())
    test_ids = sorted(int(v) for v in labelled.loc[test_mask, "match_id"].unique())
    common: dict[str, Any] = {
        "competitions": [
            f"{c} {s}" for c, s in
            listing[["competition", "season"]].drop_duplicates().to_numpy().tolist()
        ],
        "n_matches": int(labelled["match_id"].nunique()),
        "n_turnovers": int(len(dataset)),
        "n_labelled": int(len(labelled)),
        "target": target,
        "test_fraction": test_fraction,
        "seed": seed,
        "train_match_ids": train_ids,
        "held_out_match_ids": test_ids,
        "n_train_rows": int(train_mask.sum()),
        "n_held_out_rows": int(test_mask.sum()),
        "label_phase_means": {
            str(k): float(v)
            for k, v in labelled.groupby("phase")[target].mean().items()
        },
    }

    split_model = train(
        labelled[train_mask],
        labelled.loc[train_mask, target],
        target_name=target,
        metadata={
            **common,
            "fitted_on": "train_matches",
            "in_sample_warning": (
                "fitted on train_match_ids only; score_held_out() will score this one."
            ),
        },
    )
    table = compare_models(labelled, labelled[target], train_mask, test_mask, split_model)
    if verbose:
        print(table.round(4).to_string(index=False), flush=True)

    model = train(
        labelled,
        labelled[target],
        target_name=target,
        metadata={
            **common,
            "fitted_on": "all_labelled",
            "in_sample_warning": (
                "fitted on EVERY labelled row, including held_out_match_ids. Scoring this "
                "artefact on those matches is in-sample and will flatter it. Use the "
                "'split' artefact and score_held_out()."
            ),
            "held_out_metrics": table.to_dict(orient="records"),
        },
    )
    split_model.metadata["held_out_metrics"] = table.to_dict(orient="records")
    model.metadata["predicted_phase_means"] = {
        str(k): float(v)
        for k, v in pd.Series(model.predict(labelled)).groupby(
            labelled["phase"].to_numpy()
        ).mean().items()
    }
    return model, split_model, table, dataset


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point: ``python -m pressing.proxy``.

    Args:
        argv: Arguments, or None to read ``sys.argv``.

    Returns:
        Process exit code.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m pressing.proxy",
        description="Train the event-only counter-press proxy on StatsBomb 360.",
    )
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--out", default=None, help="Refit artefact path.")
    parser.add_argument(
        "--out-split", default=None,
        help="Split-model artefact path. This is the model the metrics describe.",
    )
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Fetch missing matches at one request per second (about 10 MB each).",
    )
    args = parser.parse_args(argv)

    model, split_model, _, _ = train_from_cache(
        args.cache_dir,
        target=args.target,
        test_fraction=args.test_fraction,
        seed=args.seed,
        jobs=args.jobs,
        download=args.download,
    )
    for fitted, out, variant in (
        (model, args.out, "refit"),
        (split_model, args.out_split, "split"),
    ):
        path = save_model(fitted, out if out is not None else bundled_model_path(variant))
        print(f"wrote {path} ({path.stat().st_size} bytes, fitted_on="
              f"{fitted.metadata['fitted_on']})")
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
