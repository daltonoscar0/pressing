"""The conditional baseline: what recovery a loss in that place, with that
shape, was worth before anybody pressed.

Why this module exists
----------------------
``recovery`` is a control share, and a control share is mostly geography. Lose
the ball on the opponent's byline and your whole team is already upfield around
it; lose it on your own edge-of-box and the opponent owns everything in front
of you. Measured on Metrica game 1, over the 255 scored losses, a straight line
in the attacking fraction of the turnover explains ``R2 = 0.8212`` of the
loss-level variance, and averaged up to the 27 players with at least one
credited loss, a cubic in that same fraction predicts the printed
``Recovery 5 s`` column at ``r = +0.9771`` (``R2 = 0.9547``, Spearman
``+0.9402``). The player's own mean turnover position, with nothing else in the
model, reaches ``R2 = 0.9583``. Over only the 22 rows that print a value the
cubic still reaches ``r = +0.9691`` (``R2 = 0.9391``).

So a per-player *level* is a map of where that player stands. The only number
with a chance of being about pressing is a **residual**::

    recovery_residual = recovery - recovery_expected

where ``recovery_expected`` is what a team losing the ball *there*, *in that
shape*, would have got. This module fits that expectation and hands back both
columns.

What goes into the expectation
------------------------------
Two blocks, both read off the tracking frame at ``t0`` and both about the
counter-pressing team -- which is the team that just lost the ball.

**Position** (:data:`POSITION_FEATURES`) -- the attacking fraction ``x_norm``
of the turnover, entered as a polynomial of degree
:attr:`BaselineSpec.position_degree`, plus the normalised width
``width_norm`` and its square when :attr:`BaselineSpec.use_width` is set. A
wide loss is not the same problem as a central one at the same ``x``.

**Shape** (:data:`SHAPE_FEATURES`) -- five things a coach would actually call
shape, all computable from two matches of tracking:

``n_goalside``
    Counter-pressing outfielders between the ball and their own goal.
``ball_to_centroid_m``
    Distance from the ball to the counter-pressing team's outfield centroid.
``spread_m``
    Root-mean-square distance of those outfielders from that centroid --
    compactness, low is compact.
``n_within_10m``
    Counter-pressing outfielders inside
    :data:`DEFAULT_NEAR_RADIUS_M` of the ball. The swarm.
``line_height``
    Attacking fraction of the :data:`DEFAULT_LINE_PLAYERS`-th deepest
    counter-pressing outfielder -- the defensive line.

The goalkeeper is excluded from every one of them. A keeper standing on his
line drags a centroid twenty metres and tells you nothing about a press.

Does shape earn its place? Measured, not assumed
------------------------------------------------
Cross-validated **by match** -- fit on Metrica game 1, predict game 2, and the
reverse, so no match is ever fitted and scored together -- over the 487 scored
losses of the two matches. R2 is against the grand mean of the scored rows:

============================================  =========  =============
model                                         in-sample  out-of-sample
============================================  =========  =============
mean only (predict the other match's mean)    0.0000     -0.0008
``x_norm`` linear                             0.8289     0.8233
``x_norm`` cubic                              0.8399     0.8352
``x_norm`` cubic + width                      0.8418     0.8368
``x_norm`` cubic + width + all five shape     0.9277     0.9237
:data:`DEFAULT_SPEC` (the four of them)       0.9277     0.9240
============================================  =========  =============

So shape earns ``+0.0872`` of out-of-sample R2 over position and width. It is
not decoration. Dropping one term at a time from :data:`DEFAULT_SPEC` costs
0.0103 for ``line_height``, 0.0086 for ``spread_m``, 0.0018 for ``n_goalside``
and 0.0006 for ``ball_to_centroid_m``, so two of the four carry nearly all of
it and the biggest single thing beyond the ball's own position is where the
team's defensive line was. The in-sample-to-out-of-sample gap is 0.0037 on nine
terms and 487 rows, so the model is not straining.

None of that should be mistaken for insight. A control share is a function of
where twenty-two people are standing, so summary statistics of where they are
standing predict it well almost by construction. The right way to read
``R2 = 0.9240`` is: **the five-second recovery number is very nearly the frozen
instant it starts from.** Measured directly, ``r(recovery, share_t0_area) =
+0.9740`` over the same 487 losses. The residual is what the five seconds add
to the geometry -- which is the only part that could be pressing.

Width, for the record, earns ``+0.0015`` on top of position alone and
``+0.0005`` on top of the full model. Wide losses are not meaningfully
different from central ones once ``x`` and the shape are known. It is kept
because it costs nothing and because the question was worth answering.

The permutation test is the point
---------------------------------
Residualising is only worth doing if the residual carries between-player signal
that the raw column does not. :func:`permutation_between_player` runs the
5,000-shuffle test the audit ran: hold every player's loss count fixed, shuffle
the values across losses, and ask how often chance spreads the per-player means
as far apart as reality did.

Every p below is a Monte-Carlo estimate, so it is only reproducible with its
recipe stated. All of them: ``min_count=5``, ``n_shuffles=5000``, and a fresh
``numpy.random.default_rng(0)`` for each row of each table. A different seed
moves these p-values by about ±0.01 and moves none of the conclusions.

**Metrica game 1 alone**, on ``data/demo/metrica_game_1_tracking.parquet`` --
the asset the report itself reads, 805 credited presser-losses over 255 scored
losses. Values are shuffled across all losses, not within a squad. The residual
here is the **in-sample** one: with a single match in the frame
:func:`add_expected_recovery` fits and scores on the same rows and sets
``residual_out_of_sample`` to ``False``, and that is the number the report
computes for its own match.

=========================  =========  ===========  ===========  =======
column                     players    observed sd  null median  p
=========================  =========  ===========  ===========  =======
``recovery`` pooled        26         0.0581       0.0415       0.0126
``recovery`` build-up      14         0.0174       0.0221       0.8806
``recovery`` middle        21         0.0355       0.0289       0.1200
``recovery`` final third   19         0.0336       0.0279       0.1662
residual pooled            26         0.0109       0.0110       0.5167
residual build-up          14         0.0134       0.0197       0.9638
residual middle            21         0.0158       0.0143       0.2785
residual final third       19         0.0137       0.0152       0.7073
=========================  =========  ===========  ===========  =======

Up to v0.1.1 those residual rows read 0.0115 / 0.0112 / 0.4371 and so on. Those
were not the in-sample residual at all: they were the **two-match out-of-fold**
residual restricted to game-1 rows, which is a different fit under a caption
saying "the report's own view of its own match". For the record, that fit gives
pooled 0.0116 / 0.0112 / ``p = 0.4303``, build-up 0.0154 / 0.0196 / 0.8810,
middle 0.0154 / 0.0147 / 0.4029, final third 0.0141 / 0.0154 / 0.6715. Both fits
say noise; only one of them is the one the report runs.

The report does not print the unstratified pooled p either. It shuffles inside a
squad, so that a difference between the two teams cannot arrive as a difference
between players: on the same match and the same in-sample residual that is
``p = 0.6643``, against 0.5167 unstratified.

**Both matches together**, from the raw Metrica CSVs, 49 player-match units,
shuffling only within a team and a match so that neither a fixture effect nor a
team effect can masquerade as a player one. The residual here is out-of-fold by
construction -- each match is predicted by a model fitted on the other:

=========================  =========  ===========  ===========  =======
column                     players    observed sd  null median  p
=========================  =========  ===========  ===========  =======
``recovery`` pooled        49         0.0508       0.0447       0.0976
``recovery`` build-up      33         0.0222       0.0293       0.9960
``recovery`` middle        40         0.0374       0.0389       0.6747
``recovery`` final third   40         0.0409       0.0277       0.0004
residual pooled            49         0.0144       0.0134       0.1902
residual build-up          33         0.0224       0.0262       0.9984
residual middle            40         0.0155       0.0181       0.9556
residual final third       40         0.0165       0.0184       0.8676
=========================  =========  ===========  ===========  =======

Read those two tables honestly, because they say the same thing twice.

The three per-phase columns -- the comparison the report tells a coach to make
-- do not clear the test. In build-up the real spread between players is
*smaller* than chance, in both matches and by a wide margin. The one column
that does clear it is ``recovery`` in the final third across both matches, at
``p = 0.0004``; residualise it and it evaporates to ``p = 0.8676``. That is the
audit's whole point restated by the fix: what separates players in that column
is where they were standing, and nothing else.

And the residual, the number this module exists to compute, separates players at
``p = 0.5167`` on the report's own match -- ``p = 0.6643`` shuffled inside a
squad, which is the figure the report prints -- and at ``p = 0.1902`` on both
matches pooled. **There is no per-player counter-pressing effect detectable in
this data.** The residual is the honest number and the honest number is noise.
Print it with its interval, or do not print it.

The minimum count
-----------------
The rule, stated before the arithmetic: *print a per-player value only when its
95% bootstrap interval is narrower than the whole spread of the column it sits
in.* An interval wider than the table's entire range orders nobody against
anybody; it is a decoration with three decimal places.

The two matches yield **1,604** player-loss credits (806 in game 1, 798 in game
2) over 53 player-match units. The per-loss sd below is measured over all 1,604.
The spread of the means is measured over the **1,595** credits belonging to the
49 units with at least five, because a unit with one credit has a mean that is
that credit and would widen the range without carrying any information:

=================  ============  ==============  ================  ==========
column             per-loss sd   range of means  needed for range  for one sd
=================  ============  ==============  ================  ==========
``recovery``       0.1978        0.2432          10.2 -> 11        233.0
residual           0.0511        0.0703          8.1 -> 9          193.9
=================  ============  ==============  ================  ==========

"Needed for range" is ``(2 * 1.96 * per-loss sd / range of means) ** 2``, the
count at which the interval's full width drops below the range; "for one sd"
puts the between-unit sd (0.0508 and 0.0144) in the denominator instead.
Restricting the per-loss sd to the same 1,595 credits changes nothing that
matters: 10.1 -> 11, 232.2, 8.1 -> 9, 193.4.

:data:`MIN_LOSSES_FOR_PLAYER` is therefore **11**, the stricter of the two, so
one threshold covers both columns. On Metrica game 1 the report prints 28 rows,
14 per squad, of which the threshold lets 22 show a value and suppresses 6.
(27 of the 28 carry at least one credited loss -- 13 home, 14 away -- and that
27 is the population the position-dominance figure above is measured over, not
the number of printed rows.)

The right-hand column is the one to dwell on. Separating two *typical* players
-- one between-player standard deviation apart -- needs about 233 nearby losses
for the recovery column and 194 for the residual. The median player in Metrica
game 1 gets 24. That is roughly **eight to ten full matches per player** before
a per-player counter-press number means anything, and it is why
:func:`reportable_mask` is a suppression rule and not a confidence rule. Eleven
losses buys the right to be printed next to an interval. It does not buy the
right to be believed.

Using it
--------
::

    features = loss_features(losses, tracking)
    scored = add_expected_recovery(losses, features)
    scored[["recovery", "recovery_expected", "recovery_residual"]]

With one match in the frame the model is fitted on that match and the residual
is in-sample; ``scored.attrs["residual_out_of_sample"]`` says ``False`` and the
report must say so too. With two or more matches the residual is out-of-fold
by construction and the flag says ``True``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd

from pressing.schema import PITCH_LENGTH, PITCH_WIDTH

__all__ = [
    "ALL_FEATURES",
    "BaselineCV",
    "BaselineModel",
    "BaselineSpec",
    "DEFAULT_LINE_PLAYERS",
    "DEFAULT_NEAR_RADIUS_M",
    "DEFAULT_SPEC",
    "FEATURE_COLUMNS",
    "MIN_LOSSES_FOR_PLAYER",
    "POSITION_FEATURES",
    "PermutationResult",
    "RESIDUAL_COLUMNS",
    "SHAPE_FEATURES",
    "add_expected_recovery",
    "bootstrap_mean_ci",
    "crossval_by_match",
    "fit_baseline",
    "loss_features",
    "permutation_between_player",
    "reportable_mask",
]

#: Radius in metres for ``n_within_10m``. Ten metres is roughly the distance a
#: presser can close inside the first second of the window, which is what makes
#: the count a shape feature rather than a restatement of the ball position.
DEFAULT_NEAR_RADIUS_M: Final[float] = 10.0

#: How many deep outfielders define the defensive line for ``line_height``. Four
#: is the flat back four; the feature is the attacking fraction of the fourth
#: deepest, so a back three plus a dropping midfielder reads the same way.
DEFAULT_LINE_PLAYERS: Final[int] = 4

#: Position block. ``x_norm`` enters as a polynomial; the rest enter raw.
POSITION_FEATURES: Final[tuple[str, ...]] = ("x_norm", "width_norm")

#: Every shape feature :func:`loss_features` computes, all measured on the
#: counter-pressing team's outfielders at t0. Not all of them are in
#: :data:`DEFAULT_SPEC` -- see :data:`DEFAULT_SHAPE`.
SHAPE_FEATURES: Final[tuple[str, ...]] = (
    "n_goalside",
    "ball_to_centroid_m",
    "spread_m",
    "n_within_10m",
    "line_height",
)

#: The four shape features the shipped baseline actually uses.
#:
#: ``n_within_10m`` is computed and then deliberately left out, on two grounds
#: that agree. Measured: dropping it RAISES out-of-sample R2 from 0.9237 to
#: 0.9240, so it earns nothing. Argued: it counts the players within ten metres
#: of the ball, which is very nearly the set of players the per-player table
#: credits with the loss, so conditioning on it would subtract the swarming the
#: residual is supposed to detect.
DEFAULT_SHAPE: Final[tuple[str, ...]] = (
    "n_goalside",
    "ball_to_centroid_m",
    "spread_m",
    "line_height",
)

#: Every feature :func:`loss_features` emits, position block first.
ALL_FEATURES: Final[tuple[str, ...]] = (*POSITION_FEATURES, *SHAPE_FEATURES)

#: Everything :func:`loss_features` returns, features and keys together.
FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "loss_id",
    "match_id",
    "phase",
    "recovery",
    "scored",
    *ALL_FEATURES,
)

#: The two columns :func:`add_expected_recovery` adds to a scored-loss frame.
RESIDUAL_COLUMNS: Final[tuple[str, ...]] = ("recovery_expected", "recovery_residual")

#: Fewest nearby losses a player needs before a per-player recovery or residual
#: is worth printing at all.
#:
#: Derived, not chosen: it is the count at which a 95% bootstrap interval on the
#: player's mean becomes narrower than the full range of player means in the
#: same column, which is the least a number must manage before it can order
#: anybody against anybody. Measured over the 1,604 player-loss credits of the
#: two Metrica matches -- against the spread of means of the 49 player-match
#: units holding 1,595 of them -- that is 11 for ``recovery`` and 9 for the
#: residual; 11 is the stricter and covers both. See the module docstring,
#: including the 194 to 233 losses a player would actually need before the
#: number separates them from a typical teammate.
MIN_LOSSES_FOR_PLAYER: Final[int] = 11

_TEAMS: Final[tuple[str, str]] = ("home", "away")


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #


def _attacking_fraction(
    x: np.ndarray | float,
    direction: int,
    pitch_length: float,
) -> np.ndarray | float:
    """Vectorised twin of :func:`pressing.phases.attacking_fraction`.

    The one in :mod:`pressing.phases` takes a scalar, and this module needs a
    whole team's worth at once. The convention is identical: ``0`` at the
    team's own goal line, ``1`` at the goal it attacks.

    Args:
        x: Canonical metres along the pitch, scalar or array.
        direction: ``+1`` if the team attacks ``+x``, ``-1`` otherwise.
        pitch_length: Pitch length in metres.

    Returns:
        The attacking fraction, same shape as ``x``.

    Raises:
        ValueError: If ``direction`` is not ``+1`` or ``-1``.
    """
    if direction not in (1, -1):
        raise ValueError(f"direction must be +1 or -1, got {direction!r}")
    fraction = np.asarray(x, dtype=np.float64) / float(pitch_length)
    return fraction if direction == 1 else 1.0 - fraction


def _team_shape(
    px: np.ndarray,
    py: np.ndarray,
    ball_x: float,
    ball_y: float,
    direction: int,
    *,
    near_radius_m: float,
    line_players: int,
    pitch_length: float,
) -> dict[str, float]:
    """Shape features for one team at one instant.

    Args:
        px: Outfield x positions in canonical metres.
        py: Outfield y positions in canonical metres.
        ball_x: Ball x in canonical metres.
        ball_y: Ball y in canonical metres.
        direction: ``+1`` if this team attacks ``+x``, ``-1`` otherwise.
        near_radius_m: Radius for ``n_within_10m``.
        line_players: Rank of the deep player that defines ``line_height``.
        pitch_length: Pitch length in metres.

    Returns:
        One value per name in :data:`SHAPE_FEATURES`, all NaN when the team has
        no tracked outfielder at the instant.
    """
    blank = dict.fromkeys(SHAPE_FEATURES, float("nan"))
    finite = np.isfinite(px) & np.isfinite(py)
    px, py = px[finite], py[finite]
    if px.size == 0 or not (np.isfinite(ball_x) and np.isfinite(ball_y)):
        return blank

    u_players = np.asarray(_attacking_fraction(px, direction, pitch_length), dtype=np.float64)
    u_ball = float(np.asarray(_attacking_fraction(ball_x, direction, pitch_length)))
    cx, cy = float(px.mean()), float(py.mean())
    distance = np.hypot(px - cx, py - cy)
    to_ball = np.hypot(px - ball_x, py - ball_y)
    line_rank = min(line_players, px.size) - 1
    return {
        "n_goalside": float(np.count_nonzero(u_players < u_ball)),
        "ball_to_centroid_m": float(np.hypot(cx - ball_x, cy - ball_y)),
        "spread_m": float(np.sqrt(np.mean(distance**2))),
        "n_within_10m": float(np.count_nonzero(to_ball <= near_radius_m)),
        "line_height": float(np.sort(u_players)[line_rank]),
    }


def loss_features(
    losses: pd.DataFrame,
    tracking: pd.DataFrame,
    *,
    attack_direction: dict[str, int] | None = None,
    goalkeepers: dict[str, Any] | None = None,
    near_radius_m: float = DEFAULT_NEAR_RADIUS_M,
    line_players: int = DEFAULT_LINE_PLAYERS,
    pitch_length: float = PITCH_LENGTH,
    pitch_width: float = PITCH_WIDTH,
) -> pd.DataFrame:
    """Position and shape features for every loss, read at ``t0``.

    Only the few hundred frames the losses actually land on are touched, so
    this is a merge and a groupby, not a scan of the whole match.

    Args:
        losses: Scored losses from :func:`pressing.counterpress.score_losses`.
        tracking: Canonical tracking frame for the same match or matches.
        attack_direction: ``{"home": +1, "away": -1}``. Defaults to the
            tracking frame's ``attrs["attack_direction"]``, then to the
            canonical home-attacks-``+x`` convention.
        goalkeepers: ``{"home": player_id, "away": player_id}``. Defaults to
            the tracking frame's ``attrs["goalkeeper"]``. Keepers are dropped
            from every shape feature.
        near_radius_m: Radius for ``n_within_10m``.
        line_players: Rank of the deep player defining ``line_height``.
        pitch_length: Pitch length in metres.
        pitch_width: Pitch width in metres.

    Returns:
        One row per input loss with :data:`FEATURE_COLUMNS`, in the input's
        order. A loss whose frame is missing from the tracking keeps its row
        with NaN shape features.
    """
    direction = dict(
        attack_direction
        if attack_direction is not None
        else tracking.attrs.get("attack_direction", {"home": 1, "away": -1})
    )
    keepers = dict(goalkeepers if goalkeepers is not None else tracking.attrs.get("goalkeeper", {}))
    keeper_ids = {str(value) for value in keepers.values()}

    half_width = float(pitch_width) / 2.0
    out = pd.DataFrame(
        {
            "loss_id": losses["loss_id"].to_numpy(),
            "match_id": losses["match_id"].astype(str).to_numpy()
            if "match_id" in losses.columns
            else np.full(len(losses), str(tracking.attrs.get("match_id", "match"))),
            "phase": losses["phase"].astype("object").to_numpy(),
            "recovery": losses["recovery"].to_numpy(dtype=np.float64),
            "scored": losses["scored"].to_numpy(dtype=bool),
            "x_norm": losses["x_norm"].to_numpy(dtype=np.float64),
            "width_norm": np.abs(losses["y"].to_numpy(dtype=np.float64) - half_width)
            / half_width,
        }
    )
    for name in SHAPE_FEATURES:
        out[name] = np.nan

    wanted = losses.loc[:, ["loss_id", "frame", "losing_team", "x", "y"]]
    usable = wanted[
        np.isfinite(wanted["x"].to_numpy(dtype=np.float64))
        & np.isfinite(wanted["y"].to_numpy(dtype=np.float64))
        & wanted["losing_team"].isin(_TEAMS).to_numpy(dtype=bool)
    ]
    if usable.empty:
        return out.loc[:, list(FEATURE_COLUMNS)]

    frames = np.unique(usable["frame"].to_numpy(dtype=np.int64))
    rows = tracking.loc[
        tracking["team"].isin(_TEAMS).to_numpy(dtype=bool)
        & tracking["frame"].isin(frames).to_numpy(dtype=bool),
        ["frame", "team", "player_id", "x", "y"],
    ]
    rows = rows[~rows["player_id"].astype(str).isin(keeper_ids).to_numpy(dtype=bool)]
    if rows.empty:
        return out.loc[:, list(FEATURE_COLUMNS)]

    by_frame_team: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {
        (int(frame), str(team)): (
            block["x"].to_numpy(dtype=np.float64),
            block["y"].to_numpy(dtype=np.float64),
        )
        for (frame, team), block in rows.groupby(["frame", "team"], observed=True, sort=False)
    }

    index = {int(loss_id): position for position, loss_id in enumerate(out["loss_id"].to_numpy())}
    computed: dict[str, list[float]] = {name: [] for name in SHAPE_FEATURES}
    positions: list[int] = []
    for record in usable.to_dict("records"):
        team = str(record["losing_team"])
        px, py = by_frame_team.get(
            (int(record["frame"]), team), (np.empty(0), np.empty(0))
        )
        shape = _team_shape(
            px,
            py,
            float(record["x"]),
            float(record["y"]),
            int(direction.get(team, 1)),
            near_radius_m=near_radius_m,
            line_players=line_players,
            pitch_length=pitch_length,
        )
        positions.append(index[int(record["loss_id"])])
        for name in SHAPE_FEATURES:
            computed[name].append(shape[name])

    where = np.asarray(positions, dtype=np.int64)
    for name in SHAPE_FEATURES:
        column = out[name].to_numpy(dtype=np.float64).copy()
        column[where] = np.asarray(computed[name], dtype=np.float64)
        out[name] = column
    return out.loc[:, list(FEATURE_COLUMNS)]


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BaselineSpec:
    """Which features the expected-recovery model is allowed to use.

    Attributes:
        position_degree: Degree of the polynomial in ``x_norm``. One is a
            straight line, three is the cubic the audit used.
        use_width: Whether ``width_norm`` and its square join the position
            block.
        shape: Which of :data:`SHAPE_FEATURES` join the model. Empty is a
            position-only baseline.
        ridge: Ridge penalty on the standardised design matrix. The intercept
            is never penalised, which is what makes the fitting-set residuals
            exactly zero-mean.
    """

    position_degree: int = 3
    use_width: bool = True
    shape: tuple[str, ...] = DEFAULT_SHAPE
    ridge: float = 1e-3

    def __post_init__(self) -> None:
        """Validate the specification.

        Raises:
            ValueError: On a degree below one, a negative ridge penalty, or a
                shape feature that is not in :data:`SHAPE_FEATURES`.
        """
        if self.position_degree < 1:
            raise ValueError(f"position_degree must be at least 1, got {self.position_degree}")
        if self.ridge < 0:
            raise ValueError(f"ridge must be non-negative, got {self.ridge}")
        unknown = tuple(name for name in self.shape if name not in SHAPE_FEATURES)
        if unknown:
            raise ValueError(f"unknown shape features {unknown}, expected {SHAPE_FEATURES}")
        object.__setattr__(self, "shape", tuple(self.shape))

    @property
    def term_names(self) -> tuple[str, ...]:
        """Design-matrix column names, in order, excluding the intercept."""
        terms = [f"x_norm^{power}" for power in range(1, self.position_degree + 1)]
        if self.use_width:
            terms += ["width_norm", "width_norm^2"]
        return (*terms, *self.shape)


#: The shipped baseline: a cubic in position, width, and the four shape terms
#: of :data:`DEFAULT_SHAPE`. Measured at 0.9240 out-of-sample R2 against 0.8368
#: for the same model without the shape block -- see the module docstring.
DEFAULT_SPEC: Final[BaselineSpec] = BaselineSpec()


def _design(features: pd.DataFrame, spec: BaselineSpec) -> np.ndarray:
    """Build the design matrix for ``spec`` from a feature frame.

    Args:
        features: Output of :func:`loss_features`.
        spec: Which terms to include.

    Returns:
        A ``(n_rows, len(spec.term_names))`` float64 array.
    """
    x = features["x_norm"].to_numpy(dtype=np.float64)
    columns = [x**power for power in range(1, spec.position_degree + 1)]
    if spec.use_width:
        width = features["width_norm"].to_numpy(dtype=np.float64)
        columns += [width, width**2]
    columns += [features[name].to_numpy(dtype=np.float64) for name in spec.shape]
    return np.column_stack(columns) if columns else np.empty((len(features), 0))


@dataclass(frozen=True)
class BaselineModel:
    """A fitted expected-recovery model.

    Attributes:
        spec: The specification it was fitted to.
        term_names: Design-matrix column names, in coefficient order.
        coef: Ridge coefficients on the STANDARDISED design matrix.
        intercept: Fitted intercept, which equals the training mean of
            ``recovery`` because the standardised columns are centred.
        centre: Per-column training mean used to standardise.
        scale: Per-column training standard deviation used to standardise. A
            constant column gets a scale of one so it simply drops out.
        n_train: Rows the fit used.
        train_matches: Match ids the fit saw. :func:`crossval_by_match` checks
            this against the evaluation matches to prove there is no leakage.
        fill: Per-column training mean used for a missing feature value, so a
            loss with an untracked frame still scores at the training average
            rather than falling out of the table.
    """

    spec: BaselineSpec
    term_names: tuple[str, ...]
    coef: np.ndarray
    intercept: float
    centre: np.ndarray
    scale: np.ndarray
    n_train: int
    train_matches: tuple[str, ...]
    fill: np.ndarray

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Expected recovery for every row of ``features``.

        The prediction is NOT clipped to ``[0, 1]``. Clipping would break the
        zero-mean-residual guarantee, and an expectation that strays outside
        the unit interval is a fact about the fit worth seeing.

        Args:
            features: Output of :func:`loss_features`.

        Returns:
            A float64 array, one expected recovery per row.
        """
        design = _design(features, self.spec)
        missing = ~np.isfinite(design)
        if missing.any():
            design = np.where(missing, np.broadcast_to(self.fill, design.shape), design)
        standardised = (design - self.centre) / self.scale
        return standardised @ self.coef + self.intercept


def fit_baseline(
    features: pd.DataFrame,
    spec: BaselineSpec | None = None,
) -> BaselineModel:
    """Fit expected recovery on every scored row of ``features``.

    The fit is ridge with an unpenalised intercept on standardised columns, so
    the residuals on the fitting set sum to exactly zero. That is a property
    the tests pin, not an accident.

    Args:
        features: Output of :func:`loss_features`. Unscored rows and rows with
            a non-finite ``recovery`` are dropped from the fit.
        spec: Which terms to use. Defaults to :data:`DEFAULT_SPEC`.

    Returns:
        The fitted :class:`BaselineModel`.

    Raises:
        ValueError: If no scored row survives.
    """
    spec = DEFAULT_SPEC if spec is None else spec
    y_all = features["recovery"].to_numpy(dtype=np.float64)
    usable = features["scored"].to_numpy(dtype=bool) & np.isfinite(y_all)
    rows = features.loc[usable]
    if rows.empty:
        raise ValueError("cannot fit a baseline on zero scored losses")

    design = _design(rows, spec)
    y = y_all[usable]
    fill = np.array(
        [
            float(np.nanmean(column)) if np.isfinite(column).any() else 0.0
            for column in design.T
        ],
        dtype=np.float64,
    )
    design = np.where(~np.isfinite(design), np.broadcast_to(fill, design.shape), design)

    centre = design.mean(axis=0)
    scale = design.std(axis=0, ddof=0)
    scale = np.where(scale > 0, scale, 1.0)
    standardised = (design - centre) / scale

    width = standardised.shape[1]
    gram = standardised.T @ standardised + spec.ridge * np.eye(width)
    coef = np.linalg.solve(gram, standardised.T @ (y - y.mean())) if width else np.empty(0)
    matches = tuple(sorted({str(value) for value in rows["match_id"]}))
    return BaselineModel(
        spec=spec,
        term_names=spec.term_names,
        coef=coef,
        intercept=float(y.mean()),
        centre=centre,
        scale=scale,
        n_train=int(len(rows)),
        train_matches=matches,
        fill=fill,
    )


# --------------------------------------------------------------------------- #
# Cross-validation by match
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BaselineCV:
    """Leave-one-match-out results for a :class:`BaselineSpec`.

    Attributes:
        spec: What was cross-validated.
        prediction: Out-of-fold expected recovery, aligned to the input frame's
            rows. NaN wherever a row was not scored, or was the only match.
        r2_oos: Out-of-sample R2 of the pooled out-of-fold predictions against
            the grand mean of the scored rows. Negative means the baseline is
            worse than printing the overall average.
        r2_in: In-sample R2 of a single fit on everything, for the gap.
        fold_r2: Per-fold out-of-sample R2, keyed by the held-out match, each
            measured against its own TRAINING mean -- the number a forecaster
            standing before that match could have had. It is therefore not
            directly comparable with ``r2_oos``, which uses the grand mean.
        folds: ``(train_matches, test_match)`` for every fold. The tests assert
            the two never intersect.
        n_scored: Scored rows the cross-validation used.
    """

    spec: BaselineSpec
    prediction: np.ndarray
    r2_oos: float
    r2_in: float
    fold_r2: dict[str, float]
    folds: tuple[tuple[tuple[str, ...], str], ...]
    n_scored: int


def _r2(observed: np.ndarray, predicted: np.ndarray, baseline: float) -> float:
    """R2 of ``predicted`` against ``observed`` about a stated ``baseline``."""
    total = float(np.sum((observed - baseline) ** 2))
    if total <= 0:
        return float("nan")
    return float(1.0 - np.sum((observed - predicted) ** 2) / total)


def crossval_by_match(
    features: pd.DataFrame,
    spec: BaselineSpec | None = None,
) -> BaselineCV:
    """Leave-one-match-out cross-validation of the expected-recovery model.

    Splitting by match is the only split worth doing here. Losses inside one
    match share a pitch, two squads, a referee and a scoreline, so a random
    row split would let the model memorise the match and report a number
    nobody can use on the next one.

    Args:
        features: Output of :func:`loss_features`, covering two or more
            matches.
        spec: Which terms to use. Defaults to :data:`DEFAULT_SPEC`.

    Returns:
        A :class:`BaselineCV`.

    Raises:
        ValueError: If fewer than two matches carry a scored loss.
    """
    spec = DEFAULT_SPEC if spec is None else spec
    y_all = features["recovery"].to_numpy(dtype=np.float64)
    usable = features["scored"].to_numpy(dtype=bool) & np.isfinite(y_all)
    match_id = features["match_id"].astype(str).to_numpy()
    matches = sorted(set(match_id[usable]))
    if len(matches) < 2:
        raise ValueError(
            f"cross-validation by match needs at least two matches, got {matches}"
        )

    prediction = np.full(len(features), np.nan)
    folds: list[tuple[tuple[str, ...], str]] = []
    fold_r2: dict[str, float] = {}
    for held_out in matches:
        train = usable & (match_id != held_out)
        test = usable & (match_id == held_out)
        model = fit_baseline(features.loc[train], spec)
        if held_out in model.train_matches:
            raise AssertionError(
                f"fold leaked: {held_out} appears in the training matches {model.train_matches}"
            )
        predicted = model.predict(features.loc[test])
        prediction[test] = predicted
        folds.append((model.train_matches, held_out))
        fold_r2[held_out] = _r2(y_all[test], predicted, float(np.mean(y_all[train])))

    y = y_all[usable]
    full = fit_baseline(features.loc[usable], spec)
    return BaselineCV(
        spec=spec,
        prediction=prediction,
        r2_oos=_r2(y, prediction[usable], float(y.mean())),
        r2_in=_r2(y, full.predict(features.loc[usable]), float(y.mean())),
        fold_r2=fold_r2,
        folds=tuple(folds),
        n_scored=int(usable.sum()),
    )


def _loss_key(frame: pd.DataFrame) -> pd.Index:
    """The identity of a loss: ``"<match_id>#<loss_id>"``.

    ``loss_id`` restarts at zero in every match, so joining two matches on it
    alone silently produces a cross product. This is the key that does not.

    Args:
        frame: Anything carrying ``loss_id``, and ``match_id`` when it has one.

    Returns:
        A string index, one entry per row.
    """
    match = (
        frame["match_id"].astype(str)
        if "match_id" in frame.columns
        else pd.Series(["match"] * len(frame), index=frame.index)
    )
    return pd.Index(match.to_numpy(dtype=str) + "#" + frame["loss_id"].astype(str).to_numpy())


def add_expected_recovery(
    losses: pd.DataFrame,
    features: pd.DataFrame,
    spec: BaselineSpec | None = None,
) -> pd.DataFrame:
    """Attach ``recovery_expected`` and ``recovery_residual`` to a loss frame.

    With two or more matches present the expectation is out-of-fold: every
    loss is scored by a model that never saw its match, so the residual is
    honest. With one match there is nothing to hold out, the model is fitted on
    that match and the residual is in-sample; ``attrs["residual_out_of_sample"]``
    is False and any report built on it has to say so.

    Args:
        losses: Scored losses. Returned unmodified except for the two new
            columns.
        features: Output of :func:`loss_features` for the same losses, matched
            on ``(match_id, loss_id)``. Loss ids restart at zero in every match,
            so the pair is the key and ``loss_id`` alone is not.
        spec: Which terms to use. Defaults to :data:`DEFAULT_SPEC`.

    Returns:
        A copy of ``losses`` with :data:`RESIDUAL_COLUMNS` appended and
        ``attrs["residual_out_of_sample"]``, ``attrs["baseline_spec"]`` and
        ``attrs["baseline_r2"]`` set. Unscored losses get NaN in both columns.

    Raises:
        ValueError: If ``features`` does not cover every loss, or covers one
            twice.
    """
    spec = DEFAULT_SPEC if spec is None else spec
    order = features.set_index(_loss_key(features))
    if order.index.has_duplicates:
        raise ValueError("features contain duplicate (match_id, loss_id) pairs")
    wanted = _loss_key(losses)
    missing = wanted.difference(order.index)
    if len(missing):
        raise ValueError(
            f"features are missing {len(missing)} losses, e.g. {list(missing[:5])}"
        )
    aligned = order.loc[wanted].reset_index(drop=True)

    scored = aligned["scored"].to_numpy(dtype=bool) & np.isfinite(
        aligned["recovery"].to_numpy(dtype=np.float64)
    )
    n_matches = len({str(value) for value in aligned.loc[scored, "match_id"]})
    expected = np.full(len(aligned), np.nan)
    if n_matches >= 2:
        cv = crossval_by_match(aligned, spec)
        expected = cv.prediction
        out_of_sample = True
        r2 = cv.r2_oos
    else:
        model = fit_baseline(aligned, spec)
        expected[scored] = model.predict(aligned.loc[scored])
        out_of_sample = False
        y = aligned["recovery"].to_numpy(dtype=np.float64)[scored]
        r2 = _r2(y, expected[scored], float(y.mean()))

    out = losses.copy()
    out.attrs.update(losses.attrs)
    out["recovery_expected"] = expected
    out["recovery_residual"] = out["recovery"].to_numpy(dtype=np.float64) - expected
    out.attrs["residual_out_of_sample"] = out_of_sample
    out.attrs["baseline_spec"] = spec
    out.attrs["baseline_r2"] = float(r2)
    return out


# --------------------------------------------------------------------------- #
# Uncertainty and the print/suppress decision
# --------------------------------------------------------------------------- #


def bootstrap_mean_ci(
    values: np.ndarray | pd.Series,
    *,
    n_boot: int = 2000,
    level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of ``values``.

    Resampling a player's own losses with replacement is the right null here:
    it asks what this player's mean would have looked like had the match handed
    them a different draw of the same kind of turnovers. It says nothing about
    whether the *kind* was representative, which is a different and larger
    problem the module docstring covers.

    Args:
        values: The player's per-loss values. Non-finite entries are dropped.
        n_boot: Bootstrap resamples.
        level: Interval mass, e.g. ``0.95``.
        rng: Random generator, for reproducibility.

    Returns:
        ``(low, high)``, or ``(nan, nan)`` when nothing finite survives.

    Raises:
        ValueError: If ``level`` is not strictly inside ``(0, 1)`` or
            ``n_boot`` is below one.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    if n_boot < 1:
        raise ValueError(f"n_boot must be at least 1, got {n_boot}")
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return (float("nan"), float("nan"))
    if data.size == 1:
        return (float(data[0]), float(data[0]))
    generator = np.random.default_rng() if rng is None else rng
    draws = generator.integers(0, data.size, size=(n_boot, data.size))
    means = data[draws].mean(axis=1)
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return (float(low), float(high))


def reportable_mask(
    counts: np.ndarray | pd.Series,
    minimum: int = MIN_LOSSES_FOR_PLAYER,
) -> np.ndarray:
    """Which rows have enough losses behind them to print.

    Args:
        counts: Per-row loss counts.
        minimum: Threshold, inclusive. Defaults to
            :data:`MIN_LOSSES_FOR_PLAYER`.

    Returns:
        A boolean array, True where the value may be printed.
    """
    numbers = np.asarray(counts, dtype=np.float64)
    return np.isfinite(numbers) & (numbers >= float(minimum))


# --------------------------------------------------------------------------- #
# The permutation test
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PermutationResult:
    """Whether a column separates players by more than chance.

    Attributes:
        n_groups: Players in the test.
        n_values: Losses in the test.
        observed_sd: Standard deviation of the per-player means, as measured.
        null_median: Median of that statistic over the shuffles.
        null_p95: 95th percentile of the null.
        p_value: ``(1 + #{null >= observed}) / (1 + n_shuffles)``. The plus-one
            is the standard finite-sample correction; it can never report zero.
        n_shuffles: Shuffles run.
        group_counts: Losses behind each player, in group order.
    """

    n_groups: int
    n_values: int
    observed_sd: float
    null_median: float
    null_p95: float
    p_value: float
    n_shuffles: int
    group_counts: tuple[int, ...] = ()


def permutation_between_player(
    values: np.ndarray | pd.Series,
    groups: np.ndarray | pd.Series,
    *,
    strata: np.ndarray | pd.Series | None = None,
    n_shuffles: int = 5000,
    min_count: int = 5,
    rng: np.random.Generator | None = None,
) -> PermutationResult:
    """Test whether per-player means spread further apart than chance allows.

    The statistic is the standard deviation of the per-player means. The null
    shuffles the values across losses while holding every player's loss count
    fixed, which is exactly the "these players are interchangeable" hypothesis
    a coach implicitly rejects when they read the column.

    Pass ``strata`` -- the match id, normally -- whenever the values span more
    than one match. Without it the shuffle moves a value from one match into
    another, so any match-level offset in the column masquerades as
    between-player spread and the test reports a signal that is really a
    fixture. Measured on the two Metrica matches, whose mean out-of-fold
    residual per credited presser differs by 0.0103 (``-0.00474`` in game 1
    against ``+0.00552`` in game 2), the pooled residual test returns
    ``p = 0.0024`` unstratified, ``p = 0.0236`` stratified by match and
    ``p = 0.1902`` stratified by match and team -- one apparent finding and two
    honest nulls, from the same 5,000 shuffles, the same numbers and the same
    ``default_rng(0)``.

    Args:
        values: Per-loss values, e.g. ``recovery`` or ``recovery_residual``.
        groups: Player key per loss, the same length as ``values``.
        strata: Optional block label per loss. Values are only ever permuted
            within a block. A player must sit inside exactly one block.
        n_shuffles: Shuffles. 5,000 is what the audit used.
        min_count: Players below this many losses are excluded before the test,
            because a one-loss player's mean is that loss.
        rng: Random generator, for reproducibility.

    Returns:
        A :class:`PermutationResult`. When fewer than two players clear
        ``min_count`` the statistic is NaN and ``p_value`` is NaN.

    Raises:
        ValueError: If ``values``, ``groups`` or ``strata`` differ in length,
            if ``n_shuffles`` is below one, or if a player spans two strata.
    """
    if n_shuffles < 1:
        raise ValueError(f"n_shuffles must be at least 1, got {n_shuffles}")
    data = np.asarray(values, dtype=np.float64)
    keys = np.asarray(groups, dtype=object)
    if data.shape[0] != keys.shape[0]:
        raise ValueError(f"values and groups differ in length: {data.shape[0]} vs {keys.shape[0]}")
    blocks = np.zeros(data.shape[0], dtype=object) if strata is None else np.asarray(
        strata, dtype=object
    )
    if blocks.shape[0] != data.shape[0]:
        raise ValueError(
            f"values and strata differ in length: {data.shape[0]} vs {blocks.shape[0]}"
        )

    finite = np.isfinite(data)
    data, keys, blocks = data[finite], keys[finite], blocks[finite]
    codes, _ = pd.factorize(pd.Series(keys, dtype="object"))
    counts = np.bincount(codes, minlength=codes.max() + 1 if codes.size else 0)
    keep_group = counts >= min_count
    keep = keep_group[codes] if codes.size else np.zeros(0, dtype=bool)
    data, codes, blocks = data[keep], codes[keep], blocks[keep]
    codes, _ = pd.factorize(pd.Series(codes, dtype="object"))
    block_codes, _ = pd.factorize(pd.Series(blocks, dtype="object"))
    n_groups = int(codes.max() + 1) if codes.size else 0
    if n_groups >= 2:
        spanning = pd.DataFrame({"group": codes, "block": block_codes}).groupby("group")[
            "block"
        ].nunique()
        if int(spanning.max()) > 1:
            raise ValueError(
                f"{int((spanning > 1).sum())} players span more than one stratum; "
                "a player key must be unique to its block"
            )
    if n_groups < 2:
        return PermutationResult(
            n_groups=n_groups,
            n_values=int(data.size),
            observed_sd=float("nan"),
            null_median=float("nan"),
            null_p95=float("nan"),
            p_value=float("nan"),
            n_shuffles=n_shuffles,
            group_counts=tuple(int(value) for value in np.bincount(codes, minlength=n_groups)),
        )

    sizes = np.bincount(codes, minlength=n_groups).astype(np.float64)

    def spread(sample: np.ndarray) -> float:
        totals = np.bincount(codes, weights=sample, minlength=n_groups)
        return float((totals / sizes).std(ddof=1))

    observed = spread(data)
    generator = np.random.default_rng() if rng is None else rng
    slots = [np.flatnonzero(block_codes == block) for block in range(block_codes.max() + 1)]
    null = np.empty(n_shuffles, dtype=np.float64)
    shuffled = np.empty_like(data)
    for index in range(n_shuffles):
        for slot in slots:
            shuffled[slot] = generator.permutation(data[slot])
        null[index] = spread(shuffled)
    return PermutationResult(
        n_groups=n_groups,
        n_values=int(data.size),
        observed_sd=observed,
        null_median=float(np.median(null)),
        null_p95=float(np.percentile(null, 95)),
        p_value=float((np.count_nonzero(null >= observed) + 1) / (n_shuffles + 1)),
        n_shuffles=n_shuffles,
        group_counts=tuple(int(value) for value in sizes),
    )
