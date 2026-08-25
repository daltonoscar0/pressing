"""Event-only proxy for the tracking-derived counter-press metric.

Most matches have event data and no tracking. This module trains a gradient
boosted model that predicts the tracking-derived counter-press value from event
features alone -- turnover location, time to the next event, the next three
actions, and pressure flags -- so the same report can be produced from an event
file, watermarked as an estimate.

Training uses the Metrica and SkillCorner overlap. Splits are by match, never by
row, so no match contributes to both train and test. The correlation achieved on
held-out matches is reported in the README as measured, not as hoped for.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ProxyMetrics:
    """Held-out performance of the proxy model.

    Attributes:
        r: Pearson correlation between predicted and tracking-derived values.
        mae: Mean absolute error on the held-out matches.
        n_losses: Number of scored losses in the held-out set.
        n_matches: Number of held-out matches.
    """

    r: float
    mae: float
    n_losses: int
    n_matches: int


def build_features(events: pd.DataFrame) -> pd.DataFrame:
    """Derive proxy model features from an events frame.

    Features per turnover: location, distance and angle to goal, time to the
    next event, the types of the next three actions, and pressure flags on
    those actions.

    Args:
        events: Canonical events frame.

    Returns:
        One row per turnover, with a ``match_id`` column retained so splits can
        be made by match.
    """
    raise NotImplementedError


def train(features: pd.DataFrame, targets: pd.Series, group_col: str = "match_id") -> object:
    """Fit the gradient boosted proxy model.

    Args:
        features: Feature frame from :func:`build_features`.
        targets: Tracking-derived counter-press values, aligned to ``features``.
        group_col: Column identifying the match, used to keep splits leak-free.

    Returns:
        The fitted estimator.
    """
    raise NotImplementedError


def evaluate(model: object, features: pd.DataFrame, targets: pd.Series) -> ProxyMetrics:
    """Score the proxy model on held-out matches.

    Args:
        model: Fitted estimator from :func:`train`.
        features: Held-out feature frame.
        targets: Held-out tracking-derived values.

    Returns:
        Measured held-out metrics.
    """
    raise NotImplementedError


def predict(model: object, events: pd.DataFrame) -> pd.DataFrame:
    """Estimate counter-press values for an event-only match.

    Args:
        model: Fitted estimator from :func:`train`, or the bundled default.
        events: Canonical events frame for the match.

    Returns:
        One row per turnover with an estimated ``recovery`` and ``epv_conceded``.
    """
    raise NotImplementedError


def load_bundled_model(path: Path | None = None) -> object:
    """Load the model shipped with the package.

    Args:
        path: Optional override path to a serialized model.

    Returns:
        The fitted estimator.
    """
    raise NotImplementedError
