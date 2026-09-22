"""Trophic state (Carlson 1977) and WHO bloom alert level from chlorophyll-a."""

import math

import numpy as np
import pandas as pd

from .config import TROPHIC_CLASSES


def carlson_tsi(chlorophyll) -> np.ndarray:
    return 9.81 * np.log(np.asarray(chlorophyll, dtype=float)) + 30.6


def chlorophyll_at(tsi: float) -> float:
    """Chlorophyll-a (ug/L) at which the index reaches `tsi`."""
    return math.exp((tsi - 30.6) / 9.81)


def trophic_class(tsi) -> np.ndarray:
    tsi = np.asarray(tsi, dtype=float)
    labels = np.array([name for _, name in TROPHIC_CLASSES], dtype=object)
    bounds = [upper for upper, _ in TROPHIC_CLASSES[:-1]]
    return labels[np.searchsorted(bounds, tsi, side="right")]


def class_probabilities(sensor, X: pd.DataFrame) -> pd.DataFrame:
    """Chance of each trophic class, from the soft sensor's out-of-sample error distribution."""
    log_pred = sensor.predict_log(X)
    above = [sensor.p_above(log_pred, chlorophyll_at(upper)) for upper, _ in TROPHIC_CLASSES[:-1]]
    edges = [np.ones(len(X)), *above, np.zeros(len(X))]
    return pd.DataFrame(
        {f"p_{name}": edges[i] - edges[i + 1] for i, (_, name) in enumerate(TROPHIC_CLASSES)}, index=X.index
    )


def who_level(p_above_12, p_above_24, alert_probability: float) -> np.ndarray:
    level = np.where(np.asarray(p_above_12) >= alert_probability, "Alert Level 1", "Vigilance")
    return np.where(np.asarray(p_above_24) >= alert_probability, "Alert Level 1+ (check for scum)", level)
