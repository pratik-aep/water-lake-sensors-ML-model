"""Conformalized quantile regression: stretch or shrink predicted ranges until they hold their promised coverage."""

import numpy as np


def horizon_bucket(horizons, buckets) -> np.ndarray:
    labels = np.full(len(horizons), "", dtype=object)
    for lo, hi in buckets:
        labels[(horizons >= lo) & (horizons <= hi)] = f"{lo}-{hi}h"
    return labels


# Fewer verified forecasts than this in a group and its offset would be one or two outcomes' luck: the group keeps
# the calibration it already has.
MIN_CALIBRATION_SAMPLES = 30


def fit_offsets(lo, hi, actual, groups, coverage: float, min_count: int = MIN_CALIBRATION_SAMPLES) -> dict:
    """Per group with at least `min_count` verified forecasts, the amount to widen [lo, hi] so it covers `coverage`
    of calibration outcomes (Romano et al. 2019)."""
    offsets = {}
    # Only forecasts that were actually made and verified: a gap at the origin (no current reading, e.g. a
    # reading the anomaly detector removed) leaves no range, and one missing score would blank the whole group.
    known = ~np.isnan(actual) & ~np.isnan(lo) & ~np.isnan(hi)
    for group in np.unique(groups[known]):
        m = known & (groups == group)
        if m.sum() < min_count:
            continue
        scores = np.maximum(lo[m] - actual[m], actual[m] - hi[m])
        level = min(1.0, np.ceil((m.sum() + 1) * coverage) / m.sum())
        offsets[group] = float(np.quantile(scores, level, method="higher"))
    return offsets


def widen(lo, mid, hi, groups, offsets: dict):
    q = np.array([offsets.get(g, 0.0) for g in groups])
    return np.minimum(lo - q, mid), mid, np.maximum(hi + q, mid)


def prob_below(threshold, lo, mid, hi, levels=(0.1, 0.5, 0.9)) -> np.ndarray:
    """P(value < threshold) from a piecewise-linear distribution through the three quantiles, extended linearly."""
    p_lo, p_mid, p_hi = levels
    lower_slope = (p_mid - p_lo) / np.maximum(mid - lo, 1e-9)
    upper_slope = (p_hi - p_mid) / np.maximum(hi - mid, 1e-9)
    p = np.where(threshold <= mid, p_mid - (mid - threshold) * lower_slope, p_mid + (threshold - mid) * upper_slope)
    return np.clip(p, 0.0, 1.0)
