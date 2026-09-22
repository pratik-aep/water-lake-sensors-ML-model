"""Layer 1: QARTOD-style rule tests on one station's readings for one sensor."""

import numpy as np
import pandas as pd

from ..wqi.config import VALID_RANGES

QC_PASS, QC_SUSPECT, QC_FAIL, QC_MISSING = 1, 3, 4, 9
# When several tests fire on a reading, the reported reason is the first match in this order.
REASONS = ["missing", "out_of_range", "flatline", "spike", "rate_of_change", "climatology"]
# Tests whose pattern points at the sensor itself, whatever the water is doing.
SHAPE_FAULTS = {"out_of_range", "flatline", "spike"}
LOG_SENSORS = {"turbidity"}


def to_working(values: pd.Series, sensor: str) -> pd.Series:
    # Turbidity noise grows with its level; on log1p(NTU) the noise is roughly constant.
    return np.log1p(values.clip(lower=0)) if sensor in LOG_SENSORS else values


def from_working(values, sensor: str):
    return np.expm1(values) if sensor in LOG_SENSORS else values


def _run_length(same: pd.Series) -> pd.Series:
    return same.astype(int).groupby((~same).cumsum()).cumsum()


def rule_tests(
    raw: pd.Series, sensor: str, step_scale: float, clim_low, clim_high, settings: dict
) -> tuple[pd.Series, pd.Series]:
    """QC flag and reason per reading; expects one station's readings on a regular grid."""
    low, high = VALID_RANGES[sensor]
    missing = raw.isna()
    out_of_range = raw.notna() & ~raw.between(low, high)

    repeats = _run_length(raw.diff().abs() <= settings["resolution"][sensor]) + 1
    flat_suspect = repeats >= settings["flat_suspect_points"]
    flat_fail = repeats >= settings["flat_fail_points"]

    working = to_working(raw.mask(out_of_range), sensor)
    median = working.rolling(settings["spike_window"], center=True, min_periods=3).median()
    deviation = (working - median).abs() / step_scale
    spike_suspect = deviation >= settings["spike_suspect"]
    spike_fail = deviation >= settings["spike_fail"]
    rate_of_change = working.diff().abs() / step_scale >= settings["rate_of_change_limit"]
    if clim_low is None:
        climatology = pd.Series(False, index=raw.index)
    else:
        climatology = (working < clim_low) | (working > clim_high)

    fired = {
        "missing": missing,
        "out_of_range": out_of_range,
        "flatline": flat_suspect,
        "spike": spike_suspect,
        "rate_of_change": rate_of_change,
        "climatology": climatology,
    }
    reason = pd.Series("", index=raw.index)
    for name in reversed(REASONS):
        reason = reason.mask(fired[name], name)

    flag = pd.Series(QC_PASS, index=raw.index)
    flag = flag.mask(climatology | rate_of_change | spike_suspect | flat_suspect, QC_SUSPECT)
    flag = flag.mask(out_of_range | spike_fail | flat_fail, QC_FAIL)
    flag = flag.mask(missing, QC_MISSING)
    return flag, reason
