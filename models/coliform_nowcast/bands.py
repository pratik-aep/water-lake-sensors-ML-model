"""Which CPCB coliform band a count (or a nowcast) falls in."""

import numpy as np

BANDS = ["A (<=50)", "B (<=500)", "C (<=5000)", "above C"]
LIMITS = [50.0, 500.0, 5000.0]


def band_of_count(count) -> np.ndarray:
    return np.array(BANDS, dtype=object)[np.searchsorted(LIMITS, np.asarray(count, dtype=float), side="left")]


def band_of_nowcast(p_above: dict) -> np.ndarray:
    """The cleanest band whose limit the count more likely than not stays within."""
    n = len(next(iter(p_above.values())))
    band = np.full(n, BANDS[-1], dtype=object)
    for limit, name in reversed(list(zip(LIMITS, BANDS))):
        band = np.where(1 - np.asarray(p_above[limit], dtype=float) >= 0.5, name, band)
    return band
