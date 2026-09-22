"""Weighted Arithmetic Water Quality Index and dissolved-oxygen unit conversion."""

import numpy as np

from .config import PARAMETERS, WQI_CLASSES

_K = 1.0 / sum(1.0 / p["si"] for p in PARAMETERS.values())
WEIGHTS = {name: _K / p["si"] for name, p in PARAMETERS.items()}


def quality_rating(name: str, value: float) -> float:
    p = PARAMETERS[name]
    si, ideal = p["si"], p["ideal"]
    if value < ideal and "lower_si" in p:
        # pH is two-sided: acidic readings are rated against the lower limit instead of 8.5.
        si = p["lower_si"]
    return max(0.0, 100.0 * (value - ideal) / (si - ideal))


def compute_wqi(reading) -> float:
    return sum(WEIGHTS[name] * quality_rating(name, reading[name]) for name in PARAMETERS)


def classify_wqi(wqi: float) -> str:
    return next(label for upper, label in WQI_CLASSES if wqi <= upper)


def do_saturation_mgl(temp_c, pressure_atm=1.0):
    """Freshwater DO saturation in mg/L (Benson & Krause 1984, as used in APHA 4500-O)."""
    t = np.asarray(temp_c, dtype=float) + 273.15
    ln_c = -139.34411 + 1.575701e5 / t - 6.642308e7 / t**2 + 1.243800e10 / t**3 - 8.621949e11 / t**4
    # Linear pressure scaling skips the water-vapour term; error is <0.5% at Udaipur's ~600 m altitude.
    return np.exp(ln_c) * pressure_atm


def do_percent_to_mgl(percent, temp_c, pressure_atm=1.0):
    return np.asarray(percent, dtype=float) / 100.0 * do_saturation_mgl(temp_c, pressure_atm)
