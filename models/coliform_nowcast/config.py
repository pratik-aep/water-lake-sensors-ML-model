"""Settings for the total coliform nowcast."""

from ..bod_surrogate.config import DEFAULT_SETTINGS as SOFT_SENSOR_SETTINGS

TARGET = "total_coliform"  # MPN/100 mL from the lab's multiple-tube (or equivalent) test

DEFAULT_SETTINGS = {
    **SOFT_SENSOR_SETTINGS,
    "limits": (0.0, 1e8),
    # CPCB total coliform limits: Class A 50, Class B (bathing) 500, Class C (treated drinking source) 5000.
    "exceedance_levels": [50.0, 500.0, 5000.0],
}
