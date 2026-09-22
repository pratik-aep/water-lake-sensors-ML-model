"""Settings for the chlorophyll-a soft sensor, trophic-state bands and WHO bloom alert levels."""

import math

from ..bod_surrogate.config import DEFAULT_SETTINGS as SOFT_SENSOR_SETTINGS

TARGET = "chlorophyll_a"  # ug/L, from lab extraction of grab samples

DEFAULT_SETTINGS = {
    **SOFT_SENSOR_SETTINGS,
    "limits": (0.0, 1000.0),
    # WHO (2021) recreational-water framework, for lakes where cyanobacteria dominate: 12-24 ug/L is Alert
    # Level 1; above 24 ug/L, look for scum or low transparency (Alert Level 2).
    "exceedance_levels": [12.0, 24.0],
}

# Carlson (1977) trophic state index from chlorophyll-a: TSI = 9.81 ln(Chl) + 30.6. Upper TSI bound per class.
TROPHIC_CLASSES = [(40.0, "oligotrophic"), (50.0, "mesotrophic"), (70.0, "eutrophic"), (math.inf, "hypereutrophic")]
