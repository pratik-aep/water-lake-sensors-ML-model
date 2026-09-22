"""WQI parameter standards, class bands and model feature sets."""

# Weighted Arithmetic WQI (Brown et al., 1972). `si` = permissible standard, `ideal` = value in pure water.
# Standards are the IS 10500:2012 / ICMR values commonly used in Indian WAWQI studies.
PARAMETERS = {
    "ph": {"si": 8.5, "ideal": 7.0, "lower_si": 6.5},
    "dissolved_oxygen": {"si": 5.0, "ideal": 14.6},  # mg/L
    "turbidity": {"si": 5.0, "ideal": 0.0},  # NTU
    "bod": {"si": 5.0, "ideal": 0.0},  # mg/L
    "conductivity": {"si": 300.0, "ideal": 0.0},  # µS/cm
    "nitrate": {"si": 45.0, "ideal": 0.0},  # mg/L
}

# Inclusive upper WQI bound per class, cleanest first (Brown et al., 1972 scale).
WQI_CLASSES = [
    (25.0, "Excellent"),
    (50.0, "Good"),
    (75.0, "Poor"),
    (100.0, "Very Poor"),
    (float("inf"), "Unsuitable"),
]
CLASS_ORDER = [label for _, label in WQI_CLASSES]

# Physically plausible bounds for lake readings; anything outside is a sensor or data-entry fault, not pollution.
VALID_RANGES = {
    "temperature": (0.0, 45.0),
    "ph": (0.0, 14.0),
    "turbidity": (0.0, 4000.0),
    "dissolved_oxygen": (0.0, 25.0),
    "bod": (0.0, 500.0),
    "conductivity": (0.0, 20000.0),
    "nitrate": (0.0, 500.0),
}

# A WQI parameter can carry a separate reference measurement, e.g. `lab_ph` next to the sensor's `ph`.
# The label is computed from the reference value; the model still learns from the sensor reading.
LAB_PREFIX = "lab_"

FEATURE_SETS = {
    # What the lake station senses continuously. Level is left out: it is hydrological, not a quality parameter.
    "sensor": ["temperature", "ph", "turbidity", "dissolved_oxygen"],
    # Sensor readings plus the lab-sampled parameters the WQI label is computed from.
    "full": ["temperature", "ph", "turbidity", "dissolved_oxygen", "bod", "conductivity", "nitrate"],
}
