"""Air pollutants, the vendor's sensor ranges, and India's National Air Quality Index (CPCB, 2014)."""

# Columns after import. Particles in ug/m3; gases as the vendor reports them, in ppm.
PARTICLES = ["pm1", "pm25", "pm10", "tsp"]
GASES = ["o3", "no2", "co", "so2", "nox", "h2s", "co2", "voc"]
WEATHER_COLUMNS = ["air_temperature", "relative_humidity", "wind_speed", "boundary_layer_height", "rain_mm"]

# Molecular weights (g/mol), to turn ppm into the mass concentrations the index is defined in.
MOLECULAR_WEIGHT = {"o3": 48.00, "no2": 46.01, "co": 28.01, "so2": 64.07, "h2s": 34.08}
# Vendor spec: top of each gas module's range (ppm), where readings stop meaning "this much" and mean "at least".
SENSOR_MAX_PPM = {"o3": 0.4, "no2": 0.1, "co": 20.0, "so2": 10.0, "nox": 0.5, "h2s": 8.0, "co2": 2000.0, "voc": 20.0}
# Lower detectable limits (ppm): electrochemical cells read slightly negative in clean air.
DETECTION_LIMIT_PPM = {"o3": 0.001, "no2": 0.001, "co": 0.040, "so2": 0.009, "nox": 0.001, "h2s": 0.012, "co2": 10.0, "voc": 0.010}
PARTICLE_MAX = {"pm1": 170.0, "pm25": 2000.0, "pm10": 4500.0, "tsp": 5000.0}  # the optical particle profiler's ranges

# NAQI breakpoints: concentration band edges for index bands 0-50-100-200-300-400-500. Units: ug/m3, except CO in
# mg/m3. Averaging: 24 h, except CO and O3 (highest 8-hour mean). The open-ended "Severe" band is closed here at the
# width of the band below it, a common convention, and the index is capped at 500.
INDEX_EDGES = [0, 50, 100, 200, 300, 400, 500]
BREAKPOINTS = {
    "pm10": [0, 50, 100, 250, 350, 430, 510],
    "pm25": [0, 30, 60, 90, 120, 250, 380],
    "no2": [0, 40, 80, 180, 280, 400, 520],
    "o3": [0, 50, 100, 168, 208, 748, 1288],
    "co": [0, 1.0, 2.0, 10.0, 17.0, 34.0, 51.0],
    "so2": [0, 40, 80, 380, 800, 1600, 2400],
}
EIGHT_HOUR = {"co", "o3"}
CATEGORIES = [
    (50, "Good", "Minimal impact"),
    (100, "Satisfactory", "Minor breathing discomfort to sensitive people"),
    (200, "Moderate", "Breathing discomfort to people with lung or heart disease, children and older adults"),
    (300, "Poor", "Breathing discomfort to most people on prolonged exposure"),
    (400, "Very Poor", "Respiratory illness on prolonged exposure"),
    (500, "Severe", "Affects healthy people; serious impact on those with existing disease"),
]
MIN_HOURS_24H = 16  # CPCB: a 24-hour average needs at least 16 hourly values
MIN_HOURS_8H = 6
MIN_POLLUTANTS = 3  # an index needs three pollutants, one of them PM2.5 or PM10

DEFAULT_SETTINGS = {
    "min_readings_per_hour": 2,
    "flat_hours": 6,  # identical hourly values this long = a stuck sensor
    "spike_z": 8.0,  # robust z of an hour against its 24 h neighbourhood
    # Optical particle counters over-read in humid air as particles swell with water (kappa-Koehler growth,
    # Crilley et al. 2018). kappa ~0.4 suits mixed urban aerosol; above 95% RH the correction is not trusted.
    "hygroscopic_kappa": 0.4,
    "humidity_trust_limit": 95.0,
    # PM ordering allowance: PM1 <= PM2.5 <= PM10 <= TSP within sensor noise.
    "pm_order_tolerance": (5.0, 0.15),  # ug/m3 plus this share of the larger value
    # Forecast
    "targets": ["pm25", "pm10"],
    "horizons": list(range(1, 49)),
    "train_horizons": [1, 2, 3, 6, 9, 12, 18, 24, 30, 36, 42, 48],
    "train_origin_every_hours": 3,
    "eval_origin_every_hours": 6,
    "lags_hours": [1, 2, 3, 6, 12, 24],
    "quantiles": (0.1, 0.5, 0.9),
    "horizon_buckets": [(1, 6), (7, 24), (25, 48)],
    "gbm": {"max_iter": 300, "learning_rate": 0.06, "max_leaf_nodes": 31, "min_samples_leaf": 40, "l2_regularization": 1.0},
    "days_ahead": [1, 2],
    "alert_probability": 0.5,
    "alert_index": 200,  # warn when tomorrow's index is likely Poor or worse
    "calibration_days": 30,
    "test_days": 60,
}
