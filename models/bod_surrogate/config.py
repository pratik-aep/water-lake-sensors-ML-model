"""Settings for the BOD soft sensor, and the CPCB class criteria it checks against."""

DEFAULT_SETTINGS = {
    "window_hours": 24,  # sensor summary window ending at each sample time
    "min_window_coverage": 0.7,  # share of hourly values the window needs, else the sample is skipped
    # Darkness for the night-time regression (wraps past midnight), and daylight for photosynthesis.
    "night_hours": (21, 5),
    "day_hours": (6, 19),
    "min_night_points": 6,
    # Reaeration is pooled over this many nights: one night rarely varies enough to pin it down alone.
    "reaeration_pool_nights": 14,
    "metabolism_days": 3,  # metabolism features average the last few complete days
    "pressure_atm": 0.93,  # Udaipur, ~600 m
    "min_training_samples": 30,
    "cv_folds": 5,
    "test_fraction": 0.2,  # most recent share of lab samples held out
    "coverage": 0.8,
    "exceedance_levels": [2.0, 3.0],  # CPCB BOD limits for Class A, and Classes B and C
    "alert_probability": 0.5,
    "estimate_hour": 10,  # daily estimates at the usual grab-sampling time
}

# CPCB designated-best-use criteria this system can check: minimum DO (mg/L), maximum BOD (mg/L), maximum total
# coliform (MPN/100 mL) and pH range. Free ammonia (D) and conductivity, SAR and boron (E) need lab tests and are
# reported as not assessed; so are coliforms unless the coliform nowcast supplies them.
CPCB_CLASSES = [
    ("A", "Drinking water source without conventional treatment, after disinfection", 6.0, 2.0, 50.0, (6.5, 8.5)),
    ("B", "Outdoor bathing", 5.0, 3.0, 500.0, (6.5, 8.5)),
    ("C", "Drinking water source after conventional treatment", 4.0, 3.0, 5000.0, (6.0, 9.0)),
    ("D", "Propagation of wildlife and fisheries", 4.0, None, None, (6.5, 8.5)),
    ("E", "Irrigation, industrial cooling, controlled waste disposal", None, None, None, (6.0, 8.5)),
]
NOT_ASSESSED = {
    "A": "total coliforms", "B": "total coliforms", "C": "total coliforms",
    "D": "free ammonia", "E": "conductivity, SAR, boron",
}
