"""Targets, horizons and model settings for the forecaster."""

WEATHER = ["air_temperature", "cloud_cover", "rain_mm"]

# Copied into every trained forecaster, so a saved model keeps the settings it was fit with.
DEFAULT_SETTINGS = {
    "horizons": list(range(1, 49)),  # hourly look-ahead
    # Horizon is a model input, so a subset is enough to learn from; the model then answers for every hour.
    "train_horizons": [1, 2, 3, 4, 6, 9, 12, 15, 18, 24, 30, 36, 42, 48],
    "train_origin_every_hours": 3,
    "eval_origin_every_hours": 3,
    "horizon_buckets": [(1, 3), (4, 6), (7, 12), (13, 24), (25, 36), (37, 48)],
    "quantiles": (0.1, 0.5, 0.9),  # the outer two form an 80% range
    "min_valid_per_hour": 3,  # of the six 10-minute readings
    "lags_hours": [1, 2, 3, 6, 12, 24],
    "pressure_atm": 0.93,  # Udaipur, ~600 m; used for the oxygen saturation feature
    # Nightly oxygen minimum: the pre-dawn low is when fish kills happen.
    "night_hours": (0, 9),
    "night_min_valid_hours": 5,
    "night_issue_hour": 18,
    "night_days": 7,
    "do_alert_mg_l": 4.0,  # CPCB Class D (wildlife and fisheries) minimum
    "alert_probability": 0.5,
    # Ranges are recalibrated on this many days of recent, already-verified forecasts, so they track the season.
    "recalibrate_days": 30,
    "recalibrate_every_days": 7,
    # Training uses recorded weather, but live forecasts use predicted weather; adding error that grows with
    # lead time keeps the model from trusting weather more than a forecast deserves. Turn off when training
    # on archived weather forecasts.
    "weather_forecast_noise": True,
    "gbm": {"max_iter": 300, "learning_rate": 0.08, "max_leaf_nodes": 31, "min_samples_leaf": 40, "l2_regularization": 1.0},
    "challenger": {
        "lookback_hours": 168,
        "origin_every_hours": 2,
        "hidden": 48,
        "epochs": 15,
        "patience": 3,
        "batch_size": 256,
        "learning_rate": 0.002,
    },
}
