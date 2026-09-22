"""Features for coliform: the soft-sensor inputs plus rain timing, the main driver of faecal bacteria in lakes."""

import numpy as np
import pandas as pd

from ..bod_surrogate.features import sample_features as base_features

WEATHER = ("rain_24h", "rain_72h", "rain_7d", "hours_since_rain", "cloud_24h",
           "log_rain_24h", "log_rain_72h", "log_rain_7d", "log_hours_since_rain")
# Runoff washes bacteria in and murky water carries them, so more rain or turbidity may only raise the estimate.
MONOTONE_INCREASING = ["rain_24h", "rain_72h", "rain_7d", "log_rain_24h", "log_rain_72h", "log_rain_7d",
                       "mean_turbidity", "max_turbidity"]
DRY_SPELL_CAP_HOURS = 24 * 14


def sample_features(samples: pd.DataFrame, hourly: pd.DataFrame, weather, settings: dict):
    X, kept = base_features(samples, hourly, weather, settings)
    if weather is not None and len(X):
        rain = weather.set_index("timestamp")["rain_mm"]
        wet_hours = rain.index[rain >= 1.0]
        rain_24h, rain_7d, since = [], [], []
        for when in samples["timestamp"].to_numpy()[kept]:
            hour = pd.Timestamp(when).floor("h")
            rain_24h.append(rain.loc[hour - pd.Timedelta(hours=24) : hour - pd.Timedelta(hours=1)].sum(min_count=1))
            rain_7d.append(rain.loc[hour - pd.Timedelta(days=7) : hour - pd.Timedelta(hours=1)].sum(min_count=1))
            earlier = wet_hours[wet_hours < hour]
            hours = (hour - earlier[-1]) / pd.Timedelta(hours=1) if len(earlier) else np.inf
            since.append(min(hours, DRY_SPELL_CAP_HOURS))
        X["rain_24h"], X["rain_7d"], X["hours_since_rain"] = rain_24h, rain_7d, since
        # Bacteria counts grow with the log of the runoff, so linear models need the rain on a log scale too.
        for col in ("rain_24h", "rain_72h", "rain_7d"):
            X[f"log_{col}"] = np.log1p(X[col].astype(float))
        X["log_hours_since_rain"] = np.log1p(X["hours_since_rain"].astype(float))
    return X, kept
