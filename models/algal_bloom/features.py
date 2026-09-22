"""Sensor features for chlorophyll-a: the BOD soft sensor's inputs plus algae-specific productivity signals."""

import numpy as np
import pandas as pd

from ..bod_surrogate.features import METABOLISM, sample_features as base_features

# Photosynthesis is algal biomass times light, so the day's oxygen production and pH swing (CO2 drawn down by
# day) grow with chlorophyll; the model may only ever raise its estimate as they rise.
MONOTONE_INCREASING = ["gpp", "log_gpp", "do_amplitude", "swing_ph"]
ALGAE_METABOLISM = [*METABOLISM, "log_gpp"]


def sample_features(samples: pd.DataFrame, hourly: pd.DataFrame, weather, settings: dict):
    X, kept = base_features(samples, hourly, weather, settings)
    if len(X):
        # Biomass scales production multiplicatively, so on the log scale it is a straight-line signal.
        X["log_gpp"] = np.log(X["gpp"].clip(lower=1e-3))
        X["swing_ph"] = X["max_ph"] - X["min_ph"]
    return X, kept
