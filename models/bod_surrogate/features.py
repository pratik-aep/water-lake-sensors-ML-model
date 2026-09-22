"""What the sensors said before each sample time: recent readings, daily swings, and lake metabolism."""

import numpy as np
import pandas as pd

from ..anomaly_detection.config import SENSORS
from ..anomaly_detection.rules import to_working
from ..wqi.calculator import do_saturation_mgl
from .metabolism import daily_metabolism

METABOLISM = ["respiration", "log_respiration", "reaeration", "night_temperature", "gpp", "nep", "do_amplitude"]
# Physics the model must respect: more oxygen consumption, murkier water or a larger oxygen deficit can
# only raise the BOD estimate.
MONOTONE_INCREASING = ["respiration", "log_respiration", "mean_turbidity", "mean_do_deficit"]


class StationHistory:
    """One station's hourly readings (working scale) plus its daily metabolism table."""

    def __init__(self, hourly: pd.DataFrame, settings: dict):
        frame = hourly.set_index("timestamp")[SENSORS]
        self.frame = pd.DataFrame({s: to_working(frame[s], s) for s in SENSORS})
        self.deficit = do_saturation_mgl(self.frame["temperature"], settings["pressure_atm"]) - self.frame["dissolved_oxygen"]
        self.metabolism = daily_metabolism(hourly, settings)


def features_at(history: StationHistory, when: pd.Timestamp, weather: pd.DataFrame | None, settings: dict):
    """Features for a sample taken at `when`, from complete hours before it only; None if the window is too sparse."""
    s = settings
    hour = when.floor("h")
    window = history.frame.loc[hour - pd.Timedelta(hours=s["window_hours"]) : hour - pd.Timedelta(hours=1)]
    if window.notna().all(axis=1).sum() < s["min_window_coverage"] * s["window_hours"]:
        return None
    feats = {f"now_{c}": window[c].dropna().iloc[-1] if window[c].notna().any() else np.nan for c in SENSORS}
    for c in SENSORS:
        feats[f"mean_{c}"] = window[c].mean()
        feats[f"min_{c}"] = window[c].min()
        feats[f"max_{c}"] = window[c].max()
    feats["mean_do_deficit"] = history.deficit.loc[window.index].mean()

    # Last night's respiration counts once it has ended; daytime production only for days already over.
    table = history.metabolism
    night_done = when.normalize() if when.hour >= s["night_hours"][1] else when.normalize() - pd.Timedelta(days=1)
    nights = table.loc[: night_done, ["respiration", "reaeration", "night_temperature"]].dropna().tail(s["metabolism_days"])
    days = table.loc[: when.normalize() - pd.Timedelta(days=1), ["gpp", "nep", "do_amplitude"]].dropna().tail(s["metabolism_days"])
    feats.update(nights.mean().to_dict() if len(nights) else dict.fromkeys(["respiration", "reaeration", "night_temperature"], np.nan))
    feats.update(days.mean().to_dict() if len(days) else dict.fromkeys(["gpp", "nep", "do_amplitude"], np.nan))
    # Organic matter scales respiration multiplicatively, so on the log scale it becomes a straight-line signal
    # that linear models can use alongside log BOD.
    feats["log_respiration"] = np.log(max(feats["respiration"], 1e-3)) if not np.isnan(feats["respiration"]) else np.nan

    if weather is not None:
        w = weather.set_index("timestamp")
        feats["rain_72h"] = w.loc[hour - pd.Timedelta(hours=72) : hour - pd.Timedelta(hours=1), "rain_mm"].sum(min_count=1)
        feats["cloud_24h"] = w.loc[hour - pd.Timedelta(hours=24) : hour - pd.Timedelta(hours=1), "cloud_cover"].mean()
    return feats


def sample_features(samples: pd.DataFrame, hourly: pd.DataFrame, weather: pd.DataFrame | None, settings: dict):
    """Feature table for (station_id, timestamp) rows; returns (features, mask of rows that had enough sensor data)."""
    histories = {st: StationHistory(g.reset_index(drop=True), settings) for st, g in hourly.groupby("station_id")}
    rows, kept = [], []
    for station, when in zip(samples["station_id"], samples["timestamp"]):
        feats = features_at(histories[station], pd.Timestamp(when), weather, settings) if station in histories else None
        kept.append(feats is not None)
        if feats is not None:
            rows.append(feats)
    return pd.DataFrame(rows), np.array(kept, dtype=bool)
