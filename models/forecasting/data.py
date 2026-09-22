"""Load sensor history and weather, and put both on an hourly grid."""

import numpy as np
import pandas as pd

from ..anomaly_detection.config import SENSORS
from ..anomaly_detection.data import prepare
from ..wqi.dataset import parse_numeric
from .config import WEATHER


def load_history(path) -> tuple[pd.DataFrame, dict]:
    return history_from_frame(pd.read_csv(path, low_memory=False))


def history_from_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """10-minute readings; values the anomaly detector blamed on a faulty sensor (its *_fault columns) are dropped."""
    raw = raw.copy()
    masked = 0
    for sensor in SENSORS:
        verdict = f"{sensor}_fault"
        if verdict in raw.columns:
            faulty = raw[verdict].fillna("").astype(str).ne("")
            raw.loc[faulty, sensor] = np.nan
            masked += int(faulty.sum())
    readings, report = prepare(raw)
    report["faulty_readings_masked"] = masked
    return readings, report


def to_hourly(readings: pd.DataFrame, min_valid: int) -> pd.DataFrame:
    """Hourly means per station on a gap-free grid; an hour with fewer than min_valid readings is left empty."""
    frame = readings.assign(timestamp=readings["timestamp"].dt.floor("h"))
    grouped = frame.groupby(["station_id", "timestamp"])[SENSORS]
    hourly = grouped.mean().where(grouped.count() >= min_valid).reset_index()
    grids = []
    for station, g in hourly.groupby("station_id", sort=True):
        slots = pd.date_range(g["timestamp"].min(), g["timestamp"].max(), freq="h")
        grid = g.set_index("timestamp").reindex(slots).rename_axis("timestamp").reset_index()
        grids.append(grid.assign(station_id=station))
    return pd.concat(grids, ignore_index=True)[["station_id", "timestamp", *SENSORS]]


def load_weather(path) -> pd.DataFrame:
    """Hourly city weather; short gaps (up to 3 hours) are interpolated."""
    raw = pd.read_csv(path)
    missing = [c for c in ["timestamp", *WEATHER] if c not in raw.columns]
    if missing:
        raise ValueError(f"Weather data is missing columns {missing}")
    weather = raw[["timestamp", *WEATHER]].copy()
    weather["timestamp"] = pd.to_datetime(weather["timestamp"], errors="coerce").dt.floor("h")
    weather = weather.dropna(subset=["timestamp"]).drop_duplicates("timestamp").set_index("timestamp").sort_index()
    for col in WEATHER:
        weather[col] = parse_numeric(weather[col])
    weather = weather.reindex(pd.date_range(weather.index.min(), weather.index.max(), freq="h"))
    return weather.interpolate(limit=3, limit_area="inside").rename_axis("timestamp").reset_index()
