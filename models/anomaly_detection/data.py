"""Load sensor readings and put each station on a regular time grid."""

import pandas as pd

from ..wqi.dataset import parse_numeric
from .config import INTERVAL, SENSORS

REQUIRED = ["station_id", "timestamp", *SENSORS]


def prepare(df: pd.DataFrame, interval: str = INTERVAL) -> tuple[pd.DataFrame, dict]:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Sensor data is missing columns {missing}")

    df = df[REQUIRED].copy()
    report = {"rows_in": len(df)}
    df["station_id"] = df["station_id"].astype(str)
    # Logger clocks jitter by a few seconds; snap each reading to its scan slot.
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.round(interval)
    bad_time = df["timestamp"].isna()
    report["bad_timestamp_dropped"] = int(bad_time.sum())
    df = df[~bad_time]
    for sensor in SENSORS:
        df[sensor] = parse_numeric(df[sensor])

    before = len(df)
    df = df.drop_duplicates(["station_id", "timestamp"])
    report["duplicates_dropped"] = before - len(df)
    if df.empty:
        raise ValueError(f"No usable readings left after cleaning: {report}")

    grids = []
    for station, g in df.groupby("station_id", sort=True):
        slots = pd.date_range(g["timestamp"].min(), g["timestamp"].max(), freq=interval)
        grid = g.set_index("timestamp").reindex(slots).rename_axis("timestamp").reset_index()
        grids.append(grid.assign(station_id=station))
    out = pd.concat(grids, ignore_index=True)[REQUIRED]

    report["gap_rows_added"] = len(out) - len(df)
    report["stations"] = sorted(out["station_id"].unique().tolist())
    report["date_range"] = [out["timestamp"].min().isoformat(), out["timestamp"].max().isoformat()]
    report["rows_out"] = len(out)
    return out, report


def load_readings(path, interval: str = INTERVAL) -> tuple[pd.DataFrame, dict]:
    return prepare(pd.read_csv(path, low_memory=False), interval)
