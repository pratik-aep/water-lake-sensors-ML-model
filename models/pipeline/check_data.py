"""Before training on real data: is there enough of it, in the right units, for each model?"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..anomaly_detection.config import DEFAULT_SETTINGS as ANOMALY
from ..anomaly_detection.config import SENSORS
from ..anomaly_detection.data import prepare
from ..bod_surrogate.config import DEFAULT_SETTINGS as SOFT
from ..forecasting.data import to_hourly
from .ingest import load_mapping, normalise, turbidity_unit

HERE = Path(__file__).parent
TEST_DAYS = 60  # train_all's default holdout
FORECAST_HISTORY_DAYS = 60 + 30 + TEST_DAYS  # training history, then calibration, then test
AIR_HISTORY_DAYS = 45 + 30 + TEST_DAYS
LAB_TARGETS = {"bod": "BOD soft sensor", "chlorophyll_a": "algae soft sensor", "total_coliform": "coliform nowcast"}
# DO above this in "mg/L" is almost certainly % saturation: water can't hold that much oxygen at lake temperatures.
DO_MGL_CEILING = 25.0


def covered_samples(lab: pd.DataFrame, hourly: pd.DataFrame) -> int:
    """Lab samples with enough complete sensor hours in the day before them for a soft sensor to use."""
    complete = hourly.dropna(subset=SENSORS).set_index("timestamp")
    need = SOFT["min_window_coverage"] * SOFT["window_hours"]
    count = 0
    for station, when in zip(lab["station_id"].astype(str), pd.to_datetime(lab["timestamp"], errors="coerce")):
        if pd.isna(when):
            continue
        hours = complete[complete["station_id"] == station].loc[
            when.floor("h") - pd.Timedelta(hours=SOFT["window_hours"]) : when.floor("h") - pd.Timedelta(hours=1)
        ]
        count += len(hours) >= need
    return int(count)


def readiness(readings: pd.DataFrame, lab: pd.DataFrame | None, weather: pd.DataFrame | None,
              air: pd.DataFrame | None = None) -> tuple[dict, list]:
    """Per station coverage, plus one (model, ready, reason) row per model."""
    days = (readings.groupby("station_id")["timestamp"].agg(lambda t: (t.max() - t.min()) / pd.Timedelta(days=1)))
    coverage = readings.groupby("station_id")[SENSORS].agg(lambda v: round(float(v.notna().mean()), 3))
    stations = {st: {"days": round(float(days[st]), 1), "share_of_readings_present": coverage.loc[st].to_dict()} for st in days.index}
    span = float(days.min())
    rows = [
        ("anomaly detector", span >= ANOMALY["cv_block_days"] * 2 + TEST_DAYS,
         f"{span:.0f} days per station; needs {ANOMALY['cv_block_days'] * 2 + TEST_DAYS} (28 to learn, {TEST_DAYS} to test)"),
        ("forecaster", span >= FORECAST_HISTORY_DAYS, f"{span:.0f} days; needs {FORECAST_HISTORY_DAYS} (60 history, 30 calibration, {TEST_DAYS} test)"),
    ]
    if weather is None:
        rows.append(("weather", False, "no weather file: forecasts lose about 17% accuracy and turbidity after rain looks like faults"))
    else:
        w = pd.to_datetime(weather["timestamp"])
        covered = readings["timestamp"].between(w.min(), w.max()).mean()
        rows.append(("weather", covered > 0.95, f"covers {covered:.0%} of the readings' time span"))

    needed = int(np.ceil(SOFT["min_training_samples"] / (1 - SOFT["test_fraction"])))
    hourly = to_hourly(readings, 3)
    for target, name in LAB_TARGETS.items():
        if lab is None or target not in lab.columns:
            rows.append((name, False, f"no '{target}' column in the lab results"))
            continue
        have = covered_samples(lab[lab[target].notna()], hourly)
        rows.append((name, have >= needed, f"{have} lab samples with sensor data around them; needs {needed}"))
    if air is not None:
        stamps = pd.to_datetime(air["timestamp"], errors="coerce")
        air_days = float(stamps.groupby(air["station_id"]).agg(lambda t: (t.max() - t.min()) / pd.Timedelta(days=1)).min())
        missing = [c for c in ("pm25", "pm10") if c not in air.columns]
        rows.append(("air quality", not missing and air_days >= AIR_HISTORY_DAYS,
                     f"missing columns {missing}" if missing else f"{air_days:.0f} days per station; needs {AIR_HISTORY_DAYS} (45 history, 30 calibration, {TEST_DAYS} test)"))
        if weather is not None and not {"wind_speed", "boundary_layer_height"} <= set(weather.columns):
            rows.append(("air weather", False, "weather lacks wind_speed and boundary_layer_height; fetch it with models.pipeline.weather"))
    wqi_ready = lab is not None and {"bod", "conductivity", "nitrate"} <= set(lab.columns)
    rows.append(("WQI classifier", wqi_ready and len(lab) >= 50,
                 "needs lab bod, conductivity and nitrate on 50+ samples" if not wqi_ready else f"{len(lab)} lab samples; needs 50"))
    return stations, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readings", type=Path, required=True, help="sensor export")
    parser.add_argument("--vendor-mapping", type=Path, help="column names and units of the export (see vendor_mapping.json)")
    parser.add_argument("--lab", type=Path)
    parser.add_argument("--weather", type=Path)
    parser.add_argument("--air-readings", type=Path, help="air station export, if there are air stations")
    args = parser.parse_args(argv)

    mapping = load_mapping(args.vendor_mapping)
    normalised, ingest = normalise(pd.read_csv(args.readings, low_memory=False), mapping)
    readings, grid = prepare(normalised)
    lab = pd.read_csv(args.lab) if args.lab else None
    weather = pd.read_csv(args.weather) if args.weather else None
    air = pd.read_csv(args.air_readings, low_memory=False) if args.air_readings else None
    stations, rows = readiness(readings, lab, weather, air)

    print(f"Readings: {grid['rows_in']} rows, {len(stations)} stations, {grid['date_range'][0]} to {grid['date_range'][1]}")
    print(f"Turbidity is in {turbidity_unit(mapping)}; oxygen in mg/L after import")
    for sensor, n in ingest["censored_at_sensor_limit"].items():
        print(f"  {n} {sensor} readings sat at the sensor's range limit (the lake went past what it can measure)")
    median_do = readings["dissolved_oxygen"].median()
    if median_do > DO_MGL_CEILING:
        print(f"  WARNING: median oxygen is {median_do:.0f} 'mg/L', which is physically impossible: the export is almost "
              "certainly % saturation. Set units.dissolved_oxygen to 'percent' in the mapping")
    for station, s in stations.items():
        print(f"  {station}: {s['days']} days; share of readings present {s['share_of_readings_present']}")
    print("\nModel readiness:")
    for name, ready, reason in rows:
        print(f"  {'READY  ' if ready else 'NOT YET'}  {name:<20} {reason}")


if __name__ == "__main__":
    main()
