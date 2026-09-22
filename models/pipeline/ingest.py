"""Turn a vendor sensor export into the readings every model expects: our column names, oxygen in mg/L."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..anomaly_detection.config import SENSORS
from ..wqi.calculator import do_percent_to_mgl
from ..wqi.dataset import parse_numeric

HERE = Path(__file__).parent
REQUIRED = ["station_id", "timestamp", *SENSORS]
# What the models assume when no mapping is given: our own column names and units.
DEFAULT_MAPPING = {
    "columns": {c: c for c in REQUIRED},
    "stations": {},
    "units": {"dissolved_oxygen": "mg/L", "turbidity": "NTU"},
    "turbidity_percent_to_ntu": None,
    "sensor_ranges": {},
    "pressure_atm": 0.93,
    "timezone": None,
}
UNITS = {"dissolved_oxygen": ("mg/L", "percent"), "turbidity": ("NTU", "percent")}


def load_mapping(path) -> dict:
    """A mapping file (see vendor_mapping.json) layered over the defaults."""
    mapping = {**DEFAULT_MAPPING, **json.loads(Path(path).read_text())} if path else dict(DEFAULT_MAPPING)
    mapping["columns"] = {**DEFAULT_MAPPING["columns"], **mapping.get("columns", {})}
    mapping["units"] = {**DEFAULT_MAPPING["units"], **mapping.get("units", {})}
    for sensor, unit in mapping["units"].items():
        if unit not in UNITS.get(sensor, (unit,)):
            raise ValueError(f"unit {unit!r} for {sensor} is not one of {UNITS[sensor]}")
    return mapping


def turbidity_unit(mapping: dict) -> str:
    """The unit turbidity is reported in after normalising: NTU, or the vendor's % scale if uncalibrated."""
    if mapping["units"]["turbidity"] == "percent" and not mapping.get("turbidity_percent_to_ntu"):
        return "%"
    return "NTU"


def normalise(raw: pd.DataFrame, mapping: dict) -> tuple[pd.DataFrame, dict]:
    """Rename, convert and censor a vendor export; returns (readings, report of what was changed)."""
    renamed = {vendor: ours for ours, vendor in mapping["columns"].items()}
    missing = [vendor for vendor, ours in renamed.items() if vendor not in raw.columns]
    if missing:
        raise ValueError(f"the export has no column(s) {missing}; it has {list(raw.columns)}. Edit 'columns' in the mapping")
    df = raw[list(renamed)].rename(columns=renamed)
    report = {"rows": len(df), "units_in": dict(mapping["units"]), "turbidity_unit_out": turbidity_unit(mapping)}

    df["station_id"] = df["station_id"].astype(str).str.strip().replace(mapping.get("stations") or {})
    stamps = pd.to_datetime(df["timestamp"], errors="coerce", utc=bool(mapping.get("timezone")))
    if mapping.get("timezone"):
        # Exports in UTC (or with offsets) become local clock time, which the daily cycle features rely on.
        stamps = stamps.dt.tz_convert(mapping["timezone"]).dt.tz_localize(None)
    df["timestamp"] = stamps
    for sensor in SENSORS:
        df[sensor] = parse_numeric(df[sensor])

    # A reading pinned at the sensor's range limit only says the true value is at least (or at most) that:
    # pH 9 on a 4-9 probe during an afternoon bloom, or 100% oxygen on a sensor that stops there.
    # A limit of null means the range ends where nature does (oxygen can't go below 0), so it censors nothing.
    censored, by_station = {}, {}
    for sensor, (low, high) in (mapping.get("sensor_ranges") or {}).items():
        below = df[sensor] <= low if low is not None else False
        above = df[sensor] >= high if high is not None else False
        at_limit = df[sensor].notna() & (below | above)
        if at_limit.any():
            censored[sensor] = int(at_limit.sum())
            by_station.update({(st, sensor): int(n) for st, n in df.loc[at_limit, "station_id"].value_counts().items()})
            df.loc[at_limit, sensor] = np.nan
    report["censored_at_sensor_limit"] = censored
    report["censored_by_station"] = {st: {s: n for (st2, s), n in by_station.items() if st2 == st} for st, _ in by_station}

    if mapping["units"]["dissolved_oxygen"] == "percent":
        df["dissolved_oxygen"] = do_percent_to_mgl(df["dissolved_oxygen"], df["temperature"], mapping["pressure_atm"])
        report["dissolved_oxygen"] = "converted from % saturation to mg/L using water temperature"
    calibration = mapping.get("turbidity_percent_to_ntu")
    if mapping["units"]["turbidity"] == "percent" and calibration:
        slope, intercept = calibration
        df["turbidity"] = (slope * df["turbidity"] + intercept).clip(lower=0)
        report["turbidity"] = f"converted from % to NTU as {slope:g} x % + {intercept:g}"
    report["rows_without_time_dropped"] = int(df["timestamp"].isna().sum())
    return df.dropna(subset=["timestamp"]).reset_index(drop=True), report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True, help="CSV exported from the vendor's app or cloud")
    parser.add_argument("--mapping", type=Path, default=HERE / "vendor_mapping.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    mapping = load_mapping(args.mapping)
    readings, report = normalise(pd.read_csv(args.export, low_memory=False), mapping)
    readings.to_csv(args.out, index=False)
    print(json.dumps(report, indent=2))
    for sensor, n in report["censored_at_sensor_limit"].items():
        print(f"Note: {n} {sensor} readings sat at the sensor's range limit and were set aside; "
              "if this happens daily, the lake goes beyond what the sensor can measure")
    print(f"Wrote {len(readings)} readings to {args.out}")


if __name__ == "__main__":
    main()
