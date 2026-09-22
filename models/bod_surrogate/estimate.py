"""Estimate daily BOD for each lake from sensor readings, with ranges, limit exceedance odds and CPCB class."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..forecasting.data import load_history, load_weather, to_hourly
from .cpcb import assess
from .features import sample_features

HERE = Path(__file__).parent


def daily_times(hourly: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """One estimate a day per station, at the usual sampling hour, once enough history exists."""
    rows = []
    lead_in = pd.Timedelta(days=settings["metabolism_days"] + 1)
    for station, g in hourly.groupby("station_id"):
        first, last = g["timestamp"].min() + lead_in, g["timestamp"].max()
        for day in pd.date_range(first.normalize(), last.normalize(), freq="D"):
            when = day + pd.Timedelta(hours=settings["estimate_hour"])
            if first <= when <= last:
                rows.append({"station_id": station, "timestamp": when})
    return pd.DataFrame(rows, columns=["station_id", "timestamp"])


def estimate_daily(bundle: dict, readings: pd.DataFrame, weather: pd.DataFrame | None) -> tuple[pd.DataFrame, int]:
    """Daily BOD estimates with ranges, exceedance odds and CPCB class; also returns how many days were skipped."""
    s, sensor = bundle["settings"], bundle["sensor"]
    hourly = to_hourly(readings, 3)
    when = daily_times(hourly, s)
    X, kept = sample_features(when, hourly, weather if bundle["uses_weather"] else None, s)
    if not kept.any():
        return pd.DataFrame(), int(len(kept))
    when = when[kept].reset_index(drop=True)
    estimate = sensor.estimate(X)
    levels = s["exceedance_levels"]
    cpcb = assess(X["now_dissolved_oxygen"], X["now_ph"], {lv: 1 - estimate[f"p_above_{lv:g}"].to_numpy() for lv in levels})
    result = pd.concat([when, estimate, cpcb], axis=1)
    result["dissolved_oxygen"] = X["now_dissolved_oxygen"].to_numpy()
    result["ph"] = X["now_ph"].to_numpy()
    result["metabolism_missing"] = np.isnan(X[["respiration", "gpp"]].to_numpy()).any(axis=1)
    return result, int((~kept).sum())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="10-minute readings covering at least five days")
    parser.add_argument("--weather", type=Path, help="hourly weather (needed if the model was trained with it)")
    parser.add_argument("--model", type=Path, default=HERE / "artifacts" / "bod_soft_sensor.joblib")
    parser.add_argument("--show-days", type=int, default=7, help="days per station to print")
    parser.add_argument("--out", type=Path, help="write every daily estimate to this CSV")
    args = parser.parse_args(argv)

    bundle = joblib.load(args.model)
    s = bundle["settings"]
    if bundle["uses_weather"] and args.weather is None:
        parser.error("this model was trained with weather; pass --weather")
    readings, _ = load_history(args.data)
    result, skipped = estimate_daily(bundle, readings, load_weather(args.weather) if bundle["uses_weather"] else None)
    if result.empty:
        parser.error("no day has enough sensor data for an estimate")
    levels = s["exceedance_levels"]
    print(f"{len(result)} daily estimates ({skipped} days skipped for lack of sensor data); model: {bundle['model']}, "
          f"trained on {bundle['lab_samples']} lab samples")
    for station, g in result.groupby("station_id"):
        print(f"\n=== {station} ===")
        for _, r in g.tail(args.show_days).iterrows():
            odds = "  ".join(f"P(>{lv:g})={r[f'p_above_{lv:g}']:.0%}" for lv in levels)
            print(f"  {r['timestamp'].date()}  BOD {r['bod']:.1f} mg/L ({r['bod_lo']:.1f}-{r['bod_hi']:.1f})  {odds}  "
                  f"CPCB class {r['cpcb_class']}")
    print("\nCPCB class uses DO, pH and estimated BOD only; coliforms, free ammonia and conductivity still need lab tests.")
    if result["metabolism_missing"].any():
        print("Note: some days lacked a complete night or day of oxygen data, so metabolism inputs were missing there.")
    if args.out:
        result.to_csv(args.out, index=False)
        print(f"Wrote {len(result)} estimates to {args.out}")


if __name__ == "__main__":
    main()
