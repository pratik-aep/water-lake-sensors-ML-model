"""Estimate daily chlorophyll-a for each lake from sensor readings, with trophic state and WHO bloom alert level."""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from ..bod_surrogate.estimate import daily_times
from ..forecasting.data import load_history, load_weather, to_hourly
from .config import TARGET
from .features import sample_features
from .trophic import carlson_tsi, class_probabilities, trophic_class, who_level

HERE = Path(__file__).parent


def estimate_daily(bundle: dict, readings: pd.DataFrame, weather: pd.DataFrame | None) -> tuple[pd.DataFrame, int]:
    """Daily chlorophyll-a with a range, trophic class odds and WHO level; also returns how many days were skipped."""
    s, sensor = bundle["settings"], bundle["sensor"]
    hourly = to_hourly(readings, 3)
    when = daily_times(hourly, s)
    X, kept = sample_features(when, hourly, weather if bundle["uses_weather"] else None, s)
    if not kept.any():
        return pd.DataFrame(), int(len(kept))
    when = when[kept].reset_index(drop=True)
    estimate = sensor.estimate(X)
    tsi = carlson_tsi(estimate[TARGET])
    result = pd.concat([when, estimate, class_probabilities(sensor, X)], axis=1)
    result["tsi"] = tsi.round(1)
    result["trophic_class"] = trophic_class(tsi)
    result["who_level"] = who_level(estimate["p_above_12"], estimate["p_above_24"], s["alert_probability"])
    result["metabolism_missing"] = X["gpp"].isna().to_numpy()
    return result, int((~kept).sum())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="10-minute readings covering at least five days")
    parser.add_argument("--weather", type=Path, help="hourly weather (needed if the model was trained with it)")
    parser.add_argument("--model", type=Path, default=HERE / "artifacts" / "chlorophyll_soft_sensor.joblib")
    parser.add_argument("--show-days", type=int, default=7, help="days per station to print")
    parser.add_argument("--out", type=Path, help="write every daily estimate to this CSV")
    args = parser.parse_args(argv)

    bundle = joblib.load(args.model)
    if bundle["uses_weather"] and args.weather is None:
        parser.error("this model was trained with weather; pass --weather")
    readings, _ = load_history(args.data)
    result, skipped = estimate_daily(bundle, readings, load_weather(args.weather) if bundle["uses_weather"] else None)
    if result.empty:
        parser.error("no day has enough sensor data for an estimate")

    print(f"{len(result)} daily estimates ({skipped} days skipped for lack of sensor data); model: {bundle['model']}, "
          f"trained on {bundle['lab_samples']} lab samples")
    for station, g in result.groupby("station_id"):
        print(f"\n=== {station} ===")
        for _, r in g.tail(args.show_days).iterrows():
            print(f"  {r['timestamp'].date()}  chlorophyll-a {r[TARGET]:.1f} ug/L ({r[f'{TARGET}_lo']:.1f}-{r[f'{TARGET}_hi']:.1f})  "
                  f"TSI {r['tsi']:.0f} {r['trophic_class']}  P(>12)={r['p_above_12']:.0%} P(>24)={r['p_above_24']:.0%}  "
                  f"WHO: {r['who_level']}")
    print("\nWHO alert levels assume cyanobacteria dominate the algae; confirm with a microscope count before acting.")
    if args.out:
        result.to_csv(args.out, index=False)
        print(f"Wrote {len(result)} estimates to {args.out}")


if __name__ == "__main__":
    main()
