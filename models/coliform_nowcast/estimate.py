"""Nowcast daily total coliform for each lake from sensor readings and rain, with CPCB band odds."""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from ..bod_surrogate.estimate import daily_times
from ..forecasting.data import load_history, load_weather, to_hourly
from .bands import band_of_nowcast
from .config import TARGET
from .features import sample_features

HERE = Path(__file__).parent


def estimate_daily(bundle: dict, readings: pd.DataFrame, weather: pd.DataFrame | None) -> tuple[pd.DataFrame, int]:
    """Daily coliform nowcast with a range and the chance of exceeding each CPCB limit; also returns skipped days."""
    s, sensor = bundle["settings"], bundle["sensor"]
    hourly = to_hourly(readings, 3)
    when = daily_times(hourly, s)
    X, kept = sample_features(when, hourly, weather if bundle["uses_weather"] else None, s)
    if not kept.any():
        return pd.DataFrame(), int(len(kept))
    when = when[kept].reset_index(drop=True)
    estimate = sensor.estimate(X)
    result = pd.concat([when, estimate], axis=1)
    result["cpcb_band"] = band_of_nowcast({lv: estimate[f"p_above_{lv:g}"].to_numpy() for lv in s["exceedance_levels"]})
    return result, int((~kept).sum())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="10-minute readings covering at least five days")
    parser.add_argument("--weather", type=Path, help="hourly weather (needed if the model was trained with it)")
    parser.add_argument("--model", type=Path, default=HERE / "artifacts" / "coliform_nowcast.joblib")
    parser.add_argument("--show-days", type=int, default=7, help="days per station to print")
    parser.add_argument("--out", type=Path, help="write every daily nowcast to this CSV")
    args = parser.parse_args(argv)

    bundle = joblib.load(args.model)
    if bundle["uses_weather"] and args.weather is None:
        parser.error("this model was trained with weather; pass --weather")
    readings, _ = load_history(args.data)
    result, skipped = estimate_daily(bundle, readings, load_weather(args.weather) if bundle["uses_weather"] else None)
    if result.empty:
        parser.error("no day has enough sensor data for a nowcast")

    print(f"{len(result)} daily nowcasts ({skipped} days skipped for lack of sensor data); model: {bundle['model']}, "
          f"trained on {bundle['lab_samples']} lab samples")
    for station, g in result.groupby("station_id"):
        print(f"\n=== {station} ===")
        for _, r in g.tail(args.show_days).iterrows():
            print(f"  {r['timestamp'].date()}  total coliform {r[TARGET]:.0f} MPN/100 mL ({r[f'{TARGET}_lo']:.0f}-{r[f'{TARGET}_hi']:.0f})  "
                  f"P(>500)={r['p_above_500']:.0%} P(>5000)={r['p_above_5000']:.0%}  CPCB band {r['cpcb_band']}")
    print("\nA nowcast guides sampling and warnings; confirm any unsafe call with a lab count before acting on it.")
    if args.out:
        result.to_csv(args.out, index=False)
        print(f"Wrote {len(result)} nowcasts to {args.out}")


if __name__ == "__main__":
    main()
