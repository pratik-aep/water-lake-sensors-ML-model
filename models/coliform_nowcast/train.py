"""Fit the total coliform nowcast on lab samples, test it on the most recent ones, and save it."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn

from ..bod_surrogate.data import load_lab
from ..bod_surrogate.train import sha256
from ..bod_surrogate.training import fit_soft_sensor, input_groups, print_summary
from ..forecasting.data import load_history, load_weather, to_hourly
from .bands import band_of_count, band_of_nowcast
from .config import DEFAULT_SETTINGS, TARGET
from .features import MONOTONE_INCREASING, WEATHER, sample_features

HERE = Path(__file__).parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "data" / "synthetic_sensor_readings.csv",
                        help="10-minute readings; the anomaly detector's --out file works too (faulty values are dropped)")
    parser.add_argument("--lab", type=Path, default=HERE / "data" / "synthetic_lab_results.csv",
                        help=f"lab results with station_id, timestamp and {TARGET} (MPN/100 mL)")
    parser.add_argument("--weather", type=Path, default=HERE / "data" / "synthetic_weather.csv",
                        help="hourly weather; rain timing is the strongest coliform signal")
    parser.add_argument("--no-weather", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=HERE / "artifacts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    for path in (args.data, args.lab):
        if not path.exists():
            parser.error(f"{path} not found; run `python -m models.coliform_nowcast.simulate` or pass it explicitly")

    s = dict(DEFAULT_SETTINGS)
    readings, sensor_report = load_history(args.data)
    hourly = to_hourly(readings, 3)
    use_weather = not args.no_weather and args.weather.exists()
    weather = load_weather(args.weather) if use_weather else None
    lab, lab_report = load_lab(args.lab, TARGET, s["limits"])
    X, kept = sample_features(lab, hourly, weather, s)
    lab_report["no_sensor_coverage_dropped"] = int((~kept).sum())
    lab = lab[kept].reset_index(drop=True)
    if len(lab) < s["min_training_samples"] / (1 - s["test_fraction"]):
        parser.error(f"only {len(lab)} lab samples have sensor data around them; need more to train and test")

    def extra(X_test, actual, estimate):
        # Does the nowcast put each sample in the same CPCB coliform band (A, B, C or above) as the lab did?
        nowcast = band_of_nowcast({lv: estimate[f"p_above_{lv:g}"].to_numpy() for lv in s["exceedance_levels"]})
        return {"cpcb_band_agreement_with_lab": round(float(np.mean(band_of_count(actual) == nowcast)), 3)}

    groups = input_groups(X.columns, weather=WEATHER)
    final, results = fit_soft_sensor(X, lab, TARGET, s, args.seed, groups, MONOTONE_INCREASING, extra)
    bundle = {
        "sensor": final,
        "settings": s,
        "uses_weather": use_weather and "weather" not in results["dropped_input_groups"],
        "model": results["model"],
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "data_file": str(args.data),
        "data_sha256": sha256(args.data),
        "lab_file": str(args.lab),
        "lab_sha256": sha256(args.lab),
        "weather_file": str(args.weather) if use_weather else None,
        "lab_samples": len(lab),
        "stations": sorted(lab["station_id"].unique()),
        "date_range": [str(lab["timestamp"].min()), str(lab["timestamp"].max())],
    }
    metrics = {**{k: v for k, v in bundle.items() if k != "sensor"}, "sensor_report": sensor_report, "lab_report": lab_report, **results}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "coliform_nowcast.joblib"
    joblib.dump(bundle, model_path)
    (args.out_dir / "coliform_nowcast_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    print_summary(results, lab_report, "total coliform", "MPN/100 mL")
    print(f"CPCB coliform band (A/B/C/above) matches the lab's {results['test']['cpcb_band_agreement_with_lab']:.0%} of the time")
    print(f"\nSaved {results['model']} nowcast (refit on all {len(lab)} samples) to {model_path}")


if __name__ == "__main__":
    main()
