"""Fit the BOD soft sensor on lab samples, test it on the most recent ones, and save it."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from ..forecasting.data import load_history, load_weather, to_hourly
from .config import DEFAULT_SETTINGS
from .cpcb import assess
from .data import load_lab
from .features import MONOTONE_INCREASING, sample_features
from .training import fit_soft_sensor, input_groups, print_summary

HERE = Path(__file__).parent


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cpcb_agreement(X: pd.DataFrame, lab_bod, estimate: pd.DataFrame, levels) -> dict:
    truth = assess(X["now_dissolved_oxygen"], X["now_ph"], {lv: (np.asarray(lab_bod) <= lv).astype(float) for lv in levels})
    guess = assess(X["now_dissolved_oxygen"], X["now_ph"], {lv: 1 - estimate[f"p_above_{lv:g}"].to_numpy() for lv in levels})
    return {"cpcb_class_agreement_with_lab": round(float((truth["cpcb_class"] == guess["cpcb_class"]).mean()), 3)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "data" / "synthetic_sensor_readings.csv",
                        help="10-minute readings; the anomaly detector's --out file works too (faulty values are dropped)")
    parser.add_argument("--lab", type=Path, default=HERE / "data" / "synthetic_lab_bod.csv",
                        help="lab results with station_id, timestamp (when sampled) and bod (mg/L)")
    parser.add_argument("--weather", type=Path, default=HERE / "data" / "synthetic_weather.csv")
    parser.add_argument("--no-weather", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=HERE / "artifacts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    for path in (args.data, args.lab):
        if not path.exists():
            parser.error(f"{path} not found; run `python -m models.bod_surrogate.simulate` or pass it explicitly")

    s = dict(DEFAULT_SETTINGS)
    readings, sensor_report = load_history(args.data)
    hourly = to_hourly(readings, 3)
    use_weather = not args.no_weather and args.weather.exists()
    weather = load_weather(args.weather) if use_weather else None
    lab, lab_report = load_lab(args.lab)
    X, kept = sample_features(lab, hourly, weather, s)
    lab_report["no_sensor_coverage_dropped"] = int((~kept).sum())
    lab = lab[kept].reset_index(drop=True)
    if len(lab) < s["min_training_samples"] / (1 - s["test_fraction"]):
        parser.error(f"only {len(lab)} lab samples have sensor data around them; need more to train and test")

    def extra(X_test, actual, estimate):
        return cpcb_agreement(X_test, actual, estimate, s["exceedance_levels"])

    final, results = fit_soft_sensor(X, lab, "bod", s, args.seed, input_groups(X.columns), MONOTONE_INCREASING, extra)
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
    model_path = args.out_dir / "bod_soft_sensor.joblib"
    joblib.dump(bundle, model_path)
    (args.out_dir / "bod_soft_sensor_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    print_summary(results, lab_report, "BOD", "mg/L")
    print(f"CPCB class (DO, pH, BOD criteria) matches the lab-based class "
          f"{results['test']['cpcb_class_agreement_with_lab']:.0%} of the time")
    print(f"\nSaved {results['model']} soft sensor (refit on all {len(lab)} samples) to {model_path}")


if __name__ == "__main__":
    main()
