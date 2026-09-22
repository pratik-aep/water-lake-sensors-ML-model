"""Run a trained anomaly detector over a CSV of sensor readings and report the incidents."""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from ..forecasting.data import load_weather
from .config import INTERVAL
from .data import load_readings
from .evaluate import alarm_incidents, consistency_coverage

LOW_COVERAGE = 0.8

HERE = Path(__file__).parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="CSV with station_id, timestamp and sensor columns")
    parser.add_argument("--model", type=Path, default=HERE / "artifacts" / "anomaly_detector.joblib")
    parser.add_argument(
        "--since",
        type=pd.Timestamp,
        help="only report readings from this time on; earlier rows serve as history (include a week or more)",
    )
    parser.add_argument("--weather", type=Path, help="hourly weather covering the readings (if the detector uses it)")
    parser.add_argument("--out", type=Path, help="write every reported reading with its flags and verdict to this CSV")
    args = parser.parse_args(argv)

    bundle = joblib.load(args.model)
    detector = bundle["detector"]
    if detector.uses_weather and args.weather is None:
        parser.error("this detector was trained with weather; pass --weather")
    df, report = load_readings(args.data)
    unknown = sorted(set(report["stations"]) - set(detector.stations_))
    if unknown:
        print(f"Warning: stations {unknown} were not in training; all-station error scales are used for them")

    result = detector.detect(df, load_weather(args.weather) if detector.uses_weather else None)
    if args.since is not None:
        result = result[result["timestamp"] >= args.since]
    merge_gap = pd.Timedelta(INTERVAL) * (detector.settings["merge_gap_points"] + 1)
    incidents = alarm_incidents(result, merge_gap)

    print(f"Checked {len(result)} readings (model trained {bundle['trained_at']} on {bundle['date_range'][0]} to {bundle['date_range'][1]})")
    print("Verdicts:", result["verdict"].value_counts().to_dict())
    coverage = consistency_coverage(result)
    print("Cross-sensor checks ran on this share of readings:", coverage)
    paused = [s for s, share in coverage.items() if share < LOW_COVERAGE]
    if paused:
        print(f"Warning: checks for {paused} were often paused because conditions were outside the training "
              "range; retrain on recent data (rule checks still ran)")
    print(f"\n{len(incidents)} incident(s):")
    if len(incidents):
        print(incidents.to_string(index=False))
    if args.out:
        result.to_csv(args.out, index=False)
        print(f"\nWrote per-reading results to {args.out}")


if __name__ == "__main__":
    main()
