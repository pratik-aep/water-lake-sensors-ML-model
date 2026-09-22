"""Fit the anomaly detector, test it on held-out readings with injected faults, and save it."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from ..forecasting.data import load_weather
from .config import INTERVAL
from .data import load_readings
from .detector import AnomalyDetector
from .evaluate import evaluate
from .inject import KINDS, inject

HERE = Path(__file__).parent
NO_EPISODES = pd.DataFrame(columns=["station_id", "kind", "sensor", "start", "end"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "data" / "synthetic_sensor_readings.csv")
    parser.add_argument("--test-days", type=int, default=90, help="most recent days held out for testing")
    parser.add_argument("--per-kind", type=int, default=2, help="injected episodes of each kind per station")
    parser.add_argument("--known-episodes", type=Path,
                        help="known faults and events (station_id, kind, sensor, start, end), e.g. from the maintenance "
                             "log; alarms they explain are not counted as false")
    parser.add_argument("--weather", type=Path,
                        help="hourly weather (timestamp, rain_mm); lets the detector tell runoff from a fouled sensor")
    parser.add_argument("--out-dir", type=Path, default=HERE / "artifacts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if not args.data.exists():
        parser.error(f"{args.data} not found; run `python -m models.anomaly_detection.simulate` or pass --data")

    df, report = load_readings(args.data)
    cutoff = df["timestamp"].max() - pd.Timedelta(days=args.test_days)
    train_df, holdout = df[df["timestamp"] <= cutoff], df[df["timestamp"] > cutoff]
    if train_df.empty or holdout.empty:
        parser.error(f"the data spans less than --test-days ({args.test_days}); lower it")

    weather = load_weather(args.weather) if args.weather else None
    detector = AnomalyDetector(seed=args.seed).fit(train_df, weather)
    merge_gap = pd.Timedelta(INTERVAL) * (detector.settings["merge_gap_points"] + 1)
    # Score the holdout as a live stream would see it: with the preceding week as history for rolling baselines.
    context = train_df[train_df["timestamp"] > cutoff - pd.Timedelta(days=detector.settings["baseline_days"])]

    def detect_holdout(readings):
        result = detector.detect(pd.concat([context, readings]), weather)
        return result[result["timestamp"] > cutoff]

    known = NO_EPISODES
    if args.known_episodes:
        known = pd.read_csv(args.known_episodes, parse_dates=["start", "end"]).fillna({"sensor": ""})
        known = known[known["end"] > cutoff]
    clean_scores = evaluate(detect_holdout(holdout), known, merge_gap)
    dirty, episodes = inject(holdout, seed=args.seed, per_kind=args.per_kind)
    scores = evaluate(detect_holdout(dirty), pd.concat([episodes, known], ignore_index=True), merge_gap)

    final = AnomalyDetector(seed=args.seed).fit(df, weather)
    bundle = {
        "detector": final,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "data_file": str(args.data),
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "rows": len(df),
        "stations": report["stations"],
        "date_range": report["date_range"],
        "settings": final.settings,
        "drift_limit": final.drift_limit_,
        "uses_weather": final.uses_weather,
    }
    metrics = {
        **{k: v for k, v in bundle.items() if k != "detector"},
        "data_report": report,
        "holdout": {
            "from": str(holdout["timestamp"].min()),
            "days": args.test_days,
            "note": "Scores come from a detector fit before the holdout; the saved detector is refit on all rows.",
        },
        "clean_holdout": {
            k: clean_scores[k] for k in ("incidents", "false_alarms_per_station_week", "consistency_coverage", "per_kind")
        },
        "known_episodes_in_holdout": len(known),
        "injected_holdout": scores,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "anomaly_detector.joblib"
    joblib.dump(bundle, model_path)
    (args.out_dir / "anomaly_detector_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    print(f"Data: {report['rows_out']} readings across {len(report['stations'])} stations, {report['date_range'][0]} to {report['date_range'][1]}")
    print(f"Fit on {len(train_df)} readings; tested on the last {args.test_days} days ({len(holdout)} readings)\n")
    print(f"Injected-fault test ({args.per_kind} of each kind per station, scored per episode):")
    print(f"  {'kind':<12} {'caught':>8} {'recall':>7} {'median delay':>13}")
    for kind in KINDS:
        row = scores["per_kind"].get(kind)
        if row:
            delay = "-" if row["median_delay_hours"] is None else f"{row['median_delay_hours']:.1f} h"
            print(f"  {kind:<12} {row['detected']:>4}/{row['episodes']:<3} {row['recall']:>7.0%} {delay:>13}")
    print(f"\nEvents wrongly blamed on a sensor: {scores['events_blamed_on_a_sensor']}")
    print(f"Alarm incidents: {scores['incidents']}, of which unexplained (false): {scores['false_incidents']}")
    holdout_name = "the holdout (known episodes excluded)" if len(known) else "the clean holdout"
    print(f"False alarms per station-week: {scores['false_alarms_per_station_week']} with faults present, "
          f"{clean_scores['false_alarms_per_station_week']} on {holdout_name}")
    if len(known):
        caught = sum(v["detected"] for v in clean_scores["per_kind"].values())
        print(f"Known episodes already in the holdout: {caught} of {len(known)} caught "
              f"({ {k: v['detected'] for k, v in clean_scores['per_kind'].items()} })")
    print(f"Cross-sensor checks ran on this share of readings: {clean_scores['consistency_coverage']}")
    print(f"\nDrift alarm levels (average disagreement, in error scales): { {k: round(v, 2) for k, v in final.drift_limit_.items()} }")
    print(f"Saved detector (refit on all {len(df)} readings) to {model_path}")


if __name__ == "__main__":
    main()
