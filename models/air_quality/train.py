"""Fit the particulate forecasters, test them on the most recent period, and save them."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from ..forecasting.calibration import horizon_bucket
from ..forecasting.evaluate import alert_scores
from .config import DEFAULT_SETTINGS
from .forecast import AirTimeline, DailyForecaster, HourlyForecaster
from .naqi import category
from .qc import check, to_hourly

HERE = Path(__file__).parent


def load(readings_path, weather_path, settings):
    readings = pd.read_csv(readings_path, low_memory=False)
    weather = pd.read_csv(weather_path) if weather_path and Path(weather_path).exists() else None
    if weather is not None:
        weather["timestamp"] = pd.to_datetime(weather["timestamp"])
    return check(to_hourly(readings, settings), weather, settings), weather


def timelines_for(hourly: pd.DataFrame, weather, settings) -> dict:
    ahead = max(max(settings["horizons"]), 24 * (max(settings["days_ahead"]) + 1))
    return {st: AirTimeline(st, g[["timestamp", *settings["targets"]]], weather, settings["targets"], ahead)
            for st, g in hourly.groupby("station_id", sort=True)}


def _weeks(start, end, calibration_days):
    week = start
    while week < end:
        yield week - pd.Timedelta(days=calibration_days), week, min(week + pd.Timedelta(days=7), end)
        week += pd.Timedelta(days=7)


def _mae(a, b) -> float:
    ok = ~(np.isnan(a) | np.isnan(b))
    return round(float(np.mean(np.abs(a[ok] - b[ok]))), 2)


def hourly_scores(fc: pd.DataFrame, settings) -> dict:
    fc = fc[fc["actual"].notna()].assign(bucket=lambda d: horizon_bucket(d["horizon"].to_numpy(), settings["horizon_buckets"]))
    out = {}
    for (pollutant, bucket), g in fc.groupby(["pollutant", "bucket"], sort=False):
        model, persistence = _mae(g["mid"].to_numpy(), g["actual"].to_numpy()), _mae(g["persistence"].to_numpy(), g["actual"].to_numpy())
        out.setdefault(pollutant, {})[bucket] = {
            "mae": model, "mae_persistence": persistence, "skill": round(1 - model / persistence, 3) if persistence else None,
            "coverage_80": round(float(g["actual"].between(g["lo"], g["hi"]).mean()), 3),
        }
    return out


def daily_scores(fc: pd.DataFrame, settings) -> dict:
    fc = fc[fc["actual_pm_index"].notna()]
    out = {}
    for days, g in fc.groupby("days_ahead"):
        row = {}
        for t in settings["targets"]:
            actual = g[f"{t}_actual"].to_numpy()
            persistence = _mae(g[f"{t}_last24h"].to_numpy(), actual)
            row[t] = {"mae": _mae(g[t].to_numpy(), actual), "mae_persistence": persistence,
                      "coverage_80": round(float(g[f"{t}_actual"].between(g[f"{t}_lo"], g[f"{t}_hi"]).mean()), 3)}
        row["category_agreement"] = round(float((g["pm_category"] == category(g["actual_pm_index"])).mean()), 3)
        event = (g["actual_pm_index"] > settings["alert_index"]).to_numpy()
        chance = g["p_poor_or_worse"].to_numpy()
        row["poor_day_alerts"] = alert_scores(chance, chance >= settings["alert_probability"], event)
        out[int(days)] = row
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "data" / "synthetic_air_readings.csv",
                        help="air readings: station_id, timestamp, pm1, pm25, pm10, tsp, and gases in ppm")
    parser.add_argument("--weather", type=Path, default=HERE / "data" / "synthetic_weather.csv",
                        help="hourly weather with wind_speed, boundary_layer_height, relative_humidity (models.pipeline.weather)")
    parser.add_argument("--test-days", type=int, default=DEFAULT_SETTINGS["test_days"])
    parser.add_argument("--out-dir", type=Path, default=HERE / "artifacts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if not args.data.exists():
        parser.error(f"{args.data} not found; run `python -m models.air_quality.simulate` or pass --data")

    s = {**DEFAULT_SETTINGS, "test_days": args.test_days}
    hourly, weather = load(args.data, args.weather, s)
    timelines = timelines_for(hourly, weather, s)
    codes = {st: float(i) for i, st in enumerate(sorted(timelines))}
    end = hourly["timestamp"].max() + pd.Timedelta(hours=1)
    test_start = end - pd.Timedelta(days=s["test_days"])
    cal_start = test_start - pd.Timedelta(days=s["calibration_days"])
    if cal_start - hourly["timestamp"].min() < pd.Timedelta(days=45):
        parser.error("need at least 45 days of readings before the calibration and test periods")

    def rng(stream):  # simulated weather-forecast error, reproducible per use
        return np.random.default_rng(args.seed + stream) if weather is not None else None

    hourly_model = HourlyForecaster(s, args.seed).fit(timelines, codes, cal_start, rng(1))
    daily_model = DailyForecaster(s, args.seed).fit(timelines, codes, cal_start, rng(2))
    hourly_parts, daily_parts = [], []
    for recal_from, week, stop in _weeks(test_start, end, s["calibration_days"]):
        hourly_model.calibrate(timelines, recal_from, week, rng(3))
        origins = hourly_model.origins(timelines, week, stop, s["eval_origin_every_hours"])
        if any(len(o) for o in origins.values()):
            hourly_parts.append(hourly_model.predict(timelines, origins, rng(4)))
        daily_model.calibrate(timelines, recal_from, week, rng(5), hourly_model)
        issue = daily_model.issue_dates(timelines, week, min(stop, end - pd.Timedelta(days=max(s["days_ahead"]) + 1)))
        if any(len(d) for d in issue.values()):
            daily_parts.append(daily_model.predict(timelines, issue, rng(6), hourly_model))
    hourly_test, daily_test = pd.concat(hourly_parts, ignore_index=True), pd.concat(daily_parts, ignore_index=True)

    # Deploy: refit on everything; ranges calibrated on the last month of forecasts the refit has not trained on.
    recent = end - pd.Timedelta(days=s["calibration_days"])
    hourly_model.calibrate(timelines, recent, end, rng(3))
    daily_model.calibrate(timelines, recent, end, rng(5), hourly_model)
    final_hourly = HourlyForecaster(s, args.seed).fit(timelines, codes, end, rng(1))
    final_daily = DailyForecaster(s, args.seed).fit(timelines, codes, end, rng(2))
    final_hourly.offsets_, final_daily.offsets_ = hourly_model.offsets_, daily_model.offsets_

    bundle = {
        "hourly": final_hourly, "daily": final_daily, "codes": codes, "settings": s,
        "uses_weather": weather is not None,
        "weather_columns": next(iter(timelines.values())).weather_columns,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "data_file": str(args.data), "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "stations": sorted(codes), "date_range": [str(hourly["timestamp"].min()), str(hourly["timestamp"].max())],
    }
    flags = {c.removesuffix("_flag"): hourly[c].replace("", np.nan).value_counts().to_dict() for c in hourly if c.endswith("_flag")}
    metrics = {**{k: v for k, v in bundle.items() if k not in ("hourly", "daily")},
               "test_from": str(test_start), "qc_flags": {k: v for k, v in flags.items() if v},
               "hourly_test": hourly_scores(hourly_test, s), "daily_test": daily_scores(daily_test, s)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.out_dir / "air_forecaster.joblib")
    (args.out_dir / "air_forecaster_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    print(f"Air data: {len(hourly)} station-hours, {bundle['date_range'][0]} to {bundle['date_range'][1]}; test from {test_start.date()}")
    print(f"Sensor checks set aside or flagged: {metrics['qc_flags']}")
    print("\nHourly forecast (ug/m3): MAE vs no-change forecast, and how often the 80% range held")
    for pollutant, buckets in metrics["hourly_test"].items():
        for bucket, m in buckets.items():
            print(f"  {pollutant:<5} {bucket:>7}  MAE {m['mae']:>6}  (no change {m['mae_persistence']:>6}, skill {m['skill']})  coverage {m['coverage_80']}")
    print("\nDaily means and the particulate AQI:")
    for days, row in metrics["daily_test"].items():
        a = row["poor_day_alerts"]
        print(f"  {days} day(s) ahead: PM2.5 MAE {row['pm25']['mae']} (no change {row['pm25']['mae_persistence']}; "
              f"coverage {row['pm25']['coverage_80']}), "
              f"category right {row['category_agreement']:.0%}; Poor+ days {a['events']}, detected {a['probability_of_detection']}, "
              f"false-alarm ratio {a['false_alarm_ratio']}, Brier skill {a['brier_skill']}")
    print(f"\nSaved to {args.out_dir / 'air_forecaster.joblib'}")


if __name__ == "__main__":
    main()
