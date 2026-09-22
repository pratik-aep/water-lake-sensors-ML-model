"""Fit the forecasters, test them on the most recent period, keep the best hourly model, and save."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from .config import DEFAULT_SETTINGS
from .data import load_history, load_weather, to_hourly
from .evaluate import hourly_report, nightly_report, relative_error, summary_table
from .forecaster import HourlyForecaster, NightlyForecaster, make_timelines, station_codes

HERE = Path(__file__).parent


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _weeks(settings, start, end):
    """Consecutive test weeks, each with the recalibration window that precedes it."""
    step = pd.Timedelta(days=settings["recalibrate_every_days"])
    week = start
    while week < end:
        yield week - pd.Timedelta(days=settings["recalibrate_days"]), week, min(week + step, end)
        week += step


def _test_hourly(model, timelines, settings, test_start, end):
    """Test forecasts as live use would make them: ranges recalibrated weekly on the previous verified forecasts."""
    horizon = pd.Timedelta(hours=max(settings["horizons"]))
    parts = []
    for recal_from, week, stop in _weeks(settings, test_start, end):
        origins = model.origins(timelines, week, min(stop + horizon, end))
        if any(len(o) for o in origins.values()):
            model.calibrate(timelines, recal_from, week)
            parts.append(model.predict(timelines, origins, simulated_forecast=True))
    return pd.concat(parts, ignore_index=True)


def _test_nightly(model, hourly, timelines, settings, test_start, end):
    last_issue = end - pd.Timedelta(days=settings["night_days"] + 1)
    parts = []
    for recal_from, week, stop in _weeks(settings, test_start, last_issue):
        issue = model.issue_dates(timelines, week, stop)
        if any(len(d) for d in issue.values()):
            model.calibrate(timelines, recal_from, week, hourly)
            parts.append(model.predict(timelines, issue, simulated_forecast=True, hourly=hourly))
    return pd.concat(parts, ignore_index=True)


def _validate(model, timelines, cal_start, test_start):
    """Median accuracy on the validation window, relative to 'no change'; used to choose the hourly model."""
    return relative_error(model.predict(timelines, model.origins(timelines, cal_start, test_start), simulated_forecast=True))


def _refit(settings, kind, fitted, timelines, codes, seed, until):
    """The chosen hourly model refit on all data; a CNN-LSTM keeps the epoch count that validated best."""
    if kind == "ensemble":
        return HourlyForecaster.combine(
            [_refit(settings, k, fitted, timelines, codes, seed, until) for k in ("gradient_boosting", "cnn_lstm")]
        )
    if kind == "cnn_lstm":
        settings = {**settings, "challenger": {**settings["challenger"], "epochs": fitted[kind].model_.epochs_run_}}
    return HourlyForecaster(settings, kind, seed).fit(timelines, codes, until)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "data" / "synthetic_sensor_readings.csv",
                        help="10-minute readings; the anomaly detector's --out file works too (faulty values are dropped)")
    parser.add_argument("--weather", type=Path, default=HERE / "data" / "synthetic_weather.csv",
                        help="hourly air_temperature, cloud_cover, rain_mm")
    parser.add_argument("--no-weather", action="store_true", help="train without weather inputs")
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--calibration-days", type=int, default=30)
    parser.add_argument("--no-challenger", action="store_true", help="skip the CNN-LSTM challenger and the ensemble")
    parser.add_argument("--out-dir", type=Path, default=HERE / "artifacts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if not args.data.exists():
        parser.error(f"{args.data} not found; run `python -m models.forecasting.simulate` or pass --data")

    settings = dict(DEFAULT_SETTINGS)
    readings, report = load_history(args.data)
    hourly = to_hourly(readings, settings["min_valid_per_hour"])
    use_weather = not args.no_weather and args.weather.exists()
    weather = load_weather(args.weather) if use_weather else None

    end = hourly["timestamp"].max() + pd.Timedelta(hours=1)
    test_start = end - pd.Timedelta(days=args.test_days)
    cal_start = test_start - pd.Timedelta(days=args.calibration_days)
    if cal_start - hourly["timestamp"].min() < pd.Timedelta(days=60):
        parser.error("need at least 60 days of history before the calibration and test periods")
    codes = station_codes(hourly["station_id"].unique())
    timelines = make_timelines(hourly, weather, settings)

    fitted = {}
    for kind in ["gradient_boosting"] + ([] if args.no_challenger else ["cnn_lstm"]):
        fitted[kind] = HourlyForecaster(settings, kind, args.seed).fit(timelines, codes, cal_start, validation=(cal_start, test_start))
    if len(fitted) > 1:
        fitted["ensemble"] = HourlyForecaster.combine([fitted["gradient_boosting"], fitted["cnn_lstm"]])
    # Chosen on the validation window, never on the test period, so the test scores stay honest.
    validation_scores = {kind: _validate(model, timelines, cal_start, test_start) for kind, model in fitted.items()}
    winner = min(validation_scores, key=validation_scores.get)
    tests = {kind: _test_hourly(model, timelines, settings, test_start, end) for kind, model in fitted.items()}

    nightly = NightlyForecaster(settings, args.seed).fit(timelines, codes, cal_start)
    nightly_test = _test_nightly(nightly, fitted[winner], timelines, settings, test_start, end)

    weather_value = None
    if use_weather:
        # Like for like: the same gradient-boosting setup with and without weather inputs.
        dry = make_timelines(hourly, None, settings)
        dry_hourly = HourlyForecaster(settings, "gradient_boosting", args.seed).fit(dry, codes, cal_start)
        dry_nightly = NightlyForecaster(settings, args.seed).fit(dry, codes, cal_start)
        wet_nightly = NightlyForecaster(settings, args.seed).fit(timelines, codes, cal_start)
        compare = {
            "with_weather": (tests["gradient_boosting"], _test_nightly(wet_nightly, fitted["gradient_boosting"], timelines, settings, test_start, end)),
            "without_weather": (_test_hourly(dry_hourly, dry, settings, test_start, end), _test_nightly(dry_nightly, dry_hourly, dry, settings, test_start, end)),
        }
        weather_value = {}
        for name, (hourly_fc, nightly_fc) in compare.items():
            alerts = nightly_report(nightly_fc, settings["do_alert_mg_l"], settings["alert_probability"])["alerts"]
            weather_value[name] = {
                "hourly_relative_error": round(relative_error(hourly_fc), 4),
                "nightly_alerts": {g: {k: a[k] for k in ("critical_success_index", "brier_skill")} for g, a in alerts.items()},
            }

    # Deploy: refit on everything. Ranges are calibrated on the last month of forecasts made before that data
    # was trained on, so they reflect unseen conditions; forecast.py recalibrates as new data arrives.
    recent = end - pd.Timedelta(days=settings["recalibrate_days"])
    fitted[winner].calibrate(timelines, recent, end)
    nightly.calibrate(timelines, recent, end, fitted[winner])
    final_hourly = _refit(settings, winner, fitted, timelines, codes, args.seed, end)
    final_hourly.offsets_ = fitted[winner].offsets_
    final_nightly = NightlyForecaster(settings, args.seed).fit(timelines, codes, end)
    final_nightly.offsets_ = nightly.offsets_

    bundle = {
        "hourly": final_hourly,
        "nightly": final_nightly,
        "codes": codes,
        "settings": settings,
        "uses_weather": use_weather,
        "hourly_model": winner,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "data_file": str(args.data),
        "data_sha256": _sha256(args.data),
        "weather_file": str(args.weather) if use_weather else None,
        "weather_sha256": _sha256(args.weather) if use_weather else None,
        "stations": sorted(codes),
        "date_range": [str(hourly["timestamp"].min()), str(hourly["timestamp"].max())],
    }
    hourly_reports = {kind: hourly_report(test, settings["horizon_buckets"]) for kind, test in tests.items()}
    nightly_scores = nightly_report(nightly_test, settings["do_alert_mg_l"], settings["alert_probability"])
    metrics = {
        **{k: v for k, v in bundle.items() if k not in ("hourly", "nightly")},
        "data_report": report,
        "periods": {"validation_from": str(cal_start), "test_from": str(test_start), "end": str(end)},
        "validation_relative_error": validation_scores,
        "test_relative_error": {kind: relative_error(test) for kind, test in tests.items()},
        "hourly_test": hourly_reports,
        "nightly_test": nightly_scores,
        "weather_value": weather_value,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "forecaster.joblib"
    joblib.dump(bundle, model_path)
    (args.out_dir / "forecaster_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    print(f"Data: {len(hourly)} station-hours, {bundle['date_range'][0]} to {bundle['date_range'][1]}; "
          f"weather {'used' if use_weather else 'not used'}")
    print(f"Validation from {cal_start.date()}, test from {test_start.date()} ({args.test_days} days)\n")
    print("Hourly model error relative to 'no change' (lower is better); chosen on validation:")
    for kind, score in validation_scores.items():
        print(f"  {kind:<18} validation {score:.3f}   test {metrics['test_relative_error'][kind]:.3f}"
              f"{'   <- kept' if kind == winner else ''}")
    columns = ["sensor", "horizon", "mae", "mae_persistence", "mae_same_hour_yesterday", "skill", "coverage_80"]
    print(f"\nHourly test results ({winner}); skill is vs the better baseline, coverage is for the 80% range:")
    print(summary_table(hourly_reports[winner])[columns].to_string(index=False))
    print("\nNightly oxygen minimum (mg/L):")
    print(pd.DataFrame(nightly_scores["by_days_ahead"]).T.to_string())
    print(f"\nAlerts for nights below {settings['do_alert_mg_l']} mg/L (probability >= {settings['alert_probability']}):")
    for group, s in nightly_scores["alerts"].items():
        print(f"  {group:<9} events {s['events']:>3}  detected {s['probability_of_detection']}  "
              f"false-alarm ratio {s['false_alarm_ratio']}  CSI {s['critical_success_index']}  "
              f"Brier skill {s['brier_skill']}")
    if weather_value:
        print("\nValue of weather forecasts (gradient boosting with vs without weather inputs):")
        for name, v in weather_value.items():
            csi = {g: a["critical_success_index"] for g, a in v["nightly_alerts"].items()}
            print(f"  {name:<16} hourly relative error {v['hourly_relative_error']:.3f}; nightly alert CSI {csi}")
    print(f"\nSaved {winner} hourly + nightly forecasters (refit on all data) to {model_path}")


if __name__ == "__main__":
    main()
