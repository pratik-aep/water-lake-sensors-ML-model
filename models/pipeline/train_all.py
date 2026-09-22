"""Train every model on one dataset: the anomaly detector first, then the rest on the readings it cleaned."""

import argparse
import contextlib
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from ..air_quality import train as air_train
from ..algal_bloom import train as algae_train
from ..algal_bloom.config import TARGET as CHLOROPHYLL
from ..anomaly_detection import train as anomaly_train
from ..anomaly_detection.config import SENSORS
from ..anomaly_detection.data import load_readings
from ..bod_surrogate import train as bod_train
from ..coliform_nowcast import train as coliform_train
from ..coliform_nowcast.config import TARGET as COLIFORM
from ..forecasting import train as forecast_train
from ..forecasting.data import history_from_frame, load_weather, to_hourly
from ..wqi import train as wqi_train
from .ingest import load_mapping, normalise, turbidity_unit

HERE = Path(__file__).parent
MODEL_FILES = {
    "anomaly_detection": "anomaly_detector.joblib",
    "forecasting": "forecaster.joblib",
    "bod_surrogate": "bod_soft_sensor.joblib",
    "wqi": "wqi_sensor.joblib",
    "algal_bloom": "chlorophyll_soft_sensor.joblib",
    "coliform_nowcast": "coliform_nowcast.joblib",
    "air_quality": "air_forecaster.joblib",
}


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lab_with_sensors(lab: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    """Each lab sample beside the sensors' last complete hour before it was taken: the WQI model's training rows."""
    # A lab's own pH, turbidity or oxygen becomes the WQI label source (lab_*); the sensors stay the model inputs.
    lab = lab.rename(columns={s: f"lab_{s}" for s in SENSORS if s in lab.columns})
    lab = lab.assign(timestamp=pd.to_datetime(lab["timestamp"], errors="coerce"), station_id=lab["station_id"].astype(str))
    lab = lab.dropna(subset=["timestamp"]).sort_values("timestamp")
    hour_ends = hourly.assign(timestamp=hourly["timestamp"] + pd.Timedelta(hours=1)).sort_values("timestamp")
    joined = pd.merge_asof(
        lab, hour_ends, on="timestamp", by="station_id", direction="backward", tolerance=pd.Timedelta(hours=2)
    )
    return joined.dropna(subset=SENSORS).reset_index(drop=True)


def _read_metrics(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def attempt(train) -> str | None:
    """Run a model's training command; None if it trained, else why it stopped (too little data, a missing column).
    Models the system can run without are skipped this way instead of stopping the whole run."""
    errors = io.StringIO()
    try:
        with contextlib.redirect_stderr(errors):
            train()
    except SystemExit as stop:
        if stop.code:
            return (errors.getvalue().strip().splitlines() or [str(stop.code)])[-1].split("error: ")[-1]
    return None


def _dig(value, keys):
    for key in keys:
        value = value.get(key) if isinstance(value, dict) else None
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readings", type=Path, default=HERE / "data" / "sensor_readings.csv", help="10-minute sensor export")
    parser.add_argument("--vendor-mapping", type=Path,
                        help="column names and units of a vendor export (see vendor_mapping.json); the daily run reuses it")
    parser.add_argument("--weather", type=Path, default=HERE / "data" / "weather.csv", help="hourly weather")
    parser.add_argument("--no-weather", action="store_true")
    parser.add_argument("--lab", type=Path, default=HERE / "data" / "lab_results.csv",
                        help="lab results (optional): station_id, timestamp, bod, conductivity, nitrate, and optionally "
                             "chlorophyll_a, total_coliform; without them only the sensor-based models are trained")
    parser.add_argument("--air-readings", type=Path, default=HERE / "data" / "air_readings.csv",
                        help="air station readings (used if the file exists): station_id, timestamp, pm1..tsp, gases in ppm")
    parser.add_argument("--known-episodes", type=Path, default=HERE / "data" / "injected_episodes.csv",
                        help="known sensor faults and pollution events (e.g. the maintenance log), used if the file exists")
    parser.add_argument("--test-days", type=int, default=60, help="most recent days held out when testing each model")
    parser.add_argument("--no-challenger", action="store_true", help="skip the forecaster's CNN-LSTM (faster)")
    parser.add_argument("--out-dir", type=Path, default=HERE / "artifacts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if not args.readings.exists():
        parser.error(f"{args.readings} not found; run `python -m models.pipeline.simulate` or pass it explicitly")

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    source = args.readings
    mapping = load_mapping(args.vendor_mapping)
    if args.vendor_mapping:
        # Every model learns from the same normalised readings: our column names, oxygen in mg/L.
        normalised, ingest_report = normalise(pd.read_csv(args.readings, low_memory=False), mapping)
        args.readings = out / "normalised_readings.csv"
        normalised.to_csv(args.readings, index=False)
        print(f"Normalised the vendor export: {ingest_report}")
    use_weather = not args.no_weather and args.weather.exists()
    weather_args = ["--weather", str(args.weather)] if use_weather else ["--no-weather"]
    seed = ["--seed", str(args.seed)]
    has_lab = args.lab.exists()
    # Chlorophyll-a and total coliform are optional in a lab panel; without them their models are simply not trained.
    lab_columns = pd.read_csv(args.lab, nrows=1).columns if has_lab else []
    planned = ["anomaly_detection", "forecasting"]
    planned += ["bod_surrogate", "wqi"] if has_lab else []
    planned += ["algal_bloom"] if CHLOROPHYLL in lab_columns else []
    planned += ["coliform_nowcast"] if COLIFORM in lab_columns else []
    planned += ["air_quality"] if args.air_readings.exists() else []
    skipped = {} if has_lab else {name: "no lab results" for name in ("bod_surrogate", "wqi")}
    if has_lab and turbidity_unit(mapping) == "%" and "turbidity" not in lab_columns:
        # The WQI rates turbidity against 5 NTU; on the vendor's uncalibrated % scale that would mislabel every sample.
        planned.remove("wqi")
        skipped["wqi"] = ("turbidity is on the vendor's % scale and the lab gives no turbidity (NTU): calibrate the "
                          "sensor (turbidity_percent_to_ntu) or add lab turbidity")

    def stage(name, title):
        print(f"\n===== {planned.index(name) + 1}/{len(planned)}  {title} =====")


    # The core needs only the sensors: sensor checks run from the first months of data, and every other model is
    # added once there is enough data for it.
    stage("anomaly_detection", "Anomaly detector")
    known = ["--known-episodes", str(args.known_episodes)] if args.known_episodes.exists() else []
    anomaly_weather = ["--weather", str(args.weather)] if use_weather else []
    anomaly_train.main(["--data", str(args.readings), "--test-days", str(args.test_days), *known, *anomaly_weather,
                        "--out-dir", str(out / "anomaly_detection"), *seed])
    detector = joblib.load(out / "anomaly_detection" / MODEL_FILES["anomaly_detection"])["detector"]
    readings, _ = load_readings(args.readings)
    flagged = detector.detect(readings, load_weather(args.weather) if detector.uses_weather else None)
    cleaned = out / "cleaned_readings.csv"
    flagged.to_csv(cleaned, index=False)
    faulty = int(flagged[[f"{s}_fault" for s in SENSORS]].ne("").to_numpy().sum())
    print(f"Flagged {faulty} sensor values as faulty; the other models learn from the rest ({cleaned.name})")

    trained = ["anomaly_detection"]

    lab_args = ["--data", str(cleaned), "--lab", str(args.lab), *weather_args]
    wqi_rows = out / "wqi_training_rows.csv"
    rows = pd.DataFrame()
    if has_lab:
        rows = lab_with_sensors(pd.read_csv(args.lab), to_hourly(history_from_frame(flagged)[0], 3))
        rows.to_csv(wqi_rows, index=False)
    jobs = [
        ("forecasting", "Forecaster", lambda: forecast_train.main(
            ["--data", str(cleaned), *weather_args, "--test-days", str(args.test_days), "--out-dir", str(out / "forecasting"),
             *seed, *(["--no-challenger"] if args.no_challenger else [])])),
        ("bod_surrogate", "BOD soft sensor", lambda: bod_train.main([*lab_args, "--out-dir", str(out / "bod_surrogate"), *seed])),
        ("wqi", "WQI classifier", lambda: wqi_train.main(["--data", str(wqi_rows), "--features", "sensor", "--out-dir", str(out / "wqi"), *seed])),
        ("algal_bloom", "Algae (chlorophyll-a) soft sensor", lambda: algae_train.main([*lab_args, "--out-dir", str(out / "algal_bloom"), *seed])),
        ("coliform_nowcast", "Total coliform nowcast", lambda: coliform_train.main([*lab_args, "--out-dir", str(out / "coliform_nowcast"), *seed])),
        ("air_quality", "Air quality (AQI and particulate forecasts)", lambda: air_train.main(
            ["--data", str(args.air_readings), *(["--weather", str(args.weather)] if use_weather else []),
             "--test-days", str(args.test_days), "--out-dir", str(out / "air_quality"), *seed])),
    ]
    for name, title, train in jobs:
        if name not in planned:
            continue
        stage(name, title)
        reason = attempt(train)
        if reason is None:
            trained.append(name)
        else:
            skipped[name] = reason
            print(f"Skipped: {reason}")

    metrics = {name: _read_metrics(out / name / MODEL_FILES[name].replace(".joblib", "_metrics.json")) for name in trained}
    def get(name, *keys):
        return _dig(metrics.get(name, {}), keys)

    summary = {
        "anomaly_false_alarms_per_station_week": get("anomaly_detection", "clean_holdout", "false_alarms_per_station_week"),
        "forecaster_model": get("forecasting", "hourly_model"),
        "forecaster_low_oxygen_alert_csi": get("forecasting", "nightly_test", "alerts", "all", "critical_success_index"),
        "bod_model": get("bod_surrogate", "model"),
        "bod_test_mae_mg_l": get("bod_surrogate", "test", "soft_sensor", "mae"),
        "bod_last_lab_value_mae_mg_l": get("bod_surrogate", "test", "last_lab_value", "mae"),
        "wqi_model": get("wqi", "model_name"),
        "wqi_holdout_accuracy": get("wqi", "holdout_evaluation", "report", "accuracy"),
        "chlorophyll_test_mae_ug_l": get("algal_bloom", "test", "soft_sensor", "mae"),
        "chlorophyll_trophic_class_agreement": get("algal_bloom", "test", "trophic_class_agreement_with_lab"),
        "coliform_model": get("coliform_nowcast", "model"),
        "coliform_cpcb_band_agreement": get("coliform_nowcast", "test", "cpcb_band_agreement_with_lab"),
        "air_tomorrow_pm25_mae_ug_m3": get("air_quality", "daily_test", "1", "pm25", "mae"),
        "air_tomorrow_category_agreement": get("air_quality", "daily_test", "1", "category_agreement"),
    }
    summary = {k: v for k, v in summary.items() if v is not None}
    manifest = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "readings": str(source),
        "readings_sha256": _sha256(source),
        # The daily run applies the same mapping, so its readings arrive in the units the models learned.
        "vendor_mapping": mapping,
        "turbidity_unit": turbidity_unit(mapping),
        "weather": str(args.weather) if use_weather else None,
        "lab": str(args.lab) if has_lab else None,
        "lab_sha256": _sha256(args.lab) if has_lab else None,
        "models": {name: f"{name}/{MODEL_FILES[name]}" for name in trained},
        "skipped": skipped,
        "sensor_values_flagged_faulty": faulty,
        "wqi_training_rows": len(rows),
        "summary": summary,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\n===== Trained {len(trained)} model{'' if len(trained) == 1 else 's'} =====")
    for key, value in summary.items():
        print(f"  {key:<40} {value}")
    for name, reason in skipped.items():
        print(f"  not trained: {name:<27} {reason}")
    print(f"Manifest: {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
