"""Train every model on one dataset: the anomaly detector first, then the rest on the readings it cleaned."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

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

HERE = Path(__file__).parent
MODEL_FILES = {
    "anomaly_detection": "anomaly_detector.joblib",
    "forecasting": "forecaster.joblib",
    "bod_surrogate": "bod_soft_sensor.joblib",
    "wqi": "wqi_sensor.joblib",
    "algal_bloom": "chlorophyll_soft_sensor.joblib",
    "coliform_nowcast": "coliform_nowcast.joblib",
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readings", type=Path, default=HERE / "data" / "sensor_readings.csv", help="10-minute sensor export")
    parser.add_argument("--weather", type=Path, default=HERE / "data" / "weather.csv", help="hourly weather")
    parser.add_argument("--no-weather", action="store_true")
    parser.add_argument("--lab", type=Path, default=HERE / "data" / "lab_results.csv",
                        help="lab results: station_id, timestamp, bod, conductivity, nitrate (lab pH/turbidity/DO optional)")
    parser.add_argument("--known-episodes", type=Path, default=HERE / "data" / "injected_episodes.csv",
                        help="known sensor faults and pollution events (e.g. the maintenance log), used if the file exists")
    parser.add_argument("--test-days", type=int, default=60, help="most recent days held out when testing each model")
    parser.add_argument("--no-challenger", action="store_true", help="skip the forecaster's CNN-LSTM (faster)")
    parser.add_argument("--out-dir", type=Path, default=HERE / "artifacts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    for path in (args.readings, args.lab):
        if not path.exists():
            parser.error(f"{path} not found; run `python -m models.pipeline.simulate` or pass it explicitly")

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    use_weather = not args.no_weather and args.weather.exists()
    weather_args = ["--weather", str(args.weather)] if use_weather else ["--no-weather"]
    seed = ["--seed", str(args.seed)]

    # Chlorophyll-a and total coliform are optional in a lab panel; without them their models are simply not trained.
    lab_columns = pd.read_csv(args.lab, nrows=1).columns
    has_chlorophyll, has_coliform = CHLOROPHYLL in lab_columns, COLIFORM in lab_columns
    stages = 4 + has_chlorophyll + has_coliform

    print(f"\n===== 1/{stages}  Anomaly detector =====")
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

    print(f"\n===== 2/{stages}  Forecaster =====")
    forecast_train.main(["--data", str(cleaned), *weather_args, "--test-days", str(args.test_days),
                         "--out-dir", str(out / "forecasting"), *seed, *(["--no-challenger"] if args.no_challenger else [])])

    print(f"\n===== 3/{stages}  BOD soft sensor =====")
    bod_train.main(["--data", str(cleaned), "--lab", str(args.lab), *weather_args, "--out-dir", str(out / "bod_surrogate"), *seed])

    print(f"\n===== 4/{stages}  WQI classifier =====")
    rows = lab_with_sensors(pd.read_csv(args.lab), to_hourly(history_from_frame(flagged)[0], 3))
    wqi_rows = out / "wqi_training_rows.csv"
    rows.to_csv(wqi_rows, index=False)
    wqi_train.main(["--data", str(wqi_rows), "--features", "sensor", "--out-dir", str(out / "wqi"), *seed])

    trained = ["anomaly_detection", "forecasting", "bod_surrogate", "wqi"]
    if has_chlorophyll:
        print(f"\n===== 5/{stages}  Algae (chlorophyll-a) soft sensor =====")
        algae_train.main(["--data", str(cleaned), "--lab", str(args.lab), *weather_args, "--out-dir", str(out / "algal_bloom"), *seed])
        trained.append("algal_bloom")
    if has_coliform:
        print(f"\n===== {len(trained) + 1}/{stages}  Total coliform nowcast =====")
        coliform_train.main(["--data", str(cleaned), "--lab", str(args.lab), *weather_args,
                             "--out-dir", str(out / "coliform_nowcast"), *seed])
        trained.append("coliform_nowcast")

    metrics = {name: _read_metrics(out / name / MODEL_FILES[name].replace(".joblib", "_metrics.json")) for name in trained}
    summary = {
        "anomaly_false_alarms_per_station_week": metrics["anomaly_detection"].get("clean_holdout", {}).get("false_alarms_per_station_week"),
        "forecaster_model": metrics["forecasting"].get("hourly_model"),
        "forecaster_low_oxygen_alert_csi": metrics["forecasting"].get("nightly_test", {}).get("alerts", {}).get("all", {}).get("critical_success_index"),
        "bod_model": metrics["bod_surrogate"].get("model"),
        "bod_test_mae_mg_l": metrics["bod_surrogate"].get("test", {}).get("soft_sensor", {}).get("mae"),
        "bod_last_lab_value_mae_mg_l": metrics["bod_surrogate"].get("test", {}).get("last_lab_value", {}).get("mae"),
        "wqi_model": metrics["wqi"].get("model_name"),
        "wqi_holdout_accuracy": metrics["wqi"].get("holdout_evaluation", {}).get("report", {}).get("accuracy"),
    }
    if has_chlorophyll:
        algae = metrics["algal_bloom"]
        summary["chlorophyll_test_mae_ug_l"] = algae.get("test", {}).get("soft_sensor", {}).get("mae")
        summary["chlorophyll_trophic_class_agreement"] = algae.get("test", {}).get("trophic_class_agreement_with_lab")
    if has_coliform:
        coliform = metrics["coliform_nowcast"]
        summary["coliform_model"] = coliform.get("model")
        summary["coliform_cpcb_band_agreement"] = coliform.get("test", {}).get("cpcb_band_agreement_with_lab")
    manifest = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "readings": str(args.readings),
        "readings_sha256": _sha256(args.readings),
        "weather": str(args.weather) if use_weather else None,
        "lab": str(args.lab),
        "lab_sha256": _sha256(args.lab),
        "models": {name: f"{name}/{MODEL_FILES[name]}" for name in trained},
        "sensor_values_flagged_faulty": faulty,
        "wqi_training_rows": len(rows),
        "summary": summary,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print("\n===== All models trained =====")
    for key, value in summary.items():
        print(f"  {key:<40} {value}")
    print(f"Manifest: {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
