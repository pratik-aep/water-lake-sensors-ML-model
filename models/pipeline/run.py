"""Daily run: check every sensor, forecast, estimate BOD, algae and coliform, and classify each lake, in one report."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from ..air_quality.estimate import air_status
from ..algal_bloom.estimate import estimate_daily as estimate_algae
from ..anomaly_detection.config import INTERVAL, SENSORS
from ..anomaly_detection.data import prepare
from ..anomaly_detection.evaluate import alarm_incidents
from ..bod_surrogate.cpcb import assess
from ..bod_surrogate.estimate import estimate_daily
from ..coliform_nowcast.estimate import estimate_daily as estimate_coliform
from ..forecasting.config import DEFAULT_SETTINGS as FORECAST_SETTINGS
from ..forecasting.data import history_from_frame, load_weather, to_hourly
from ..forecasting.forecast import make_forecast, with_wqi_class
from ..wqi.predict import classify
from .ingest import load_mapping, normalise, turbidity_unit
from .dashboard import air_cards
from .dashboard import render as render_dashboard
from .report import SEVERITY, air_report, render_text, station_report

HERE = Path(__file__).parent
# Report-level settings; each model keeps its own thresholds in its saved bundle.
STALE_HOURS = 3


def cpcb_with_coliform(bod: pd.DataFrame, coliform: pd.DataFrame) -> pd.DataFrame:
    """Re-grade each day's CPCB class with the coliform nowcast alongside BOD, DO and pH."""
    probs = [c for c in coliform.columns if c.startswith("p_above_")]
    both = bod.drop(columns=["cpcb_class", "cpcb_confidence", "cpcb_not_assessed"]).merge(
        coliform[["station_id", "timestamp", *probs]].rename(columns={p: f"coliform_{p}" for p in probs}),
        on=["station_id", "timestamp"], how="left",
    )
    has = both[[f"coliform_{p}" for p in probs]].notna().all(axis=1).to_numpy()
    bod_ok = {lv: 1 - both[f"p_above_{lv:g}"].to_numpy() for lv in (2.0, 3.0)}
    graded = assess(both["dissolved_oxygen"], both["ph"], bod_ok)
    if has.any():
        coliform_ok = {float(p.removeprefix("p_above_")): 1 - both[f"coliform_{p}"].to_numpy()[has] for p in probs}
        with_c = assess(both["dissolved_oxygen"][has], both["ph"][has], {lv: v[has] for lv, v in bod_ok.items()}, coliform_ok)
        graded.loc[has] = with_c.to_numpy()
    return pd.concat([both, graded], axis=1)


def latest_wqi(wqi_bundle: dict, cleaned: pd.DataFrame) -> dict:
    """WQI class from each station's latest hour with every sensor reading."""
    hourly = to_hourly(cleaned, 3).dropna(subset=SENSORS)
    latest = hourly.groupby("station_id").tail(1).reset_index(drop=True)
    classes = classify(wqi_bundle, latest)
    return {station: classes.iloc[i] for i, station in enumerate(latest["station_id"])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readings", type=Path, required=True, help="recent 10-minute sensor export (two weeks or more)")
    parser.add_argument("--vendor-mapping", type=Path, help="override the column and unit mapping saved at training")
    parser.add_argument("--air-readings", type=Path, help="recent air station readings (eight days or more), if there are air stations")
    parser.add_argument("--weather", type=Path, help="hourly weather including forecasts for the coming week")
    parser.add_argument("--models-dir", type=Path, default=HERE / "artifacts", help="output of train_all")
    parser.add_argument("--window-hours", type=int, default=24, help="how far back incidents are reported")
    parser.add_argument("--out-dir", type=Path, default=HERE / "reports")
    args = parser.parse_args(argv)

    manifest = json.loads((args.models_dir / "manifest.json").read_text())
    bundles = {name: joblib.load(args.models_dir / path) for name, path in manifest["models"].items()}
    # The fault detector always exists; the forecaster and lab-based models once there was enough data to train them.
    forecaster, bod_model, wqi_model = bundles.get("forecasting"), bundles.get("bod_surrogate"), bundles.get("wqi")
    algae_model, coliform_model = bundles.get("algal_bloom"), bundles.get("coliform_nowcast")
    detector = bundles["anomaly_detection"]["detector"]
    needs_weather = detector.uses_weather or any(
        b["uses_weather"] for b in (forecaster, bod_model, algae_model, coliform_model) if b is not None
    )
    if needs_weather and args.weather is None:
        parser.error("these models use weather; pass --weather with forecasts for the coming week")
    air_model = bundles.get("air_quality")
    if air_model is not None and args.air_readings is not None and air_model["uses_weather"] and args.weather is None:
        parser.error("the air model uses weather; pass --weather")
    weather = load_weather(args.weather) if needs_weather else None

    mapping = load_mapping(args.vendor_mapping) if args.vendor_mapping else manifest.get("vendor_mapping", load_mapping(None))
    normalised, ingest_report = normalise(pd.read_csv(args.readings, low_memory=False), mapping)
    readings, _ = prepare(normalised)
    flagged = detector.detect(readings, weather if detector.uses_weather else None)
    now = flagged["timestamp"].max()
    recent = flagged[flagged["timestamp"] > now - pd.Timedelta(hours=args.window_hours)]
    merge_gap = pd.Timedelta(INTERVAL) * (detector.settings["merge_gap_points"] + 1)
    incidents = alarm_incidents(recent, merge_gap)

    cleaned, _ = history_from_frame(flagged)
    forecast = make_forecast(forecaster, cleaned, weather) if forecaster is not None else {
        "hourly": pd.DataFrame(), "nightly": pd.DataFrame(), "recalibrated": False, "weather_short": False, "skipped": []}
    hourly_fc = forecast["hourly"]
    if wqi_model is not None and len(hourly_fc):
        hourly_fc = with_wqi_class(hourly_fc, wqi_model)
    bod = estimate_daily(bod_model, cleaned, weather)[0] if bod_model is not None else pd.DataFrame()
    coliform = estimate_coliform(coliform_model, cleaned, weather)[0] if coliform_model is not None else pd.DataFrame()
    coliform_latest = {st: g.iloc[-1] for st, g in coliform.groupby("station_id")} if len(coliform) else {}
    if len(bod) and len(coliform):
        bod = cpcb_with_coliform(bod, coliform)
    bod_latest = {st: g.iloc[-1] for st, g in bod.groupby("station_id")} if len(bod) else {}
    algae = estimate_algae(algae_model, cleaned, weather)[0] if algae_model is not None else pd.DataFrame()
    algae_latest = {st: g.iloc[-1] for st, g in algae.groupby("station_id")} if len(algae) else {}
    wqi_now = latest_wqi(wqi_model, cleaned) if wqi_model is not None else {}

    forecast_settings = forecaster["settings"] if forecaster is not None else FORECAST_SETTINGS
    settings = {
        "alert_probability": forecast_settings["alert_probability"],
        "do_alert_mg_l": forecast_settings["do_alert_mg_l"],
        "stale_hours": STALE_HOURS,
        "turbidity_unit": turbidity_unit(mapping),
    }
    stations, alerts = {}, []
    for station in sorted(flagged["station_id"].unique()):
        stations[station], raised = station_report(
            station, recent, incidents, hourly_fc, forecast["nightly"], bod_latest.get(station), wqi_now.get(station),
            settings, algae_latest.get(station), coliform_latest.get(station),
        )
        alerts += raised
    notes = [(st, "medium", "No forecast today: not every sensor has reported in the last 24 hours") for st in forecast["skipped"]]
    for station, sensors in ingest_report["censored_by_station"].items():
        for sensor, n in sensors.items():
            low, high = mapping["sensor_ranges"][sensor]
            span = f"{'' if low is None else low}-{'' if high is None else high}"
            notes.append((station, "low", f"{sensor} sat at the sensor's range limit ({span}) on {n} readings in this "
                                          "report's data: the lake went beyond what the sensor can measure"))
    for station, severity, message in notes:
        note = {"severity": severity, "station": station, "message": message}
        alerts.append(note)
        if station in stations:
            stations[station]["alerts"].append(note)
    air_sections, air = {}, None
    if air_model is not None and args.air_readings is not None:
        air_weather = pd.read_csv(args.weather) if args.weather else None  # all columns: wind, mixing height, humidity
        air = air_status(air_model, pd.read_csv(args.air_readings, low_memory=False), air_weather)
        current = {row["station_id"]: row for _, row in air["current"].iterrows()}
        for station in sorted(set(current) | set(air["flags"])):
            days = air["daily"][air["daily"]["station_id"] == station] if len(air["daily"]) else air["daily"]
            air_sections[station], raised = air_report(station, current.get(station), air["flags"].get(station, {}), days,
                                                       air_model["settings"])
            alerts += raised
    report = {
        "generated_at": now.isoformat(timespec="minutes"),
        "window_hours": args.window_hours,
        "models_trained_at": manifest["trained_at"],
        "forecast_ranges_recalibrated": forecast["recalibrated"],
        "weather_forecast_short": forecast["weather_short"],
        "alerts": sorted(alerts, key=lambda a: SEVERITY.index(a["severity"])),
        "stations": stations,
        **({"air": air_sections} if air_sections else {}),
    }

    out = args.out_dir / now.strftime("%Y-%m-%d_%H%M")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    text = render_text(report)
    (out / "report.txt").write_text(text)
    extra = air_cards(report.get("air", {}), air["hourly"]) if air is not None else ""
    (out / "dashboard.html").write_text(render_dashboard(report, hourly_fc, forecast["nightly"], settings, extra))
    if air is not None:
        air["hourly"].to_csv(out / "air_hourly_forecast.csv", index=False)
        air["daily"].to_csv(out / "air_daily_forecast.csv", index=False)
    recent.to_csv(out / "sensor_flags.csv", index=False)
    if len(hourly_fc):
        hourly_fc.to_csv(out / "hourly_forecast.csv", index=False)
        forecast["nightly"].to_csv(out / "nightly_forecast.csv", index=False)
    if len(bod):
        bod.to_csv(out / "bod_estimates.csv", index=False)
    if len(algae):
        algae.to_csv(out / "algae_estimates.csv", index=False)
    if len(coliform):
        coliform.to_csv(out / "coliform_nowcasts.csv", index=False)
    print(text)
    print(f"\nReport, dashboard and data files written to {out}")
    return out


if __name__ == "__main__":
    main()
