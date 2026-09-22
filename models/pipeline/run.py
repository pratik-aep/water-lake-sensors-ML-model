"""Daily run: check every sensor, forecast, estimate BOD, algae and coliform, and classify each lake, in one report."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from ..algal_bloom.estimate import estimate_daily as estimate_algae
from ..anomaly_detection.config import INTERVAL, SENSORS
from ..anomaly_detection.data import load_readings
from ..anomaly_detection.evaluate import alarm_incidents
from ..bod_surrogate.cpcb import assess
from ..bod_surrogate.estimate import estimate_daily
from ..coliform_nowcast.estimate import estimate_daily as estimate_coliform
from ..forecasting.data import history_from_frame, load_weather, to_hourly
from ..forecasting.forecast import make_forecast, with_wqi_class
from ..wqi.predict import classify
from .report import SEVERITY, render_text, station_report

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
    parser.add_argument("--weather", type=Path, help="hourly weather including forecasts for the coming week")
    parser.add_argument("--models-dir", type=Path, default=HERE / "artifacts", help="output of train_all")
    parser.add_argument("--window-hours", type=int, default=24, help="how far back incidents are reported")
    parser.add_argument("--out-dir", type=Path, default=HERE / "reports")
    args = parser.parse_args(argv)

    manifest = json.loads((args.models_dir / "manifest.json").read_text())
    bundles = {name: joblib.load(args.models_dir / path) for name, path in manifest["models"].items()}
    forecaster, bod_model = bundles["forecasting"], bundles["bod_surrogate"]
    algae_model, coliform_model = bundles.get("algal_bloom"), bundles.get("coliform_nowcast")
    detector = bundles["anomaly_detection"]["detector"]
    needs_weather = detector.uses_weather or any(
        b["uses_weather"] for b in (forecaster, bod_model, algae_model, coliform_model) if b is not None
    )
    if needs_weather and args.weather is None:
        parser.error("these models use weather; pass --weather with forecasts for the coming week")
    weather = load_weather(args.weather) if needs_weather else None

    readings, _ = load_readings(args.readings)
    flagged = detector.detect(readings, weather if detector.uses_weather else None)
    now = flagged["timestamp"].max()
    recent = flagged[flagged["timestamp"] > now - pd.Timedelta(hours=args.window_hours)]
    merge_gap = pd.Timedelta(INTERVAL) * (detector.settings["merge_gap_points"] + 1)
    incidents = alarm_incidents(recent, merge_gap)

    cleaned, _ = history_from_frame(flagged)
    forecast = make_forecast(forecaster, cleaned, weather)
    hourly_fc = with_wqi_class(forecast["hourly"], bundles["wqi"])
    bod, _ = estimate_daily(bod_model, cleaned, weather)
    coliform = estimate_coliform(coliform_model, cleaned, weather)[0] if coliform_model is not None else pd.DataFrame()
    coliform_latest = {st: g.iloc[-1] for st, g in coliform.groupby("station_id")} if len(coliform) else {}
    if len(bod) and len(coliform):
        bod = cpcb_with_coliform(bod, coliform)
    bod_latest = {st: g.iloc[-1] for st, g in bod.groupby("station_id")} if len(bod) else {}
    algae = estimate_algae(algae_model, cleaned, weather)[0] if algae_model is not None else pd.DataFrame()
    algae_latest = {st: g.iloc[-1] for st, g in algae.groupby("station_id")} if len(algae) else {}
    wqi_now = latest_wqi(bundles["wqi"], cleaned)

    settings = {
        "alert_probability": forecaster["settings"]["alert_probability"],
        "do_alert_mg_l": forecaster["settings"]["do_alert_mg_l"],
        "stale_hours": STALE_HOURS,
    }
    stations, alerts = {}, []
    for station in sorted(flagged["station_id"].unique()):
        stations[station], raised = station_report(
            station, recent, incidents, hourly_fc, forecast["nightly"], bod_latest.get(station), wqi_now.get(station),
            settings, algae_latest.get(station), coliform_latest.get(station),
        )
        alerts += raised
    report = {
        "generated_at": now.isoformat(timespec="minutes"),
        "window_hours": args.window_hours,
        "models_trained_at": manifest["trained_at"],
        "forecast_ranges_recalibrated": forecast["recalibrated"],
        "weather_forecast_short": forecast["weather_short"],
        "alerts": sorted(alerts, key=lambda a: SEVERITY.index(a["severity"])),
        "stations": stations,
    }

    out = args.out_dir / now.strftime("%Y-%m-%d_%H%M")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    text = render_text(report)
    (out / "report.txt").write_text(text)
    recent.to_csv(out / "sensor_flags.csv", index=False)
    hourly_fc.to_csv(out / "hourly_forecast.csv", index=False)
    forecast["nightly"].to_csv(out / "nightly_forecast.csv", index=False)
    bod.to_csv(out / "bod_estimates.csv", index=False)
    if len(algae):
        algae.to_csv(out / "algae_estimates.csv", index=False)
    if len(coliform):
        coliform.to_csv(out / "coliform_nowcasts.csv", index=False)
    print(text)
    print(f"\nReport and data files written to {out}")


if __name__ == "__main__":
    main()
