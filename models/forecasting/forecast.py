"""Forecast every sensor for the next 48 hours and each lake's pre-dawn oxygen minimum for the coming nights."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..anomaly_detection.config import SENSORS
from .data import load_history, load_weather, to_hourly
from .forecaster import make_timelines

HERE = Path(__file__).parent


# A forecast starts from the latest hour with every sensor reading; one older than this would be stale.
MAX_ORIGIN_AGE_HOURS = 24


def latest_complete_hour(tl) -> int | None:
    """Position of the latest hour with every sensor reading, or None if there is none recent enough."""
    complete = np.flatnonzero(~np.isnan(tl.x).any(axis=1))
    if not len(complete) or tl.n - 1 - complete[-1] > MAX_ORIGIN_AGE_HOURS:
        return None
    return int(complete[-1])


def with_wqi_class(hourly_fc: pd.DataFrame, wqi_bundle: dict) -> pd.DataFrame:
    """Expected WQI class for each forecast hour, from the median forecasts, via the WQI sensor model."""
    from ..wqi.predict import classify

    wide = hourly_fc.pivot_table(index=["station_id", "target_time"], columns="sensor", values="mid").reset_index()
    wide["wqi_class"] = classify(wqi_bundle, wide)["predicted_class"].to_numpy()
    return hourly_fc.merge(wide[["station_id", "target_time", "wqi_class"]], on=["station_id", "target_time"])


def make_forecast(bundle: dict, readings: pd.DataFrame, weather: pd.DataFrame | None) -> dict:
    """Hourly forecasts from each station's latest complete hour, and nightly oxygen minima for the coming nights."""
    settings = bundle["settings"]
    timelines = make_timelines(
        to_hourly(readings, settings["min_valid_per_hour"]), weather if bundle["uses_weather"] else None, settings
    )
    hourly_model, nightly_model = bundle["hourly"], bundle["nightly"]

    # With a month of history the model has never seen, refresh the ranges so they follow the current season.
    data_end = max(tl.clock[tl.n - 1] for tl in timelines.values()) + pd.Timedelta(hours=1)
    recent = data_end - pd.Timedelta(days=settings["recalibrate_days"])
    recalibrated = recent > pd.Timestamp(bundle["date_range"][1])
    if recalibrated:
        hourly_model.calibrate(timelines, recent, data_end)
        nightly_model.calibrate(timelines, recent, data_end, hourly_model)

    # A lake whose sensors have not all reported recently (a dead probe, a long gap) gets no forecast today;
    # the other lakes still do.
    last = {station: pos for station, tl in timelines.items() if (pos := latest_complete_hour(tl)) is not None}
    skipped = sorted(set(timelines) - set(last))
    if not last:
        return {"hourly": pd.DataFrame(), "nightly": pd.DataFrame(), "issued_at": {}, "recalibrated": recalibrated,
                "weather_short": False, "skipped": skipped}
    timelines = {station: timelines[station] for station in last}
    issued_at = {station: tl.clock[last[station]] for station, tl in timelines.items()}
    hourly_fc = hourly_model.predict(timelines, {st: np.array([pos]) for st, pos in last.items()})

    issue = {}
    for station, now in issued_at.items():
        evening = now.normalize() + pd.Timedelta(hours=settings["night_issue_hour"])
        issue[station] = pd.DatetimeIndex([(evening if evening <= now else evening - pd.Timedelta(days=1)).normalize()])
    nightly_fc = nightly_model.predict(timelines, issue, hourly=hourly_model)
    night_end = pd.Timedelta(hours=settings["night_hours"][1])
    nightly_fc = nightly_fc[nightly_fc["night_of"] + night_end > nightly_fc["station_id"].map(issued_at)]

    weather_short = False
    if weather is not None and bundle["uses_weather"]:
        # Forecast weather is used up to the last forecast hour and the day before the last forecast night.
        ahead = max(hourly_fc["target_time"].max(), nightly_fc["night_of"].max())
        known = weather.set_index("timestamp").reindex(pd.date_range(hourly_fc["origin"].min(), ahead, freq="h"))
        weather_short = bool(known.isna().any(axis=1).mean() > 0.1)
    return {
        "hourly": hourly_fc.drop(columns=["actual", "origin_pos", "target_pos"]),
        "nightly": nightly_fc.drop(columns=["actual"]),
        "issued_at": issued_at,
        "recalibrated": recalibrated,
        "weather_short": weather_short,
        "skipped": skipped,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="recent 10-minute readings (at least 8 days)")
    parser.add_argument("--weather", type=Path, help="hourly weather, including forecast rows for the coming week")
    parser.add_argument("--model", type=Path, default=HERE / "artifacts" / "forecaster.joblib")
    parser.add_argument("--wqi-model", type=Path, help="also give the expected WQI class for each forecast hour")
    parser.add_argument("--out-dir", type=Path, help="write hourly_forecast.csv and nightly_forecast.csv here")
    args = parser.parse_args(argv)

    bundle = joblib.load(args.model)
    settings = bundle["settings"]
    if bundle["uses_weather"] and args.weather is None:
        parser.error("this model was trained with weather; pass --weather with forecasts for the coming days")
    readings, _ = load_history(args.data)
    result = make_forecast(bundle, readings, load_weather(args.weather) if bundle["uses_weather"] else None)
    hourly_fc, nightly_fc = result["hourly"], result["nightly"]
    if result["recalibrated"]:
        print(f"Ranges recalibrated on the last {settings['recalibrate_days']} days of verified forecasts")
    if result["weather_short"]:
        print("Warning: weather forecasts don't cover the coming week; later forecasts will be less reliable")
    if args.wqi_model:
        hourly_fc = with_wqi_class(hourly_fc, joblib.load(args.wqi_model))

    alert = settings["do_alert_mg_l"]
    for station, issued in result["issued_at"].items():
        fc = hourly_fc[hourly_fc["station_id"] == station]
        do = fc[fc["sensor"] == "dissolved_oxygen"]
        low = do.loc[do["mid"].idxmin()]
        print(f"\n=== {station} (forecast from {issued}) ===")
        print(f"Next 48 h: lowest oxygen {low['mid']:.2f} mg/L at {low['target_time']} "
              f"(80% range {low['lo']:.2f}-{low['hi']:.2f})")
        for sensor in SENSORS:
            s = fc[fc["sensor"] == sensor]
            at24 = s[s["horizon"] == 24].iloc[0]
            print(f"  {sensor:<17} in 24 h: {at24['mid']:.2f} ({at24['lo']:.2f}-{at24['hi']:.2f})")
        if "wqi_class" in fc:
            classes = fc.drop_duplicates("target_time")["wqi_class"].value_counts()
            print(f"  expected WQI class over the next 48 h: {classes.to_dict()}")
        nights = nightly_fc[nightly_fc["station_id"] == station]
        print(f"Pre-dawn oxygen minimum (alert below {alert} mg/L):")
        for _, n in nights.iterrows():
            flag = "  ALERT" if n["p_below_alert"] >= settings["alert_probability"] else ""
            print(f"  {n['night_of'].date()}  {n['mid']:.2f} mg/L ({n['lo']:.2f}-{n['hi']:.2f})  "
                  f"P(<{alert:g}) = {n['p_below_alert']:.0%}{flag}")

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        hourly_fc.to_csv(args.out_dir / "hourly_forecast.csv", index=False)
        nightly_fc.to_csv(args.out_dir / "nightly_forecast.csv", index=False)
        print(f"\nWrote hourly_forecast.csv and nightly_forecast.csv to {args.out_dir}")


if __name__ == "__main__":
    main()
