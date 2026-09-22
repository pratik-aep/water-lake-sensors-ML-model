"""Each air station now: the National AQI with its main pollutant and sensor checks, and particulate forecasts."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .naqi import advice, averaged, naqi
from .qc import POLLUTANTS, check, to_hourly
from .train import timelines_for

HERE = Path(__file__).parent


def air_status(bundle: dict, readings: pd.DataFrame, weather: pd.DataFrame | None) -> dict:
    """{"current": latest index per station, "flags": sensor checks over the last day, "hourly": 48 h forecasts,
    "daily": tomorrow and the day after}. Needs at least eight days of readings (the forecasts use a week's history)."""
    s = bundle["settings"]
    if weather is not None:
        weather = weather.assign(timestamp=pd.to_datetime(weather["timestamp"]))
    hourly = check(to_hourly(readings, s), weather, s)
    index = naqi(averaged(hourly))
    current = index.dropna(subset=["aqi"]).groupby("station_id").tail(1).reset_index(drop=True)
    current["advice"] = current["category"].map(advice)

    last_day = hourly[hourly["timestamp"] > hourly["timestamp"].max() - pd.Timedelta(hours=24)]
    flags = {
        st: {p: sorted(set(g[f"{p}_flag"]) - {""}) for p in POLLUTANTS if f"{p}_flag" in g and (g[f"{p}_flag"] != "").any()}
        for st, g in last_day.groupby("station_id")
    }

    timelines = timelines_for(hourly, weather if bundle["uses_weather"] else None, s)
    hourly_model, daily_model = bundle["hourly"], bundle["daily"]
    origins, issue = {}, {}
    for st, tl in timelines.items():
        complete = np.flatnonzero(~np.isnan(tl.y).any(axis=1))
        if len(complete) and complete[-1] >= 24 * 7:
            origins[st] = complete[-1:]
            now = tl.clock[complete[-1]]
            evening = now.normalize() + pd.Timedelta(hours=18)
            issue[st] = pd.DatetimeIndex([(evening if evening <= now else evening - pd.Timedelta(days=1)).normalize()])
    hourly_fc = hourly_model.predict(timelines, origins) if origins else pd.DataFrame()
    daily_fc = daily_model.predict(timelines, issue, hourly=hourly_model) if issue else pd.DataFrame()
    if len(daily_fc):
        latest = hourly.groupby("station_id")["timestamp"].max().dt.normalize()
        daily_fc = daily_fc[daily_fc["date"] > daily_fc["station_id"].map(latest)]
    return {"current": current, "flags": flags, "hourly": hourly_fc, "daily": daily_fc}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="recent air readings (eight days or more)")
    parser.add_argument("--weather", type=Path, help="hourly weather including the coming days")
    parser.add_argument("--model", type=Path, default=HERE / "artifacts" / "air_forecaster.joblib")
    args = parser.parse_args(argv)
    bundle = joblib.load(args.model)
    if bundle["uses_weather"] and args.weather is None:
        parser.error("this model was trained with weather; pass --weather")
    status = air_status(bundle, pd.read_csv(args.data, low_memory=False), pd.read_csv(args.weather) if args.weather else None)
    for _, row in status["current"].iterrows():
        print(f"\n=== {row['station_id']} ({row['timestamp']}) ===")
        print(f"AQI {row['aqi']:.0f} {row['category']} (main pollutant {row['prominent_pollutant']}): {row['advice']}")
        if status["flags"].get(row["station_id"]):
            print(f"Sensor checks, last 24 h: {status['flags'][row['station_id']]}")
        for _, d in status["daily"][status["daily"]["station_id"] == row["station_id"]].iterrows():
            print(f"  {d['date'].date()}: PM2.5 {d['pm25']:.0f} ug/m3 ({d['pm25_lo']:.0f}-{d['pm25_hi']:.0f}), "
                  f"particulate index {d['pm_index']:.0f} {d['pm_category']}, chance Poor or worse {d['p_poor_or_worse']:.0%}")


if __name__ == "__main__":
    main()
