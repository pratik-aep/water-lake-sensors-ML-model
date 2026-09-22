"""Fetch Udaipur's hourly weather (recent past plus the week ahead) from Open-Meteo, in the models' CSV format."""

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
UDAIPUR = (24.58, 73.68)  # the four lakes lie within ~5 km of this point
TIMEZONE = "Asia/Kolkata"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
# Open-Meteo variable -> our column, and the factor that converts its unit to ours.
VARIABLES = {
    "temperature_2m": ("air_temperature", 1.0),
    "cloud_cover": ("cloud_cover", 0.01),  # % -> 0-1
    "precipitation": ("rain_mm", 1.0),
    "wind_speed_10m": ("wind_speed", 1.0),  # m/s, requested below
    "relative_humidity_2m": ("relative_humidity", 1.0),
    "boundary_layer_height": ("boundary_layer_height", 1.0),  # m; low at night, which traps air pollution
}


def _get(url: str, params: dict, timeout: float = 60) -> dict:
    with urllib.request.urlopen(f"{url}?{urllib.parse.urlencode(params)}", timeout=timeout) as response:
        return json.load(response)


def to_frame(payload: dict) -> pd.DataFrame:
    hourly = payload["hourly"]
    frame = pd.DataFrame({"timestamp": pd.to_datetime(hourly["time"])})
    for name, (column, factor) in VARIABLES.items():
        if name in hourly:
            frame[column] = pd.to_numeric(pd.Series(hourly[name]), errors="coerce") * factor
    return frame


def fetch(start=None, end=None, past_days: int = 14, forecast_days: int = 8, location=UDAIPUR) -> pd.DataFrame:
    """Hourly weather in local time: the archive for [start, end] if given, else recent days plus the forecast."""
    params = {
        "latitude": location[0],
        "longitude": location[1],
        "hourly": ",".join(VARIABLES),
        "timezone": TIMEZONE,
        "wind_speed_unit": "ms",
    }
    if start is not None:
        payload = _get(ARCHIVE_URL, {**params, "start_date": str(pd.Timestamp(start).date()), "end_date": str(pd.Timestamp(end).date())})
    else:
        payload = _get(FORECAST_URL, {**params, "past_days": past_days, "forecast_days": forecast_days})
    return to_frame(payload)


def merge(existing: pd.DataFrame | None, fresh: pd.DataFrame) -> pd.DataFrame:
    """Add fresh hours to a weather file; where both have an hour, the fresh value (the newer forecast) wins."""
    if existing is None or existing.empty:
        return fresh.sort_values("timestamp").reset_index(drop=True)
    existing = existing.assign(timestamp=pd.to_datetime(existing["timestamp"]))
    combined = pd.concat([existing, fresh], ignore_index=True)
    return combined.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "data" / "weather.csv", help="weather CSV to create or update")
    parser.add_argument("--start", help="fetch recorded history from this date (YYYY-MM-DD) instead of the forecast")
    parser.add_argument("--end", help="end date for --start (default: yesterday)")
    parser.add_argument("--past-days", type=int, default=14)
    args = parser.parse_args(argv)

    if args.start:
        end = args.end or (pd.Timestamp.now().normalize() - pd.Timedelta(days=1)).date()
        fresh = fetch(args.start, end)
    else:
        fresh = fetch(past_days=args.past_days)
    existing = pd.read_csv(args.out) if args.out.exists() else None
    weather = merge(existing, fresh)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    weather.to_csv(args.out, index=False)
    print(f"Fetched {len(fresh)} hours ({fresh['timestamp'].min()} to {fresh['timestamp'].max()}); "
          f"{args.out} now holds {len(weather)} hours")


if __name__ == "__main__":
    main()
