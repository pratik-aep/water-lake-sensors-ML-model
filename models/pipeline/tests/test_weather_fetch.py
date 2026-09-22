import pandas as pd
import pytest

from models.forecasting.data import load_weather
from models.pipeline import weather

PAYLOAD = {
    "hourly": {
        "time": ["2025-07-01T00:00", "2025-07-01T01:00", "2025-07-01T02:00"],
        "temperature_2m": [27.1, 26.8, 26.5],
        "cloud_cover": [100, 50, 0],
        "precipitation": [2.5, 0.0, None],
        "wind_speed_10m": [3.0, 2.0, 1.0],
    }
}


def test_open_meteo_hours_become_the_models_weather_columns(monkeypatch):
    calls = []
    monkeypatch.setattr(weather, "_get", lambda url, params: calls.append((url, params)) or PAYLOAD)
    frame = weather.fetch(past_days=3)
    assert calls[0][0] == weather.FORECAST_URL and calls[0][1]["timezone"] == "Asia/Kolkata"
    assert list(frame["cloud_cover"]) == [1.0, 0.5, 0.0]  # percent to a 0-1 fraction
    assert frame["rain_mm"].iloc[0] == 2.5 and pd.isna(frame["rain_mm"].iloc[2])
    weather.fetch("2025-07-01", "2025-07-02")
    assert calls[1][0] == weather.ARCHIVE_URL and calls[1][1]["start_date"] == "2025-07-01"


def test_newer_forecasts_replace_older_ones_and_the_file_loads(tmp_path, monkeypatch):
    old = weather.to_frame(PAYLOAD).assign(air_temperature=10.0)
    newer = weather.to_frame({"hourly": {**PAYLOAD["hourly"], "time": ["2025-07-01T02:00", "2025-07-01T03:00", "2025-07-01T04:00"]}})
    merged = weather.merge(old, newer)
    assert len(merged) == 5
    assert merged.set_index("timestamp").loc[pd.Timestamp("2025-07-01 02:00"), "air_temperature"] == pytest.approx(27.1)
    merged.to_csv(tmp_path / "weather.csv", index=False)
    loaded = load_weather(tmp_path / "weather.csv")
    assert {"air_temperature", "cloud_cover", "rain_mm"} <= set(loaded.columns)
