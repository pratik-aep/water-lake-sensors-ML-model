import numpy as np
import pandas as pd
import pytest

from models.anomaly_detection.config import SENSORS
from models.forecasting.config import DEFAULT_SETTINGS, WEATHER
from models.forecasting.features import Timeline, forecast_error, hourly_rows, nightly_rows

S = DEFAULT_SETTINGS
WEATHER_FEATURES_AHEAD = {"cloud_before_target_12h", "rain_before_target_24h", "air_at_target", "air_mean_ahead"}


def timeline(days=12, seed=0, with_weather=True):
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2025-03-01", periods=days * 24, freq="h")
    diel = np.sin(2 * np.pi * (hours.hour - 9) / 24)
    hourly = pd.DataFrame(
        {
            "station_id": "a",
            "timestamp": hours,
            "temperature": 25 + diel + rng.normal(0, 0.1, len(hours)),
            "ph": 7.8 + 0.1 * diel,
            "turbidity": 3.0 + rng.random(len(hours)),
            "dissolved_oxygen": 6 + 1.5 * diel + rng.normal(0, 0.1, len(hours)),
        }
    )
    weather = None
    if with_weather:
        clock = pd.date_range(hours[0], periods=len(hours) + 24 * 8, freq="h")
        weather = pd.DataFrame({"timestamp": clock, "air_temperature": 28.0, "cloud_cover": 0.3, "rain_mm": 0.0})
    return hourly, Timeline("a", hourly, weather, future_hours=24 * 8)


def test_targets_are_changes_from_the_origin():
    hourly, tl = timeline()
    X, Y, info = hourly_rows(tl, np.array([100]), [1, 24], S, 0.0)
    do = hourly["dissolved_oxygen"].to_numpy()
    assert Y["dissolved_oxygen"].tolist() == pytest.approx([do[101] - do[100], do[124] - do[100]])
    assert list(info["target_pos"]) == [101, 124]


def test_features_never_see_readings_after_the_origin():
    hourly, tl = timeline()
    X, Y, _ = hourly_rows(tl, np.array([100]), [6, 30], S, 0.0)
    changed = hourly.copy()
    changed.loc[101:, SENSORS] += 5.0
    X2, Y2, _ = hourly_rows(Timeline("a", changed, None, 24 * 8), np.array([100]), [6, 30], S, 0.0)
    sensor_features = [c for c in X.columns if c not in WEATHER_FEATURES_AHEAD | {"rain_past_24h", "cloud_past_12h"}]
    pd.testing.assert_frame_equal(X[sensor_features], X2[sensor_features])
    assert not Y.equals(Y2)


def test_same_hour_yesterday_always_comes_from_the_past():
    hourly, tl = timeline()
    X, _, _ = hourly_rows(tl, np.array([100]), [5, 30], S, 0.0)
    do = hourly["dissolved_oxygen"].to_numpy()
    assert X["same_hour_last_day_dissolved_oxygen"].tolist() == pytest.approx([do[81] - do[100], do[82] - do[100]])


def test_nightly_target_is_the_change_in_predawn_minimum():
    hourly, tl = timeline()
    issue = pd.DatetimeIndex([pd.Timestamp("2025-03-08")])
    X, y, info = nightly_rows(tl, issue, S, 0.0)
    do = hourly.set_index("timestamp")["dissolved_oxygen"]

    def night(date):
        return do[(do.index.normalize() == date) & (do.index.hour < 9)].min()

    assert X["last_night_min"].iloc[0] == pytest.approx(night(pd.Timestamp("2025-03-08")))
    assert y.iloc[2] == pytest.approx(night(pd.Timestamp("2025-03-11")) - night(pd.Timestamp("2025-03-08")))
    assert X["now_dissolved_oxygen"].iloc[0] == pytest.approx(do[pd.Timestamp("2025-03-08 18:00")])
    assert list(info["days_ahead"]) == list(range(1, S["night_days"] + 1))


def test_forecast_weather_error_grows_with_lead_time():
    rng = np.random.default_rng(0)
    clear = np.full(20000, 0.5)
    assert np.std(forecast_error("cloud", clear, 48, rng)) > 2 * np.std(forecast_error("cloud", clear, 1, rng))
    rain = np.full(20000, 10.0)
    assert (forecast_error("rain", rain, 72, rng) == 0).mean() > (forecast_error("rain", rain, 1, rng) == 0).mean()
    assert set(WEATHER) == {"air_temperature", "cloud_cover", "rain_mm"}
