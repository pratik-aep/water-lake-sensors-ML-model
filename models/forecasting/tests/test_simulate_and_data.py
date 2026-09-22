import numpy as np
import pandas as pd
import pytest

from models.forecasting.data import load_history, load_weather, to_hourly
from models.forecasting.simulate import simulate
from models.wqi.generate_data import STATIONS


@pytest.fixture(scope="module")
def sim():
    return simulate(days=200, seed=5, stations={"goverdhan_sagar": STATIONS["goverdhan_sagar"]})


def test_weather_is_physically_plausible(sim):
    _, weather = sim
    assert weather["cloud_cover"].between(0, 1).all()
    assert (weather["rain_mm"] >= 0).all()
    assert weather["rain_mm"].gt(0).any()


def test_oxygen_rises_by_day_and_falls_overnight(sim):
    readings, _ = sim
    by_day = readings.set_index("timestamp")["dissolved_oxygen"]
    afternoon = by_day[(by_day.index.hour >= 13) & (by_day.index.hour < 17)].groupby(lambda t: t.date()).mean()
    predawn = by_day[by_day.index.hour < 9].groupby(lambda t: t.date()).min()
    assert (afternoon - predawn).median() > 1.0


def test_cloudy_days_lead_to_lower_oxygen_the_next_night(sim):
    readings, weather = sim
    do = readings.set_index("timestamp")["dissolved_oxygen"]
    night_min = do[do.index.hour < 9].groupby(do.index[do.index.hour < 9].normalize()).min()
    daytime = weather.set_index("timestamp")["cloud_cover"].between_time("06:00", "17:59")
    cloud = daytime.groupby(daytime.index.normalize()).mean()
    change_next_night = night_min.shift(-1) - night_min
    assert pd.concat([cloud, change_next_night], axis=1).dropna().corr().iloc[0, 1] < -0.1


def test_hourly_mean_needs_enough_readings():
    ts = pd.date_range("2025-01-01", periods=12, freq="10min")
    readings = pd.DataFrame(
        {"station_id": "a", "timestamp": ts, "temperature": 20.0, "ph": 7.5, "turbidity": 2.0, "dissolved_oxygen": np.arange(12.0)}
    )
    readings.loc[7:11, "dissolved_oxygen"] = np.nan
    hourly = to_hourly(readings, min_valid=3)
    assert hourly["dissolved_oxygen"].iloc[0] == pytest.approx(2.5)
    assert np.isnan(hourly["dissolved_oxygen"].iloc[1])


def test_history_drops_values_the_anomaly_detector_blamed_on_a_sensor(tmp_path):
    ts = pd.date_range("2025-01-01", periods=3, freq="10min")
    flagged = pd.DataFrame(
        {
            "station_id": "a",
            "timestamp": ts,
            "temperature": 20.0,
            "ph": 7.5,
            "turbidity": [2.0, 50.0, 2.0],
            "dissolved_oxygen": 8.0,
            "turbidity_fault": ["", "spike", ""],
        }
    )
    flagged.to_csv(tmp_path / "flagged.csv", index=False)
    readings, report = load_history(tmp_path / "flagged.csv")
    assert np.isnan(readings.loc[1, "turbidity"])
    assert report["faulty_readings_masked"] == 1


def test_weather_short_gaps_are_filled_long_gaps_are_not(tmp_path):
    hours = pd.date_range("2025-01-01", periods=12, freq="h")
    weather = pd.DataFrame({"timestamp": hours, "air_temperature": np.arange(12.0), "cloud_cover": 0.5, "rain_mm": 0.0})
    weather = weather.drop(index=[2, 3, 6, 7, 8, 9])
    weather.to_csv(tmp_path / "weather.csv", index=False)
    loaded = load_weather(tmp_path / "weather.csv").set_index("timestamp")["air_temperature"]
    assert loaded.iloc[2] == pytest.approx(2.0)
    assert loaded.iloc[6:10].isna().any()

    weather.drop(columns="rain_mm").to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="rain_mm"):
        load_weather(tmp_path / "bad.csv")
