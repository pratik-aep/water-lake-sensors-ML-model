import json

import joblib
import pandas as pd

from models.air_quality import train
from models.air_quality.estimate import air_status
from models.air_quality.simulate import simulate


def test_train_then_estimate_the_air_quality_end_to_end(tmp_path):
    readings, weather = simulate(days=140, seed=3, start="2024-09-15")
    readings.to_csv(tmp_path / "air.csv", index=False)
    weather.to_csv(tmp_path / "weather.csv", index=False)
    train.main(["--data", str(tmp_path / "air.csv"), "--weather", str(tmp_path / "weather.csv"),
                "--test-days", "21", "--out-dir", str(tmp_path)])
    metrics = json.loads((tmp_path / "air_forecaster_metrics.json").read_text())
    tomorrow = metrics["daily_test"]["1"]
    # Tomorrow from the hourly path should beat assuming tomorrow repeats today.
    assert tomorrow["pm25"]["mae"] < tomorrow["pm25"]["mae_persistence"]
    assert 0.6 < metrics["hourly_test"]["pm25"]["1-6h"]["coverage_80"] < 0.95

    recent = readings[readings["timestamp"] > readings["timestamp"].max() - pd.Timedelta(days=10)]
    status = air_status(joblib.load(tmp_path / "air_forecaster.joblib"), recent, weather)
    assert set(status["current"]["station_id"]) == {"air_city_centre", "air_madri_industrial"}
    assert status["current"]["category"].isin(["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]).all()
    assert sorted(status["daily"]["days_ahead"].unique()) == [1, 2]
    assert status["hourly"]["horizon"].max() == 48
