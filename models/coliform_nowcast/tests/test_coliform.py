import json

import numpy as np
import pandas as pd
import pytest

from models.bod_surrogate.cpcb import assess
from models.coliform_nowcast import estimate, train
from models.coliform_nowcast.bands import band_of_count, band_of_nowcast
from models.coliform_nowcast.config import DEFAULT_SETTINGS
from models.coliform_nowcast.features import sample_features
from models.coliform_nowcast.simulate import simulate
from models.wqi.generate_data import STATIONS


def test_bands_follow_the_cpcb_coliform_limits():
    assert band_of_count([10, 50, 51, 500, 4999, 6000]).tolist() == [
        "A (<=50)", "A (<=50)", "B (<=500)", "B (<=500)", "C (<=5000)", "above C"
    ]
    nowcast = band_of_nowcast({50.0: [0.2, 0.9, 0.9, 0.9], 500.0: [0.1, 0.3, 0.8, 0.9], 5000.0: [0.0, 0.1, 0.2, 0.7]})
    assert nowcast.tolist() == ["A (<=50)", "B (<=500)", "C (<=5000)", "above C"]


def test_coliform_limits_join_bod_in_the_cpcb_class():
    certain = {2.0: [1.0, 1.0], 3.0: [1.0, 1.0]}
    without = assess([7.0, 7.0], [7.5, 7.5], certain)
    with_coliform = assess([7.0, 7.0], [7.5, 7.5], certain, {50.0: [0.9, 0.1], 500.0: [1.0, 0.2], 5000.0: [1.0, 0.9]})
    assert without["cpcb_class"].tolist() == ["A", "A"]
    assert with_coliform["cpcb_class"].tolist() == ["A", "C"]
    assert with_coliform.loc[0, "cpcb_not_assessed"] == "" and without.loc[0, "cpcb_not_assessed"] == "total coliforms"


def test_rain_timing_features_look_back_from_the_sample():
    hours = pd.date_range("2025-07-01", periods=24 * 10, freq="h")
    hourly = pd.DataFrame(
        {"station_id": "a", "timestamp": hours, "temperature": 25.0, "ph": 7.8, "turbidity": 5.0, "dissolved_oxygen": 7.0}
    )
    rain = np.zeros(len(hours))
    rain[hours.get_loc(pd.Timestamp("2025-07-08 03:00"))] = 12.0
    rain[hours.get_loc(pd.Timestamp("2025-07-09 12:00"))] = 30.0  # after the sample: must not count
    weather = pd.DataFrame({"timestamp": hours, "air_temperature": 28.0, "cloud_cover": 0.5, "rain_mm": rain})
    samples = pd.DataFrame({"station_id": ["a"], "timestamp": [pd.Timestamp("2025-07-08 10:30")]})
    X, kept = sample_features(samples, hourly, weather, DEFAULT_SETTINGS)
    assert kept.all()
    assert X.loc[0, "rain_24h"] == 12.0 and X.loc[0, "rain_7d"] == 12.0
    assert X.loc[0, "hours_since_rain"] == pytest.approx(7.0)


def test_train_and_nowcast_command_lines_end_to_end(tmp_path, capsys):
    stations = {k: STATIONS[k] for k in ("fateh_sagar", "goverdhan_sagar", "pichola")}
    readings, weather, lab = simulate(days=220, seed=6, stations=stations)
    readings.to_csv(tmp_path / "readings.csv", index=False)
    weather.to_csv(tmp_path / "weather.csv", index=False)
    lab.to_csv(tmp_path / "lab.csv", index=False)

    train.main(["--data", str(tmp_path / "readings.csv"), "--lab", str(tmp_path / "lab.csv"),
                "--weather", str(tmp_path / "weather.csv"), "--out-dir", str(tmp_path)])
    metrics = json.loads((tmp_path / "coliform_nowcast_metrics.json").read_text())
    assert metrics["cv_rmse_log"][metrics["model"]] < metrics["cv_rmse_log"]["mean_baseline"]

    recent = readings[readings["timestamp"] > readings["timestamp"].max() - pd.Timedelta(days=10)]
    recent.to_csv(tmp_path / "recent.csv", index=False)
    estimate.main(["--data", str(tmp_path / "recent.csv"), "--weather", str(tmp_path / "weather.csv"),
                   "--model", str(tmp_path / "coliform_nowcast.joblib"), "--out", str(tmp_path / "coliform.csv")])
    out = pd.read_csv(tmp_path / "coliform.csv")
    assert set(out["station_id"]) == set(stations)
    assert (out["p_above_50"] >= out["p_above_500"]).all() and (out["p_above_500"] >= out["p_above_5000"]).all()
    assert "CPCB band" in capsys.readouterr().out
