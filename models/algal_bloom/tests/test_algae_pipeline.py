import json

import numpy as np
import pandas as pd

from models.algal_bloom import estimate, train
from models.algal_bloom.simulate import simulate
from models.wqi.generate_data import STATIONS


def test_on_days_with_similar_cloud_more_chlorophyll_means_a_bigger_oxygen_swing():
    readings, weather, lab = simulate(days=200, seed=3, stations={"goverdhan_sagar": STATIONS["goverdhan_sagar"]})
    assert "chlorophyll_a" in lab.columns and "algae" not in readings.columns
    oxygen = readings.set_index("timestamp")["dissolved_oxygen"]
    swing = oxygen.groupby(oxygen.index.normalize()).agg(lambda x: x.max() - x.min())
    daytime = weather.set_index("timestamp")["cloud_cover"].between_time("06:00", "17:59")
    cloud = daytime.groupby(daytime.index.normalize()).mean()
    days = pd.to_datetime(lab["timestamp"]).dt.normalize()
    d = pd.DataFrame(
        {
            "chl": np.log(pd.to_numeric(lab["chlorophyll_a"], errors="coerce").to_numpy()),
            "swing": np.log(swing.reindex(days).to_numpy()),
            "cloud": cloud.reindex(days).to_numpy(),
        }
    ).dropna()
    # Cloud cuts photosynthesis for every alga alike, so take its effect out before comparing.
    without_cloud = d["swing"] - np.polyval(np.polyfit(d["cloud"], d["swing"], 1), d["cloud"])
    assert d["chl"].corr(without_cloud) > 0.5


def test_train_and_estimate_command_lines_end_to_end(tmp_path, capsys):
    stations = {k: STATIONS[k] for k in ("fateh_sagar", "goverdhan_sagar", "pichola")}
    readings, weather, lab = simulate(days=220, seed=6, stations=stations)
    readings.to_csv(tmp_path / "readings.csv", index=False)
    weather.to_csv(tmp_path / "weather.csv", index=False)
    lab.to_csv(tmp_path / "lab.csv", index=False)

    train.main(["--data", str(tmp_path / "readings.csv"), "--lab", str(tmp_path / "lab.csv"),
                "--weather", str(tmp_path / "weather.csv"), "--out-dir", str(tmp_path)])
    metrics = json.loads((tmp_path / "chlorophyll_soft_sensor_metrics.json").read_text())
    assert metrics["test"]["soft_sensor"]["mae"] < metrics["test"]["station_median"]["mae"]
    assert 0 <= metrics["test"]["trophic_class_agreement_with_lab"] <= 1

    recent = readings[readings["timestamp"] > readings["timestamp"].max() - pd.Timedelta(days=10)]
    recent.to_csv(tmp_path / "recent.csv", index=False)
    estimate.main(["--data", str(tmp_path / "recent.csv"), "--weather", str(tmp_path / "weather.csv"),
                   "--model", str(tmp_path / "chlorophyll_soft_sensor.joblib"), "--out", str(tmp_path / "chl.csv")])
    out = pd.read_csv(tmp_path / "chl.csv")
    assert set(out["station_id"]) == set(stations)
    assert (out["chlorophyll_a_lo"] <= out["chlorophyll_a"]).all() and (out["chlorophyll_a"] <= out["chlorophyll_a_hi"]).all()
    odds = out[["p_oligotrophic", "p_mesotrophic", "p_eutrophic", "p_hypereutrophic"]].sum(axis=1)
    assert ((odds - 1).abs() < 1e-9).all()
    assert "WHO" in capsys.readouterr().out
