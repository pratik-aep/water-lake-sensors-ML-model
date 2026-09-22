import json

import joblib
import pandas as pd

from models.bod_surrogate import estimate, train
from models.bod_surrogate.simulate import simulate
from models.wqi.generate_data import STATIONS


def test_simulated_lab_results_look_like_a_weekly_programme():
    readings, _, lab = simulate(days=120, seed=4, stations={"pichola": STATIONS["pichola"]})
    assert {"load", "labile"}.isdisjoint(readings.columns)
    gaps = lab["timestamp"].diff().dt.days.dropna()
    assert gaps.median() == 7
    assert set(lab["timestamp"].dt.hour) == {10}


def test_train_and_estimate_command_lines_end_to_end(tmp_path, capsys):
    stations = {k: STATIONS[k] for k in ("fateh_sagar", "goverdhan_sagar", "pichola")}
    readings, weather, lab = simulate(days=220, seed=6, stations=stations)
    readings.to_csv(tmp_path / "readings.csv", index=False)
    weather.to_csv(tmp_path / "weather.csv", index=False)
    lab.to_csv(tmp_path / "lab.csv", index=False)

    train.main(["--data", str(tmp_path / "readings.csv"), "--lab", str(tmp_path / "lab.csv"),
                "--weather", str(tmp_path / "weather.csv"), "--out-dir", str(tmp_path)])
    metrics = json.loads((tmp_path / "bod_soft_sensor_metrics.json").read_text())
    assert metrics["cv_rmse_log"][metrics["model"]] < metrics["cv_rmse_log"]["mean_baseline"]
    assert metrics["test"]["soft_sensor"]["mae"] < metrics["test"]["station_median"]["mae"]
    bundle = joblib.load(tmp_path / "bod_soft_sensor.joblib")

    recent = readings[readings["timestamp"] > readings["timestamp"].max() - pd.Timedelta(days=10)]
    recent.to_csv(tmp_path / "recent.csv", index=False)
    args = ["--data", str(tmp_path / "recent.csv"), "--model", str(tmp_path / "bod_soft_sensor.joblib"),
            "--out", str(tmp_path / "estimates.csv")]
    if bundle["uses_weather"]:
        args += ["--weather", str(tmp_path / "weather.csv")]
    estimate.main(args)
    out = pd.read_csv(tmp_path / "estimates.csv")
    assert set(out["station_id"]) == set(stations)
    assert (out["bod_lo"] <= out["bod"]).all() and (out["bod"] <= out["bod_hi"]).all()
    assert out["cpcb_class"].isin(["A", "B", "C", "D", "E", "below E", "unknown"]).all()
    assert "CPCB class" in capsys.readouterr().out
