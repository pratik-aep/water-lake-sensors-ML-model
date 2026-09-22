import json

import pandas as pd
import pytest

from models.forecasting import train as forecast_train
from models.forecasting.config import DEFAULT_SETTINGS as FORECAST_SETTINGS
from models.pipeline import run, train_all
from models.pipeline.simulate import simulate
from models.wqi.generate_data import STATIONS


def test_lab_samples_meet_the_last_complete_sensor_hour_before_them():
    hourly = pd.DataFrame(
        {
            "station_id": "a",
            "timestamp": pd.date_range("2025-01-01 06:00", periods=6, freq="h"),
            "temperature": [20.0, 21, 22, 23, 24, 25],
            "ph": 7.5,
            "turbidity": 2.0,
            "dissolved_oxygen": 8.0,
        }
    )
    lab = pd.DataFrame(
        {
            "station_id": ["a", "a"],
            "timestamp": ["2025-01-01 09:20", "2025-01-01 20:00"],
            "bod": ["2.0", "<1"],
            "conductivity": [400, 410],
            "nitrate": [3.0, 3.1],
            "ph": [7.9, 7.8],
        }
    )
    rows = train_all.lab_with_sensors(lab, hourly)
    assert len(rows) == 1  # the 20:00 sample has no sensor hour within two hours
    assert rows.loc[0, "temperature"] == 22.0  # 08:00-09:00, the last hour complete before 09:20
    assert rows.loc[0, "lab_ph"] == 7.9 and rows.loc[0, "ph"] == 7.5


def test_train_everything_then_produce_a_daily_report(tmp_path, monkeypatch, capsys):
    stations = {k: STATIONS[k] for k in ("fateh_sagar", "goverdhan_sagar", "pichola")}
    readings, weather, lab, episodes = simulate(days=170, seed=5, stations=stations, chunk_days=60)
    readings.to_csv(tmp_path / "readings.csv", index=False)
    weather.to_csv(tmp_path / "weather.csv", index=False)
    lab.to_csv(tmp_path / "lab.csv", index=False)
    episodes.to_csv(tmp_path / "episodes.csv", index=False)
    small = {
        **FORECAST_SETTINGS,
        "train_horizons": [1, 3, 6, 12, 24, 36, 48],
        "gbm": {**FORECAST_SETTINGS["gbm"], "max_iter": 60},
    }
    monkeypatch.setattr(forecast_train, "DEFAULT_SETTINGS", small)

    models = tmp_path / "models"
    train_all.main(["--readings", str(tmp_path / "readings.csv"), "--weather", str(tmp_path / "weather.csv"),
                    "--lab", str(tmp_path / "lab.csv"), "--known-episodes", str(tmp_path / "episodes.csv"),
                    "--test-days", "20", "--no-challenger", "--out-dir", str(models)])
    manifest = json.loads((models / "manifest.json").read_text())
    assert set(manifest["models"]) == {"anomaly_detection", "forecasting", "bod_surrogate", "wqi", "algal_bloom", "coliform_nowcast"}
    assert all((models / path).exists() for path in manifest["models"].values())
    assert manifest["summary"]["bod_test_mae_mg_l"] is not None
    assert manifest["summary"]["chlorophyll_test_mae_ug_l"] is not None
    assert manifest["summary"]["coliform_cpcb_band_agreement"] is not None

    now = readings["timestamp"].max() - pd.Timedelta(days=8)
    recent = readings[(readings["timestamp"] > now - pd.Timedelta(days=16)) & (readings["timestamp"] <= now)]
    recent.to_csv(tmp_path / "recent.csv", index=False)
    run.main(["--readings", str(tmp_path / "recent.csv"), "--weather", str(tmp_path / "weather.csv"),
              "--models-dir", str(models), "--out-dir", str(tmp_path / "reports")])

    report_dir = next((tmp_path / "reports").iterdir())
    report = json.loads((report_dir / "report.json").read_text())
    assert set(report["stations"]) == set(stations)
    for section in report["stations"].values():
        assert section["now"]["wqi_class"] in {"Excellent", "Good", "Poor", "Very Poor", "Unsuitable"}
        assert section["now"]["cpcb_class"] in {"A", "B", "C", "D", "E", "below E", "unknown"}
        assert section["outlook_48h"]["lowest_oxygen"]["mg_l"] is not None
        assert len(section["nights"]) >= 5
        assert section["algae"]["trophic_class"] in {"oligotrophic", "mesotrophic", "eutrophic", "hypereutrophic"}
        assert section["coliform"]["cpcb_band"] in {"A (<=50)", "B (<=500)", "C (<=5000)", "above C"}
        # With a coliform nowcast the CPCB class no longer leaves coliform unassessed.
        assert "coliform" not in (section["now"]["cpcb_not_assessed"] or "")
    severities = [a["severity"] for a in report["alerts"]]
    assert severities == sorted(severities, key=["high", "medium", "low"].index)
    assert (report_dir / "report.txt").read_text().startswith("Lake water quality report")
    assert "=== pichola ===" in capsys.readouterr().out

    # The forecaster was trained with weather, so the daily run must not quietly run without it.
    with pytest.raises(SystemExit):
        run.main(["--readings", str(tmp_path / "recent.csv"), "--models-dir", str(models)])
