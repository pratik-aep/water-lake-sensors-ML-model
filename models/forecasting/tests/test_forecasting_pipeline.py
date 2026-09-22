import joblib
import numpy as np
import pandas as pd
import pytest

from models.anomaly_detection.data import prepare
from models.forecasting import forecast, train
from models.forecasting.config import DEFAULT_SETTINGS
from models.forecasting.data import to_hourly
from models.forecasting.evaluate import hourly_report
from models.forecasting.forecaster import HourlyForecaster, NightlyForecaster, make_timelines, station_codes
from models.forecasting.simulate import simulate
from models.wqi.generate_data import STATIONS

SMALL = {
    **DEFAULT_SETTINGS,
    "train_horizons": [1, 3, 6, 12, 24, 36, 48],
    "gbm": {**DEFAULT_SETTINGS["gbm"], "max_iter": 60},
    "challenger": {**DEFAULT_SETTINGS["challenger"], "lookback_hours": 48, "hidden": 8, "epochs": 1},
}
TWO = {k: STATIONS[k] for k in ("fateh_sagar", "goverdhan_sagar")}


@pytest.fixture(scope="module")
def world():
    raw, weather = simulate(days=110, seed=3, stations=TWO)
    readings, _ = prepare(raw)
    hourly = to_hourly(readings, SMALL["min_valid_per_hour"])
    end = hourly["timestamp"].max() + pd.Timedelta(hours=1)
    cal_start, test_start = end - pd.Timedelta(days=30), end - pd.Timedelta(days=15)
    timelines = make_timelines(hourly, weather, SMALL)
    codes = station_codes(timelines)
    gbm = HourlyForecaster(SMALL, "gradient_boosting").fit(timelines, codes, cal_start).calibrate(timelines, cal_start, test_start)
    return raw, weather, timelines, codes, gbm, cal_start, test_start, end


def test_hourly_forecasts_are_ordered_calibrated_and_beat_no_change(world):
    _, _, timelines, _, gbm, _, test_start, end = world
    fc = gbm.predict(timelines, gbm.origins(timelines, test_start, end), simulated_forecast=True)
    assert (fc["lo"] <= fc["mid"]).all() and (fc["mid"] <= fc["hi"]).all()
    assert set(fc["horizon"]) == set(range(1, 49))
    oxygen = hourly_report(fc, SMALL["horizon_buckets"])["dissolved_oxygen"]
    assert oxygen["1-3h"]["skill"] > 0.2
    assert 0.6 <= oxygen["4-6h"]["coverage_80"] <= 0.97


def test_nightly_forecast_uses_the_hourly_path_for_the_nights_it_covers(world):
    _, _, timelines, codes, gbm, cal_start, test_start, end = world
    nightly = NightlyForecaster(SMALL).fit(timelines, codes, cal_start)
    issue = nightly.issue_dates(timelines, test_start, end - pd.Timedelta(days=8))
    alone = nightly.predict(timelines, issue)
    joined = nightly.predict(timelines, issue, hourly=gbm)
    assert joined["p_below_alert"].between(0, 1).all()
    near = joined["days_ahead"] <= 2
    assert not np.allclose(joined.loc[near, "mid"], alone.loc[near, "mid"])
    assert np.allclose(joined.loc[~near, "mid"], alone.loc[~near, "mid"])


def test_cnn_lstm_and_ensemble_produce_sorted_quantile_paths(world):
    _, _, timelines, codes, gbm, cal_start, test_start, end = world
    cnn = HourlyForecaster(SMALL, "cnn_lstm").fit(timelines, codes, cal_start, validation=(cal_start, test_start))
    origins = {st: np.array([tl.n - 60]) for st, tl in timelines.items()}
    info, deltas = cnn.model_.predict_deltas(timelines, codes, origins)
    assert len(info) == 2 * 48
    assert all((np.diff(d, axis=1) >= 0).all() for d in deltas.values())
    ensemble = HourlyForecaster.combine([gbm, cnn])
    _, mixed = ensemble.model_.predict_deltas(timelines, codes, origins)
    _, from_gbm = gbm.model_.predict_deltas(timelines, codes, origins)
    assert np.allclose(mixed["ph"], (from_gbm["ph"] + deltas["ph"]) / 2)


def test_train_and_forecast_command_lines_end_to_end(world, tmp_path, monkeypatch, capsys):
    raw, weather, *_ = world
    raw.to_csv(tmp_path / "readings.csv", index=False)
    weather.to_csv(tmp_path / "weather.csv", index=False)
    monkeypatch.setattr(train, "DEFAULT_SETTINGS", SMALL)
    train.main(["--data", str(tmp_path / "readings.csv"), "--weather", str(tmp_path / "weather.csv"),
                "--test-days", "15", "--calibration-days", "15", "--no-challenger", "--out-dir", str(tmp_path)])
    bundle = joblib.load(tmp_path / "forecaster.joblib")
    assert bundle["hourly_model"] == "gradient_boosting" and bundle["uses_weather"]

    recent = raw[raw["timestamp"] > raw["timestamp"].max() - pd.Timedelta(days=12)]
    recent[recent["timestamp"] <= recent["timestamp"].max() - pd.Timedelta(days=3)].to_csv(tmp_path / "recent.csv", index=False)
    forecast.main(["--data", str(tmp_path / "recent.csv"), "--weather", str(tmp_path / "weather.csv"),
                   "--model", str(tmp_path / "forecaster.joblib"), "--out-dir", str(tmp_path / "out")])
    hourly_fc = pd.read_csv(tmp_path / "out" / "hourly_forecast.csv")
    nightly_fc = pd.read_csv(tmp_path / "out" / "nightly_forecast.csv")
    assert len(hourly_fc) == 2 * 48 * 4
    assert nightly_fc["p_below_alert"].between(0, 1).all()
    assert "Pre-dawn oxygen minimum" in capsys.readouterr().out

    with pytest.raises(SystemExit):
        forecast.main(["--data", str(tmp_path / "recent.csv"), "--model", str(tmp_path / "forecaster.joblib")])
