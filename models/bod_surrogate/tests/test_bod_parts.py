import numpy as np
import pandas as pd
import pytest

from models.anomaly_detection.config import SENSORS
from models.bod_surrogate.config import DEFAULT_SETTINGS
from models.bod_surrogate.cpcb import assess
from models.bod_surrogate.data import load_lab
from models.bod_surrogate.features import StationHistory, features_at
from models.bod_surrogate.models import SoftSensor, candidates

S = DEFAULT_SETTINGS


def test_lab_results_are_parsed_and_cleaned(tmp_path):
    pd.DataFrame(
        {
            "station_id": ["a", "a", "a", "a", "b"],
            "timestamp": ["2025-01-01 10:00", "2025-01-08 10:00", "2025-01-08 10:00", "bad", "2025-01-01 10:00"],
            "bod": ["<1", "2.5", "2.6", "3.0", "0"],
        }
    ).to_csv(tmp_path / "lab.csv", index=False)
    lab, report = load_lab(tmp_path / "lab.csv")
    assert lab["bod"].tolist() == [0.5, 2.5]
    assert report["duplicates_dropped"] == 1 and report["unusable_dropped"] == 2

    pd.DataFrame({"station_id": ["a"], "timestamp": ["2025-01-01"]}).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="bod"):
        load_lab(tmp_path / "bad.csv")


def hourly_frame(days=8, seed=0):
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2025-04-01", periods=days * 24, freq="h")
    diel = np.sin(2 * np.pi * (hours.hour - 9) / 24)
    return pd.DataFrame(
        {
            "timestamp": hours,
            "temperature": 25 + diel,
            "ph": 7.8 + 0.1 * diel,
            "turbidity": 3 + rng.random(len(hours)),
            "dissolved_oxygen": 7 + 1.5 * diel + rng.normal(0, 0.05, len(hours)),
        }
    )


def test_features_only_use_complete_hours_before_the_sample():
    hourly = hourly_frame()
    when = pd.Timestamp("2025-04-06 10:20")
    before = features_at(StationHistory(hourly, S), when, None, S)
    changed = hourly.copy()
    changed.loc[changed["timestamp"] >= when.floor("h"), SENSORS] += 5
    after = features_at(StationHistory(changed, S), when, None, S)
    assert before == pytest.approx(after, nan_ok=True)
    assert before["now_temperature"] == pytest.approx(hourly.set_index("timestamp").loc["2025-04-06 09:00", "temperature"])


def test_sparse_windows_are_skipped():
    hourly = hourly_frame()
    hourly.loc[hourly["timestamp"].between("2025-04-05 12:00", "2025-04-06 09:00"), "dissolved_oxygen"] = np.nan
    assert features_at(StationHistory(hourly, S), pd.Timestamp("2025-04-06 10:00"), None, S) is None


def test_cpcb_class_follows_the_designated_best_use_limits():
    table = assess(
        dissolved_oxygen=[7.0, 7.0, 4.5, 4.5, 3.0, 7.0, np.nan],
        ph=[7.5, 7.5, 7.5, 7.5, 7.5, 9.5, 7.5],
        p_bod_at_most={2.0: [0.9, 0.3, 0.2, 0.1, 0.1, 0.9, 0.9], 3.0: [0.95, 0.8, 0.7, 0.2, 0.2, 0.95, 0.95]},
    )
    assert table["cpcb_class"].tolist() == ["A", "B", "C", "D", "E", "below E", "unknown"]
    assert table.loc[0, "cpcb_not_assessed"] == "total coliforms"


def test_soft_sensor_ranges_cover_fresh_data_and_odds_fall_with_the_limit():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"x": rng.normal(0, 1, 600), "z": rng.normal(0, 1, 600)})
    bod = np.exp(1.0 + 0.4 * X["x"] + rng.normal(0, 0.2, 600))
    sensor = SoftSensor("ridge", candidates(list(X.columns), 0)["ridge"], S).fit(X.iloc[:400], bod[:400])
    est = sensor.estimate(X.iloc[400:])
    inside = (bod[400:] >= est["bod_lo"]) & (bod[400:] <= est["bod_hi"])
    assert inside.mean() == pytest.approx(S["coverage"], abs=0.07)
    assert (est["bod_lo"] <= est["bod"]).all() and (est["bod"] <= est["bod_hi"]).all()
    assert (est["p_above_2"] >= est["p_above_3"]).all()
