import numpy as np
import pandas as pd
import pytest

from models.anomaly_detection.config import SENSORS
from models.anomaly_detection.detector import TrendPlusTrees, ewma, isolate, sustained


def test_drift_average_follows_a_persistent_bias_and_stays_bounded_under_autocorrelated_noise():
    z = np.zeros((400, 1))
    z[200:, 0] = 1.5
    stat = np.abs(ewma(z, weight=0.1, reset=np.zeros(400, bool), cap=np.inf))
    assert stat[199, 0] == 0
    assert stat[-1, 0] == pytest.approx(1.5)
    # A slow random wander (strongly autocorrelated, like 10-minute residuals) never pushes it past its own size.
    wander = np.cumsum(np.random.default_rng(0).normal(0, 0.05, (5000, 1)), axis=0)
    wander = np.clip(wander - wander.mean(), -2, 2)
    assert np.abs(ewma(wander, 0.1, np.zeros(5000, bool), np.inf)).max() <= 2


def test_drift_average_carries_over_missing_values_restarts_on_reset_and_respects_the_cap():
    z = np.full((12, 1), 2.0)
    z[3, 0] = np.nan
    reset = np.zeros(12, bool)
    reset[6] = True
    stat = np.abs(ewma(z, weight=0.5, reset=reset, cap=1.8))
    assert stat[3, 0] == stat[2, 0]
    assert stat[6, 0] == 0 and stat[7, 0] == pytest.approx(1.0)
    assert stat.max() == pytest.approx(1.8)


def test_drift_average_reset_can_target_single_columns():
    z = np.full((50, 2), 2.0)
    reset = np.zeros((50, 2), bool)
    reset[:, 1] = True
    stat = np.abs(ewma(z, weight=0.5, reset=reset, cap=np.inf))
    assert stat[-1, 0] == pytest.approx(2.0) and stat[-1, 1] == 0


def test_sustained_is_causal_and_restarts_at_each_station():
    mask = pd.Series([True, True, True, True, False, True, True, True])
    station = pd.Series(["a"] * 6 + ["b"] * 2)
    assert list(sustained(mask, station, 3)) == [False, False, True, True, False, False, False, False]


def frame(values):
    return pd.DataFrame([values], columns=SENSORS)


def others_frame(values_by_sensor):
    return {f: pd.DataFrame([{s: v for s, v in values_by_sensor[f].items()}]) for f in SENSORS}


def test_isolate_blames_the_sensor_whose_reconstruction_restores_agreement():
    own = frame([0.2, 3.0, 0.5, 2.5])
    others = others_frame(
        {
            "temperature": {"ph": 3.0, "turbidity": 0.5, "dissolved_oxygen": 2.5},
            "ph": {"temperature": 0.2, "turbidity": 0.4, "dissolved_oxygen": 0.6},
            "turbidity": {"temperature": 0.2, "ph": 3.0, "dissolved_oxygen": 2.5},
            "dissolved_oxygen": {"temperature": 0.2, "ph": 2.0, "turbidity": 0.5},
        }
    )
    assert isolate(own, others).iloc[0] == "ph"


def test_isolate_returns_blank_when_no_single_sensor_explains_it():
    own = frame([0.2, 3.0, 0.5, 2.5])
    everyone_still_off = {"temperature": 0.2, "ph": 3.0, "turbidity": 0.5, "dissolved_oxygen": 2.5}
    others = others_frame({f: {s: v for s, v in everyone_still_off.items() if s != f} for f in SENSORS})
    assert isolate(own, others).iloc[0] == ""


def test_trend_plus_trees_keeps_extrapolating_a_trend():
    x = np.linspace(0, 10, 200)
    X = pd.DataFrame({"x": x, "station": 0.0})
    model = TrendPlusTrees(random_state=0).fit(X, 2 * x)
    beyond = model.predict(pd.DataFrame({"x": [15.0], "station": [0.0]}))[0]
    assert beyond == pytest.approx(30.0, abs=1.5)
