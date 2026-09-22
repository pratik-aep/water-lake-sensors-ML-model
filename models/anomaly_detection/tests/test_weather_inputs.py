import numpy as np
import pandas as pd
import pytest

from models.anomaly_detection.data import prepare
from models.anomaly_detection.detector import AnomalyDetector, weather_memory
from models.anomaly_detection.inject import apply_episode
from models.forecasting.simulate import simulate
from models.wqi.generate_data import STATIONS


def test_weather_memory_decays_and_never_looks_ahead():
    hours = pd.date_range("2025-07-01", periods=100, freq="h")
    rain = np.zeros(100)
    rain[10] = 20.0
    weather = pd.DataFrame({"timestamp": hours, "rain_mm": rain})
    when = pd.Series(pd.to_datetime(["2025-07-01 09:50", "2025-07-01 10:20", "2025-07-02 10:00", "2025-07-06 10:00"]))
    memory = weather_memory(weather, when, (6, 36))
    assert (memory.iloc[0] == 0).all()  # before the storm
    assert memory.loc[1, "rain_6h"] == pytest.approx(np.log1p(20.0))
    assert memory.loc[2, "rain_6h"] < memory.loc[2, "rain_36h"]  # short memory has faded, long memory hasn't
    assert memory.loc[3, "rain_36h"] == 0.0  # after the weather ends there is no rain on record


@pytest.fixture(scope="module")
def monsoon():
    # Mid-monsoon, when runoff drives turbidity up for a day or two after each storm.
    raw, weather = simulate(days=60, seed=8, start="2024-07-01", stations={k: STATIONS[k] for k in ("pichola", "goverdhan_sagar")})
    readings, _ = prepare(raw)
    return readings, weather


def test_rain_inputs_explain_turbidity_after_storms(monsoon):
    readings, weather = monsoon
    cutoff = readings["timestamp"].min() + pd.Timedelta(days=46)
    train = readings[readings["timestamp"] < cutoff]
    stream = readings[readings["timestamp"] >= cutoff - pd.Timedelta(days=7)]

    def wet_disagreement(detector, w):
        # How far turbidity sits from what the detector expects, in the day or so after rain.
        result = detector.detect(stream, w)
        result = result[result["timestamp"] >= cutoff]
        wet = weather_memory(weather, result["timestamp"], (36,))["rain_36h"] > 1
        return result.loc[wet, "turbidity_z"].abs().mean()

    blind = wet_disagreement(AnomalyDetector(seed=0).fit(train), None)
    with_rain = AnomalyDetector(seed=0).fit(train, weather)
    assert wet_disagreement(with_rain, weather) < blind
    with pytest.raises(ValueError, match="weather"):
        with_rain.detect(stream)


def test_air_temperature_lets_a_slow_temperature_drift_show(monsoon):
    # Checked only against its own trailing median, a slowly drifting thermometer carries its reference with it.
    readings, weather = monsoon
    cutoff = readings["timestamp"].min() + pd.Timedelta(days=46)
    detector = AnomalyDetector(seed=0).fit(readings[readings["timestamp"] < cutoff], weather)
    stream = readings[readings["timestamp"] >= cutoff - pd.Timedelta(days=7)].reset_index(drop=True)
    start = cutoff + pd.Timedelta(days=2)
    rows = stream.index[(stream["station_id"] == "pichola") & stream["timestamp"].between(start, start + pd.Timedelta(days=5))]
    apply_episode(stream, rows, "drift", "temperature", 1.2)
    flagged = detector.detect(stream, weather).loc[rows, "temperature_fault"].ne("")
    assert flagged.iloc[len(rows) // 2 :].mean() > 0.5  # by the second half of the drift it is plainly visible
