import joblib
import pandas as pd
import pytest

from models.anomaly_detection.config import SENSORS
from models.anomaly_detection.data import prepare
from models.anomaly_detection.detector import AnomalyDetector
from models.anomaly_detection.inject import apply_episode
from models.anomaly_detection.simulate import simulate
from models.wqi.generate_data import STATIONS

STATION = "goverdhan_sagar"


@pytest.fixture(scope="module")
def setup():
    raw = simulate(days=52, seed=3, stations={k: STATIONS[k] for k in ("fateh_sagar", STATION)})
    df, _ = prepare(raw)
    cutoff = df["timestamp"].min() + pd.Timedelta(days=42)
    detector = AnomalyDetector(seed=0).fit(df[df["timestamp"] < cutoff])
    # Holdout plus the preceding week, which rolling baselines use as history.
    stream = df[df["timestamp"] >= cutoff - pd.Timedelta(days=7)].reset_index(drop=True)
    return detector, stream, cutoff


def detect_with(detector, stream, cutoff, kind, sensor="", size=1.2, hours=12):
    dirty = stream.copy()
    start = cutoff + pd.Timedelta(days=4)
    rows = dirty.index[(dirty["station_id"] == STATION) & dirty["timestamp"].between(start, start + pd.Timedelta(hours=hours))]
    apply_episode(dirty, rows, kind, sensor, size)
    result = detector.detect(dirty)
    window = result[(result["station_id"] == STATION) & result["timestamp"].between(start, start + pd.Timedelta(hours=hours))]
    return result, window


def test_clean_stream_raises_few_alarms(setup):
    detector, stream, cutoff = setup
    result = detector.detect(stream)
    result = result[result["timestamp"] >= cutoff]
    alarm_share = result["verdict"].isin(["sensor_fault", "possible_event", "needs_review"]).mean()
    assert alarm_share < 0.02


@pytest.mark.parametrize("sensor", SENSORS)
def test_an_offset_is_blamed_on_the_right_sensor(setup, sensor):
    _, window = detect_with(*setup, "offset", sensor)
    blamed = {s: int(window[f"{s}_fault"].ne("").sum()) for s in SENSORS}
    assert blamed[sensor] > 0
    assert blamed[sensor] >= 5 * max(v for s, v in blamed.items() if s != sensor)


def test_a_spike_is_flagged_as_a_spike(setup):
    _, window = detect_with(*setup, "spike", "turbidity", hours=0.2)
    assert window["turbidity_fault"].eq("spike").any()


def test_a_pollution_event_is_an_event_not_a_sensor_fault(setup):
    _, window = detect_with(*setup, "event", size=1.2, hours=24)
    assert window["verdict"].eq("possible_event").mean() > 0.4
    assert window["verdict"].eq("sensor_fault").mean() < 0.1


def test_a_drift_alarm_clears_soon_after_the_sensor_is_repaired(setup):
    detector, stream, cutoff = setup
    result, window = detect_with(detector, stream, cutoff, "drift", "dissolved_oxygen", size=-1.2, hours=72)
    assert window["dissolved_oxygen_fault"].ne("").iloc[-72:].any()

    # An open alarm is held while the check is paused (here the lake sits at the edge of its training range),
    # so allow a few hours rather than the one hour it takes with unbroken checks.
    repaired_at = window["timestamp"].max()
    after = result[
        (result["station_id"] == STATION)
        & result["timestamp"].between(repaired_at + pd.Timedelta(hours=3), repaired_at + pd.Timedelta(days=1))
    ]
    assert after["dissolved_oxygen_fault"].eq("").all()


def test_a_saved_detector_gives_identical_results(setup, tmp_path):
    detector, stream, _ = setup
    joblib.dump(detector, tmp_path / "detector.joblib")
    reloaded = joblib.load(tmp_path / "detector.joblib")
    pd.testing.assert_frame_equal(detector.detect(stream), reloaded.detect(stream))
