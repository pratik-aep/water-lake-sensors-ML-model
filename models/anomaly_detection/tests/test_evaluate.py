import pandas as pd
import pytest

from models.anomaly_detection.config import SENSORS
from models.anomaly_detection.evaluate import alarm_incidents, evaluate

GAP = pd.Timedelta("70min")


def result(verdicts, faulty_sensor="ph"):
    ts = pd.date_range("2025-01-01", periods=len(verdicts), freq="10min")
    out = pd.DataFrame({"station_id": "pichola", "timestamp": ts, "verdict": verdicts, "event": False})
    for s in SENSORS:
        out[s] = 1.0
        out[f"{s}_z"] = 0.0
        out[f"{s}_fault"] = ["spike" if (s == faulty_sensor and v == "sensor_fault") else "" for v in verdicts]
    return out


def test_alarms_less_than_the_gap_apart_merge_into_one_incident():
    verdicts = ["sensor_fault"] + ["normal"] * 5 + ["sensor_fault"] + ["normal"] * 10 + ["sensor_fault"]
    incidents = alarm_incidents(result(verdicts), GAP)
    assert len(incidents) == 2
    assert incidents.loc[0, "sensors"] == "ph"


def test_evaluate_scores_detection_delay_and_false_alarms():
    # The last alarm is well past the episode's six-hour aftermath, so it counts as false.
    verdicts = ["normal"] * 3 + ["sensor_fault"] * 2 + ["normal"] * 50 + ["sensor_fault"]
    res = result(verdicts)
    ts = res["timestamp"]
    episodes = pd.DataFrame(
        [{"station_id": "pichola", "kind": "spike", "sensor": "ph", "start": ts[2], "end": ts[4]}]
    )
    scores = evaluate(res, episodes, GAP)
    spike = scores["per_kind"]["spike"]
    assert (spike["episodes"], spike["detected"], spike["recall"]) == (1, 1, 1.0)
    assert spike["median_delay_hours"] == pytest.approx(1 / 6, abs=0.01)
    assert scores["incidents"] == 2
    assert scores["false_incidents"] == 1
    assert scores["consistency_coverage"]["ph"] == 1.0
