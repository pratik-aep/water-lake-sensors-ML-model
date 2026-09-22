import argparse
import json

import numpy as np
import pandas as pd
import pytest

from models.pipeline import check_data
from models.pipeline.ingest import load_mapping, normalise, turbidity_unit
from models.pipeline.train_all import attempt
from models.wqi.calculator import do_saturation_mgl


def vendor_export():
    return pd.DataFrame({
        "Device": ["LAKE-01", "LAKE-01", "LAKE-01", "LAKE-02"],
        "Time (UTC)": ["2025-07-01T04:30:00Z", "2025-07-01T04:40:00Z", "2025-07-01T04:50:00Z", "2025-07-01T04:30:00Z"],
        "Water Temp": [25.0, 25.0, 25.0, 30.0],
        "pH": [7.8, 9.0, 8.1, 4.0],
        "Turbidity %": [10.0, 12.0, 11.0, 50.0],
        "DO %": [80.0, 100.0, 0.0, 50.0],
    })


def vendor_mapping(tmp_path, **extra):
    mapping = {
        "columns": {"station_id": "Device", "timestamp": "Time (UTC)", "temperature": "Water Temp", "ph": "pH",
                    "turbidity": "Turbidity %", "dissolved_oxygen": "DO %"},
        "stations": {"LAKE-01": "pichola", "LAKE-02": "fateh_sagar"},
        "units": {"dissolved_oxygen": "percent", "turbidity": "percent"},
        "sensor_ranges": {"ph": [4, 9], "dissolved_oxygen": [None, 100]},
        "timezone": "Asia/Kolkata",
        **extra,
    }
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(mapping))
    return load_mapping(path)


def test_a_vendor_export_becomes_model_readings(tmp_path):
    readings, report = normalise(vendor_export(), vendor_mapping(tmp_path))
    assert list(readings["station_id"]) == ["pichola", "pichola", "pichola", "fateh_sagar"]
    # UTC 04:30 is 10:00 in Udaipur; the daily-cycle features need local clock time.
    assert readings["timestamp"].iloc[0] == pd.Timestamp("2025-07-01 10:00")
    # 80% saturation at 25 C and Udaipur's pressure, in mg/L.
    assert readings["dissolved_oxygen"].iloc[0] == pytest.approx(0.8 * do_saturation_mgl(25.0, 0.93))
    # Pinned at the top of the range: set aside. Zero oxygen is a real (and alarming) reading: kept.
    assert np.isnan(readings["dissolved_oxygen"].iloc[1]) and readings["dissolved_oxygen"].iloc[2] == 0
    assert np.isnan(readings["ph"].iloc[1]) and np.isnan(readings["ph"].iloc[3])
    assert report["censored_at_sensor_limit"] == {"ph": 2, "dissolved_oxygen": 1}
    assert report["censored_by_station"] == {"pichola": {"ph": 1, "dissolved_oxygen": 1}, "fateh_sagar": {"ph": 1}}
    assert report["turbidity_unit_out"] == "%" and readings["turbidity"].iloc[0] == 10.0


def test_a_turbidity_calibration_turns_percent_into_ntu(tmp_path):
    mapping = vendor_mapping(tmp_path, turbidity_percent_to_ntu=[2.0, 0.5])
    readings, _ = normalise(vendor_export(), mapping)
    assert turbidity_unit(mapping) == "NTU" and readings["turbidity"].iloc[0] == pytest.approx(20.5)


def test_a_misnamed_column_is_reported_with_what_the_export_has(tmp_path):
    raw = vendor_export().rename(columns={"DO %": "Dissolved Oxygen"})
    with pytest.raises(ValueError, match="DO %.*Dissolved Oxygen"):
        normalise(raw, vendor_mapping(tmp_path))


def test_unknown_units_are_refused(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"units": {"dissolved_oxygen": "ppm"}}))
    with pytest.raises(ValueError, match="ppm"):
        load_mapping(path)


def test_readiness_says_what_each_model_still_needs():
    stamps = pd.date_range("2025-01-01", periods=24 * 6 * 40, freq="10min")
    readings = pd.DataFrame({"station_id": "a", "timestamp": stamps, "temperature": 25.0, "ph": 7.5,
                             "turbidity": 5.0, "dissolved_oxygen": 7.0})
    lab = pd.DataFrame({"station_id": "a", "timestamp": pd.date_range("2025-01-05 10:00", periods=5, freq="7D"),
                        "bod": 2.0})
    _, rows = check_data.readiness(readings, lab, None)
    verdict = {name: (ready, reason) for name, ready, reason in rows}
    assert not verdict["anomaly detector"][0] and "40 days" in verdict["anomaly detector"][1]
    assert verdict["BOD soft sensor"] == (False, "5 lab samples with sensor data around them; needs 38")
    assert "no 'chlorophyll_a' column" in verdict["algae soft sensor"][1]


def test_a_model_without_enough_data_is_skipped_with_the_reason():
    def too_little():
        argparse.ArgumentParser(prog="bod").error("only 12 lab samples have sensor data around them; need more")

    assert attempt(too_little) == "only 12 lab samples have sensor data around them; need more"
    assert attempt(lambda: None) is None
