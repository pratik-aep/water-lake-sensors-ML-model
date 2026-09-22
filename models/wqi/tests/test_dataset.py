import math

import pandas as pd
import pytest

from models.wqi.calculator import classify_wqi
from models.wqi.config import FEATURE_SETS
from models.wqi.dataset import prepare

SENSOR = FEATURE_SETS["sensor"]


def row(**overrides):
    base = {
        "station_id": "pichola",
        "timestamp": "2025-01-01",
        "temperature": 22.0,
        "ph": 7.6,
        "turbidity": 3.0,
        "dissolved_oxygen": 7.0,
        "bod": 2.0,
        "conductivity": 400.0,
        "nitrate": 4.0,
    }
    return {**base, **overrides}


def frame(*rows):
    return pd.DataFrame(list(rows))


def test_missing_required_column_raises():
    with pytest.raises(ValueError, match="bod"):
        prepare(frame(row()).drop(columns="bod"), SENSOR)


def test_out_of_range_sensor_value_becomes_missing_but_row_is_kept():
    out, report = prepare(frame(row(temperature=99.0)), SENSOR)
    assert len(out) == 1
    assert math.isnan(out.loc[0, "temperature"])
    assert report["out_of_range_set_to_missing"] == {"temperature": 1}


def test_out_of_range_label_value_drops_the_row():
    out, report = prepare(frame(row(), row(timestamp="2025-01-02", ph=15.0)), SENSOR)
    assert len(out) == 1
    assert report["unlabeled_dropped"] == 1


def test_below_detection_limit_uses_half_the_limit_and_junk_is_dropped():
    out, report = prepare(frame(row(bod="<1"), row(timestamp="2025-01-02", bod="n/a")), SENSOR)
    assert out.loc[0, "bod"] == 0.5
    assert report["unlabeled_dropped"] == 1


def test_lab_value_drives_the_label_while_the_feature_keeps_the_sensor_reading():
    sensor_only_wqi = prepare(frame(row()), SENSOR)[0].loc[0, "wqi"]
    out, _ = prepare(frame(row(lab_turbidity=20.0)), SENSOR)
    assert out.loc[0, "wqi"] > sensor_only_wqi
    assert out.loc[0, "turbidity"] == 3.0


def test_duplicate_station_and_timestamp_is_dropped():
    out, report = prepare(frame(row(), row(ph=7.9)), SENSOR)
    assert len(out) == 1
    assert report["duplicates_dropped"] == 1


def test_bad_timestamp_is_dropped():
    out, report = prepare(frame(row(), row(timestamp="not a date")), SENSOR)
    assert len(out) == 1
    assert report["bad_timestamp_dropped"] == 1


def test_row_with_no_sensor_values_is_dropped_even_when_lab_values_exist():
    lab = {"lab_ph": 7.5, "lab_turbidity": 2.0, "lab_dissolved_oxygen": 7.0}
    no_sensors = row(timestamp="2025-01-02", temperature=None, ph=None, turbidity=None, dissolved_oxygen=None)
    out, report = prepare(frame(row(**lab), {**no_sensors, **lab}), SENSOR)
    assert len(out) == 1
    assert report["no_feature_values_dropped"] == 1


def test_supplied_label_column_is_ignored_and_recomputed():
    out, _ = prepare(frame(row(wqi_class="Excellent")), SENSOR)
    assert out.loc[0, "wqi_class"] == classify_wqi(out.loc[0, "wqi"])
    assert out.loc[0, "wqi_class"] != "Excellent"
