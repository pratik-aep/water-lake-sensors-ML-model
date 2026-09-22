import pandas as pd
import pytest

from models.anomaly_detection.data import prepare


def readings(timestamps, **overrides):
    n = len(timestamps)
    base = {
        "station_id": ["pichola"] * n,
        "timestamp": timestamps,
        "temperature": [20.0] * n,
        "ph": [7.5] * n,
        "turbidity": [2.0] * n,
        "dissolved_oxygen": [8.0] * n,
    }
    return pd.DataFrame({**base, **overrides})


def test_readings_snap_to_a_regular_grid_with_gaps_as_empty_rows():
    df = readings(["2025-01-01 00:00:04", "2025-01-01 00:10:00", "2025-01-01 00:40:00"], temperature=[20.0, 20.1, "err"])
    out, report = prepare(df)
    assert list(out["timestamp"].dt.strftime("%H:%M")) == ["00:00", "00:10", "00:20", "00:30", "00:40"]
    assert report["gap_rows_added"] == 2
    assert out["temperature"].isna().sum() == 3
    assert (out["station_id"] == "pichola").all()


def test_duplicates_and_bad_timestamps_are_dropped():
    out, report = prepare(readings(["2025-01-01 00:00", "2025-01-01 00:00", "not a time", "2025-01-01 00:10"]))
    assert len(out) == 2
    assert report["duplicates_dropped"] == 1
    assert report["bad_timestamp_dropped"] == 1


def test_missing_column_raises():
    with pytest.raises(ValueError, match="turbidity"):
        prepare(readings(["2025-01-01 00:00"]).drop(columns="turbidity"))
