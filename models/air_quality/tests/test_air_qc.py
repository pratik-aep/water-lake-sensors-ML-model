import numpy as np
import pandas as pd
import pytest

from models.air_quality.config import DEFAULT_SETTINGS
from models.air_quality.qc import check, to_hourly
from models.air_quality.simulate import humidity_growth


def readings(hours=48, **overrides):
    stamps = pd.date_range("2025-01-01", periods=hours * 4, freq="15min")
    rng = np.random.default_rng(0)
    base = {
        "station_id": "a", "timestamp": stamps,
        "pm1": 30 + rng.normal(0, 1, len(stamps)), "pm25": 45 + rng.normal(0, 1.5, len(stamps)),
        "pm10": 80 + rng.normal(0, 2, len(stamps)), "tsp": 100 + rng.normal(0, 2, len(stamps)),
        "no2": 0.02 + rng.normal(0, 0.001, len(stamps)), "o3": 0.03 + rng.normal(0, 0.001, len(stamps)),
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_readings_become_hourly_means_and_a_ceiling_reading_is_marked():
    no2 = np.full(48 * 4, 0.02)
    no2[40:44] = 0.1  # hour 10 at the NO2 cell's ceiling
    hourly = to_hourly(readings(no2=no2), DEFAULT_SETTINGS)
    assert len(hourly) == 48 and hourly["no2_at_limit"].sum() == 1
    cleaned = check(hourly, None, DEFAULT_SETTINGS)
    assert cleaned.loc[10, "no2_flag"] == "at_sensor_limit" and cleaned.loc[10, "no2"] == pytest.approx(188.2, abs=0.2)


def test_stuck_and_spiking_sensors_are_set_aside_but_real_peaks_are_kept():
    pm25 = 45 + np.random.default_rng(1).normal(0, 1.5, 48 * 4)
    pm25[80:84] = 900.0  # one hour far above both neighbours: the sensor
    smoke = {name: np.where((np.arange(48 * 4) >= 120) & (np.arange(48 * 4) < 136), level, base)
             for name, level, base in (("pm1", 95.0, 30.0), ("pm10", 190.0, 80.0), ("tsp", 230.0, 100.0))}
    pm25[120:136] = 140.0  # four hours of real smoke, in every size fraction
    o3 = np.full(48 * 4, 0.03)
    o3[:40] += np.random.default_rng(2).normal(0, 0.002, 40)  # then stuck on one value
    cleaned = check(to_hourly(readings(pm25=pm25, o3=o3, **smoke), DEFAULT_SETTINGS), None, DEFAULT_SETTINGS)
    assert cleaned.loc[20, "pm25_flag"] == "spike" and np.isnan(cleaned.loc[20, "pm25"])
    assert cleaned.loc[31, "pm25_flag"] == "" and cleaned.loc[31, "pm25"] > 100
    assert (cleaned["o3_flag"] == "stuck").sum() > 20


def test_humid_air_is_corrected_and_fog_is_set_aside():
    hourly = to_hourly(readings(hours=3), DEFAULT_SETTINGS)
    weather = pd.DataFrame({"timestamp": hourly["timestamp"], "relative_humidity": [50.0, 85.0, 98.0]})
    cleaned = check(hourly, weather, DEFAULT_SETTINGS)
    assert cleaned.loc[1, "pm25"] == pytest.approx(hourly.loc[1, "pm25"] / humidity_growth(85.0), rel=1e-6)
    assert np.isnan(cleaned.loc[2, "pm25"]) and cleaned.loc[2, "pm25_flag"] == "too_humid"


def test_pm2_5_above_pm10_is_flagged_on_both():
    hourly = to_hourly(readings(hours=2, pm25=np.full(8, 120.0)), DEFAULT_SETTINGS)
    cleaned = check(hourly, None, DEFAULT_SETTINGS)
    assert (cleaned["pm25_flag"] == "size_order").all() and (cleaned["pm10_flag"] == "size_order").all()
