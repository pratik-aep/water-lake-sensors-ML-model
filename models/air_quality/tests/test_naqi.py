import numpy as np
import pandas as pd
import pytest

from models.air_quality.naqi import averaged, category, naqi, ppm_to_mass, sub_index


def test_sub_indices_follow_the_cpcb_breakpoints():
    # PM2.5 30 -> 50 and 60 -> 100 (band edges); 45 sits halfway; beyond the last band the index stops at 500.
    assert sub_index([30, 45, 60, 90, 1000], "pm25").tolist() == [50, 75, 100, 200, 500]
    assert sub_index([100, 250], "pm10").tolist() == [100, 200]
    assert np.isnan(sub_index([np.nan], "no2")[0])


def test_categories_use_whole_number_bands():
    assert category([50, 51, 100, 101, 200, 201, 301, 401]).tolist() == [
        "Good", "Satisfactory", "Satisfactory", "Moderate", "Moderate", "Poor", "Very Poor", "Severe"]
    assert category([np.nan])[0] == "insufficient data"


def test_ppm_becomes_the_mass_units_of_the_index():
    assert ppm_to_mass(0.1, "no2") == pytest.approx(188.2, abs=0.1)  # the vendor NO2 cell's ceiling, in ug/m3
    assert ppm_to_mass(1.0, "co") == pytest.approx(1.146, abs=0.001)  # CO is indexed in mg/m3


def hourly(hours, **values):
    return pd.DataFrame({"station_id": "a", "timestamp": pd.date_range("2025-01-01", periods=hours, freq="h"), **values})


def test_a_24_hour_average_needs_16_hours_and_ozone_uses_its_worst_8_hours():
    ozone = np.full(24, 40.0)
    ozone[12:20] = 120.0  # an afternoon peak
    pm = np.full(24, 50.0)
    pm[:10] = np.nan  # only 14 hours of particles
    avg = averaged(hourly(24, pm25=pm, o3=ozone))
    assert np.isnan(avg["pm25"].iloc[-1])
    assert avg["o3"].iloc[-1] == pytest.approx(120.0)


def test_an_index_needs_three_pollutants_including_particles():
    base = hourly(1, pm25=[45.0], no2=[30.0], so2=[10.0])
    result = naqi(base)
    assert result.loc[0, "aqi"] == 75 and result.loc[0, "prominent_pollutant"] == "pm25"
    assert result.loc[0, "category"] == "Satisfactory"
    no_particles = naqi(hourly(1, no2=[300.0], so2=[10.0], o3=[20.0]))
    assert np.isnan(no_particles.loc[0, "aqi"]) and no_particles.loc[0, "category"] == "insufficient data"
