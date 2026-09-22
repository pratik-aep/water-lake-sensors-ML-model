import numpy as np
import pandas as pd
import pytest

from models.bod_surrogate.config import DEFAULT_SETTINGS
from models.bod_surrogate.metabolism import daily_metabolism
from models.wqi.calculator import do_saturation_mgl

K, R, P = 0.08, 0.3, 1.0  # reaeration (1/h), respiration (mg/L/h), peak photosynthesis (mg/L/h)


def lake(days=30, seed=0):
    """Oxygen integrated at 10-minute steps from known metabolism, then averaged to hourly like real sensors."""
    rng = np.random.default_rng(seed)
    steps = pd.date_range("2025-04-01", periods=days * 144, freq="10min")
    hour = steps.hour + steps.minute / 60
    temperature = 25 + 0.5 * np.sin(2 * np.pi * (hour - 10) / 24)
    saturation = do_saturation_mgl(temperature, DEFAULT_SETTINGS["pressure_atm"])
    light = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None)
    oxygen = np.empty(len(steps))
    oxygen[0] = saturation[0]
    for t in range(1, len(steps)):
        oxygen[t] = oxygen[t - 1] + (P * light[t] - R + K * (saturation[t] - oxygen[t - 1])) / 6
    frame = pd.DataFrame({"temperature": temperature, "dissolved_oxygen": oxygen + rng.normal(0, 0.02, len(steps))}, index=steps)
    hourly = frame.resample("h").mean().rename_axis("timestamp").reset_index()
    daily_production = pd.Series(P * light / 6, index=steps)
    return hourly, daily_production.between_time("06:00", "18:59").resample("D").sum()


def test_night_regression_recovers_reaeration_and_respiration():
    hourly, _ = lake()
    table = daily_metabolism(hourly, DEFAULT_SETTINGS).iloc[3:]
    assert table["reaeration"].median() == pytest.approx(K, rel=0.15)
    assert table["respiration"].median() == pytest.approx(R, rel=0.1)


def test_daytime_production_matches_the_light_driven_truth():
    hourly, true_gpp = lake()
    table = daily_metabolism(hourly, DEFAULT_SETTINGS).iloc[3:-1]
    ratio = (table["gpp"] / true_gpp.reindex(table.index)).median()
    assert ratio == pytest.approx(1.0, abs=0.15)
    assert (table["nep"] < table["gpp"]).all()


def test_missing_nights_are_left_empty_not_invented():
    hourly, _ = lake(days=10)
    hourly.loc[(hourly["timestamp"] >= "2025-04-05 20:00") & (hourly["timestamp"] <= "2025-04-06 06:00"), "dissolved_oxygen"] = np.nan
    table = daily_metabolism(hourly, DEFAULT_SETTINGS)
    assert np.isnan(table.loc["2025-04-06", "respiration"])
    assert not np.isnan(table.loc["2025-04-07", "respiration"])
