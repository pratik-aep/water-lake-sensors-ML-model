import numpy as np
import pandas as pd
import pytest

from models.forecasting.calibration import fit_offsets, prob_below, widen
from models.forecasting.evaluate import hourly_report, nightly_report


def test_conformal_offsets_restore_the_promised_coverage():
    rng = np.random.default_rng(0)
    groups = np.array(["x"] * 4000)
    actual = rng.normal(0, 1, 4000)
    # Ranges that are far too narrow (+-0.5) should be widened to about +-1.28, an 80% normal range.
    offsets = fit_offsets(np.full(4000, -0.5), np.full(4000, 0.5), actual, groups, coverage=0.8)
    fresh = rng.normal(0, 1, 4000)
    lo, _, hi = widen(np.full(4000, -0.5), np.zeros(4000), np.full(4000, 0.5), groups, offsets)
    assert ((fresh >= lo) & (fresh <= hi)).mean() == pytest.approx(0.8, abs=0.03)


def test_forecasts_missing_for_a_gap_do_not_blank_the_calibration():
    rng = np.random.default_rng(1)
    groups = np.array(["x"] * 1000)
    lo, hi = np.full(1000, -0.5), np.full(1000, 0.5)
    lo[::50] = hi[::50] = np.nan  # no current reading at these origins, so no forecast was made
    offsets = fit_offsets(lo, hi, rng.normal(0, 1, 1000), groups, coverage=0.8)
    assert np.isfinite(offsets["x"]) and offsets["x"] > 0.5


def test_widen_never_lets_the_range_exclude_the_median():
    lo, mid, hi = widen(np.array([1.0]), np.array([2.0]), np.array([3.0]), np.array(["x"]), {"x": -5.0})
    assert lo[0] <= mid[0] <= hi[0]


def test_probability_below_matches_the_quantiles_and_is_monotone():
    lo, mid, hi = np.array([3.0]), np.array([4.0]), np.array([6.0])
    assert prob_below(3.0, lo, mid, hi)[0] == pytest.approx(0.1)
    assert prob_below(4.0, lo, mid, hi)[0] == pytest.approx(0.5)
    assert prob_below(6.0, lo, mid, hi)[0] == pytest.approx(0.9)
    probs = [prob_below(t, lo, mid, hi)[0] for t in np.linspace(0, 10, 50)]
    assert probs == sorted(probs) and probs[0] == 0.0 and probs[-1] == 1.0


def test_hourly_skill_is_measured_against_the_better_baseline():
    df = pd.DataFrame(
        {
            "sensor": "ph",
            "horizon": [1, 2],
            "mid": [7.0, 7.0],
            "lo": [6.9, 6.9],
            "hi": [7.1, 7.1],
            "actual": [7.05, 6.95],
            "persistence": [7.5, 6.5],
            "same_hour_last_day": [7.2, 6.8],
        }
    )
    report = hourly_report(df, [(1, 3)])["ph"]["1-3h"]
    assert report["skill"] == pytest.approx(1 - 0.05 / 0.15, abs=1e-3)
    assert report["coverage_80"] == 1.0


def test_alert_verification_counts_hits_misses_and_false_alarms():
    df = pd.DataFrame(
        {
            "days_ahead": [1, 1, 1, 1],
            "mid": [3.0, 5.0, 3.5, 6.0],
            "lo": [2.0, 4.0, 3.0, 5.0],
            "hi": [4.0, 6.0, 4.0, 7.0],
            "actual": [3.2, 3.5, 4.5, 6.2],
            "last_night": [3.0, 4.0, 4.0, 6.0],
            "week_mean": [3.0, 4.0, 4.0, 6.0],
            "p_below_alert": [0.9, 0.2, 0.7, 0.0],
        }
    )
    alerts = nightly_report(df, threshold=4.0, alert_probability=0.5)["alerts"]["all"]
    assert (alerts["hits"], alerts["misses"], alerts["false_alarms"]) == (1, 1, 1)
    assert alerts["critical_success_index"] == pytest.approx(1 / 3, abs=1e-3)
    assert alerts["brier_score"] == pytest.approx(np.mean([0.01, 0.64, 0.49, 0.0]), abs=1e-4)


def test_a_group_with_too_few_verified_forecasts_is_left_as_it_was():
    groups = np.array(["busy"] * 200 + ["sparse"] * 5)
    lo, hi = np.full(205, -0.5), np.full(205, 0.5)
    offsets = fit_offsets(lo, hi, np.random.default_rng(2).normal(0, 1, 205), groups, coverage=0.8)
    assert set(offsets) == {"busy"}  # five outcomes would set the sparse group's range by luck
