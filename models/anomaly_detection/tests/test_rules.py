import numpy as np
import pandas as pd

from models.anomaly_detection.config import DEFAULT_SETTINGS
from models.anomaly_detection.rules import QC_FAIL, QC_MISSING, QC_PASS, QC_SUSPECT, rule_tests

S = DEFAULT_SETTINGS


def noisy(n=200, level=7.0, sd=0.05, seed=0):
    return pd.Series(level + np.random.default_rng(seed).normal(0, sd, n))


def run(series, low=None, high=None):
    return rule_tests(series, "dissolved_oxygen", 0.07, low, high, S)


def test_clean_series_passes():
    flag, reason = run(noisy())
    assert (flag == QC_PASS).all()
    assert (reason == "").all()


def test_missing_and_out_of_range_without_blaming_neighbours():
    s = noisy()
    s.iloc[10] = np.nan
    s.iloc[20] = -5.0
    flag, reason = run(s)
    assert (flag.iloc[10], reason.iloc[10]) == (QC_MISSING, "missing")
    assert (flag.iloc[20], reason.iloc[20]) == (QC_FAIL, "out_of_range")
    assert flag.iloc[19] == QC_PASS and flag.iloc[21] == QC_PASS


def test_two_point_spike_is_a_spike_but_a_step_is_not():
    s = noisy()
    s.iloc[50:52] += 3.0
    s.iloc[120:] += 3.0
    _, reason = run(s)
    assert list(reason.iloc[50:52]) == ["spike", "spike"]
    assert reason.iloc[120] == "rate_of_change"
    assert "spike" not in set(reason.iloc[110:130])


def test_flatline_turns_suspect_then_fail():
    s = noisy()
    s.iloc[100:140] = 6.9
    flag, reason = run(s)
    suspect_at = 100 + S["flat_suspect_points"] - 1
    fail_at = 100 + S["flat_fail_points"] - 1
    assert reason.iloc[suspect_at - 1] == ""
    assert (reason.iloc[suspect_at], flag.iloc[suspect_at]) == ("flatline", QC_SUSPECT)
    assert flag.iloc[fail_at] == QC_FAIL


def test_climatology_band_marks_suspect_only_where_a_band_exists():
    s = noisy()
    s.iloc[60] = 7.3
    low = pd.Series(6.8, index=s.index)
    high = pd.Series(7.2, index=s.index)
    high.iloc[60] = np.nan
    flag, reason = run(s, low, high)
    assert reason.iloc[60] == ""

    high.iloc[60] = 7.2
    flag, reason = run(s, low, high)
    assert (reason.iloc[60], flag.iloc[60]) == ("climatology", QC_SUSPECT)
