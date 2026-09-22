import pytest

from models.wqi.calculator import (
    WEIGHTS,
    classify_wqi,
    compute_wqi,
    do_percent_to_mgl,
    do_saturation_mgl,
    quality_rating,
)
from models.wqi.config import PARAMETERS


def test_weights_sum_to_one():
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)


def test_rating_is_0_at_ideal_and_100_at_standard():
    for name, p in PARAMETERS.items():
        assert quality_rating(name, p["ideal"]) == pytest.approx(0.0)
        assert quality_rating(name, p["si"]) == pytest.approx(100.0)


def test_acidic_ph_is_rated_against_lower_limit():
    assert quality_rating("ph", 6.5) == pytest.approx(100.0)
    assert quality_rating("ph", 6.75) == pytest.approx(50.0)


def test_supersaturated_do_is_not_negative():
    assert quality_rating("dissolved_oxygen", 16.0) == 0.0


def test_wqi_is_0_at_ideal_and_100_at_standard():
    assert compute_wqi({n: p["ideal"] for n, p in PARAMETERS.items()}) == pytest.approx(0.0)
    assert compute_wqi({n: p["si"] for n, p in PARAMETERS.items()}) == pytest.approx(100.0)


@pytest.mark.parametrize(
    "wqi, expected",
    [
        (0, "Excellent"),
        (25, "Excellent"),
        (25.01, "Good"),
        (50, "Good"),
        (75, "Poor"),
        (100, "Very Poor"),
        (100.01, "Unsuitable"),
        (450, "Unsuitable"),
    ],
)
def test_class_boundaries(wqi, expected):
    assert classify_wqi(wqi) == expected


def test_do_saturation_matches_usgs_table():
    assert do_saturation_mgl(20.0) == pytest.approx(9.09, abs=0.01)
    assert do_saturation_mgl(25.0) == pytest.approx(8.26, abs=0.01)


def test_do_percent_conversion():
    assert do_percent_to_mgl(50.0, 20.0) == pytest.approx(9.09 / 2, abs=0.01)
    assert do_percent_to_mgl(100.0, 20.0, pressure_atm=0.93) == pytest.approx(9.09 * 0.93, abs=0.01)
