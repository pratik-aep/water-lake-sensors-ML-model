import numpy as np
import pandas as pd
import pytest

from models.algal_bloom.config import DEFAULT_SETTINGS
from models.algal_bloom.trophic import carlson_tsi, chlorophyll_at, class_probabilities, trophic_class, who_level
from models.bod_surrogate.models import SoftSensor, candidates


def test_carlson_index_matches_the_published_class_boundaries():
    # Carlson (1977): about 2.6, 7.3 and 56 ug/L chlorophyll-a sit at TSI 40, 50 and 70.
    assert carlson_tsi([2.6, 7.3, 56.0]) == pytest.approx([40.0, 50.1, 70.1], abs=0.2)
    assert chlorophyll_at(50.0) == pytest.approx(7.23, abs=0.01)
    assert trophic_class([35, 45, 60, 80]).tolist() == ["oligotrophic", "mesotrophic", "eutrophic", "hypereutrophic"]


def test_trophic_class_odds_sum_to_one_and_follow_the_estimate():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.normal(0, 1, 400)})
    chl = np.exp(2.5 + 0.8 * X["x"] + rng.normal(0, 0.2, 400))
    settings = {**DEFAULT_SETTINGS, "cv_folds": 5}
    sensor = SoftSensor("ridge", candidates(["x"], 0)["ridge"], settings, "chlorophyll_a").fit(X, chl)
    odds = class_probabilities(sensor, pd.DataFrame({"x": [-3.0, 0.0, 3.0]}))
    assert odds.sum(axis=1).to_numpy() == pytest.approx([1.0, 1.0, 1.0])
    assert odds["p_oligotrophic"].iloc[0] > 0.5 and odds["p_hypereutrophic"].iloc[2] > 0.5


def test_who_level_steps_up_with_the_odds_of_crossing_12_and_24():
    levels = who_level([0.1, 0.7, 0.9], [0.0, 0.2, 0.6], alert_probability=0.5)
    assert levels.tolist() == ["Vigilance", "Alert Level 1", "Alert Level 1+ (check for scum)"]
