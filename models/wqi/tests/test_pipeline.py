import joblib
import pandas as pd
import pytest

from models.wqi import predict, train
from models.wqi.config import CLASS_ORDER, FEATURE_SETS
from models.wqi.dataset import prepare
from models.wqi.generate_data import generate


@pytest.fixture(scope="module")
def trained_bundle(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("wqi")
    data = out_dir / "readings.csv"
    generate(days=150, seed=1).to_csv(data, index=False)
    train.main(["--data", str(data), "--out-dir", str(out_dir), "--cv-folds", "3"])
    return joblib.load(out_dir / "wqi_sensor.joblib")


def test_bundle_records_how_the_model_was_trained(trained_bundle):
    assert trained_bundle["model_name"] in train.CANDIDATES
    assert trained_bundle["features"] == FEATURE_SETS["sensor"]
    assert len(trained_bundle["data_sha256"]) == 64
    assert set(trained_bundle["feature_ranges"]) == set(FEATURE_SETS["sensor"])


def test_classify_returns_a_class_and_probabilities_that_sum_to_one(trained_bundle):
    readings = pd.DataFrame([{"temperature": 25, "ph": 7.8, "turbidity": 4.0, "dissolved_oxygen": 6.5}])
    result = predict.classify(trained_bundle, readings)
    assert result.loc[0, "predicted_class"] in CLASS_ORDER
    assert result.filter(like="p_").sum(axis=1).iloc[0] == pytest.approx(1.0)


def test_check_reading_blocks_impossible_values_and_warns_on_unseen_ones(trained_bundle):
    ok = {"temperature": 25, "ph": 7.8, "turbidity": 4.0, "dissolved_oxygen": 6.5}
    assert predict.check_reading(trained_bundle, ok) == ([], [])

    errors, _ = predict.check_reading(trained_bundle, {**ok, "ph": 15.0})
    assert any("physical range" in e for e in errors)

    errors, _ = predict.check_reading(trained_bundle, {k: v for k, v in ok.items() if k != "ph"})
    assert any("'ph'" in e for e in errors)

    _, warnings = predict.check_reading(trained_bundle, {**ok, "turbidity": 3000.0})
    assert any("training range" in w for w in warnings)


def test_time_holdout_never_trains_on_the_future():
    df, _ = prepare(generate(days=100, seed=2), FEATURE_SETS["sensor"])
    train_df, test_df = train.split_holdout(df, "time", 0.2, seed=0)
    assert train_df["timestamp"].max() < test_df["timestamp"].min()
