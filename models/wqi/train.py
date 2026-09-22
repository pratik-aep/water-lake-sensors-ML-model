"""Select, evaluate and save the WQI classifier."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import CLASS_ORDER, FEATURE_SETS
from .dataset import load_dataset

HERE = Path(__file__).parent
MACRO_F1 = make_scorer(f1_score, average="macro", zero_division=0)

# Every real candidate reweights classes: polluted readings are rare, and missing them is the costly error.
CANDIDATES = {
    "majority_baseline": (DummyClassifier(strategy="most_frequent"), {}),
    "logistic_regression": (
        make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000),
        ),
        {"logisticregression__C": [0.1, 1.0, 10.0]},
    ),
    "random_forest": (
        RandomForestClassifier(n_estimators=300, class_weight="balanced"),
        {"max_depth": [None, 10], "min_samples_leaf": [1, 3], "max_features": ["sqrt", 1.0]},
    ),
    "hist_gradient_boosting": (
        HistGradientBoostingClassifier(max_iter=300, class_weight="balanced"),
        {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31], "l2_regularization": [0.0, 1.0]},
    ),
}


def split_holdout(df: pd.DataFrame, method: str, test_size: float, seed: int):
    if method == "time":
        is_test = df["timestamp"] > df["timestamp"].quantile(1 - test_size)
        return df[~is_test], df[is_test]
    y = df["wqi_class"]
    stratify = y if y.value_counts().min() >= 2 else None
    return train_test_split(df, test_size=test_size, stratify=stratify, random_state=seed)


def cv_splitter(df: pd.DataFrame, folds: int, seed: int):
    """Readings from the same station and week share a fold, so near-duplicates can't leak across folds."""
    if "timestamp" not in df.columns:
        return StratifiedKFold(folds, shuffle=True, random_state=seed), None
    groups = df["timestamp"].dt.to_period("W").astype(str)
    if "station_id" in df.columns:
        groups = df["station_id"].astype(str) + "|" + groups
    return StratifiedGroupKFold(folds, shuffle=True, random_state=seed), groups


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "data" / "synthetic_lake_readings.csv")
    parser.add_argument("--features", choices=FEATURE_SETS, default="sensor")
    parser.add_argument(
        "--holdout",
        choices=["auto", "time", "random"],
        default="auto",
        help="time = test on the most recent readings; auto picks time whenever a timestamp column exists",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--models", nargs="+", choices=CANDIDATES, default=list(CANDIDATES))
    parser.add_argument("--out-dir", type=Path, default=HERE / "artifacts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    features = FEATURE_SETS[args.features]
    df, data_report = load_dataset(args.data, features)
    holdout = args.holdout
    if holdout == "auto":
        holdout = "time" if "timestamp" in df.columns else "random"
    elif holdout == "time" and "timestamp" not in df.columns:
        parser.error("--holdout time needs a 'timestamp' column")

    train_df, test_df = split_holdout(df, holdout, args.test_size, args.seed)
    X_train, y_train = train_df[features], train_df["wqi_class"]
    X_test, y_test = test_df[features], test_df["wqi_class"]
    cv, groups = cv_splitter(train_df, args.cv_folds, args.seed)

    searches = {}
    for name in args.models:
        estimator, grid = CANDIDATES[name]
        estimator = clone(estimator)
        if "random_state" in estimator.get_params():
            estimator.set_params(random_state=args.seed)
        search = GridSearchCV(estimator, grid, scoring=MACRO_F1, cv=cv, n_jobs=-1, error_score="raise")
        searches[name] = search.fit(X_train, y_train, groups=groups)

    leaderboard = sorted(
        (
            {
                "model": name,
                "cv_macro_f1": round(float(s.best_score_), 4),
                "cv_std": round(float(s.cv_results_["std_test_score"][s.best_index_]), 4),
                "params": s.best_params_,
            }
            for name, s in searches.items()
        ),
        key=lambda row: row["cv_macro_f1"],
        reverse=True,
    )
    best_name = leaderboard[0]["model"]
    best = searches[best_name].best_estimator_

    y_pred = best.predict(X_test)
    labels = [c for c in CLASS_ORDER if c in set(y_train) | set(y_test)]
    rank = {label: i for i, label in enumerate(CLASS_ORDER)}
    steps_too_clean = pd.Series([rank[t] - rank[p] for t, p in zip(y_test, y_pred)])
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    perm = permutation_importance(
        best, X_test, y_test, scoring=MACRO_F1, n_repeats=10, random_state=args.seed, n_jobs=-1
    )
    importance = sorted(
        ((f, float(m), float(s)) for f, m, s in zip(features, perm.importances_mean, perm.importances_std)),
        key=lambda row: row[1],
        reverse=True,
    )

    final = clone(best).fit(df[features], df["wqi_class"])
    bundle = {
        "model": final,
        "model_name": best_name,
        "params": searches[best_name].best_params_,
        "feature_set": args.features,
        "features": features,
        "feature_ranges": {f: [float(df[f].min()), float(df[f].max())] for f in features},
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "data_file": str(args.data),
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "rows": len(df),
    }
    metrics = {
        **{k: v for k, v in bundle.items() if k != "model"},
        "data_report": data_report,
        "holdout": {
            "method": holdout,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "note": "Holdout scores come from the model fit on the training split; the saved model is refit on all rows.",
        },
        "leaderboard": leaderboard,
        "holdout_evaluation": {
            "report": classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0),
            "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
            "rate_predicted_cleaner_than_actual": round(float((steps_too_clean > 0).mean()), 4),
            "count_predicted_2plus_classes_cleaner": int((steps_too_clean >= 2).sum()),
        },
        "permutation_importance": {f: {"mean": round(m, 4), "std": round(s, 4)} for f, m, s in importance},
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / f"wqi_{args.features}.joblib"
    joblib.dump(bundle, model_path)
    (args.out_dir / f"wqi_{args.features}_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    dropped = {k: v for k, v in data_report.items() if k.endswith("_dropped") and v}
    print(f"Data: {data_report['rows_in']} rows in, {data_report['rows_out']} usable. Dropped: {dropped or 'none'}")
    if data_report["out_of_range_set_to_missing"]:
        print(f"Out-of-range values set to missing: {data_report['out_of_range_set_to_missing']}")
    print(f"Classes: {data_report['class_counts']}")
    print(f"Features ({args.features}): {features}")
    if holdout == "time":
        print(f"Holdout: most recent readings, from {test_df['timestamp'].min().date()} ({len(test_df)} rows)")
    else:
        print(f"Holdout: random stratified sample ({len(test_df)} rows); scores may be optimistic for time-series data")

    print(f"\n{args.cv_folds}-fold CV leaderboard (macro-F1, training split):")
    for row in leaderboard:
        print(f"  {row['model']:<24} {row['cv_macro_f1']:.3f} ± {row['cv_std']:.3f}  {row['params']}")

    print(f"\nSelected: {best_name}. Holdout results:")
    print(classification_report(y_test, y_pred, labels=labels, zero_division=0))
    print("Confusion matrix (rows = actual, cols = predicted):")
    print(pd.DataFrame(cm, index=labels, columns=labels).to_string(), "\n")
    evaluation = metrics["holdout_evaluation"]
    print(f"Predicted cleaner than actual: {evaluation['rate_predicted_cleaner_than_actual']:.1%} of holdout rows")
    print(f"Predicted 2+ classes cleaner:  {evaluation['count_predicted_2plus_classes_cleaner']} rows\n")
    print("Permutation importance (macro-F1 drop when a feature is shuffled):")
    for feature, mean, std in importance:
        print(f"  {feature:<17} {mean:.3f} ± {std:.3f}")
    print(f"\nSaved {best_name} (refit on all {len(df)} rows) to {model_path}")


if __name__ == "__main__":
    main()
