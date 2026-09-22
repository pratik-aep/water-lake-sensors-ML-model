"""Shared training for soft sensors of a lab-measured value: model choice, input selection and honest testing."""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .evaluate import accuracy, exceedance, range_quality
from .features import METABOLISM
from .models import SoftSensor, candidates, out_of_fold, time_folds


def input_groups(columns, metabolism=METABOLISM, weather=("rain_72h", "cloud_24h")) -> dict:
    groups = {
        "oxygen metabolism": [c for c in columns if c in metabolism],
        "pH": [c for c in columns if c.endswith("_ph")],
        "turbidity": [c for c in columns if c.endswith("_turbidity")],
        "weather": [c for c in columns if c in weather],
    }
    return {g: cols for g, cols in groups.items() if cols}


def fit_soft_sensor(X: pd.DataFrame, lab: pd.DataFrame, target: str, settings: dict, seed: int, groups: dict,
                    monotone, extra_scores=None) -> tuple[SoftSensor, dict]:
    """Choose, test and refit a soft sensor; X and lab are aligned, time-sorted rows, one per lab sample."""
    s = settings
    values = lab[target].to_numpy()
    n_test = max(1, round(len(lab) * s["test_fraction"]))
    train, test = np.arange(len(lab) - n_test), np.arange(len(lab) - n_test, len(lab))
    folds = time_folds(len(train), s["cv_folds"])
    log_train = np.log(values[train])

    def cv_error(columns, name) -> float:
        oof = out_of_fold(candidates(columns, seed, monotone)[name], X.iloc[train][columns], log_train, folds)
        return round(float(np.sqrt(np.mean((oof - log_train) ** 2))), 4)

    cv_rmse = {name: cv_error(list(X.columns), name) for name in candidates(list(X.columns), seed, monotone)}
    best = min(cv_rmse, key=cv_rmse.get)

    # Which sensor information carries the signal at these lakes. With few lab samples an uninformative group
    # mostly adds noise, so groups are dropped one at a time while that clearly lowers the cross-validated error.
    input_check = {"all inputs": cv_rmse[best]}
    input_check.update({f"without {g}": cv_error([c for c in X.columns if c not in cols], best) for g, cols in groups.items()})
    selected, current, dropped = list(X.columns), cv_rmse[best], []
    while True:
        trials = {g: [c for c in selected if c not in cols] for g, cols in groups.items() if g not in dropped}
        scores = {g: cv_error(cols, best) for g, cols in trials.items() if cols}
        if not scores or min(scores.values()) > 0.98 * current:
            break
        group = min(scores, key=scores.get)
        selected, current = trials[group], scores[group]
        dropped.append(group)

    def make():
        return SoftSensor(best, candidates(selected, seed, monotone)[best], s, target)

    Xs = X[selected]
    sensor = make().fit(Xs.iloc[train], values[train])
    estimate = sensor.estimate(Xs.iloc[test])
    actual = values[test]
    # What a manager assumes between samples today: the station's previous lab result.
    last_value = lab.groupby("station_id")[target].shift().to_numpy()[test]
    has_previous = ~np.isnan(last_value)
    station_median = lab.iloc[train].groupby("station_id")[target].median()
    usual = lab.iloc[test]["station_id"].map(station_median).fillna(np.median(values[train])).to_numpy()
    test_scores = {
        "samples": int(n_test),
        "from": str(lab["timestamp"].iloc[test[0]]),
        "soft_sensor": {
            **accuracy(actual, estimate[target]),
            **range_quality(actual, estimate[f"{target}_lo"], estimate[f"{target}_hi"]),
        },
        "last_lab_value": accuracy(actual[has_previous], last_value[has_previous]),
        "station_median": accuracy(actual, usual),
        "exceedance": {
            f">{lv:g}": exceedance(actual, estimate[f"p_above_{lv:g}"], lv, s["alert_probability"]) for lv in s["exceedance_levels"]
        },
    }
    if extra_scores is not None:
        test_scores.update(extra_scores(X.iloc[test], actual, estimate))
    importance = permutation_importance(
        sensor.model_, Xs.iloc[test], np.log(actual), scoring="neg_mean_absolute_error", n_repeats=20, random_state=seed
    )
    top = sorted(zip(selected, importance.importances_mean), key=lambda kv: kv[1], reverse=True)[:8]

    # A lake the model has never seen: train on the others, predict this one.
    unseen_lake = {}
    for station in sorted(lab["station_id"].unique()):
        inside = (lab["station_id"] != station).to_numpy()
        if inside.sum() >= s["min_training_samples"]:
            held_out = make().fit(Xs[inside], values[inside]).estimate(Xs[~inside])
            unseen_lake[station] = accuracy(values[~inside], held_out[target])

    # How accuracy grows with the number of lab samples, to plan the sampling programme.
    rng = np.random.default_rng(seed)
    learning = {}
    for fraction in (0.25, 0.5, 0.75, 1.0):
        size = max(s["min_training_samples"] // 2, int(len(train) * fraction))
        maes = []
        for _ in range(1 if fraction == 1.0 else 5):
            subset = np.sort(rng.choice(train, size=size, replace=False))
            maes.append(accuracy(actual, make().fit(Xs.iloc[subset], values[subset]).estimate(Xs.iloc[test])[target])["mae"])
        learning[int(size)] = round(float(np.mean(maes)), 3)

    results = {
        "model": best,
        "cv_rmse_log": cv_rmse,
        "input_check_cv_rmse_log": input_check,
        "dropped_input_groups": dropped,
        "cv_rmse_log_selected_inputs": current,
        "test": test_scores,
        "top_features": {name: round(float(v), 4) for name, v in top},
        "unseen_lake": unseen_lake,
        "learning_curve_test_mae": learning,
    }
    return make().fit(Xs, values), results


def print_summary(results: dict, lab_report: dict, label: str, unit: str) -> None:
    test = results["test"]
    print(f"Lab samples: {lab_report['samples_in']} in, {lab_report['samples_out'] - lab_report['no_sensor_coverage_dropped']} usable ({lab_report})")
    print(f"Test: the most recent {test['samples']} samples, from {pd.Timestamp(test['from']).date()}\n")
    print(f"Model choice, cross-validated on earlier samples (RMSE of log {label}; 0.1 is about 10% error):")
    for name, score in results["cv_rmse_log"].items():
        print(f"  {name:<14} {score:.3f}{'  <- kept' if name == results['model'] else ''}")
    print("Which inputs matter (same model, cross-validated; higher without a group = that group helps):")
    for name, score in results["input_check_cv_rmse_log"].items():
        print(f"  {name:<26} {score:.3f}")
    print(f"Input groups dropped: {results['dropped_input_groups'] or 'none'} "
          f"(cross-validated error {results['cv_rmse_log_selected_inputs']:.3f})")
    print(f"\nTest accuracy ({unit}):")
    for name in ("soft_sensor", "last_lab_value", "station_median"):
        a = test[name]
        print(f"  {name:<15} MAE {a['mae']:.2f}  RMSE {a['rmse']:.2f}  median error {a['median_abs_pct_error']}%  R2(log) {a['r2_log']}")
    ss = test["soft_sensor"]
    print(f"  80% range contains the lab value {ss['coverage']:.0%} of the time; typical range spans x{ss['median_width_factor']}")
    for level, e in test["exceedance"].items():
        print(f"  {label} {level} {unit}: {e['events']} cases, detected {e['probability_of_detection']}, "
              f"false-alarm ratio {e['false_alarm_ratio']}, Brier skill {e['brier_skill']}")
    print("\nMost useful inputs (permutation importance):", ", ".join(f"{n} {v:.3f}" for n, v in list(results["top_features"].items())[:6]))
    print(f"A lake the model never saw (MAE {unit}):", {k: v["mae"] for k, v in results["unseen_lake"].items()})
    print("Test MAE by number of training samples:", results["learning_curve_test_mae"])
