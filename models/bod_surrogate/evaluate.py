"""BOD soft-sensor scoring: accuracy, range quality, and how well limit exceedances are called."""

import numpy as np

from ..forecasting.evaluate import alert_scores


def accuracy(actual, estimate) -> dict:
    actual, estimate = np.asarray(actual, dtype=float), np.asarray(estimate, dtype=float)
    log_error = np.log(estimate) - np.log(actual)
    log_actual = np.log(actual)
    return {
        "mae": round(float(np.mean(np.abs(estimate - actual))), 3),
        "rmse": round(float(np.sqrt(np.mean((estimate - actual) ** 2))), 3),
        "median_abs_pct_error": round(float(np.median(np.abs(estimate / actual - 1)) * 100), 1),
        "rmse_log": round(float(np.sqrt(np.mean(log_error**2))), 3),
        "r2_log": round(float(1 - np.sum(log_error**2) / np.sum((log_actual - log_actual.mean()) ** 2)), 3),
        "median_ratio": round(float(np.median(estimate / actual)), 3),  # 1 = unbiased
    }


def range_quality(actual, lo, hi) -> dict:
    actual = np.asarray(actual, dtype=float)
    return {
        "coverage": round(float(np.mean((actual >= lo) & (actual <= hi))), 3),
        "median_width_factor": round(float(np.median(np.asarray(hi) / np.asarray(lo))), 2),  # hi / lo
    }


def exceedance(actual, probability, level: float, alert_probability: float) -> dict:
    probability = np.asarray(probability, dtype=float)
    return alert_scores(probability, probability >= alert_probability, np.asarray(actual) > level)
