"""Forecast scoring: accuracy against baselines, range coverage, and alert hits, misses and false alarms."""

import numpy as np
import pandas as pd

from ..anomaly_detection.config import SENSORS
from .calibration import horizon_bucket

LEAD_GROUPS = [(1, 2), (3, 4), (5, 7)]


def _mae(pred, actual) -> float:
    return float(np.mean(np.abs(pred - actual)))


def hourly_report(forecasts: pd.DataFrame, buckets) -> dict:
    """Per sensor and horizon bucket: MAE of the median vs both baselines, skill, and 80%-range coverage and width."""
    df = forecasts[forecasts["actual"].notna()].assign(bucket=lambda d: horizon_bucket(d["horizon"].to_numpy(), buckets))
    report = {}
    for (sensor, bucket), g in df.groupby(["sensor", "bucket"], sort=False):
        model = _mae(g["mid"], g["actual"])
        persistence = _mae(g["persistence"], g["actual"])
        same_hour = _mae(g["same_hour_last_day"].fillna(g["persistence"]), g["actual"])
        report.setdefault(sensor, {})[bucket] = {
            "mae": round(model, 4),
            "mae_persistence": round(persistence, 4),
            "mae_same_hour_yesterday": round(same_hour, 4),
            # Skill against the better of the two simple forecasts: 0 = no better, 1 = perfect.
            "skill": round(1 - model / min(persistence, same_hour), 3),
            "coverage_80": round(float(g["actual"].between(g["lo"], g["hi"]).mean()), 3),
            "mean_width": round(float((g["hi"] - g["lo"]).mean()), 4),
        }
    return report


def relative_error(forecasts: pd.DataFrame) -> float:
    """Mean over sensors of model MAE / persistence MAE; used to pick between models, lower is better."""
    df = forecasts[forecasts["actual"].notna()]
    ratios = [
        _mae(g["mid"], g["actual"]) / _mae(g["persistence"], g["actual"]) for _, g in df.groupby("sensor") if len(g)
    ]
    return float(np.mean(ratios))


def _ratio(a: int, b: int):
    return round(a / b, 3) if b else None


def alert_scores(probability, alert, event) -> dict:
    hits = int((alert & event).sum())
    misses = int((~alert & event).sum())
    false_alarms = int((alert & ~event).sum())
    brier = float(np.mean((probability - event) ** 2)) if len(event) else None
    base_rate = event.mean() if len(event) else 0
    reference = base_rate * (1 - base_rate)
    return {
        "events": int(event.sum()),
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "probability_of_detection": _ratio(hits, hits + misses),
        "false_alarm_ratio": _ratio(false_alarms, hits + false_alarms),
        "critical_success_index": _ratio(hits, hits + misses + false_alarms),
        # Brier score judges the probabilities themselves; skill compares it with always forecasting the base rate.
        "brier_score": None if brier is None else round(brier, 4),
        "brier_skill": round(1 - brier / reference, 3) if brier is not None and reference > 0 else None,
    }


def nightly_report(forecasts: pd.DataFrame, threshold: float, alert_probability: float) -> dict:
    df = forecasts[forecasts["actual"].notna()]
    by_day = {}
    for days, g in df.groupby("days_ahead"):
        model = _mae(g["mid"], g["actual"])
        last = _mae(g["last_night"], g["actual"])
        week = _mae(g["week_mean"].fillna(g["last_night"]), g["actual"])
        by_day[int(days)] = {
            "mae": round(model, 3),
            "mae_last_night": round(last, 3),
            "mae_week_mean": round(week, 3),
            "skill": round(1 - model / min(last, week), 3),
            "coverage_80": round(float(g["actual"].between(g["lo"], g["hi"]).mean()), 3),
        }
    probability = df["p_below_alert"].to_numpy()
    alert = probability >= alert_probability
    event = (df["actual"] < threshold).to_numpy()
    days = df["days_ahead"].to_numpy()
    alerts = {"all": alert_scores(probability, alert, event)}
    for lo, hi in LEAD_GROUPS:
        m = (days >= lo) & (days <= hi)
        alerts[f"{lo}-{hi} days"] = alert_scores(probability[m], alert[m], event[m])
    return {"by_days_ahead": by_day, "alerts": alerts}


def summary_table(report: dict, sensors=SENSORS) -> pd.DataFrame:
    rows = [
        {"sensor": s, "horizon": b, **m} for s in sensors if s in report for b, m in report[s].items()
    ]
    return pd.DataFrame(rows)
