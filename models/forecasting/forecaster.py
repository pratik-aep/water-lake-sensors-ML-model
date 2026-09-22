"""Hourly and nightly forecasters: a model plus calibrated ranges, reported in each sensor's own units."""

import numpy as np
import pandas as pd

from ..anomaly_detection.config import SENSORS
from ..anomaly_detection.rules import from_working
from .calibration import fit_offsets, horizon_bucket, prob_below, widen
from .config import DEFAULT_SETTINGS
from .features import Timeline, nightly_rows, take_rows
from .models import EnsembleHourlyModel, GBMHourlyModel, QuantileGBM


def make_timelines(hourly: pd.DataFrame, weather: pd.DataFrame | None, settings: dict) -> dict:
    future_hours = max(max(settings["horizons"]), (settings["night_days"] + 1) * 24)
    return {
        station: Timeline(station, g.reset_index(drop=True), weather, future_hours)
        for station, g in hourly.groupby("station_id", sort=True)
    }


def station_codes(stations) -> dict:
    return {station: float(i) for i, station in enumerate(sorted(stations))}


def _uses_weather(timelines: dict) -> bool:
    return next(iter(timelines.values())).w is not None


class HourlyForecaster:
    """Every sensor, every hour for the next 48 hours, as an 80% range around a median."""

    def __init__(self, settings: dict | None = None, kind: str = "gradient_boosting", seed: int = 42):
        self.settings = {**DEFAULT_SETTINGS, **(settings or {})}
        self.kind = kind
        self.seed = seed

    def _rng(self, timelines, stream: int):
        """Weather-forecast error, simulated from recorded weather (training and evaluation only)."""
        if _uses_weather(timelines) and self.settings["weather_forecast_noise"]:
            return np.random.default_rng(self.seed + stream)
        return None

    def fit(self, timelines: dict, codes: dict, until, validation=None) -> "HourlyForecaster":
        self.codes_ = codes
        self.uses_weather_ = _uses_weather(timelines)
        if self.kind == "cnn_lstm":
            from .challenger import CNNLSTMHourlyModel

            self.model_ = CNNLSTMHourlyModel(self.settings, self.seed)
        else:
            self.model_ = GBMHourlyModel(self.settings, self.seed)
        self.model_.fit(timelines, codes, pd.Timestamp(until), self._rng(timelines, 1), validation)
        self.offsets_ = {}
        return self

    @classmethod
    def combine(cls, members: list) -> "HourlyForecaster":
        """An ensemble of already-fitted hourly forecasters."""
        ensemble = cls(members[0].settings, "ensemble", members[0].seed)
        ensemble.codes_ = members[0].codes_
        ensemble.uses_weather_ = members[0].uses_weather_
        ensemble.model_ = EnsembleHourlyModel([m.model_ for m in members])
        ensemble.offsets_ = {}
        return ensemble

    def origins(self, timelines: dict, start, end) -> dict:
        """Forecast origins every few hours in [start, end), leaving room for the full horizon to be observed."""
        every, horizon = self.settings["eval_origin_every_hours"], max(self.settings["horizons"])
        return {
            station: np.arange(max(24, tl.position(start)), min(tl.n, tl.position(end)) - horizon, every)
            for station, tl in timelines.items()
        }

    def working_forecast(self, timelines, origins, simulated_forecast):
        """Uncalibrated quantile deltas plus the readings needed to turn them into values, all on the working scale."""
        info, deltas = self.model_.predict_deltas(
            timelines, self.codes_, origins, self._rng(timelines, 2) if simulated_forecast else None
        )
        now, actual, same_hour = (np.full((len(info), len(SENSORS)), np.nan) for _ in range(3))
        h = info["horizon"].to_numpy()
        for station, idx in info.groupby("station_id").indices.items():
            tl = timelines[station]
            i, j = info["origin_pos"].to_numpy()[idx], info["target_pos"].to_numpy()[idx]
            now[idx] = take_rows(tl.x, i)
            actual[idx] = take_rows(tl.x, np.where(j < tl.n, j, -1))
            same_hour[idx] = take_rows(tl.x, j - 24 * np.ceil(h[idx] / 24).astype(int))
        return info, deltas, now, actual, same_hour

    def calibrate(self, timelines: dict, start, end) -> "HourlyForecaster":
        info, deltas, now, actual, _ = self.working_forecast(timelines, self.origins(timelines, start, end), True)
        buckets = horizon_bucket(info["horizon"].to_numpy(), self.settings["horizon_buckets"])
        coverage = self.settings["quantiles"][-1] - self.settings["quantiles"][0]
        for k, s in enumerate(SENSORS):
            found = fit_offsets(now[:, k] + deltas[s][:, 0], now[:, k] + deltas[s][:, 2], actual[:, k], buckets, coverage)
            self.offsets_.update({f"{s}|{b}": q for b, q in found.items()})
        return self

    def predict(self, timelines: dict, origins: dict, simulated_forecast: bool = False) -> pd.DataFrame:
        """Long table: one row per origin, horizon and sensor, in the sensor's own units."""
        info, deltas, now, actual, same_hour = self.working_forecast(timelines, origins, simulated_forecast)
        buckets = horizon_bucket(info["horizon"].to_numpy(), self.settings["horizon_buckets"])
        parts = []
        for k, s in enumerate(SENSORS):
            groups = np.array([f"{s}|{b}" for b in buckets])
            lo, mid, hi = widen(*(now[:, k] + deltas[s][:, q] for q in range(3)), groups, self.offsets_)
            values = {"lo": lo, "mid": mid, "hi": hi, "actual": actual[:, k], "persistence": now[:, k], "same_hour_last_day": same_hour[:, k]}
            parts.append(info.assign(sensor=s, **{name: from_working(v, s) for name, v in values.items()}))
        return pd.concat(parts, ignore_index=True)


class NightlyForecaster:
    """Pre-dawn oxygen minimum for each of the next nights, with the chance it falls below the alert level."""

    def __init__(self, settings: dict | None = None, seed: int = 42):
        self.settings = {**DEFAULT_SETTINGS, **(settings or {})}
        self.seed = seed

    def _rng(self, timelines, stream: int):
        if _uses_weather(timelines) and self.settings["weather_forecast_noise"]:
            return np.random.default_rng(self.seed + stream)
        return None

    def issue_dates(self, timelines: dict, start, end) -> dict:
        """Evening issue dates in [start, end) with a week of history behind them."""
        hour = pd.Timedelta(hours=self.settings["night_issue_hour"])
        out = {}
        for station, tl in timelines.items():
            first = max(pd.Timestamp(start), tl.start + pd.Timedelta(days=7)).normalize()
            dates = pd.date_range(first, pd.Timestamp(end).normalize(), freq="D")
            out[station] = dates[(dates + hour >= pd.Timestamp(start)) & (dates + hour < pd.Timestamp(end)) & (dates + hour <= tl.clock[tl.n - 1])]
        return out

    def _rows(self, timelines, issue, rng):
        parts = [nightly_rows(timelines[st], dates, self.settings, self.codes_.get(st, np.nan), rng) for st, dates in issue.items() if len(dates)]
        X, y, info = (pd.concat(p, ignore_index=True) for p in zip(*parts))
        return X, y, info

    def fit(self, timelines: dict, codes: dict, until) -> "NightlyForecaster":
        self.codes_ = codes
        until = pd.Timestamp(until)
        night_end = pd.Timedelta(hours=self.settings["night_hours"][1])
        X, y, info = self._rows(timelines, self.issue_dates(timelines, pd.Timestamp.min, until), self._rng(timelines, 3))
        keep = (info["night_of"] + night_end <= until).to_numpy() & y.notna().to_numpy()
        self.gbm_ = QuantileGBM(self.settings["quantiles"], self.settings["gbm"], self.seed).fit(
            X[keep].reset_index(drop=True), y[keep].to_frame().reset_index(drop=True)
        )
        self.offsets_ = {}
        return self

    def _hourly_minima(self, timelines, info, hourly, simulated_forecast):
        """For nights the hourly horizon fully covers, the minimum of the hourly forecast path over the night."""
        s = self.settings
        first, last = s["night_hours"]
        issued_at = info["issue_date"] + pd.Timedelta(hours=s["night_issue_hour"])
        covered = info["night_of"] + pd.Timedelta(hours=last - 1) <= issued_at + pd.Timedelta(hours=max(s["horizons"]))
        origins = {
            station: np.unique([timelines[station].position(t) for t in issued_at[covered & (info["station_id"] == station)]])
            for station in info.loc[covered, "station_id"].unique()
        }
        if not origins:
            return np.zeros(len(info), dtype=bool), np.full((len(info), 3), np.nan)
        path_info, deltas, now, _, _ = hourly.working_forecast(timelines, origins, simulated_forecast)
        k = SENSORS.index("dissolved_oxygen")
        path = path_info[["station_id", "origin", "target_time"]].assign(
            **{q: now[:, k] + deltas["dissolved_oxygen"][:, i] for i, q in enumerate(("lo", "mid", "hi"))}
        )
        hour = path["target_time"].dt.hour
        path = path[(hour >= first) & (hour < last)]
        minima = path.groupby(["station_id", "origin", path["target_time"].dt.normalize()])[["lo", "mid", "hi"]].min()
        aligned = minima.reindex(pd.MultiIndex.from_arrays([info["station_id"], issued_at, info["night_of"]])).to_numpy()
        return covered.to_numpy() & ~np.isnan(aligned).any(axis=1), aligned

    def _working(self, timelines, issue, simulated_forecast, hourly=None):
        X, y, info = self._rows(timelines, issue, self._rng(timelines, 4) if simulated_forecast else None)
        delta = self.gbm_.predict(X)["night_min_change"]
        last = X["last_night_min"].to_numpy()
        lo, mid, hi = (last + delta[:, q] for q in range(3))
        if hourly is not None:
            # Within the hourly horizon, the hour-by-hour path is the sharper forecast of the night's low.
            use, path_min = self._hourly_minima(timelines, info, hourly, simulated_forecast)
            lo, mid, hi = (np.where(use, path_min[:, q], v) for q, v in enumerate((lo, mid, hi)))
        return info, lo, mid, hi, last + y.to_numpy(), last, last + X["week_mean_night_min"].to_numpy()

    def calibrate(self, timelines: dict, start, end, hourly=None) -> "NightlyForecaster":
        end = pd.Timestamp(end) - pd.Timedelta(days=self.settings["night_days"] + 1)
        info, lo, _, hi, actual, _, _ = self._working(timelines, self.issue_dates(timelines, start, end), True, hourly)
        coverage = self.settings["quantiles"][-1] - self.settings["quantiles"][0]
        found = fit_offsets(lo, hi, actual, info["days_ahead"].astype(str).to_numpy(), coverage)
        self.offsets_.update({str(k): v for k, v in found.items()})
        return self

    def predict(self, timelines: dict, issue: dict, simulated_forecast: bool = False, hourly=None) -> pd.DataFrame:
        """Pass the fitted HourlyForecaster as `hourly` to use its path for the nights it covers."""
        info, lo, mid, hi, actual, last, week = self._working(timelines, issue, simulated_forecast, hourly)
        lo, mid, hi = widen(lo, mid, hi, info["days_ahead"].astype(str).to_numpy(), self.offsets_)
        alert_level = self.settings["do_alert_mg_l"]
        return info.assign(
            lo=np.clip(lo, 0, None),
            mid=np.clip(mid, 0, None),
            hi=np.clip(hi, 0, None),
            p_below_alert=prob_below(alert_level, lo, mid, hi, self.settings["quantiles"]),
            actual=actual,
            last_night=last,
            week_mean=week,
        )
