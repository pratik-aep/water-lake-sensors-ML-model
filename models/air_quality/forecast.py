"""Particulate forecasts: hourly PM2.5/PM10 for 48 h, and tomorrow's daily means as an AQI category with odds."""

import numpy as np
import pandas as pd

from ..forecasting.calibration import fit_offsets, horizon_bucket, prob_below, widen
from ..forecasting.features import forecast_error, take_rows
from ..forecasting.models import QuantileGBM
from .config import BREAKPOINTS
from .naqi import category, sub_index

AHEAD = ["wind_speed", "boundary_layer_height", "relative_humidity", "air_temperature", "rain_mm"]


def _noisy(name: str, values, lead_hours, rng):
    """What a weather forecast issued `lead_hours` earlier would have said (training and testing only)."""
    if rng is None:
        return values
    lead = np.asarray(lead_hours, dtype=float)
    if name in ("wind_speed", "boundary_layer_height"):
        return values * rng.lognormal(0, 0.15 + 0.01 * lead)
    if name == "relative_humidity":
        return np.clip(values + rng.normal(0, 3 + 0.1 * lead), 0, 100)
    return forecast_error("rain" if name == "rain_mm" else "air", values, lead, rng)


class AirTimeline:
    """One station's hourly log particle levels with weather on the same clock, running into the future."""

    def __init__(self, station: str, hourly: pd.DataFrame, weather: pd.DataFrame | None, targets, future_hours: int):
        g = hourly.set_index("timestamp").sort_index().asfreq("h")
        self.station, self.start, self.n = station, g.index[0], len(g)
        self.clock = pd.date_range(self.start, periods=self.n + future_hours, freq="h")
        self.y = np.log1p(g[targets].clip(lower=0).to_numpy(dtype=float))
        # Means of the concentrations themselves (then logged), like the daily means the index is built from.
        level = pd.DataFrame(np.expm1(self.y), index=g.index)
        self.day_mean = np.log1p(level.rolling(24, min_periods=16).mean().to_numpy())
        self.week_mean = np.log1p(level.rolling(24 * 7, min_periods=24 * 3).mean().to_numpy())
        self.weather_columns = [c for c in AHEAD if weather is not None and c in weather]
        self.w = None
        if self.weather_columns:
            w = weather.assign(timestamp=pd.to_datetime(weather["timestamp"])).drop_duplicates("timestamp")
            self.w = w.set_index("timestamp").reindex(self.clock)[self.weather_columns].to_numpy(dtype=float)

    def position(self, when) -> int:
        return int((pd.Timestamp(when) - self.start) // pd.Timedelta(hours=1))


def _weather_block(tl, positions, lead, rng, prefix) -> dict:
    feats = {}
    if tl.w is None:
        return feats
    for k, name in enumerate(tl.weather_columns):
        if name == "rain_mm":  # rain over the six hours up to then: it washes particles out
            values = np.nansum(np.stack([take_rows(tl.w, positions - d)[:, k] for d in range(6)]), axis=0)
        else:
            values = take_rows(tl.w, positions)[:, k]
        values = _noisy(name, values, lead, rng)
        feats[f"{prefix}{name}"] = np.log(np.clip(values, 10, None)) if name == "boundary_layer_height" else values
    return feats


def hourly_rows(tl: AirTimeline, origins, horizons, settings, code: float, rng=None):
    horizons = np.asarray(horizons, dtype=int)
    i = np.repeat(np.asarray(origins, dtype=int), len(horizons))
    h = np.tile(horizons, len(origins))
    j = i + h
    targets = settings["targets"]
    now = take_rows(tl.y, i)
    feats = {"horizon": h.astype(float), "station": np.full(len(i), code)}
    for name, pos in (("issue", i), ("target", j)):
        angle = 2 * np.pi * ((tl.start.hour + pos) % 24) / 24
        feats[f"{name}_hour_sin"], feats[f"{name}_hour_cos"] = np.sin(angle), np.cos(angle)
    feats["target_weekend"] = (tl.clock[np.minimum(j, len(tl.clock) - 1)].dayofweek >= 5).astype(float)
    for k, t in enumerate(targets):
        feats[f"now_{t}"] = now[:, k]
        for lag in settings["lags_hours"]:
            feats[f"lag{lag}_{t}"] = take_rows(tl.y, i - lag)[:, k] - now[:, k]
        feats[f"day_mean_{t}"] = take_rows(tl.day_mean, i)[:, k] - now[:, k]
        feats[f"week_mean_{t}"] = take_rows(tl.week_mean, i)[:, k] - now[:, k]
        feats[f"same_hour_{t}"] = take_rows(tl.y, j - 24 * np.ceil(h / 24).astype(int))[:, k] - now[:, k]
    feats.update(_weather_block(tl, i, np.zeros(len(i)), None, "now_"))
    feats.update(_weather_block(tl, j, h, rng, "target_"))
    actual = take_rows(tl.y, np.where(j < tl.n, j, -1))
    Y = pd.DataFrame({t: actual[:, k] - now[:, k] for k, t in enumerate(targets)})
    info = pd.DataFrame({"station_id": tl.station, "origin": tl.clock[i], "horizon": h,
                         "target_time": tl.clock[np.minimum(j, len(tl.clock) - 1)]})
    return pd.DataFrame(feats), Y, info, now, actual


class HourlyForecaster:
    """PM2.5 and PM10 for each of the next 48 hours, as an 80% range around a median (log scale inside)."""

    def __init__(self, settings: dict, seed: int = 42):
        self.settings, self.seed = settings, seed

    def origins(self, timelines, start, end, every) -> dict:
        horizon = max(self.settings["horizons"])
        return {st: np.arange(max(24 * 7, tl.position(max(pd.Timestamp(start), tl.start))), min(tl.n, tl.position(end)) - horizon, every)
                for st, tl in timelines.items()}

    def _rows(self, timelines, origins, horizons, rng):
        parts = [hourly_rows(timelines[st], o, horizons, self.settings, self.codes_.get(st, np.nan), rng)
                 for st, o in origins.items() if len(o)]
        return [pd.concat([p[k] for p in parts], ignore_index=True) if k < 3 else np.vstack([p[k] for p in parts]) for k in range(5)]

    def fit(self, timelines, codes, until, rng=None):
        self.codes_ = codes
        s = self.settings
        X, Y, info, _, _ = self._rows(timelines, self.origins(timelines, pd.Timestamp.min, until, s["train_origin_every_hours"]),
                                      s["train_horizons"], rng)
        keep = (info["target_time"] < pd.Timestamp(until)).to_numpy()
        self.gbm_ = QuantileGBM(s["quantiles"], s["gbm"], self.seed).fit(X[keep].reset_index(drop=True), Y[keep].reset_index(drop=True))
        self.offsets_ = {}
        return self

    def _working(self, timelines, origins, rng):
        X, _, info, now, actual = self._rows(timelines, origins, self.settings["horizons"], rng)
        return info, self.gbm_.predict(X), now, actual

    def calibrate(self, timelines, start, end, rng=None):
        info, deltas, now, actual = self._working(timelines, self.origins(timelines, start, end, self.settings["eval_origin_every_hours"]), rng)
        buckets = horizon_bucket(info["horizon"].to_numpy(), self.settings["horizon_buckets"])
        coverage = self.settings["quantiles"][-1] - self.settings["quantiles"][0]
        self.offsets_ = {}
        for k, t in enumerate(self.settings["targets"]):
            found = fit_offsets(now[:, k] + deltas[t][:, 0], now[:, k] + deltas[t][:, 2], actual[:, k], buckets, coverage)
            self.offsets_.update({f"{t}|{b}": q for b, q in found.items()})
        return self

    def predict(self, timelines, origins, rng=None) -> pd.DataFrame:
        info, deltas, now, actual = self._working(timelines, origins, rng)
        buckets = horizon_bucket(info["horizon"].to_numpy(), self.settings["horizon_buckets"])
        parts = []
        for k, t in enumerate(self.settings["targets"]):
            groups = np.array([f"{t}|{b}" for b in buckets])
            lo, mid, hi = widen(*(now[:, k] + deltas[t][:, q] for q in range(3)), groups, self.offsets_)
            parts.append(info.assign(pollutant=t, lo=np.expm1(lo), mid=np.expm1(mid), hi=np.expm1(hi),
                                     actual=np.expm1(actual[:, k]), persistence=np.expm1(now[:, k])))
        return pd.concat(parts, ignore_index=True)


def daily_rows(tl: AirTimeline, issue_dates, settings, code: float, rng=None):
    """Issued at 18:00 on each date: the mean of each of the coming days (log scale), from today's levels and weather."""
    days = np.asarray(settings["days_ahead"], dtype=int)
    issue = pd.DatetimeIndex(np.repeat(pd.DatetimeIndex(issue_dates).values, len(days)))
    ahead = np.tile(days, len(issue_dates))
    positions = np.array([tl.position(d + pd.Timedelta(hours=18)) for d in issue])
    targets = settings["targets"]
    now = take_rows(tl.day_mean, np.where(positions < tl.n, positions, -1))
    feats = {"days_ahead": ahead.astype(float), "station": np.full(len(issue), code)}
    target_day = issue + pd.to_timedelta(ahead, unit="D")
    feats["target_weekend"] = (target_day.dayofweek >= 5).astype(float)
    actual = np.full((len(issue), len(targets)), np.nan)
    for k, t in enumerate(targets):
        feats[f"last24_{t}"] = now[:, k]
        feats[f"week_mean_{t}"] = take_rows(tl.week_mean, positions)[:, k] - now[:, k]
    if tl.w is not None:
        starts = np.array([tl.position(d) for d in target_day])
        hours = starts[:, None] + np.arange(24)[None, :]
        lead = (ahead - 1) * 24 + 6
        for k, name in enumerate(tl.weather_columns):
            block = np.stack([take_rows(tl.w, hours[:, m])[:, k] for m in range(24)], axis=1)
            with np.errstate(all="ignore"):
                if name == "boundary_layer_height":  # how high the day mixes, and how low the night traps
                    for label, summary in (("day_mixing_max", np.nanmax(block, axis=1)), ("night_mixing_min", np.nanmin(block, axis=1))):
                        feats[label] = np.log(np.clip(_noisy(name, summary, lead, rng), 10, None))
                else:
                    summary = np.nansum(block, axis=1) if name == "rain_mm" else np.nanmean(block, axis=1)
                    feats[f"day_{name}"] = _noisy(name, summary, lead, rng)
    for k, _ in enumerate(targets):
        starts = np.array([tl.position(d) for d in target_day])
        hours = starts[:, None] + np.arange(24)[None, :]
        block = np.stack([take_rows(tl.y, np.where(hours[:, m] < tl.n, hours[:, m], -1))[:, k] for m in range(24)], axis=1)
        enough = (~np.isnan(block)).sum(axis=1) >= 16
        with np.errstate(all="ignore"):
            actual[:, k] = np.where(enough, np.log1p(np.nanmean(np.expm1(block), axis=1)), np.nan)
    Y = pd.DataFrame({t: actual[:, k] - now[:, k] for k, t in enumerate(targets)})
    info = pd.DataFrame({"station_id": tl.station, "issue_date": issue, "days_ahead": ahead, "date": target_day})
    return pd.DataFrame(feats), Y, info, now, actual


class DailyForecaster:
    """Tomorrow's and the next day's mean PM2.5 and PM10, and what they mean for the AQI."""

    def __init__(self, settings: dict, seed: int = 42):
        self.settings, self.seed = settings, seed

    def issue_dates(self, timelines, start, end) -> dict:
        out = {}
        for st, tl in timelines.items():
            first = max(pd.Timestamp(start), tl.start + pd.Timedelta(days=7)).normalize()
            dates = pd.date_range(first, pd.Timestamp(end).normalize(), freq="D")
            issued = dates + pd.Timedelta(hours=18)
            out[st] = dates[(issued >= pd.Timestamp(start)) & (issued < pd.Timestamp(end)) & (issued <= tl.clock[tl.n - 1])]
        return out

    def _rows(self, timelines, issue, rng):
        parts = [daily_rows(timelines[st], d, self.settings, self.codes_.get(st, np.nan), rng) for st, d in issue.items() if len(d)]
        return [pd.concat([p[k] for p in parts], ignore_index=True) if k < 3 else np.vstack([p[k] for p in parts]) for k in range(5)]

    def fit(self, timelines, codes, until, rng=None):
        self.codes_ = codes
        X, Y, info, _, _ = self._rows(timelines, self.issue_dates(timelines, pd.Timestamp.min, until), rng)
        keep = ((info["date"] + pd.Timedelta(days=1)) <= pd.Timestamp(until)).to_numpy()
        self.gbm_ = QuantileGBM(self.settings["quantiles"], self.settings["gbm"], self.seed).fit(
            X[keep].reset_index(drop=True), Y[keep].reset_index(drop=True))
        self.offsets_ = {}
        return self

    def _hourly_means(self, timelines, info, hourly, rng):
        """For days the hourly horizon fully covers, the mean of the hourly forecast path (log scale, per quantile)."""
        horizon, issue_hour = max(hourly.settings["horizons"]), 18
        covered = (info["days_ahead"] * 24 + 23 - issue_hour <= horizon).to_numpy()
        issued_at = info["issue_date"] + pd.Timedelta(hours=issue_hour)
        origins = {st: np.unique([timelines[st].position(x) for x in issued_at[covered & (info["station_id"] == st).to_numpy()]])
                   for st in info.loc[covered, "station_id"].unique()}
        path = {t: np.full((len(info), 3), np.nan) for t in self.settings["targets"]}
        if not origins:
            return covered, path
        h_info, deltas, now, _ = hourly._working(timelines, origins, rng)
        day = (h_info["target_time"].dt.normalize() - h_info["origin"].dt.normalize()).dt.days
        keys = pd.MultiIndex.from_arrays([info["station_id"], issued_at, info["days_ahead"]])
        for k, t in enumerate(self.settings["targets"]):
            levels = pd.DataFrame(np.expm1(now[:, [k]] + deltas[t]), columns=["lo", "mid", "hi"]).assign(
                station_id=h_info["station_id"], origin=h_info["origin"], day=day)
            means = np.log1p(levels.groupby(["station_id", "origin", "day"])[["lo", "mid", "hi"]].mean())
            path[t] = means.reindex(keys).to_numpy()
        return covered & ~np.isnan(path[self.settings["targets"][0]]).any(axis=1), path

    def _working(self, timelines, issue, rng, hourly=None):
        X, _, info, now, actual = self._rows(timelines, issue, rng)
        deltas = self.gbm_.predict(X)
        quantiles = {t: now[:, [k]] + deltas[t] for k, t in enumerate(self.settings["targets"])}
        if hourly is not None:
            # Within the hourly horizon (tomorrow, from an 18:00 issue) the hour-by-hour path is the sharper forecast.
            use, path = self._hourly_means(timelines, info, hourly, rng)
            quantiles = {t: np.where(use[:, None], path[t], q) for t, q in quantiles.items()}
        return info, quantiles, now, actual

    def calibrate(self, timelines, start, end, rng=None, hourly=None):
        end = pd.Timestamp(end) - pd.Timedelta(days=max(self.settings["days_ahead"]) + 1)
        info, quantiles, _, actual = self._working(timelines, self.issue_dates(timelines, start, end), rng, hourly)
        coverage = self.settings["quantiles"][-1] - self.settings["quantiles"][0]
        groups = info["days_ahead"].astype(str).to_numpy()
        self.offsets_ = {}
        for k, t in enumerate(self.settings["targets"]):
            found = fit_offsets(quantiles[t][:, 0], quantiles[t][:, 2], actual[:, k], groups, coverage)
            self.offsets_.update({f"{t}|{g}": q for g, q in found.items()})
        return self

    def predict(self, timelines, issue, rng=None, hourly=None) -> pd.DataFrame:
        """One row per station, issue date and day ahead: each target's range, the PM index and P(index > alert).
        Pass the fitted HourlyForecaster as `hourly` to use its path for the day it covers."""
        s = self.settings
        info, quantiles, now, actual = self._working(timelines, issue, rng, hourly)
        out = info.copy()
        chances, indices, actual_index = [], [], []
        for k, t in enumerate(s["targets"]):
            groups = np.array([f"{t}|{d}" for d in info["days_ahead"]])
            lo, mid, hi = widen(*(quantiles[t][:, q] for q in range(3)), groups, self.offsets_)
            out[f"{t}_lo"], out[t], out[f"{t}_hi"] = np.expm1(lo), np.expm1(mid), np.expm1(hi)
            out[f"{t}_actual"] = np.expm1(actual[:, k])
            out[f"{t}_last24h"] = np.expm1(now[:, k])  # the no-change forecast: tomorrow like the past day
            # The concentration at which this pollutant alone would push the index past the alert level.
            limit = np.interp(s["alert_index"], [0, 50, 100, 200, 300, 400, 500], BREAKPOINTS[t])
            chances.append(1 - prob_below(np.log1p(limit), lo, mid, hi, s["quantiles"]))
            indices.append(sub_index(out[t], t))
            actual_index.append(sub_index(out[f"{t}_actual"], t))
        out["pm_index"] = np.nanmax(np.column_stack(indices), axis=1).round(0)
        out["pm_category"] = category(out["pm_index"])
        # Either pollutant can do it; they rise together, so the larger chance is the honest lower bound.
        out["p_poor_or_worse"] = np.max(np.column_stack(chances), axis=1).round(3)
        with np.errstate(all="ignore"):
            out["actual_pm_index"] = np.nanmax(np.column_stack(actual_index), axis=1)
        return out
