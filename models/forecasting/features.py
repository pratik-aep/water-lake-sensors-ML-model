"""Forecasting rows: what is known at the forecast origin, and the change to predict at each horizon."""

import numpy as np
import pandas as pd

from ..anomaly_detection.config import SENSORS
from ..anomaly_detection.rules import to_working
from ..wqi.calculator import do_saturation_mgl
from .config import WEATHER

TEMP, DO = SENSORS.index("temperature"), SENSORS.index("dissolved_oxygen")
AIR, CLOUD, RAIN = (WEATHER.index(c) for c in ("air_temperature", "cloud_cover", "rain_mm"))


def forecast_error(kind: str, values: np.ndarray, lead_hours, rng) -> np.ndarray:
    """Degrade recorded weather into what a forecast issued `lead_hours` earlier would have said."""
    lead = np.asarray(lead_hours, dtype=float)
    if kind == "cloud":
        return np.clip(values + rng.normal(0, 0.05 + 0.005 * lead), 0, 1)
    if kind == "rain":
        missed = rng.random(values.shape) < 0.05 + 0.004 * lead
        return np.where(missed, 0.0, values * rng.lognormal(0, 0.3 + 0.01 * lead))
    return values + rng.normal(0, 0.3 + 0.03 * lead)


def take_rows(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """x[idx] with NaN rows wherever idx falls outside x."""
    inside = (idx >= 0) & (idx < len(x))
    out = np.full((len(idx), *x.shape[1:]), np.nan)
    out[inside] = x[idx[inside]]
    return out


class Timeline:
    """One station's hourly readings (working scale) with city weather on the same clock, running into the future."""

    def __init__(self, station: str, hourly: pd.DataFrame, weather: pd.DataFrame | None, future_hours: int):
        self.station = station
        self.start = pd.Timestamp(hourly["timestamp"].iloc[0])
        self.n = len(hourly)
        self.x = np.column_stack([to_working(hourly[s], s).to_numpy(dtype=float) for s in SENSORS])
        self.clock = pd.date_range(self.start, periods=self.n + future_hours, freq="h")
        frame = pd.DataFrame(self.x, index=self.clock[: self.n])
        self.rolling = np.stack([getattr(frame.rolling(24, min_periods=12), f)().to_numpy() for f in ("mean", "min", "max")], axis=1)
        self.w = None
        if weather is not None:
            self.w = weather.set_index("timestamp").reindex(self.clock)[WEATHER].to_numpy(dtype=float)
            self._sum = np.vstack([np.zeros(len(WEATHER)), np.cumsum(np.nan_to_num(self.w), axis=0)])
            self._count = np.vstack([np.zeros(len(WEATHER)), np.cumsum(~np.isnan(self.w), axis=0)])

    def position(self, when) -> int:
        return int((pd.Timestamp(when) - self.start) // pd.Timedelta(hours=1))

    def hour_of(self, pos):
        return (self.start.hour + pos) % 24

    def weather_window(self, col: int, start, stop, how: str = "mean") -> np.ndarray:
        """Mean or sum of a weather column over positions [start, stop); NaN where no weather is known."""
        a = np.clip(start, 0, len(self.clock))
        b = np.clip(stop, 0, len(self.clock))
        total = self._sum[b, col] - self._sum[a, col]
        count = self._count[b, col] - self._count[a, col]
        with np.errstate(invalid="ignore", divide="ignore"):
            value = total / count if how == "mean" else total
        return np.where(count > 0, value, np.nan)

    def nights(self, settings) -> pd.DataFrame:
        """Per date: pre-dawn oxygen minimum, daytime sensor means before the evening issue hour, and daily weather."""
        first, last = settings["night_hours"]
        frame = pd.DataFrame(self.x, index=self.clock[: self.n], columns=SENSORS)
        hour = frame.index.hour
        night = frame.loc[(hour >= first) & (hour < last), "dissolved_oxygen"]
        grouped = night.groupby(night.index.normalize())
        table = pd.DataFrame({"night_min": grouped.min().where(grouped.count() >= settings["night_min_valid_hours"])})
        before_issue = frame[hour < settings["night_issue_hour"]]
        day_means = before_issue.groupby(before_issue.index.normalize()).mean()
        table = table.join(day_means.add_prefix("day_mean_"), how="outer")
        if self.w is not None:
            w = pd.DataFrame(self.w, index=self.clock, columns=WEATHER)
            daytime = w[(w.index.hour >= 6) & (w.index.hour < 18)]
            daily = pd.DataFrame(
                {
                    "cloud_day": daytime["cloud_cover"].groupby(daytime.index.normalize()).mean(),
                    "rain_day": w["rain_mm"].groupby(w.index.normalize()).sum(min_count=1),
                    "air_day": w["air_temperature"].groupby(w.index.normalize()).mean(),
                }
            )
            daily["rain_3day"] = daily["rain_day"].rolling(3, min_periods=1).sum()
            table = table.join(daily, how="outer")
        return table.sort_index()


def hourly_rows(tl: Timeline, origins: np.ndarray, horizons, settings: dict, station_code: float, rng=None):
    """Features, targets (change from the origin, working scale) and row info for every origin x horizon."""
    horizons = np.asarray(horizons, dtype=int)
    i = np.repeat(np.asarray(origins, dtype=int), len(horizons))
    h = np.tile(horizons, len(origins))
    j = i + h
    now = take_rows(tl.x, i)
    feats = {"horizon": h.astype(float), "station": np.full(len(i), station_code, dtype=float)}
    for name, pos in (("issue", i), ("target", j)):
        angle = 2 * np.pi * tl.hour_of(pos) / 24
        feats[f"{name}_hour_sin"], feats[f"{name}_hour_cos"] = np.sin(angle), np.cos(angle)
    for k, s in enumerate(SENSORS):
        feats[f"now_{s}"] = now[:, k]
    feats["do_deficit"] = do_saturation_mgl(now[:, TEMP], settings["pressure_atm"]) - now[:, DO]
    # Everything below is relative to the current reading, so trees never need to extrapolate a level.
    for lag in settings["lags_hours"]:
        lagged = take_rows(tl.x, i - lag)
        for k, s in enumerate(SENSORS):
            feats[f"lag{lag}_{s}"] = lagged[:, k] - now[:, k]
    rolling = take_rows(tl.rolling, i)
    for m, stat in enumerate(("mean", "min", "max")):
        for k, s in enumerate(SENSORS):
            feats[f"day_{stat}_{s}"] = rolling[:, m, k] - now[:, k]
    same_hour = take_rows(tl.x, j - 24 * np.ceil(h / 24).astype(int))
    for k, s in enumerate(SENSORS):
        feats[f"same_hour_last_day_{s}"] = same_hour[:, k] - now[:, k]

    if tl.w is not None:
        feats["rain_past_24h"] = tl.weather_window(RAIN, i - 23, i + 1, "sum")
        feats["cloud_past_12h"] = tl.weather_window(CLOUD, i - 11, i + 1)
        ahead = {
            "cloud_before_target_12h": ("cloud", tl.weather_window(CLOUD, j - 11, j + 1)),
            "rain_before_target_24h": ("rain", tl.weather_window(RAIN, j - 23, j + 1, "sum")),
            "air_at_target": ("air", take_rows(tl.w, j)[:, AIR]),
            "air_mean_ahead": ("air", tl.weather_window(AIR, i + 1, j + 1)),
        }
        for name, (kind, values) in ahead.items():
            feats[name] = values if rng is None else forecast_error(kind, values, h, rng)

    targets = pd.DataFrame({s: take_rows(tl.x, np.where(j < tl.n, j, -1))[:, k] - now[:, k] for k, s in enumerate(SENSORS)})
    return pd.DataFrame(feats), targets, row_info(tl, i, h)


def row_info(tl: Timeline, i: np.ndarray, h: np.ndarray) -> pd.DataFrame:
    j = i + h
    return pd.DataFrame(
        {
            "station_id": tl.station,
            "origin": tl.clock[i],
            "horizon": h,
            "target_time": tl.clock[np.minimum(j, len(tl.clock) - 1)],
            "origin_pos": i,
            "target_pos": j,
        }
    )


def nightly_rows(tl: Timeline, issue_dates: pd.DatetimeIndex, settings: dict, station_code: float, rng=None):
    """Features, target (change in pre-dawn oxygen minimum from last night) and row info for each issue date x night."""
    nights = tl.nights(settings)
    days = np.arange(1, settings["night_days"] + 1)
    issue = pd.DatetimeIndex(np.repeat(issue_dates.values, len(days)))
    ahead = np.tile(days, len(issue_dates))
    target_date = issue + pd.to_timedelta(ahead, unit="D")

    def at(dates, col):
        return nights[col].reindex(dates).to_numpy(dtype=float)

    last = at(issue, "night_min")
    week = nights["night_min"].rolling(7, min_periods=4).mean()
    positions = np.array([tl.position(t) for t in issue + pd.Timedelta(hours=settings["night_issue_hour"])])
    now = take_rows(tl.x, np.where(positions < tl.n, positions, -1))
    feats = {
        "days_ahead": ahead.astype(float),
        "station": np.full(len(issue), station_code, dtype=float),
        "last_night_min": last,
        "prev1_night_min": at(issue - pd.Timedelta(days=1), "night_min") - last,
        "prev2_night_min": at(issue - pd.Timedelta(days=2), "night_min") - last,
        "week_mean_night_min": week.reindex(issue).to_numpy(dtype=float) - last,
        "do_deficit": do_saturation_mgl(now[:, TEMP], settings["pressure_atm"]) - now[:, DO],
    }
    for k, s in enumerate(SENSORS):
        feats[f"now_{s}"] = now[:, k]
        feats[f"day_mean_{s}"] = at(issue, f"day_mean_{s}")
    # Warming raises respiration, so the recent temperature trend helps predict where minima are heading.
    feats["temperature_trend_3d"] = feats["day_mean_temperature"] - at(issue - pd.Timedelta(days=3), "day_mean_temperature")
    if tl.w is not None:
        # Today's daytime cloud is already observed at the evening issue time; comparing it with the forecast
        # for the day before each target night tells the model whether photosynthesis will rise or fall.
        feats["cloud_day_today"] = at(issue, "cloud_day")
        day_before = target_date - pd.Timedelta(days=1)
        lead = (ahead - 1) * 24
        for name, kind in (("cloud_day", "cloud"), ("rain_day", "rain"), ("rain_3day", "rain"), ("air_day", "air")):
            values = at(day_before, name)
            feats[f"{name}_before_night"] = values if rng is None else forecast_error(kind, values, lead, rng)

    target = at(target_date, "night_min") - last
    info = pd.DataFrame({"station_id": tl.station, "issue_date": issue, "days_ahead": ahead, "night_of": target_date})
    return pd.DataFrame(feats), pd.Series(target, name="night_min_change"), info
