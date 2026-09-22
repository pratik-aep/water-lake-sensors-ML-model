"""Layers 2-3: cross-sensor consistency, drift and multivariate checks, then fault-versus-event attribution."""

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..wqi.config import VALID_RANGES
from .config import DEFAULT_SETTINGS, INTERVAL, SENSORS
from .rules import QC_PASS, SHAPE_FAULTS, rule_tests, to_working

MAD_TO_STD = 1.4826
ALL = "__all__"
PER_DAY = pd.Timedelta(days=1) // pd.Timedelta(INTERVAL)
# Faults inferred from disagreement with other sensors, as opposed to a sensor's own signal shape.
SOFT_FAULTS = ["drift", "inconsistent", "rate_of_change"]


def robust_std(values, floor: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return floor
    return max(MAD_TO_STD * float(np.median(np.abs(values - np.median(values)))), floor)


def ewma(z: np.ndarray, weight: float, reset: np.ndarray, cap) -> np.ndarray:
    """Exponentially weighted mean per column, held within +-cap; `reset` (per row, or per row and column)
    zeroes it and missing values carry it."""
    # Unlike a CUSUM, the average stays bounded when readings are strongly autocorrelated (10-minute residuals
    # are), so its alarm level can be read straight off normal data.
    reset = np.broadcast_to(reset.reshape(len(z), -1), z.shape)
    level = np.zeros(z.shape[1])
    out = np.empty(z.shape)
    for i, row in enumerate(z):
        ok = ~np.isnan(row) & ~reset[i]
        level = np.where(ok, np.clip(level + weight * (np.where(ok, row, 0.0) - level), -cap, cap), level)
        level = np.where(reset[i], 0.0, level)
        out[i] = level
    return out


def sustained(mask: pd.Series, station: pd.Series, min_points: int) -> pd.Series:
    """True from the min_points-th consecutive True row of a run onward, per station (no look-ahead)."""
    run = ((mask != mask.shift()) | (station != station.shift())).cumsum()
    return mask & (mask.groupby(run).cumcount() + 1 >= min_points)


def isolate(own: pd.DataFrame, others: dict) -> pd.Series:
    """Per row, the sensor whose reconstruction brings every other sensor back within limits (1 = limit), else ''."""
    culprit = pd.Series("", index=own.index)
    best = pd.Series(np.inf, index=own.index)
    for f in SENSORS:
        remaining = others[f].max(axis=1).fillna(0.0)
        explains = (own[f] > 1) & (remaining <= 1) & (remaining < best)
        culprit = culprit.mask(explains, f)
        best = best.mask(explains, remaining)
    return culprit


class TrendPlusTrees(RegressorMixin, BaseEstimator):
    """Ridge carries the broad relationships into unseen seasons; boosted trees add curvature inside the known range."""

    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, X, y):
        numeric = [c for c in X.columns if c != "station"]
        self.linear_ = make_pipeline(
            ColumnTransformer(
                [("num", StandardScaler(), numeric), ("station", OneHotEncoder(handle_unknown="ignore"), ["station"])]
            ),
            Ridge(alpha=1.0),
        ).fit(X, y)
        self.trees_ = HistGradientBoostingRegressor(
            max_iter=200, categorical_features=["station"], random_state=self.random_state
        ).fit(X, y - self.linear_.predict(X))
        return self

    def predict(self, X):
        return self.linear_.predict(X) + self.trees_.predict(X)


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)


def _working(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {s: to_working(df[s].where(df[s].between(*VALID_RANGES[s])), s) for s in SENSORS}, index=df.index
    )


def _others(sensor: str) -> list[str]:
    return [s for s in SENSORS if s != sensor]


def weather_memory(weather: pd.DataFrame, timestamps: pd.Series, rain_hours, air_hours=()) -> pd.DataFrame:
    """Weather as the lake feels it at each timestamp's hour: log rain remembered with each decay time (hours; no
    weather row = no rain), and, if the weather has them, air temperature averaged over each time scale (water
    warms and cools days behind the air) and cloud over the last few hours."""
    w = weather.assign(timestamp=pd.to_datetime(weather["timestamp"])).groupby("timestamp").mean(numeric_only=True)
    w = w.asfreq("h")
    hour = timestamps.dt.floor("h")
    out = {}
    rain = w["rain_mm"].astype(float).fillna(0.0).to_numpy()
    for h in rain_hours:
        remembered = pd.Series(lfilter([1.0], [1.0, -np.exp(-1 / h)], rain), index=w.index)
        out[f"rain_{h}h"] = np.log1p(remembered.reindex(hour).fillna(0.0).to_numpy())
    if len(air_hours) and {"air_temperature", "cloud_cover"} <= set(w.columns):
        for h in air_hours:
            out[f"air_{h}h"] = w["air_temperature"].ewm(alpha=1 - np.exp(-1 / h)).mean().reindex(hour).to_numpy()
        out["cloud_3h"] = w["cloud_cover"].ewm(alpha=1 - np.exp(-1 / 3)).mean().reindex(hour).to_numpy()
    return pd.DataFrame(out, index=timestamps.index)


class AnomalyDetector:
    """Expects readings from data.prepare: one row per station per scan slot, gaps as empty rows."""

    def __init__(self, settings: dict | None = None, seed: int = 42):
        self.settings = {**DEFAULT_SETTINGS, **(settings or {})}
        self.seed = seed

    def fit(self, df: pd.DataFrame, weather: pd.DataFrame | None = None) -> "AnomalyDetector":
        """Learn normal behaviour from 10-minute readings, and from hourly weather (timestamp, rain_mm, and ideally
        air_temperature and cloud_cover) if given."""
        df = _sorted(df)
        self.weather_inputs_ = []
        if weather is not None:
            s = self.settings
            self.weather_inputs_ = list(weather_memory(weather, df["timestamp"].head(1), s["rain_memory_hours"], s["air_memory_hours"]).columns)
        self._fit_once(df, weather)
        # Real history contains faults and pollution episodes; learning "normal" from them blunts the detector.
        # So fit, flag the history, and refit with flagged readings (and whole event rows) left out.
        for _ in range(self.settings["training_passes"] - 1):
            flagged = self.detect(df, weather)
            cleaned = df.copy()
            for sen in SENSORS:
                cleaned.loc[flagged[f"{sen}_fault"].ne("").to_numpy() | flagged["event"].to_numpy(), sen] = np.nan
            self._fit_once(cleaned, weather)
        return self

    def _fit_once(self, df: pd.DataFrame, weather) -> None:
        s = self.settings
        station = df["station_id"]
        self.stations_ = sorted(station.unique().tolist())
        work = _working(df)
        self.step_scale_ = {
            (st, sen): robust_std(work.loc[station == st, sen].diff(), s["resolution"][sen])
            for st in self.stations_
            for sen in SENSORS
        }

        # Learn "normal" only from readings the rule tests don't already reject.
        _, reason = self._layer1(df, clim=None)
        clean = work.mask(reason.isin([*SHAPE_FAULTS, "rate_of_change"]))
        self._fit_climatology(df, clean)
        self._fit_envelope(station, clean)
        departure = clean - self._baseline(df, clean)
        self.departure_scale_ = {sen: self._scales(station, departure[sen], s["min_scale"][sen]) for sen in SENSORS}
        dep_z = self._departure_z(df, clean)
        candidate = self._departures(dep_z, clean)[1]

        design = self._design(df, clean, weather)
        blocks = (df["timestamp"] - df["timestamp"].min()) // pd.Timedelta(days=s["cv_block_days"])
        if blocks.nunique() < 2:
            raise ValueError(f"Need at least {2 * s['cv_block_days']} days of readings to fit the detector")
        folds = GroupKFold(min(5, blocks.nunique()))
        self.models_, self.resid_scale_, oof = {}, {}, {}
        for sen in SENSORS:
            complete = clean[sen].notna() & design[self._needs(sen)].notna().all(axis=1)
            X, y = design.loc[complete, self._inputs(sen)], clean.loc[complete, sen]
            model = TrendPlusTrees(random_state=self.seed)
            # Predict each held-out block from the others, so scales reflect error on unseen stretches of time.
            pred = pd.Series(np.nan, index=df.index)
            pred[complete] = cross_val_predict(model, X, y, cv=folds, groups=blocks[complete])
            oof[sen] = pred
            self.resid_scale_[sen] = self._scales(station, self._detrend(clean[sen] - pred, station), s["min_scale"][sen])
            self.models_[sen] = model.fit(X, y)

        self.pair_scale_ = {}
        for f in SENSORS:
            swapped = design.copy()
            swapped[f] = oof[f]
            for sen in _others(f):
                resid = self._detrend(clean[sen] - self._predict_complete(sen, swapped), station)
                self.pair_scale_[(sen, f)] = self._scales(station, resid, s["min_scale"][sen])

        z = pd.DataFrame({sen: self._z(clean[sen] - oof[sen], station, self.resid_scale_[sen]) for sen in SENSORS})
        clip = s["drift_clip"]
        stat = np.abs(self._ewma_by_station(station, z.clip(-clip, clip).to_numpy(), candidate.to_numpy(), np.inf))
        self.drift_limit_ = {
            sen: max(s["drift_min"], float(np.nanquantile(stat[:, j], s["drift_quantile"])))
            for j, sen in enumerate(SENSORS)
        }

        features = self._iforest_features(df, clean, z, dep_z)
        self.iforest_ = IsolationForest(n_estimators=200, random_state=self.seed).fit(features)
        self.iforest_threshold_ = float(np.quantile(-self.iforest_.score_samples(features), s["iforest_quantile"]))

    def detect(self, df: pd.DataFrame, weather: pd.DataFrame | None = None) -> pd.DataFrame:
        if self.uses_weather and weather is None:
            raise ValueError("this detector was trained with weather; pass the hourly weather covering the readings")
        s = self.settings
        df = _sorted(df)
        station = df["station_id"]
        qc, reason = self._layer1(df, self._clim_lookup(df))
        clean = _working(df).mask(reason.isin(list(SHAPE_FAULTS)))
        dep_z = self._departure_z(df, clean)
        n_departing, candidate = self._departures(dep_z, clean)
        event = sustained(candidate, station, s["event_min_points"])

        z, z_without, raw = self._residuals(df, clean, weather)
        limit = s["residual_z"]
        instant = isolate(z.abs() / limit, {f: z_without[f].abs() / limit for f in SENSORS})
        rate_of_change = reason.eq("rate_of_change")
        drift = isolate(*self._drift_statistics(station, z, z_without, raw, candidate, instant, rate_of_change))
        # A sensor that is clearly wrong right now explains its neighbours' smaller disagreements.
        drift = drift.mask((instant != "") & (instant != drift), "")

        lone_step = rate_of_change.sum(axis=1) == 1
        fault = pd.DataFrame("", index=df.index, columns=SENSORS)
        for sen in SENSORS:
            f = pd.Series("", index=df.index)
            f = f.mask(rate_of_change[sen] & lone_step & ~candidate, "rate_of_change")
            f = f.mask((instant == sen) & ~event, "inconsistent")
            f = f.mask((drift == sen) & ~event, "drift")
            f = f.mask(reason[sen].isin(list(SHAPE_FAULTS)), reason[sen])
            fault[sen] = f

        onset = self._event_onsets(event, fault, station)
        event = event | onset
        fault = fault.mask(fault.isin(SOFT_FAULTS) & onset.to_numpy()[:, None], "")
        any_fault = fault.ne("").any(axis=1)

        score = pd.Series(-self.iforest_.score_samples(self._iforest_features(df, clean, z, dep_z)), index=df.index)
        review = sustained((score > self.iforest_threshold_) & ~any_fault & ~event, station, s["review_min_points"])
        all_missing = df[SENSORS].isna().all(axis=1)
        verdict = np.select(
            [all_missing, event, any_fault, review],
            ["missing", "possible_event", "sensor_fault", "needs_review"],
            default="normal",
        )

        out = df[["station_id", "timestamp", *SENSORS]].copy()
        for sen in SENSORS:
            out[f"{sen}_qc"] = qc[sen]
            out[f"{sen}_fault"] = fault[sen]
            # Empty where the consistency check was paused (missing inputs or conditions unseen in training).
            out[f"{sen}_z"] = z[sen].round(2)
        out["n_departing"] = n_departing
        out["anomaly_score"] = score.round(3)
        out["event"] = event
        out["verdict"] = verdict
        return out

    def _layer1(self, df, clim):
        qc = pd.DataFrame(QC_PASS, index=df.index, columns=SENSORS)
        reason = pd.DataFrame("", index=df.index, columns=SENSORS)
        for st, idx in df.groupby("station_id", sort=False).indices.items():
            rows = df.index[idx]
            for sen in SENSORS:
                low = high = None
                if clim is not None:
                    low, high = clim.loc[rows, f"{sen}_low"], clim.loc[rows, f"{sen}_high"]
                flag, why = rule_tests(df.loc[rows, sen], sen, self._step_scale(st, sen), low, high, self.settings)
                qc.loc[rows, sen] = flag.to_numpy()
                reason.loc[rows, sen] = why.to_numpy()
        return qc, reason

    def _step_scale(self, station: str, sensor: str) -> float:
        known = self.step_scale_.get((station, sensor))
        if known is not None:
            return known
        return float(np.median([v for (_, sen), v in self.step_scale_.items() if sen == sensor]))

    def _band(self, frame: pd.DataFrame, quantiles, margin) -> pd.Series:
        band = {}
        for sen in SENSORS:
            v = frame[sen].dropna().to_numpy()
            lo = hi = np.nan
            if len(v) >= PER_DAY:
                lo, hi = np.quantile(v, quantiles)
                pad = margin * (hi - lo)
                lo, hi = lo - pad, hi + pad
            band[f"{sen}_low"], band[f"{sen}_high"] = lo, hi
        return pd.Series(band, dtype=float)

    def _fit_climatology(self, df, clean):
        s = self.settings
        frame = clean.assign(station_id=df["station_id"], month=df["timestamp"].dt.month)
        self.clim_month_ = (
            frame.groupby(["station_id", "month"])
            .apply(self._band, s["climatology_quantiles"], s["climatology_margin"])
            .reset_index()
        )

    def _clim_lookup(self, df) -> pd.DataFrame:
        """Seasonal band per row; NaN (test not evaluated) where that station-month was never seen."""
        keys = pd.DataFrame({"station_id": df["station_id"].to_numpy(), "month": df["timestamp"].dt.month.to_numpy()})
        cols = [c for c in self.clim_month_.columns if c not in ("station_id", "month")]
        out = keys.merge(self.clim_month_, on=["station_id", "month"], how="left")[cols]
        out.index = df.index
        return out

    def _fit_envelope(self, station, clean):
        s = self.settings
        codes = station.map({st: float(i) for i, st in enumerate(self.stations_)})
        per_station = clean.groupby(codes).apply(self._band, s["envelope_quantiles"], s["envelope_margin"])
        overall = self._band(clean, s["envelope_quantiles"], s["envelope_margin"])
        self.envelope_ = {
            sen: {
                "low": per_station[f"{sen}_low"].to_dict(),
                "high": per_station[f"{sen}_high"].to_dict(),
                ALL: (overall[f"{sen}_low"], overall[f"{sen}_high"]),
            }
            for sen in SENSORS
        }

    def _trusted(self, design, sensor) -> pd.Series:
        """Whether every sensor input of `sensor`'s model lies inside the training envelope."""
        ok = pd.Series(True, index=design.index)
        for sen in self._needs(sensor):
            if sen in SENSORS:
                env = self.envelope_[sen]
                low = design["station"].map(env["low"]).fillna(env[ALL][0])
                high = design["station"].map(env["high"]).fillna(env[ALL][1])
                ok &= design[sen].between(low, high)
        return ok

    def _baseline(self, df, clean) -> pd.DataFrame:
        window = self.settings["baseline_days"] * PER_DAY
        return clean.groupby(df["station_id"]).transform(lambda x: x.rolling(window, min_periods=PER_DAY).median())

    def _departure_z(self, df, clean) -> pd.DataFrame:
        departure = clean - self._baseline(df, clean)
        station = df["station_id"]
        return pd.DataFrame({s: departure[s] / self._scale(station, self.departure_scale_[s]) for s in SENSORS})

    def _departures(self, dep_z, clean):
        departing = (dep_z.abs() > self.settings["event_z"]) & clean.notna()
        n = departing.sum(axis=1)
        return n, n >= self.settings["event_min_sensors"]

    def _event_onsets(self, event, fault, station) -> pd.Series:
        """Rows with only soft faults in the lead-up to a confirmed event: the event's leading edge."""
        window = self.settings["event_onset_hours"] * PER_DAY // 24
        soft_only = fault.isin(SOFT_FAULTS).any(axis=1) & ~fault.isin(list(SHAPE_FAULTS)).any(axis=1)
        starts = event & ~(event.shift(fill_value=False) & (station == station.shift()))
        onset = np.zeros(len(event), dtype=bool)
        same_station = station.to_numpy()
        soft = soft_only.to_numpy()
        for pos in np.flatnonzero(starts.to_numpy()):
            lead = np.arange(max(0, pos - window), pos)
            lead = lead[same_station[lead] == same_station[pos]]
            onset[lead[soft[lead]]] = True
        return pd.Series(onset, index=event.index)

    def _design(self, df, clean, weather) -> pd.DataFrame:
        # Hour of day only: calendar features would stop generalising the moment a new season starts.
        hour = 2 * np.pi * (df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60) / 24
        window = self.settings["temperature_anchor_hours"] * PER_DAY // 24
        X = clean[SENSORS].copy()
        X["temperature_anchor"] = (
            clean["temperature"]
            .groupby(df["station_id"])
            # A full day minimum: a shorter window's median would lean toward whichever part of the daily cycle it saw.
            .transform(lambda x: x.rolling(window, min_periods=PER_DAY).median())
        )
        X["hour_sin"], X["hour_cos"] = np.sin(hour), np.cos(hour)
        X["station"] = df["station_id"].map({st: float(i) for i, st in enumerate(self.stations_)})
        if self.uses_weather:
            s = self.settings
            memory = weather_memory(weather, df["timestamp"], s["rain_memory_hours"], s.get("air_memory_hours", ()))
            X = X.join(memory[self._weather_columns()])
        return X

    def _weather_columns(self) -> list[str]:
        if hasattr(self, "weather_inputs_"):
            return self.weather_inputs_
        # Detectors saved before air temperature was used knew only rain, if anything.
        return [f"rain_{h}h" for h in self.settings["rain_memory_hours"]] if getattr(self, "uses_rain_", False) else []

    @property
    def uses_weather(self) -> bool:
        return bool(self._weather_columns())

    def _needs(self, sensor: str) -> list[str]:
        """Inputs that must be present to predict `sensor`: the other sensors for chemistry; for temperature, the
        air temperature history, or failing that its own trailing median (which cannot reveal a slow drift)."""
        if sensor != "temperature":
            return _others(sensor)
        return [c for c in self._weather_columns() if c.startswith("air_")] or ["temperature_anchor"]

    def _inputs(self, sensor: str) -> list[str]:
        needs = self._needs(sensor)
        weather = [c for c in self._weather_columns() if c not in needs]
        return [*needs, *weather, "hour_sin", "hour_cos", "station"]

    def _predict_complete(self, sensor, design) -> pd.Series:
        """Prediction of `sensor`, only where its inputs are present and inside the training envelope."""
        ok = design[self._needs(sensor)].notna().all(axis=1) & self._trusted(design, sensor)
        pred = pd.Series(np.nan, index=design.index)
        if ok.any():
            pred[ok] = self.models_[sensor].predict(design.loc[ok, self._inputs(sensor)])
        return pred

    @staticmethod
    def _scales(station, resid, floor) -> dict:
        scales = {st: robust_std(resid[station == st], floor) for st in station.unique()}
        scales[ALL] = robust_std(resid, floor)
        return scales

    @staticmethod
    def _scale(station, scales) -> pd.Series:
        return station.map({k: v for k, v in scales.items() if k != ALL}).fillna(scales[ALL]).astype(float)

    def _residuals(self, df, clean, weather):
        """Each sensor vs what its inputs predict; plus the same with each sensor swapped for its reconstruction."""
        station = df["station_id"]
        design = self._design(df, clean, weather)
        pred = {sen: self._predict_complete(sen, design) for sen in SENSORS}
        resid = pd.DataFrame({sen: clean[sen] - pred[sen] for sen in SENSORS})
        z = pd.DataFrame({sen: self._z(resid[sen], station, self.resid_scale_[sen]) for sen in SENSORS})
        # Also without removing the trend: a drifting sensor stays off the model while the fault lasts, and
        # returns to it the moment the sensor is cleaned.
        raw = {"": pd.DataFrame({sen: resid[sen] / self._scale(station, self.resid_scale_[sen]) for sen in SENSORS})}
        z_without = {}
        for f in SENSORS:
            swapped = design.copy()
            swapped[f] = pred[f]
            resid_f = {sen: clean[sen] - self._predict_complete(sen, swapped) for sen in _others(f)}
            z_without[f] = pd.DataFrame({sen: self._z(r, station, self.pair_scale_[(sen, f)]) for sen, r in resid_f.items()})
            raw[f] = pd.DataFrame({sen: r / self._scale(station, self.pair_scale_[(sen, f)]) for sen, r in resid_f.items()})
        return z, z_without, raw

    def _z(self, resid: pd.Series, station: pd.Series, scales: dict) -> pd.Series:
        scale = self._scale(station, scales)
        raw = resid / scale
        if not self.settings["residual_detrend_days"]:
            return raw
        trend_free = self._detrend(resid, station) / scale
        # A fault must show against both the model and its own recent trend. That keeps slow model error out,
        # and stops a repaired sensor looking wrong in the opposite direction while the trend catches up.
        agree = np.sign(raw) == np.sign(trend_free)
        both = np.sign(raw) * np.minimum(raw.abs(), trend_free.abs())
        return both.where(agree, 0.0).where(raw.notna())

    def _detrend(self, resid: pd.Series, station: pd.Series) -> pd.Series:
        if not self.settings["residual_detrend_days"]:
            return resid
        window = self.settings["residual_detrend_days"] * PER_DAY
        trend = resid.groupby(station).transform(lambda x: x.rolling(window, min_periods=PER_DAY).median())
        return resid - trend.fillna(0.0)

    def _ewma_by_station(self, station, values, reset, cap, hours=None) -> np.ndarray:
        weight = 1 - np.exp(-1 / ((hours or self.settings["drift_hours"]) * PER_DAY / 24))
        out = np.empty(values.shape)
        for idx in station.groupby(station, sort=False).indices.values():
            out[idx] = ewma(values[idx], weight, reset[idx], cap)
        return out

    def _drift_statistics(self, station, z, z_without, raw, candidate, instant, jumped):
        """Long-run average of every residual, scaled so 1 is that sensor's alarm level."""
        s = self.settings
        clip, cap = s["drift_clip"], s["drift_cap"]
        # Restart during events, and while a different sensor is clearly faulty: that disagreement is borrowed.

        def scaled(frame, reconstructed=None):
            limit = np.array([self.drift_limit_[sen] for sen in frame.columns])
            # Also restart when the sensor (or the one being reconstructed) jumps: cleaning a fouled sensor snaps
            # it back in one step, and the drift it had built up is over.
            swapped = jumped[reconstructed] if reconstructed else False
            reset = np.column_stack(
                [
                    (candidate | ((instant != "") & (instant != sen) & (instant != reconstructed)) | jumped[sen] | swapped).to_numpy()
                    for sen in frame.columns
                ]
            )
            level = self._ewma_by_station(station, frame.clip(-clip, clip).to_numpy(), reset, cap * limit)
            # The last few hours of plain disagreement must still lean the same way: once a drifting sensor is
            # cleaned the alarm clears within the hour, not after the long average has slowly wound down.
            untrended = raw[reconstructed or ""][frame.columns].clip(-clip, clip).to_numpy()
            recent = self._ewma_by_station(station, untrended, reset, np.inf, s["drift_confirm_hours"])
            confirmed = (np.sign(recent) == np.sign(level)) & (np.abs(recent) >= s["drift_confirm_fraction"] * limit)
            return pd.DataFrame(np.where(confirmed, np.abs(level) / limit, 0.0), columns=frame.columns, index=frame.index)

        return scaled(z), {f: scaled(z_without[f], f) for f in SENSORS}

    def _iforest_features(self, df, clean, z, dep_z) -> pd.DataFrame:
        station = df["station_id"]
        step = pd.DataFrame(
            {
                sen: clean[sen].groupby(station).diff()
                / station.map({st: self._step_scale(st, sen) for st in station.unique()})
                for sen in SENSORS
            }
        )
        wobble = step.groupby(station).transform(lambda x: x.rolling(6, min_periods=2).std())
        features = pd.concat(
            [dep_z.add_suffix("_dep"), z.add_suffix("_resid"), step.add_suffix("_step"), wobble.add_suffix("_wobble")],
            axis=1,
        )
        return features.clip(-10, 10).fillna(0.0)
