"""Gradient-boosted quantile models."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import hourly_rows


class QuantileGBM:
    """One boosted-tree model per target and quantile; predictions are sorted so quantiles never cross."""

    def __init__(self, quantiles, params: dict, seed: int):
        self.quantiles = tuple(quantiles)
        self.params = params
        self.seed = seed

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> "QuantileGBM":
        self.columns_ = list(X.columns)
        self.targets_ = list(Y.columns)
        self.models_ = {}
        for target in self.targets_:
            ok = Y[target].notna().to_numpy()
            for q in self.quantiles:
                model = HistGradientBoostingRegressor(
                    loss="quantile", quantile=q, categorical_features=["station"], random_state=self.seed, **self.params
                )
                self.models_[(target, q)] = model.fit(X.loc[ok], Y.loc[ok, target])
        return self

    def predict(self, X: pd.DataFrame) -> dict:
        X = X[self.columns_]
        return {
            target: np.sort(np.column_stack([self.models_[(target, q)].predict(X) for q in self.quantiles]), axis=1)
            for target in self.targets_
        }


class EnsembleHourlyModel:
    """Averages the members' quantiles (Vincentisation); combinations tend to beat any single forecaster."""

    kind = "ensemble"

    def __init__(self, members: list):
        self.members = members

    def predict_deltas(self, timelines: dict, codes: dict, origins: dict, rng=None):
        outputs = [m.predict_deltas(timelines, codes, origins, rng) for m in self.members]
        info = outputs[0][0]
        return info, {s: np.mean([deltas[s] for _, deltas in outputs], axis=0) for s in outputs[0][1]}


class GBMHourlyModel:
    """Direct multi-horizon forecaster: the horizon is an input, so one model per sensor covers every hour ahead."""

    kind = "gradient_boosting"

    def __init__(self, settings: dict, seed: int):
        self.settings = settings
        self.seed = seed

    def fit(self, timelines: dict, codes: dict, until: pd.Timestamp, rng, validation=None) -> "GBMHourlyModel":
        s = self.settings
        X_parts, Y_parts = [], []
        for station, tl in timelines.items():
            origins = np.arange(24, min(tl.n, tl.position(until)), s["train_origin_every_hours"])
            X, Y, info = hourly_rows(tl, origins, s["train_horizons"], s, codes[station], rng)
            keep = (info["target_time"] < until).to_numpy()
            X_parts.append(X[keep])
            Y_parts.append(Y[keep])
        self.gbm = QuantileGBM(s["quantiles"], s["gbm"], self.seed).fit(
            pd.concat(X_parts, ignore_index=True), pd.concat(Y_parts, ignore_index=True)
        )
        return self

    def predict_deltas(self, timelines: dict, codes: dict, origins: dict, rng=None):
        X_parts, info_parts = [], []
        for station, positions in origins.items():
            tl = timelines[station]
            X, _, info = hourly_rows(tl, positions, self.settings["horizons"], self.settings, codes.get(station, np.nan), rng)
            X_parts.append(X)
            info_parts.append(info)
        return pd.concat(info_parts, ignore_index=True), self.gbm.predict(pd.concat(X_parts, ignore_index=True))
