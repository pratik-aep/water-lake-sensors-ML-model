"""Candidate soft sensors fitted to the log of a lab-measured value, with ranges from out-of-sample residuals."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import MONOTONE_INCREASING


def candidates(feature_names, seed: int, monotone=MONOTONE_INCREASING) -> dict:
    """`monotone` lists inputs that may only ever raise the estimate (physics the model must respect)."""
    scaled = [SimpleImputer(strategy="median"), StandardScaler()]
    return {
        "mean_baseline": DummyRegressor(),
        "ridge": make_pipeline(*scaled, RidgeCV(alphas=np.logspace(-3, 3, 13))),
        # Partial least squares: the classic soft-sensor method for many correlated inputs and few lab samples.
        "pls": GridSearchCV(
            make_pipeline(*[clone(step) for step in scaled], PLSRegression()),
            {"plsregression__n_components": [1, 2, 3, 4, 5]},
            cv=KFold(4),
            scoring="neg_mean_squared_error",
        ),
        "monotonic_gbm": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=8,
            min_samples_leaf=8,
            l2_regularization=1.0,
            monotonic_cst={f: 1 for f in monotone if f in feature_names},
            random_state=seed,
        ),
    }


def time_folds(n_samples: int, wanted: int) -> KFold:
    """Contiguous folds over time-sorted samples, so validation never interleaves with training weeks."""
    return KFold(max(2, min(wanted, n_samples // 10)))


def out_of_fold(estimator, X: pd.DataFrame, log_values: np.ndarray, folds) -> np.ndarray:
    return np.ravel(cross_val_predict(clone(estimator), X, log_values, cv=folds))


class SoftSensor:
    """A fitted candidate plus its out-of-fold residuals, which turn the log estimate into ranges and odds."""

    def __init__(self, name: str, estimator, settings: dict, target: str = "bod"):
        self.name = name
        self.estimator = estimator
        self.settings = settings
        self.target = target

    def fit(self, X: pd.DataFrame, values) -> "SoftSensor":
        self.features_ = list(X.columns)
        log_values = np.log(np.asarray(values, dtype=float))
        folds = time_folds(len(X), self.settings["cv_folds"])
        self.residuals_ = np.sort(log_values - out_of_fold(self.estimator, X, log_values, folds))
        self.model_ = clone(self.estimator).fit(X, log_values)
        return self

    def predict_log(self, X: pd.DataFrame) -> np.ndarray:
        return np.ravel(self.model_.predict(X[self.features_]))

    def estimate(self, X: pd.DataFrame) -> pd.DataFrame:
        """Median estimate, a conformal range, and the chance of exceeding each configured limit."""
        log_pred = self.predict_log(X)
        r = self.residuals_
        n = len(r)
        alpha = 1 - self.settings["coverage"]
        lo_idx = max(int(np.floor((n + 1) * alpha / 2)) - 1, 0)
        hi_idx = min(int(np.ceil((n + 1) * (1 - alpha / 2))) - 1, n - 1)
        out = pd.DataFrame(
            {
                self.target: np.exp(log_pred + np.median(r)),
                f"{self.target}_lo": np.exp(log_pred + r[lo_idx]),
                f"{self.target}_hi": np.exp(log_pred + r[hi_idx]),
            },
            index=X.index,
        )
        for level in self.settings["exceedance_levels"]:
            out[f"p_above_{level:g}"] = self.p_above(log_pred, level)
        return out

    def p_above(self, log_pred: np.ndarray, level: float) -> np.ndarray:
        return 1 - np.searchsorted(self.residuals_, np.log(level) - log_pred, side="right") / len(self.residuals_)
