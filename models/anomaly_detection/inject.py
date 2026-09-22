"""Inject labelled sensor faults and pollution events into clean readings, to test the detector."""

import numpy as np
import pandas as pd

from .config import INTERVAL, SENSORS
from .rules import from_working, to_working

# The fault types Leigh et al. (2019) rank as most important for in-situ water-quality sensors.
FAULT_KINDS = ["spike", "flatline", "offset", "drift", "oscillation"]
KINDS = [*FAULT_KINDS, "event", "dropout"]
# Size on the working scale (turbidity is log1p NTU); each injection multiplies it by a random 0.8-1.5.
MAGNITUDE = {
    "spike": {"temperature": 4.0, "ph": 1.0, "turbidity": 1.5, "dissolved_oxygen": 3.0},
    "offset": {"temperature": 2.0, "ph": 0.5, "turbidity": 0.8, "dissolved_oxygen": 1.5},
    "drift": {"temperature": 1.5, "ph": 0.5, "turbidity": 1.0, "dissolved_oxygen": 2.0},
    "oscillation": {"temperature": 1.0, "ph": 0.3, "turbidity": 0.5, "dissolved_oxygen": 1.0},
}
# Fouling makes turbidity read high and oxygen read low; other faults can go either way.
FIXED_SIGN = {("drift", "turbidity"): 1, ("drift", "dissolved_oxygen"): -1, ("spike", "turbidity"): 1}
# A pollution event moves several sensors together: cloudier water, less oxygen, higher pH.
EVENT_EFFECT = {"turbidity": 1.2, "dissolved_oxygen": -2.5, "ph": 0.4}
DURATION_HOURS = {
    "spike": (0.17, 0.34),
    "flatline": (3, 12),
    "offset": (12, 96),
    "drift": (72, 240),
    "oscillation": (2, 8),
    "event": (6, 48),
    "dropout": (1, 12),
}


def _shift(df, rows, sensor, delta):
    shifted = from_working(to_working(df.loc[rows, sensor], sensor) + delta, sensor)
    df.loc[rows, sensor] = shifted.clip(lower=0) if sensor in ("turbidity", "dissolved_oxygen") else shifted


def apply_episode(df: pd.DataFrame, rows, kind: str, sensor: str = "", size: float = 1.0) -> None:
    """Modify df in place; `size` is a signed multiplier on the kind's standard magnitude."""
    n = len(rows)
    if kind == "dropout":
        df.loc[rows, SENSORS] = np.nan
    elif kind == "event":
        t = np.linspace(0, 1, n)
        pulse = np.clip(np.minimum(t / 0.2, (1 - t) / 0.4), 0, 1)
        for sen, effect in EVENT_EFFECT.items():
            _shift(df, rows, sen, abs(size) * effect * pulse)
    elif kind == "flatline":
        held = df.loc[rows, sensor].dropna()
        if len(held):
            df.loc[rows, sensor] = held.iloc[0]
    else:
        magnitude = size * MAGNITUDE[kind][sensor]
        shape = {
            "spike": np.ones(n),
            "offset": np.ones(n),
            "drift": np.linspace(0, 1, n),
            "oscillation": np.where(np.arange(n) % 2 == 0, 1.0, -1.0),
        }[kind]
        _shift(df, rows, sensor, magnitude * shape)


def inject(df: pd.DataFrame, seed: int, per_kind: int = 2, interval: str = INTERVAL):
    """Return (dirty copy, episodes); episodes are spaced 1-2 days apart so each one can be scored on its own."""
    rng = np.random.default_rng(seed)
    out = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    slot = pd.Timedelta(interval)
    per_day = pd.Timedelta(days=1) // slot
    episodes = []
    for station, idx in out.groupby("station_id", sort=False).indices.items():
        kinds = KINDS * per_kind
        rng.shuffle(kinds)
        cursor = per_day
        for kind in kinds:
            lo, hi = DURATION_HOURS[kind]
            length = max(1, round(rng.uniform(lo, hi) * pd.Timedelta(hours=1) / slot))
            start = cursor + round(rng.uniform(1, 2) * per_day)
            if start + length >= len(idx):
                break
            rows = out.index[idx[start : start + length]]
            sensor = "" if kind in ("event", "dropout") else str(rng.choice(SENSORS))
            sign = FIXED_SIGN.get((kind, sensor), rng.choice([-1, 1]))
            apply_episode(out, rows, kind, sensor, sign * rng.uniform(0.8, 1.5))
            episodes.append(
                {
                    "station_id": station,
                    "kind": kind,
                    "sensor": sensor,
                    "start": out.at[rows[0], "timestamp"],
                    "end": out.at[rows[-1], "timestamp"],
                }
            )
            cursor = start + length
    return out, pd.DataFrame(episodes, columns=["station_id", "kind", "sensor", "start", "end"])
