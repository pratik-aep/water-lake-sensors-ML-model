"""Generate synthetic lake readings shaped like the real station + lab data will be."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .calculator import do_saturation_mgl
from .config import FEATURE_SETS
from .dataset import prepare

DEFAULT_OUT = Path(__file__).parent / "data" / "synthetic_lake_readings.csv"
SENSOR_COLUMNS = FEATURE_SETS["sensor"]
# Typical pollution load per lake, ordered as in the Udaipur lake studies (Goverdhan Sagar worst, Fateh Sagar best).
STATIONS = {"fateh_sagar": 0.22, "pichola": 0.32, "swaroop_sagar": 0.32, "goverdhan_sagar": 0.55}


def _station(name: str, base_load: float, days: pd.DatetimeIndex, rng) -> pd.DataFrame:
    n = len(days)
    doy = days.dayofyear.to_numpy()

    # AR(1) drift: pollution episodes persist for days, so neighbouring readings are correlated.
    drift = np.zeros(n)
    for i in range(1, n):
        drift[i] = 0.95 * drift[i - 1] + rng.normal(0, 0.05)
    monsoon_runoff = 0.1 * np.exp(-(((doy - 220) / 30) ** 2))
    load = np.clip(base_load + monsoon_runoff + drift, 0.0, 1.0)
    temperature = 25 + 6 * np.sin(2 * np.pi * (doy - 59) / 365) + rng.normal(0, 0.8, n)

    ph = np.clip(7.2 + 1.3 * load + rng.normal(0, 0.2, n), 6.0, 9.8)
    turbidity = 0.5 + 20 * load**2 * rng.lognormal(0, 0.35, n)
    dissolved_oxygen = np.clip(
        do_saturation_mgl(temperature) * (1.05 - 0.85 * load) + rng.normal(0, 0.3, n), 0.2, None
    )
    # The optical turbidity window fouls and reads high until it is cleaned, every 30 days here.
    fouling = 1 + 0.3 * (np.arange(n) % 30) / 30

    df = pd.DataFrame(
        {
            "station_id": name,
            "timestamp": days,
            "temperature": temperature,
            "ph": ph + rng.normal(0, 0.1, n),
            "turbidity": turbidity * fouling * rng.lognormal(0, 0.1, n),
            "dissolved_oxygen": np.clip(dissolved_oxygen + rng.normal(0, 0.3, n), 0.0, None),
            "lab_ph": ph,
            "lab_turbidity": turbidity,
            "lab_dissolved_oxygen": dissolved_oxygen,
            "bod": 0.5 + 10 * load**1.5 * rng.lognormal(0, 0.3, n),
            "conductivity": np.clip(250 + 900 * load + rng.normal(0, 80, n), 50, None),
            "nitrate": 0.5 + 25 * load**2 * rng.lognormal(0, 0.4, n),
        }
    )
    # Real sensors drop out now and then; the models have to cope with gaps.
    df[SENSOR_COLUMNS] = df[SENSOR_COLUMNS].mask(rng.random((n, len(SENSOR_COLUMNS))) < 0.02)
    return df


def generate(days: int = 1250, seed: int = 42, start: str = "2023-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    df = pd.concat([_station(name, load, dates, rng) for name, load in STATIONS.items()], ignore_index=True)
    numeric = df.select_dtypes("number").columns
    df[numeric] = df[numeric].round(2)
    return df


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=1250, help="days of daily readings per station")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    df = generate(args.days, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    _, report = prepare(df, SENSOR_COLUMNS)
    print(f"Wrote {len(df)} rows ({len(STATIONS)} stations x {args.days} days) to {args.out}")
    print("Class counts:", report["class_counts"])


if __name__ == "__main__":
    main()
