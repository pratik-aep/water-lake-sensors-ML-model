"""Simulate clean 10-minute sensor readings for the Udaipur lake stations."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from ..wqi.calculator import do_saturation_mgl
from ..wqi.generate_data import STATIONS
from .config import INTERVAL, SENSORS

DEFAULT_OUT = Path(__file__).parent / "data" / "synthetic_sensor_readings.csv"


def _station(name: str, base_load: float, ts: pd.DatetimeIndex, rng) -> pd.DataFrame:
    n = len(ts)
    doy = ts.dayofyear.to_numpy()
    hour = ts.hour.to_numpy() + ts.minute.to_numpy() / 60

    # Pollution load wanders slowly (AR(1) with roughly a two-week memory) plus a monsoon runoff bump.
    wander = lfilter([1.0], [1.0, -0.9995], rng.normal(0, 0.003, n))
    load = np.clip(base_load + 0.1 * np.exp(-(((doy - 220) / 30) ** 2)) + wander, 0.0, 1.0)
    # Photosynthesis peaks mid-afternoon: oxygen and pH rise by day and fall overnight, more in richer water.
    daylight = np.sin(2 * np.pi * (hour - 9) / 24)
    temperature = 25 + 6 * np.sin(2 * np.pi * (doy - 59) / 365) + 0.8 * daylight
    dissolved_oxygen = do_saturation_mgl(temperature) * (1.05 - 0.85 * load) + 0.6 * (0.5 + load) * daylight
    ph = 7.2 + 1.3 * load + 0.12 * (0.5 + load) * daylight
    turbidity = 0.5 + 20 * load**2

    df = pd.DataFrame(
        {
            "station_id": name,
            "timestamp": ts,
            "temperature": (temperature + rng.normal(0, 0.03, n)).round(3),
            "ph": (ph + rng.normal(0, 0.01, n)).round(3),
            "turbidity": (turbidity * rng.lognormal(0, 0.03, n)).round(2),
            "dissolved_oxygen": np.clip(dissolved_oxygen + rng.normal(0, 0.04, n), 0, None).round(3),
        }
    )
    # The odd missed transmission.
    df.loc[rng.random(n) < 0.002, SENSORS] = np.nan
    return df


def simulate(days: int = 455, seed: int = 7, start: str = "2024-01-01", stations: dict = STATIONS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=days * pd.Timedelta(days=1) // pd.Timedelta(INTERVAL), freq=INTERVAL)
    return pd.concat([_station(name, load, ts, rng) for name, load in stations.items()], ignore_index=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=455, help="a year to learn from plus a held-out test period")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    df = simulate(args.days, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows ({df['station_id'].nunique()} stations x {args.days} days at {INTERVAL}) to {args.out}")


if __name__ == "__main__":
    main()
