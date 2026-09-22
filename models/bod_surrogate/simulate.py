"""Simulate sensors, weather and weekly lab BOD results driven by the same hidden pollution load."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from ..forecasting.simulate import simulate as simulate_lakes
from ..wqi.generate_data import STATIONS

HERE = Path(__file__).parent


def simulate(
    days: int = 455,
    seed: int = 21,
    stations: dict = STATIONS,
    every_days: int = 7,
    missed: float = 0.08,
    lab_cv: float = 0.15,
    algae: bool = False,
    coliform: bool = False,
):
    """Return (10-minute readings, hourly weather, weekly lab results: BOD, conductivity, nitrate and options)."""
    readings, weather = simulate_lakes(days, seed, stations=stations, include_load=True, labile=True, algae=algae)
    rng = np.random.default_rng(seed + 7)
    # The rest of the lab panel comes from the same visit; separate streams keep the BOD results unchanged.
    panel_rng = np.random.default_rng(seed + 9)
    algae_rng = np.random.default_rng(seed + 11)
    coliform_rng = np.random.default_rng(seed + 13)
    # Rain washes faecal bacteria in from the catchment; they die off over roughly a day and a half.
    runoff = pd.Series(lfilter([1.0], [1.0, -np.exp(-1 / 36)], weather["rain_mm"].to_numpy()), index=weather["timestamp"])
    samples = []
    for station, g in readings.groupby("station_id", sort=True):
        hidden = g.set_index("timestamp")[["load", "labile", "algae"]]
        first = hidden.index.min().normalize() + pd.Timedelta(days=int(rng.integers(1, every_days + 1)))
        dates = pd.date_range(first, hidden.index.max() - pd.Timedelta(days=2), freq=f"{every_days}D")
        # Labile organic matter drives both BOD and the lake's night respiration, so respiration reveals part of
        # it; a smaller share of organic differences shows in no sensor at all, so no soft sensor can be perfect.
        unseen = np.exp(lfilter([1.0], [1.0, -0.9], rng.normal(0, 0.03, len(dates))))
        for date, factor in zip(dates, unseen):
            if rng.random() < missed:
                continue
            when = date + pd.Timedelta(days=int(rng.integers(-1, 2)), hours=10, minutes=int(rng.integers(0, 60)))
            load, labile = hidden["load"].asof(when), hidden["labile"].asof(when)
            true_bod = (0.6 + 7 * load**1.4) * labile * factor
            measured = true_bod * rng.lognormal(0, lab_cv)  # BOD5 lab repeatability is roughly +-15%
            sample = {
                "station_id": station,
                "timestamp": when,
                "bod": "<1" if measured < 1 else round(measured, 1),
                "conductivity": round(max(250 + 900 * load + panel_rng.normal(0, 60), 50.0)),
                "nitrate": round(0.5 + 25 * load**2 * panel_rng.lognormal(0, 0.3), 2),
            }
            if algae:
                # Chlorophyll-a tracks algal biomass; pigment per cell varies with the species present, and
                # extraction in the lab adds roughly 12% error.
                chl = 20 * hidden["algae"].asof(when) ** 2 * algae_rng.lognormal(0, 0.08) * algae_rng.lognormal(0, 0.12)
                sample["chlorophyll_a"] = "<0.5" if chl < 0.5 else round(chl, 1)
            if coliform:
                # Sewage raises the baseline, runoff adds pulses; an unseen source share and the MPN method's
                # own imprecision (about +-0.25 log10) keep any nowcast from being exact.
                log10_count = (
                    1.3 + 2.0 * load + 0.8 * np.log10(1 + runoff.asof(when.floor("h")))
                    + coliform_rng.normal(0, 0.15) + coliform_rng.normal(0, 0.25)
                )
                count = 10**log10_count
                sample["total_coliform"] = "<2" if count < 2 else int(round(count))
            samples.append(sample)
    lab = pd.DataFrame(samples).sort_values("timestamp").reset_index(drop=True)
    return readings.drop(columns=["load", "labile", "algae"]), weather, lab


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=455)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--out-dir", type=Path, default=HERE / "data")
    args = parser.parse_args(argv)

    readings, weather, lab = simulate(args.days, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    readings.to_csv(args.out_dir / "synthetic_sensor_readings.csv", index=False)
    weather.to_csv(args.out_dir / "synthetic_weather.csv", index=False)
    lab.to_csv(args.out_dir / "synthetic_lab_bod.csv", index=False)
    print(f"Wrote {len(readings)} sensor readings, {len(weather)} hours of weather and {len(lab)} lab BOD results "
          f"to {args.out_dir}")


if __name__ == "__main__":
    main()
