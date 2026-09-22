"""Simulate sensors, weather and weekly lab results (including chlorophyll-a) for lakes with algal blooms."""

import argparse
from pathlib import Path

from ..bod_surrogate.simulate import simulate as simulate_lab_world
from ..wqi.generate_data import STATIONS

HERE = Path(__file__).parent


def simulate(days: int = 455, seed: int = 51, stations: dict = STATIONS):
    """Return (10-minute sensor readings, hourly weather, weekly lab results with chlorophyll_a)."""
    return simulate_lab_world(days, seed, stations, algae=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=455)
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--out-dir", type=Path, default=HERE / "data")
    args = parser.parse_args(argv)

    readings, weather, lab = simulate(args.days, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    readings.to_csv(args.out_dir / "synthetic_sensor_readings.csv", index=False)
    weather.to_csv(args.out_dir / "synthetic_weather.csv", index=False)
    lab.to_csv(args.out_dir / "synthetic_lab_results.csv", index=False)
    print(f"Wrote {len(readings)} sensor readings, {len(weather)} hours of weather and {len(lab)} lab results "
          f"to {args.out_dir}")


if __name__ == "__main__":
    main()
