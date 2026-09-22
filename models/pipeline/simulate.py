"""One synthetic world for the whole system: sensor export (with faults and pollution events), weather and lab results."""

import argparse
from pathlib import Path

import pandas as pd

from ..anomaly_detection.data import prepare
from ..anomaly_detection.inject import inject
from ..bod_surrogate.simulate import simulate as simulate_lakes_and_lab
from ..wqi.generate_data import STATIONS

HERE = Path(__file__).parent


def simulate(days: int = 455, seed: int = 31, stations: dict = STATIONS, chunk_days: int = 120):
    """Return (10-minute readings with injected faults and events, hourly weather, lab results, injected episodes)."""
    readings, weather, lab = simulate_lakes_and_lab(days, seed, stations, algae=True, coliform=True)
    readings, _ = prepare(readings)
    # Real sensor history contains faults and pollution episodes all year round, so inject them chunk by chunk.
    start = readings["timestamp"].min()
    chunk = ((readings["timestamp"] - start) // pd.Timedelta(days=chunk_days)).to_numpy()
    parts, episodes = [], []
    for i, n in enumerate(sorted(set(chunk))):
        dirty, found = inject(readings[chunk == n], seed=seed + i, per_kind=1)
        parts.append(dirty)
        episodes.append(found)
    dirty = pd.concat(parts, ignore_index=True).sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    return dirty, weather, lab, pd.concat(episodes, ignore_index=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=455)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--out-dir", type=Path, default=HERE / "data")
    args = parser.parse_args(argv)

    readings, weather, lab, episodes = simulate(args.days, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    readings.to_csv(args.out_dir / "sensor_readings.csv", index=False)
    weather.to_csv(args.out_dir / "weather.csv", index=False)
    lab.to_csv(args.out_dir / "lab_results.csv", index=False)
    episodes.to_csv(args.out_dir / "injected_episodes.csv", index=False)
    print(f"Wrote {len(readings)} sensor readings ({len(episodes)} injected faults and events), {len(weather)} hours "
          f"of weather and {len(lab)} lab results to {args.out_dir}")


if __name__ == "__main__":
    main()
