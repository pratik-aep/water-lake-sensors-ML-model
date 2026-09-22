"""Simulate Udaipur weather and weather-driven lake sensors, with oxygen from a lake metabolism model."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter, lfilter_zi

from ..anomaly_detection.config import INTERVAL, SENSORS
from ..wqi.calculator import do_saturation_mgl
from ..wqi.generate_data import STATIONS

HERE = Path(__file__).parent
STEPS_PER_HOUR = pd.Timedelta(hours=1) // pd.Timedelta(INTERVAL)


def _smooth(x, alpha):
    """Exponential smoothing that starts at the first value instead of zero."""
    zi = lfilter_zi([alpha], [1, alpha - 1]) * x[0]
    return lfilter([alpha], [1, alpha - 1], x, zi=zi)[0]


def simulate_weather(days: int, seed: int, start: str) -> pd.DataFrame:
    """Hourly air temperature (deg C), cloud cover (0-1) and rain (mm) for one city."""
    rng = np.random.default_rng(seed)
    doy = pd.date_range(start, periods=days, freq="D").dayofyear.to_numpy()
    monsoon = np.exp(-(((doy - 210) / 40) ** 2))
    cloud_day = np.clip(0.15 + 0.6 * monsoon + lfilter([1.0], [1.0, -0.6], rng.normal(0, 0.15, days)), 0, 1)
    wet = rng.random(days) < 0.03 + 0.5 * monsoon
    rain_day = np.where(wet, rng.gamma(0.8, 8 + 25 * monsoon), 0.0)
    warm_spell = lfilter([1.0], [1.0, -0.7], rng.normal(0, 1.0, days))

    hours = pd.date_range(start, periods=days * 24, freq="h")
    day = np.repeat(np.arange(days), 24)
    hour = hours.hour.to_numpy()
    cloud = np.clip(cloud_day[day] + rng.normal(0, 0.05, len(hours)), 0, 1)
    rain = np.zeros(len(hours))
    for d in np.flatnonzero(wet):
        first = d * 24 + rng.integers(10, 20)
        length = rng.integers(1, 7)
        rain[first : first + length] += rain_day[d] / length
    season = 24 + 9 * np.sin(2 * np.pi * (hours.dayofyear.to_numpy() - 59) / 365) - 3 * monsoon[day]
    air = season + warm_spell[day] + 5 * (1 - 0.6 * cloud) * np.sin(2 * np.pi * (hour - 9) / 24)
    return pd.DataFrame(
        {"timestamp": hours, "air_temperature": air.round(2), "cloud_cover": cloud.round(3), "rain_mm": rain.round(2)}
    )


def _station(
    name: str, base_load: float, weather: pd.DataFrame, rng, labile: bool = False, algae: bool = False
) -> pd.DataFrame:
    ts = pd.date_range(weather["timestamp"].iloc[0], periods=len(weather) * STEPS_PER_HOUR, freq=INTERVAL)
    n = len(ts)
    air, cloud, rain = (np.repeat(weather[c].to_numpy(), STEPS_PER_HOUR) for c in ("air_temperature", "cloud_cover", "rain_mm"))
    rain = rain / STEPS_PER_HOUR
    hour = ts.hour.to_numpy() + ts.minute.to_numpy() / 60
    monsoon = np.exp(-(((ts.dayofyear.to_numpy() - 210) / 40) ** 2))

    # Surface water follows air temperature with about four days of lag, plus a small afternoon warming.
    water_temp = _smooth(air, 1 / (4 * 24 * STEPS_PER_HOUR)) + 0.8 * (1 - 0.5 * cloud) * np.sin(2 * np.pi * (hour - 10) / 24)
    # Pollution load wanders slowly; rain washes in nutrients that fade over about five days.
    wander = lfilter([1.0], [1.0, -0.9995], rng.normal(0, 0.003, n))
    nutrients = lfilter([0.004], [1.0, -np.exp(-1 / (5 * 24 * STEPS_PER_HOUR))], rain)
    load = np.clip(base_load + 0.08 * monsoon + wander + nutrients, 0, 1)
    # Rain carries sediment in; it settles out over about a day and a half.
    sediment = lfilter([1.2], [1.0, -np.exp(-1 / (1.5 * 24 * STEPS_PER_HOUR))], rain)
    turbidity = 0.5 + 20 * load**2 + sediment

    # Lake metabolism: photosynthesis (sunlight, dimmed by cloud and murky water) minus respiration (faster when
    # warm), plus exchange with the air pulling oxygen toward saturation.
    light = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None) * (1 - 0.65 * cloud) * (1 - 0.3 * np.minimum(turbidity / 30, 1))
    # Optionally, algal biomass also rises and falls on its own (blooms) and scales photosynthesis; it is what
    # lab chlorophyll-a measures.
    bloom = np.exp(lfilter([1.0], [1.0, -0.9993], rng.normal(0, 0.01, n))) if algae else np.ones(n)
    biomass = (0.4 + load) * bloom
    production = biomass * light
    # Optionally, easily decomposed ("labile") organic matter wanders on its own, speeding respiration without
    # showing in any other sensor; it is what a lab BOD test measures.
    organic = np.exp(lfilter([1.0], [1.0, -0.9995], rng.normal(0, 0.004, n))) if labile else np.ones(n)
    respiration = 0.30 * (0.3 + load) * organic * 1.02 ** (water_temp - 20)
    saturation = do_saturation_mgl(water_temp, 0.93)
    oxygen = np.empty(n)
    oxygen[0] = saturation[0]
    for t in range(1, n):
        change = production[t] - respiration[t] + 0.08 * (saturation[t] - oxygen[t - 1])
        oxygen[t] = max(oxygen[t - 1] + change / STEPS_PER_HOUR, 0.0)
    ph = np.clip(7.2 + 1.3 * load + 0.6 * (oxygen / saturation - 1), 6.0, 10.0)

    df = pd.DataFrame(
        {
            "station_id": name,
            "timestamp": ts,
            "temperature": (water_temp + rng.normal(0, 0.03, n)).round(3),
            "ph": (ph + rng.normal(0, 0.01, n)).round(3),
            "turbidity": (turbidity * rng.lognormal(0, 0.03, n)).round(2),
            "dissolved_oxygen": np.clip(oxygen + rng.normal(0, 0.04, n), 0, None).round(3),
        }
    )
    df.loc[rng.random(n) < 0.002, SENSORS] = np.nan
    # Hidden state no sensor measures; other simulators (e.g. lab BOD) derive their ground truth from it.
    df["load"] = load
    df["labile"] = organic
    df["algae"] = biomass
    return df


# Starting in June, the default 455 days end in late August, so a held-out test period covers the monsoon
# months when low-oxygen nights actually happen.
def simulate(
    days: int = 455,
    seed: int = 11,
    start: str = "2024-06-01",
    stations: dict = STATIONS,
    include_load: bool = False,
    labile: bool = False,
    algae: bool = False,
):
    """Return (10-minute sensor readings for every station, hourly city weather)."""
    # include_load keeps the hidden `load`, `labile` and `algae` columns, which no sensor measures.
    weather = simulate_weather(days, seed, start)
    rng = np.random.default_rng(seed + 1)
    readings = pd.concat(
        [_station(name, load, weather, rng, labile, algae) for name, load in stations.items()], ignore_index=True
    )
    return (readings if include_load else readings.drop(columns=["load", "labile", "algae"])), weather


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=455)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out-dir", type=Path, default=HERE / "data")
    args = parser.parse_args(argv)

    readings, weather = simulate(args.days, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    readings.to_csv(args.out_dir / "synthetic_sensor_readings.csv", index=False)
    weather.to_csv(args.out_dir / "synthetic_weather.csv", index=False)
    print(f"Wrote {len(readings)} sensor readings and {len(weather)} hours of weather to {args.out_dir}")


if __name__ == "__main__":
    main()
