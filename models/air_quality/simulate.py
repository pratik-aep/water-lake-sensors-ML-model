"""Simulate an Udaipur-like city's air stations (the vendor's gases in ppm, particles in ug/m3) and their weather."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from ..forecasting.simulate import simulate_weather

HERE = Path(__file__).parent
STEPS_PER_HOUR = 4  # 15-minute readings
# Station: (traffic weight, industry weight). Madri is Udaipur's industrial area.
STATIONS = {"air_city_centre": (1.0, 0.2), "air_madri_industrial": (0.5, 1.0)}


def add_air_weather(weather: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Wind, mixing height and humidity consistent with the temperature, cloud and rain already simulated."""
    rng = np.random.default_rng(seed)
    t = pd.to_datetime(weather["timestamp"])
    hour, doy = t.dt.hour.to_numpy(), t.dt.dayofyear.to_numpy()
    monsoon = np.exp(-(((doy - 210) / 40) ** 2))
    winter = 0.5 * (1 + np.cos(2 * np.pi * (doy - 15) / 365))
    day = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None)
    gust = np.exp(lfilter([1.0], [1.0, -0.9], rng.normal(0, 0.12, len(t))))
    wind = (1.0 + 2.5 * day + 2.0 * monsoon) * gust
    # The mixed layer collapses at night (pollution trapped near the ground) and grows with afternoon heating.
    sun = day * (1 - 0.6 * weather["cloud_cover"].to_numpy())
    mixing = 150 + (2600 - 1500 * winter - 800 * monsoon) * sun + 60 * wind
    humidity = np.clip(35 + 45 * monsoon + 20 * winter - 1.2 * (weather["air_temperature"].to_numpy() - 25)
                       + 25 * (weather["rain_mm"].to_numpy() > 0) - 15 * day + rng.normal(0, 4, len(t)), 8, 100)
    return weather.assign(wind_speed=wind.round(2), boundary_layer_height=mixing.round(0), relative_humidity=humidity.round(1))


def humidity_growth(relative_humidity, kappa: float = 0.4):
    """How much an optical counter over-reads particle mass as particles take up water (kappa-Koehler)."""
    water = np.clip(np.asarray(relative_humidity, dtype=float), 0, 95) / 100
    return 1 + (kappa / 1.65) / (1 / water - 1)


def _station(name: str, traffic_w: float, industry_w: float, weather: pd.DataFrame, rng) -> pd.DataFrame:
    w = weather.set_index("timestamp")
    ts = pd.date_range(w.index[0], periods=len(w) * STEPS_PER_HOUR, freq="15min")
    hourly = {c: np.repeat(w[c].to_numpy(), STEPS_PER_HOUR) for c in ("air_temperature", "cloud_cover", "rain_mm", "wind_speed",
                                                                     "boundary_layer_height", "relative_humidity")}
    n, hour, doy = len(ts), ts.hour.to_numpy() + ts.minute.to_numpy() / 60, ts.dayofyear.to_numpy()
    weekday = ts.dayofweek.to_numpy() < 5
    winter = 0.5 * (1 + np.cos(2 * np.pi * (doy - 15) / 365))
    dust_season = np.exp(-(((doy - 135) / 30) ** 2))  # April-June: dry, windy, dusty
    burning = np.exp(-(((doy - 310) / 12) ** 2))  # late October-November: crop burning and festival fireworks

    traffic = (0.35 + np.exp(-((hour - 9) / 1.5) ** 2) + 0.9 * np.exp(-((hour - 19.5) / 2) ** 2)) * np.where(weekday, 1.0, 0.75)
    emissions = traffic_w * traffic + industry_w * 0.6
    ventilation = np.clip(hourly["wind_speed"] * hourly["boundary_layer_height"], 150, None)
    rain_6h = lfilter(np.ones(6 * STEPS_PER_HOUR), [1.0], hourly["rain_mm"] / STEPS_PER_HOUR)
    washout = np.exp(-0.25 * rain_6h)
    wander = np.exp(lfilter([1.0], [1.0, -0.995], rng.normal(0, 0.03, n)))

    regional = (12 + 38 * winter + 55 * burning) * wander
    fine = (regional + 2600 * emissions / ventilation) * washout
    coarse = (18 * traffic_w * traffic + 45 * dust_season * hourly["wind_speed"] ** 1.2 * wander) * washout
    pm25, pm10 = fine, fine + coarse

    sun = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None) * (1 - 0.7 * hourly["cloud_cover"])
    no2 = (0.004 + 1.4 * emissions / ventilation * (1 + 0.5 * winter)) * wander
    o3 = np.clip(0.018 + 0.045 * sun * np.clip(hourly["air_temperature"] / 35, 0.3, 1.3) - 0.35 * no2, 0.002, None)
    co = (0.25 + 45 * emissions / ventilation + 0.8 * burning) * wander
    so2 = 0.0015 + 0.9 * industry_w / ventilation * wander
    co2 = 415 + 9000 * (traffic_w * traffic + industry_w) / ventilation + rng.normal(0, 2, n)

    def noisy(value, absolute, relative):
        return np.clip(value * (1 + rng.normal(0, relative, n)) + rng.normal(0, absolute, n), -3 * absolute, None)

    growth = humidity_growth(hourly["relative_humidity"])  # what the optical counter adds in humid air
    df = pd.DataFrame({
        "station_id": name,
        "timestamp": ts,
        "pm1": noisy(0.65 * pm25 * growth, 1.0, 0.04).clip(0).round(1),
        "pm25": noisy(pm25 * growth, 2.0, 0.05).clip(0).round(1),
        "pm10": noisy(pm10 * growth, 3.0, 0.05).clip(0).round(1),
        "tsp": noisy(1.3 * pm10 * growth, 4.0, 0.05).clip(0).round(1),
        "o3": noisy(o3, 0.001, 0.02).round(4),
        "no2": np.minimum(noisy(no2, 0.001, 0.03), 0.1).round(4),  # the NO2 cell tops out at 0.1 ppm
        "nox": noisy(no2 * 1.6, 0.001, 0.03).round(4),
        "co": noisy(co, 0.02, 0.03).round(3),
        "so2": noisy(so2, 0.004, 0.05).round(4),
        "h2s": noisy(0.002 + 0.003 * industry_w * wander, 0.006, 0.05).round(4),
        "co2": co2.round(0),
        "voc": noisy(0.05 + 3 * emissions / ventilation, 0.005, 0.05).round(3),
    })
    gaps = rng.random(n) < 0.004
    df.loc[gaps, df.columns[2:]] = np.nan
    return df


def simulate(days: int = 455, seed: int = 71, start: str = "2023-10-01", stations: dict = STATIONS, weather=None):
    """Return (readings, hourly weather). The default period ends in late December, so a held-out test covers the
    post-monsoon burning season and early winter, when Poor air days happen."""
    if weather is None:
        weather = simulate_weather(days, seed, start)
    weather = add_air_weather(weather, seed + 1)
    rng = np.random.default_rng(seed + 2)
    readings = pd.concat([_station(n, t, i, weather, rng) for n, (t, i) in stations.items()], ignore_index=True)
    return readings, weather


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=455)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--out-dir", type=Path, default=HERE / "data")
    args = parser.parse_args(argv)
    readings, weather = simulate(args.days, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    readings.to_csv(args.out_dir / "synthetic_air_readings.csv", index=False)
    weather.to_csv(args.out_dir / "synthetic_weather.csv", index=False)
    print(f"Wrote {len(readings)} air readings and {len(weather)} hours of weather to {args.out_dir}")


if __name__ == "__main__":
    main()
