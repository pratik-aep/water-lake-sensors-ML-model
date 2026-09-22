"""Air sensor checks and corrections: hourly means, stuck and spiking sensors, PM size order, humidity, units."""

import numpy as np
import pandas as pd

from .config import BREAKPOINTS, DETECTION_LIMIT_PPM, GASES, PARTICLE_MAX, PARTICLES, SENSOR_MAX_PPM
from .naqi import ppm_to_mass
from .simulate import humidity_growth

POLLUTANTS = [*PARTICLES, *GASES]


def to_hourly(readings: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Hourly means per station (readings at the start of an hour count toward that hour); too few readings = missing.
    Also marks hours when a reading sat at the top of its sensor's range (<pollutant>_at_limit)."""
    df = readings.assign(timestamp=pd.to_datetime(readings["timestamp"]), hour=lambda d: d["timestamp"].dt.floor("h"))
    columns = [p for p in POLLUTANTS if p in df]
    limits = {**SENSOR_MAX_PPM, **PARTICLE_MAX}
    for p in columns:
        df[f"{p}_at_limit"] = df[p] >= limits[p]
    grouped = df.groupby(["station_id", "hour"])
    means = grouped[columns].mean().where(grouped[columns].count() >= settings["min_readings_per_hour"])
    at_limit = grouped[[f"{p}_at_limit" for p in columns]].any()
    return means.join(at_limit).reset_index().rename(columns={"hour": "timestamp"})


def _stuck(values: pd.Series, hours: int) -> pd.Series:
    same = values.diff().eq(0)
    run = same.groupby((~same).cumsum()).cumsum()
    return run >= hours - 1


def _isolated_spike(values: pd.Series, z_limit: float) -> pd.Series:
    median = values.rolling(25, center=True, min_periods=6).median()
    spread = (values - median).abs().rolling(25, center=True, min_periods=6).median() * 1.4826
    z = (values - median) / spread.clip(lower=1e-9)
    # A real pollution peak lasts hours; a lone reading far above both neighbours is the sensor.
    alone = (z > z_limit) & (z.shift(1).fillna(0) < z_limit / 2) & (z.shift(-1).fillna(0) < z_limit / 2)
    return alone


def check(hourly: pd.DataFrame, weather: pd.DataFrame | None, settings: dict) -> pd.DataFrame:
    """Cleaned hourly values (particles humidity-corrected in ug/m3; index gases in ug/m3, CO in mg/m3; other gases
    in ppm), with a <pollutant>_flag column saying why a value was removed or should be read with care."""
    out = hourly.sort_values(["station_id", "timestamp"]).reset_index(drop=True).copy()
    columns = [p for p in POLLUTANTS if p in out]
    flags = pd.DataFrame("", index=out.index, columns=columns)

    for p in columns:
        if p in DETECTION_LIMIT_PPM:
            floor = 3 * DETECTION_LIMIT_PPM[p]
            too_low = out[p] < -floor
            flags.loc[too_low, p] = "negative"
            out.loc[too_low, p] = np.nan
            out[p] = out[p].clip(lower=0)  # clean-air readings within the sensor's noise
        # A reading at the sensor's ceiling is a real (and high) level, not a fault: kept, as "at least this".
        limit = out.get(f"{p}_at_limit", pd.Series(False, index=out.index)).fillna(False).astype(bool)
        for _, idx in out.groupby("station_id").groups.items():
            values = out.loc[idx, p]
            stuck = _stuck(values, settings["flat_hours"]) & values.notna()
            spike = _isolated_spike(values, settings["spike_z"])
            flags.loc[idx[stuck.to_numpy()], p] = "stuck"
            flags.loc[idx[spike.to_numpy()], p] = "spike"
        flags.loc[limit, p] = "at_sensor_limit"
        out.loc[flags[p].isin(["stuck", "spike"]), p] = np.nan

    particles = [p for p in PARTICLES if p in out]
    if weather is not None and "relative_humidity" in weather:
        humidity = out["timestamp"].map(pd.to_datetime(weather["timestamp"]).pipe(
            lambda t: pd.Series(weather["relative_humidity"].to_numpy(), index=t))).astype(float)
        growth = pd.Series(humidity_growth(humidity, settings["hygroscopic_kappa"]), index=out.index)
        foggy = humidity > settings["humidity_trust_limit"]
        for p in particles:
            out[p] = out[p] / growth.fillna(1.0)
            flags.loc[foggy & out[p].notna(), p] = "too_humid"
            out.loc[foggy, p] = np.nan

    absolute, share = settings["pm_order_tolerance"]
    for small, large in zip(particles, particles[1:]):
        wrong = out[small] > out[large] + absolute + share * out[large]
        for p in (small, large):
            flags.loc[wrong & flags[p].eq(""), p] = "size_order"

    for gas in BREAKPOINTS:
        if gas in out and gas in DETECTION_LIMIT_PPM:
            out[gas] = ppm_to_mass(out[gas], gas)
    for p in columns:
        out[f"{p}_flag"] = flags[p]
    return out.drop(columns=[c for c in out if c.endswith("_at_limit")])
