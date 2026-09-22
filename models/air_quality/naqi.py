"""India's National Air Quality Index from hourly pollutant data (CPCB method)."""

import numpy as np
import pandas as pd

from .config import (
    BREAKPOINTS,
    CATEGORIES,
    EIGHT_HOUR,
    INDEX_EDGES,
    MIN_HOURS_8H,
    MIN_HOURS_24H,
    MIN_POLLUTANTS,
    MOLECULAR_WEIGHT,
)

MOLAR_VOLUME_25C = 24.45  # litres per mole at 25 C and 1 atm, the reference conditions for reporting


def ppm_to_mass(ppm, gas: str):
    """ppm to ug/m3 (mg/m3 for CO, the unit its breakpoints use), at 25 C and 1 atm."""
    ugm3 = np.asarray(ppm, dtype=float) * MOLECULAR_WEIGHT[gas] * 1000 / MOLAR_VOLUME_25C
    return ugm3 / 1000 if gas == "co" else ugm3


def sub_index(concentration, pollutant: str) -> np.ndarray:
    """Linear interpolation within the breakpoint band the concentration falls in; 500 beyond the last band."""
    c = np.asarray(concentration, dtype=float)
    return np.where(np.isnan(c), np.nan, np.interp(np.clip(c, 0, None), BREAKPOINTS[pollutant], INDEX_EDGES))


def category(index) -> np.ndarray:
    """Category of a (whole-number) index: Good up to 50, Satisfactory 51-100, and so on."""
    index = np.round(np.asarray(index, dtype=float))
    names = np.array([name for _, name, _ in CATEGORIES], dtype=object)
    position = np.searchsorted([upper for upper, _, _ in CATEGORIES], np.nan_to_num(index), side="left")
    return np.where(np.isnan(index), "insufficient data", names[position.clip(0, len(names) - 1)])


def advice(name: str) -> str:
    return {label: text for _, label, text in CATEGORIES}.get(name, "")


def averaged(hourly: pd.DataFrame) -> pd.DataFrame:
    """Per station and hour, each pollutant averaged the way its index is defined, ending at that hour."""
    out = []
    for station, g in hourly.groupby("station_id", sort=True):
        g = g.set_index("timestamp").sort_index().asfreq("h")
        frame = pd.DataFrame(index=g.index)
        for p in BREAKPOINTS:
            if p not in g:
                continue
            if p in EIGHT_HOUR:
                eight = g[p].rolling(8, min_periods=MIN_HOURS_8H).mean()
                frame[p] = eight.rolling(24, min_periods=1).max()  # the day's worst 8-hour stretch
            else:
                frame[p] = g[p].rolling(24, min_periods=MIN_HOURS_24H).mean()
        out.append(frame.assign(station_id=station).reset_index())
    return pd.concat(out, ignore_index=True)


def naqi(concentrations: pd.DataFrame) -> pd.DataFrame:
    """Index from averaged concentrations (ug/m3; CO mg/m3): sub-indices, the index, its category and the
    pollutant setting it. Needs three pollutants, one of them particulate, else 'insufficient data'."""
    pollutants = [p for p in BREAKPOINTS if p in concentrations]
    subs = pd.DataFrame({p: sub_index(concentrations[p], p) for p in pollutants}, index=concentrations.index)
    enough = (subs.notna().sum(axis=1) >= MIN_POLLUTANTS) & subs[[p for p in ("pm25", "pm10") if p in subs]].notna().any(axis=1)
    index = subs.max(axis=1).where(enough)
    prominent = subs.fillna(-1).idxmax(axis=1).where(enough, "")
    result = concentrations.copy()
    for p in pollutants:
        result[f"si_{p}"] = subs[p].round(0)
    result["aqi"] = index.round(0)
    result["category"] = category(index)
    result["prominent_pollutant"] = prominent
    return result
