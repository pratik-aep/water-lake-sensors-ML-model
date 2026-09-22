"""Lake metabolism from the daily oxygen curve (the diel oxygen technique)."""

import pandas as pd

from ..wqi.calculator import do_saturation_mgl


def hourly_changes(hourly: pd.DataFrame, pressure_atm: float) -> pd.DataFrame:
    """Each hour-to-hour step: oxygen change (mg/L/h), mean saturation deficit and temperature over the step."""
    frame = hourly.set_index("timestamp")[["temperature", "dissolved_oxygen"]]
    deficit = do_saturation_mgl(frame["temperature"], pressure_atm) - frame["dissolved_oxygen"]
    steps = pd.DataFrame(
        {
            "change": frame["dissolved_oxygen"].shift(-1) - frame["dissolved_oxygen"],
            "deficit": (deficit + deficit.shift(-1)) / 2,
            "temperature": (frame["temperature"] + frame["temperature"].shift(-1)) / 2,
        },
        index=frame.index,
    )
    return steps.dropna()


def daily_metabolism(hourly: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Per date: respiration R and reaeration k from the night ending that morning, then that day's production."""
    # Night-time regression (Odum 1956; Hornberger & Kelly 1975): in darkness dDO/dt = -R + k * deficit.
    s = settings
    steps = hourly_changes(hourly, s["pressure_atm"])
    hour = steps.index.hour
    night_start, night_end = s["night_hours"]
    dark = steps[(hour >= night_start) | (hour < night_end)]
    # A night is labelled by the date it ends on: 21:00-04:59 belongs to the following morning.
    night_of = (dark.index + pd.Timedelta(hours=24 - night_start)).normalize()

    grouped = dark.assign(dd=dark["deficit"] ** 2, dc=dark["deficit"] * dark["change"]).groupby(night_of)
    sums = pd.DataFrame(
        {
            "n": grouped.size(),
            "d": grouped["deficit"].sum(),
            "c": grouped["change"].sum(),
            "dd": grouped["dd"].sum(),
            "dc": grouped["dc"].sum(),
        }
    )
    sums = sums[sums["n"] >= s["min_night_points"]]
    # Slope from within-night variation only (each night gets its own intercept): nights with more respiration
    # also have bigger deficits, and pooling them naively would drag the reaeration estimate toward zero.
    within = pd.DataFrame(
        {"dd": sums["dd"] - sums["d"] ** 2 / sums["n"], "dc": sums["dc"] - sums["d"] * sums["c"] / sums["n"]}
    )
    pooled = within.rolling(s["reaeration_pool_nights"], min_periods=1).sum()
    k = (pooled["dc"] / pooled["dd"].where(pooled["dd"] > 1e-9)).where(lambda v: v > 0)
    # Where the pooled fit is not physical, fall back to the running median of earlier estimates (no look-ahead).
    k = k.fillna(k.expanding().median().shift()).clip(upper=0.5)
    nights = pd.DataFrame({"reaeration": k})
    nights["respiration"] = (nights["reaeration"] * sums["d"] - sums["c"]) / sums["n"]
    nights["night_temperature"] = dark.groupby(night_of)["temperature"].mean()

    day_start, day_end = s["day_hours"]
    light = steps[(hour >= day_start) & (hour < day_end)]
    date = light.index.normalize()
    joined = light.join(nights[["reaeration", "respiration"]], on=date)
    production = joined["change"] + joined["respiration"] - joined["reaeration"] * joined["deficit"]
    by_day = production.groupby(date)
    enough = by_day.count() >= 0.7 * (day_end - day_start)
    days = pd.DataFrame({"gpp": by_day.sum().clip(lower=0).where(enough)})

    oxygen = hourly.set_index("timestamp")["dissolved_oxygen"]
    days["do_amplitude"] = oxygen.groupby(oxygen.index.normalize()).agg(lambda x: x.max() - x.min())
    table = nights.join(days, how="outer")
    table["nep"] = table["gpp"] - 24 * table["respiration"]
    return table.sort_index()
