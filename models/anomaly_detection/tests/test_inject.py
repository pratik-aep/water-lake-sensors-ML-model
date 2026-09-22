import numpy as np
import pandas as pd

from models.anomaly_detection.config import SENSORS
from models.anomaly_detection.inject import KINDS, apply_episode, inject
from models.anomaly_detection.simulate import simulate
from models.wqi.generate_data import STATIONS

TWO_STATIONS = {k: STATIONS[k] for k in ("fateh_sagar", "goverdhan_sagar")}


def test_every_kind_is_injected_and_labelled_and_nothing_else_changes():
    clean = simulate(days=90, seed=1, stations=TWO_STATIONS)
    dirty, episodes = inject(clean, seed=3, per_kind=1)

    assert set(episodes["kind"]) == set(KINDS)
    assert episodes.groupby("station_id").size().eq(len(KINDS)).all()

    clean = clean.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    inside = pd.Series(False, index=dirty.index)
    for ep in episodes.itertuples():
        inside |= (dirty["station_id"] == ep.station_id) & dirty["timestamp"].between(ep.start, ep.end)
    assert dirty.loc[~inside, SENSORS].equals(clean.loc[~inside, SENSORS])
    assert not dirty.loc[inside, SENSORS].equals(clean.loc[inside, SENSORS])


def test_injection_is_reproducible():
    clean = simulate(days=60, seed=1, stations=TWO_STATIONS)
    a, ea = inject(clean, seed=9, per_kind=1)
    b, eb = inject(clean, seed=9, per_kind=1)
    assert a.equals(b) and ea.equals(eb)


def test_episode_shapes():
    df = pd.DataFrame({s: np.linspace(1.0, 2.0, 20) for s in SENSORS})
    rows = df.index[5:15]

    flat = df.copy()
    apply_episode(flat, rows, "flatline", "ph")
    assert flat.loc[rows, "ph"].nunique() == 1

    gone = df.copy()
    apply_episode(gone, rows, "dropout")
    assert gone.loc[rows, SENSORS].isna().all().all()

    drift = df.copy()
    apply_episode(drift, rows, "drift", "dissolved_oxygen", size=-1.0)
    added = drift.loc[rows, "dissolved_oxygen"] - df.loc[rows, "dissolved_oxygen"]
    assert added.iloc[0] == 0 and added.is_monotonic_decreasing and added.iloc[-1] < -1.5

    event = df.copy()
    apply_episode(event, rows, "event", size=1.0)
    peak = rows[4]
    assert event.at[peak, "turbidity"] > df.at[peak, "turbidity"]
    assert event.at[peak, "dissolved_oxygen"] < df.at[peak, "dissolved_oxygen"]
    assert event.at[peak, "ph"] > df.at[peak, "ph"]
