"""Score detector output per episode (not per reading) against known faults and events."""

import pandas as pd

from .config import SENSORS

ALARM_VERDICTS = ["sensor_fault", "possible_event", "needs_review"]
DETECTION_GRACE = pd.Timedelta(hours=1)
# Drift alarms take a few hours to settle after the sensor is cleaned; alarms that soon belong to the episode.
AFTERMATH = pd.Timedelta(hours=6)


def alarm_incidents(result: pd.DataFrame, merge_gap: pd.Timedelta) -> pd.DataFrame:
    """Group alarm rows into incidents an operator would see, merging alarms closer than merge_gap."""
    rows = []
    alarms = result[result["verdict"].isin(ALARM_VERDICTS)]
    for station, g in alarms.groupby("station_id", sort=False):
        incident = (g["timestamp"].diff() > merge_gap).cumsum()
        for _, part in g.groupby(incident):
            rows.append(
                {
                    "station_id": station,
                    "start": part["timestamp"].iloc[0],
                    "end": part["timestamp"].iloc[-1],
                    "verdicts": ",".join(sorted(part["verdict"].unique())),
                    "sensors": ",".join(s for s in SENSORS if part[f"{s}_fault"].ne("").any()),
                }
            )
    return pd.DataFrame(rows, columns=["station_id", "start", "end", "verdicts", "sensors"])


def _station_weeks(result: pd.DataFrame) -> float:
    spans = result.groupby("station_id")["timestamp"].agg(lambda t: t.max() - t.min())
    return float(spans.sum() / pd.Timedelta(days=7))


def consistency_coverage(result: pd.DataFrame) -> dict:
    """Share of each sensor's readings on which the cross-sensor check ran (it pauses in unseen conditions)."""
    return {s: round(float(result[f"{s}_z"].notna().sum() / max(result[s].notna().sum(), 1)), 3) for s in SENSORS}


def evaluate(result: pd.DataFrame, episodes: pd.DataFrame, merge_gap: pd.Timedelta) -> dict:
    episodes = episodes.assign(start=pd.to_datetime(episodes["start"]), end=pd.to_datetime(episodes["end"]))
    fault_cols = [f"{s}_fault" for s in SENSORS]
    per_episode = []
    for ep in episodes.itertuples(index=False):
        # Only flags on the episode's own readings count: the jump back when a drifting sensor is cleaned is
        # easy to flag, but by then the fault has already gone unnoticed for its whole length.
        window = result[(result["station_id"] == ep.station_id) & result["timestamp"].between(ep.start, ep.end)]
        if ep.kind == "event":
            hit = window["event"]
        elif ep.kind == "dropout":
            hit = window["verdict"].eq("missing")
        else:
            hit = window[f"{ep.sensor}_fault"].ne("")
        detected = bool(hit.any())
        first = window.loc[hit, "timestamp"].min() if detected else None
        per_episode.append(
            {
                **ep._asdict(),
                "detected": detected,
                "delay_hours": round((first - ep.start) / pd.Timedelta(hours=1), 2) if detected else None,
                "blamed_on_sensor": ep.kind == "event" and not detected and bool(window[fault_cols].ne("").any().any()),
            }
        )

    table = pd.DataFrame(per_episode, columns=[*episodes.columns, "detected", "delay_hours", "blamed_on_sensor"])
    per_kind = {}
    for kind, g in table.groupby("kind", sort=False):
        per_kind[kind] = {
            "episodes": len(g),
            "detected": int(g["detected"].sum()),
            "recall": round(float(g["detected"].mean()), 3),
            "median_delay_hours": None if not g["detected"].any() else float(g["delay_hours"].median()),
        }

    incidents = alarm_incidents(result, merge_gap)
    explained = [
        bool(
            (
                (episodes["station_id"] == inc.station_id)
                & (episodes["start"] - DETECTION_GRACE <= inc.end)
                & (episodes["end"] + AFTERMATH >= inc.start)
            ).any()
        )
        for inc in incidents.itertuples(index=False)
    ]
    false_incidents = len(incidents) - sum(explained)
    return {
        "per_kind": per_kind,
        "per_episode": per_episode,
        "incidents": len(incidents),
        "false_incidents": false_incidents,
        "false_alarms_per_station_week": round(false_incidents / _station_weeks(result), 3),
        "incident_precision": round(sum(explained) / len(incidents), 3) if len(incidents) else None,
        "events_blamed_on_a_sensor": int(table["blamed_on_sensor"].sum()),
        "consistency_coverage": consistency_coverage(result),
    }
