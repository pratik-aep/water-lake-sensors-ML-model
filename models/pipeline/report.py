"""The daily per-lake report and its prioritised alerts."""

import pandas as pd

from ..anomaly_detection.config import SENSORS

SEVERITY = ["high", "medium", "low"]


def _time(value) -> str:
    return pd.Timestamp(value).isoformat(timespec="minutes")


def _round(value, digits=2):
    return None if pd.isna(value) else round(float(value), digits)


def station_report(station, flagged, incidents, forecast_hourly, nightly, bod, wqi_now, settings, algae=None,
                   coliform=None) -> tuple[dict, list]:
    """One lake's section of the report, plus the alerts it raises."""
    alerts = []

    def alert(severity, message):
        alerts.append({"severity": severity, "station": station, "message": message})

    rows = flagged[flagged["station_id"] == station]
    latest = rows.dropna(subset=SENSORS, how="all").iloc[-1] if rows[SENSORS].notna().any().any() else None
    section = {"latest": None if latest is None else {"time": _time(latest["timestamp"]), **{s: _round(latest[s], 3) for s in SENSORS}}}
    if latest is None or pd.Timestamp(latest["timestamp"]) < rows["timestamp"].max() - pd.Timedelta(hours=settings["stale_hours"]):
        alert("medium", "No recent sensor data: check power and the GSM link")

    section["sensor_health"] = {
        s: {
            "faulty_readings": int(rows[f"{s}_fault"].ne("").sum()),
            "fault_types": sorted(set(rows[f"{s}_fault"]) - {""}),
        }
        for s in SENSORS
    }
    section["incidents"] = []
    for inc in incidents[incidents["station_id"] == station].itertuples(index=False):
        span = f"{_time(inc.start)} to {_time(inc.end)}"
        section["incidents"].append({"start": _time(inc.start), "end": _time(inc.end), "kinds": inc.verdicts, "sensors": inc.sensors})
        if "possible_event" in inc.verdicts:
            alert("high", f"Possible pollution event {span}: several sensors changed together")
        if "sensor_fault" in inc.verdicts:
            alert("medium", f"Sensor fault on {inc.sensors or 'a sensor'} {span}: check or clean it")
        if inc.verdicts == "needs_review":
            alert("low", f"Unusual readings {span}: worth a look")

    now = {}
    if wqi_now is not None:
        now["wqi_class"] = wqi_now["predicted_class"]
        now["wqi_probability"] = _round(wqi_now[f"p_{wqi_now['predicted_class']}"])
        if wqi_now["predicted_class"] == "Unsuitable":
            alert("medium", "Water quality index is Unsuitable at the latest reading")
    if bod is not None:
        now.update({
            "date": _time(bod["timestamp"]),
            "bod_mg_l": _round(bod["bod"]),
            "bod_range_mg_l": [_round(bod["bod_lo"]), _round(bod["bod_hi"])],
            "p_bod_above_3": _round(bod["p_above_3"]),
            "cpcb_class": bod["cpcb_class"],
            "cpcb_not_assessed": bod["cpcb_not_assessed"],
        })
        if bod["p_above_3"] >= settings["alert_probability"]:
            alert("medium", f"BOD likely above the CPCB Class B/C limit of 3 mg/L: {bod['bod']:.1f} mg/L "
                            f"({bod['bod_lo']:.1f}-{bod['bod_hi']:.1f})")
    section["now"] = now

    if algae is not None:
        section["algae"] = {
            "date": _time(algae["timestamp"]),
            "chlorophyll_a_ug_l": _round(algae["chlorophyll_a"], 1),
            "range_ug_l": [_round(algae["chlorophyll_a_lo"], 1), _round(algae["chlorophyll_a_hi"], 1)],
            "trophic_state_index": _round(algae["tsi"], 1),
            "trophic_class": algae["trophic_class"],
            "p_above_12": _round(algae["p_above_12"]),
            "p_above_24": _round(algae["p_above_24"]),
            "who_level": algae["who_level"],
        }
        if algae["p_above_24"] >= settings["alert_probability"]:
            alert("high", f"Possible algal bloom: chlorophyll-a likely above 24 ug/L ({algae['chlorophyll_a']:.0f}); "
                          "look for scum and confirm cyanobacteria with a microscope count")
        elif algae["p_above_12"] >= settings["alert_probability"]:
            alert("medium", f"Chlorophyll-a likely above the WHO Alert Level 1 threshold of 12 ug/L ({algae['chlorophyll_a']:.0f})")

    if coliform is not None:
        section["coliform"] = {
            "date": _time(coliform["timestamp"]),
            "total_coliform_mpn_100ml": _round(coliform["total_coliform"], 0),
            "range_mpn_100ml": [_round(coliform["total_coliform_lo"], 0), _round(coliform["total_coliform_hi"], 0)],
            "p_above_500": _round(coliform["p_above_500"]),
            "p_above_5000": _round(coliform["p_above_5000"]),
            "cpcb_band": coliform["cpcb_band"],
        }
        if coliform["p_above_5000"] >= settings["alert_probability"]:
            alert("high", f"Total coliform likely above 5000 MPN/100 mL ({coliform['total_coliform']:.0f}), the CPCB "
                          "Class C limit: likely sewage or runoff; take a lab sample and warn water users")
        elif coliform["p_above_500"] >= settings["alert_probability"]:
            alert("medium", f"Total coliform likely above 500 MPN/100 mL ({coliform['total_coliform']:.0f}), "
                            "the CPCB bathing (Class B) limit")

    fc = forecast_hourly[forecast_hourly["station_id"] == station]
    outlook = {}
    if len(fc):
        oxygen = fc[fc["sensor"] == "dissolved_oxygen"]
        low = oxygen.loc[oxygen["mid"].idxmin()]
        murky = fc[fc["sensor"] == "turbidity"]
        high = murky.loc[murky["mid"].idxmax()]
        outlook = {
            "issued_at": _time(fc["origin"].iloc[0]),
            "lowest_oxygen": {"mg_l": _round(low["mid"]), "time": _time(low["target_time"]), "range": [_round(low["lo"]), _round(low["hi"])]},
            "highest_turbidity": {"value": _round(high["mid"], 1), "unit": settings["turbidity_unit"], "time": _time(high["target_time"])},
        }
        if "wqi_class" in fc:
            outlook["wqi_classes_next_48h"] = fc.drop_duplicates("target_time")["wqi_class"].value_counts().to_dict()
    section["outlook_48h"] = outlook

    section["nights"] = []
    for n in nightly[nightly["station_id"] == station].itertuples(index=False):
        flagged_night = n.p_below_alert >= settings["alert_probability"]
        section["nights"].append({
            "date": str(pd.Timestamp(n.night_of).date()),
            "oxygen_min_mg_l": _round(n.mid),
            "range": [_round(n.lo), _round(n.hi)],
            "p_below_alert": _round(n.p_below_alert),
            "alert": bool(flagged_night),
        })
        if flagged_night:
            alert("high" if n.days_ahead <= 2 else "medium",
                  f"Oxygen may fall below {settings['do_alert_mg_l']:g} mg/L before dawn on {pd.Timestamp(n.night_of).date()} "
                  f"(chance {n.p_below_alert:.0%})")
    section["alerts"] = alerts
    return section, alerts


AIR_SEVERITY = {"Poor": "medium", "Very Poor": "high", "Severe": "high"}
GAS_NAMES = {"no2": "NO2", "o3": "O3", "co": "CO", "so2": "SO2", "nox": "NOx", "h2s": "H2S", "co2": "CO2", "voc": "VOC",
             "pm1": "PM1", "pm25": "PM2.5", "pm10": "PM10", "tsp": "TSP"}


def air_report(station, current, flags: dict, daily: pd.DataFrame, settings) -> tuple[dict, list]:
    """One air station's section: the AQI now, sensor checks, and the coming days' particulate outlook."""
    alerts = []

    def alert(severity, message):
        alerts.append({"severity": severity, "station": station, "message": message})

    section = {"flags_last_24h": flags}
    if current is not None:
        subs = {p.removeprefix("si_"): _round(current[p], 0) for p in current.index if p.startswith("si_") and not pd.isna(current[p])}
        section["now"] = {"time": _time(current["timestamp"]), "aqi": _round(current["aqi"], 0), "category": current["category"],
                          "prominent_pollutant": GAS_NAMES.get(current["prominent_pollutant"], current["prominent_pollutant"]),
                          "sub_indices": subs, "advice": current["advice"]}
        if current["category"] in AIR_SEVERITY:
            alert(AIR_SEVERITY[current["category"]], f"Air quality {current['category']} (AQI {current['aqi']:.0f}, mainly "
                  f"{section['now']['prominent_pollutant']}): {current['advice']}")
    saturated = sorted(GAS_NAMES[p] for p, kinds in flags.items() if "at_sensor_limit" in kinds)
    if saturated:
        alert("low", f"{', '.join(saturated)} reached the top of the sensor's range: the true level was higher than reported")
    section["days"] = []
    for d in daily.itertuples(index=False):
        section["days"].append({"date": str(pd.Timestamp(d.date).date()), "pm25_ug_m3": _round(d.pm25, 0),
                                "pm25_range": [_round(d.pm25_lo, 0), _round(d.pm25_hi, 0)], "pm_index": _round(d.pm_index, 0),
                                "category": d.pm_category, "p_poor_or_worse": _round(d.p_poor_or_worse)})
        if d.p_poor_or_worse >= settings["alert_probability"]:
            alert("medium", f"Air likely Poor or worse on {pd.Timestamp(d.date).date()} (chance {d.p_poor_or_worse:.0%}; "
                            f"PM2.5 around {d.pm25:.0f} ug/m3): consider advisories for outdoor activity")
    section["alerts"] = alerts
    return section, alerts


def render_text(report: dict) -> str:
    lines = [f"Lake water quality report, {report['generated_at']} (incidents from the last {report['window_hours']} h)", ""]
    if report["alerts"]:
        lines.append("ALERTS")
        lines += [f"  [{a['severity'].upper():<6}] {a['station']}: {a['message']}" for a in report["alerts"]]
    else:
        lines.append("No alerts.")
    for station, s in report["stations"].items():
        lines += ["", f"=== {station} ==="]
        if s["latest"]:
            readings = ", ".join(f"{k} {v}" for k, v in s["latest"].items() if k != "time")
            lines.append(f"Latest ({s['latest']['time']}): {readings}")
        faults = {k: v["faulty_readings"] for k, v in s["sensor_health"].items() if v["faulty_readings"]}
        lines.append(f"Sensor health: {'all good' if not faults else 'faulty readings ' + str(faults)}")
        now = s["now"]
        if now:
            lines.append(f"Now: WQI {now.get('wqi_class')}; BOD {now.get('bod_mg_l')} mg/L {now.get('bod_range_mg_l')}; "
                         f"CPCB class {now.get('cpcb_class')} (not assessed: {now.get('cpcb_not_assessed') or '-'})")
        if s.get("algae"):
            a = s["algae"]
            lines.append(f"Algae: chlorophyll-a {a['chlorophyll_a_ug_l']} ug/L {a['range_ug_l']}, TSI {a['trophic_state_index']} "
                         f"({a['trophic_class']}), WHO: {a['who_level']}")
        if s.get("coliform"):
            c = s["coliform"]
            lines.append(f"Total coliform: {c['total_coliform_mpn_100ml']:.0f} MPN/100 mL {c['range_mpn_100ml']}, "
                         f"P(>500) {c['p_above_500']:.0%}, CPCB band {c['cpcb_band']}")
        if s["outlook_48h"]:
            o = s["outlook_48h"]
            lines.append(f"Next 48 h: lowest oxygen {o['lowest_oxygen']['mg_l']} mg/L at {o['lowest_oxygen']['time']}, "
                         f"turbidity peak {o['highest_turbidity']['value']} {o['highest_turbidity']['unit']}")
        if s["nights"]:
            lines.append("Pre-dawn oxygen minimum: " + "; ".join(
                f"{n['date']} {n['oxygen_min_mg_l']} ({n['p_below_alert']:.0%}){' !' if n['alert'] else ''}" for n in s["nights"]
            ))
    for station, s in report.get("air", {}).items():
        lines += ["", f"=== {station} (air) ==="]
        if s.get("now"):
            n = s["now"]
            lines.append(f"AQI {n['aqi']:.0f} {n['category']} at {n['time']} (mainly {n['prominent_pollutant']}): {n['advice']}")
        if s["flags_last_24h"]:
            lines.append(f"Sensor checks, last 24 h: {s['flags_last_24h']}")
        if s["days"]:
            lines.append("Coming days (particulate index): " + "; ".join(
                f"{d['date']} {d['category']} ({d['p_poor_or_worse']:.0%} chance Poor+)" for d in s["days"]))
    return "\n".join(lines)
