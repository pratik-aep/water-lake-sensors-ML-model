"""A self-contained HTML dashboard for one report: alerts first, then a card per lake with its forecasts."""

from html import escape

import numpy as np
import pandas as pd

from .report import GAS_NAMES

CSS = """
:root { --bg:#f6f7f9; --card:#fff; --ink:#1d2330; --muted:#5d6675; --line:#e2e5ea; --accent:#1f6feb;
  --band:rgba(31,111,235,.16); --high:#c62828; --medium:#b26a00; --low:#51606f; --ok:#2e7d32; }
@media (prefers-color-scheme: dark) { :root { --bg:#0f1318; --card:#171c23; --ink:#e6e9ee; --muted:#9aa4b2;
  --line:#2a313b; --accent:#6ea8ff; --band:rgba(110,168,255,.2); --high:#ff6b6b; --medium:#f0a93b; --low:#9aa4b2; --ok:#5cc26a; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width:1180px; margin:0 auto; padding:24px 16px 48px; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:17px; margin:0; } h3 { font-size:13px; margin:14px 0 6px; color:var(--muted);
  text-transform:uppercase; letter-spacing:.04em; }
.sub { color:var(--muted); margin:0 0 20px; }
.panel, .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; }
.alerts { margin-bottom:20px; } .alerts ul { list-style:none; margin:8px 0 0; padding:0; }
.alerts li { padding:7px 0; border-top:1px solid var(--line); display:flex; gap:10px; align-items:baseline; }
.chip { font-size:11px; font-weight:700; padding:2px 7px; border-radius:99px; color:#fff; flex:none; text-transform:uppercase; }
.chip.high { background:var(--high); } .chip.medium { background:var(--medium); } .chip.low { background:var(--low); }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(340px,1fr)); gap:16px; }
.head { display:flex; justify-content:space-between; align-items:baseline; gap:8px; }
.badge { font-size:12px; color:var(--muted); }
dl { display:grid; grid-template-columns:auto 1fr; gap:3px 12px; margin:0; font-size:14px; }
dt { color:var(--muted); } dd { margin:0; font-variant-numeric:tabular-nums; }
svg { width:100%; height:auto; display:block; } svg text { fill:var(--muted); font-size:10px; }
.good { color:var(--ok); } .bad { color:var(--high); }
footer { color:var(--muted); font-size:13px; margin-top:24px; }
.section-title { margin:28px 0 12px; }
.aqi { display:flex; align-items:center; gap:12px; margin:10px 0 4px; }
.aqi .num { font-size:30px; font-weight:700; line-height:1; padding:6px 10px; border-radius:8px; color:#fff; font-variant-numeric:tabular-nums; }
.cat-Good { background:#00893a; } .cat-Satisfactory { background:#6aa84f; } .cat-Moderate { background:#c9a200; }
.cat-Poor { background:#e07b00; } .cat-Very-Poor { background:#cc0000; } .cat-Severe { background:#7e0023; }
"""


def _fmt(value, digits=1, unit=""):
    return "–" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:.{digits}f}{unit}"


def band_chart(times, lo, mid, hi, threshold=None, width=320, height=120, digits=1, time_format="%d %b %H:%M") -> str:
    """Median line inside its 80% band, with a dashed alert line; empty string when there is nothing to draw."""
    times, lo, mid, hi = pd.to_datetime(pd.Series(times)), np.asarray(lo, float), np.asarray(mid, float), np.asarray(hi, float)
    if not len(times) or np.isnan(mid).all():
        return ""
    pad_l, pad_r, pad_t, pad_b = 34, 6, 8, 18
    marks = [] if threshold is None else [threshold]
    data_top, data_bottom = np.nanmax([*hi, *marks]), np.nanmin([*lo, *marks])
    span = (data_top - data_bottom) or 1.0
    top, bottom = data_top + 0.08 * span, data_bottom - 0.08 * span
    t0, t1 = times.iloc[0], times.iloc[-1]
    total = (t1 - t0) / pd.Timedelta(hours=1) or 1.0

    def x(t):
        return pad_l + (width - pad_l - pad_r) * ((t - t0) / pd.Timedelta(hours=1)) / total

    def y(v):
        return pad_t + (height - pad_t - pad_b) * (top - v) / (top - bottom)

    xs = [x(t) for t in times]
    band = " ".join(f"{a:.1f},{y(b):.1f}" for a, b in zip(xs, hi)) + " " + " ".join(
        f"{a:.1f},{y(b):.1f}" for a, b in zip(reversed(xs), reversed(lo)))
    line = " ".join(f"{a:.1f},{y(b):.1f}" for a, b in zip(xs, mid))
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img">',
             f'<polygon points="{band}" fill="var(--band)"/>',
             f'<polyline points="{line}" fill="none" stroke="var(--accent)" stroke-width="2"/>']
    if threshold is not None:
        parts.append(f'<line x1="{pad_l}" x2="{width - pad_r}" y1="{y(threshold):.1f}" y2="{y(threshold):.1f}" '
                     f'stroke="var(--high)" stroke-dasharray="4 3"/>')
    for v in sorted({data_top, data_bottom, *marks}):
        parts.append(f'<text x="{pad_l - 4}" y="{y(v) + 3:.1f}" text-anchor="end">{v:.{digits}f}</text>')
    parts.append(f'<text x="{pad_l}" y="{height - 4}">{t0.strftime(time_format)}</text>')
    parts.append(f'<text x="{width - pad_r}" y="{height - 4}" text-anchor="end">{t1.strftime(time_format)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _lake_card(station: str, s: dict, hourly: pd.DataFrame, nightly: pd.DataFrame, settings: dict) -> str:
    latest, now = s.get("latest") or {}, s.get("now") or {}
    faults = {k: v["faulty_readings"] for k, v in s["sensor_health"].items() if v["faulty_readings"]}
    unit = settings["turbidity_unit"]
    rows = [
        ("Latest reading", escape(latest.get("time", "–"))),
        ("Water temperature", _fmt(latest.get("temperature"), 1, " °C")),
        ("pH", _fmt(latest.get("ph"), 2)),
        ("Turbidity", _fmt(latest.get("turbidity"), 1, f" {unit}")),
        ("Dissolved oxygen", _fmt(latest.get("dissolved_oxygen"), 2, " mg/L")),
        ("Sensor health", '<span class="good">all good</span>' if not faults else
         f'<span class="bad">faulty readings: {escape(", ".join(f"{k} {v}" for k, v in faults.items()))}</span>'),
    ]
    estimates = [
        ("WQI class", escape(str(now.get("wqi_class", "–")))),
        ("CPCB class", escape(str(now.get("cpcb_class", "–"))) + (f' <span class="badge">(not assessed: {escape(now["cpcb_not_assessed"])})</span>' if now.get("cpcb_not_assessed") else "")),
        ("BOD", f'{_fmt(now.get("bod_mg_l"), 1)} mg/L <span class="badge">{_fmt((now.get("bod_range_mg_l") or [None])[0], 1)}–{_fmt((now.get("bod_range_mg_l") or [None, None])[1], 1)}</span>'),
    ]
    if s.get("algae"):
        a = s["algae"]
        estimates.append(("Chlorophyll-a", f'{_fmt(a["chlorophyll_a_ug_l"], 1)} µg/L <span class="badge">{escape(a["trophic_class"])}; WHO {escape(a["who_level"])}</span>'))
    if s.get("coliform"):
        c = s["coliform"]
        estimates.append(("Total coliform", f'{_fmt(c["total_coliform_mpn_100ml"], 0)} MPN/100 mL <span class="badge">band {escape(c["cpcb_band"])}</span>'))

    oxygen = hourly[(hourly["station_id"] == station) & (hourly["sensor"] == "dissolved_oxygen")].sort_values("target_time") if len(hourly) else hourly
    nights = nightly[nightly["station_id"] == station].sort_values("night_of") if len(nightly) else nightly
    alert_level = settings["do_alert_mg_l"]
    html = [f'<section class="card"><div class="head"><h2>{escape(station.replace("_", " ").title())}</h2>'
            f'<span class="badge">{len(s["alerts"])} alert(s)</span></div>',
            "<h3>Now</h3><dl>" + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows) + "</dl>",
            "<h3>Estimated from the sensors</h3><dl>" + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in estimates) + "</dl>"
            if now or s.get("algae") or s.get("coliform") else ""]
    if len(oxygen):
        html.append("<h3>Oxygen, next 48 h (mg/L, 80% range)</h3>")
        html.append(band_chart(oxygen["target_time"], oxygen["lo"], oxygen["mid"], oxygen["hi"], alert_level))
    if len(nights):
        html.append("<h3>Lowest oxygen before dawn, coming nights</h3>")
        html.append(band_chart(nights["night_of"], nights["lo"], nights["mid"], nights["hi"], alert_level, time_format="%d %b"))
    html.append("</section>")
    return "".join(html)


def render(report: dict, hourly: pd.DataFrame, nightly: pd.DataFrame, settings: dict, extra_sections: str = "") -> str:
    alerts = report["alerts"]
    alert_items = "".join(
        f'<li><span class="chip {a["severity"]}">{a["severity"]}</span><span><b>{escape(a["station"])}</b>: {escape(a["message"])}</span></li>'
        for a in alerts
    ) or '<li><span class="good">No alerts.</span></li>'
    cards = "".join(_lake_card(st, s, hourly, nightly, settings) for st, s in report["stations"].items())
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Udaipur Lakes Dashboard</title><style>{CSS}</style></head>
<body><main><h1>Udaipur lakes{" and air" if extra_sections else ""}: daily status</h1>
<p class="sub">Report {escape(report["generated_at"])} · models trained {escape(str(report["models_trained_at"])[:10])}</p>
<section class="panel alerts"><h2>Alerts ({len(alerts)})</h2><ul>{alert_items}</ul></section>
<div class="grid">{cards}</div>{extra_sections}
<footer>BOD, algae and coliform values are model estimates from the sensors, not lab results: confirm anything
that matters with a lab sample. Ranges show where the true value falls 8 times out of 10.</footer></main></body></html>"""


NATIONAL_PM25_24H = 60  # ug/m3, India's 24-hour ambient standard (NAAQS 2009)


def air_cards(sections: dict, hourly: pd.DataFrame) -> str:
    """Cards for the air stations: the index now, its main pollutant, the coming days and a PM2.5 chart."""
    if not sections:
        return ""
    cards = []
    for station, s in sections.items():
        now = s.get("now") or {}
        html = [f'<section class="card"><div class="head"><h2>{escape(station.replace("_", " ").title())}</h2>'
                f'<span class="badge">{len(s["alerts"])} alert(s)</span></div>']
        if now:
            level = escape(now["category"].replace(" ", "-"))
            html.append(f'<div class="aqi"><span class="num cat-{level}">{_fmt(now["aqi"], 0)}</span>'
                        f'<span><b>{escape(now["category"])}</b><br>'
                        f'<span class="badge">mainly {escape(now["prominent_pollutant"])} · {escape(now["time"])}</span></span></div>')
            html.append(f'<p class="badge">{escape(now["advice"])}</p>')
            subs = ", ".join(f"{escape(GAS_NAMES.get(k, k))} {_fmt(v, 0)}" for k, v in now["sub_indices"].items())
            html.append(f"<dl><dt>Sub-indices</dt><dd>{subs}</dd>")
        else:
            html.append("<dl>")
        checks = "; ".join(f"{escape(k)}: {escape(', '.join(v))}" for k, v in s["flags_last_24h"].items())
        html.append(f"<dt>Sensor checks (24 h)</dt><dd>{checks or '<span class="good">all good</span>'}</dd></dl>")
        if s["days"]:
            html.append("<h3>Coming days (particulate index)</h3><dl>" + "".join(
                f"<dt>{escape(d['date'])}</dt><dd>{escape(d['category'])} · PM2.5 {_fmt(d['pm25_ug_m3'], 0)} µg/m³ "
                f"<span class=\"badge\">{d['p_poor_or_worse']:.0%} chance Poor or worse</span></dd>" for d in s["days"]) + "</dl>")
        pm = hourly[(hourly["station_id"] == station) & (hourly["pollutant"] == "pm25")].sort_values("target_time") if len(hourly) else hourly
        if len(pm):
            html.append(f"<h3>PM2.5, next 48 h (µg/m³; dashed: 24-h standard {NATIONAL_PM25_24H})</h3>")
            html.append(band_chart(pm["target_time"], pm["lo"], pm["mid"], pm["hi"], NATIONAL_PM25_24H, digits=0))
        html.append("</section>")
        cards.append("".join(html))
    return f'<h2 class="section-title">Air quality</h2><div class="grid">{"".join(cards)}</div>'
