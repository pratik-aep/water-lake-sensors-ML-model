"""The whole daily job: take in new vendor exports, refresh the weather, write the report and dashboard, send alerts.

Drop each CSV the vendor's app exports into the inbox folder (air station exports into inbox_air); this adds it to
the stored history (keeping the vendor's own format, which run converts with the mapping saved at training) and
moves it to processed/.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

from . import notify, run, weather
from .ingest import load_mapping

HERE = Path(__file__).parent


def take_in(inbox: Path, history: Path, mapping: dict) -> dict:
    """Append every export waiting in the inbox to the history file; the newest copy of a reading wins."""
    files = sorted(inbox.glob("*.csv"))
    if not files:
        return {"files": 0, "new_rows": 0}
    station, stamp = mapping["columns"]["station_id"], mapping["columns"]["timestamp"]
    parts = [pd.read_csv(history, low_memory=False)] if history.exists() else []
    before = len(parts[0]) if parts else 0
    parts += [pd.read_csv(f, low_memory=False) for f in files]
    combined = pd.concat(parts, ignore_index=True)
    order = pd.to_datetime(combined[stamp], errors="coerce", utc=bool(mapping.get("timezone")))
    combined = combined.assign(_order=order).sort_values([station, "_order"], kind="stable")
    combined = combined.drop_duplicates([station, stamp], keep="last").drop(columns="_order")
    history.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(history, index=False)
    done = inbox / "processed"
    done.mkdir(exist_ok=True)
    for f in files:
        shutil.move(str(f), done / f.name)
    return {"files": len(files), "new_rows": len(combined) - before}


def recent(history: Path, mapping: dict, days: int, out: Path) -> Path:
    """The last `days` of history (what run needs for baselines, forecasts and metabolism), as its own file."""
    raw = pd.read_csv(history, low_memory=False)
    stamps = pd.to_datetime(raw[mapping["columns"]["timestamp"]], errors="coerce", utc=bool(mapping.get("timezone")))
    raw[stamps > stamps.max() - pd.Timedelta(days=days)].to_csv(out, index=False)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inbox", type=Path, default=HERE / "inbox")
    parser.add_argument("--history", type=Path, default=HERE / "history" / "readings.csv")
    parser.add_argument("--air-inbox", type=Path, default=HERE / "inbox_air", help="air station exports (our column names)")
    parser.add_argument("--air-history", type=Path, default=HERE / "history" / "air_readings.csv")
    parser.add_argument("--weather", type=Path, default=HERE / "history" / "weather.csv")
    parser.add_argument("--no-weather-fetch", action="store_true", help="use the weather file as it is (e.g. offline)")
    parser.add_argument("--models-dir", type=Path, default=HERE / "artifacts")
    parser.add_argument("--out-dir", type=Path, default=HERE / "reports")
    parser.add_argument("--days", type=int, default=45,
                        help="days of recent readings each report looks at; 45 leaves 30 days of already-verified "
                             "forecasts to keep the forecast ranges calibrated to the season")
    parser.add_argument("--notify", action="store_true", help="send alerts (configured by environment, see notify.py)")
    parser.add_argument("--dry-run-notify", action="store_true", help="print the alert message instead of sending it")
    args = parser.parse_args(argv)

    manifest = json.loads((args.models_dir / "manifest.json").read_text())
    mapping = manifest.get("vendor_mapping") or load_mapping(None)
    args.inbox.mkdir(parents=True, exist_ok=True)
    taken = take_in(args.inbox, args.history, mapping)
    print(f"Inbox: {taken['files']} export(s), {taken['new_rows']} new readings")
    if not args.history.exists():
        sys.exit(f"no readings yet: put a vendor export in {args.inbox}")
    air_mapping = {"columns": {"station_id": "station_id", "timestamp": "timestamp"}}
    if "air_quality" in manifest["models"]:
        args.air_inbox.mkdir(parents=True, exist_ok=True)
        air_taken = take_in(args.air_inbox, args.air_history, air_mapping)
        print(f"Air inbox: {air_taken['files']} export(s), {air_taken['new_rows']} new readings")

    if not args.no_weather_fetch:
        try:
            fresh = weather.fetch()
            existing = pd.read_csv(args.weather) if args.weather.exists() else None
            args.weather.parent.mkdir(parents=True, exist_ok=True)
            weather.merge(existing, fresh).to_csv(args.weather, index=False)
            print(f"Weather refreshed up to {fresh['timestamp'].max()}")
        except (OSError, ValueError, KeyError) as error:  # offline, or a bad reply: carry on with the stored weather
            print(f"Warning: weather fetch failed ({error}); using {args.weather} as it is")

    window = recent(args.history, mapping, args.days, args.history.parent / "recent.csv")
    run_args = ["--readings", str(window), "--models-dir", str(args.models_dir), "--out-dir", str(args.out_dir)]
    if args.weather.exists():
        run_args += ["--weather", str(args.weather)]
    if "air_quality" in manifest["models"] and args.air_history.exists():
        air_window = recent(args.air_history, air_mapping, args.days, args.air_history.parent / "recent_air.csv")
        run_args += ["--air-readings", str(air_window)]
    out = run.main(run_args)

    if args.notify or args.dry_run_notify:
        report = json.loads((out / "report.json").read_text())
        sent = notify.notify(report, notify.config_from_env(), dry_run=args.dry_run_notify)
        print(f"Alerts: {sent['alerts']} to send; email sent: {sent['email']}; webhook sent: {sent['webhook']}")
    print(f"Dashboard: {out / 'dashboard.html'}")


if __name__ == "__main__":
    main()
