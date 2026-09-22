# Udaipur lakes and air: monitoring models

Seven models turn the vendor's lake and air stations into daily warnings.
- **Lake buoys:** four sensors (temperature, pH, turbidity, dissolved oxygen) every 10 minutes.
- **Air stations:** particles PM1 to TSP, plus eight gases.

Optional inputs make them better: hourly weather (fetched automatically), weekly lab results and a maintenance log. Each model lives in its own folder and can be trained and run alone. `pipeline/` joins them into one daily report, dashboard and alert message.

| Folder | What it answers | Method |
|---|---|---|
| `anomaly_detection/` | Is a lake sensor broken, or is something happening in the lake? | QARTOD-style rule tests; cross-sensor consistency models, with rain memory and air temperature as the water-temperature reference when weather is given; drift tracking (a 2-day average of each sensor's disagreement); Isolation Forest; fault-vs-event verdict |
| `forecasting/` | What will the next 48 h look like? Will oxygen crash before dawn this week? | GBM quantile and CNN-LSTM ensemble with conformal ranges, recalibrated weekly; nightly DO minimum for 7 nights |
| `bod_surrogate/` | Today's BOD, and which CPCB class (A–E) is the lake in? | Diel-oxygen metabolism (night-time regression) feeding a soft sensor; ridge, PLS or monotonic GBM chosen by time-blocked CV |
| `wqi/` | The weighted-arithmetic WQI class | Classifier trained on lab-computed WQI, with sensor features |
| `algal_bloom/` | How much algae (chlorophyll-a)? Trophic state and WHO bloom level? | Soft sensor; Carlson TSI and WHO 2021 alert levels (12 and 24 µg/L) |
| `coliform_nowcast/` | Is the water safe for contact (the CPCB bacteria limits)? | Soft sensor with rain-timing and turbidity inputs; P(>50/500/5000 MPN) |
| `air_quality/` | India's National AQI now, and will tomorrow be a Poor air day? | CPCB NAQI (24 h / 8 h averaging rules); sensor checks (stuck, spikes, PM size order, humidity correction, sensor ceilings); 48-hour PM2.5/PM10 quantile forecasts and next-day index with conformal ranges |

Every estimate comes with a range and the chance of crossing each legal limit. Every model is scored against a simple rule it has to beat, for example "same as the last lab value" or "no change".

## Daily operation

```bash
# One-off: train on history (about 25 min). Vendor exports go through the mapping below.
.venv/bin/python -m models.pipeline.train_all --readings lake_history.csv --air-readings air_history.csv \
    --weather weather.csv --lab lab_results.csv --known-episodes maintenance_log.csv \
    --vendor-mapping models/pipeline/vendor_mapping.json
# Pass every path: the defaults point at the synthetic demo data made by `models.pipeline.simulate`.

# Every morning (or scheduled, below):
.venv/bin/python -m models.pipeline.daily --notify
```

The daily job does five things:
1. Adds every export waiting in `pipeline/inbox/` (lakes) and `pipeline/inbox_air/` (air) to the stored history in `pipeline/history/`.
2. Refreshes the hourly weather from Open-Meteo: the recent past plus 8 days ahead. It needs no key, and if it's offline the stored weather is used.
3. Looks at the last 45 days of readings (30 of them keep the forecast ranges calibrated to the season) and writes `reports/<time>/`:
   - `report.json` and `report.txt`
   - `dashboard.html`: one self-contained page, with a card per lake and per air station
   - CSVs of every forecast and estimate
4. Sends alerts at or above the chosen severity.
5. Moves the processed exports to `processed/`.

**Scheduling:** `pipeline/schedule/` holds a macOS launchd file and a Linux crontab. The crontab also retrains everything on the 1st of each month.

**Alerts:** email over any SMTP server (for Gmail, use an app password) and/or a webhook (Slack, Teams, Discord, or a WhatsApp/SMS gateway). They are configured by environment variables, so no password lives in the code:

| Variable | Meaning |
|---|---|
| `ALERT_MIN_SEVERITY` | `high`, `medium` (default) or `low` |
| `ALERT_EMAIL_TO`, `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `ALERT_EMAIL_FROM` | email |
| `ALERT_WEBHOOK_URL` | chat or SMS gateway |

Try it without sending anything: `python -m models.pipeline.notify --report <report.json> --dry-run`.

## Getting real data in

- **`pipeline/vendor_mapping.json`** maps the vendor's column names and units onto ours. The spec says oxygen and turbidity are reported as 0–100%.
  - Oxygen is converted to mg/L using the water temperature.
  - Turbidity stays on the vendor's % scale until a few side-by-side lab turbidimeter readings give a `[slope, intercept]` to convert it to NTU.
  - Readings pinned at a sensor's range limit are set aside, not used as real values: pH 9 on the 4–9 probe, oxygen 100%.
  - The mapping used at training is saved and reapplied by every daily run.
- **`python -m models.pipeline.check_data --readings export.csv --vendor-mapping ... --lab ... --weather ... --air-readings ...`** reports:
  - whether the units look right;
  - how many readings sat at sensor limits;
  - which models have enough data to train, and what each still needs.
- **`python -m models.pipeline.weather --start 2024-01-01`** downloads Udaipur's recorded weather history for training.
- **Air station exports** use these columns: `station_id, timestamp, pm1, pm25, pm10, tsp` (µg/m³) and `o3, no2, co, so2, nox, h2s, co2, voc` (ppm, as the vendor reports them).
- **Grow into it.** `train_all` trains whatever the data allows and says what it skipped and why. The fault detector comes first (88 days of sensor data). The forecaster, lab-based models and air model follow as their data accumulates. The daily report works with any subset of trained models.
- **WQI and turbidity.** The WQI rates turbidity in NTU, so it is trained only when turbidity is calibrated to NTU or the lab measures turbidity.
- **Minimum data to train:**
  - Fault detector: 88 days.
  - Forecaster: 150 days.
  - Each lab-based model: 38 lab samples taken with the sensors running.
  - WQI: 50 lab samples with BOD, conductivity and nitrate.
  - Air model: 135 days (45 to learn, 30 to calibrate, 60 to test).

## Current results (synthetic data, most recent period held out)

| Model | Result | Simple rule it must beat |
|---|---|---|
| Fault detector | Monsoon test: spikes, flatlines, offsets, oscillation and dropouts 8/8 each; drift 7/8 (median 47 h); pollution events 5/8; 1.1 false alarms per station-week, mostly turbidity in storms. Calmer standalone test: all kinds 8/8, 0.02 false alarms | – |
| Lake forecaster | 80% ranges hold 77–81% of the time; low-oxygen alert CSI 0.47 (0.77 for the next 2 nights) | no change |
| BOD | MAE 0.92 mg/L | last lab value 1.22 |
| WQI | 96% correct | – |
| Chlorophyll-a | MAE 5.2 µg/L; trophic class matches the lab 89% of the time | – |
| Total coliform | CPCB band matches the lab 77% of the time (lab noise sets the ceiling) | – |
| Air quality | Hourly PM2.5 15–25% better than no change, ranges hold 78–81%; tomorrow's PM2.5 MAE 9.4 µg/m³ (no change: 11.7), category right 70%; Poor-day warnings catch 50% with 33% false (Brier skill 0.38) | no change |

These numbers come from simulated lakes and air that were designed alongside the models, so real data will score lower. Retrain on real data and judge the models by the metrics files it writes (`*_metrics.json` beside each model).

## Known limits

- **Vendor sensor ranges hide the worst readings.**
  - pH is read only to 9, while eutrophic lakes pass 9 on bloom afternoons.
  - Oxygen is read only to 100%, while afternoon supersaturation is common.
  - NO₂ tops out at 0.1 ppm (188 µg/m³), so the station cannot show NO₂ beyond the start of "Poor".
  - The report says when a reading hit a ceiling. Wider-range sensors would fix this properly.
- **Missed pollution events.** In the monsoon test, 3 of 8 events are missed. `event_z` in `anomaly_detection/config.py` trades sensitivity for false alarms: at 2.5 instead of 3.0 it caught 7 of 8, but tripled false event alarms.
- **Free ammonia** (the CPCB class D criterion) can't be measured by these sensors, so it is always "not assessed".
- **New stations.** Drift is judged against learned normal behaviour, so a new station starts with a conservative alarm level, and drift takes about two days to catch.
- **Tomorrow's AQI** comes from particulates only. PM2.5 usually sets Udaipur's index, but a high-ozone day can be under-called.

## Tests

`.venv/bin/python -m pytest models -q -p no:warnings`: all tests, about 10–15 min. They also run on GitHub on every push (`.github/workflows/tests.yml`).
