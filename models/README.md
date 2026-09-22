# Lake water quality models (Udaipur lakes)

Six models turn the four sensors on each buoy (temperature, pH, turbidity and dissolved oxygen, every 10 minutes) into warnings. A few inputs are optional: hourly weather, weekly lab results and a maintenance log. Each model lives in its own folder and can be trained and run on its own. `pipeline/` joins them into one daily report.

| Folder | What it answers | Method |
|---|---|---|
| `anomaly_detection/` | Is a sensor broken, or is something happening in the lake? | QARTOD-style rule tests; cross-sensor consistency models, with rain memory and air temperature as the water-temperature reference when weather is given; drift tracking (a 2-day average of each sensor's disagreement); Isolation Forest; and a fault-vs-event verdict |
| `forecasting/` | What will the next 48 h look like? Will oxygen crash before dawn this week? | GBM quantile and CNN-LSTM ensemble with conformal ranges, recalibrated weekly; nightly DO minimum for 7 nights |
| `bod_surrogate/` | What is today's BOD, and which CPCB class (A–E) is the lake in? | Diel-oxygen metabolism (night-time regression) feeding a soft sensor; ridge, PLS or monotonic GBM chosen by time-blocked CV |
| `wqi/` | What is the weighted-arithmetic WQI class? | Classifier trained on lab-computed WQI, with sensor features |
| `algal_bloom/` | How much algae (chlorophyll-a)? Trophic state and WHO bloom level? | Soft sensor; outputs Carlson TSI and WHO 2021 alert levels 12/24 µg/L |
| `coliform_nowcast/` | Is the water safe for contact (total coliform, the CPCB bacteria limits)? | Soft sensor with rain-timing and turbidity inputs; outputs P(>50/500/5000 MPN) |

Every soft sensor reports a range and the chance of exceeding each legal limit, not just one number. Each model is also compared against the "last lab value" baseline.

## Running the whole system

```bash
.venv/bin/python -m models.pipeline.simulate          # synthetic world (skip once real data exists)
.venv/bin/python -m models.pipeline.train_all         # ~20 min; add --no-challenger to skip the CNN-LSTM
.venv/bin/python -m models.pipeline.run --readings recent.csv --weather weather.csv
```

Real data plugs in through the same arguments:

| Argument | Contents |
|---|---|
| `--readings` | Vendor CSV export: `station_id, timestamp, temperature, ph, turbidity, dissolved_oxygen` |
| `--weather` | Hourly: `timestamp, air_temperature, cloud_cover, rain_mm`, including forecast hours for the week ahead |
| `--lab` | `station_id, timestamp, bod, conductivity, nitrate`, plus optional `chlorophyll_a` and `total_coliform` |
| `--known-episodes` | The maintenance log; faults it records are not counted as false alarms when the detector is scored |

The algae and coliform stages are trained only when the lab panel contains their columns.

`run` writes `reports/<time>/report.json` and `report.txt`, with prioritised alerts per lake and CSVs of every estimate.

## Current results (synthetic world, most recent 60 days held out)

| Model | Result | Baseline |
|---|---|---|
| Anomaly detector | Monsoon test: spikes, flatlines, offsets, oscillation and dropouts 8/8 each; drift 7/8 (median 47 h); pollution events 5/8; 1.1 false alarms per station-week, mostly turbidity during storms. Calmer standalone test: every kind 8/8, 0.02 false alarms | – |
| Forecaster | Ensemble; 80% ranges contain the outcome 77–81% of the time; low-oxygen alert CSI 0.47 (0.77 for the next 2 nights) | – |
| BOD | MAE 0.92 mg/L | Last lab value: 1.22 |
| WQI | Holdout accuracy 96% | – |
| Chlorophyll-a | MAE 5.2 µg/L; trophic class matches the lab 89% of the time | – |
| Total coliform | CPCB band matches the lab 77% of the time; above-500 alerts: detection 0.93, false-alarm ratio 0.03 (standalone run) | – |

These numbers come from simulated lakes. Retrain and re-read the metrics JSON files once real data arrives.

## Known limits

- The CPCB class D criterion for free ammonia cannot be measured by these sensors, so it is always reported as "not assessed".
- In the monsoon test, 3 of 8 pollution events are missed. Against daily oxygen swings and storm-driven turbidity they don't stand far enough from the past week's median. Comparing with the same time of day on previous days caught one more, but doubled false event alarms, so it was not adopted.
- Drift is judged against normal behaviour learned from history. A newly installed station has little of it, so a conservative alarm floor applies and drifts are typically caught after about two days.
- Weather is read from a CSV file. Connecting an online weather service (IMD / Open-Meteo) is an open decision.
- Tests: `.venv/bin/python -m pytest models -q -p no:warnings` (102 tests, about 10 min).
