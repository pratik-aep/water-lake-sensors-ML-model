"""Sensors, sampling interval and detection settings for the anomaly detector."""

SENSORS = ["temperature", "ph", "turbidity", "dissolved_oxygen"]
INTERVAL = "10min"  # the vendor station's default scan rate

# Copied into every trained detector, so a saved model keeps the settings it was fit with.
DEFAULT_SETTINGS = {
    # Fit, flag the training history, then refit without what was flagged (2 passes = one self-cleaning round).
    "training_passes": 2,
    # Logger resolution: a reading within this of the previous one counts as "unchanged" for the flat-line test.
    "resolution": {"temperature": 0.001, "ph": 0.001, "turbidity": 0.01, "dissolved_oxygen": 0.001},
    # Smallest spread assumed for model errors: about half of typical field-sensor accuracy, so differences a
    # sensor can't reliably measure never raise alarms. Turbidity is in log1p units: "±0.3 NTU or 2%" is ~±10%
    # in clear lakes of 1-3 NTU.
    "min_scale": {"temperature": 0.1, "ph": 0.05, "turbidity": 0.05, "dissolved_oxygen": 0.1},
    "flat_suspect_points": 12,  # 2 hours of identical readings
    "flat_fail_points": 24,
    # Centred window, so a spike is confirmed 20 minutes after it happens.
    "spike_window": 5,
    "spike_suspect": 6.0,  # distance from the rolling median, in multiples of the sensor's normal step size
    "spike_fail": 12.0,
    "rate_of_change_limit": 8.0,  # one-step change, in multiples of the normal step size
    "climatology_quantiles": (0.001, 0.999),
    "climatology_margin": 0.1,  # widen the learned seasonal band by this fraction of its width
    "residual_z": 5.0,  # this far from what the other sensors predict = inconsistent
    # Water temperature follows the weather, not the chemistry. With hourly weather it is checked against air
    # temperature averaged over these time scales (hours; a lake lags the air by days); without weather, against
    # its own trailing median over this many hours, which catches jumps but not a slow drift.
    "air_memory_hours": (24, 96, 240),
    "temperature_anchor_hours": 48,
    # With hourly weather, the consistency models also see recent rain, remembered over these time scales
    # (hours): runoff raises turbidity for a day or two, and without rain a monsoon storm looks like a fouled sensor.
    "rain_memory_hours": (6, 36, 120),
    # Error scales and alarm thresholds are calibrated on predictions for whole held-out blocks of this length,
    # matching a detector that is retrained about this often.
    "cv_block_days": 14,
    # A consistency prediction is trusted only when its inputs lie inside the range seen in training (these
    # quantiles, widened by this fraction of their width). Outside it the check pauses rather than guess.
    "envelope_quantiles": (0.001, 0.999),
    "envelope_margin": 0.1,
    # Model error that builds slowly as seasons move beyond the training data is removed by subtracting its
    # trailing median. The window is a trade-off: a slow drift hides inside it too, so a week lets a sensor
    # drifting over several days stand out (3 days caught 3 of 8 test drifts; 7 days, 7 of 8).
    "residual_detrend_days": 7,
    # Drift: an exponentially weighted average of each sensor's disagreement, with this time constant (hours);
    # two days separates a sensor that keeps drifting from a spell of unusual weather. Its alarm level is this
    # quantile of the same average on training data (never below drift_min).
    "drift_hours": 48,
    "drift_quantile": 0.999,
    "drift_min": 1.0,  # a short training history shows too little slow error to set the level from alone
    "drift_clip": 4.0,  # caps one reading's push on the average, so a single spike can't raise a drift alarm
    # The average is held within this multiple of its alarm level, so an alarm clears within a few hours of
    # the sensor being fixed however far it had drifted.
    "drift_cap": 1.5,
    # The alarm holds only while the plain (not detrended) disagreement over the last hour still leans the same
    # way, at this share of the alarm level at least, so it clears soon after the sensor is cleaned.
    "drift_confirm_hours": 1,
    "drift_confirm_fraction": 0.5,
    # Events are judged against each station's trailing median, so they work before a full year of history exists.
    "baseline_days": 7,
    "event_z": 3.0,  # departure from that baseline that counts toward a pollution event
    "event_min_sensors": 2,
    "event_min_points": 3,
    # One sensor usually reacts first when an event starts; its flags in this lead-up are relabelled as the event.
    "event_onset_hours": 3,
    "iforest_quantile": 0.9995,
    "review_min_points": 2,
    "merge_gap_points": 6,  # alarms less than an hour apart are reported as one incident
}
