"""Classify lake readings with a trained WQI model."""

import argparse
import json
import math
from pathlib import Path

import joblib
import pandas as pd

from .calculator import classify_wqi, compute_wqi, do_percent_to_mgl
from .config import CLASS_ORDER, PARAMETERS, VALID_RANGES

HERE = Path(__file__).parent


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isnan(value)


def check_reading(bundle: dict, reading: dict) -> tuple[list[str], list[str]]:
    """Errors block a prediction; warnings flag values the model never saw in training."""
    errors, warnings = [], []
    for name in bundle["features"]:
        value = reading.get(name)
        if not _is_number(value):
            errors.append(f"'{name}' must be a number, got {value!r}")
            continue
        low, high = VALID_RANGES[name]
        if not low <= value <= high:
            errors.append(f"'{name}'={value} is outside the physical range {low:g}-{high:g}")
            continue
        seen_low, seen_high = bundle["feature_ranges"][name]
        if not seen_low <= value <= seen_high:
            warnings.append(
                f"'{name}'={value} is outside the training range {seen_low:g}-{seen_high:g}; trust this prediction less"
            )
    return errors, warnings


def classify(bundle: dict, readings: pd.DataFrame) -> pd.DataFrame:
    model = bundle["model"]
    proba = model.predict_proba(readings[bundle["features"]])
    result = pd.DataFrame(proba, columns=[f"p_{c}" for c in model.classes_], index=readings.index)
    result.insert(0, "predicted_class", model.classes_[proba.argmax(axis=1)])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reading",
        help='JSON object, e.g. \'{"temperature": 26, "ph": 7.9, "turbidity": 6.5, "dissolved_oxygen": 5.8}\'',
    )
    parser.add_argument("--model", type=Path, default=HERE / "artifacts" / "wqi_sensor.joblib")
    parser.add_argument(
        "--do-percent", action="store_true", help="dissolved_oxygen is %% saturation (vendor sensor output), not mg/L"
    )
    parser.add_argument(
        "--pressure-atm", type=float, default=1.0, help="barometric pressure for the %% to mg/L conversion (~0.93 at Udaipur)"
    )
    args = parser.parse_args(argv)

    reading = json.loads(args.reading)
    if not isinstance(reading, dict):
        parser.error("reading must be a JSON object")
    if args.do_percent:
        if not (_is_number(reading.get("dissolved_oxygen")) and _is_number(reading.get("temperature"))):
            parser.error("--do-percent needs numeric 'dissolved_oxygen' and 'temperature'")
        reading["dissolved_oxygen"] = round(
            float(do_percent_to_mgl(reading["dissolved_oxygen"], reading["temperature"], args.pressure_atm)), 2
        )
        print(f"Dissolved oxygen converted to {reading['dissolved_oxygen']} mg/L")

    bundle = joblib.load(args.model)
    errors, warnings = check_reading(bundle, reading)
    if errors:
        parser.error("; ".join(errors))
    for warning in warnings:
        print(f"Warning: {warning}")

    result = classify(bundle, pd.DataFrame([reading])).iloc[0]
    print(f"Predicted class: {result['predicted_class']}  ({bundle['model_name']}, trained {bundle['trained_at']})")
    for label in CLASS_ORDER:
        if f"p_{label}" in result:
            print(f"  {label:<11} {result[f'p_{label}']:.2f}")
    if all(_is_number(reading.get(name)) for name in PARAMETERS):
        wqi = compute_wqi(reading)
        print(f"Computed WQI from all parameters: {wqi:.1f} -> {classify_wqi(wqi)}")


if __name__ == "__main__":
    main()
