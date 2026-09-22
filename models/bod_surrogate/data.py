"""Load lab results for a soft sensor."""

import pandas as pd

from ..wqi.config import VALID_RANGES
from ..wqi.dataset import parse_numeric


def load_lab(path, target: str = "bod", limits: tuple | None = None) -> tuple[pd.DataFrame, dict]:
    """Lab results for `target`; "<1"-style values below the detection limit become half the limit."""
    raw = pd.read_csv(path)
    required = ["station_id", "timestamp", target]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Lab data is missing columns {missing}")
    lab = raw[required].copy()
    report = {"samples_in": len(lab)}
    lab["station_id"] = lab["station_id"].astype(str)
    lab["timestamp"] = pd.to_datetime(lab["timestamp"], errors="coerce")
    lab[target] = parse_numeric(lab[target])
    low, high = limits or VALID_RANGES[target]
    # Values are modelled on a log scale, so a zero result can't be used; "<1" arrives as 0.5 and is kept.
    usable = lab["timestamp"].notna() & lab[target].gt(0) & lab[target].between(low, high)
    report["unusable_dropped"] = int((~usable).sum())
    lab = lab[usable]
    before = len(lab)
    lab = lab.drop_duplicates(["station_id", "timestamp"])
    report["duplicates_dropped"] = before - len(lab)
    report["samples_out"] = len(lab)
    return lab.sort_values("timestamp").reset_index(drop=True), report
