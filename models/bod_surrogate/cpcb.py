"""The best CPCB designated-best-use class a reading meets, on the criteria this system can measure."""

import numpy as np
import pandas as pd

from .config import CPCB_CLASSES, NOT_ASSESSED


def assess(dissolved_oxygen, ph, p_bod_at_most: dict, p_coliform_at_most: dict | None = None) -> pd.DataFrame:
    """Best class per row, from measured DO and pH and P(value <= limit) keyed by each class's BOD / coliform limit."""
    # A class is met when DO and pH meet its limits and BOD (and coliform, if supplied) are more likely than not
    # within theirs; the two are treated as independent.
    do = np.asarray(dissolved_oxygen, dtype=float)
    ph = np.asarray(ph, dtype=float)
    best = np.full(len(do), "below E", dtype=object)
    confidence = np.full(len(do), np.nan)
    unassessed = np.full(len(do), "", dtype=object)
    decided = np.isnan(do) | np.isnan(ph)
    best[decided] = "unknown"
    ones = np.ones(len(do))
    for name, _, do_min, bod_max, coliform_max, (ph_lo, ph_hi) in CPCB_CLASSES:
        p_bod = ones if bod_max is None else np.asarray(p_bod_at_most[bod_max], dtype=float)
        coliform_known = coliform_max is not None and p_coliform_at_most is not None
        p_coliform = np.asarray(p_coliform_at_most[coliform_max], dtype=float) if coliform_known else ones
        p_limits = p_bod * p_coliform
        meets = (do >= (do_min or -np.inf)) & (ph >= ph_lo) & (ph <= ph_hi) & (p_limits >= 0.5)
        take = meets & ~decided
        best[take] = name
        confidence[take] = p_limits[take]
        unassessed[take] = "" if coliform_known else NOT_ASSESSED[name]
        decided |= take
    return pd.DataFrame({"cpcb_class": best, "cpcb_confidence": confidence, "cpcb_not_assessed": unassessed})
