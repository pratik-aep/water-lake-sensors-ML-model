"""Load, clean and label lake readings for the WQI model."""

import pandas as pd

from .calculator import classify_wqi, compute_wqi
from .config import CLASS_ORDER, LAB_PREFIX, PARAMETERS, VALID_RANGES


def label_column(columns, name: str) -> str:
    lab = LAB_PREFIX + name
    return lab if lab in columns else name


def parse_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    text = series.astype("string").str.strip()
    below_limit = text.str.startswith("<").fillna(False).astype(bool)
    values = pd.to_numeric(text.str.lstrip("<"), errors="coerce").astype(float)
    # "<1" means below the lab detection limit; substituting half the limit is the usual convention.
    return values.where(~below_limit, values / 2)


def prepare(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, dict]:
    label_cols = {name: label_column(df.columns, name) for name in PARAMETERS}
    needed = list(dict.fromkeys([*features, *label_cols.values()]))
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"Data is missing columns {missing}. WQI parameters may be given as '<name>' or '{LAB_PREFIX}<name>'."
        )

    df = df.copy()
    report = {"rows_in": len(df)}

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        bad_time = df["timestamp"].isna()
        report["bad_timestamp_dropped"] = int(bad_time.sum())
        df = df[~bad_time]

    out_of_range = {}
    for col in needed:
        values = parse_numeric(df[col])
        low, high = VALID_RANGES[col.removeprefix(LAB_PREFIX)]
        bad = values.notna() & ~values.between(low, high)
        if bad.any():
            out_of_range[col] = int(bad.sum())
        df[col] = values.mask(bad)
    report["out_of_range_set_to_missing"] = out_of_range

    keys = [c for c in ("station_id", "timestamp") if c in df.columns]
    before = len(df)
    df = df.drop_duplicates(subset=keys if "timestamp" in keys else None)
    report["duplicates_dropped"] = before - len(df)

    unlabeled = df[list(label_cols.values())].isna().any(axis=1)
    no_features = df[features].isna().all(axis=1)
    report["unlabeled_dropped"] = int(unlabeled.sum())
    report["no_feature_values_dropped"] = int((no_features & ~unlabeled).sum())
    df = df[~unlabeled & ~no_features].copy()
    if df.empty:
        raise ValueError(f"No usable labeled rows left after cleaning: {report}")

    reference = df[list(label_cols.values())].rename(columns={col: name for name, col in label_cols.items()})
    wqi = reference.apply(compute_wqi, axis=1)
    df["wqi_class"] = wqi.map(classify_wqi)
    df["wqi"] = wqi.round(2)

    if "timestamp" in df.columns:
        df = df.sort_values([c for c in ("timestamp", "station_id") if c in df.columns])
        report["date_range"] = [df["timestamp"].min().isoformat(), df["timestamp"].max().isoformat()]
    if "station_id" in df.columns:
        report["stations"] = sorted(df["station_id"].astype(str).unique().tolist())
    report["missing_feature_values"] = {c: int(n) for c, n in df[features].isna().sum().items() if n}
    report["rows_out"] = len(df)
    counts = df["wqi_class"].value_counts().reindex(CLASS_ORDER, fill_value=0)
    report["class_counts"] = {label: int(n) for label, n in counts.items()}
    return df.reset_index(drop=True), report


def load_dataset(path, features: list[str]) -> tuple[pd.DataFrame, dict]:
    return prepare(pd.read_csv(path), features)
