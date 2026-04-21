"""
Feature engineering pipeline.
Input:  data/processed/cleaned_results.parquet
Output: data/processed/model_table.parquet

One row = one driver-race instance.
All rolling features are computed using only past races (no leakage).
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io_utils import load_settings, load_parquet, save_parquet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _rolling_past(series: pd.Series, window: int, agg: str) -> pd.Series:
    """
    Compute a rolling statistic over the past `window` races,
    excluding the current row (shift by 1 to prevent leakage).
    """
    return series.shift(1).rolling(window=window, min_periods=1).agg(agg)


def add_driver_rolling_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Rolling driver-level features sorted chronologically within each driver."""
    df = df.sort_values(["driver_id", "season", "round"]).copy()

    grp = df.groupby("driver_id")["finish_position"]

    df["avg_finish_last5"] = grp.transform(lambda s: _rolling_past(s, window, "mean"))
    df["std_finish_last5"] = grp.transform(lambda s: _rolling_past(s, window, "std")).fillna(0)
    df["best_finish_last5"] = grp.transform(lambda s: _rolling_past(s, window, "min"))
    df["recent_form_last3"] = grp.transform(lambda s: _rolling_past(s, 3, "mean"))

    # DNF rate over last 10 races
    if "is_dnf" in df.columns:
        df["dnf_rate_last10"] = (
            df.groupby("driver_id")["is_dnf"]
            .transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
        )
    else:
        df["dnf_rate_last10"] = 0.0

    # Points momentum
    grp_pts = df.groupby("driver_id")["points"]
    df["avg_points_last5"] = grp_pts.transform(lambda s: _rolling_past(s, window, "mean"))

    return df


def add_constructor_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Rolling constructor performance: avg points per race over last N races."""
    df = df.sort_values(["constructor_id", "season", "round"]).copy()

    grp = df.groupby("constructor_id")["points"]
    df["constructor_avg_pts_last5"] = grp.transform(lambda s: _rolling_past(s, window, "mean"))
    df["constructor_avg_finish_last5"] = (
        df.groupby("constructor_id")["finish_position"]
        .transform(lambda s: _rolling_past(s, window, "mean"))
    )
    return df


def add_track_features(df: pd.DataFrame, track_window: int = 3) -> pd.DataFrame:
    """
    Historical performance at the same circuit (last N visits).
    Grouped by (driver_id, circuit_id), sorted by (season, round).
    """
    if "circuit_id" not in df.columns:
        df["track_avg_finish"] = np.nan
        df["track_best_finish"] = np.nan
        return df

    df = df.sort_values(["driver_id", "circuit_id", "season", "round"]).copy()

    grp = df.groupby(["driver_id", "circuit_id"])["finish_position"]
    df["track_avg_finish"] = grp.transform(lambda s: _rolling_past(s, track_window, "mean"))
    df["track_best_finish"] = grp.transform(lambda s: _rolling_past(s, track_window, "min"))

    return df


def add_qualifying_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Qualifying-derived features.
    - qualifying_position (already in df from clean step)
    - quali_gap_to_pole: driver's best_quali_ms minus session minimum (relative pace signal)
    """
    if "qualifying_position" not in df.columns:
        df["qualifying_position"] = np.nan
        df["quali_gap_to_pole"] = np.nan
        return df

    if "best_quali_ms" in df.columns:
        session_best = df.groupby(["season", "round"])["best_quali_ms"].transform("min")
        df["quali_gap_to_pole"] = df["best_quali_ms"] - session_best
    else:
        df["quali_gap_to_pole"] = np.nan

    return df


def add_pace_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Relative lap pace from OpenF1 (if available).
    Normalise median lap time against the race leader's median.
    """
    if "openf1_median_lap_s" not in df.columns:
        df["pace_ratio"] = np.nan
        return df

    race_min = df.groupby(["season", "round"])["openf1_median_lap_s"].transform("min")
    df["pace_ratio"] = df["openf1_median_lap_s"] / race_min
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode driver_id and constructor_id for tree-based models."""
    for col in ["driver_id", "constructor_id", "circuit_id"]:
        if col in df.columns:
            df[f"{col}_enc"] = df[col].astype("category").cat.codes
    return df


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    df["is_top3"] = (df["finish_position"] <= 3).astype(int)
    df["is_winner"] = (df["finish_position"] == 1).astype(int)
    return df


FEATURE_COLS = [
    "qualifying_position",
    "grid",
    "avg_finish_last5",
    "std_finish_last5",
    "best_finish_last5",
    "recent_form_last3",
    "dnf_rate_last10",
    "avg_points_last5",
    "constructor_avg_pts_last5",
    "constructor_avg_finish_last5",
    "track_avg_finish",
    "track_best_finish",
    "quali_gap_to_pole",
    "pace_ratio",
    "driver_id_enc",
    "constructor_id_enc",
]

TARGET_COLS = ["is_top3", "is_winner"]
META_COLS = ["season", "round", "driver_id", "constructor_id", "circuit_id", "finish_position", "points"]


def main() -> None:
    settings = load_settings()
    processed_dir = Path(settings["paths"]["processed_data"])

    src = processed_dir / "cleaned_results.parquet"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found — run clean_data.py first.")

    df = load_parquet(src)
    log.info(f"Loaded cleaned results: {len(df)} rows")

    window = settings["model"]["rolling_window"]
    track_window = settings["model"]["track_window"]

    df = build_targets(df)
    df = add_driver_rolling_features(df, window=window)
    df = add_constructor_features(df, window=window)
    df = add_track_features(df, track_window=track_window)
    df = add_qualifying_features(df)
    df = add_pace_features(df)
    df = encode_categoricals(df)

    # Drop rows where all rolling features are NaN (first races of a driver's career)
    core_features = ["avg_finish_last5", "qualifying_position"]
    before = len(df)
    df = df.dropna(subset=[c for c in core_features if c in df.columns], how="all")
    log.info(f"Dropped {before - len(df)} rows with all-NaN core features.")

    # Remaining NaN fill: median impute per season
    for col in FEATURE_COLS:
        if col in df.columns:
            season_median = df.groupby("season")[col].transform("median")
            global_median = df[col].median()
            df[col] = df[col].fillna(season_median).fillna(global_median)

    keep_cols = [c for c in META_COLS + FEATURE_COLS + TARGET_COLS if c in df.columns]
    df = df[keep_cols].sort_values(["season", "round", "finish_position"]).reset_index(drop=True)

    out_path = processed_dir / "model_table.parquet"
    save_parquet(df, out_path)

    log.info(
        f"Model table saved → {out_path}\n"
        f"  Shape: {df.shape}\n"
        f"  is_top3 positive rate: {df['is_top3'].mean():.2%}\n"
        f"  is_winner positive rate: {df['is_winner'].mean():.2%}\n"
        f"  Feature columns: {[c for c in FEATURE_COLS if c in df.columns]}"
    )


if __name__ == "__main__":
    main()
