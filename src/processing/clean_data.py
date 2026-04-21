"""
Merge and clean data from all three ingestion sources into a single
canonical results table saved to data/processed/cleaned_results.parquet.

Canonical key: (season, round, driver_id)
Driver identity mapping: Jolpica driver_id ↔ driver_number (integer) ↔ FastF1 3-letter code
"""
import logging
from pathlib import Path

import pandas as pd
import numpy as np

from src.utils.io_utils import load_settings, load_csv, save_parquet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _parse_lap_time_to_ms(t: str) -> float | None:
    """Convert 'M:SS.mmm' string to float milliseconds."""
    if not t or pd.isna(t):
        return None
    try:
        parts = str(t).strip().split(":")
        if len(parts) == 2:
            minutes, seconds = parts
            return (int(minutes) * 60 + float(seconds)) * 1000
        return float(t) * 1000
    except Exception:
        return None


def build_driver_number_map(drivers_df: pd.DataFrame, openf1_drivers_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Build a mapping table: driver_id (Jolpica) → driver_number (integer) → name_acronym (FastF1 code).
    Jolpica drivers table has driver_number; OpenF1 drivers table has driver_number + name_acronym.
    """
    mapping = drivers_df[["driver_id", "code", "driver_number", "given_name", "family_name"]].copy()
    mapping["driver_number"] = pd.to_numeric(mapping["driver_number"], errors="coerce")

    if openf1_drivers_df is not None and not openf1_drivers_df.empty:
        acronym_map = (
            openf1_drivers_df[["driver_number", "name_acronym"]]
            .dropna(subset=["driver_number"])
            .drop_duplicates("driver_number")
        )
        acronym_map["driver_number"] = pd.to_numeric(acronym_map["driver_number"], errors="coerce")
        mapping = mapping.merge(acronym_map, on="driver_number", how="left")
        # Fall back to Jolpica 3-letter code where acronym is missing
        mapping["name_acronym"] = mapping["name_acronym"].fillna(mapping["code"])
    else:
        mapping["name_acronym"] = mapping["code"]

    return mapping.dropna(subset=["driver_id"])


def load_raw(raw_dir: Path) -> dict[str, pd.DataFrame | None]:
    def _safe_load(path: Path) -> pd.DataFrame | None:
        if path.exists():
            df = load_csv(path)
            log.info(f"  Loaded {path.name}: {len(df)} rows")
            return df
        log.warning(f"  Not found: {path.name} — skipping.")
        return None

    return {
        "drivers": _safe_load(raw_dir / "ergast_drivers.csv"),
        "constructors": _safe_load(raw_dir / "ergast_constructors.csv"),
        "races": _safe_load(raw_dir / "ergast_races.csv"),
        "results": _safe_load(raw_dir / "ergast_results.csv"),
        "qualifying": _safe_load(raw_dir / "ergast_qualifying.csv"),
        "openf1_sessions": _safe_load(raw_dir / "openf1_sessions.csv"),
        "openf1_laps": _safe_load(raw_dir / "openf1_laps.csv"),
        "openf1_stints": _safe_load(raw_dir / "openf1_stints.csv"),
        "openf1_drivers": _safe_load(raw_dir / "openf1_drivers.csv"),
    }


def clean_results(results: pd.DataFrame) -> pd.DataFrame:
    df = results.copy()
    df["season"] = df["season"].astype(int)
    df["round"] = df["round"].astype(int)
    df["grid"] = pd.to_numeric(df["grid"], errors="coerce").fillna(0).astype(int)
    df["finish_position"] = pd.to_numeric(df["finish_position"], errors="coerce")
    df["points"] = pd.to_numeric(df["points"], errors="coerce").fillna(0.0)
    df["laps"] = pd.to_numeric(df["laps"], errors="coerce").fillna(0).astype(int)
    df["is_dnf"] = ~df["status"].str.lower().str.startswith("finished").fillna(False) & \
                   (df["finish_position"].isna())
    df = df.drop_duplicates(subset=["season", "round", "driver_id"])
    return df


def clean_qualifying(qualifying: pd.DataFrame) -> pd.DataFrame:
    df = qualifying.copy()
    df["season"] = df["season"].astype(int)
    df["round"] = df["round"].astype(int)
    df["qualifying_position"] = pd.to_numeric(df["qualifying_position"], errors="coerce")
    for col in ["q1_time", "q2_time", "q3_time"]:
        df[f"{col}_ms"] = df[col].apply(_parse_lap_time_to_ms)
    df["best_quali_ms"] = df[["q1_time_ms", "q2_time_ms", "q3_time_ms"]].min(axis=1)
    df = df.drop_duplicates(subset=["season", "round", "driver_id"])
    return df


def build_openf1_lap_summary(
    openf1_laps: pd.DataFrame,
    openf1_sessions: pd.DataFrame,
    driver_map: pd.DataFrame,
) -> pd.DataFrame | None:
    """
    Aggregate OpenF1 lap data to one row per (session_key, driver_number):
    median lap pace, best lap, sector medians, max speed.
    Then join session year/circuit info and driver_id from the driver map.
    """
    if openf1_laps is None or openf1_laps.empty:
        return None

    laps = openf1_laps.copy()
    laps["lap_duration"] = pd.to_numeric(laps["lap_duration"], errors="coerce")
    laps["driver_number"] = pd.to_numeric(laps["driver_number"], errors="coerce")

    # Remove outliers (pit laps, safety car laps) — keep laps within 120% of session median
    session_medians = laps.groupby("session_key")["lap_duration"].transform("median")
    laps = laps[laps["lap_duration"] <= session_medians * 1.2]
    laps = laps[laps.get("is_pit_out_lap", False) != True]

    agg = (
        laps.groupby(["session_key", "driver_number"])
        .agg(
            openf1_median_lap_s=("lap_duration", "median"),
            openf1_best_lap_s=("lap_duration", "min"),
            openf1_median_s1=("sector1_ms", "median"),
            openf1_median_s2=("sector2_ms", "median"),
            openf1_median_s3=("sector3_ms", "median"),
            openf1_max_speed_st=("st_speed", "max"),
        )
        .reset_index()
    )

    if openf1_sessions is not None and not openf1_sessions.empty:
        sessions = openf1_sessions[["session_key", "year"]].drop_duplicates()
        sessions["session_key"] = pd.to_numeric(sessions["session_key"], errors="coerce")
        agg = agg.merge(sessions, on="session_key", how="left")

    # Map driver_number → driver_id
    num_to_id = driver_map.dropna(subset=["driver_number", "driver_id"]).copy()
    num_to_id["driver_number"] = pd.to_numeric(num_to_id["driver_number"], errors="coerce")
    agg = agg.merge(num_to_id[["driver_number", "driver_id"]], on="driver_number", how="left")

    return agg


def main() -> None:
    settings = load_settings()
    raw_dir = Path(settings["paths"]["raw_data"])
    processed_dir = Path(settings["paths"]["processed_data"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading raw files...")
    raw = load_raw(raw_dir)

    if raw["results"] is None:
        raise FileNotFoundError("ergast_results.csv not found — run ingest_ergast.py first.")

    # Driver mapping table
    driver_map = build_driver_number_map(raw["drivers"], raw["openf1_drivers"])
    log.info(f"Driver map: {len(driver_map)} entries")

    # Clean core tables
    results = clean_results(raw["results"])
    qualifying = clean_qualifying(raw["qualifying"]) if raw["qualifying"] is not None else pd.DataFrame()

    # Merge qualifying into results on (season, round, driver_id)
    if not qualifying.empty:
        quali_cols = [
            "season", "round", "driver_id",
            "qualifying_position", "best_quali_ms", "q1_time_ms", "q2_time_ms", "q3_time_ms",
        ]
        quali_merge = qualifying[[c for c in quali_cols if c in qualifying.columns]]
        results = results.merge(quali_merge, on=["season", "round", "driver_id"], how="left")

    # Merge race metadata (circuit_id, race_date)
    if raw["races"] is not None:
        races = raw["races"][["season", "round", "circuit_id", "circuit_name", "country", "race_date"]].copy()
        races["season"] = races["season"].astype(int)
        races["round"] = races["round"].astype(int)
        results = results.merge(races, on=["season", "round"], how="left")

    # Merge OpenF1 lap aggregates
    openf1_summary = build_openf1_lap_summary(
        raw["openf1_laps"], raw["openf1_sessions"], driver_map
    )
    if openf1_summary is not None:
        openf1_cols = [
            "driver_id", "year",
            "openf1_median_lap_s", "openf1_best_lap_s",
            "openf1_median_s1", "openf1_median_s2", "openf1_median_s3",
            "openf1_max_speed_st",
        ]
        openf1_merge = openf1_summary[[c for c in openf1_cols if c in openf1_summary.columns]].copy()
        openf1_merge = openf1_merge.rename(columns={"year": "season"})
        openf1_merge["season"] = pd.to_numeric(openf1_merge["season"], errors="coerce")
        results = results.merge(
            openf1_merge.drop_duplicates(subset=["season", "driver_id"]),
            on=["season", "driver_id"],
            how="left",
        )

    # Final type enforcement
    results["finish_position"] = pd.to_numeric(results["finish_position"], errors="coerce")
    results["grid"] = pd.to_numeric(results["grid"], errors="coerce").fillna(0).astype(int)
    results["qualifying_position"] = pd.to_numeric(
        results.get("qualifying_position", pd.Series(dtype=float)), errors="coerce"
    )

    # Median-impute numeric columns by (driver_id, season)
    numeric_cols = [
        "finish_position", "qualifying_position", "grid",
        "openf1_median_lap_s", "openf1_best_lap_s", "openf1_max_speed_st",
    ]
    for col in numeric_cols:
        if col in results.columns:
            group_median = results.groupby(["driver_id", "season"])[col].transform("median")
            global_median = results[col].median()
            results[col] = results[col].fillna(group_median).fillna(global_median)

    results = results.drop_duplicates(subset=["season", "round", "driver_id"])
    results = results.sort_values(["season", "round", "finish_position"]).reset_index(drop=True)

    out_path = processed_dir / "cleaned_results.parquet"
    save_parquet(results, out_path)
    log.info(
        f"Cleaned results saved → {out_path} | "
        f"{len(results)} rows | seasons {results['season'].min()}–{results['season'].max()}"
    )


if __name__ == "__main__":
    main()
