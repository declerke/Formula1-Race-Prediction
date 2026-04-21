"""
Ingest lap-level session data via FastF1 Python library.
Covers race sessions for the configured year (default: 2025).
FastF1 uses a local cache to avoid repeated downloads — enable it before any call.
"""
import logging
import argparse
from pathlib import Path

import pandas as pd
import fastf1

from src.utils.io_utils import load_settings, save_parquet, save_csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

LAP_COLS = [
    "Driver", "DriverNumber", "Team",
    "LapNumber", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
    "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
    "Compound", "TyreLife", "FreshTyre",
    "PitInTime", "PitOutTime",
    "IsPersonalBest", "IsAccurate",
]

WEATHER_COLS = ["Time", "AirTemp", "TrackTemp", "Humidity", "WindSpeed", "Rainfall"]


def _time_to_ms(t) -> float | None:
    """Convert a pandas Timedelta or NaT to milliseconds."""
    try:
        return t.total_seconds() * 1000
    except Exception:
        return None


def load_race_session(year: int, round_number: int, cache_dir: Path) -> pd.DataFrame:
    """
    Load a single race session and return a lap-level DataFrame with
    sector times converted to milliseconds and weather columns merged.
    """
    try:
        session = fastf1.get_session(year, round_number, "R")
        session.load(weather=True, telemetry=False)
    except Exception as e:
        log.warning(f"  Round {round_number} failed to load: {e}")
        return pd.DataFrame()

    laps = session.laps.copy()
    existing_cols = [c for c in LAP_COLS if c in laps.columns]
    laps = laps[existing_cols].copy()

    laps["season"] = year
    laps["round"] = round_number
    laps["race_name"] = session.event["EventName"]
    laps["circuit"] = session.event.get("Location", "")

    # Convert Timedelta columns to float milliseconds
    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        if col in laps.columns:
            laps[col] = laps[col].apply(_time_to_ms)

    # Merge average weather per lap via nearest time join
    try:
        weather = session.weather_data
        if weather is not None and not weather.empty:
            weather_cols_present = [c for c in WEATHER_COLS if c in weather.columns]
            weather = weather[weather_cols_present].copy()
            avg_weather = weather.drop(columns=["Time"], errors="ignore").mean()
            for col, val in avg_weather.items():
                laps[f"weather_{col.lower()}"] = val
    except Exception:
        pass

    return laps.reset_index(drop=True)


def main(year: int, rounds: list[int] | None, raw_dir: Path, cache_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    fastf1.Cache.enable_cache(str(cache_dir))

    # Determine rounds to process
    if rounds is None:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        rounds = schedule["RoundNumber"].dropna().astype(int).tolist()

    all_laps = []
    for rnd in rounds:
        log.info(f"Loading round {rnd} / {year}...")
        laps_df = load_race_session(year, rnd, cache_dir)
        if not laps_df.empty:
            all_laps.append(laps_df)
            save_parquet(laps_df, raw_dir / f"fastf1_laps_{year}_r{rnd:02d}.parquet")
            log.info(f"  Round {rnd}: {len(laps_df)} laps saved.")

    if all_laps:
        combined = pd.concat(all_laps, ignore_index=True)
        save_parquet(combined, raw_dir / f"fastf1_laps_{year}_all.parquet")
        # Also save CSV summary (aggregated per driver per race for joining)
        summary = (
            combined.groupby(["season", "round", "Driver", "DriverNumber", "Team"])
            .agg(
                median_lap_ms=("LapTime", "median"),
                best_lap_ms=("LapTime", "min"),
                median_s1_ms=("Sector1Time", "median"),
                median_s2_ms=("Sector2Time", "median"),
                median_s3_ms=("Sector3Time", "median"),
                max_speed_st=("SpeedST", "max"),
                total_laps=("LapNumber", "count"),
                compounds_used=("Compound", lambda x: "|".join(x.dropna().unique())),
            )
            .reset_index()
        )
        save_csv(summary, raw_dir / f"fastf1_summary_{year}.csv")
        log.info(
            f"FastF1 ingestion complete — {len(combined)} total laps across "
            f"{len(all_laps)} rounds for {year}."
        )
    else:
        log.warning("No FastF1 data collected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest F1 session data via FastF1")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument(
        "--rounds", type=int, nargs="+", default=None,
        help="Specific round numbers to load (default: all rounds in the season)"
    )
    args = parser.parse_args()

    settings = load_settings()
    raw_dir = Path(settings["paths"]["raw_data"])
    cache_dir = Path(settings["paths"]["fastf1_cache"])
    main(args.year, args.rounds, raw_dir, cache_dir)
