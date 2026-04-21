"""
Ingest F1 session, lap, and stint data from OpenF1 API.
Coverage: 2023-2025 (historical data, no auth required).
Rate limit: 3 req/s / 30 req/min — we use 0.5s delay per request.
"""
import time
import logging
import argparse
from pathlib import Path

import requests
import pandas as pd

from src.utils.io_utils import load_settings, save_csv
from src.utils.db import upsert_df

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_URL = "https://api.openf1.org/v1"
DELAY = 2.0  # 30 req/min limit → 2s is safe

LAP_COLS = [
    "session_key", "driver_number", "lap_number", "lap_duration",
    "duration_sector_1", "duration_sector_2", "duration_sector_3",
    "i1_speed", "i2_speed", "st_speed", "is_pit_out_lap",
]
STINT_COLS = [
    "session_key", "driver_number", "stint_number",
    "lap_start", "lap_end", "compound", "tyre_age_at_start",
]
SESSION_COLS = [
    "session_key", "meeting_key", "year", "session_name", "session_type",
    "date_start", "date_end", "circuit_short_name", "country_name", "location",
]
DRIVER_COLS = [
    "session_key", "driver_number", "name_acronym", "full_name",
    "team_name", "team_colour", "country_code",
]


def _get(endpoint: str, params: dict | None = None, retries: int = 4) -> list[dict]:
    url = f"{BASE_URL}/{endpoint}"
    for attempt in range(retries):
        resp = requests.get(url, params=params, timeout=60)
        time.sleep(DELAY)
        if resp.status_code == 404:
            log.debug(f"  404 for {url} — no data for this session, skipping.")
            return []
        if resp.status_code == 429:
            wait = 10 * (attempt + 1)
            log.warning(f"  429 rate-limited — waiting {wait}s before retry {attempt + 1}/{retries}")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []
    log.error(f"  Gave up after {retries} retries: {url}")
    return []


def _keep_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[[c for c in cols if c in df.columns]]


def fetch_sessions(year: int, session_type: str = "Race") -> pd.DataFrame:
    log.info(f"Fetching {session_type} sessions for {year}...")
    data = _get("sessions", {"year": year, "session_type": session_type})
    if not data:
        return pd.DataFrame()
    return _keep_cols(pd.DataFrame(data), SESSION_COLS)


def fetch_laps(session_key: int) -> pd.DataFrame:
    log.info(f"  Fetching laps for session_key={session_key}...")
    data = _get("laps", {"session_key": session_key})
    if not data:
        return pd.DataFrame()
    df = _keep_cols(pd.DataFrame(data), LAP_COLS)
    # Rename to match DB schema
    df = df.rename(columns={
        "duration_sector_1": "sector1_ms",
        "duration_sector_2": "sector2_ms",
        "duration_sector_3": "sector3_ms",
    })
    return df


def fetch_stints(session_key: int) -> pd.DataFrame:
    log.info(f"  Fetching stints for session_key={session_key}...")
    data = _get("stints", {"session_key": session_key})
    if not data:
        return pd.DataFrame()
    return _keep_cols(pd.DataFrame(data), STINT_COLS)


def fetch_drivers(session_key: int) -> pd.DataFrame:
    data = _get("drivers", {"session_key": session_key})
    if not data:
        return pd.DataFrame()
    return _keep_cols(pd.DataFrame(data), DRIVER_COLS)


def main(years: list[int], raw_dir: Path, write_db: bool = False) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_sessions, all_laps, all_stints, all_drivers = [], [], [], []

    for year in years:
        sessions_df = fetch_sessions(year)
        if sessions_df.empty:
            log.warning(f"No race sessions found for {year} — skipping.")
            continue
        all_sessions.append(sessions_df)

        for _, row in sessions_df.iterrows():
            sk = int(row["session_key"])

            laps_df = fetch_laps(sk)
            if not laps_df.empty:
                all_laps.append(laps_df)

            stints_df = fetch_stints(sk)
            if not stints_df.empty:
                all_stints.append(stints_df)

            drivers_df = fetch_drivers(sk)
            if not drivers_df.empty:
                all_drivers.append(drivers_df)

        log.info(f"Year {year}: {len(sessions_df)} race sessions processed.")

    def _concat_save(frames: list, path: Path, label: str) -> pd.DataFrame | None:
        if not frames:
            log.warning(f"No {label} data collected.")
            return None
        combined = pd.concat(frames, ignore_index=True)
        save_csv(combined, path)
        log.info(f"  {label}: {len(combined)} rows saved.")
        return combined

    sessions_out = _concat_save(all_sessions, raw_dir / "openf1_sessions.csv", "sessions")
    laps_out = _concat_save(all_laps, raw_dir / "openf1_laps.csv", "laps")
    stints_out = _concat_save(all_stints, raw_dir / "openf1_stints.csv", "stints")

    if all_drivers:
        drivers_out = (
            pd.concat(all_drivers, ignore_index=True)
            .drop_duplicates(subset=["driver_number", "team_name"])
        )
        save_csv(drivers_out, raw_dir / "openf1_drivers.csv")
        log.info(f"  drivers: {len(drivers_out)} unique driver-team rows saved.")

    if write_db and laps_out is not None:
        upsert_df(laps_out, "lap_times", ["session_key", "driver_number", "lap_number"])
    if write_db and stints_out is not None:
        upsert_df(stints_out, "telemetry", ["session_key", "driver_number", "stint_number"])

    log.info("OpenF1 ingestion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest F1 data from OpenF1 API")
    parser.add_argument("--year", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--write-db", action="store_true")
    args = parser.parse_args()

    settings = load_settings()
    raw_dir = Path(settings["paths"]["raw_data"])
    main(args.year, raw_dir, args.write_db)
