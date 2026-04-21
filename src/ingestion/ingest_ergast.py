"""
Ingest historical F1 data from Jolpica API (drop-in Ergast replacement).
Covers seasons 2021-2025: races, results, qualifying, drivers, constructors.
Jolpica base URL: https://api.jolpi.ca/ergast/f1
Rate limit: 4 req/s / 500 req/hr — we use 0.3s delay per request.
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

BASE_URL = "https://api.jolpi.ca/ergast/f1"
DELAY = 0.3


def _get(url: str, params: dict | None = None) -> dict:
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    time.sleep(DELAY)
    return resp.json()


def _paginated(url: str, key_path: list[str]) -> list[dict]:
    """Fetch all pages and return the combined list at the nested key_path."""
    limit = 100
    offset = 0
    results = []
    while True:
        data = _get(url, params={"limit": limit, "offset": offset})
        mr = data["MRData"]
        total = int(mr["total"])
        node = mr
        for k in key_path:
            node = node[k]
        results.extend(node)
        offset += limit
        if offset >= total:
            break
    return results


def fetch_drivers() -> pd.DataFrame:
    log.info("Fetching drivers...")
    drivers = _paginated(f"{BASE_URL}/drivers.json", ["DriverTable", "Drivers"])
    rows = [
        {
            "driver_id": d["driverId"],
            "code": d.get("code", ""),
            "given_name": d["givenName"],
            "family_name": d["familyName"],
            "nationality": d.get("nationality", ""),
            "driver_number": d.get("permanentNumber"),
            "dob": d.get("dateOfBirth"),
        }
        for d in drivers
    ]
    return pd.DataFrame(rows)


def fetch_constructors() -> pd.DataFrame:
    log.info("Fetching constructors...")
    constructors = _paginated(f"{BASE_URL}/constructors.json", ["ConstructorTable", "Constructors"])
    rows = [
        {
            "constructor_id": c["constructorId"],
            "name": c["name"],
            "nationality": c.get("nationality", ""),
        }
        for c in constructors
    ]
    return pd.DataFrame(rows)


def fetch_races(season: int) -> pd.DataFrame:
    log.info(f"Fetching races for {season}...")
    races = _paginated(f"{BASE_URL}/{season}/races.json", ["RaceTable", "Races"])
    rows = [
        {
            "season": int(r["season"]),
            "round": int(r["round"]),
            "race_name": r["raceName"],
            "circuit_id": r["Circuit"]["circuitId"],
            "circuit_name": r["Circuit"]["circuitName"],
            "country": r["Circuit"]["Location"]["country"],
            "locality": r["Circuit"]["Location"]["locality"],
            "race_date": r.get("date"),
            "race_time": r.get("time"),
        }
        for r in races
    ]
    return pd.DataFrame(rows)


def fetch_results(season: int) -> pd.DataFrame:
    log.info(f"Fetching results for {season}...")
    races = _paginated(f"{BASE_URL}/{season}/results.json", ["RaceTable", "Races"])
    rows = []
    for race in races:
        for res in race.get("Results", []):
            pos = res.get("position", "")
            rows.append(
                {
                    "season": int(race["season"]),
                    "round": int(race["round"]),
                    "driver_id": res["Driver"]["driverId"],
                    "constructor_id": res["Constructor"]["constructorId"],
                    "grid": int(res.get("grid", 0)),
                    "finish_position": int(pos) if pos.isdigit() else None,
                    "position_text": res.get("positionText", ""),
                    "points": float(res.get("points", 0)),
                    "laps": int(res.get("laps", 0)),
                    "status": res.get("status", ""),
                    "finish_time_ms": res.get("Time", {}).get("millis"),
                    "fastest_lap_rank": res.get("FastestLap", {}).get("rank"),
                    "fastest_lap_time": res.get("FastestLap", {}).get("Time", {}).get("time"),
                }
            )
    return pd.DataFrame(rows)


def fetch_qualifying(season: int) -> pd.DataFrame:
    log.info(f"Fetching qualifying for {season}...")
    races = _paginated(f"{BASE_URL}/{season}/qualifying.json", ["RaceTable", "Races"])
    rows = []
    for race in races:
        for q in race.get("QualifyingResults", []):
            rows.append(
                {
                    "season": int(race["season"]),
                    "round": int(race["round"]),
                    "driver_id": q["Driver"]["driverId"],
                    "constructor_id": q["Constructor"]["constructorId"],
                    "qualifying_position": int(q["position"]),
                    "q1_time": q.get("Q1"),
                    "q2_time": q.get("Q2"),
                    "q3_time": q.get("Q3"),
                }
            )
    return pd.DataFrame(rows)


def main(start_season: int, end_season: int, raw_dir: Path, write_db: bool = False) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)

    drivers_df = fetch_drivers()
    save_csv(drivers_df, raw_dir / "ergast_drivers.csv")
    log.info(f"  Drivers: {len(drivers_df)} rows")

    constructors_df = fetch_constructors()
    save_csv(constructors_df, raw_dir / "ergast_constructors.csv")
    log.info(f"  Constructors: {len(constructors_df)} rows")

    all_races, all_results, all_qualifying = [], [], []
    for season in range(start_season, end_season + 1):
        races_df = fetch_races(season)
        results_df = fetch_results(season)
        qualifying_df = fetch_qualifying(season)

        all_races.append(races_df)
        all_results.append(results_df)
        all_qualifying.append(qualifying_df)

        log.info(
            f"  Season {season}: {len(races_df)} races | "
            f"{len(results_df)} results | {len(qualifying_df)} qualifying rows"
        )

    races_combined = pd.concat(all_races, ignore_index=True)
    results_combined = pd.concat(all_results, ignore_index=True)
    qualifying_combined = pd.concat(all_qualifying, ignore_index=True)

    save_csv(races_combined, raw_dir / "ergast_races.csv")
    save_csv(results_combined, raw_dir / "ergast_results.csv")
    save_csv(qualifying_combined, raw_dir / "ergast_qualifying.csv")

    if write_db:
        upsert_df(drivers_df, "drivers", ["driver_id"])
        upsert_df(constructors_df, "constructors", ["constructor_id"])
        upsert_df(races_combined, "races", ["season", "round"])
        upsert_df(results_combined, "results", ["season", "round", "driver_id"])
        upsert_df(qualifying_combined, "qualifying", ["season", "round", "driver_id"])
        log.info("DB upserts complete.")

    log.info(
        f"Ergast/Jolpica ingestion complete — "
        f"{len(races_combined)} races | {len(results_combined)} results | "
        f"{len(qualifying_combined)} qualifying rows saved."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest F1 data from Jolpica (Ergast-compatible) API")
    parser.add_argument("--start-season", type=int, default=2021)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--write-db", action="store_true", help="Also upsert into PostgreSQL")
    args = parser.parse_args()

    settings = load_settings()
    raw_dir = Path(settings["paths"]["raw_data"])
    main(args.start_season, args.end_season, raw_dir, args.write_db)
