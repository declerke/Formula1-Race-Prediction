"""
Unit tests for the feature engineering pipeline.
Tests verify: no data leakage, output shape, required columns, target integrity.
"""
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.build_features import (
    add_driver_rolling_features,
    add_constructor_features,
    add_track_features,
    add_qualifying_features,
    build_targets,
    FEATURE_COLS,
    TARGET_COLS,
)


# ── Fixtures ────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Synthetic driver-race table: 3 drivers over 3 seasons × 10 rounds."""
    rng = np.random.default_rng(42)
    drivers = ["hamilton", "verstappen", "leclerc"]
    seasons = [2021, 2022, 2023]
    rounds = list(range(1, 11))
    constructors = {"hamilton": "mercedes", "verstappen": "red_bull", "leclerc": "ferrari"}
    circuits = [f"circuit_{i}" for i in range(1, 11)]

    rows = []
    for season in seasons:
        for rnd in rounds:
            for driver in drivers:
                finish = rng.integers(1, 21)
                rows.append({
                    "season": season,
                    "round": rnd,
                    "driver_id": driver,
                    "constructor_id": constructors[driver],
                    "circuit_id": circuits[rnd - 1],
                    "finish_position": finish,
                    "grid": rng.integers(1, 21),
                    "qualifying_position": rng.integers(1, 21),
                    "best_quali_ms": float(rng.integers(70000, 90000)),
                    "points": max(0.0, float(26 - finish * 1.3)),
                    "is_dnf": bool(rng.random() < 0.05),
                    "openf1_median_lap_s": float(rng.uniform(85, 100)),
                })
    return pd.DataFrame(rows)


# ── No data leakage ─────────────────────────────────────────────────────────────

def test_rolling_feature_no_leakage(sample_df):
    """
    Rolling features for race N must not use data from race N.
    Verify by checking that for a driver's very first race,
    avg_finish_last5 is NaN (no past data to compute from).
    """
    df = add_driver_rolling_features(sample_df.copy(), window=5)
    df_sorted = df.sort_values(["driver_id", "season", "round"])
    first_races = df_sorted.groupby("driver_id").head(1)
    # First race has no prior data — rolling feature should be NaN
    assert first_races["avg_finish_last5"].isna().all(), (
        "avg_finish_last5 should be NaN for a driver's first race (no prior data)"
    )


def test_rolling_feature_uses_only_past_data(sample_df):
    """
    For the second race of a driver, avg_finish_last5 should equal
    only the first race's finish_position (shift-1 rolling mean of window 1).
    """
    df = add_driver_rolling_features(sample_df.copy(), window=5)
    df_sorted = df.sort_values(["driver_id", "season", "round"]).reset_index(drop=True)
    for driver in sample_df["driver_id"].unique():
        driver_df = df_sorted[df_sorted["driver_id"] == driver].reset_index(drop=True)
        first_finish = driver_df.iloc[0]["finish_position"]
        second_row_feature = driver_df.iloc[1]["avg_finish_last5"]
        assert abs(second_row_feature - first_finish) < 1e-6, (
            f"Driver {driver}: avg_finish_last5 at race 2 should equal race 1 finish "
            f"({first_finish}), got {second_row_feature}"
        )


# ── Output shape and columns ────────────────────────────────────────────────────

def test_rolling_features_output_columns(sample_df):
    df = add_driver_rolling_features(sample_df.copy())
    expected = ["avg_finish_last5", "std_finish_last5", "best_finish_last5", "recent_form_last3"]
    for col in expected:
        assert col in df.columns, f"Expected column '{col}' not found in output"


def test_constructor_features_output(sample_df):
    df = add_constructor_features(sample_df.copy())
    assert "constructor_avg_pts_last5" in df.columns
    assert "constructor_avg_finish_last5" in df.columns
    assert len(df) == len(sample_df), "Row count must not change"


def test_track_features_output(sample_df):
    df = add_track_features(sample_df.copy())
    assert "track_avg_finish" in df.columns
    assert "track_best_finish" in df.columns


def test_qualifying_features_output(sample_df):
    df = add_qualifying_features(sample_df.copy())
    assert "quali_gap_to_pole" in df.columns
    gap = df["quali_gap_to_pole"].dropna()
    assert (gap >= 0).all(), "quali_gap_to_pole must be non-negative (driver time >= session best)"


# ── Target integrity ────────────────────────────────────────────────────────────

def test_targets_binary(sample_df):
    df = build_targets(sample_df.copy())
    assert df["is_top3"].isin([0, 1]).all()
    assert df["is_winner"].isin([0, 1]).all()


def test_is_winner_subset_of_is_top3(sample_df):
    df = build_targets(sample_df.copy())
    assert (df[df["is_winner"] == 1]["is_top3"] == 1).all(), (
        "Every race winner must also be in top 3"
    )


def test_exactly_one_winner_per_race(sample_df):
    """Each (season, round) should have at most 1 winner."""
    df = build_targets(sample_df.copy())
    winners_per_race = df.groupby(["season", "round"])["is_winner"].sum()
    assert (winners_per_race <= 1).all(), "More than one winner found in a single race"


def test_is_top3_rate_near_expected(sample_df):
    """With 20 drivers per race, is_top3 rate should be ~15%."""
    df = build_targets(sample_df.copy())
    rate = df["is_top3"].mean()
    assert 0.10 <= rate <= 0.25, f"is_top3 rate {rate:.2%} is outside expected 10-25% range"


# ── Edge cases ──────────────────────────────────────────────────────────────────

def test_no_constructor_features_on_empty_df():
    empty = pd.DataFrame(columns=["season", "round", "driver_id", "constructor_id",
                                  "finish_position", "points"])
    result = add_constructor_features(empty)
    assert len(result) == 0


def test_track_features_with_missing_circuit_id(sample_df):
    df = sample_df.drop(columns=["circuit_id"])
    result = add_track_features(df)
    # Should fill with NaN gracefully, not raise
    assert "track_avg_finish" in result.columns
