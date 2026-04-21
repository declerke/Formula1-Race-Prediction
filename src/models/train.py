"""
Train Logistic Regression (baseline) and XGBoost (advanced) classifiers
for both targets: is_top3 and is_winner.

Time-aware split: train on historical seasons, validate on intermediate, test on latest.
Artifacts saved to artifacts/ as .pkl files.
"""
import json
import logging
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

from src.utils.io_utils import load_settings, load_parquet
from src.features.build_features import FEATURE_COLS, TARGET_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def time_split(
    df: pd.DataFrame,
    train_seasons: list[int],
    val_seasons: list[int],
    test_seasons: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["season"].isin(train_seasons)]
    val = df[df["season"].isin(val_seasons)]
    test = df[df["season"].isin(test_seasons)]
    return train, val, test


def get_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    available = [c for c in FEATURE_COLS if c in df.columns]
    return df[available], available


def build_logistic_regression(pos_weight: float) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=0.5,
            solver="lbfgs",
            random_state=42,
        )),
    ])


def build_xgboost(scale_pos_weight: float, random_state: int, n_estimators: int,
                  max_depth: int, learning_rate: float) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )


def train_for_target(
    train: pd.DataFrame,
    val: pd.DataFrame,
    target: str,
    feature_cols: list[str],
    settings: dict,
    artifacts_dir: Path,
) -> dict:
    X_train = train[feature_cols].fillna(0)
    y_train = train[target]
    X_val = val[feature_cols].fillna(0) if not val.empty else X_train
    y_val = val[target] if not val.empty else y_train

    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / max(pos_count, 1)
    log.info(f"  [{target}] pos={pos_count}, neg={neg_count}, scale_pos_weight={scale_pos_weight:.2f}")

    # Logistic Regression
    log.info(f"  [{target}] Training Logistic Regression...")
    lgr = build_logistic_regression(scale_pos_weight)
    lgr.fit(X_train, y_train)
    lgr_report = classification_report(y_val, lgr.predict(X_val), output_dict=True, zero_division=0)
    joblib.dump(lgr, artifacts_dir / f"lgr_{target}.pkl")

    # XGBoost
    log.info(f"  [{target}] Training XGBoost...")
    xgb = build_xgboost(
        scale_pos_weight=scale_pos_weight,
        random_state=settings["model"]["random_state"],
        n_estimators=settings["model"]["xgb_n_estimators"],
        max_depth=settings["model"]["xgb_max_depth"],
        learning_rate=settings["model"]["xgb_learning_rate"],
    )
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    xgb_report = classification_report(y_val, xgb.predict(X_val), output_dict=True, zero_division=0)
    joblib.dump(xgb, artifacts_dir / f"xgb_{target}.pkl")

    return {
        "target": target,
        "logistic_regression": lgr_report,
        "xgboost": xgb_report,
        "feature_importances": dict(zip(feature_cols, xgb.feature_importances_.tolist())),
    }


def main(targets: list[str]) -> None:
    settings = load_settings()
    processed_dir = Path(settings["paths"]["processed_data"])
    artifacts_dir = Path(settings["paths"]["artifacts"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_table = processed_dir / "model_table.parquet"
    if not model_table.exists():
        raise FileNotFoundError(f"{model_table} not found — run build_features.py first.")

    df = load_parquet(model_table)
    log.info(f"Loaded model table: {df.shape}")

    _, feature_cols = get_features(df)
    log.info(f"Features ({len(feature_cols)}): {feature_cols}")

    # Save feature names for the Streamlit app
    with open(artifacts_dir / "feature_names.json", "w") as f:
        json.dump(feature_cols, f)

    train_seasons = settings["model"]["train_seasons"]
    val_seasons = settings["model"]["val_seasons"]
    test_seasons = settings["model"]["test_seasons"]

    train, val, test = time_split(df, train_seasons, val_seasons, test_seasons)
    log.info(
        f"Split — train: {len(train)} rows ({train_seasons}) | "
        f"val: {len(val)} rows ({val_seasons}) | "
        f"test: {len(test)} rows ({test_seasons})"
    )

    # Save test set for evaluate.py
    test.to_parquet(processed_dir / "test_set.parquet", index=False)

    all_results = []
    for target in targets:
        if target not in df.columns:
            log.warning(f"Target '{target}' not in model table — skipping.")
            continue
        log.info(f"Training models for target: {target}")
        result = train_for_target(train, val, target, feature_cols, settings, artifacts_dir)
        all_results.append(result)
        log.info(
            f"  [{target}] LR val accuracy: "
            f"{result['logistic_regression'].get('accuracy', 0):.3f} | "
            f"XGB val accuracy: {result['xgboost'].get('accuracy', 0):.3f}"
        )

    metrics_dir = Path(settings["paths"]["metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "training_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    log.info(f"Training complete. Artifacts saved to {artifacts_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train F1 prediction models")
    parser.add_argument(
        "--target",
        type=str,
        nargs="+",
        default=["is_top3", "is_winner"],
        choices=["is_top3", "is_winner"],
    )
    args = parser.parse_args()
    main(args.target)
