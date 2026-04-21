"""
Evaluate trained models on the held-out test set.
Produces metrics/*.json with accuracy, precision, recall, F1, and confusion matrix.
"""
import json
import logging
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)

from src.utils.io_utils import load_settings, load_parquet
from src.features.build_features import FEATURE_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    target: str,
) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    cm = confusion_matrix(y_test, y_pred).tolist()
    metrics = {
        "model": model_name,
        "target": target,
        "n_test": int(len(y_test)),
        "n_positive": int(y_test.sum()),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "confusion_matrix": cm,
        "confusion_matrix_labels": ["Negative", "Positive"],
    }
    if y_proba is not None and len(np.unique(y_test)) > 1:
        metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_proba)), 4)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    metrics["classification_report"] = report

    return metrics


def interpretation(metrics: dict) -> str:
    target = metrics["target"]
    recall = metrics["recall"]
    precision = metrics["precision"]
    roc = metrics.get("roc_auc", "N/A")
    lines = [
        f"Target: {target}",
        f"  Recall  {recall:.2%} — of all true {'podium' if 'top3' in target else 'winners'}, "
        f"the model captured this fraction.",
        f"  Precision {precision:.2%} — of all predicted positives, this fraction were correct.",
        f"  ROC-AUC: {roc}",
    ]
    if recall < 0.4:
        lines.append("  ⚠ Low recall: model misses many true positives — consider lowering threshold.")
    if precision < 0.3:
        lines.append("  ⚠ Low precision: high false-positive rate — class imbalance may be dominating.")
    return "\n".join(lines)


def main(targets: list[str]) -> None:
    settings = load_settings()
    artifacts_dir = Path(settings["paths"]["artifacts"])
    processed_dir = Path(settings["paths"]["processed_data"])
    metrics_dir = Path(settings["paths"]["metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    test_path = processed_dir / "test_set.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"{test_path} not found — run train.py first.")

    test = load_parquet(test_path)
    feature_names_path = artifacts_dir / "feature_names.json"
    if feature_names_path.exists():
        with open(feature_names_path) as f:
            feature_cols = json.load(f)
    else:
        feature_cols = [c for c in FEATURE_COLS if c in test.columns]

    X_test = test[feature_cols].fillna(0)
    all_metrics = []

    for target in targets:
        if target not in test.columns:
            log.warning(f"Target '{target}' not in test set — skipping.")
            continue

        y_test = test[target]
        log.info(f"\nEvaluating target: {target} | {int(y_test.sum())} positives / {len(y_test)} rows")

        for model_type in ["lgr", "xgb"]:
            artifact = artifacts_dir / f"{model_type}_{target}.pkl"
            if not artifact.exists():
                log.warning(f"  {artifact.name} not found — skipping.")
                continue

            model = joblib.load(artifact)
            model_name = "Logistic Regression" if model_type == "lgr" else "XGBoost"
            metrics = evaluate_model(model, X_test, y_test, model_name, target)
            all_metrics.append(metrics)

            log.info(
                f"  {model_name}: accuracy={metrics['accuracy']:.3f} | "
                f"precision={metrics['precision']:.3f} | recall={metrics['recall']:.3f} | "
                f"F1={metrics['f1']:.3f} | ROC-AUC={metrics.get('roc_auc', 'N/A')}"
            )
            log.info(interpretation(metrics))

    out_path = metrics_dir / "evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    log.info(f"\nEvaluation complete — results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate F1 prediction models")
    parser.add_argument(
        "--target",
        type=str,
        nargs="+",
        default=["is_top3", "is_winner"],
        choices=["is_top3", "is_winner"],
    )
    args = parser.parse_args()
    main(args.target)
