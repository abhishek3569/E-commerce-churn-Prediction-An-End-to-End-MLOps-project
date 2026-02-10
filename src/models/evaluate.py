"""
Model evaluation with business metrics and comparison to previous versions.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from src.config import get_settings
from src.features.feature_engineering import TARGET_COLUMN

logger = logging.getLogger(__name__)

# Business cost assumptions (configurable)
COST_FALSE_POSITIVE = 10  # Unnecessary retention offer cost
COST_FALSE_NEGATIVE = 100  # Lost customer cost


def load_model_from_mlflow(run_id: str) -> Tuple[any, str]:
    """Load model artifact from MLflow run. Returns (model, model_type)."""
    import mlflow.sklearn
    import mlflow.xgboost
    import mlflow.lightgbm

    client = mlflow.MlflowClient()
    run = client.get_run(run_id)
    model_type = run.data.params.get("model_type", "unknown")
    model_uri = f"runs:/{run_id}/model"
    if model_type in ("logistic_regression", "random_forest"):
        model = mlflow.sklearn.load_model(model_uri)
    elif model_type == "xgboost":
        model = mlflow.xgboost.load_model(model_uri)
    elif model_type == "lightgbm":
        model = mlflow.lightgbm.load_model(model_uri)
    else:
        model = mlflow.pyfunc.load_model(model_uri)
    return model, model_type


def evaluate_on_test(
    run_id: str,
    test_features_path: Optional[Path] = None,
) -> Dict:
    """
    Evaluate model on test set and compute business metrics.
    """
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)

    processed_dir = Path(settings.data.processed_data_dir)
    test_path = test_features_path or processed_dir / settings.data.features_test_file

    if not test_path.exists():
        raise FileNotFoundError(f"Test features not found at {test_path}")

    df = pd.read_parquet(test_path)
    y_true = df[TARGET_COLUMN].values
    X = df.drop(columns=[TARGET_COLUMN])

    model, model_type = load_model_from_mlflow(run_id)
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (np.array(y_proba) >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else 0.0,
    }

    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
        if cm.shape[0] >= 1 and cm.shape[1] >= 1:
            tp = int(cm[1, 1]) if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
            fn = int(cm[1, 0]) if cm.shape[0] > 1 else 0
            fp = int(cm[0, 1]) if cm.shape[1] > 1 else 0
            tn = int(cm[0, 0])
    cost_fp = fp * COST_FALSE_POSITIVE
    cost_fn = fn * COST_FALSE_NEGATIVE
    total_cost = cost_fp + cost_fn

    # Expected ROI: assume intervention saves 30% of churners, costs $20 per intervention
    intervention_cost = 20
    potential_saves = fn * 0.3 * COST_FALSE_NEGATIVE
    intervention_spend = (tp + fp) * intervention_cost
    expected_roi = (potential_saves - intervention_spend) / max(intervention_spend, 1)

    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    metrics["cost_false_positives"] = cost_fp
    metrics["cost_false_negatives"] = cost_fn
    metrics["total_cost"] = total_cost
    metrics["expected_roi"] = expected_roi
    metrics["model_type"] = model_type

    return metrics


def compare_with_production(
    new_run_id: str,
    production_run_id: Optional[str] = None,
) -> Dict:
    """
    Compare new model against current production model.
    """
    new_metrics = evaluate_on_test(new_run_id)
    result = {"new_model": new_metrics, "production_model": None, "improvement": {}}

    if production_run_id:
        try:
            prod_metrics = evaluate_on_test(production_run_id)
            result["production_model"] = prod_metrics
            for k in ["auc_roc", "f1", "precision", "recall"]:
                if k in new_metrics and k in prod_metrics:
                    result["improvement"][k] = new_metrics[k] - prod_metrics[k]
        except Exception as e:
            logger.warning("Could not load production model for comparison: %s", e)

    return result


def generate_evaluation_report(
    run_id: str,
    output_path: Optional[Path] = None,
    production_run_id: Optional[str] = None,
) -> Dict:
    """
    Generate full evaluation report with visualizations.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    comparison = compare_with_production(run_id, production_run_id)
    metrics = comparison["new_model"]

    report_path = output_path or Path("data/processed/evaluation_report.json")
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as f:
        json.dump(comparison, f, indent=2)

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    metric_names = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    values = [metrics.get(k, 0) for k in metric_names]
    axes[0].bar(metric_names, values)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Test Set Metrics")
    axes[0].tick_params(axis="x", rotation=45)

    cost_names = ["cost_false_positives", "cost_false_negatives"]
    cost_vals = [metrics.get(k, 0) for k in cost_names]
    axes[1].bar(cost_names, cost_vals)
    axes[1].set_title("Business Cost Breakdown")
    axes[1].tick_params(axis="x", rotation=45)
    plt.tight_layout()
    viz_path = report_path.parent / "evaluation_metrics.png"
    plt.savefig(viz_path, dpi=100, bbox_inches="tight")
    plt.close()

    logger.info("Evaluation report saved to %s", report_path)
    return comparison


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="MLflow run ID to evaluate")
    parser.add_argument("--production-run-id", default=None)
    parser.add_argument("--output", default="data/processed/evaluation_report.json")
    args = parser.parse_args()
    generate_evaluation_report(args.run_id, Path(args.output), args.production_run_id)
