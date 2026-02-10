"""
Model registry: promotion logic, validation, versioning, and rollback.
"""
import logging
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd

from src.config import get_settings

logger = logging.getLogger(__name__)


def _check_auc_threshold(run_id: str) -> Tuple[bool, float]:
    """Validate model meets minimum AUC-ROC threshold."""
    client = mlflow.MlflowClient()
    run = client.get_run(run_id)
    metrics = run.data.metrics
    auc = metrics.get("val_auc_roc") or metrics.get("auc_roc", 0.0)
    min_auc = get_settings().registry.min_auc_roc
    return auc >= min_auc, float(auc)


def _check_better_than_production(run_id: str) -> Tuple[bool, Optional[float]]:
    """Check if new model outperforms current production on AUC-ROC."""
    client = mlflow.MlflowClient()
    model_name = get_settings().registry.model_name

    try:
        prod_version = client.get_latest_versions(model_name, stages=["Production"])
        if not prod_version:
            return True, None
        prod_run_id = prod_version[0].run_id
        prod_run = client.get_run(prod_run_id)
        prod_auc = prod_run.data.metrics.get("val_auc_roc") or prod_run.data.metrics.get("auc_roc", 0.0)
    except Exception:
        return True, None

    new_run = client.get_run(run_id)
    new_auc = new_run.data.metrics.get("val_auc_roc") or new_run.data.metrics.get("auc_roc", 0.0)
    return new_auc >= prod_auc, float(prod_auc)


def _check_prediction_bias(
    run_id: str,
    test_df: pd.DataFrame,
    segment_col: Optional[str] = None,
    max_auc_gap: float = 0.1,
) -> Tuple[bool, Dict]:
    """
    Check for significant bias across segments.
    Returns (passed, segment_metrics).
    """
    if segment_col is None or segment_col not in test_df.columns:
        return True, {}

    client = mlflow.MlflowClient()
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.pyfunc.load_model(model_uri)

    X = test_df.drop(columns=["churned"], errors="ignore")
    y_true = test_df["churned"].values if "churned" in test_df.columns else None

    segments = test_df[segment_col].unique()
    segment_aucs: Dict[str, float] = {}

    for seg in segments:
        mask = test_df[segment_col] == seg
        X_seg = X[mask]
        preds = model.predict(X_seg)
        if hasattr(preds, "__len__") and len(np.array(preds).shape) > 1:
            preds = np.array(preds)[:, 1]
        if y_true is not None:
            y_seg = y_true[mask]
            if len(np.unique(y_seg)) > 1:
                from sklearn.metrics import roc_auc_score
                segment_aucs[str(seg)] = float(roc_auc_score(y_seg, preds))

    if len(segment_aucs) < 2:
        return True, segment_aucs

    aucs = list(segment_aucs.values())
    gap = max(aucs) - min(aucs)
    return gap <= max_auc_gap, segment_aucs


def validate_for_promotion(
    run_id: str,
    test_df: Optional[pd.DataFrame] = None,
) -> Tuple[bool, List[str]]:
    """
    Run all validation checks before promoting model.
    Returns (passed, list of failure reasons).
    """
    failures: List[str] = []

    ok, auc = _check_auc_threshold(run_id)
    if not ok:
        failures.append(f"AUC-ROC {auc:.4f} below threshold {get_settings().registry.min_auc_roc}")

    ok, prod_auc = _check_better_than_production(run_id)
    if not ok and prod_auc is not None:
        failures.append(f"New model does not beat production AUC {prod_auc:.4f}")

    if test_df is not None:
        ok, seg_metrics = _check_prediction_bias(run_id, test_df)
        if not ok:
            failures.append(f"Segment bias detected: {seg_metrics}")

    return len(failures) == 0, failures


def promote_to_staging(run_id: str) -> str:
    """Transition model to Staging in registry."""
    client = mlflow.MlflowClient()
    model_name = get_settings().registry.model_name
    result = client.search_model_versions(f"run_id='{run_id}'")
    if not result:
        raise ValueError(f"No model version found for run_id={run_id}")

    version = result[0]
    client.transition_model_version_stage(model_name, version.version, "Staging")
    logger.info("Promoted run %s to Staging (version %s)", run_id, version.version)
    return version.version


def promote_to_production(run_id: str) -> str:
    """Transition model to Production and archive previous production."""
    client = mlflow.MlflowClient()
    model_name = get_settings().registry.model_name

    # Archive current production
    try:
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        for pv in prod_versions:
            client.transition_model_version_stage(model_name, pv.version, "Archived")
    except Exception as e:
        logger.debug("No existing production to archive: %s", e)

    # Promote new version
    result = client.search_model_versions(f"run_id='{run_id}'")
    if not result:
        # Register if not yet registered
        mlflow.register_model(f"runs:/{run_id}/model", model_name)
        result = client.search_model_versions(f"run_id='{run_id}'")

    if not result:
        raise ValueError(f"No model version found for run_id={run_id}")

    version = result[0]
    client.transition_model_version_stage(model_name, version.version, "Production")
    logger.info("Promoted run %s to Production (version %s)", run_id, version.version)
    return version.version


def rollback_to_version(version: str) -> None:
    """Rollback production to a specific version."""
    client = mlflow.MlflowClient()
    model_name = get_settings().registry.model_name

    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    for pv in prod_versions:
        client.transition_model_version_stage(model_name, pv.version, "Archived")

    client.transition_model_version_stage(model_name, version, "Production")
    logger.info("Rolled back to version %s", version)


def get_production_model_uri() -> Optional[str]:
    """Return model URI for current production model."""
    client = mlflow.MlflowClient()
    model_name = get_settings().registry.model_name
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if versions:
            return f"models:/{model_name}/Production"
    except Exception:
        pass
    return None
