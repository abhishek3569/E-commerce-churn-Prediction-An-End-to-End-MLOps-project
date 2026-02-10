"""Tests for model training, evaluation, and registry."""
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from src.models.train import _split_data, _get_model_and_params, _compute_metrics
from src.models.evaluate import evaluate_on_test
from src.models.model_registry import validate_for_promotion, _check_auc_threshold
from src.features.feature_engineering import TARGET_COLUMN


@pytest.fixture
def sample_features_df() -> pd.DataFrame:
    n = 500
    X = np.random.randn(n, 5)
    y = (X[:, 0] + X[:, 1] + np.random.randn(n) * 0.5 > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df[TARGET_COLUMN] = y
    return df


def test_split_data(sample_features_df: pd.DataFrame) -> None:
    X_train, y_train, X_val, y_val, X_test, y_test = _split_data(sample_features_df)
    assert len(X_train) + len(X_val) + len(X_test) == len(sample_features_df)
    assert 0.6 < len(X_train) / len(sample_features_df) < 0.8
    assert len(X_val) > 0 and len(X_test) > 0


def test_get_model_and_params() -> None:
    for model_type in ["logistic_regression", "random_forest", "xgboost", "lightgbm"]:
        model, params = _get_model_and_params(model_type)
        assert model is not None
        assert isinstance(params, dict)
        assert len(params) >= 1


def test_compute_metrics() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_proba = np.array([0.1, 0.9, 0.8, 0.9])
    m = _compute_metrics(y_true, y_pred, y_proba)
    assert "accuracy" in m
    assert "auc_roc" in m
    assert 0 <= m["auc_roc"] <= 1


def test_validate_for_promotion_no_production() -> None:
    with patch("src.models.model_registry.mlflow") as mock_mlflow:
        client = MagicMock()
        client.get_run.return_value.data.metrics = {"val_auc_roc": 0.85}
        mock_mlflow.MlflowClient.return_value = client
        passed, failures = validate_for_promotion("fake_run_id")
        assert isinstance(passed, bool)
        assert isinstance(failures, list)
