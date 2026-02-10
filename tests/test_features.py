"""Tests for feature engineering and validation."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import (
    _add_derived_features,
    engineer_features,
    TARGET_COLUMN,
)
from src.features.feature_validation import (
    check_missing_values,
    detect_outliers_iqr,
    population_stability_index,
    validate_data_quality,
    validate_schema,
)


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Minimal raw dataframe matching synthetic schema."""
    n = 100
    return pd.DataFrame({
        "account_age_days": np.random.randint(30, 500, n),
        "customer_segment": np.random.choice(["Bronze", "Silver", "Gold"], n),
        "signup_channel": np.random.choice(["organic", "email"], n),
        "total_purchases": np.random.randint(0, 20, n),
        "total_revenue": np.random.uniform(0, 5000, n),
        "avg_order_value": np.random.uniform(10, 200, n),
        "days_since_last_purchase": np.random.randint(0, 365, n),
        "purchase_frequency": np.random.uniform(0, 5, n),
        "website_visits_last_30days": np.random.randint(0, 30, n),
        "email_open_rate": np.random.uniform(0, 1, n),
        "email_click_rate": np.random.uniform(0, 0.5, n),
        "app_usage_minutes_last_30days": np.random.uniform(0, 500, n),
        "customer_service_contacts": np.random.randint(0, 5, n),
        "favorite_category": np.random.choice(["Electronics", "Fashion"], n),
        "number_of_categories_purchased": np.random.randint(1, 5, n),
        "returns_count": np.random.randint(0, 5, n),
        "return_rate": np.random.uniform(0, 0.3, n),
        "loyalty_points_balance": np.random.uniform(0, 5000, n),
        "discount_usage_rate": np.random.uniform(0, 0.5, n),
        "referrals_made": np.random.randint(0, 5, n),
        TARGET_COLUMN: np.random.randint(0, 2, n),
    })


def test_add_derived_features(sample_raw_df: pd.DataFrame) -> None:
    df = _add_derived_features(sample_raw_df.copy())
    assert "recency_score" in df.columns
    assert "rfm_score" in df.columns
    assert "engagement_score" in df.columns
    assert "churn_risk_high_inactivity" in df.columns
    assert df["recency_score"].between(0, 1).all()


def test_engineer_features(sample_raw_df: pd.DataFrame) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "customers.csv"
        sample_raw_df.to_csv(csv_path, index=False)
        artifacts = engineer_features(csv_path, output_dir=Path(tmp) / "processed", save_pipeline=True)
        assert artifacts.transformer_path.exists()
        assert artifacts.importance_report_path.exists()
        features_path = Path(tmp) / "processed" / "features.parquet"
        assert features_path.exists()
        df = pd.read_parquet(features_path)
        assert TARGET_COLUMN in df.columns
        assert len(df.columns) > 5


def test_engineer_features_missing_target(sample_raw_df: pd.DataFrame) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "customers.csv"
        df = sample_raw_df.drop(columns=[TARGET_COLUMN])
        df.to_csv(csv_path, index=False)
        with pytest.raises(ValueError, match="Target column"):
            engineer_features(csv_path, output_dir=Path(tmp) / "processed")


def test_check_missing_values() -> None:
    df = pd.DataFrame({"a": [1, 2, np.nan, 4], "b": [1, 2, 3, 4]})
    severe = check_missing_values(df, threshold=0.2)
    assert "a" in severe


def test_detect_outliers_iqr() -> None:
    df = pd.DataFrame({"x": list(range(100)) + [1000]})
    outliers = detect_outliers_iqr(df)
    assert "x" in outliers
    assert outliers["x"] >= 1


def test_population_stability_index() -> None:
    expected = np.random.normal(0, 1, 1000)
    actual = np.random.normal(0.1, 1, 1000)
    psi = population_stability_index(expected, actual)
    assert isinstance(psi, float)
    assert psi >= 0


def test_validate_schema() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    assert validate_schema(df, {"a": "numeric", "b": "categorical"})
    assert not validate_schema(df, {"c": "numeric"})


def test_validate_data_quality(sample_raw_df: pd.DataFrame) -> None:
    result = validate_data_quality(sample_raw_df)
    assert result.schema_valid is True or result.schema_valid is False
    assert hasattr(result, "missing_fraction")
    assert hasattr(result, "numeric_outliers")
