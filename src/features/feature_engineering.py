from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from src.config import get_settings


TARGET_COLUMN = "churned"


@dataclass
class FeatureEngineeringArtifacts:
    feature_names: List[str]
    target_name: str
    transformer_path: Path
    importance_report_path: Path


def _build_preprocessor(
    numerical_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    settings = get_settings()
    scaler_cls = StandardScaler if settings.features.numerical_scaler == "standard" else RobustScaler

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler_cls()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived business features such as RFM and engagement scores."""
    df = df.copy()

    # Recency: invert days since last purchase (higher is more recent)
    max_days = df["days_since_last_purchase"].max()
    df["recency_score"] = 1 - (df["days_since_last_purchase"] / (max_days + 1e-6))

    # Frequency score: normalized purchase frequency
    max_freq = df["purchase_frequency"].replace([np.inf, -np.inf], np.nan).max()
    df["frequency_score"] = df["purchase_frequency"] / (max_freq + 1e-6)

    # Monetary score: normalized total revenue
    max_revenue = df["total_revenue"].max()
    df["monetary_score"] = df["total_revenue"] / (max_revenue + 1e-6)

    settings = get_settings()
    df["rfm_score"] = (
        settings.features.rfm_recency_weight * df["recency_score"]
        + settings.features.rfm_frequency_weight * df["frequency_score"]
        + settings.features.rfm_monetary_weight * df["monetary_score"]
    )

    # Engagement score: composite of visits, email, and app usage
    engagement_components = [
        df["website_visits_last_30days"].fillna(0) / (df["website_visits_last_30days"].max() + 1e-6),
        df["email_open_rate"].fillna(0),
        df["email_click_rate"].fillna(0),
        df["app_usage_minutes_last_30days"].fillna(0)
        / (df["app_usage_minutes_last_30days"].max() + 1e-6),
    ]
    df["engagement_score"] = np.mean(engagement_components, axis=0)

    # Satisfaction score: fewer returns and support contacts is better
    returns_norm = df["returns_count"].fillna(0) / (df["returns_count"].max() + 1e-6)
    cs_norm = df["customer_service_contacts"].fillna(0) / (
        df["customer_service_contacts"].max() + 1e-6
    )
    df["satisfaction_score"] = 1 - 0.5 * returns_norm - 0.5 * cs_norm

    # CLV estimate: RFM * average order * projected months
    df["clv_estimate"] = (
        df["rfm_score"] * df["avg_order_value"] * get_settings().features.clv_months
    )

    # Churn risk indicators
    df["churn_risk_high_inactivity"] = (df["days_since_last_purchase"] > 180).astype(int)
    df["churn_risk_low_engagement"] = (df["website_visits_last_30days"] < 2).astype(int)
    df["churn_risk_low_email"] = (df["email_open_rate"] < 0.1).astype(int)
    df["churn_risk_high_returns"] = (df["return_rate"] > 0.3).astype(int)

    return df


def engineer_features(
    input_csv: Path,
    output_dir: Path | None = None,
    save_pipeline: bool = True,
) -> FeatureEngineeringArtifacts:
    """
    Load raw CSV, engineer features, fit preprocessing pipeline, and persist artifacts.
    """
    settings = get_settings()
    if output_dir is None:
        output_dir = settings.data.processed_data_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in input data.")

    df = _add_derived_features(df)

    y = df[TARGET_COLUMN]
    drop_cols = [
        TARGET_COLUMN,
        "churn_probability",
        "customer_id",
        "data_generation_timestamp",
        "last_purchase_date",
    ]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocessor = _build_preprocessor(numerical_features, categorical_features)
    X_transformed = preprocessor.fit_transform(X)

    # Feature names
    ohe = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    cat_feature_names = ohe.get_feature_names_out(categorical_features).tolist()
    feature_names = numerical_features + cat_feature_names

    # Mutual information as a simple feature importance proxy
    mi = mutual_info_classif(X_transformed, y, discrete_features=False)
    importance_df = pd.DataFrame({"feature": feature_names, "mutual_information": mi})
    importance_df.sort_values("mutual_information", ascending=False, inplace=True)

    transformer_path = output_dir / "feature_pipeline.joblib"
    importance_report_path = output_dir / "feature_importance.csv"

    if save_pipeline:
        joblib.dump(preprocessor, transformer_path)
        importance_df.to_csv(importance_report_path, index=False)

    # Save transformed features for training pipeline
    features_path = output_dir / "features.parquet"
    features_df = pd.DataFrame(X_transformed, columns=feature_names)
    features_df[TARGET_COLUMN] = y.reset_index(drop=True)
    features_df.to_parquet(features_path, index=False)

    return FeatureEngineeringArtifacts(
        feature_names=feature_names,
        target_name=TARGET_COLUMN,
        transformer_path=transformer_path,
        importance_report_path=importance_report_path,
    )


__all__: List[str] = ["engineer_features", "FeatureEngineeringArtifacts", "TARGET_COLUMN"]

