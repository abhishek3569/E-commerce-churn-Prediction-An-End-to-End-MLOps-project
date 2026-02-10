from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import AnyHttpUrl, Field, PositiveInt
from pydantic_settings import BaseSettings


class DataSettings(BaseSettings):
    raw_data_dir: Path = Field(default=Path("data/raw"))
    processed_data_dir: Path = Field(default=Path("data/processed"))
    features_train_file: str = "features_train.parquet"
    features_val_file: str = "features_val.parquet"
    features_test_file: str = "features_test.parquet"


class ModelHyperParams(BaseSettings):
    # Logistic Regression
    logreg_c: float = 1.0
    logreg_max_iter: PositiveInt = 500

    # Random Forest
    rf_n_estimators: PositiveInt = 300
    rf_max_depth: Optional[int] = None
    rf_min_samples_split: PositiveInt = 2

    # XGBoost
    xgb_n_estimators: PositiveInt = 400
    xgb_learning_rate: float = 0.05
    xgb_max_depth: PositiveInt = 5

    # LightGBM
    lgbm_n_estimators: PositiveInt = 400
    lgbm_learning_rate: float = 0.05
    lgbm_num_leaves: PositiveInt = 31


class MLflowSettings(BaseSettings):
    tracking_uri: str = Field(default="http://localhost:5000", env="MLFLOW_TRACKING_URI")
    experiment_name: str = "ecommerce_churn_experiment"
    model_registry_uri: Optional[AnyHttpUrl] = None


class RegistrySettings(BaseSettings):
    model_name: str = "ecommerce_churn_model"
    min_auc_roc: float = Field(default=0.75, env="MODEL_MIN_AUC")
    max_churn_rate_delta: float = Field(default=0.2, env="MODEL_CHURN_RATE_DRIFT_THRESHOLD")


class FeatureEngineeringSettings(BaseSettings):
    numerical_scaler: str = "standard"  # or "robust"
    categorical_encoding: str = "onehot"
    rfm_recency_weight: float = 0.3
    rfm_frequency_weight: float = 0.3
    rfm_monetary_weight: float = 0.4
    clv_months: PositiveInt = 12


class APISettings(BaseSettings):
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: PositiveInt = Field(default=8000, env="API_PORT")
    reload: bool = False


class MonitoringSettings(BaseSettings):
    drift_psi_threshold: float = Field(default=0.25, env="DRIFT_PSI_THRESHOLD")
    latency_threshold_ms: PositiveInt = Field(default=100, env="LATENCY_THRESHOLD_MS")
    churn_rate_change_threshold: float = Field(default=0.2, env="MODEL_CHURN_RATE_DRIFT_THRESHOLD")


class TrainingScheduleSettings(BaseSettings):
    cron: str = Field(default="0 3 * * 0", env="TRAINING_SCHEDULE_CRON")
    min_new_rows_for_retrain: PositiveInt = 10_000


class Settings(BaseSettings):
    """
    Central application configuration using Pydantic.
    Values can be overridden via environment variables or .env files.
    """

    env: str = Field(default="local")

    data: DataSettings = DataSettings()
    model_hyperparams: ModelHyperParams = ModelHyperParams()
    mlflow: MLflowSettings = MLflowSettings()
    registry: RegistrySettings = RegistrySettings()
    features: FeatureEngineeringSettings = FeatureEngineeringSettings()
    api: APISettings = APISettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    training_schedule: TrainingScheduleSettings = TrainingScheduleSettings()

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"


@lru_cache()
def get_settings() -> Settings:
    """
    Cached access to application settings.
    """
    return Settings()


__all__: List[str] = ["Settings", "get_settings"]

