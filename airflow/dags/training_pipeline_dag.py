"""
Airflow DAG: Weekly model training and promotion.
Schedule: Weekly on Sunday at 3 AM
"""
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _load_latest_data(**context):
    import pandas as pd

    raw_dir = _PROJECT_ROOT / "data" / "raw"
    csvs = list(raw_dir.glob("customers_*.csv"))
    if not csvs:
        raise ValueError("No raw customer data found in data/raw")
    latest = max(csvs, key=lambda p: p.stat().st_mtime)
    context["ti"].xcom_push(key="raw_path", value=str(latest))
    df = pd.read_csv(latest)
    if len(df) < 1000:
        raise ValueError(f"Data volume {len(df)} too low for training")
    return str(latest)


def _run_feature_engineering(**context):
    from src.features.feature_engineering import engineer_features

    raw_path = context["ti"].xcom_pull(task_ids="load_latest_data", key="raw_path")
    if not raw_path:
        raise ValueError("No raw path from load_latest_data")
    engineer_features(Path(raw_path), output_dir=_PROJECT_ROOT / "data" / "processed")
    return "Feature engineering completed"


def _validate_data(**context):
    from src.features.feature_validation import validate_data_quality
    import pandas as pd

    settings = __import__("src.config", fromlist=["get_settings"]).get_settings()
    features_path = _PROJECT_ROOT / settings.data.processed_data_dir / "features.parquet"
    if not features_path.exists():
        raise ValueError("features.parquet not found after feature engineering")
    df = pd.read_parquet(features_path)
    result = validate_data_quality(df)
    if not result.schema_valid:
        raise ValueError("Data validation failed")
    return "Validation passed"


def _train_models(**context):
    from src.models.train import run_training_pipeline

    report = run_training_pipeline()
    best_run_id = report.get("best_run_id")
    context["ti"].xcom_push(key="best_run_id", value=best_run_id)
    return best_run_id


def _evaluate_model(**context):
    from src.models.evaluate import generate_evaluation_report

    run_id = context["ti"].xcom_pull(task_ids="train_models", key="best_run_id")
    if not run_id:
        raise ValueError("No best run id from training")
    generate_evaluation_report(run_id)
    context["ti"].xcom_push(key="run_id", value=run_id)
    return run_id


def _validate_for_promotion(**context):
    from src.models.model_registry import validate_for_promotion
    import pandas as pd

    run_id = context["ti"].xcom_pull(task_ids="evaluate_model", key="run_id")
    if not run_id:
        raise ValueError("No run_id from evaluation")
    settings = __import__("src.config", fromlist=["get_settings"]).get_settings()
    test_path = _PROJECT_ROOT / settings.data.processed_data_dir / settings.data.features_test_file
    test_df = pd.read_parquet(test_path) if test_path.exists() else None
    passed, failures = validate_for_promotion(run_id, test_df)
    if not passed:
        raise ValueError(f"Promotion validation failed: {failures}")
    return "Validation passed"


def _promote_to_staging(**context):
    from src.models.model_registry import promote_to_staging

    run_id = context["ti"].xcom_pull(task_ids="evaluate_model", key="run_id")
    promote_to_staging(run_id)
    return "Promoted to Staging"


def _promote_to_production(**context):
    from src.models.model_registry import promote_to_production

    run_id = context["ti"].xcom_pull(task_ids="evaluate_model", key="run_id")
    promote_to_production(run_id)
    return "Promoted to Production"


def _notify_success(**context):
    # Placeholder: integrate with Slack/email
    return "Training pipeline completed successfully"


default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": True,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="training_pipeline_dag",
    default_args=default_args,
    description="Weekly model training and promotion",
    schedule_interval="0 3 * * 0",  # Sunday 3 AM
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["training", "mlflow"],
) as dag:
    load_data = PythonOperator(
        task_id="load_latest_data",
        python_callable=_load_latest_data,
        provide_context=True,
    )
    feature_eng = PythonOperator(
        task_id="feature_engineering",
        python_callable=_run_feature_engineering,
        provide_context=True,
    )
    validate_data_task = PythonOperator(
        task_id="validate_data",
        python_callable=_validate_data,
        provide_context=True,
    )
    train_models = PythonOperator(
        task_id="train_models",
        python_callable=_train_models,
        provide_context=True,
    )
    evaluate_model = PythonOperator(
        task_id="evaluate_model",
        python_callable=_evaluate_model,
        provide_context=True,
    )
    validate_promotion = PythonOperator(
        task_id="validate_promotion",
        python_callable=_validate_for_promotion,
        provide_context=True,
    )
    promote_staging = PythonOperator(
        task_id="promote_staging",
        python_callable=_promote_to_staging,
        provide_context=True,
    )
    promote_production = PythonOperator(
        task_id="promote_production",
        python_callable=_promote_to_production,
        provide_context=True,
    )
    notify = PythonOperator(
        task_id="notify",
        python_callable=_notify_success,
        provide_context=True,
    )

    load_data >> feature_eng >> validate_data_task >> train_models >> evaluate_model >> validate_promotion
    validate_promotion >> promote_staging >> promote_production >> notify
