"""
Airflow DAG: Daily synthetic data generation and validation.
Schedule: Daily at 2 AM
"""
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

import sys
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _run_data_generation(**context):
    from data.synthetic_data_generator import _generate_dataset, _data_quality_report
    import pandas as pd
    from datetime import datetime

    n = 1000  # Incremental batch size for daily runs
    df = _generate_dataset(n, seed=int(datetime.utcnow().timestamp()) % 10000)
    out_dir = _PROJECT_ROOT / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    date_suffix = datetime.utcnow().strftime("%Y%m%d")
    csv_path = out_dir / f"customers_{date_suffix}.csv"
    df.to_csv(csv_path, index=False)
    report = _data_quality_report(df)
    report_path = out_dir / f"data_quality_report_{date_suffix}.txt"
    report_path.write_text(report, encoding="utf-8")
    context["ti"].xcom_push(key="output_path", value=str(csv_path))


def _validate_data_quality(**context):
    from src.features.feature_validation import validate_data_quality
    import pandas as pd

    output_path = context["ti"].xcom_pull(task_ids="generate_data", key="output_path")
    if not output_path:
        raise ValueError("No output path from generate_data task")
    df = pd.read_csv(output_path)
    result = validate_data_quality(df)
    if not result.schema_valid:
        raise ValueError("Schema validation failed")
    if result.missing_fraction:
        raise ValueError(f"High missing values: {result.missing_fraction}")
    return "Validation passed"


def _trigger_feature_engineering(**context):
    from src.features.feature_engineering import engineer_features

    output_path = context["ti"].xcom_pull(task_ids="generate_data", key="output_path")
    if not output_path:
        raise ValueError("No output path from generate_data task")
    engineer_features(Path(output_path), output_dir=_PROJECT_ROOT / "data" / "processed")
    return "Feature engineering completed"


default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": True,
    "email": ["admin@example.com"],
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="data_generation_dag",
    default_args=default_args,
    description="Generate and validate synthetic e-commerce data",
    schedule_interval="0 2 * * *",  # 2 AM daily
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["data", "generation"],
) as dag:
    generate_data = PythonOperator(
        task_id="generate_data",
        python_callable=_run_data_generation,
        provide_context=True,
    )
    validate_quality = PythonOperator(
        task_id="validate_quality",
        python_callable=_validate_data_quality,
        provide_context=True,
    )
    feature_engineering = PythonOperator(
        task_id="feature_engineering",
        python_callable=_trigger_feature_engineering,
        provide_context=True,
    )

    generate_data >> validate_quality >> feature_engineering
