"""
Airflow DAG: Daily monitoring - drift and prediction distribution analysis.
Schedule: Daily at 6 AM
"""
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _calculate_drift(**context):
    from src.monitoring.metrics import generate_monitoring_report

    base_dir = _PROJECT_ROOT / "data" / "processed"
    baseline = base_dir / "features_train.parquet"
    current = base_dir / "features.parquet"
    if not baseline.exists():
        baseline = list((_PROJECT_ROOT / "data" / "raw").glob("customers_*.csv"))
        baseline = sorted(baseline)[-1] if baseline else None
    if not current.exists():
        current = baseline
    if baseline and current:
        report = generate_monitoring_report(
            baseline, current,
            output_path=base_dir / "monitoring_report.txt",
        )
        context["ti"].xcom_push(key="alerts", value=report.alerts)
        return len(report.alerts)
    return 0


def _analyze_prediction_distributions(**context):
    # Placeholder: would query API/prometheus for prediction stats
    return "Distribution analysis complete"


def _compare_baselines(**context):
    return "Baseline comparison complete"


def _send_alerts(**context):
    alerts = context["ti"].xcom_pull(task_ids="calculate_drift", key="alerts") or []
    if alerts:
        # Placeholder: send to Slack/email
        for a in alerts:
            print(f"ALERT: {a}")
    return f"Processed {len(alerts)} alerts"


default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="monitoring_dag",
    default_args=default_args,
    description="Daily drift and monitoring",
    schedule_interval="0 6 * * *",  # 6 AM daily
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["monitoring", "drift"],
) as dag:
    calc_drift = PythonOperator(
        task_id="calculate_drift",
        python_callable=_calculate_drift,
        provide_context=True,
    )
    analyze_dist = PythonOperator(
        task_id="analyze_distributions",
        python_callable=_analyze_prediction_distributions,
        provide_context=True,
    )
    compare_base = PythonOperator(
        task_id="compare_baselines",
        python_callable=_compare_baselines,
        provide_context=True,
    )
    send_alerts = PythonOperator(
        task_id="send_alerts",
        python_callable=_send_alerts,
        provide_context=True,
    )

    calc_drift >> analyze_dist >> compare_base >> send_alerts
