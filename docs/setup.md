# Setup Guide

## Prerequisites

- Python 3.11+
- Docker and Docker Compose
- (Optional) Conda or venv for local development

## Environment Setup

### 1. Clone and Install Dependencies

```bash
cd ecommerce-churn-mlops
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Variables

Copy `env.example` to `.env` and adjust:

```bash
cp env.example .env
```

Key variables:

- `MLFLOW_TRACKING_URI`: MLflow server URL (default: http://localhost:5000)
- `MODEL_MIN_AUC`: Minimum AUC for model promotion (default: 0.75)
- `DRIFT_PSI_THRESHOLD`: PSI threshold for drift alerts (default: 0.25)

## Docker Compose Up

Start the full stack:

```bash
docker compose up -d
```

Services:

| Service           | Port | Description                |
|-------------------|------|----------------------------|
| MLflow            | 5000 | Experiment tracking        |
| Model API         | 8000 | Prediction endpoints       |
| Airflow Webserver | 8080 | DAG UI (admin/admin)       |
| Prometheus        | 9090 | Metrics                    |
| Grafana           | 3000 | Dashboards (admin/admin)   |
| PostgreSQL        | 5432 | MLflow + Airflow metadata  |

Wait for `airflow-init` to finish before using Airflow. Initialize the database once:

```bash
docker compose run --rm airflow-init
```

## Airflow Configuration

1. Open http://localhost:8080
2. Login: admin / admin
3. Enable DAGs: `data_generation_dag`, `training_pipeline_dag`, `monitoring_dag`
4. Ensure project root is mounted; DAGs use `_PROJECT_ROOT` for paths

## MLflow UI Access

- URL: http://localhost:5000
- Experiments: `ecommerce_churn_experiment`
- Models: `ecommerce_churn_model`

## API Testing Examples

### Health Check

```bash
curl http://localhost:8000/health
```

### Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "account_age_days": 365,
      "customer_segment": "Gold",
      "signup_channel": "organic",
      "total_purchases": 10,
      "total_revenue": 2500,
      "avg_order_value": 250,
      "days_since_last_purchase": 30,
      "purchase_frequency": 1.5,
      "website_visits_last_30days": 15,
      "email_open_rate": 0.4,
      "email_click_rate": 0.2,
      "app_usage_minutes_last_30days": 120,
      "customer_service_contacts": 1,
      "favorite_category": "Electronics",
      "number_of_categories_purchased": 3,
      "returns_count": 0,
      "return_rate": 0.05,
      "loyalty_points_balance": 3000,
      "discount_usage_rate": 0.3,
      "referrals_made": 2
    }
  }'
```

## Local Run (Without Docker)

### 1. Generate Data

```bash
python data/synthetic_data_generator.py --n_customers 10000 --output_dir data/raw
```

### 2. Feature Engineering

```python
from pathlib import Path
from src.features.feature_engineering import engineer_features

latest = max(Path("data/raw").glob("customers_*.csv"), key=lambda p: p.stat().st_mtime)
engineer_features(latest)
```

### 3. Train Models

Start MLflow first:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

Then:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
python -m src.models.train
```

### 4. Run API

```bash
uvicorn src.serving.api:app --reload --port 8000
```

## Run Tests

```bash
pytest tests/ -v
```
