# Architecture

## System Overview

The E-commerce Churn Prediction MLOps pipeline is a production-ready batch ML system that generates synthetic customer data, engineers features, trains churn prediction models, serves predictions via API, and monitors for drift and degradation.

## Architecture Diagram (Conceptual)

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Data Generator │────>│  Raw Data (CSV)  │────>│ Feature Engine  │
│  (Synthetic)    │     │  data/raw/       │     │ + Validation    │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          v
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Model API      │<────│  MLflow Registry │<────│  Model Training │
│  (FastAPI)      │     │  + Artifacts     │     │  (4 model types)│
└────────┬────────┘     └──────────────────┘     └─────────────────┘
         │
         v
┌─────────────────┐     ┌──────────────────┐
│  Prometheus     │<────│  Metrics         │
│  + Grafana      │     │  /metrics        │
└─────────────────┘     └──────────────────┘

         ┌──────────────────────────────────────┐
         │  Airflow Orchestration                │
         │  - data_generation_dag (daily 2 AM)   │
         │  - training_pipeline_dag (Sun 3 AM)   │
         │  - monitoring_dag (daily 6 AM)        │
         └──────────────────────────────────────┘
```

## Data Flow

1. **Data Generation**: Synthetic data generator produces customer records with realistic churn patterns (100K records for initial, 1K for daily incremental). Output: `data/raw/customers_YYYYMMDD.csv`.
2. **Feature Engineering**: Raw CSV → derived features (RFM, engagement scores, churn risk indicators) → preprocessing (impute, encode, scale) → `data/processed/features.parquet` and `feature_pipeline.joblib`.
3. **Training**: Load features → 70/15/15 stratified split → train Logistic Regression, Random Forest, XGBoost, LightGBM → log to MLflow → save best model.
4. **Serving**: Load production model + feature pipeline → accept raw features via `/predict` or `/predict_batch` → preprocess → predict → return probability + binary label.
5. **Monitoring**: Compute PSI for numerical drift, chi-square for categorical drift, churn rate deltas; export to Prometheus; alert on threshold breaches.

## Component Interactions

- **MLflow**: Tracks experiments, stores models, manages registry (Staging → Production).
- **PostgreSQL**: Backend for MLflow and Airflow metadata.
- **Airflow**: Schedules DAGs; DAGs import project `src` and `data` modules.
- **FastAPI**: Serves predictions; loads model from MLflow registry or latest run.
- **Prometheus**: Scrapes `/metrics` from the API.
- **Grafana**: Dashboards over Prometheus metrics.

## Technology Stack

| Component     | Technology        | Rationale                          |
|---------------|-------------------|------------------------------------|
| ML Framework  | scikit-learn, XGBoost, LightGBM | Strong performance, MLflow integration |
| Experiment Tracking | MLflow        | Standard for model versioning and registry |
| API           | FastAPI           | Fast, async, OpenAPI docs          |
| Orchestration | Apache Airflow    | Mature DAG scheduler               |
| Monitoring    | Prometheus + Grafana | Industry standard observability |
| Config        | Pydantic Settings | Type-safe, env-based configuration |

## Scalability Considerations

- **Data volume**: Feature engineering and training are batch jobs; for 1M+ rows, consider Spark or Dask.
- **API throughput**: Stateless FastAPI; scale horizontally behind a load balancer.
- **MLflow**: Use S3/GCS for artifact storage for large models.
- **Airflow**: Move to CeleryExecutor or KubernetesExecutor for distributed task execution.
