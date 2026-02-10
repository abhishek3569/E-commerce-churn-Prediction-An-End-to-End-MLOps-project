# E-commerce Customer Churn Prediction - MLOps Pipeline

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![MLflow](https://img.shields.io/badge/MLflow-2.x-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.1xx-green)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🎯 Project Overview

Production-ready MLOps pipeline for predicting customer churn in e-commerce using automated training, experiment tracking, model registry, and REST API deployment.

**Key Features:**
- ✅ End-to-end ML pipeline with MLflow tracking
- ✅ Automated model training and evaluation
- ✅ Model versioning and registry management
- ✅ REST API for real-time predictions
- ✅ Data drift monitoring
- ✅ Airflow orchestration for scheduled training
- ✅ Docker containerization

## 📊 Business Impact

- **Churn Prediction Accuracy**: 85%+ AUC-ROC
- **API Response Time**: <100ms
- **Automated Retraining**: Weekly schedule
- **Model Versioning**: Complete lineage tracking

## 🏗️ Architecture

<img width="4170" height="2985" alt="mlops_platform_architecture" src="https://github.com/user-attachments/assets/19430f70-a7c4-4eba-82e3-51cbdb711658" />


**Technology Stack:**
- **ML Framework**: Scikit-learn, XGBoost, LightGBM
- **Experiment Tracking**: MLflow
- **Orchestration**: Apache Airflow
- **API**: FastAPI, Uvicorn
- **Containerization**: Docker, Docker Compose
- **Data Processing**: Pandas, NumPy
- **Monitoring**: Prometheus, Grafana (optional)

## 📂 Project Structure
```
ecommerce-churn-mlops/
├── data/
│   ├── raw/              # Raw customer data
│   ├── processed/        # Processed features
│   └── synthetic_data_generator.py
├── src/
│   ├── features/         # Feature engineering
│   ├── models/           # Model training & evaluation
│   ├── serving/          # FastAPI application
│   └── monitoring/       # Drift detection
├── airflow/
│   └── dags/             # Airflow DAGs
├── notebooks/            # Exploratory analysis
├── tests/                # Unit & integration tests
├── docs/                 # Documentation
└── docker-compose.yml    # Container orchestration
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Docker & Docker Compose (optional)
- pip

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/ecommerce-churn-mlops.git
cd ecommerce-churn-mlops

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Running the Pipeline

**1. Start MLflow Server**
```bash
mlflow server --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlflow-artifacts
```

**2. Initialize MLflow**
```bash
python scripts/init_mlflow.py
```

**3. Generate Synthetic Data**
```bash
python data/synthetic_data_generator.py
```

**4. Train Model**
```bash
python -m src.models.train
```

**5. Start Prediction API**
```bash
uvicorn src.serving.api:app --host 127.0.0.1 --port 8000
```

**6. Access Services**
- MLflow UI: http://127.0.0.1:5000
- API Docs: http://127.0.0.1:8000/docs
- Predictions: http://127.0.0.1:8000/predict

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.78 | 0.75 | 0.72 | 0.73 | 0.82 |
| Random Forest | 0.84 | 0.82 | 0.80 | 0.81 | 0.89 |
| **XGBoost** | **0.87** | **0.85** | **0.83** | **0.84** | **0.92** |
| LightGBM | 0.86 | 0.84 | 0.82 | 0.83 | 0.91 |

## 🔍 Key Features

### 1. Feature Engineering
- RFM (Recency, Frequency, Monetary) analysis
- Engagement scoring (web visits, email interaction, app usage)
- Customer satisfaction metrics
- CLV (Customer Lifetime Value) estimation

### 2. MLflow Integration
- Experiment tracking with hyperparameters
- Metric logging (accuracy, AUC-ROC, F1)
- Model registry with staging/production stages
- Artifact storage (models, plots, feature importance)

### 3. Model Serving API
**Endpoints:**
- `POST /predict` - Single customer prediction
- `POST /predict_batch` - Batch predictions
- `GET /health` - Service health check
- `GET /model_info` - Current model metadata

**Example Request:**
```python
import requests

data = {
    "customer_id": "CUST_001",
    "total_purchases": 25,
    "days_since_last_purchase": 15,
    "email_open_rate": 0.65,
    # ... other features
}

response = requests.post("http://127.0.0.1:8000/predict", json=data)
print(response.json())
# Output: {"customer_id": "CUST_001", "churn_probability": 0.23, "prediction": 0}
```

### 4. Automated Pipeline (Airflow)
- **Data Generation DAG**: Daily synthetic data generation
- **Training DAG**: Weekly model retraining
- **Monitoring DAG**: Daily drift detection

## 📊 Monitoring & Observability

- **Data Drift Detection**: PSI (Population Stability Index) monitoring
- **Model Performance**: Prediction distribution tracking
- **API Metrics**: Latency, throughput, error rates

## 🧪 Testing
```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_features.py
pytest tests/test_models.py
pytest tests/test_api.py
```

## 📝 Documentation

- [Architecture Documentation](docs/architecture.md)
- [Setup Guide](docs/setup.md)
- [API Documentation](docs/api_documentation.md)
- [Model Card](docs/model_card.md)

## 🎓 What I Learned

- Implementing end-to-end MLOps workflows
- Model versioning and experiment tracking with MLflow
- Building production-ready ML APIs with FastAPI
- Orchestrating ML pipelines with Airflow
- Data drift detection and model monitoring
- Containerization and deployment best practices

## 🔮 Future Improvements

- [ ] Add Kubernetes deployment manifests
- [ ] Implement A/B testing framework
- [ ] Add real-time streaming predictions (Kafka)
- [ ] Integration with cloud platforms (AWS SageMaker, GCP Vertex AI)
- [ ] Enhanced monitoring with Prometheus/Grafana
- [ ] CI/CD pipeline with GitHub Actions
- [ ] Model explainability with SHAP

## 📄 License

MIT License - see [LICENSE](LICENSE) file

