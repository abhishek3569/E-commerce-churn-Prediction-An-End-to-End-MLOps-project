"""
FastAPI model serving application for churn predictions.
"""
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, generate_latest

from src.config import get_settings
from src.features.feature_engineering import _add_derived_features
from src.models.model_registry import get_production_model_uri
from src.serving.schemas import (
    BatchPredictRequest,
    BatchPredictionResult,
    CustomerFeatures,
    HealthResponse,
    ModelInfo,
    PredictionResult,
    PredictRequest,
)

logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTION_COUNTER = Counter("churn_predictions_total", "Total predictions served")
PREDICTION_LATENCY = Histogram("churn_prediction_latency_seconds", "Prediction latency")
PREDICTION_PROB_HISTOGRAM = Histogram(
    "churn_prediction_probability", "Distribution of churn probabilities", buckets=[0.1 * i for i in range(11)]
)

# Global model and pipeline
_model: Optional[Any] = None
_preprocessor: Optional[Any] = None
_model_version: Optional[str] = None


def _features_to_dataframe(f: CustomerFeatures) -> pd.DataFrame:
    """Convert CustomerFeatures to a single-row DataFrame for preprocessing."""
    return pd.DataFrame([{
        "account_age_days": f.account_age_days,
        "customer_segment": f.customer_segment,
        "signup_channel": f.signup_channel,
        "total_purchases": f.total_purchases,
        "total_revenue": f.total_revenue,
        "avg_order_value": f.avg_order_value,
        "days_since_last_purchase": f.days_since_last_purchase,
        "purchase_frequency": f.purchase_frequency,
        "website_visits_last_30days": f.website_visits_last_30days,
        "email_open_rate": f.email_open_rate,
        "email_click_rate": f.email_click_rate,
        "app_usage_minutes_last_30days": f.app_usage_minutes_last_30days,
        "customer_service_contacts": f.customer_service_contacts,
        "favorite_category": f.favorite_category,
        "number_of_categories_purchased": f.number_of_categories_purchased,
        "returns_count": f.returns_count,
        "return_rate": f.return_rate,
        "loyalty_points_balance": f.loyalty_points_balance,
        "discount_usage_rate": f.discount_usage_rate,
        "referrals_made": f.referrals_made,
    }])


def _load_native_model(uri: str) -> Any:
    """Load model via native flavor to support predict_proba."""
    import mlflow.sklearn
    import mlflow.xgboost
    import mlflow.lightgbm
    for loader in [mlflow.sklearn.load_model, mlflow.xgboost.load_model, mlflow.lightgbm.load_model]:
        try:
            return loader(uri)
        except Exception:
            continue
    return mlflow.pyfunc.load_model(uri)


def _load_model() -> bool:
    """Load production model and feature pipeline. Returns True if loaded."""
    global _model, _preprocessor, _model_version
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)

    model_uri = get_production_model_uri()
    if not model_uri:
        pipeline_path = Path(settings.data.processed_data_dir) / "feature_pipeline.joblib"
        if pipeline_path.exists():
            _preprocessor = joblib.load(pipeline_path)
        experiment = mlflow.get_experiment_by_name(settings.mlflow.experiment_name)
        if experiment:
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["metrics.val_auc_roc DESC"],
                max_results=1,
            )
            if len(runs) > 0:
                run_id = runs.iloc[0]["run_id"]
                model_uri = f"runs:/{run_id}/model"
        if not model_uri:
            logger.warning("No production model or recent run found")
            return False

    try:
        _model = _load_native_model(model_uri)
        _model_version = model_uri.split("/")[-1] if "/" in model_uri else model_uri
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        return False

    if _preprocessor is None:
        pipeline_path = Path(settings.data.processed_data_dir) / "feature_pipeline.joblib"
        if pipeline_path.exists():
            _preprocessor = joblib.load(pipeline_path)
        else:
            logger.error("Feature pipeline not found")
            return False

    return True


def _predict_proba(features_df: pd.DataFrame) -> np.ndarray:
    """Run full preprocessing + model prediction."""
    df = _add_derived_features(features_df)
    X = df.drop(columns=["churn_probability"], errors="ignore")
    X_transformed = _preprocessor.transform(X)
    if hasattr(_model, "predict_proba"):
        proba = _model.predict_proba(X_transformed)
        return np.asarray(proba)[:, 1].flatten()
    preds = _model.predict(X_transformed)
    return np.asarray(preds).flatten().astype(float)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    _load_model()
    yield
    # Cleanup if needed
    pass


app = FastAPI(title="E-commerce Churn Prediction API", version="1.0", lifespan=lifespan)


@app.post("/predict", response_model=PredictionResult)
async def predict(request: PredictRequest) -> PredictionResult:
    """Single customer churn prediction."""
    if _model is None or _preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    start = time.perf_counter()
    try:
        df = _features_to_dataframe(request.features)
        proba = _predict_proba(df)[0]
        churned = proba >= 0.5
        confidence = max(proba, 1 - proba)
        PREDICTION_COUNTER.inc()
        PREDICTION_PROB_HISTOGRAM.observe(proba)
        return PredictionResult(
            churn_probability=round(proba, 4),
            churned=bool(churned),
            confidence=round(confidence, 4),
            customer_id=request.customer_id,
        )
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        PREDICTION_LATENCY.observe(time.perf_counter() - start)


@app.post("/predict_batch", response_model=BatchPredictionResult)
async def predict_batch(request: BatchPredictRequest) -> BatchPredictionResult:
    """Batch churn predictions."""
    if _model is None or _preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    start = time.perf_counter()
    try:
        rows: List[Dict[str, Any]] = []
        customer_ids: List[Optional[str]] = []
        for r in request.customers:
            rows.append(_features_to_dataframe(r.features).iloc[0].to_dict())
            customer_ids.append(r.customer_id)
        df = pd.DataFrame(rows)
        probas = _predict_proba(df)
        results = [
            PredictionResult(
                churn_probability=round(float(p), 4),
                churned=p >= 0.5,
                confidence=round(max(p, 1 - p), 4),
                customer_id=cid,
            )
            for p, cid in zip(probas, customer_ids)
        ]
        PREDICTION_COUNTER.inc(len(results))
        for p in probas:
            PREDICTION_PROB_HISTOGRAM.observe(p)
        return BatchPredictionResult(predictions=results, count=len(results))
    except Exception as e:
        logger.exception("Batch prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        PREDICTION_LATENCY.observe(time.perf_counter() - start)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check."""
    return HealthResponse(
        status="healthy",
        model_loaded=_model is not None and _preprocessor is not None,
    )


@app.get("/model_info", response_model=ModelInfo)
async def model_info() -> ModelInfo:
    """Current model version and metadata."""
    settings = get_settings()
    return ModelInfo(
        model_version=_model_version,
        model_name=settings.registry.model_name,
        loaded=_model is not None,
    )


@app.get("/metrics")
async def metrics() -> bytes:
    """Prometheus metrics."""
    return generate_latest()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(app, host=settings.api.host, port=settings.api.port)
