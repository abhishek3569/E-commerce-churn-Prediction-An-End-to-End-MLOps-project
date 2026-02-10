"""Tests for FastAPI serving endpoints."""
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.serving.api import app
from src.serving.schemas import CustomerFeatures, PredictRequest


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def sample_features() -> dict:
    return {
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
        "referrals_made": 2,
    }


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    assert data["status"] == "healthy"


def test_model_info(client: TestClient) -> None:
    r = client.get("/model_info")
    assert r.status_code == 200
    data = r.json()
    assert "loaded" in data


def test_predict_endpoint(client: TestClient, sample_features: dict) -> None:
    r = client.post(
        "/predict",
        json={"features": sample_features},
    )
    # 200 if model loaded, 503 if no model available
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        data = r.json()
        assert "churn_probability" in data
        assert "churned" in data
        assert 0 <= data["churn_probability"] <= 1


def test_predict_validation_error(client: TestClient) -> None:
    r = client.post("/predict", json={"features": {"invalid": "data"}})
    assert r.status_code == 422


def test_metrics_endpoint(client: TestClient) -> None:
    r = client.get("/metrics")
    assert r.status_code == 200
    assert b"churn" in r.content or b"#" in r.content
