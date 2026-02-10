"""
Pydantic schemas for API request/response validation.
"""
from typing import List, Optional

from pydantic import BaseModel, Field


# Allowed values for categorical features
CUSTOMER_SEGMENTS = ["Bronze", "Silver", "Gold", "Platinum"]
SIGNUP_CHANNELS = ["organic", "paid_social", "email", "referral", "direct"]
FAVORITE_CATEGORIES = ["Electronics", "Fashion", "Home", "Sports", "Beauty"]


class CustomerFeatures(BaseModel):
    """Input schema matching training features (raw, pre-derivation)."""

    account_age_days: int = Field(..., ge=0, le=5000)
    customer_segment: str = Field(..., description="Bronze, Silver, Gold, or Platinum")
    signup_channel: str = Field(..., description="organic, paid_social, email, referral, or direct")
    total_purchases: int = Field(..., ge=0, le=200)
    total_revenue: float = Field(..., ge=0, le=100_000)
    avg_order_value: float = Field(..., ge=0, le=1000)
    days_since_last_purchase: int = Field(..., ge=0, le=500)
    purchase_frequency: float = Field(..., ge=0)
    website_visits_last_30days: int = Field(..., ge=0, le=200)
    email_open_rate: float = Field(..., ge=0, le=1)
    email_click_rate: float = Field(..., ge=0, le=1)
    app_usage_minutes_last_30days: float = Field(..., ge=0, le=2000)
    customer_service_contacts: int = Field(..., ge=0, le=50)
    favorite_category: str = Field(..., description="Electronics, Fashion, Home, Sports, or Beauty")
    number_of_categories_purchased: int = Field(..., ge=1, le=20)
    returns_count: int = Field(..., ge=0, le=50)
    return_rate: float = Field(..., ge=0, le=1)
    loyalty_points_balance: float = Field(..., ge=0, le=50_000)
    discount_usage_rate: float = Field(..., ge=0, le=1)
    referrals_made: int = Field(..., ge=0, le=50)


class PredictRequest(BaseModel):
    """Single prediction request."""

    customer_id: Optional[str] = None
    features: CustomerFeatures


class BatchPredictRequest(BaseModel):
    """Batch prediction request."""

    customers: List[PredictRequest]


class PredictionResult(BaseModel):
    """Single prediction response."""

    churn_probability: float = Field(..., ge=0, le=1)
    churned: bool = Field(..., description="Binary prediction (probability >= 0.5)")
    confidence: float = Field(..., ge=0, le=1)
    customer_id: Optional[str] = None


class BatchPredictionResult(BaseModel):
    """Batch prediction response."""

    predictions: List[PredictionResult]
    count: int


class ModelInfo(BaseModel):
    """Current model metadata."""

    model_version: Optional[str] = None
    model_name: Optional[str] = None
    loaded: bool = False


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    model_loaded: bool = False
