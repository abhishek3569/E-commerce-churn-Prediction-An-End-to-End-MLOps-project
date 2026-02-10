# API Documentation

## Base URL

- Local: `http://localhost:8000`
- Docker: `http://model-api:8000` (internal) or `http://localhost:8000` (external)

## Endpoints

### Health Check

**GET** `/health`

Returns service health and model load status.

**Response**

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

### Model Info

**GET** `/model_info`

Returns current model version and metadata.

**Response**

```json
{
  "model_version": "1",
  "model_name": "ecommerce_churn_model",
  "loaded": true
}
```

---

### Single Prediction

**POST** `/predict`

Predict churn probability for a single customer.

**Request Body**

```json
{
  "customer_id": "optional-string",
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
}
```

**Response**

```json
{
  "churn_probability": 0.234,
  "churned": false,
  "confidence": 0.766,
  "customer_id": "optional-string"
}
```

---

### Batch Prediction

**POST** `/predict_batch`

Predict churn for multiple customers.

**Request Body**

```json
{
  "customers": [
    {
      "customer_id": "c1",
      "features": { ... }
    },
    {
      "customer_id": "c2",
      "features": { ... }
    }
  ]
}
```

**Response**

```json
{
  "predictions": [
    {
      "churn_probability": 0.12,
      "churned": false,
      "confidence": 0.88,
      "customer_id": "c1"
    }
  ],
  "count": 2
}
```

---

### Prometheus Metrics

**GET** `/metrics`

Returns Prometheus-format metrics for monitoring.

- `churn_predictions_total`: Counter of predictions served
- `churn_prediction_latency_seconds`: Histogram of latency
- `churn_prediction_probability`: Distribution of predicted probabilities

---

## Error Codes

| Code | Description                    |
|------|--------------------------------|
| 422  | Validation error (invalid input) |
| 500  | Internal server error          |
| 503  | Service unavailable (model not loaded) |

## Rate Limiting

Not implemented. Add middleware (e.g. slowapi) for production.

## Authentication

Placeholder for future: API keys or JWT. No auth currently.
