import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# Set tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Set experiment
mlflow.set_experiment("ecommerce_churn_experiment")

print("Creating test dataset...")
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Starting MLflow run...")
with mlflow.start_run(run_name="test_run"):
    # Log parameters
    params = {
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42
    }
    mlflow.log_params(params)
    
    # Train model
    print("Training model...")
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Log metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_pred_proba)
    }
    mlflow.log_metrics(metrics)
    
    # Log model
    mlflow.sklearn.log_model(model, "model", registered_model_name="ecommerce_churn_model")
    
    print(f"\n✓ MLflow run completed!")
    print(f"  - Accuracy: {metrics['accuracy']:.4f}")
    print(f"  - AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"\nCheck MLflow UI at: http://127.0.0.1:5000")