import mlflow
from mlflow.tracking import MlflowClient

def initialize_mlflow():
    """Initialize MLflow experiment and model registry"""
    
    # Set tracking URI (adjust based on your setup)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    client = MlflowClient()
    
    # Create experiment if it doesn't exist
    experiment_name = "ecommerce_churn_experiment"
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"✓ Created experiment: {experiment_name} (ID: {experiment_id})")
        else:
            print(f"✓ Experiment already exists: {experiment_name} (ID: {experiment.experiment_id})")
    except Exception as e:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"✓ Created experiment: {experiment_name} (ID: {experiment_id})")
    
    # Note: Registered models are created automatically when you log a model
    # But we can check if it exists
    model_name = "ecommerce_churn_model"
    try:
        client.get_registered_model(model_name)
        print(f"✓ Registered model already exists: {model_name}")
    except:
        print(f"ℹ Registered model '{model_name}' will be created when first model is logged")
    
    print("\nMLflow initialization complete!")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment: {experiment_name}")
    print(f"Model Registry: {model_name}")

if __name__ == "__main__":
    initialize_mlflow()
