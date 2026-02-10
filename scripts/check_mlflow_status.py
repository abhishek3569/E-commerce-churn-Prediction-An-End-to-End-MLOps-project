import mlflow
from mlflow.tracking import MlflowClient

# Set tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

print("=" * 60)
print("MLflow Status Check")
print("=" * 60)
print(f"Tracking URI: {mlflow.get_tracking_uri()}\n")

# List all experiments
try:
    experiments = client.search_experiments()
    print(f"Total Experiments: {len(experiments)}\n")
    
    for exp in experiments:
        print(f"Experiment: {exp.name}")
        print(f"  - ID: {exp.experiment_id}")
        print(f"  - Lifecycle Stage: {exp.lifecycle_stage}")
        
        # Get runs for this experiment
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        print(f"  - Total Runs: {len(runs)}")
        
        if runs:
            print(f"  - Latest Run:")
            latest_run = runs[0]
            print(f"    * Run ID: {latest_run.info.run_id}")
            print(f"    * Status: {latest_run.info.status}")
            print(f"    * Start Time: {latest_run.info.start_time}")
            print(f"    * Metrics: {latest_run.data.metrics}")
        print()
        
except Exception as e:
    print(f"Error: {e}")

# Check for registered models
try:
    models = client.search_registered_models()
    print(f"Total Registered Models: {len(models)}\n")
    
    for model in models:
        print(f"Model: {model.name}")
        versions = client.search_model_versions(f"name='{model.name}'")
        print(f"  - Total Versions: {len(versions)}")
        for v in versions:
            print(f"    * Version {v.version} - Stage: {v.current_stage}")
        print()
        
except Exception as e:
    print(f"Error checking models: {e}")