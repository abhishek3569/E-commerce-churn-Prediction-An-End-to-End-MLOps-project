import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

print("=" * 60)
print("Model Artifacts Verification")
print("=" * 60)

# Get all runs from the experiment
experiment = client.get_experiment_by_name("ecommerce_churn_experiment")
if experiment:
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    
    if runs:
        for run in runs[:3]:  # Check last 3 runs
            print(f"\nRun ID: {run.info.run_id}")
            print(f"Status: {run.info.status}")
            
            # List artifacts
            try:
                artifacts = client.list_artifacts(run.info.run_id)
                print(f"Artifacts:")
                for artifact in artifacts:
                    print(f"  - {artifact.path} (is_dir: {artifact.is_dir})")
                    
                    # If it's the model directory, list its contents
                    if artifact.path == "model" and artifact.is_dir:
                        model_artifacts = client.list_artifacts(run.info.run_id, "model")
                        for ma in model_artifacts:
                            print(f"    └─ {ma.path}")
            except Exception as e:
                print(f"  Error listing artifacts: {e}")
    else:
        print("No runs found!")
else:
    print("Experiment not found!")

# Check registered models
print("\n" + "=" * 60)
print("Registered Models")
print("=" * 60)

try:
    models = client.search_registered_models()
    for model in models:
        print(f"\nModel: {model.name}")
        versions = client.search_model_versions(f"name='{model.name}'")
        for v in versions:
            print(f"  Version {v.version}:")
            print(f"    - Stage: {v.current_stage}")
            print(f"    - Run ID: {v.run_id}")
            print(f"    - Source: {v.source}")
except Exception as e:
    print(f"Error: {e}")