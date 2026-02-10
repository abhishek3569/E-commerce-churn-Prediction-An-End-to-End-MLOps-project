import matplotlib.pyplot as plt
from pathlib import Path
import mlflow

# Setup output directory
output_dir = Path('outputs/training_artifacts')
output_dir.mkdir(parents=True, exist_ok=True)

def plot_feature_importance(model, feature_names):
    """Plot and log feature importance"""
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importance)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance Plot')
    plt.tight_layout()
    
    # Save to file
    plot_path = output_dir / 'feature_importance.png'
    plt.savefig(plot_path)
    plt.close()
    
    # Log to MLflow
    mlflow.log_artifact(str(plot_path))
    
    return plot_path

def plot_confusion_matrix(y_true, y_pred):
    """Plot and log confusion matrix"""
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    
    # Save to file
    plot_path = output_dir / 'confusion_matrix.png'
    plt.savefig(plot_path)
    plt.close()
    
    # Log to MLflow
    mlflow.log_artifact(str(plot_path))
    
    return plot_path