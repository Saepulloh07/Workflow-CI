import os
import pickle
import argparse
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score
)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train heart disease classification models')
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="all",
        help="Model name or 'all' to train all models"
    )
    return parser.parse_args()


def setup_environment():
    """Setup directories and MLflow tracking"""
    os.makedirs("csv_output", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Heart_Disease_Classification")
    
    print("✓ Environment setup complete")


def load_preprocessed_data():
    """Load preprocessed training and test data"""
    data_path = "data/preprocessing_objects.pkl"
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Preprocessing data not found at {data_path}")
    
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    feature_names = data.get("feature_names", [])
    
    print(f"✓ Data loaded → Train {X_train.shape}, Test {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, feature_names


def get_models():
    """Return dictionary of models to train"""
    return {
        "Logistic_Regression": LogisticRegression(
            random_state=42, 
            max_iter=1000
        ),
        "Decision_Tree": DecisionTreeClassifier(
            random_state=42
        ),
        "Random_Forest": RandomForestClassifier(
            random_state=42, 
            n_estimators=200
        ),
        "Gradient_Boosting": GradientBoostingClassifier(
            random_state=42, 
            n_estimators=200
        ),
        "SVM": SVC(
            random_state=42, 
            probability=True
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5
        ),
        "Naive_Bayes": GaussianNB(),
    }


def calculate_metrics(y_test, y_pred, y_prob=None):
    """Calculate all evaluation metrics"""
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
    }
    
    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
        except Exception as e:
            print(f"Warning: Could not calculate ROC AUC - {e}")
            metrics['roc_auc'] = None
    else:
        metrics['roc_auc'] = None
    
    return metrics


def train_and_evaluate_model(model_name, model, X_train, X_test, y_train, y_test):
    """Train a single model and evaluate it"""
    print(f"\n{'='*60}")
    print(f"Training → {model_name}")
    print(f"{'='*60}")
    
    # Create nested run for this model
    with mlflow.start_run(run_name=model_name, nested=True):
        # Log model parameters
        if hasattr(model, 'get_params'):
            params = model.get_params()
            for key, value in params.items():
                mlflow.log_param(key, value)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get probability predictions if available
        y_prob = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        
        # Log metrics to MLflow
        for metric_name, metric_value in metrics.items():
            if metric_value is not None:
                mlflow.log_metric(metric_name, metric_value)
        
        # Log model to nested run
        mlflow.sklearn.log_model(
            model, 
            "model",
            input_example=X_train[:5],
            signature=mlflow.models.infer_signature(X_train, y_pred)
        )
        
        # Print results
        print(f"✓ Accuracy  = {metrics['accuracy']:.4f}")
        print(f"✓ Precision = {metrics['precision']:.4f}")
        print(f"✓ Recall    = {metrics['recall']:.4f}")
        print(f"✓ F1-Score  = {metrics['f1_score']:.4f}")
        if metrics['roc_auc'] is not None:
            print(f"✓ ROC-AUC   = {metrics['roc_auc']:.4f}")
        
        # Return results and trained model
        return {
            "Model": model_name,
            "Accuracy": round(metrics['accuracy'], 4),
            "Precision": round(metrics['precision'], 4),
            "Recall": round(metrics['recall'], 4),
            "F1-Score": round(metrics['f1_score'], 4),
            "ROC-AUC": round(metrics['roc_auc'], 4) if metrics['roc_auc'] is not None else "N/A"
        }, model


def save_results_and_best_model(results, trained_models, X_train, y_train):
    """Save comparison results and best model to parent run"""
    # Create results dataframe
    results_df = pd.DataFrame(results).sort_values(
        "Accuracy", 
        ascending=False
    ).reset_index(drop=True)
    
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # Save comparison CSV
    csv_path = "csv_output/model_comparison_results.csv"
    results_df.to_csv(csv_path, index=False)
    mlflow.log_artifact(csv_path)
    print(f"\n✓ Results saved to {csv_path}")
    
    # Get best model
    best_name = results_df.iloc[0]["Model"]
    best_acc = results_df.iloc[0]["Accuracy"]
    best_model = trained_models[best_name]
    
    # Retrain best model on full training data to ensure consistency
    print(f"\n{'='*60}")
    print(f"Retraining best model: {best_name}")
    print(f"{'='*60}")
    best_model.fit(X_train, y_train)
    
    # Save best model locally
    model_path = f"data/best_model_{best_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"✓ Model saved locally to {model_path}")
    
    # Log best model to PARENT run (not nested)
    # This is critical for CI/CD to find the model
    try:
        mlflow.sklearn.log_model(
            best_model, 
            "best_model",  # This artifact path is used by CI/CD
            input_example=X_train[:5],
            signature=mlflow.models.infer_signature(X_train, best_model.predict(X_train[:5]))
        )
        print("✓ Best model logged to MLflow parent run")
    except Exception as e:
        print(f"Warning: Failed to log best model to MLflow: {e}")
    
    # Log the pickle file as well
    mlflow.log_artifact(model_path)
    
    # Log best model metadata
    mlflow.log_param("best_model_name", best_name)
    mlflow.log_metric("best_accuracy", best_acc)
    
    # Log all metrics from best model
    best_metrics = results_df.iloc[0].to_dict()
    for key, value in best_metrics.items():
        if key != "Model" and value != "N/A":
            mlflow.log_metric(f"best_{key.lower().replace('-', '_')}", float(value))
    
    print(f"✓ Best Model: {best_name} (Accuracy = {best_acc})")
    
    return best_name, best_acc


def main():
    """Main execution function"""
    # Parse arguments
    args = parse_arguments()
    
    # Setup environment
    setup_environment()
    
    # Load data
    X_train, X_test, y_train, y_test, feature_names = load_preprocessed_data()
    
    # Get models
    models = get_models()
    
    # Filter models if specific model requested
    if args.model_name != "all" and args.model_name in models:
        models = {args.model_name: models[args.model_name]}
        print(f"\n✓ Training only: {args.model_name}")
    else:
        print(f"\n✓ Training all models ({len(models)} total)")
    
    # Start main MLflow run
    with mlflow.start_run(run_name="Heart_Disease_Training_Run") as parent_run:
        # Log parameters to parent run
        mlflow.log_param("n_models", len(models))
        mlflow.log_param("train_size", X_train.shape[0])
        mlflow.log_param("test_size", X_test.shape[0])
        mlflow.log_param("n_features", X_train.shape[1])
        
        # Train all models and store them
        results = []
        trained_models = {}
        
        for name, model in models.items():
            result, trained_model = train_and_evaluate_model(
                name, model, X_train, X_test, y_train, y_test
            )
            results.append(result)
            trained_models[name] = trained_model
        
        # Save results and best model to parent run
        best_name, best_acc = save_results_and_best_model(
            results, trained_models, X_train, y_train
        )
        
        # Log parent run ID for reference
        print(f"\n✓ Parent Run ID: {parent_run.info.run_id}")
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print("✓ All artifacts logged to MLflow")
    print("✓ Best model saved and ready for deployment")
    print("="*60)

def setup_environment():
    os.makedirs("csv_output", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Paksa local tracking di CI, hormati env jika ada
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Heart_Disease_Classification")
    
    print(f"MLflow tracking URI: {tracking_uri}")
    
if __name__ == "__main__":
    main()