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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train heart disease classification models')
    parser.add_argument("--model_name", type=str, default="all",
                        help="Model name or 'all' to train all models")
    return parser.parse_args()


def setup_environment():
    os.makedirs("csv_output", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Jangan paksa URI di sini → biarkan mlflow run atau CI yang set
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    mlflow.set_experiment("Heart_Disease_Classification")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")


def load_preprocessed_data():
    path = "data/preprocessing_objects.pkl"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Preprocessing file not found: {path}")
    
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    
    print(f"Data loaded → Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def get_models():
    return {
        "Logistic_Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision_Tree": DecisionTreeClassifier(random_state=42),
        "Random_Forest": RandomForestClassifier(random_state=42, n_estimators=200),
        "Gradient_Boosting": GradientBoostingClassifier(random_state=42, n_estimators=200),
        "SVM": SVC(random_state=42, probability=True),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive_Bayes": GaussianNB(),
    }


def calculate_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except:
            metrics["roc_auc"] = None
    return metrics


def train_and_evaluate(model_name, model, X_train, X_test, y_train, y_test):
    print(f"\nTraining → {model_name}")
    
    with mlflow.start_run(run_name=model_name, nested=True):
        mlflow.log_params(model.get_params() if hasattr(model, "get_params") else {})
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})
        
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=X_train[:5],
            signature=mlflow.models.infer_signature(X_train, y_pred)
        )
        
        print(f"Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}")
        return metrics, model


def save_best_model(results_df, trained_models, X_train, y_train):
    best_row = results_df.iloc[0]
    best_name = best_row["Model"]
    best_model = trained_models[best_name]
    
    # Retrain on full data
    best_model.fit(X_train, y_train)
    
    # Save pickle → ini yang akan dipakai Docker
    pickle_path = f"data/best_model_{best_name}.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(best_model, f)
    
    # Log ke parent run (WAJIB)
    mlflow.set_tag("best_model", best_name)  # tambahan
    mlflow.log_param("best_model_name", best_name)
    mlflow.log_metric("best_accuracy", best_row["Accuracy"])
    
    # Log model sebagai "best_model" → untuk mlflow build-docker
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="best_model",  
        input_example=X_train[:5],
        signature=mlflow.models.infer_signature(X_train, best_model.predict(X_train[:5]))
    )
    
    # Log file pickle juga
    mlflow.log_artifact(pickle_path)
    
    print(f"BEST MODEL SAVED: {best_name} → {pickle_path}")
    return best_name

def main():
    args = parse_arguments()
    setup_environment()
    
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    models = get_models()
    
    if args.model_name != "all" and args.model_name in models:
        models = {args.model_name: models[args.model_name]}
        print(f"Training only: {args.model_name}")
    else:
        print(f"Training all {len(models)} models")
    
    with mlflow.start_run(run_name="Heart_Disease_Training_Run"):
        mlflow.log_param("n_models", len(models))
        
        results = []
        trained_models = {}
        
        for name, model in models.items():
            metrics, trained_model = train_and_evaluate(name, model, X_train, X_test, y_train, y_test)
            results.append({
                "Model": name,
                "Accuracy": round(metrics["accuracy"], 4),
                "Precision": round(metrics["precision"], 4),
                "Recall": round(metrics["recall"], 4),
                "F1-Score": round(metrics["f1_score"], 4),
                "ROC-AUC": round(metrics.get("roc_auc"), 4) if metrics.get("roc_auc") else "N/A"
            })
            trained_models[name] = trained_model
        
        results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False).reset_index(drop=True)
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print(results_df.to_string(index=False))
        
        csv_path = "csv_output/model_comparison_results.csv"
        results_df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path)
        
        save_best_model(results_df, trained_models, X_train, y_train)
        
        print("\nTRAINING COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()