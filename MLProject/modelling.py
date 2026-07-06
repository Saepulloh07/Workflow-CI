import os
import shutil
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

def main():
    print("Membaca data preprocessed...")
    # Sesuaikan path karena script akan dijalankan dari dalam Workflow-CI/MLProject
    X_train = pd.read_csv("preprosessing_output/X_train.csv")
    y_train = pd.read_csv("preprosessing_output/y_train.csv")
    
    y_train_arr = y_train.values.ravel()

    # Hapus folder model lama jika ada (agar tidak error saat CI berjalan ulang)
    if os.path.exists("saved_model"):
        shutil.rmtree("saved_model")

    with mlflow.start_run():
        print("Melatih model machine learning...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train_arr)
        
        # Log model ke MLflow tracking
        mlflow.sklearn.log_model(model, "model")
        
        # Simpan artefak secara lokal untuk Docker Build CI
        mlflow.sklearn.save_model(model, "saved_model")
        print("Model berhasil dilatih dan disimpan ke direktori 'saved_model'.")

if __name__ == "__main__":
    main()