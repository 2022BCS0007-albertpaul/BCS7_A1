import json
import numpy as np
import os
import shutil

import mlflow
import mlflow.sklearn

from app.rules import calculate_risk
from app.feature_pipeline import FeatureExtractor

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

import joblib

# -----------------------------
# CLEAN MLflow state (IMPORTANT for CI/CD)
# -----------------------------
if os.path.exists("mlruns"):
    shutil.rmtree("mlruns")

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("churn-prediction")

# -----------------------------
# Ensure directories exist
# -----------------------------
os.makedirs("model", exist_ok=True)
os.makedirs("data/splits", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# -----------------------------
# Start MLflow run
# -----------------------------
with mlflow.start_run():

    # -----------------------------
    # 1. Load Data (SAFE CHECK)
    # -----------------------------
    data_path = "data/processed/processed_data.json"

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Missing dataset: {data_path}. Run preprocess.py first."
        )

    with open(data_path) as f:
        data = json.load(f)

    # -----------------------------
    # 2. Generate Labels
    # -----------------------------
    X = []
    y = []

    for customer in data:
        X.append(customer)  # keep structured input

        label = calculate_risk(customer)

        if label == "LOW":
            y.append(0)
        elif label == "MEDIUM":
            y.append(1)
        else:
            y.append(2)

    y = np.array(y)

    # -----------------------------
    # 3. Train-Test Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Save splits (for DVC / tracking)
    np.save("data/splits/y_train.npy", y_train)
    np.save("data/splits/y_test.npy", y_test)

    # -----------------------------
    # 4. Model Pipeline
    # -----------------------------
    pipeline = Pipeline([
        ("features", FeatureExtractor()),
        ("model", RandomForestClassifier(
            n_estimators=150,
            max_depth=None,
            random_state=42
        ))
    ])

    # -----------------------------
    # 5. Train
    # -----------------------------
    pipeline.fit(X_train, y_train)

    # -----------------------------
    # 6. Predictions
    # -----------------------------
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)

    # -----------------------------
    # 7. Metrics
    # -----------------------------
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr")

    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, y_pred))

    print("\n📈 Metrics:")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("ROC-AUC:", roc_auc)

    # -----------------------------
    # 8. Save logs
    # -----------------------------
    with open("model/performance_log.txt", "a") as f:
        f.write(
            f"Precision: {precision}, Recall: {recall}, "
            f"F1: {f1}, ROC-AUC: {roc_auc}\n"
            f"Accuracy: {(y_pred == y_test).mean()}\n"
        )

    # -----------------------------
    # 9. MLflow logging
    # -----------------------------
    mlflow.log_param("n_estimators", 150)
    mlflow.log_param("max_depth", None)
    mlflow.log_param("dataset_version", "v1")

    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    # -----------------------------
    # 10. Save model
    # -----------------------------
    joblib.dump(pipeline, "model/churn_pipeline.pkl")

    # MLflow model logging
    mlflow.sklearn.log_model(
        pipeline,
        "model",
        registered_model_name="ChurnPredictionModel"
    )

    print("\n✅ Training + MLflow logging completed successfully!")