import json
import numpy as np
import os

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
# Ensure directories exist
# -----------------------------
os.makedirs("model", exist_ok=True)
os.makedirs("data/splits", exist_ok=True)

# -----------------------------
# MLflow setup
# -----------------------------
mlflow.set_experiment("churn-prediction")

with mlflow.start_run():

    # -----------------------------
    # 1. Load Data
    # -----------------------------
    with open("data/processed/processed_data.json") as f:
        data = json.load(f)

    X = data
    y = []

    # -----------------------------
    # 2. Generate Labels (from rules)
    # -----------------------------
    for customer in data:
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

    # 🔥 Save splits (for DVC)
    np.save("data/splits/y_train.npy", y_train)
    np.save("data/splits/y_test.npy", y_test)

    # -----------------------------
    # 4. Hyperparameters
    # -----------------------------
    n_estimators = 150
    max_depth = None

    # -----------------------------
    # 5. Pipeline (Feature + Model)
    # -----------------------------
    pipeline = Pipeline([
        ("features", FeatureExtractor()),
        ("model", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        ))
    ])

    # -----------------------------
    # 6. Train Model
    # -----------------------------
    pipeline.fit(X_train, y_train)

    # -----------------------------
    # 7. Predictions
    # -----------------------------
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)

    # -----------------------------
    # 8. Evaluation Metrics
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
    # 9. Performance Monitoring (NEW)
    # -----------------------------
    with open("model/performance_log.txt", "a") as f:
        f.write(
            f"Precision: {precision}, Recall: {recall}, "
            f"F1: {f1}, ROC-AUC: {roc_auc}\n"
            f"Accuracy: {(y_pred == y_test).mean()}\n"
        )

    # -----------------------------
    # 10. MLflow Logging
    # -----------------------------

    # Parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("dataset_version", "v1")

    # Feature tracking
    mlflow.log_param("features", [
        "freq_7d",
        "freq_30d",
        "freq_90d",
        "complaint_count",
        "avg_gap",
        "charge_diff"
    ])

    # Metrics
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    # -----------------------------
    # 11. Save Model (Pipeline)
    # -----------------------------
    joblib.dump(pipeline, "model/churn_pipeline.pkl")

    # Register model in MLflow
    mlflow.sklearn.log_model(
        pipeline,
        "model",
        registered_model_name="ChurnPredictionModel"
    )

    print("\n✅ Model trained, monitored, and registered successfully!")