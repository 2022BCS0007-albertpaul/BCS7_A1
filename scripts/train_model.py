import json
import numpy as np
import os

from app.feature_engineering import extract_features
from app.rules import calculate_risk

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

import joblib
import matplotlib.pyplot as plt

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

# -----------------------------
# 1. Load Data
# -----------------------------
with open("data/processed_data.json") as f:
    data = json.load(f)

X = []
y = []

# -----------------------------
# 2. Feature Extraction + Label Generation
# -----------------------------
for customer in data:
    features = extract_features(customer)
    label = calculate_risk(customer)

    X.append(features)

    if label == "LOW":
        y.append(0)
    elif label == "MEDIUM":
        y.append(1)
    else:
        y.append(2)

X = np.array(X)
y = np.array(y)

# -----------------------------
# 3. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. Train Model
# -----------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# 5. Predictions
# -----------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# -----------------------------
# 6. Evaluation Metrics
# -----------------------------
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr")

print("\n📈 Metrics:")
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC-AUC:", roc_auc)

# Save metrics to file (for GitHub)
with open("model/metrics.txt", "w") as f:
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred))
    f.write("\n\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1 Score: {f1}\n")
    f.write(f"ROC-AUC: {roc_auc}\n")

# -----------------------------
# 7. ROC Curve (Multi-class)
# -----------------------------
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

n_classes = 3
fpr = dict()
tpr = dict()
roc_auc_class = dict()

labels = ["LOW", "MEDIUM", "HIGH"]

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc_class[i] = auc(fpr[i], tpr[i])

plt.figure()

for i in range(n_classes):
    plt.plot(
        fpr[i],
        tpr[i],
        label=f"{labels[i]} (AUC = {roc_auc_class[i]:.2f})"
    )

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multi-class ROC Curve")
plt.legend()

# Save ROC curve
plt.savefig("model/roc_curve.png")
plt.show()

# -----------------------------
# 8. Save Model
# -----------------------------
joblib.dump(model, "model/churn_model.pkl")

print("\n✅ Model trained, evaluated, and saved successfully!")