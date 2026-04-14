import json
from app.ml_model import predict
from app.rules import calculate_risk   

# Load new data
with open("data/processed/new_data.json") as f:
    data = json.load(f)

correct = 0

for customer in data:
    # Model prediction
    pred = predict(customer)

    # ✅ Use SAME logic as training (not fake rule)
    actual = calculate_risk(customer)

    if pred == actual:
        correct += 1

accuracy = correct / len(data)

print("Accuracy on new data:", round(accuracy, 3))

# Drift detection threshold
if accuracy < 0.7:
    print("⚠️ Concept drift detected!")
else:
    print("✅ Model still good")