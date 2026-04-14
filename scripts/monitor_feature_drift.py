import json
import numpy as np

# Load old vs new
with open("data/processed/processed_data.json") as f:
    old = json.load(f)

with open("data/processed/new_data.json") as f:
    new = json.load(f)

def get_feature(data, key):
    return np.array([c[key] for c in data])

old_charges = get_feature(old, "monthly_charges")
new_charges = get_feature(new, "monthly_charges")

drift = abs(old_charges.mean() - new_charges.mean())

print("Feature Drift (monthly_charges):", drift)

if drift > 10:
    print("⚠️ Feature drift detected!")
else:
    print("✅ No feature drift")