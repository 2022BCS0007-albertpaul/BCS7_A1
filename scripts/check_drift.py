import json
import numpy as np

# Load old data
with open("data/processed/processed_data.json") as f:
    old_data = json.load(f)

# Load new data (simulate)
with open("data/processed/new_data.json") as f:
    new_data = json.load(f)

def extract_charge(data):
    return [c["monthly_charges"] for c in data]

old_charges = np.array(extract_charge(old_data))
new_charges = np.array(extract_charge(new_data))

# Simple drift check (mean difference)
old_mean = old_charges.mean()
new_mean = new_charges.mean()

drift = abs(new_mean - old_mean)

print("Old Mean:", old_mean)
print("New Mean:", new_mean)
print("Drift:", drift)

# Threshold
if drift > 10:
    print("⚠️ Drift detected!")
    exit(1)
else:
    print("✅ No significant drift")