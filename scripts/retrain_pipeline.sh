#!/bin/bash

echo "🔍 Checking data drift..."

python -m scripts.check_drift

if [ $? -ne 0 ]; then
    echo "🚀 Drift detected. Retraining model..."
    python -m scripts.train_model
    echo "✅ Model retrained and registered!"
else
    echo "✅ No retraining needed"
fi