@echo off

echo 🔍 Checking data drift...
python scripts\check_drift.py

IF %ERRORLEVEL% NEQ 0 (
    echo 🚀 Drift detected. Retraining model...
    python -m scripts.train_model
    echo ✅ Model retrained and registered!
) ELSE (
    echo ✅ No retraining needed
)