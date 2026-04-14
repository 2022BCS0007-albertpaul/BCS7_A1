from fastapi import FastAPI, HTTPException
from app.ml_model import predict
from app.models import Customer

import time
import psutil
import os

app = FastAPI(title="Churn Prediction ML Service")

@app.get("/")
def home():
    return {"message": "ML Pipeline Churn Prediction Service Running"}


# 🔥 PIPELINE-BASED ENDPOINT WITH LATENCY
@app.post("/predict-risk")
def predict_risk(customer: Customer):
    try:
        start_time = time.time()   # ⏱️ Start timer

        risk = predict(customer.dict())

        latency = time.time() - start_time   # ⏱️ End timer

        return {
            "risk": risk,
            "latency_seconds": round(latency, 6)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ❤️ Health check
@app.get("/health")
def health():
    return {"status": "ok"}


# 🔥 SYSTEM METRICS (MEMORY + CPU)
@app.get("/metrics")
def system_metrics():
    process = psutil.Process(os.getpid())

    memory_mb = process.memory_info().rss / 1024 / 1024
    cpu_percent = psutil.cpu_percent(interval=0.1)

    return {
        "memory_usage_mb": round(memory_mb, 2),
        "cpu_usage_percent": cpu_percent
    }