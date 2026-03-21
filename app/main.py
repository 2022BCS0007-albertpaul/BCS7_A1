from fastapi import FastAPI, HTTPException
from app.feature_engineering import extract_features
from app.ml_model import predict
from app.data_loader import load_data

app = FastAPI(title="Churn Prediction ML Service")

# Load sample customers (optional endpoint use)
try:
    customers_data = load_data()
except Exception:
    customers_data = []

@app.get("/")
def home():
    return {"message": "ML-based Churn Prediction Service Running"}

# 🔥 MAIN ENDPOINT (ML-based)
@app.post("/predict-risk")
def predict_risk(customer: dict):
    try:
        # Step 1: Feature extraction
        features = extract_features(customer)

        # Step 2: ML prediction
        risk = predict(features)

        return {
            "risk": risk,
            "features_used": {
                "freq_7d": features[0],
                "freq_30d": features[1],
                "freq_90d": features[2],
                "complaint_count": features[3],
                "avg_gap": features[4],
                "charge_diff": features[5]
            }
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# 📊 Optional: Get sample customers
@app.get("/customers")
def get_customers():
    if not customers_data:
        return {"message": "No data available. Run preprocessing first."}
    return customers_data[:10]


# ❤️ Health check (useful for Kubernetes)
@app.get("/health")
def health():
    return {"status": "ok"}