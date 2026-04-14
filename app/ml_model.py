import joblib

# Load full pipeline (features + model)
pipeline = joblib.load("model/churn_pipeline.pkl")

def predict(customer):
    pred = pipeline.predict([customer])[0]

    if pred == 0:
        return "LOW"
    elif pred == 1:
        return "MEDIUM"
    else:
        return "HIGH"