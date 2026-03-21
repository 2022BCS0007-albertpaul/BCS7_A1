import joblib

model = joblib.load("model/churn_model.pkl")

def predict(features):
    pred = model.predict([features])[0]

    if pred == 0:
        return "LOW"
    elif pred == 1:
        return "MEDIUM"
    else:
        return "HIGH"