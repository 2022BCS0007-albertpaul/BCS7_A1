# 📊 Churn Risk Prediction System

## 🚀 Overview
This project implements a **Churn Risk Prediction System** in two stages, showing the transition from a rule-based system to a machine learning–based approach within a DevOps pipeline.

---

## 🔹 Stage 1: Rule-Based System

In Stage 1, a backend microservice was developed to predict customer churn risk using predefined business rules.

Customer data was taken from the Telco Customer Churn dataset and enhanced by simulating ticket logs to represent customer interactions.

### ⚙️ Business Rules
- If customer has more than 5 tickets in the last 30 days → **HIGH RISK**
- If monthly charges increased and customer has multiple tickets → **MEDIUM RISK**
- If contract type is Month-to-Month and a complaint ticket exists → **HIGH RISK**
- Otherwise → **LOW RISK**

### 🔌 API
- Endpoint: `/predict-risk`
- Input: Customer details and ticket history
- Output: Risk category (`LOW`, `MEDIUM`, `HIGH`)
- Built using FastAPI

### 🐳 Deployment
- Containerized using Docker
- Easily deployable and scalable

---

## 🔹 Stage 2: Machine Learning System

In Stage 2, the rule-based logic was replaced with a machine learning model to improve prediction accuracy and adaptability.

### 🧠 Feature Engineering
The following features were extracted:
- Ticket frequency (last 7, 30, and 90 days)
- Number of complaint tickets
- Average time between tickets
- Change in monthly charges

### 🤖 Model
- Algorithm: Random Forest Classifier
- Labels were generated using the rule-based system (bootstrapping)

### 📊 Evaluation Metrics
The model was evaluated using:
- Precision
- Recall
- F1 Score
- ROC-AUC

A multi-class ROC curve was also generated for performance visualization.

### 🔄 Updated Flow
Client → API → Feature Engineering → ML Model → Prediction

---

## ⚙️ DevOps Integration
- Dockerized microservice
- Modular project structure
- Model artifacts stored (trained model, metrics, ROC curve)
- Ready for CI/CD pipeline integration

---

## 📌 Conclusion
This project demonstrates how a rule-based system can be enhanced using machine learning to provide more flexible and data-driven churn predictions.

---

## 👨‍💻 Author
Albert Sebastian