"""
FastAPI REST API Service for Razorpay AI Risk Manager
Exposes real-time fraud scoring endpoints, model performance metrics, and system health status.
"""

import os
import pickle
import json
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

# Absolute directory setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "comparison_metrics.json")

# Initialize FastAPI App
app = FastAPI(
    title="Razorpay AI Risk Manager API",
    description="Real-Time Payment Gateway Fraud Detection & Explainable AI Endpoint",
    version="1.0.0"
)

# Enable CORS for Web Frontend consumption
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model state
artifact = None
model = None
preprocessor = None
feature_names = None
explainer = None
model_name = "XGBoost Classifier"

@app.on_event("startup")
def load_model_artifacts():
    global artifact, model, preprocessor, feature_names, explainer, model_name
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Best model artifact missing at {MODEL_PATH}. Please run model training first.")
    
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)
        
    model = artifact["model"]
    preprocessor = artifact["preprocessor"]
    feature_names = artifact["feature_names"]
    explainer = artifact.get("explainer")
    model_name = artifact.get("model_name", "XGBoost Classifier")
    print(f"[API STARTUP] Loaded model '{model_name}' successfully with {len(feature_names)} features.")

# Request Schema
class TransactionRequest(BaseModel):
    amount: float = Field(default=1000.0, example=12500.50, description="Transaction amount in INR")
    account_age_days: int = Field(default=30, example=14, description="User account age in days")
    txn_velocity_1h: int = Field(default=0, example=5, description="Number of transactions initiated in the past hour")
    txn_city: str = Field(default="Mumbai", example="Mumbai", description="City where transaction occurred")
    home_city: str = Field(default="Mumbai", example="Delhi", description="User's registered home city")
    device_new: bool = Field(default=False, example=True, description="True if transaction is initiated from a new unrecognised device")
    merchant_category: str = Field(default="Electronics", example="Electronics", description="Merchant business sector")
    payment_method: str = Field(default="Credit Card", example="Credit Card", description="Payment channel used")
    txn_hour: int = Field(default=12, ge=0, le=23, example=2, description="Hour of transaction in 24h format (0-23). Used to detect late-night activity.")
    timestamp: Optional[str] = Field(default=None, description="ISO format timestamp (defaults to current time)")

# Response Schema
class PredictionResponse(BaseModel):
    is_fraud: bool
    risk_score_pct: float
    risk_tier: str
    decision: str
    model_version: str
    explanation: List[str]
    processed_at: str

@app.get("/")
def root():
    return {
        "service": "Razorpay AI Risk Manager API",
        "status": "online",
        "documentation": "/docs",
        "health_check": "/health"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": model_name,
        "feature_count": len(feature_names) if feature_names else 0,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/metrics")
def get_metrics():
    if not os.path.exists(METRICS_PATH):
        raise HTTPException(status_code=404, detail="Metrics file not found.")
    with open(METRICS_PATH, "r") as f:
        metrics_data = json.load(f)
    return {"metrics": metrics_data, "best_model": model_name}

@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(txn: TransactionRequest):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model pipeline not initialized.")

    # 1. Construct raw DataFrame for input
    now = datetime.now()
    ts = now  # timestamp used only for record context

    # Use the user-supplied txn_hour for is_night logic
    hour_of_day = txn.txn_hour
    is_night_flag = 1 if (hour_of_day < 6 or hour_of_day > 22) else 0

    raw_df = pd.DataFrame([{
        "user_id": "REQ_API",
        "timestamp": ts,
        "amount": txn.amount,
        "merchant_category": txn.merchant_category,
        "payment_method": txn.payment_method,
        "txn_city": txn.txn_city,
        "home_city": txn.home_city,
        "device_new": txn.device_new,
        "account_age_days": txn.account_age_days,
        "txn_velocity_1h": txn.txn_velocity_1h,
        "fraud_pattern": "none"
    }])

    # 2. Derive Features — use provided txn_hour instead of parsing timestamp
    raw_df["hour_of_day"] = hour_of_day
    raw_df["is_night"] = is_night_flag
    raw_df["geo_mismatch"] = (raw_df["txn_city"] != raw_df["home_city"]).astype(int)
    raw_df["device_new"] = raw_df["device_new"].astype(int)
    raw_df["is_new_account"] = (raw_df["account_age_days"] < 30).astype(int)
    raw_df["high_value_txn"] = (raw_df["amount"] > 4000).astype(int)

    # 3. Transform via preprocessor
    X_input = preprocessor.transform(raw_df)

    # 4. Model Prediction Probability
    prob = float(model.predict_proba(X_input)[0, 1])
    risk_score_pct = round(prob * 100, 2)
    is_fraud = prob >= 0.50

    # Risk Tier assignment
    if risk_score_pct >= 70.0:
        risk_tier = "HIGH"
        decision = "BLOCK & STEP-UP AUTHENTICATION REQUIRED"
    elif risk_score_pct >= 30.0:
        risk_tier = "MEDIUM"
        decision = "MANUAL RISK ANALYST REVIEW RECOMMENDED"
    else:
        risk_tier = "LOW"
        decision = "AUTO-APPROVED"

    # 5. Generate human-readable explanation reasons
    explanations = []
    if txn.device_new:
        explanations.append("Transaction initiated from a NEW / unrecognised device (+27% risk baseline).")
    if txn.txn_city.strip().lower() != txn.home_city.strip().lower():
        explanations.append(f"Geographic anomaly: Transaction in {txn.txn_city} differs from home location ({txn.home_city}).")
    if txn.txn_velocity_1h >= 3:
        explanations.append(f"Velocity spike: High frequency of {txn.txn_velocity_1h} transactions within 1 hour.")
    if txn.amount > 4000:
        explanations.append(f"High-value payment: Transaction amount (INR {txn.amount:,.2f}) exceeds normal spend threshold.")
    if txn.account_age_days < 30:
        explanations.append(f"New account activity: Account created only {txn.account_age_days} days ago.")
    if is_night_flag == 1:
        explanations.append(f"Late-night transaction window: Transaction at {hour_of_day:02d}:00 (11 PM - 6 AM is high-risk).")
        
    if not explanations:
        explanations.append("Standard transaction profile with normal user activity metrics.")

    return PredictionResponse(
        is_fraud=is_fraud,
        risk_score_pct=risk_score_pct,
        risk_tier=risk_tier,
        decision=decision,
        model_version=f"{model_name} v1.0",
        explanation=explanations,
        processed_at=now.isoformat()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)
