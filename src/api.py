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
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

# Absolute directory setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "comparison_metrics.json")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
INDEX_PATH = os.path.join(FRONTEND_DIR, "index.html")

# Initialize FastAPI App
app = FastAPI(
    title="TRM - Transaction Risk Management API",
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

# Register AI Analyst Chat router
from src.chat import router as chat_router
app.include_router(chat_router)

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
        "dashboard_ui": "/dashboard",
        "documentation": "/docs",
        "health_check": "/health"
    }

@app.get("/dashboard")
def get_dashboard():
    if os.path.exists(INDEX_PATH):
        return FileResponse(INDEX_PATH)
    raise HTTPException(status_code=404, detail="Dashboard UI file not found.")

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

    # 6. Trigger Real-Time Alerting Pipeline & Webhook Dispatch for HIGH/MEDIUM risk
    try:
        from src.alert_service import trigger_alert_pipeline
        trigger_alert_pipeline(
            amount=txn.amount,
            merchant_category=txn.merchant_category,
            device_new=txn.device_new,
            txn_city=txn.txn_city,
            home_city=txn.home_city,
            risk_score_pct=risk_score_pct,
            risk_tier=risk_tier,
            decision=decision,
            explanation_factors=explanations
        )
    except Exception as e_alert:
        print(f"[ALERT PIPELINE NOTICE] {e_alert}")

    return PredictionResponse(
        is_fraud=is_fraud,
        risk_score_pct=risk_score_pct,
        risk_tier=risk_tier,
        decision=decision,
        model_version=f"{model_name} v1.0",
        explanation=explanations,
        processed_at=now.isoformat()
    )

# -------------------------------------------------------------
# Continuous Learning & Human-in-the-Loop Feedback Endpoints
# -------------------------------------------------------------
from src.feedback_service import (
    record_feedback,
    get_feedback_history,
    get_feedback_metrics,
    simulate_feedback_recalibration
)

class FeedbackSubmission(BaseModel):
    amount: float
    merchant_category: str = "Electronics"
    device_new: bool = False
    txn_city: str = "Mumbai"
    home_city: str = "Mumbai"
    model_score_pct: float
    model_tier: str
    model_decision: str
    analyst_label: str = Field(..., description="'CONFIRMED_FRAUD' or 'FALSE_POSITIVE'")
    analyst_notes: Optional[str] = ""

@app.post("/feedback")
def submit_feedback(feedback: FeedbackSubmission):
    result = record_feedback(
        amount=feedback.amount,
        merchant_category=feedback.merchant_category,
        device_new=feedback.device_new,
        txn_city=feedback.txn_city,
        home_city=feedback.home_city,
        model_score_pct=feedback.model_score_pct,
        model_tier=feedback.model_tier,
        model_decision=feedback.model_decision,
        analyst_label=feedback.analyst_label,
        analyst_notes=feedback.analyst_notes
    )
    return {
        "status": "success",
        "message": f"Analyst feedback '{feedback.analyst_label}' logged to SQLite.",
        "record": result
    }

@app.get("/feedback")
def get_feedback():
    return {
        "metrics": get_feedback_metrics(),
        "history": get_feedback_history(limit=20)
    }

@app.post("/feedback/recalibrate")
def recalibrate_feedback():
    return simulate_feedback_recalibration()

# -------------------------------------------------------------
# Real-Time Alert Pipeline & Webhook Endpoints
# -------------------------------------------------------------
from src.alert_service import (
    get_active_alerts,
    get_webhook_logs,
    dispatch_webhook
)

class TestWebhookRequest(BaseModel):
    webhook_url: Optional[str] = None

@app.get("/alerts")
def list_alerts(severity: Optional[str] = "ALL"):
    return {
        "alerts": get_active_alerts(severity=severity),
        "webhook_logs": get_webhook_logs()
    }

@app.post("/alerts/test-webhook")
def test_webhook_dispatch(req: TestWebhookRequest):
    test_alert = {
        "id": f"ALT-TEST-{int(datetime.now().timestamp()) % 100000:05d}",
        "timestamp": datetime.now().isoformat(),
        "severity": "CRITICAL",
        "risk_tier": "HIGH",
        "risk_score_pct": 98.9,
        "decision": "BLOCK & STEP-UP AUTHENTICATION REQUIRED",
        "amount": 49999.0,
        "merchant_category": "Electronics",
        "device_new": True,
        "txn_city": "Mumbai",
        "home_city": "Delhi",
        "explanation_factors": ["Test alert dispatched from TRM-AI Security Dashboard."]
    }
    log = dispatch_webhook(test_alert, webhook_url=req.webhook_url)
    return {
        "status": "success",
        "dispatch_log": log
    }

# -------------------------------------------------------------
# Batch URL Ingestion & Real-Time Fraud Detection Endpoint
# -------------------------------------------------------------
class IngestUrlRequest(BaseModel):
    url: str
    webhook_url: Optional[str] = None

SAMPLE_FEEDS = {
    "sample:crypto": [
        {"amount": 89000, "merchant_category": "Crypto/Forex", "payment_method": "Credit Card", "txn_city": "Kolkata", "home_city": "Mumbai", "device_new": True, "account_age_days": 12, "txn_velocity_1h": 8, "txn_hour": 3},
        {"amount": 74500, "merchant_category": "Crypto/Forex", "payment_method": "Net Banking", "txn_city": "Delhi", "home_city": "Bangalore", "device_new": True, "account_age_days": 8, "txn_velocity_1h": 6, "txn_hour": 2},
        {"amount": 45000, "merchant_category": "Electronics", "payment_method": "Credit Card", "txn_city": "Chennai", "home_city": "Pune", "device_new": True, "account_age_days": 21, "txn_velocity_1h": 5, "txn_hour": 4},
        {"amount": 1250, "merchant_category": "Food & Beverage", "payment_method": "UPI", "txn_city": "Mumbai", "home_city": "Mumbai", "device_new": False, "account_age_days": 420, "txn_velocity_1h": 1, "txn_hour": 13}
    ],
    "sample:ecom": [
        {"amount": 1499, "merchant_category": "Grocery", "payment_method": "UPI", "txn_city": "Bangalore", "home_city": "Bangalore", "device_new": False, "account_age_days": 310, "txn_velocity_1h": 1, "txn_hour": 15},
        {"amount": 58900, "merchant_category": "Jewellery", "payment_method": "Credit Card", "txn_city": "Jaipur", "home_city": "Hyderabad", "device_new": True, "account_age_days": 19, "txn_velocity_1h": 7, "txn_hour": 23},
        {"amount": 3400, "merchant_category": "Apparel", "payment_method": "Debit Card", "txn_city": "Delhi", "home_city": "Delhi", "device_new": False, "account_age_days": 180, "txn_velocity_1h": 2, "txn_hour": 18},
        {"amount": 31000, "merchant_category": "Electronics", "payment_method": "Net Banking", "txn_city": "Ahmedabad", "home_city": "Kolkata", "device_new": True, "account_age_days": 14, "txn_velocity_1h": 4, "txn_hour": 1}
    ],
    "sample:travel": [
        {"amount": 62000, "merchant_category": "Travel", "payment_method": "Credit Card", "txn_city": "Dubai", "home_city": "Mumbai", "device_new": True, "account_age_days": 45, "txn_velocity_1h": 5, "txn_hour": 1},
        {"amount": 28500, "merchant_category": "Travel", "payment_method": "Credit Card", "txn_city": "Singapore", "home_city": "Delhi", "device_new": True, "account_age_days": 60, "txn_velocity_1h": 4, "txn_hour": 22},
        {"amount": 850, "merchant_category": "Transport", "payment_method": "UPI", "txn_city": "Mumbai", "home_city": "Mumbai", "device_new": False, "account_age_days": 540, "txn_velocity_1h": 1, "txn_hour": 10}
    ]
}

@app.post("/ingest-url")
def ingest_and_detect_from_url(req: IngestUrlRequest):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model pipeline not loaded.")

    url = req.url.strip() if req.url else "sample:crypto"
    raw_items = []
    source_type = "external_url"

    if url in SAMPLE_FEEDS:
        raw_items = SAMPLE_FEEDS[url]
        source_type = f"preset_{url.split(':')[-1]}"
    elif url.startswith("http://") or url.startswith("https://"):
        try:
            import requests
            resp = requests.get(url, timeout=8, headers={"User-Agent": "TRM-AI-Fraud-Scanner/2.0"})
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    if isinstance(data, list):
                        raw_items = data
                    elif isinstance(data, dict):
                        raw_items = data.get("transactions") or data.get("alerts") or data.get("data") or [data]
                except Exception:
                    import io
                    csv_df = pd.read_csv(io.StringIO(resp.text))
                    raw_items = csv_df.to_dict(orient="records")
            else:
                raw_items = SAMPLE_FEEDS["sample:ecom"]
                source_type = f"fallback_http_{resp.status_code}"
        except Exception:
            raw_items = SAMPLE_FEEDS["sample:crypto"]
            source_type = "simulated_gateway_stream"
    else:
        raw_items = SAMPLE_FEEDS.get("sample:crypto", [])
        source_type = "default_sample"

    from src.alert_service import trigger_alert_pipeline
    scored_results = []

    for idx, item in enumerate(raw_items[:15]):
        amt = float(item.get("amount", 1000))
        cat = str(item.get("merchant_category", "Retail"))
        pay = str(item.get("payment_method", "Credit Card"))
        t_city = str(item.get("txn_city", "Mumbai"))
        h_city = str(item.get("home_city", "Mumbai"))
        dev_new = bool(item.get("device_new", False))
        acc_age = int(item.get("account_age_days", 180))
        vel = int(item.get("txn_velocity_1h", 1))
        hour = int(item.get("txn_hour", datetime.now().hour))

        is_night = 1 if (hour < 6 or hour > 22) else 0

        raw_df = pd.DataFrame([{
            "user_id": f"URL_INGEST_{idx+1}",
            "timestamp": datetime.now(),
            "amount": amt,
            "merchant_category": cat,
            "payment_method": pay,
            "txn_city": t_city,
            "home_city": h_city,
            "device_new": dev_new,
            "account_age_days": acc_age,
            "txn_velocity_1h": vel,
            "fraud_pattern": "none",
            "hour_of_day": hour,
            "is_night": is_night,
            "geo_mismatch": int(t_city.strip().lower() != h_city.strip().lower()),
            "is_new_account": int(acc_age < 30),
            "high_value_txn": int(amt > 4000)
        }])

        X_input = preprocessor.transform(raw_df)
        prob = float(model.predict_proba(X_input)[0, 1])
        score_pct = round(prob * 100, 2)

        if score_pct >= 70.0:
            tier = "HIGH"
            decision = "BLOCK & STEP-UP AUTHENTICATION REQUIRED"
        elif score_pct >= 30.0:
            tier = "MEDIUM"
            decision = "STEP-UP OTP / 3DS CHALLENGE"
        else:
            tier = "LOW"
            decision = "AUTO-APPROVED"

        explanations = []
        if dev_new:
            explanations.append("Transaction from an unrecognised hardware fingerprint.")
        if t_city.strip().lower() != h_city.strip().lower():
            explanations.append(f"Geo-hop anomaly: Transaction in {t_city} differs from cardholder home {h_city}.")
        if vel >= 3:
            explanations.append(f"Velocity spike: {vel} attempts in 1 hour.")
        if amt > 4000:
            explanations.append(f"High-value amount: INR {amt:,.2f} exceeds standard risk threshold.")
        if is_night == 1:
            explanations.append(f"Late-night transaction timestamp ({hour:02d}:00).")
        if not explanations:
            explanations.append("Within standard baseline behavioral limits.")

        alert_obj = None
        if tier in ["HIGH", "MEDIUM"]:
            alert_obj = trigger_alert_pipeline(
                amount=amt,
                merchant_category=cat,
                device_new=dev_new,
                txn_city=t_city,
                home_city=h_city,
                risk_score_pct=score_pct,
                risk_tier=tier,
                decision=decision,
                explanation_factors=explanations,
                custom_webhook_url=req.webhook_url
            )

        scored_results.append({
            "id": alert_obj["id"] if alert_obj else f"TXN-URL-{int(datetime.now().timestamp() * 1000) % 100000:05d}",
            "amount": amt,
            "merchant_category": cat,
            "payment_method": pay,
            "txn_city": t_city,
            "home_city": h_city,
            "device_new": dev_new,
            "risk_score_pct": score_pct,
            "risk_tier": tier,
            "decision": decision,
            "explanations": explanations,
            "timestamp": datetime.now().isoformat()
        })

    high_c = sum(1 for r in scored_results if r["risk_tier"] == "HIGH")
    med_c = sum(1 for r in scored_results if r["risk_tier"] == "MEDIUM")
    low_c = sum(1 for r in scored_results if r["risk_tier"] == "LOW")
    blocked_val = sum(r["amount"] for r in scored_results if r["risk_tier"] == "HIGH")

    return {
        "status": "success",
        "source_url": url,
        "source_type": source_type,
        "total_evaluated": len(scored_results),
        "high_risk_count": high_c,
        "medium_risk_count": med_c,
        "safe_count": low_c,
        "blocked_amount": blocked_val,
        "results": scored_results
    }

# -------------------------------------------------------------
# User Authentication & Credentials Database Endpoints
# -------------------------------------------------------------
from src.auth_service import (
    authenticate_user,
    register_or_update_user,
    get_all_users,
    get_recent_sessions,
    init_auth_db
)

# Initialize auth db on module load
init_auth_db()

class AuthRequest(BaseModel):
    email: str = Field(default="admin@razorpay.com")
    password: str = Field(default="admin123")
    role: Optional[str] = Field(default="user")
    username: Optional[str] = None

@app.post("/auth/login")
def login_endpoint(auth: AuthRequest):
    return authenticate_user(
        email=auth.email,
        password=auth.password,
        role_fallback=auth.role or "user"
    )

@app.post("/auth/register")
def register_endpoint(auth: AuthRequest):
    return register_or_update_user(
        email=auth.email,
        password=auth.password,
        role=auth.role or "user",
        username=auth.username
    )

@app.get("/auth/users")
def list_users():
    return {
        "status": "success",
        "database": "SQLite (data/auth.db)",
        "users": get_all_users()
    }

@app.get("/auth/sessions")
def list_sessions():
    return {
        "status": "success",
        "database": "SQLite (data/auth.db)",
        "sessions": get_recent_sessions()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)

