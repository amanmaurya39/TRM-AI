"""
Real-Time Alert & Webhook Simulator CLI
---------------------------------------
Simulates a live transaction stream entering the TRM-AI gateway,
scoring in real-time, and triggering severity-based Slack webhooks.

Run directly:
    python src/alert_simulator.py
"""

import os
import sys
import time
import random
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.alert_service import trigger_alert_pipeline, get_active_alerts, get_webhook_logs

SAMPLE_ATTACKS = [
    {
        "amount": 89000.0,
        "merchant_category": "Crypto/Forex",
        "device_new": True,
        "txn_city": "Kolkata",
        "home_city": "Mumbai",
        "risk_score_pct": 99.2,
        "risk_tier": "HIGH",
        "decision": "BLOCK & STEP-UP AUTHENTICATION REQUIRED",
        "explanation_factors": [
            "Transaction initiated from a NEW / unrecognised device (+27% risk baseline).",
            "High-value payment: INR 89,000 exceeds threshold.",
            "Geographic anomaly: Kolkata vs Home (Mumbai)"
        ]
    },
    {
        "amount": 22400.0,
        "merchant_category": "Electronics",
        "device_new": True,
        "txn_city": "Delhi",
        "home_city": "Chennai",
        "risk_score_pct": 91.5,
        "risk_tier": "HIGH",
        "decision": "BLOCK & STEP-UP AUTHENTICATION REQUIRED",
        "explanation_factors": [
            "Velocity spike: 6 transactions in past hour.",
            "New device detected with mismatched IP subnet."
        ]
    },
    {
        "amount": 16500.0,
        "merchant_category": "Travel",
        "device_new": False,
        "txn_city": "Mumbai",
        "home_city": "Mumbai",
        "risk_score_pct": 64.0,
        "risk_tier": "MEDIUM",
        "decision": "STEP-UP OTP / 3DS CHALLENGE",
        "explanation_factors": [
            "Unusual flight ticket booking frequency for this cardholder."
        ]
    }
]


def run_alert_simulation(count: int = 3):
    print("=" * 72)
    print(" [ALERTS] TRM-AI REAL-TIME ALERTING & WEBHOOK PIPELINE")
    print("=" * 72)
    print("Simulating real-time transaction ingestion and automated alerting...\n")

    slack_url = os.environ.get("SLACK_WEBHOOK_URL")
    if slack_url:
        print(f"  + Active Slack Webhook: {slack_url[:30]}... (Live Dispatch Enabled)")
    else:
        print("  + Slack Webhook: Not set in env (Simulating Mock Webhook Pipeline)")

    print()

    for idx in range(count):
        atk = SAMPLE_ATTACKS[idx % len(SAMPLE_ATTACKS)]
        print(f"--> [EVENT {idx+1}/{count}] Incoming Txn: INR {atk['amount']:,.0f} | {atk['merchant_category']}")
        print(f"    Scoring with XGBoost... Risk Score: {atk['risk_score_pct']}% ({atk['risk_tier']})")

        alert = trigger_alert_pipeline(
            amount=atk["amount"],
            merchant_category=atk["merchant_category"],
            device_new=atk["device_new"],
            txn_city=atk["txn_city"],
            home_city=atk["home_city"],
            risk_score_pct=atk["risk_score_pct"],
            risk_tier=atk["risk_tier"],
            decision=atk["decision"],
            explanation_factors=atk["explanation_factors"]
        )

        if alert:
            tag = "[CRITICAL ALERT]" if alert["severity"] == "CRITICAL" else "[WARNING ALERT]"
            print(f"    {tag} Alert ID: {alert['id']} | Action: {alert['decision']}")
            print(f"    Webhook Status: {alert['webhook_status']}")
        else:
            print("    [PASSED] Low risk, no alert needed.")

        print()
        time.sleep(0.8)

    print("=" * 72)
    print(" [SUMMARY] Active Alert Queue Count:", len(get_active_alerts()))
    print(" [SUMMARY] Webhook Dispatches Recorded:", len(get_webhook_logs()))
    print("=" * 72)


if __name__ == "__main__":
    run_alert_simulation()
