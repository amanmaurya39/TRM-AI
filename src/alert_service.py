"""
Real-Time Alert Pipeline & Webhook Dispatcher
---------------------------------------------
Manages severity-based alert queues (CRITICAL / WARNING) and dispatches
real-time alerts to Slack webhooks or mock notification endpoints.
"""

import os
import json
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional

# In-memory queues for high-speed live alerting
MAX_ALERT_HISTORY = 50
ALERTS_QUEUE: List[Dict[str, Any]] = []
WEBHOOK_LOGS: List[Dict[str, Any]] = []

# Optional Slack webhook URL from environment
DEFAULT_SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")


def format_slack_payload(alert: Dict[str, Any]) -> Dict[str, Any]:
    """Format rich Slack Block Kit message with severity color and action buttons."""
    is_critical = alert["severity"] == "CRITICAL"
    emoji = "🚨" if is_critical else "⚠️"
    color = "#ef4444" if is_critical else "#f59e0b"

    amount_str = f"INR {alert['amount']:,.2f}"
    score_str = f"{alert['risk_score_pct']:.1f}%"
    factors = "\n".join(f"• {f}" for f in alert.get("explanation_factors", [])[:3])

    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{emoji} TRM-AI Fraud Alert: {alert['severity']} Risk Detected",
                "emoji": True
            }
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Amount:*\n{amount_str}"},
                {"type": "mrkdwn", "text": f"*Fraud Risk Score:*\n`{score_str}`"},
                {"type": "mrkdwn", "text": f"*Category:*\n{alert.get('merchant_category', 'N/A')}"},
                {"type": "mrkdwn", "text": f"*Location Anomaly:*\n{alert.get('txn_city')} vs Home: {alert.get('home_city')}"},
            ]
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Key SHAP Model Risk Triggers:*\n{factors}"
            }
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"Recommended Action: *{alert['decision']}* | Time: {alert['timestamp']}"
                }
            ]
        }
    ]

    return {
        "text": f"{emoji} [{alert['severity']}] Fraud Alert - {amount_str} flagged with {score_str} risk",
        "attachments": [{"color": color, "blocks": blocks}]
    }


def dispatch_webhook(alert: Dict[str, Any], webhook_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Dispatches alert payload to configured Slack webhook or simulates
    successful webhook reception for demo pipelines.
    """
    target_url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL", "")
    timestamp = datetime.now().isoformat()

    if target_url and target_url.startswith("https://hooks.slack.com"):
        # Real Slack Webhook dispatch
        try:
            payload = format_slack_payload(alert)
            resp = requests.post(target_url, json=payload, timeout=5)
            delivery_status = "DELIVERED" if resp.status_code == 200 else f"HTTP_{resp.status_code}"
            error_msg = None if resp.status_code == 200 else resp.text
        except Exception as e:
            delivery_status = "FAILED"
            error_msg = str(e)
    else:
        # Mock Webhook Dispatcher (Operational Simulation)
        delivery_status = "DELIVERED_MOCK"
        error_msg = None
        target_url = target_url or "https://hooks.slack.com/services/MOCK/FRAUD/PIPELINE"

    log_entry = {
        "alert_id": alert["id"],
        "timestamp": timestamp,
        "target_url": target_url[:35] + "..." if len(target_url) > 35 else target_url,
        "severity": alert["severity"],
        "amount": alert["amount"],
        "risk_score_pct": alert["risk_score_pct"],
        "status": delivery_status,
        "error": error_msg
    }

    WEBHOOK_LOGS.insert(0, log_entry)
    if len(WEBHOOK_LOGS) > MAX_ALERT_HISTORY:
        WEBHOOK_LOGS.pop()

    return log_entry


def trigger_alert_pipeline(
    amount: float,
    merchant_category: str,
    device_new: bool,
    txn_city: str,
    home_city: str,
    risk_score_pct: float,
    risk_tier: str,
    decision: str,
    explanation_factors: List[str],
    custom_webhook_url: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Evaluates risk and creates a structured alert if transaction is HIGH or MEDIUM risk.
    """
    # Only alert for HIGH (Critical) or MEDIUM (Warning) risk
    if risk_tier == "LOW":
        return None

    severity = "CRITICAL" if risk_tier == "HIGH" else "WARNING"
    alert_id = f"ALT-{int(datetime.now().timestamp() * 1000) % 1000000:06d}"
    timestamp = datetime.now().isoformat()

    alert = {
        "id": alert_id,
        "timestamp": timestamp,
        "severity": severity,
        "risk_tier": risk_tier,
        "risk_score_pct": risk_score_pct,
        "decision": decision,
        "amount": amount,
        "merchant_category": merchant_category,
        "device_new": device_new,
        "txn_city": txn_city,
        "home_city": home_city,
        "explanation_factors": explanation_factors,
        "webhook_dispatched": False,
        "webhook_status": "PENDING"
    }

    # Dispatch to Webhook Pipeline
    webhook_res = dispatch_webhook(alert, webhook_url=custom_webhook_url)
    alert["webhook_dispatched"] = True
    alert["webhook_status"] = webhook_res["status"]

    # Add to in-memory ring buffer
    ALERTS_QUEUE.insert(0, alert)
    if len(ALERTS_QUEUE) > MAX_ALERT_HISTORY:
        ALERTS_QUEUE.pop()

    return alert


def get_active_alerts(severity: Optional[str] = None, limit: int = 15) -> List[Dict[str, Any]]:
    """Retrieve filtered alerts list."""
    # Seed initial alerts if empty for instant visual demonstration
    if len(ALERTS_QUEUE) == 0:
        seed_initial_alerts()

    if severity and severity.upper() != "ALL":
        filtered = [a for a in ALERTS_QUEUE if a["severity"].upper() == severity.upper()]
        return filtered[:limit]
    return ALERTS_QUEUE[:limit]


def get_webhook_logs(limit: int = 15) -> List[Dict[str, Any]]:
    return WEBHOOK_LOGS[:limit]


def seed_initial_alerts():
    """Seed initial operational alerts so dashboard renders active live feed on boot."""
    mock_alerts = [
        {
            "id": "ALT-849201",
            "timestamp": datetime.now().isoformat(),
            "severity": "CRITICAL",
            "risk_tier": "HIGH",
            "risk_score_pct": 98.6,
            "decision": "BLOCK & STEP-UP AUTHENTICATION REQUIRED",
            "amount": 75000.0,
            "merchant_category": "Crypto/Forex",
            "device_new": True,
            "txn_city": "Moscow / Proxy",
            "home_city": "Mumbai",
            "explanation_factors": [
                "Transaction initiated from a NEW / unrecognised device (+27% risk baseline).",
                "High-value payment: Transaction amount exceeds normal threshold.",
                "Late-night transaction window: 02:00 AM."
            ],
            "webhook_dispatched": True,
            "webhook_status": "DELIVERED_MOCK"
        },
        {
            "id": "ALT-849185",
            "timestamp": datetime.now().isoformat(),
            "severity": "CRITICAL",
            "risk_tier": "HIGH",
            "risk_score_pct": 94.2,
            "decision": "BLOCK & STEP-UP AUTHENTICATION REQUIRED",
            "amount": 28500.0,
            "merchant_category": "Electronics",
            "device_new": True,
            "txn_city": "Jaipur",
            "home_city": "Bangalore",
            "explanation_factors": [
                "Geographic anomaly: Transaction in Jaipur differs from home location (Bangalore).",
                "Velocity spike: 5 rapid transactions within 1 hour."
            ],
            "webhook_dispatched": True,
            "webhook_status": "DELIVERED_MOCK"
        },
        {
            "id": "ALT-849140",
            "timestamp": datetime.now().isoformat(),
            "severity": "WARNING",
            "risk_tier": "MEDIUM",
            "risk_score_pct": 68.4,
            "decision": "STEP-UP OTP / 3DS CHALLENGE",
            "amount": 14200.0,
            "merchant_category": "Travel",
            "device_new": True,
            "txn_city": "Goa",
            "home_city": "Delhi",
            "explanation_factors": [
                "Travel sector purchase flagged for secondary step-up authorization.",
                "Unrecognised browser fingerprint detected."
            ],
            "webhook_dispatched": True,
            "webhook_status": "DELIVERED_MOCK"
        }
    ]

    for a in mock_alerts:
        ALERTS_QUEUE.append(a)
        WEBHOOK_LOGS.append({
            "alert_id": a["id"],
            "timestamp": a["timestamp"],
            "target_url": "https://hooks.slack.com/services/MOCK/FRAUD/PIPELINE",
            "severity": a["severity"],
            "amount": a["amount"],
            "risk_score_pct": a["risk_score_pct"],
            "status": "DELIVERED_MOCK",
            "error": None
        })
