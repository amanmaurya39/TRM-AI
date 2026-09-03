"""
Feedback Loop & Continuous Learning Service
-------------------------------------------
Manages human-in-the-loop analyst feedback, stores ground truth labels in SQLite,
and simulates threshold recalibration and active learning retraining cycles.
"""

import os
import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "feedback.db")


def get_db_connection():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_feedback_db():
    """Create feedback table and seed with realistic analyst reviews if empty."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analyst_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            amount REAL NOT NULL,
            merchant_category TEXT NOT NULL,
            device_new INTEGER NOT NULL,
            txn_city TEXT NOT NULL,
            home_city TEXT NOT NULL,
            model_score_pct REAL NOT NULL,
            model_tier TEXT NOT NULL,
            model_decision TEXT NOT NULL,
            analyst_label TEXT NOT NULL,
            analyst_notes TEXT,
            status TEXT DEFAULT 'PENDING'
        )
    """)

    # Check if empty; if so, seed historical analyst labels for demo richness
    cursor.execute("SELECT COUNT(*) FROM analyst_feedback")
    count = cursor.fetchone()[0]

    if count == 0:
        seed_data = [
            (
                (datetime.now() - timedelta(hours=3, minutes=15)).isoformat(),
                45000.0, "Electronics", 1, "Mumbai", "Delhi", 96.8, "HIGH", "BLOCK",
                "CONFIRMED_FRAUD", "Customer confirmed unauthorized login from unknown device in Mumbai."
            ),
            (
                (datetime.now() - timedelta(hours=5, minutes=40)).isoformat(),
                18500.0, "Travel", 1, "Goa", "Mumbai", 74.2, "MEDIUM", "REVIEW",
                "FALSE_POSITIVE", "Verified genuine flight booking during vacation. Device brand matched past logins."
            ),
            (
                (datetime.now() - timedelta(hours=8, minutes=10)).isoformat(),
                82000.0, "Crypto/Forex", 1, "Kolkata", "Bangalore", 99.4, "HIGH", "BLOCK",
                "CONFIRMED_FRAUD", "Stolen card credential stuffing attack. Multiple rapid failed OTPs preceded."
            ),
            (
                (datetime.now() - timedelta(hours=11, minutes=5)).isoformat(),
                3200.0, "Gaming", 0, "Delhi", "Delhi", 42.1, "LOW", "ALLOW",
                "FALSE_POSITIVE", "High micro-transaction velocity during weekend gaming tournament, user confirmed."
            ),
            (
                (datetime.now() - timedelta(hours=14, minutes=20)).isoformat(),
                29000.0, "Electronics", 1, "Jaipur", "Ahmedabad", 88.5, "HIGH", "BLOCK",
                "CONFIRMED_FRAUD", "Courier rerouted to warehouse address; cardholder filed immediate chargeback."
            ),
            (
                (datetime.now() - timedelta(hours=18, minutes=50)).isoformat(),
                12500.0, "E-commerce", 0, "Pune", "Pune", 31.0, "LOW", "ALLOW",
                "FALSE_POSITIVE", "Regular repeat customer purchasing festival gifts."
            ),
            (
                (datetime.now() - timedelta(hours=22, minutes=30)).isoformat(),
                67000.0, "Crypto/Forex", 1, "Gurgaon", "Chennai", 98.1, "HIGH", "BLOCK",
                "CONFIRMED_FRAUD", "Account takeover via SIM-swap; confirmed by telecom carrier."
            ),
        ]

        cursor.executemany("""
            INSERT INTO analyst_feedback (
                timestamp, amount, merchant_category, device_new,
                txn_city, home_city, model_score_pct, model_tier,
                model_decision, analyst_label, analyst_notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, seed_data)

    conn.commit()
    conn.close()


def record_feedback(
    amount: float,
    merchant_category: str,
    device_new: bool,
    txn_city: str,
    home_city: str,
    model_score_pct: float,
    model_tier: str,
    model_decision: str,
    analyst_label: str,
    analyst_notes: Optional[str] = None
) -> Dict[str, Any]:
    """Store an analyst review in SQLite database."""
    init_feedback_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    ts = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO analyst_feedback (
            timestamp, amount, merchant_category, device_new,
            txn_city, home_city, model_score_pct, model_tier,
            model_decision, analyst_label, analyst_notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        ts, amount, merchant_category, int(device_new),
        txn_city, home_city, model_score_pct, model_tier,
        model_decision, analyst_label, analyst_notes or ""
    ))

    feedback_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return {
        "id": feedback_id,
        "timestamp": ts,
        "analyst_label": analyst_label,
        "status": "RECORDED"
    }


def get_feedback_history(limit: int = 20) -> List[Dict[str, Any]]:
    """Retrieve recent analyst feedback entries."""
    init_feedback_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM analyst_feedback
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_feedback_metrics() -> Dict[str, Any]:
    """Calculate summary metrics on analyst ground-truth feedback."""
    init_feedback_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM analyst_feedback")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM analyst_feedback WHERE analyst_label = 'CONFIRMED_FRAUD'")
    confirmed_fraud = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM analyst_feedback WHERE analyst_label = 'FALSE_POSITIVE'")
    false_positives = cursor.fetchone()[0]

    # Model agreement: Model said HIGH/BLOCK and Analyst said CONFIRMED_FRAUD, or Model LOW and Analyst FALSE_POSITIVE
    cursor.execute("""
        SELECT COUNT(*) FROM analyst_feedback
        WHERE (model_tier IN ('HIGH', 'MEDIUM') AND analyst_label = 'CONFIRMED_FRAUD')
           OR (model_tier = 'LOW' AND analyst_label = 'FALSE_POSITIVE')
    """)
    agreed = cursor.fetchone()[0]

    conn.close()

    agreement_rate = (agreed / total * 100) if total > 0 else 100.0
    fp_rate = (false_positives / total * 100) if total > 0 else 0.0

    return {
        "total_reviews": total,
        "confirmed_fraud_count": confirmed_fraud,
        "false_positive_count": false_positives,
        "false_positive_rate_pct": round(fp_rate, 1),
        "model_agreement_pct": round(agreement_rate, 1),
        "pending_retrain_samples": total
    }


def simulate_feedback_recalibration() -> Dict[str, Any]:
    """
    Simulates how accumulated analyst feedback is processed in a production ML pipeline:
    1. Error Analysis on False Positives.
    2. Decision Threshold Recalibration to optimize precision without degrading recall.
    3. Active Learning candidate selection for next retraining cycle.
    """
    history = get_feedback_history(limit=50)
    stats = get_feedback_metrics()

    fp_list = [h for h in history if h["analyst_label"] == "FALSE_POSITIVE"]
    cf_list = [h for h in history if h["analyst_label"] == "CONFIRMED_FRAUD"]

    # Analyze False Positives pattern
    travel_fps = sum(1 for h in fp_list if h["merchant_category"] == "Travel")
    gaming_fps = sum(1 for h in fp_list if h["merchant_category"] == "Gaming")

    # Threshold optimization simulation
    current_high_threshold = 75.0
    recommended_high_threshold = 82.5  # shifted slightly higher to filter travel FP spikes
    projected_fp_reduction_pct = 28.5
    projected_fraud_retention_pct = 97.8

    return {
        "simulation_timestamp": datetime.now().isoformat(),
        "analyzed_samples": len(history),
        "confirmed_fraud_count": len(cf_list),
        "false_positive_count": len(fp_list),
        "patterns_detected": [
            f"Travel transactions from new devices showed higher false positive clustering ({travel_fps} cases).",
            f"Micro-transaction velocity in Gaming sector occasionally triggers premature MEDIUM tier ({gaming_fps} cases).",
            "Geo-mismatch during vacation seasons accounts for 40% of disputed alerts."
        ],
        "threshold_recalibration": {
            "current_block_threshold_pct": current_high_threshold,
            "recommended_block_threshold_pct": recommended_high_threshold,
            "current_review_threshold_pct": 50.0,
            "recommended_review_threshold_pct": 55.0,
            "expected_false_positive_reduction": f"{projected_fp_reduction_pct}%",
            "projected_recall_preservation": f"{projected_fraud_retention_pct}%"
        },
        "retraining_pipeline_action": {
            "active_learning_weight_multiplier": 2.5,
            "action": "Hard-sample mining: Added all confirmed false positives with 2.5x sample weights to fine-tuning dataset.",
            "next_automated_retrain": "Scheduled at 02:00 UTC batch cycle"
        }
    }
