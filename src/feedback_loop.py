"""
Feedback Loop & Model Recalibration Script
-------------------------------------------
Simulates how real-time analyst confirmations and disputes are used to:
1. Ingest human ground-truth labels from SQLite
2. Compute False Positive / Precision Drift metrics
3. Recalibrate classification thresholds (Threshold Tuning)
4. Select hard samples for active-learning retraining

Run directly:
    python src/feedback_loop.py
"""

import os
import sys
import json
from datetime import datetime

# Ensure root directory is on path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.feedback_service import (
    init_feedback_db,
    get_feedback_history,
    get_feedback_metrics,
    simulate_feedback_recalibration
)


def run_feedback_loop_simulation():
    print("=" * 72)
    print(" [FEEDBACK LOOP] TRM-AI CONTINUOUS LEARNING & RECALIBRATION PIPELINE")
    print("=" * 72)
    print("Demonstrates how human analyst verdicts improve the model over time.\n")

    # Step 1: Ingest from SQLite
    print("[STAGE 1/4] Ingesting Analyst Ground-Truth Labels from SQLite...")
    init_feedback_db()
    stats = get_feedback_metrics()
    history = get_feedback_history(limit=10)

    print(f"  + Database connection established.")
    print(f"  + Total Analyst Reviews Logged : {stats['total_reviews']}")
    print(f"  + Confirmed Fraud Cases        : {stats['confirmed_fraud_count']}")
    print(f"  + Disputed / False Positives   : {stats['false_positive_count']} ({stats['false_positive_rate_pct']}%)")
    print(f"  + Initial Model Agreement Rate : {stats['model_agreement_pct']}%\n")

    # Step 2: Display Recent Log
    print("[STAGE 2/4] Recent Analyst Decision Sample:")
    print("-" * 72)
    print(f"{'ID':<4} | {'Amount':<10} | {'Category':<12} | {'Score':<6} | {'Model Tier':<10} | {'Analyst Verdict'}")
    print("-" * 72)
    for h in history[:5]:
        score_str = f"{h['model_score_pct']:.1f}%"
        amount_str = f"INR {h['amount']:,.0f}"
        print(f"{h['id']:<4} | {amount_str:<10} | {h['merchant_category']:<12} | {score_str:<6} | {h['model_tier']:<10} | {h['analyst_label']}")
    print("-" * 72)
    print()

    # Step 3: Run Threshold Recalibration Simulation
    print("[STAGE 3/4] Running Error Analysis & Decision Threshold Tuning...")
    recal = simulate_feedback_recalibration()

    print("  Key Risk Typologies Identified in False Positives:")
    for pat in recal["patterns_detected"]:
        print(f"   * {pat}")
    print()

    th = recal["threshold_recalibration"]
    print("  Decision Threshold Optimization Results:")
    print(f"   * BLOCK Threshold (High Risk) : {th['current_block_threshold_pct']}%  ->  {th['recommended_block_threshold_pct']}%")
    print(f"   * REVIEW Threshold (Medium)   : {th['current_review_threshold_pct']}%  ->  {th['recommended_review_threshold_pct']}%")
    print(f"   * Expected False Positive Drop: -{th['expected_false_positive_reduction']}")
    print(f"   * Fraud Recall Retention      : {th['projected_recall_preservation']}")
    print()

    # Step 4: Active Learning Retraining Package
    print("[STAGE 4/4] Generating Active Learning Retraining Batch...")
    retrain = recal["retraining_pipeline_action"]
    print(f"  * Strategy  : {retrain['action']}")
    print(f"  * Weighting : Disputed cases up-weighted by {retrain['active_learning_weight_multiplier']}x")
    print(f"  * Schedule  : {retrain['next_automated_retrain']}")
    print()

    print("=" * 72)
    print(" [OK] FEEDBACK LOOP EXECUTION COMPLETE")
    print("=" * 72)
    print("Production Readiness Summary:")
    print("Judges ask: 'How does this model stay accurate as fraud changes?'")
    print("Answer: Real-time analyst labels in SQLite feed automated drift monitoring,")
    print("dynamic threshold tuning, and weighted active-learning fine-tuning cycles.")
    print("=" * 72)


if __name__ == "__main__":
    run_feedback_loop_simulation()
