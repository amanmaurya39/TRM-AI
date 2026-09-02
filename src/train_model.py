"""
Fraud Risk Model Training
Trains an XGBoost classifier on transaction features and computes
SHAP values so every prediction can be explained in plain language.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, classification_report,
    confusion_matrix, average_precision_score
)

DATA_PATH = "/home/claude/razorpay-risk-manager/data/transactions_features.csv"
MODEL_PATH = "/home/claude/razorpay-risk-manager/models/fraud_model.pkl"
METRICS_PATH = "/home/claude/razorpay-risk-manager/models/metrics.json"

DROP_COLS = ["txn_id", "user_id", "timestamp", "txn_city", "home_city",
             "fraud_pattern", "is_fraud"]

def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    return df

def train():
    df = load_data()
    y = df["is_fraud"]
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Handle class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1
    )

    print("Training XGBoost fraud model...")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred_proba)
    ap = average_precision_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"\nROC-AUC: {auc:.4f}")
    print(f"Average Precision (PR-AUC): {ap:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix:\n{cm}")

    # SHAP explainer (TreeExplainer is fast + exact for XGBoost)
    print("\nComputing SHAP explainer...")
    explainer = shap.TreeExplainer(model)

    # Save everything needed for the dashboard
    artifact = {
        "model": model,
        "explainer": explainer,
        "feature_names": feature_names,
        "X_test": X_test,
        "y_test": y_test,
        "test_df": df.loc[X_test.index],  # keep original context columns
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)

    metrics = {
        "roc_auc": round(auc, 4),
        "avg_precision": round(ap, 4),
        "precision_fraud": round(report["1"]["precision"], 4),
        "recall_fraud": round(report["1"]["recall"], 4),
        "f1_fraud": round(report["1"]["f1-score"], 4),
        "confusion_matrix": cm,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(feature_names)
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel + explainer saved to {MODEL_PATH}")
    print(f"Metrics saved to {METRICS_PATH}")

    return model, metrics


if __name__ == "__main__":
    train()
