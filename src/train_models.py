"""
Model Training, Evaluation & Comparison Script
Trains 4 distinct machine learning algorithms (Logistic Regression, Decision Tree,
Random Forest, and XGBoost), compares performance metrics on held-out test set,
logs comparative performance, and exports the best model artifact.
"""

import os
import pickle
import json
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import shap

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_PKL = os.path.join(MODELS_DIR, "preprocessed_data.pkl")

def train_and_compare_models():
    print("=" * 60)
    print(" STEP 3 & 4: MODEL TRAINING & PERFORMANCE COMPARISON")
    print("=" * 60)

    if not os.path.exists(DATA_PKL):
        raise FileNotFoundError("Preprocessed data missing! Please run src/preprocess.py first.")

    with open(DATA_PKL, "rb") as f:
        data = pickle.load(f)

    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    feature_names = data["feature_names"]
    preprocessor = data["preprocessor"]

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # Define 4 diverse algorithms
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
            random_state=42, n_jobs=-1
        )
    }

    results = []
    trained_model_objects = {}

    print(f"\nTraining {len(models)} algorithms on {len(X_train):,} samples...\n")

    for name, model in models.items():
        print(f" -> Training {name}...")
        model.fit(X_train, y_train)
        trained_model_objects[name] = model

        # Predictions & Probabilities
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred

        # Metrics calculation
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred).tolist()

        results.append({
            "model_name": name,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "confusion_matrix": cm
        })

    # Convert results to DataFrame for clean tabular view
    results_df = pd.DataFrame(results).sort_values(by="f1_score", ascending=False)
    
    print("\n" + "=" * 60)
    print(" MODEL PERFORMANCE COMPARISON TABLE")
    print("=" * 60)
    print(results_df[["model_name", "accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]].to_string(index=False))

    # Save Comparison Metrics JSON
    metrics_json_path = os.path.join(MODELS_DIR, "comparison_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(results, f, indent=2)

    # 5. Select Best Model based on F1-Score & ROC-AUC
    best_model_name = results_df.iloc[0]["model_name"]
    best_model = trained_model_objects[best_model_name]
    best_metrics = results_df.iloc[0].to_dict()

    print(f"\n[WINNING MODEL] {best_model_name}")
    print(f" -> F1-Score: {best_metrics['f1_score']} | ROC-AUC: {best_metrics['roc_auc']} | Precision: {best_metrics['precision']} | Recall: {best_metrics['recall']}")

    # Build SHAP explainer for best model
    print("\nComputing SHAP explainer for best model...")
    try:
        if "XGBoost" in best_model_name or "Forest" in best_model_name or "Tree" in best_model_name:
            explainer = shap.TreeExplainer(best_model)
        else:
            explainer = shap.Explainer(best_model, X_train)
    except Exception as e:
        print(f"Notice: SHAP explainer fallback due to: {e}")
        explainer = None

    # Save Best Model Artifact
    best_model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    artifact = {
        "model": best_model,
        "model_name": best_model_name,
        "explainer": explainer,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "metrics": best_metrics
    }

    with open(best_model_path, "wb") as f:
        pickle.dump(artifact, f)

    print(f"\n[SUCCESS] Best model artifact exported to: {best_model_path}")
    print(f"[SUCCESS] Comparison metrics saved to: {metrics_json_path}")
    return results_df, best_model_name

if __name__ == "__main__":
    train_and_compare_models()
