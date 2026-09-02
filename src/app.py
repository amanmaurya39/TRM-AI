"""
AI Risk Manager — Fraud Detection Dashboard
A Streamlit app simulating a real-time transaction risk monitoring system.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="AI Risk Manager | Fraud Detection", layout="wide", page_icon="🛡️")

MODEL_PATH = "/home/claude/razorpay-risk-manager/models/fraud_model.pkl"
METRICS_PATH = "/home/claude/razorpay-risk-manager/models/metrics.json"

FEATURE_LABELS = {
    "amount": "Transaction amount",
    "hour_of_day": "Hour of day",
    "is_night": "Night-time transaction",
    "geo_mismatch": "Location mismatch from home city",
    "device_new": "New/unrecognized device",
    "is_new_account": "New account (<30 days)",
    "high_value_txn": "Unusually high transaction value",
    "account_age_days": "Account age",
    "txn_velocity_1h": "Transactions in last 1 hour",
}


@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
    return artifact, metrics


def human_label(col):
    if col in FEATURE_LABELS:
        return FEATURE_LABELS[col]
    if col.startswith("merchant_category_"):
        return f"Merchant: {col.replace('merchant_category_', '')}"
    if col.startswith("payment_method_"):
        return f"Payment method: {col.replace('payment_method_', '')}"
    return col


def explain_transaction(shap_values_row, feature_names, X_row, top_n=4):
    """Turn SHAP values into a plain-language risk explanation."""
    contributions = list(zip(feature_names, shap_values_row, X_row.values))
    contributions = [c for c in contributions if abs(c[1]) > 0.01]
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    top = contributions[:top_n]
    reasons = []
    for feat, val, raw in top:
        label = human_label(feat)
        direction = "increased" if val > 0 else "decreased"
        if feat == "amount":
            reasons.append(f"Amount of ₹{raw:,.0f} {direction} risk")
        elif feat in ("geo_mismatch", "device_new", "is_new_account", "high_value_txn", "is_night"):
            state = "flagged" if raw == 1 else "not flagged"
            if raw == 1:
                reasons.append(f"{label} {direction} risk")
        elif feat == "txn_velocity_1h":
            reasons.append(f"{int(raw)} transactions in the last hour {direction} risk")
        else:
            reasons.append(f"{label} {direction} risk")
    return reasons


artifact, metrics = load_artifacts()
model = artifact["model"]
explainer = artifact["explainer"]
feature_names = artifact["feature_names"]
X_test = artifact["X_test"]
test_df = artifact["test_df"]

# ---------------- HEADER ----------------
st.markdown("## 🛡️ AI Risk Manager — Real-Time Fraud Detection")
st.caption("Built on a synthetic transaction dataset modeled after payment-gateway behavior · XGBoost + SHAP explainability")

# ---------------- TOP METRICS ----------------
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
col2.metric("Precision (Fraud)", f"{metrics['precision_fraud']*100:.1f}%")
col3.metric("Recall (Fraud)", f"{metrics['recall_fraud']*100:.1f}%")
col4.metric("Transactions Scored", f"{metrics['n_test']:,}")
col5.metric("Features Used", metrics['n_features'])

st.divider()

# ---------------- SIDEBAR CONTROLS ----------------
st.sidebar.header("⚙️ Live Feed Controls")
n_show = st.sidebar.slider("Transactions to show", 10, 100, 30)
risk_threshold = st.sidebar.slider("Risk flag threshold", 0.0, 1.0, 0.5, 0.05)
sort_by_risk = st.sidebar.checkbox("Sort by risk score (highest first)", value=True)

st.sidebar.divider()
st.sidebar.markdown("**Model:** XGBoost Classifier")
st.sidebar.markdown("**Explainability:** SHAP (TreeExplainer)")
st.sidebar.markdown("**Confusion Matrix (test set):**")
cm = metrics["confusion_matrix"]
st.sidebar.write(pd.DataFrame(cm, index=["Actual: Legit", "Actual: Fraud"],
                               columns=["Pred: Legit", "Pred: Fraud"]))

# ---------------- SCORE SAMPLE TRANSACTIONS ----------------
sample_idx = X_test.sample(n=min(n_show, len(X_test)), random_state=1).index
X_sample = X_test.loc[sample_idx]
context_sample = test_df.loc[sample_idx]

risk_scores = model.predict_proba(X_sample)[:, 1]
shap_values = explainer.shap_values(X_sample)

display_df = context_sample[["txn_id", "user_id", "amount", "timestamp", "is_fraud"]].copy()
display_df["risk_score"] = risk_scores
display_df["flagged"] = display_df["risk_score"] >= risk_threshold

if sort_by_risk:
    display_df = display_df.sort_values("risk_score", ascending=False)

# ---------------- MAIN LAYOUT ----------------
left, right = st.columns([1.4, 1])

with left:
    st.subheader("📊 Live Transaction Feed")

    def risk_color(score):
        if score >= 0.7:
            return "🔴"
        elif score >= risk_threshold:
            return "🟠"
        return "🟢"

    display_df_show = display_df.copy()
    display_df_show["Risk"] = display_df_show["risk_score"].apply(lambda s: f"{risk_color(s)} {s:.2%}")
    display_df_show["Amount (₹)"] = display_df_show["amount"].apply(lambda a: f"{a:,.0f}")
    display_df_show["Actual Label"] = display_df_show["is_fraud"].apply(lambda x: "Fraud" if x == 1 else "Legit")

    st.dataframe(
        display_df_show[["txn_id", "user_id", "Amount (₹)", "timestamp", "Risk", "Actual Label"]].rename(
            columns={"txn_id": "Transaction ID", "user_id": "User", "timestamp": "Time"}
        ),
        use_container_width=True,
        height=500
    )

    flagged_count = display_df["flagged"].sum()
    st.info(f"🚨 **{flagged_count}** of {len(display_df)} shown transactions flagged as high-risk at current threshold.")

with right:
    st.subheader("🔍 Investigate a Transaction")
    selected_txn = st.selectbox("Select transaction ID", display_df["txn_id"].tolist())

    # map txn_id back to position in X_sample
    idx_label = context_sample[context_sample["txn_id"] == selected_txn].index[0]
    pos = list(X_sample.index).index(idx_label)

    score = risk_scores[pos]
    row_shap = shap_values[pos]
    row_X = X_sample.iloc[pos]
    row_ctx = context_sample.loc[idx_label]

    st.metric("Risk Score", f"{score:.1%}", delta="HIGH RISK" if score >= risk_threshold else "Low Risk",
              delta_color="inverse" if score >= risk_threshold else "normal")

    st.markdown(f"**Amount:** ₹{row_ctx['amount']:,.2f}")
    st.markdown(f"**User:** {row_ctx['user_id']}")
    st.markdown(f"**Time:** {row_ctx['timestamp']}")
    st.markdown(f"**Actual outcome:** {'🔴 Confirmed Fraud' if row_ctx['is_fraud']==1 else '🟢 Legitimate'}")

    st.markdown("#### 🧠 AI Explanation")
    reasons = explain_transaction(row_shap, feature_names, row_X)
    if reasons:
        explanation_text = "This transaction was flagged because: " + "; ".join(reasons) + "."
        st.warning(explanation_text) if score >= risk_threshold else st.success(explanation_text)
    else:
        st.write("No significant risk factors detected.")

    with st.expander("View raw SHAP feature contributions"):
        shap_df = pd.DataFrame({
            "feature": [human_label(f) for f in feature_names],
            "shap_value": row_shap
        }).sort_values("shap_value", key=abs, ascending=False).head(10)
        fig = px.bar(shap_df, x="shap_value", y="feature", orientation="h",
                     color="shap_value", color_continuous_scale=["#2ca02c", "#d62728"],
                     title="Top feature contributions to risk score")
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------- ANALYTICS ----------------
st.subheader("📈 Risk Analytics")

a1, a2 = st.columns(2)

with a1:
    fig_dist = px.histogram(display_df, x="risk_score", nbins=30, color="is_fraud",
                             labels={"risk_score": "Risk Score", "is_fraud": "Actual Fraud"},
                             color_discrete_map={0: "#2ca02c", 1: "#d62728"},
                             title="Risk Score Distribution (colored by actual outcome)")
    st.plotly_chart(fig_dist, use_container_width=True)

with a2:
    global_shap_mean = np.abs(shap_values).mean(axis=0)
    imp_df = pd.DataFrame({
        "feature": [human_label(f) for f in feature_names],
        "importance": global_shap_mean
    }).sort_values("importance", ascending=False).head(10)
    fig_imp = px.bar(imp_df, x="importance", y="feature", orientation="h",
                      title="Top Global Risk Drivers (mean |SHAP value|)")
    fig_imp.update_layout(height=400)
    st.plotly_chart(fig_imp, use_container_width=True)

st.caption("Prototype for AI Risk Manager track — synthetic data modeled on payment-gateway transaction patterns. Not connected to live Razorpay systems.")
