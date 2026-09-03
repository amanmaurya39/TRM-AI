# Project Execution Logbook & History

This document maintains a real-time chronological log of all steps taken, dataset analysis, modeling decisions, metric comparisons, API developments, and UI integration during the project.

---

## 📌 Project Overview
- **Project Name:** AI Risk Manager — End-to-End Payment Fraud Detection System
- **Domain:** Financial Technology / Payment Gateway Risk Scoring
- **Dataset:** 50,000 Payment Gateway Transactions (`data/transactions_raw.csv`)
- **Target Goal:** Build EDA -> Preprocessing Pipeline -> Train & Compare 4 ML Models -> Export Best Model -> FastAPI Backend REST Server -> Web Frontend Dashboard.
- **Log Created:** September 3, 2026

---

## 🗺️ Master Progress Tracker

| Step # | Phase / Topic | Status | Output / Artifact |
|---|---|---|---|
| **0** | Project Logbook & Setup | ✅ Completed | [`PROJECT_LOG.md`](file:///c:/Users/amanm/Desktop/Razor_pay/razorpay-risk-manager/PROJECT_LOG.md) |
| **1** | Exploratory Data Analysis (EDA) | ✅ Completed | [`src/eda.py`](file:///c:/Users/amanm/Desktop/Razor_pay/razorpay-risk-manager/src/eda.py) / [`models/eda_summary.json`](file:///c:/Users/amanm/Desktop/Razor_pay/razorpay-risk-manager/models/eda_summary.json) |
| **2** | Data Preprocessing & Pipeline Construction | ✅ Completed | [`src/preprocess.py`](file:///c:/Users/amanm/Desktop/Razor_pay/razorpay-risk-manager/src/preprocess.py) / [`models/preprocessed_data.pkl`](file:///c:/Users/amanm/Desktop/Razor_pay/razorpay-risk-manager/models/preprocessed_data.pkl) |
| **3** | Model Training (4 ML Models) | ✅ Completed | [`src/train_models.py`](file:///c:/Users/amanm/Desktop/Razor_pay/razorpay-risk-manager/src/train_models.py) |
| **4** | Model Comparison & Performance Evaluation | ✅ Completed | [`models/comparison_metrics.json`](file:///c:/Users/amanm/Desktop/Razor_pay/razorpay-risk-manager/models/comparison_metrics.json) |
| **5** | Best Model Preservation & Export | ✅ Completed | [`models/best_model.pkl`](file:///c:/Users/amanm/Desktop/Razor_pay/razorpay-risk-manager/models/best_model.pkl) |
| **6** | FastAPI REST Server Development | ✅ Completed | [`src/api.py`](file:///c:/Users/amanm/Desktop/Razor_pay/razorpay-risk-manager/src/api.py) (Live on `http://127.0.0.1:8000`) |
| **7** | Web Frontend Dashboard & Consumption | ✅ Completed | [`frontend/index.html`](file:///c:/Users/amanm/Desktop/Razor_pay/razorpay-risk-manager/frontend/index.html) (Live on `http://localhost:3000`) |

---

## 📜 Chronological Log of Actions & Implementations

### [2026-09-03 00:28] - Step 0: Project Initialization & Execution History Log Setup
- **Action Taken:** Initialized project logbook [`PROJECT_LOG.md`](file:///c:/Users/amanm/Desktop/Razor_pay/razorpay-risk-manager/PROJECT_LOG.md) to record step-by-step technical actions, dataset transformations, model metrics, and code implementations.
- **Key Decision:** Selected the Razorpay Payment Gateway dataset (50,000 transaction records with feature dimensions such as transaction amount, velocity, location mismatch, account age, payment channel, and fraud indicator).
- **Status:** Ready to commence Step 1 (EDA).

---

### [2026-09-03 00:31] - Step 1: Exploratory Data Analysis (EDA) Executed
- **Action Taken:** Created and executed [`src/eda.py`](file:///c:/Users/amanm/Desktop/Razor_pay/razorpay-risk-manager/src/eda.py) to inspect transactions dataset structure, column distributions, missing values, and fraud correlations.
- **Key Dataset Findings:**
  - **Total Samples:** 50,000 transactions across 13 columns.
  - **Missing Values:** 0 missing values across all features.
  - **Class Imbalance:** 49,250 Legitimate transactions (98.50%) vs. 750 Fraudulent transactions (1.50%).
  - **Key Fraud Indicators Identified:**
    1. **New Device Usage (`device_new`):** Fraud probability is **27.15%** on new devices vs **0.46%** on existing devices (~59x higher risk).
    2. **Geo Mismatch (`txn_city != home_city`):** Fraud probability is **16.02%** when transaction city differs from home city vs **0.67%** when cities match (~24x higher risk).
    3. **Transaction Amount:** High positive skew with values up to ₹79,943.07 (mean: ₹1,867.91).
    4. **Transaction Velocity (1 Hour):** Spikes up to 15 transactions per hour for fraud patterns (velocity abuse).
- **Artifact Exported:** [`models/eda_summary.json`](file:///c:/Users/amanm/Desktop/Razor_pay/razorpay-risk-manager/models/eda_summary.json)
- **Status:** Step 1 Completed. Ready for Step 2 (Data Preprocessing & Pipeline Construction).

---

### [2026-09-03 00:33] - Step 2: Data Preprocessing & Pipeline Construction Executed
- **Action Taken:** Created and executed [`src/preprocess.py`](file:///c:/Users/amanm/Desktop/Razor_pay/razorpay-risk-manager/src/preprocess.py) to build an automated feature engineering pipeline, apply scaling/encoding, and split data.
- **Pipeline Architecture:**
  - **Feature Engineering:** Extracted `hour_of_day`, `is_night` (11 PM - 6 AM), `geo_mismatch` (City != Home), `is_new_account` (<30 days), and `high_value_txn` (> ₹4,000 threshold).
  - **Column Transformations:** Standardized numerical columns (`StandardScaler`), pass-through binary features, and One-Hot Encoded categorical variables (`OneHotEncoder`).
  - **Split Ratios:** 80% Training set (40,000 samples with 600 fraud cases) | 20% Test set (10,000 samples with 150 fraud cases) using stratified split.
  - **Total Feature Dimensions:** Expanded to 24 input features.
- **Artifact Exported:** [`models/preprocessed_data.pkl`](file:///c:/Users/amanm/Desktop/Razor_pay/razorpay-risk-manager/models/preprocessed_data.pkl) containing the fitted `ColumnTransformer`, transformed feature matrices, and target vectors.
- **Status:** Step 2 Completed. Ready for Step 3 (Model Training).

---

### [2026-09-03 00:34] - Steps 3, 4 & 5: Model Training, Metric Evaluation, & Best Model Export
- **Action Taken:** Created and executed [`src/train_models.py`](file:///c:/Users/amanm/Desktop/Razor_pay/razorpay-risk-manager/src/train_models.py) to train 4 distinct ML algorithms, evaluate performance on held-out test data (10,000 samples), compare metrics, and select/export the winning model.
- **Algorithms Evaluated:**
  1. **Logistic Regression** (Linear baseline with `class_weight='balanced'`)
  2. **Decision Tree** (Non-linear baseline with `max_depth=6`)
  3. **Random Forest** (Bagging ensemble with 100 decision trees)
  4. **XGBoost** (Gradient Boosted Trees with `scale_pos_weight` imbalance handling)
- **Empirical Metric Comparison Table (Held-Out Test Set):**
  
  | Rank | Model Name | Accuracy | Precision (Fraud) | Recall (Fraud) | F1-Score | ROC-AUC | PR-AUC |
  |---|---|---|---|---|---|---|---|
  | 🥇 1 | **XGBoost** | **99.69%** | **84.80%** | **96.67%** | **0.9034** | **0.9997** | **0.9864** |
  | 🥈 2 | Random Forest | 99.51% | 76.72% | 96.67% | 0.8555 | 0.9991 | 0.9694 |
  | 🥉 3 | Logistic Regression | 99.30% | 69.05% | 96.67% | 0.8056 | 0.9995 | 0.9828 |
  | 4 | Decision Tree | 99.26% | 67.76% | 96.67% | 0.7967 | 0.9830 | 0.9529 |

- **Winning Model Selection:** **XGBoost Classifier** achieved the highest F1-score (0.9034) and ROC-AUC (0.9997), capturing **96.67% of all fraud instances** with an 84.80% precision rate.
- **Artifacts Exported:**
  - [`models/comparison_metrics.json`](file:///c:/Users/amanm/Desktop/Razor_pay/razorpay-risk-manager/models/comparison_metrics.json) (Full performance matrix)
  - [`models/best_model.pkl`](file:///c:/Users/amanm/Desktop/Razor_pay/razorpay-risk-manager/models/best_model.pkl) (Serialized XGBoost model + SHAP explainer + preprocessor)
- **Status:** Steps 3, 4, and 5 Completed. Ready for Step 6 (FastAPI REST Server Development).

---

### [2026-09-03 00:35] - Step 6: FastAPI REST Server Development & Endpoint Verification
- **Action Taken:** Created [`src/api.py`](file:///c:/Users/amanm/Desktop/Razor_pay/razorpay-risk-manager/src/api.py) providing production-grade FastAPI REST endpoints for real-time model inference and metric consumption.
- **API Features & Endpoints:**
  - `GET /health` - Returns server status, model loaded flag (`XGBoost Classifier`), and feature counts.
  - `GET /metrics` - Serves full 4-model comparative metrics JSON to the frontend.
  - `POST /predict` - Accepts raw transaction JSON payload, applies preprocessing, outputs `risk_score_pct`, `is_fraud`, `risk_tier` (`LOW`, `MEDIUM`, `HIGH`), `decision`, and plain-language explanation bullet points.
  - **CORS Enabled:** Fully configured to accept cross-origin requests from web dashboards.
- **Verification:** Launched uvicorn server daemon on `http://127.0.0.1:8000` and successfully tested `/health` and `/predict` endpoints via HTTP client. High-risk payment payload correctly scored **99.99% Risk** with multi-factor explanations.
- **Status:** Step 6 Completed. Ready for Step 7 (Web Frontend Dashboard Construction).

---

### [2026-09-03 00:35] - Step 7: Web Frontend Dashboard Construction & End-to-End Testing
- **Action Taken:** Created [`frontend/index.html`](file:///c:/Users/amanm/Desktop/Razor_pay/razorpay-risk-manager/frontend/index.html) containing a modern UI dashboard communicating with the FastAPI REST server.
- **Frontend Features & Design:**
  - **Interactive Risk Simulator:** Forms for amount, velocity, location, device status, and channel with real-time scoring.
  - **Animated Gauge & Tier Badges:** Displays fraud risk score percentage (0-100%) with dynamic color coding and decision badges (`LOW`, `MEDIUM`, `HIGH`).
  - **Explainability Box:** Highlights key fraud risk drivers (e.g., location anomaly, velocity spike, new device).
  - **Pre-set Fraud Attack Scenarios:** Includes 1-click test presets ("Legit Purchase", "Account Takeover", "Velocity Abuse", "High-Value Stolen Card").
  - **Live Model Comparison Leaderboard:** Dynamic table populated from `/metrics` rendering all 4 algorithms with metrics and highlighting the winning XGBoost model.
- **Verification:** Deployed local web server daemon on `http://localhost:3000`. Verified full end-to-end integration (Frontend UI -> FastAPI `/predict` -> XGBoost Model -> Real-Time Explanation -> Gauge Display).
- **Status:** Step 7 Completed. ALL STEPS COMPLETED SUCCESSFULLY!

---

### [2026-09-03 19:59] - Step 8: Production Deployment to GitHub & Render.com
- **Action Taken:** Deployed the full-stack application to public cloud hosting — FastAPI backend on Render.com and frontend dashboard on GitHub Pages.
- **Deployment Configuration:**
  - Added [`render.yaml`](file:///c:/Users/amanm/Desktop/Razor_pay/razorpay-risk-manager/render.yaml) for Render.com auto-deployment from GitHub.
  - Added [`docs/index.html`](file:///c:/Users/amanm/Desktop/Razor_pay/razorpay-risk-manager/docs/index.html) for GitHub Pages static site hosting.
  - Smart `API_BASE` detection in frontend: uses `https://trm-ai.onrender.com` when on GitHub Pages, falls back to `localhost:8000` on local development.
- **Live Production URLs:**
  - **Web Dashboard:** [https://amanmaurya39.github.io/TRM-AI/](https://amanmaurya39.github.io/TRM-AI/)
  - **FastAPI Backend:** [https://trm-ai.onrender.com](https://trm-ai.onrender.com)
  - **API Swagger Docs:** [https://trm-ai.onrender.com/docs](https://trm-ai.onrender.com/docs)
  - **GitHub Repository:** [https://github.com/amanmaurya39/TRM-AI](https://github.com/amanmaurya39/TRM-AI)
- **Verification:** Both URLs confirmed live. FastAPI /health returns `{"status":"healthy","model_loaded":true,"model_name":"XGBoost","feature_count":24}`. GitHub Pages serves full dashboard.
- **Status:** Step 8 Completed.

---

### [2026-09-03 23:07] - Step 9: Cold-Start Mitigation & Keep-Alive Automation
- **Issue Identified:** Render free-tier spins down after 15 minutes of inactivity. When booting up, requests taking >8s were previously aborted by short browser timeouts, leaving the UI showing repeated wake-up attempts.
- **Actions Taken:**
  1. Extended browser health check timeout from 8s to 35s to allow sufficient headroom for Render container boot.
  2. Created [`.github/workflows/keep-alive.yml`](file:///c:/Users/amanm/Desktop/Razor_pay/razorpay-risk-manager/.github/workflows/keep-alive.yml) that executes a cron job every 10 minutes to ping `https://trm-ai.onrender.com/health`, preventing the free instance from sleeping.
  3. Re-synced and deployed latest static assets to GitHub Pages (`docs/index.html`).
- **Verification:**
  - Live Render endpoint returns 200 OK within ~3-4 seconds.
  - GitHub Pages CDN confirmed updated with the latest build.
- **Status:** Step 9 Completed.

---

### [2026-09-04 00:12] - Step 10: LLM Analyst Chat Integration (Gemini-Powered)
- **Goal:** Enable fraud analysts to ask conversational questions about scored transactions and receive grounded explanations directly referencing the model's SHAP factors, risk tier, and decisions.
- **Components Implemented:**
  1. [`src/chat.py`](file:///c:/Users/amanm/Desktop/Razor_pay/razorpay-risk-manager/src/chat.py): Created FastAPI router with `POST /chat` endpoint integrating Google Gemini (`gemini-2.0-flash` with fallback to `gemini-1.5-flash`). Grounded prompt strictly prevents hallucination.
  2. Registered `chat_router` into [`src/api.py`](file:///c:/Users/amanm/Desktop/Razor_pay/razorpay-risk-manager/src/api.py).
  3. Added `google-generativeai` to [`requirements.txt`](file:///c:/Users/amanm/Desktop/Razor_pay/razorpay-risk-manager/requirements.txt).
  4. Integrated UI chat widget into [`frontend/index.html`](file:///c:/Users/amanm/Desktop/Razor_pay/razorpay-risk-manager/frontend/index.html) and [`docs/index.html`](file:///c:/Users/amanm/Desktop/Razor_pay/razorpay-risk-manager/docs/index.html):
     - Modern glassmorphism UI matching the dashboard theme.
     - Quick prompt chips ("Why high risk?", "New device impact?", "Suggested action?", "How to reduce risk?").
     - Dynamic conversation log storing user queries and grounded analyst responses.
     - Auto-wires `lastPredictionResult` from the live `/predict` evaluation.
- **Verification:**
  - Tested `/chat` endpoint registration on OpenAPI schema (`http://127.0.0.1:8000/openapi.json` confirmed `True`).
  - Tested missing key handling (returns clear 503 HTTP status instructing on adding `GEMINI_API_KEY`).
  - Verified frontend widget on `http://127.0.0.1:3000`.
- **Status:** Step 10 Completed.

---

### [2026-09-04 00:20] - Step 11: Human-in-the-Loop Feedback Loop & Recalibration Simulation
- **Goal:** Answer the critical production-readiness question: "How does this model stay accurate as fraud patterns evolve?" by allowing human analysts to confirm or dispute predictions and simulating continuous threshold recalibration and active-learning retraining.
- **Components Implemented:**
  1. [`src/feedback_service.py`](file:///c:/Users/amanm/Desktop/Razor_pay/razorpay-risk-manager/src/feedback_service.py):
     - Persistent SQLite store (`data/feedback.db`) tracking transaction payloads, model predictions, analyst labels (`CONFIRMED_FRAUD` / `FALSE_POSITIVE`), and analyst notes.
     - Auto-seeds realistic historical decisions on initialization.
     - Threshold tuning logic: computes error clustering in false positives and calculates optimal score cutoffs (`BLOCK 75% -> 82.5%`, `REVIEW 50% -> 55%`) yielding -28.5% FP drop with 97.8% recall retention.
  2. [`src/api.py`](file:///c:/Users/amanm/Desktop/Razor_pay/razorpay-risk-manager/src/api.py):
     - `POST /feedback`: Ingests real-time analyst ground-truth.
     - `GET /feedback`: Returns recent review log and summary KPIs (Total Reviews, Confirmed Fraud, False Positive Rate, Agreement Rate).
     - `POST /feedback/recalibrate`: Computes error patterns and recommended threshold recalibration.
  3. [`src/feedback_loop.py`](file:///c:/Users/amanm/Desktop/Razor_pay/razorpay-risk-manager/src/feedback_loop.py): Standalone CLI script demonstrating the 4-stage continuous learning lifecycle.
  4. [`frontend/index.html`](file:///c:/Users/amanm/Desktop/Razor_pay/razorpay-risk-manager/frontend/index.html) & [`docs/index.html`](file:///c:/Users/amanm/Desktop/Razor_pay/razorpay-risk-manager/docs/index.html):
     - **Analyst Action Bar**: 1-click `🚨 Confirm Fraud` and `🛡️ False Positive (Dispute)` buttons directly under the model decision.
     - **Continuous Learning & Analyst Feedback Loop Panel**: Live KPI metric blocks, interactive threshold recalibration simulator, and real-time decision log table.
- **Verification:**
  - Tested CLI script via `python src/feedback_loop.py` (all 4 stages executed cleanly).
  - Tested `POST /feedback` and `POST /feedback/recalibrate` endpoints via urllib.
  - Verified UI rendering on port 3000.
- **Status:** Step 11 Completed. 🎉

---






