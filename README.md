# TRM-AI — AI Transaction Risk Manager

**Production-Grade Real-Time Payment Fraud Detection & Explainable Risk Intelligence Platform**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-eb6134.svg?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![SQLite](https://img.shields.io/badge/SQLite-3.0+-003B57.svg?logo=sqlite&logoColor=white)](https://sqlite.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A high-performance fraud detection engine engineered for modern financial gateways (e.g. Razorpay). Achieves **99.69% accuracy**, **96.67% fraud recall**, and **0.9997 ROC-AUC** with **sub-40ms authorization latency**, glass-box **SHAP explainability**, an integrated **Gemini AI Copilot**, and a persistent **SQLite authentication gateway**.

---

## 🌐 Live Deployments

| Resource | Link | Description |
|---|---|---|
| 📊 **Interactive Web Dashboard** | [amanmaurya39.github.io/TRM-AI](https://amanmaurya39.github.io/TRM-AI/) | Full-featured SPA: Risk Scorer, Live Pipeline, Leaderboard, AI Copilot, Auth DB |
| 🔌 **FastAPI Production Backend** | [trm-ai.onrender.com](https://trm-ai.onrender.com) | Live REST inference service hosted on Render |
| 📖 **Interactive API Documentation** | [trm-ai.onrender.com/docs](https://trm-ai.onrender.com/docs) | Swagger UI for automated endpoint testing |

> **Note on Render Free Tier:** The backend automatically spins down during inactivity. Initial cold-start takes ~30 seconds; subsequent queries respond in < 40ms. The frontend dashboard includes graceful offline fallbacks so features remain testable at all times.

---

## 📌 The Problem

In high-volume digital payment gateways, risk engineering teams face a critical dilemma: **maximizing fraud prevention without sacrificing checkout conversion rates**.

1. **Massive Financial Losses:** Card-testing bots, credential stuffing, and account takeovers (ATO) lead to costly chargeback penalties, merchant loss, and operational overhead.
2. **The Flaw of Legacy Rule Engines:** Traditional gateways depend on rigid static rules (e.g., `IF amount > ₹50,000 THEN BLOCK`). These engines fail to adapt to complex fraud vectors and generate high **False Positive Rates**, blocking genuine customers and harming merchant revenue.
3. **Strict Latency Constraints:** Fraud detection must execute inside the synchronous payment authorization pipeline within a strict **< 50ms SLA** to prevent cart abandonment.
4. **The "Black Box" Trust Deficit:** Deep learning or ensemble models that output an unexplained score leave risk analysts unable to justify declines during customer dispute resolution.
5. **Operational Silos:** Store merchants and SecOps teams require different operational views; merchants need streamlined authorization tracking, while SecOps requires policy tuning, alert triage, and audit logs.

---

## 💡 My Approach & Engineering Methodology

To solve this, **TRM-AI** was built as a multi-tier, end-to-end intelligence system connecting raw data engineering to an interactive operational frontend:

```
┌─────────────────────────┐     ┌────────────────────────┐     ┌────────────────────────┐
│  50k Gateway Data       │ ──> │  Composite Feature     │ ──> │  Stratified 4-Model    │
│  (1.5% Imbalanced)      │     │  Engineering (EDA)     │     │  Benchmark & Tuning    │
└─────────────────────────┘     └────────────────────────┘     └────────────────────────┘
                                                                            │
┌─────────────────────────┐     ┌────────────────────────┐                  ▼
│  Web SPA Dashboard      │ <── │  FastAPI REST Server   │ <── ┌────────────────────────┐
│  • Risk Simulator       │     │  • < 38ms Inference    │     │  Winning Model Export  │
│  • Live Auth Pipeline   │     │  • SHAP TreeExplainer  │     │  • XGBoost v1.0        │
│  • Gemini AI Copilot    │     │  • Gemini LLM Agent    │     │  • ROC-AUC: 0.9997     │
│  • SQLite Auth DB (RBAC)│     │  • SQLite Auth / Log   │     │  • In-Memory Pipeline  │
└─────────────────────────┘     └────────────────────────┘     └────────────────────────┘
```

### 1. Data Synthesis & Exploratory Data Analysis (EDA)
- Generated a realistic 50,000-transaction financial gateway dataset matching Indian digital payment behaviors (UPI, Cards, NetBanking, Wallets).
- Identified a realistic **1.5% fraud class imbalance** (49,250 legitimate vs 750 fraud).
- Uncovered high-leverage compound risk indicators during bivariate analysis:
  - **New Device Usage (`device_new`):** Fraud rate spikes to **27.15%** on unrecognized devices vs. **0.46%** on recognized devices (**~59x risk surge**).
  - **Geographic Mismatch (`txn_city ≠ home_city`):** Fraud rate jumps to **16.02%** vs. **0.67%** when locations match (**~24x risk surge**).
  - **Late-Night Velocity:** Elevated fraud concentration between 11 PM and 6 AM combined with > 5 transactions/hour.

### 2. Feature Engineering & Preprocessing Pipeline
- Engineered high-signal composite indicators:
  - `geo_mismatch`: Binary flag when transaction city differs from account holder's home city.
  - `is_night`: Binary flag indicating transactions occurring in high-risk off-hours (23:00 to 06:00).
  - `is_new_account`: Incubating accounts younger than 30 days.
  - `high_value_txn`: Value-at-risk trigger exceeding ₹4,000 threshold.
  - `velocity_1h`: Hourly transaction frequency to capture automated card-testing bots.
- Packaged numerical scaling (`StandardScaler`) and categorical encoding (`OneHotEncoder`) into an automated `ColumnTransformer` pipeline.

### 3. Model Training & Class Imbalance Compensation
- Implemented **Stratified K-Fold splits** to preserve minority class distribution.
- Tuned XGBoost's `scale_pos_weight` hyperparameter to penalize false negatives, guaranteeing high sensitivity to fraud attacks.
- Benchmarked 4 distinct model families: **Logistic Regression, Decision Tree, Random Forest, and XGBoost**.

### 4. Glass-Box Real-Time Explainability (SHAP)
- To maintain sub-40ms latency without sacrificing interpretability, integrated a pre-compiled **`TreeExplainer`** directly into the inference payload.
- Every transaction returns exact quantitative risk contributors (e.g. `+28.4% Geo Mismatch`, `+19.1% New Device`, `-5.2% Known Payment Method`).

### 5. Multi-Persona Authentication & SQLite Database
- Implemented an **Open-Entry Gateway**: Anyone can enter custom credentials or use 1-click presets; accounts are automatically salted, hashed with **SHA-256**, and stored in SQLite ([data/auth.db](data/auth.db)).
- Role-Based Access Control (**RBAC**):
  - **Administrator (SecOps):** Full privileges to Risk Models leaderboard, real-time alert simulator, and live SQLite database inspection.
  - **Merchant User:** Clean operational view tailored strictly for payment authorizations and fraud scoring.

---

## 📊 Model Performance & Benchmark Results

### 1. 4-Model Comparative Benchmark (Held-Out Test Set: 10,000 Samples)

| Rank | Model Family | Accuracy | Precision (Fraud) | Recall (Fraud) | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| 🥇 **1** | **XGBoost** *(Production)* | **99.69%** | **84.80%** | **96.67%** | **0.9034** | **0.9997** | **0.9864** |
| 🥈 2 | Random Forest | 99.51% | 76.72% | 96.67% | 0.8555 | 0.9991 | 0.9694 |
| 🥉 3 | Logistic Regression | 99.30% | 69.05% | 96.67% | 0.8056 | 0.9995 | 0.9828 |
| 4 | Decision Tree | 99.26% | 67.76% | 96.67% | 0.7967 | 0.9830 | 0.9529 |

### 2. Confusion Matrix Analysis (Production Evaluation on 12,500 Test Samples)

```
                       PREDICTED CLEAN              PREDICTED FRAUD
ACTUAL CLEAN            12,294 (True Negative)        19 (False Positive)
ACTUAL FRAUD                 8 (False Negative)      179 (True Positive)
```

- **Fraud Capture Rate (95.72% Recall):** Intercepted 179 out of 187 active fraud attempts.
- **Ultra-Low False Positive Rate (0.15%):** Only 19 out of 12,313 clean transactions flagged, protecting merchant checkout conversion.
- **Average Inference Latency:** **38ms**, well within payment gateway SLA limits.

---

## 🎯 4-Tier Autonomous Policy Engine

Rather than binary Accept/Reject decisions, the system dynamically routes payments based on confidence thresholds:

| Policy Tier | Risk Score | Autonomous Action | Customer Experience |
|---|---|---|---|
| 🟢 **Autonomous Approve** | `0.0% – 34.9%` | Instant settlement | Frictionless 1-click checkout |
| 🟡 **Modify / Adaptive 3DS** | `35.0% – 64.9%` | Step-up authentication | OTP / Biometric verification prompt |
| 🟠 **Human Review Gate** | `65.0% – 84.9%` | Temporary hold | Queued to SecOps manual review console |
| 🔴 **Block & Quarantine** | `85.0% – 100.0%` | Hard decline & blacklist | Payment blocked; merchant balance safeguarded |

---

## 📁 Repository Structure

```
razorpay-risk-manager/
├── data/
│   ├── auth.db                    # Persistent SQLite User & Session Database
│   ├── transactions_raw.csv       # 50,000 synthetic payment transactions
│   └── transactions_features.csv  # Feature-engineered dataset
├── docs/                          # GitHub Pages static bundle
│   └── index.html                 # Hosted SPA mirror
├── frontend/
│   └── index.html                 # Main Single Page Application (HTML5/CSS/JS)
├── models/
│   ├── best_model.pkl             # Serialized XGBoost + Preprocessor + Explainer
│   ├── comparison_metrics.json    # 4-Model evaluation benchmarks
│   ├── eda_summary.json           # EDA statistics & risk ratios
│   └── metrics.json               # Production evaluation metrics
├── src/
│   ├── alert_service.py           # Real-time alert feed & Slack webhook dispatcher
│   ├── alert_simulator.py         # Automated threat feed background generator
│   ├── api.py                     # FastAPI REST server & routing
│   ├── auth_service.py            # SQLite connection, hashing & session manager
│   ├── chat.py                    # Gemini-powered SecOps AI Copilot
│   ├── data_generator.py          # Synthetic dataset generator
│   ├── eda.py                     # Exploratory Data Analysis execution script
│   ├── feedback_service.py        # Active learning human-in-the-loop feedback loop
│   ├── preprocess.py              # Preprocessing pipeline & feature engineering
│   └── train_models.py            # Model training & comparative benchmark suite
├── index.html                     # Root SPA entrypoint
├── PROJECT_LOG.md                 # Chronological development logbook
├── requirements.txt               # Python package dependencies
└── README.md                      # Comprehensive project documentation
```

---

## ⚡ Quick Start & Local Setup

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/amanmaurya39/TRM-AI.git
cd TRM-AI
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Reproduce Pipeline (Data -> EDA -> Models)
```bash
# Generate 50,000 transaction dataset
python src/data_generator.py

# Run Exploratory Data Analysis
python src/eda.py

# Build preprocessing pipeline
python src/preprocess.py

# Train & benchmark all 4 models, export best_model.pkl
python src/train_models.py
```

### 3. Run FastAPI Backend Server (Port 8000)
```bash
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```
*API Swagger documentation available at:* `http://127.0.0.1:8000/docs`

### 4. Serve Web Dashboard (Port 3000)
```bash
python -m http.server 3000 --directory frontend
```
*Open in browser:* `http://localhost:3000`

---

## 🔌 REST API Endpoints Overview

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/auth/login` | Authenticate or auto-register user, writes to SQLite |
| `POST` | `/auth/register` | Direct user account provisioning into `data/auth.db` |
| `GET` | `/auth/users` | List all registered users and login counts from SQLite |
| `GET` | `/auth/sessions` | Retrieve security audit trail of recent login sessions |
| `POST` | `/predict` | Real-time fraud scoring with SHAP feature breakdown |
| `GET` | `/metrics` | Retrieve 4-model performance benchmark comparison |
| `POST` | `/chat` | Gemini-powered SecOps analyst copilot chat |
| `POST` | `/feedback` | Submit analyst verdict (`CONFIRMED_FRAUD` / `FALSE_POSITIVE`) |
| `GET` | `/feedback` | Retrieve model agreement and active-learning metrics |
| `POST` | `/feedback/recalibrate` | Trigger threshold recalibration simulation |
| `GET` | `/alerts` | Stream real-time high-risk fraud alerts |
| `POST` | `/alerts/test-webhook` | Dispatch test notification to configured Slack webhook |
| `GET` | `/health` | Service uptime and loaded model status |

---

## 💡 What Makes the Project Useful

The primary objective of **TRM-AI** was never just to achieve a high accuracy number on a static dataset. The true value lies in demonstrating a **complete, production-grade machine learning lifecycle** that bridges the gap between raw data science and an actual, usable application:

- **End-to-End Pipeline Integration:** Rather than an isolated proof-of-concept, TRM-AI combines data generation, exploratory data analysis, automated feature engineering, multi-model benchmarking, explainable AI (SHAP), high-performance REST APIs, and modern frontend deployment into a cohesive system.
- **Moving Beyond Jupyter Notebooks:** Many ML projects remain trapped inside experimental notebooks. TRM-AI demonstrates how to take a trained model and operationalize it into an ultra-low latency (< 40ms) authorization service capable of live evaluation, security auditing, and persistent user management.
- **Glass-Box Trust & Explainability:** By providing real-time SHAP attribution factors, it removes the mystery of "black box" decisions, helping risk analysts justify declines and resolve customer disputes transparently.
- **Dual-Persona Utility:** With dedicated **Admin (SecOps)** and **Merchant User** views, it solves real workflow fragmentation—giving store owners simple payment tracking while giving security teams deep model telemetry and database inspection.

---

## 🔮 Future Scope & Roadmap

There are several key directions planned to take TRM-AI further toward enterprise scale:

1. **Massive Real-World Transaction Datasets:** Expanding training and validation on multi-million row production datasets with diverse international payment vectors, merchant-specific MCCs, and seasonal shopping surges.
2. **Continuous Active Learning & Automated Retraining:** Expanding the built-in feedback loop (`/feedback` and `/recalibrate`) to automatically detect concept drift, retrain tree weights, and recalibrate decision boundaries when new attack vectors emerge.
3. **Direct Gateway Integrations:** Building drop-in middleware and webhooks for live payment gateways (e.g., Razorpay, Stripe) to score incoming checkout webhooks inline before capture.
4. **Distributed Streaming & Feature Stores:** Scaling localized SQLite storage to distributed streaming architectures (Apache Kafka / Redis) to support real-time sliding-window velocity aggregations across millions of cardholders.
5. **Enterprise Alert Routing:** Broadening the alert simulator to trigger real-time PagerDuty escalations, Opsgenie alerts, and automated SIEM event streams (Splunk, Datadog).

---

## 🛠️ Technology Stack

- **Machine Learning Engine:** XGBoost, Scikit-learn, Pandas, NumPy
- **Model Explainability (XAI):** SHAP (`TreeExplainer`)
- **Backend Framework:** FastAPI, Uvicorn, Pydantic
- **Generative AI Copilot:** Google Gemini API (`google.genai` / `google.generativeai`)
- **Database & Storage:** SQLite3 (`data/auth.db`), Pickled Pipeline Artifacts
- **Frontend Architecture:** Vanilla HTML5, Modern CSS (Design tokens, glassmorphism, responsive grid), Vanilla JavaScript (SPA Hash Router, asynchronous polling)
- **Deployment & CI/CD:** GitHub Pages (Static frontend hosting), Render (FastAPI Cloud Container)

---

## 📜 Development Logbook

A step-by-step chronological logbook covering every technical decision, dataset finding, model loss curve, and commit history is maintained in [PROJECT_LOG.md](PROJECT_LOG.md).
