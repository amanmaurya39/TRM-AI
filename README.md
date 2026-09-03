# TRM-AI — AI Transaction Risk Manager

**End-to-End Machine Learning & Full-Stack Fraud Detection System**

A real-time payment transaction fraud detection engine built with XGBoost, FastAPI, and a modern web dashboard. Achieves **99.69% accuracy** and **0.9997 ROC-AUC** on held-out test data.

## 🌐 Live Demo

| | URL |
|---|---|
| 📊 **Web Dashboard** | [amanmaurya39.github.io/TRM-AI](https://amanmaurya39.github.io/TRM-AI/) |
| 🔌 **FastAPI Backend** | [trm-ai.onrender.com](https://trm-ai.onrender.com) |
| 📖 **API Swagger Docs** | [trm-ai.onrender.com/docs](https://trm-ai.onrender.com/docs) |

> **Note:** The free Render tier may take ~30 seconds to wake up after idle. Once warmed up, predictions are instant.

---

## 🏗️ Project Architecture

```
Data (50k Txns) → EDA → Preprocessing → Train 4 Models → Best Model → FastAPI → Web Dashboard
```

## 📁 Project Structure

```
razorpay-risk-manager/
├── data/                          # Dataset (auto-generated)
│   ├── transactions_raw.csv       # ← generate with src/data_generator.py
│   └── transactions_features.csv  # ← generated automatically
├── models/
│   ├── best_model.pkl             # Trained XGBoost + SHAP explainer + preprocessor
│   ├── comparison_metrics.json    # 4-model benchmark results
│   └── eda_summary.json           # EDA statistics
├── src/
│   ├── data_generator.py          # Synthetic dataset generator (50k transactions)
│   ├── eda.py                     # Exploratory Data Analysis
│   ├── preprocess.py              # Feature engineering & preprocessing pipeline
│   ├── train_models.py            # Train & compare 4 ML models, export best
│   └── api.py                     # FastAPI REST server (predict, metrics, health)
├── frontend/
│   └── index.html                 # Interactive web dashboard
├── PROJECT_LOG.md                 # Chronological development logbook
├── requirements.txt
└── README.md
```

---

## 📊 Model Performance (Held-Out Test Set: 10,000 samples)

| Rank | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|---|
| 🥇 1 | **XGBoost** | **99.69%** | **84.80%** | **96.67%** | **0.9034** | **0.9997** |
| 🥈 2 | Random Forest | 99.51% | 76.72% | 96.67% | 0.8555 | 0.9991 |
| 🥉 3 | Logistic Regression | 99.30% | 69.05% | 96.67% | 0.8056 | 0.9995 |
| 4 | Decision Tree | 99.26% | 67.76% | 96.67% | 0.7967 | 0.9830 |

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate the Dataset
```bash
python src/data_generator.py
```

### 3. Run EDA
```bash
python src/eda.py
```

### 4. Build Preprocessing Pipeline
```bash
python src/preprocess.py
```

### 5. Train All 4 Models & Export Best
```bash
python src/train_models.py
```

### 6. Launch FastAPI Backend (port 8000)
```bash
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### 7. Serve Web Dashboard (port 3000)
```bash
python -m http.server 3000 --directory frontend
```

### 8. Open Browser
- **Dashboard:** http://localhost:3000
- **API Docs (Swagger):** http://localhost:8000/docs

---

## 🌐 REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Server status & loaded model info |
| `GET` | `/metrics` | 4-model comparison benchmark table |
| `POST` | `/predict` | Score a transaction in real time |

### Example `/predict` Request
```json
{
  "amount": 25000.00,
  "account_age_days": 5,
  "txn_velocity_1h": 6,
  "txn_city": "Mumbai",
  "home_city": "Delhi",
  "device_new": true,
  "merchant_category": "Electronics",
  "payment_method": "Credit Card",
  "txn_hour": 2
}
```

### Example Response
```json
{
  "is_fraud": true,
  "risk_score_pct": 99.99,
  "risk_tier": "HIGH",
  "decision": "BLOCK & STEP-UP AUTHENTICATION REQUIRED",
  "model_version": "XGBoost v1.0",
  "explanation": [
    "Transaction initiated from a NEW / unrecognised device (+27% risk baseline).",
    "Geographic anomaly: Transaction in Mumbai differs from home location (Delhi).",
    "Velocity spike: High frequency of 6 transactions within 1 hour.",
    "High-value payment: Transaction amount (INR 25,000.00) exceeds normal spend threshold.",
    "New account activity: Account created only 5 days ago.",
    "Late-night transaction window: Transaction at 02:00 (11 PM - 6 AM is high-risk)."
  ],
  "processed_at": "2026-09-03T00:35:02.874936"
}
```

---

## 🔍 Key Fraud Signals Discovered in EDA

| Signal | Legit Rate | Fraud Rate | Risk Multiplier |
|---|---|---|---|
| New Device (`device_new=True`) | 0.46% | 27.15% | ~59x |
| Geo Mismatch (`txn_city ≠ home_city`) | 0.67% | 16.02% | ~24x |
| High Velocity (≥5 txns/hr) | Low | High | — |
| High Value (> ₹4,000) | Low | High | — |
| New Account (< 30 days) | Low | High | — |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Model** | XGBoost (Gradient Boosted Trees) |
| **Explainability** | SHAP TreeExplainer |
| **API** | FastAPI + Uvicorn |
| **Frontend** | HTML5 / Vanilla CSS / JavaScript |
| **Data Processing** | Pandas, Scikit-learn |
| **Data** | Synthetic (50,000 transactions with realistic Indian payment-gateway fraud patterns) |

---

## 📖 Development Log

Full step-by-step execution history documented in [PROJECT_LOG.md](PROJECT_LOG.md).
