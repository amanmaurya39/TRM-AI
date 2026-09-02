"""
Exploratory Data Analysis (EDA) Script for Razorpay Risk Manager Dataset
Performs comprehensive data inspection, statistical summary, missing value analysis,
and fraud rate distributions across key transaction attributes.
"""

import os
import json
import pandas as pd
import numpy as np

# Set project relative paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "transactions_raw.csv")

def perform_eda():
    print("=" * 60)
    print(" STEP 1: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please generate it first.")

    df = pd.read_csv(DATA_PATH)
    
    # 1. Basic Dimensions & Data Types
    n_rows, n_cols = df.shape
    print(f"\n[1] Dataset Dimensions: {n_rows:,} rows | {n_cols} columns")
    print("\nData Types:")
    for col, dtype in df.dtypes.items():
        print(f" - {col:20s}: {dtype}")
        
    # 2. Missing Values Check
    missing = df.isnull().sum()
    print("\n[2] Missing Values Summary:")
    if missing.sum() == 0:
        print(" -> No missing values found in the dataset.")
    else:
        for col, cnt in missing.items():
            if cnt > 0:
                print(f" -> {col}: {cnt} missing values ({cnt/len(df)*100:.2f}%)")

    # 3. Target Distribution (Class Imbalance)
    fraud_counts = df['is_fraud'].value_counts()
    fraud_pct = df['is_fraud'].value_counts(normalize=True) * 100
    
    print("\n[3] Target Variable Distribution (is_fraud):")
    print(f" -> Legit (0) : {fraud_counts.get(0, 0):,} ({fraud_pct.get(0, 0):.2f}%)")
    print(f" -> Fraud (1) : {fraud_counts.get(1, 0):,} ({fraud_pct.get(1, 0):.2f}%)")
    
    # 4. Numerical Features Summary
    num_cols = ['amount', 'account_age_days', 'txn_velocity_1h']
    print("\n[4] Numerical Features Summary (Describe):")
    print(df[num_cols].describe().round(2).to_string())

    # 5. Fraud Rate breakdown by Categorical / Binary attributes
    print("\n[5] Fraud Rate by Key Features:")
    
    # By Device Type
    device_fraud = df.groupby('device_new')['is_fraud'].agg(['count', 'mean'])
    device_fraud['mean'] = (device_fraud['mean'] * 100).round(2)
    print("\n - By New Device (device_new):")
    print(device_fraud)
    
    # By Geo Mismatch
    df['geo_mismatch'] = (df['txn_city'] != df['home_city']).astype(int)
    geo_fraud = df.groupby('geo_mismatch')['is_fraud'].agg(['count', 'mean'])
    geo_fraud['mean'] = (geo_fraud['mean'] * 100).round(2)
    print("\n - By Geo Mismatch (txn_city != home_city):")
    print(geo_fraud)

    # By Payment Method
    pm_fraud = df.groupby('payment_method')['is_fraud'].agg(['count', 'mean']).sort_values(by='mean', ascending=False)
    pm_fraud['mean'] = (pm_fraud['mean'] * 100).round(2)
    print("\n - By Payment Method:")
    print(pm_fraud)
    
    # By Merchant Category
    mc_fraud = df.groupby('merchant_category')['is_fraud'].agg(['count', 'mean']).sort_values(by='mean', ascending=False)
    mc_fraud['mean'] = (mc_fraud['mean'] * 100).round(2)
    print("\n - By Merchant Category:")
    print(mc_fraud)

    # 6. Save EDA Summary Artifact
    eda_summary = {
        "dataset_rows": int(n_rows),
        "dataset_cols": int(n_cols),
        "missing_values": int(missing.sum()),
        "legit_count": int(fraud_counts.get(0, 0)),
        "fraud_count": int(fraud_counts.get(1, 0)),
        "fraud_rate_pct": float(round(fraud_pct.get(1, 0), 2)),
        "amount_mean": float(round(df['amount'].mean(), 2)),
        "amount_max": float(round(df['amount'].max(), 2)),
        "avg_account_age_days": float(round(df['account_age_days'].mean(), 2)),
        "fraud_rate_by_new_device": device_fraud['mean'].to_dict(),
        "fraud_rate_by_geo_mismatch": geo_fraud['mean'].to_dict()
    }
    
    summary_path = os.path.join(BASE_DIR, "models", "eda_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(eda_summary, f, indent=2)
        
    print(f"\n[SUCCESS] EDA Completed. Summary report saved to: {summary_path}")
    return eda_summary

if __name__ == "__main__":
    perform_eda()
