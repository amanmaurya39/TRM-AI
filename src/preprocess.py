"""
Data Preprocessing & Feature Engineering Module for Razorpay Risk Manager
Transforms raw transaction data into model-ready features, performs stratified train-test splitting,
and fits a reusable ColumnTransformer pipeline (StandardScaler + OneHotEncoder).
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "transactions_raw.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Metadata/leakage columns to drop before modeling
DROP_COLS = ["txn_id", "user_id", "timestamp", "txn_city", "home_city", "fraud_pattern", "is_fraud"]

def create_features(df):
    """Derives domain-specific fraud risk indicators from raw fields."""
    df = df.copy()
    
    # Parse timestamp if string
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["is_night"] = df["hour_of_day"].apply(lambda h: 1 if (h < 6 or h > 22) else 0)
    df["geo_mismatch"] = (df["txn_city"] != df["home_city"]).astype(int)
    df["device_new"] = df["device_new"].astype(int)
    df["is_new_account"] = (df["account_age_days"] < 30).astype(int)
    
    # High value transaction threshold (> 90th percentile, ~4,000 INR)
    df["high_value_txn"] = (df["amount"] > 4000).astype(int)
    
    return df

def build_preprocessing_pipeline():
    print("=" * 60)
    print(" STEP 2: DATA PREPROCESSING & PIPELINE CONSTRUCTION")
    print("=" * 60)

    # 1. Load Raw Data
    df = pd.read_csv(DATA_PATH)
    print(f"\n[1] Loaded Raw Dataset: {len(df):,} records")

    # 2. Engineer Features
    df_featured = create_features(df)
    print(" -> Engineered Features: hour_of_day, is_night, geo_mismatch, is_new_account, high_value_txn")

    # 3. Separate Features (X) and Target (y)
    y = df_featured["is_fraud"].values
    X_raw = df_featured.drop(columns=[c for c in DROP_COLS if c in df_featured.columns])

    num_cols = ["amount", "account_age_days", "txn_velocity_1h", "hour_of_day"]
    bin_cols = ["is_night", "geo_mismatch", "device_new", "is_new_account", "high_value_txn"]
    cat_cols = ["merchant_category", "payment_method"]

    print(f" -> Numerical Columns: {num_cols}")
    print(f" -> Binary Columns: {bin_cols}")
    print(f" -> Categorical Columns: {cat_cols}")

    # 4. Define ColumnTransformer for scaling & encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("bin", "passthrough", bin_cols),
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_cols)
        ]
    )

    # 5. Stratified Train-Test Split (80% Train, 20% Test)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.20, random_state=42, stratify=y
    )

    # Fit preprocessor on train set and transform both
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    # Reconstruct feature names after OneHotEncoding
    cat_encoder = preprocessor.named_transformers_["cat"]
    encoded_cat_names = list(cat_encoder.get_feature_names_out(cat_cols))
    feature_names = num_cols + bin_cols + encoded_cat_names

    print(f"\n[2] Preprocessing Completed:")
    print(f" -> X_train Shape: {X_train.shape} (Fraud samples: {sum(y_train)})")
    print(f" -> X_test Shape : {X_test.shape} (Fraud samples: {sum(y_test)})")
    print(f" -> Total Transformed Features: {len(feature_names)}")

    # 6. Save Pipeline & Processed Arrays
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    pipeline_data = {
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "num_cols": num_cols,
        "bin_cols": bin_cols,
        "cat_cols": cat_cols,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw
    }

    pipeline_path = os.path.join(MODELS_DIR, "preprocessed_data.pkl")
    with open(pipeline_path, "wb") as f:
        pickle.dump(pipeline_data, f)

    print(f"\n[SUCCESS] Preprocessing pipeline & data saved to: {pipeline_path}")
    return pipeline_data

if __name__ == "__main__":
    build_preprocessing_pipeline()
