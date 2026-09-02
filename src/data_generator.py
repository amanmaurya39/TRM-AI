"""
Synthetic Transaction Data Generator
Simulates Razorpay-style payment transactions with realistic fraud patterns.
Since real Razorpay data isn't available, this creates a statistically
realistic dataset with known fraud signals so the model has something
meaningful to learn.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

N_TRANSACTIONS = 50000
FRAUD_RATE = 0.015  # ~1.5% fraud, realistic for payment gateways

MERCHANT_CATEGORIES = [
    "E-commerce", "Food Delivery", "Travel", "Utilities", "Education",
    "Gaming", "Subscription", "Electronics", "Fashion", "Crypto/Forex"
]

PAYMENT_METHODS = ["Credit Card", "Debit Card", "UPI", "Net Banking", "Wallet"]

CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Pune",
    "Kolkata", "Ahmedabad", "Jaipur", "Lucknow"
]


def generate_user_base(n_users=8000):
    """Create a pool of users with historical behavior baselines."""
    users = []
    for i in range(n_users):
        users.append({
            "user_id": f"U{i:06d}",
            "home_city": random.choice(CITIES),
            "avg_txn_amount": np.random.gamma(2, 800),  # INR baseline spend
            "avg_txn_per_day": np.random.gamma(1.5, 1.2),
            "account_age_days": np.random.randint(1, 1500),
            "preferred_method": random.choice(PAYMENT_METHODS),
        })
    return pd.DataFrame(users)


def generate_transactions(users_df, n=N_TRANSACTIONS):
    rows = []
    start_date = datetime(2026, 1, 1)

    n_fraud = int(n * FRAUD_RATE)
    n_legit = n - n_fraud

    user_ids = users_df["user_id"].tolist()

    # ---- LEGITIMATE TRANSACTIONS ----
    for _ in range(n_legit):
        user = users_df.iloc[random.randrange(len(users_df))]
        amount = max(10, np.random.normal(user["avg_txn_amount"], user["avg_txn_amount"] * 0.3))
        txn_time = start_date + timedelta(
            days=random.randint(0, 240),
            hours=random.choices(
                range(24),
                weights=[1,1,1,1,1,2,3,5,6,7,8,9,9,8,7,6,6,7,8,9,8,6,4,2]
            )[0],
            minutes=random.randint(0, 59)
        )
        rows.append({
            "user_id": user["user_id"],
            "timestamp": txn_time,
            "amount": round(amount, 2),
            "merchant_category": random.choice(MERCHANT_CATEGORIES),
            "payment_method": user["preferred_method"] if random.random() > 0.15 else random.choice(PAYMENT_METHODS),
            "txn_city": user["home_city"] if random.random() > 0.05 else random.choice(CITIES),
            "home_city": user["home_city"],
            "device_new": random.random() < 0.03,
            "account_age_days": user["account_age_days"],
            "txn_velocity_1h": max(0, int(np.random.poisson(0.3))),
            "is_fraud": 0
        })

    # ---- FRAUDULENT TRANSACTIONS (with realistic fraud signatures) ----
    fraud_patterns = ["account_takeover", "stolen_card", "velocity_abuse", "geo_mismatch", "new_device_high_value"]

    for _ in range(n_fraud):
        user = users_df.iloc[random.randrange(len(users_df))]
        pattern = random.choice(fraud_patterns)
        txn_time = start_date + timedelta(
            days=random.randint(0, 240),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )

        base = {
            "user_id": user["user_id"],
            "timestamp": txn_time,
            "merchant_category": random.choice(MERCHANT_CATEGORIES),
            "payment_method": random.choice(PAYMENT_METHODS),
            "home_city": user["home_city"],
            "account_age_days": user["account_age_days"],
            "is_fraud": 1,
            "fraud_pattern": pattern
        }

        if pattern == "account_takeover":
            base.update({
                "amount": round(user["avg_txn_amount"] * random.uniform(3, 8), 2),
                "txn_city": random.choice([c for c in CITIES if c != user["home_city"]]),
                "device_new": True,
                "txn_velocity_1h": random.randint(2, 6),
            })
        elif pattern == "stolen_card":
            base.update({
                "amount": round(random.uniform(5000, 50000), 2),
                "txn_city": random.choice(CITIES),
                "device_new": True,
                "txn_velocity_1h": random.randint(0, 2),
            })
        elif pattern == "velocity_abuse":
            base.update({
                "amount": round(random.uniform(500, 5000), 2),
                "txn_city": user["home_city"],
                "device_new": False,
                "txn_velocity_1h": random.randint(5, 15),
            })
        elif pattern == "geo_mismatch":
            base.update({
                "amount": round(user["avg_txn_amount"] * random.uniform(1, 3), 2),
                "txn_city": random.choice([c for c in CITIES if c != user["home_city"]]),
                "device_new": random.random() < 0.5,
                "txn_velocity_1h": random.randint(1, 3),
            })
        else:  # new_device_high_value
            base.update({
                "amount": round(random.uniform(10000, 80000), 2),
                "txn_city": user["home_city"],
                "device_new": True,
                "txn_velocity_1h": random.randint(0, 1),
            })

        rows.append(base)

    df = pd.DataFrame(rows)
    df["fraud_pattern"] = df.get("fraud_pattern", pd.Series(["none"] * len(df))).fillna("none")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["txn_id"] = [f"TXN{str(i).zfill(8)}" for i in range(len(df))]

    return df


def engineer_features(df):
    """Derive model-ready features from raw transaction fields."""
    df = df.copy()
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["is_night"] = df["hour_of_day"].apply(lambda h: 1 if (h < 6 or h > 22) else 0)
    df["geo_mismatch"] = (df["txn_city"] != df["home_city"]).astype(int)
    df["device_new"] = df["device_new"].astype(int)
    df["is_new_account"] = (df["account_age_days"] < 30).astype(int)
    df["high_value_txn"] = (df["amount"] > df["amount"].quantile(0.90)).astype(int)

    cat_cols = ["merchant_category", "payment_method"]
    df = pd.get_dummies(df, columns=cat_cols, prefix=cat_cols)

    return df


if __name__ == "__main__":
    print("Generating user base...")
    users = generate_user_base()

    print("Generating transactions...")
    txns = generate_transactions(users)

    print(f"Total transactions: {len(txns)}")
    print(f"Fraud transactions: {txns['is_fraud'].sum()} ({txns['is_fraud'].mean()*100:.2f}%)")

    txns.to_csv("/home/claude/razorpay-risk-manager/data/transactions_raw.csv", index=False)

    features = engineer_features(txns)
    features.to_csv("/home/claude/razorpay-risk-manager/data/transactions_features.csv", index=False)

    print("Saved: data/transactions_raw.csv and data/transactions_features.csv")
