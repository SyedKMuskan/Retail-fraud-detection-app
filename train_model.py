"""Train the model the app uses, and save it + the exact feature columns.
Flat paths: this file, fraud_cleaned.csv, and the output all sit in ONE folder.
Run:  python train_model.py
"""
import pandas as pd, numpy as np, joblib
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("fraud_cleaned.csv", parse_dates=["transaction_timestamp"])

df["amount_vs_personal_avg"] = df["transaction_amount"] / df["avg_transaction_amount_7d"]
flags = ["is_international", "previous_fraud_flag", "unusual_amount_flag",
         "multiple_transactions_short_time", "high_risk_device_flag", "velocity_flag"]
df["risk_flag_count"]    = df[flags].sum(axis=1)
df["amount_x_frequency"] = df["transaction_amount"] * df["transaction_frequency_24h"]

y = df["fraud_flag"]
# drop ids, leakage (fraud_risk), raw timestamp, AND account_age_days (no signal)
X = df.drop(columns=["transaction_id", "customer_id", "transaction_timestamp",
                     "fraud_risk", "fraud_flag", "account_age_days"])
X = pd.get_dummies(X, columns=["payment_method", "device_type", "location", "merchant_category"],
                   drop_first=True)

model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1).fit(X, y)
joblib.dump({"model": model, "columns": list(X.columns)}, "fraud_model.joblib")
print("Saved fraud_model.joblib with", len(X.columns), "features (account_age_days removed)")
