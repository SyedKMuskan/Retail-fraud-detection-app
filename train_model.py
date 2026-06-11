"""Train the model the app uses, and save it + the exact feature columns.
Flat paths: this file, fraud_cleaned.csv, and the output all sit in ONE folder.
Run:  python train_model.py
"""
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

HERE = Path(__file__).parent

df = pd.read_csv(HERE / "fraud_cleaned.csv", parse_dates=["transaction_timestamp"])

df["amount_vs_personal_avg"] = df["transaction_amount"] / df["avg_transaction_amount_7d"]
flags = ["is_international", "previous_fraud_flag", "unusual_amount_flag",
         "multiple_transactions_short_time", "high_risk_device_flag", "velocity_flag"]
df["risk_flag_count"]    = df[flags].sum(axis=1)
df["amount_x_frequency"] = df["transaction_amount"] * df["transaction_frequency_24h"]

y = df["fraud_flag"]
# drop ids, leakage (fraud_risk), raw timestamp, and account_age_days (no signal)
X = df.drop(columns=["transaction_id", "customer_id", "transaction_timestamp",
                      "fraud_risk", "fraud_flag", "account_age_days"])
X = pd.get_dummies(X, columns=["payment_method", "device_type", "location", "merchant_category"],
                   drop_first=True)

# 80/20 stratified split — evaluate on held-out data only
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Logistic Regression: equal accuracy to RF on this dataset, more interpretable (see README)
# Pipeline scales features first — required for LR convergence and coefficient comparability
model = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=1000, random_state=42)),
])
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("=== Held-out test set (20% = 20,000 rows) ===")
print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))
print(f"ROC-AUC:  {roc_auc_score(y_test, y_prob):.4f}")
print("\nConfusion matrix (rows=actual, cols=predicted):")
cm = confusion_matrix(y_test, y_pred)
print(f"  TN={cm[0,0]:6,}  FP={cm[0,1]:6,}")
print(f"  FN={cm[1,0]:6,}  TP={cm[1,1]:6,}")

joblib.dump({"model": model, "columns": list(X_train.columns)}, HERE / "fraud_model.joblib")
print(f"\nSaved fraud_model.joblib  ({len(X_train.columns)} features, LogisticRegression)")
