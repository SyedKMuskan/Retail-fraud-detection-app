"""
FraudShield — Streamlit app
Run locally:  streamlit run app.py
"""
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="FraudShield", page_icon="🛡️", layout="centered")

# --- Black theme (works even without config.toml) ---
st.markdown("""
<style>
.stApp { background-color: #000000; color: #f5f5f5; }
section[data-testid="stSidebar"] { background-color: #0a0a0a; }
h1, h2, h3, h4, p, label, span { color: #f5f5f5 !important; }
.stNumberInput input, .stSelectbox div[data-baseweb="select"] { background-color:#141414; color:#fff; }
div[data-testid="stExpander"] { background-color:#0d0d0d; border:1px solid #222; border-radius:8px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    art = joblib.load("fraud_model.joblib")
    return art["model"], art["columns"]


model, COLUMNS = load_model()

# --- Heading (change this one line to rename the app) ---
st.title("🛡️ FraudShield")
st.write(
    "Enter a transaction's details and get an instant risk assessment. "
    "The model weighs the amount, the customer's recent activity, the device "
    "and location, and any risk flags to estimate the chance of fraud."
)

st.subheader("Transaction details")
c1, c2 = st.columns(2)
with c1:
    transaction_amount = st.number_input("Transaction amount (£)", 0.0, 5000.0, 120.0, 10.0)
    avg_transaction_amount_7d = st.number_input("Customer's 7-day avg amount (£)", 1.0, 5000.0, 100.0, 10.0)
    transaction_frequency_24h = st.slider("Transactions in last 24h", 0, 30, 5)
    failed_transaction_count_24h = st.slider("Failed attempts in last 24h", 0, 15, 1)
with c2:
    payment_method = st.selectbox("Payment method", ["Credit Card", "Debit Card", "PayPal", "Apple Pay", "Google Pay"])
    device_type = st.selectbox("Device type", ["Mobile", "Desktop", "Tablet"])
    location = st.selectbox("Location", ["UK", "USA", "Germany", "India", "Canada", "Australia"])
    merchant_category = st.selectbox("Merchant category", ["Groceries", "Electronics", "Fashion", "Travel", "Gaming", "Luxury"])

with st.expander("Risk flags (optional — tick any that apply)"):
    f1, f2, f3 = st.columns(3)
    with f1:
        is_international = st.checkbox("International")
        previous_fraud_flag = st.checkbox("Previous fraud")
    with f2:
        unusual_amount_flag = st.checkbox("Unusual amount")
        multiple_transactions_short_time = st.checkbox("Multiple txns, short time")
    with f3:
        high_risk_device_flag = st.checkbox("High-risk device")
        velocity_flag = st.checkbox("Velocity flag")


def build_features(raw: dict) -> pd.DataFrame:
    df = pd.DataFrame([raw])
    df["amount_vs_personal_avg"] = df["transaction_amount"] / df["avg_transaction_amount_7d"]
    flags = ["is_international", "previous_fraud_flag", "unusual_amount_flag",
             "multiple_transactions_short_time", "high_risk_device_flag", "velocity_flag"]
    df["risk_flag_count"] = df[flags].sum(axis=1)
    df["amount_x_frequency"] = df["transaction_amount"] * df["transaction_frequency_24h"]
    df = pd.get_dummies(df)
    df = df.reindex(columns=COLUMNS, fill_value=0)   # prevents column-mismatch crashes
    return df


if st.button("Check transaction", type="primary"):
    raw = dict(
        transaction_amount=transaction_amount,
        transaction_frequency_24h=transaction_frequency_24h,
        avg_transaction_amount_7d=avg_transaction_amount_7d,
        failed_transaction_count_24h=failed_transaction_count_24h,
        is_international=int(is_international),
        previous_fraud_flag=int(previous_fraud_flag),
        unusual_amount_flag=int(unusual_amount_flag),
        multiple_transactions_short_time=int(multiple_transactions_short_time),
        high_risk_device_flag=int(high_risk_device_flag),
        velocity_flag=int(velocity_flag),
        payment_method=payment_method,
        device_type=device_type,
        location=location,
        merchant_category=merchant_category,
    )
    prob = float(model.predict_proba(build_features(raw))[0, 1])
    st.metric("Fraud probability", f"{prob*100:.1f}%")
    if prob >= 0.5:
        st.error("⚠️ This transaction looks risky — we recommend reviewing or blocking it.")
    else:
        st.success("✅ Sit back and relax — this looks like a safe, genuine transaction.")
    st.progress(prob)
