# ============================================================
# AI-Powered Fraud Detection System — Phase 3: Dashboard
# Run with: streamlit run dashboard.py
# ============================================================

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide"
)

# -------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("fraud_model.pkl")
    feature_names = [f"V{i}" for i in range(1, 29)] + ["scaled_amount", "scaled_time"]
    feature_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    return model, feature_names, feature_importance

model, feature_names, feature_importance = load_model()

# -------------------------------------------------------
# LLM SETUP
# -------------------------------------------------------
@st.cache_resource
def load_llm(api_key):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=256,
        api_key=api_key
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a senior fraud analyst. Write concise, professional analyst notes."),
        ("human", """A transaction has been assessed by our ML model:
- Decision: {fraud_label}
- Fraud Probability: {fraud_probability}
- Amount: ${transaction_amount}
- Hour: {transaction_hour}:00
- Top Risk Signals: {top_features}

Write a 3-4 sentence analyst note: state the decision, explain why using the signals,
and recommend a next action. No preamble.""")
    ])
    return prompt | llm

# -------------------------------------------------------
# PREDICTION FUNCTION
# -------------------------------------------------------
# def analyze_transaction(transaction_df, original_amount, original_hour, chain):
#     prob = model.predict_proba(transaction_df)[:, 1][0]
#     label = "FLAGGED AS FRAUD" if prob >= 0.5 else "CLEARED AS LEGITIMATE"

#     top_feats = feature_importance.head(3)["feature"].tolist()
#     top_feature_values = {f: round(float(transaction_df[f].values[0]), 4) for f in top_feats}
#     top_features_str = ", ".join([f"{k}: {v}" for k, v in top_feature_values.items()])

#     response = chain.invoke({
#         "fraud_label": label,
#         "fraud_probability": f"{round(prob * 100, 1)}%",
#         "transaction_amount": original_amount,
#         "transaction_hour": original_hour,
#         "top_features": top_features_str
#     })

#     return {
#         "label": label,
#         "fraud_probability": round(prob * 100, 1),
#         "top_risk_features": top_feature_values,
#         "analyst_note": response.content.strip()
#     }
from sklearn.preprocessing import StandardScaler

def analyze_transaction(transaction_df, original_amount, original_hour, chain):
    # If loaded from CSV, it will have raw Amount/Time — scale them
    if "Amount" in transaction_df.columns:
        scaler = StandardScaler()
        transaction_df = transaction_df.copy()
        transaction_df["scaled_amount"] = scaler.fit_transform(transaction_df[["Amount"]])
        transaction_df["scaled_time"] = scaler.fit_transform(transaction_df[["Time"]])
        transaction_df.drop(["Amount", "Time"], axis=1, inplace=True)

    # Reorder columns to match training
    transaction_df = transaction_df[feature_names]

    prob = model.predict_proba(transaction_df)[:, 1][0]
    label = "FLAGGED AS FRAUD" if prob >= 0.5 else "CLEARED AS LEGITIMATE"

    top_feats = feature_importance.head(3)["feature"].tolist()
    top_feature_values = {f: round(float(transaction_df[f].values[0]), 4) for f in top_feats}
    top_features_str = ", ".join([f"{k}: {v}" for k, v in top_feature_values.items()])

    print("Generating analyst note via Groq...")
    response = chain.invoke({
        "fraud_label": label,
        "fraud_probability": f"{round(prob * 100, 1)}%",
        "transaction_amount": original_amount,
        "transaction_hour": original_hour,
        "top_features": top_features_str
    })

    return {
        "label": label,
        "fraud_probability": round(prob * 100, 1),
        "top_risk_features": top_feature_values,
        "analyst_note": response.content.strip()
    }
# -------------------------------------------------------
# UI — HEADER
# -------------------------------------------------------
st.title("🔍 AI-Powered Fraud Detection System")
st.caption("ML model + LLM explanation layer | Built with Random Forest & LLaMA 3 via Groq")
st.divider()

# -------------------------------------------------------
# SIDEBAR — CONFIG
# -------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_..."
    )
    st.caption("Get a free key at console.groq.com")
    st.divider()

    st.header("🧪 Transaction Input")
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=2450.0, step=50.0)
    hour = st.slider("Hour of Transaction", 0, 23, 2,
                     help="0 = midnight, 12 = noon, 23 = 11 PM")
    st.caption(f"Selected time: {'🌙 Night' if hour < 6 or hour > 22 else '☀️ Day'} ({hour}:00)")

    use_random = st.checkbox("Use random transaction features", value=True,
                             help="Simulates V1-V28 PCA features. Uncheck to load from test set.")
    analyze_btn = st.button("🔎 Analyze Transaction", type="primary", use_container_width=True)

# -------------------------------------------------------
# MAIN — RESULTS
# -------------------------------------------------------
if analyze_btn:
    if not groq_api_key:
        st.error("Please enter your Groq API key in the sidebar.")
    else:
        # Build transaction
        if use_random:
            sample_data = pd.DataFrame([np.random.randn(30)], columns=feature_names)
        else:
            # Load from saved test set if available
            try:
                X_test = pd.read_csv("fraud_sample.csv")
                sample_data = X_test.sample(1)
            except FileNotFoundError:
                st.warning("X_test.csv not found — using random features instead.")
                sample_data = pd.DataFrame([np.random.randn(30)], columns=feature_names)

        chain = load_llm(groq_api_key)

        with st.spinner("Analyzing transaction..."):
            result = analyze_transaction(sample_data, amount, hour, chain)

        # ── Result Banner ──
        if result["label"] == "FLAGGED AS FRAUD":
            st.error(f"🚨 {result['label']}")
        else:
            st.success(f"✅ {result['label']}")

        # ── Metrics Row ──
        col1, col2, col3 = st.columns(3)
        col1.metric("Fraud Probability", f"{result['fraud_probability']}%")
        col2.metric("Transaction Amount", f"${amount:,.2f}")
        col3.metric("Transaction Hour", f"{hour}:00")

        st.divider()

        # ── Two Column Layout ──
        left, right = st.columns(2)

        with left:
            st.subheader("📊 Top Risk Signals")
            risk_df = pd.DataFrame(
                result["top_risk_features"].items(),
                columns=["Feature", "Value"]
            )
            st.dataframe(risk_df, use_container_width=True, hide_index=True)

            st.subheader("📈 Feature Importance (Top 10)")
            chart_data = feature_importance.head(10).set_index("feature")
            st.bar_chart(chart_data["importance"])

        with right:
            st.subheader("🧠 AI Analyst Note")
            st.info(result["analyst_note"])

            st.subheader("🔢 Fraud Score Gauge")
            score = result["fraud_probability"]
            color = "red" if score >= 50 else "green"
            st.markdown(f"""
                <div style='text-align:center; padding: 20px;'>
                    <h1 style='color:{color}; font-size: 64px;'>{score}%</h1>
                    <p style='color:gray;'>Fraud Probability Score</p>
                    <div style='background:#eee; border-radius:10px; height:20px;'>
                        <div style='background:{color}; width:{score}%;
                             border-radius:10px; height:20px;'></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.caption("Model: Random Forest trained on Kaggle Credit Card Fraud Dataset | "
                   "LLM: LLaMA 3.1 8B via Groq | Built by Sukriti")

else:
    # ── Empty State ──
    st.info("👈 Configure a transaction in the sidebar and click **Analyze Transaction** to begin.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Model", "Random Forest")
    col2.metric("ROC-AUC", "0.97+")
    col3.metric("Fraud Recall", "94%+")