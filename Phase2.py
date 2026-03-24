import joblib
import pandas as pd
import numpy as np
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# ── Token ──────────────────────────────────────────────
from google.colab import userdata
os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")
# (Add GROQ_API_KEY in Colab Secrets 🔑 sidebar)

# ── Load model from Phase 1 ────────────────────────────
model = joblib.load("fraud_model.pkl")
feature_names = [f"V{i}" for i in range(1, 29)] + ["scaled_amount", "scaled_time"]
feature_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

# ── LLM setup ─────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.1-8b-instant",   # LLaMA 3 8B — fast & free on Groq
    temperature=0.3,
    max_tokens=256,
)

# ── Prompt ─────────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a senior fraud analyst. Write concise, professional analyst notes."),
    ("human", """A transaction has been assessed by our ML model:
- Decision: {fraud_label}
- Fraud Probability: {fraud_probability}
- Amount: ${transaction_amount}
- Hour: {transaction_hour}:00
- Top Risk Signals: {top_features}

Write a 3-4 sentence analyst note: state the decision, explain why using the signals, and recommend a next action. No preamble.""")
])

fraud_chain = prompt | llm

# ── Pipeline ───────────────────────────────────────────
def analyze_transaction(transaction_df, original_amount, original_hour):
    prob = model.predict_proba(transaction_df)[:, 1][0]
    label = "FLAGGED AS FRAUD" if prob >= 0.5 else "CLEARED AS LEGITIMATE"

    top_feats = feature_importance.head(3)["feature"].tolist()
    top_feature_values = {f: round(float(transaction_df[f].values[0]), 4) for f in top_feats}
    top_features_str = ", ".join([f"{k}: {v}" for k, v in top_feature_values.items()])

    print("Generating analyst note via Groq...")
    response = fraud_chain.invoke({
        "fraud_label": label,
        "fraud_probability": f"{round(prob * 100, 1)}%",
        "transaction_amount": original_amount,
        "transaction_hour": original_hour,
        "top_features": top_features_str
    })

    return {
        "label": label,
        "fraud_probability": round(prob, 4),
        "top_risk_features": top_feature_values,
        "analyst_note": response.content.strip()
    }

# ── Test ───────────────────────────────────────────────
sample_data = pd.DataFrame([np.random.randn(30)], columns=feature_names)

result = analyze_transaction(sample_data, original_amount=2450.00, original_hour=2)

print("=" * 60)
print(f"Decision     : {result['label']}")
print(f"Fraud Score  : {result['fraud_probability']}")
print(f"Risk Features: {result['top_risk_features']}")
print("\n--- Analyst Note ---")
print(result["analyst_note"])
print("=" * 60)