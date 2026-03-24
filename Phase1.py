# ============================================================
# AI-Powered Fraud Detection System — Phase 1: ML Model
# Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------------
# STEP 1: Load Data
# -------------------------------------------------------
# Download from Kaggle and place creditcard.csv in same folder
df = pd.read_csv("/content/creditcard.csv")

print(f"Dataset shape: {df.shape}")
print(f"\nClass distribution:\n{df['Class'].value_counts()}")
print(f"\nFraud %: {round(df['Class'].mean() * 100, 3)}%")

# -------------------------------------------------------
# STEP 2: Preprocessing
# -------------------------------------------------------
# 'Amount' and 'Time' are the only non-PCA features — scale them
scaler = StandardScaler()
df["scaled_amount"] = scaler.fit_transform(df[["Amount"]])
df["scaled_time"] = scaler.fit_transform(df[["Time"]])

# Drop original unscaled columns
df.drop(["Amount", "Time"], axis=1, inplace=True)

# Features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# -------------------------------------------------------
# STEP 3: Train-Test Split
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# -------------------------------------------------------
# STEP 4: Handle Class Imbalance with SMOTE
# Install: pip install imbalanced-learn
# -------------------------------------------------------
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"\nAfter SMOTE — Class distribution:\n{pd.Series(y_train_resampled).value_counts()}")

# -------------------------------------------------------
# STEP 5: Train Random Forest Classifier
# -------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight="balanced",  # extra safety for imbalance
    random_state=42,
    n_jobs=-1
)

print("\nTraining model... (may take 2-3 mins)")
model.fit(X_train_resampled, y_train_resampled)

# -------------------------------------------------------
# STEP 6: Evaluate Model
# -------------------------------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

print(f"\nROC-AUC Score: {round(roc_auc_score(y_test, y_prob), 4)}")

# -------------------------------------------------------
# STEP 7: Feature Importance (good for explaining model)
# -------------------------------------------------------
feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\n=== Top 10 Most Important Features ===")
print(feature_importance.head(10))

# -------------------------------------------------------
# STEP 8: Save Model & Scaler for Phase 2 (LLM layer)
# -------------------------------------------------------
joblib.dump(model, "fraud_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\nModel saved as fraud_model.pkl")
print("Scaler saved as scaler.pkl")

# -------------------------------------------------------
# STEP 9: Sample Prediction Function (used in Phase 2)
# -------------------------------------------------------
def predict_transaction(transaction_df):
    """
    Input: a single-row DataFrame with same columns as training data
    Output: dict with fraud label, probability, and top risk features
    """
    prob = model.predict_proba(transaction_df)[:, 1][0]
    label = "FRAUD" if prob >= 0.5 else "LEGITIMATE"

    # Get top 3 features contributing to this transaction's risk
    top_features = feature_importance.head(3)["feature"].tolist()
    feature_values = {f: round(float(transaction_df[f].values[0]), 4) for f in top_features}

    return {
        "label": label,
        "fraud_probability": round(prob, 4),
        "top_risk_features": feature_values
    }

# Test on a sample transaction
sample = X_test.iloc[[0]]
result = predict_transaction(sample)
print(f"\nSample prediction: {result}")

# -------------------------------------------------------
# NEXT: Phase 2 — Pass result into LangChain LLM
# to generate a human-readable fraud explanation
# -------------------------------------------------------