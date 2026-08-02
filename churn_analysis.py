"""
churn_analysis.py
Cleans the client data, explores it, trains an interpretable churn model,
and scores every client -> outputs clients_scored.csv for Power BI.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

# 1) LOAD
df = pd.read_csv("clients_raw.csv")
print(f"Loaded {len(df)} clients\n")

# 2) CLEAN
df["plan_type"] = df["plan_type"].str.capitalize()          # fix 'pro' -> 'Pro'
df["nps_score"] = df["nps_score"].fillna(df["nps_score"].median())
df["avg_logins_per_week"] = df["avg_logins_per_week"].fillna(
    df["avg_logins_per_week"].median()
)
print("Missing values after cleaning:\n", df.isna().sum()[df.isna().sum() > 0], "\n")

# 3) QUICK EDA (numbers you'll quote in your write-up)
print("Churn rate by plan:")
print(df.groupby("plan_type")["churned"].mean().round(3), "\n")
print("Avg late payments — churned vs retained:")
print(df.groupby("churned")["late_payments_count"].mean().round(2), "\n")

# 4) MODEL
features = [
    "contract_length_months", "monthly_revenue", "onboarding_days",
    "support_tickets_90d", "avg_logins_per_week", "days_since_last_login",
    "late_payments_count", "nps_score",
]
X = df[features]
y = df["churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
scaler = StandardScaler().fit(X_train)
model = LogisticRegression(max_iter=1000).fit(scaler.transform(X_train), y_train)

pred_proba = model.predict_proba(scaler.transform(X_test))[:, 1]
pred = (pred_proba >= 0.5).astype(int)

print("=" * 40)
print("MODEL PERFORMANCE")
print("ROC AUC:", round(roc_auc_score(y_test, pred_proba), 3))
print("\nConfusion matrix:\n", confusion_matrix(y_test, pred))
print("\n", classification_report(y_test, pred))

# 5) FEATURE IMPORTANCE (drivers of churn — great for the dashboard/story)
importance = pd.DataFrame({
    "feature": features,
    "coefficient": model.coef_[0],
}).sort_values("coefficient", key=abs, ascending=False)
print("Top churn drivers:\n", importance, "\n")

# 6) SCORE EVERY CLIENT -> feeds Power BI
df["churn_probability"] = model.predict_proba(scaler.transform(X))[:, 1].round(3)
df["risk_tier"] = pd.cut(
    df["churn_probability"],
    bins=[-0.01, 0.33, 0.66, 1.0],
    labels=["Low", "Medium", "High"],
)
df["revenue_at_risk"] = (df["monthly_revenue"] * df["churn_probability"]).round(2)

df.to_csv("clients_scored.csv", index=False)
importance.to_csv("churn_drivers.csv", index=False)
print("Saved clients_scored.csv and churn_drivers.csv")
print("High-risk clients:", (df.risk_tier == "High").sum())
print("Total monthly revenue at risk: $", round(df.revenue_at_risk.sum(), 2))
