"""Load the trained model, score customers and explain individual predictions."""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from churn.pipeline import (
    CATEGORICAL, NUMERIC, RANDOM_STATE, ROOT, add_features, build_models, load_raw, split_xy,
)

MODEL_PATH = ROOT / "models" / "churn_model.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"

# Readable names for the explanation panel
LABELS = {
    "tenure": "Months with the company", "TenureGroup": "Tenure band", "Contract": "Contract type",
    "InternetService": "Internet service", "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges", "AvgMonthlySpend": "Average monthly spend",
    "ChargeIncrease": "Recent price increase", "NumServices": "Number of add-on services",
    "PaymentMethod": "Payment method", "PaperlessBilling": "Paperless billing",
    "OnlineSecurity": "Online security", "TechSupport": "Tech support",
    "OnlineBackup": "Online backup", "DeviceProtection": "Device protection",
    "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming movies",
    "MultipleLines": "Multiple lines", "PhoneService": "Phone service",
    "SeniorCitizen": "Senior citizen", "Partner": "Has partner", "Dependents": "Has dependents",
    "gender": "Gender",
}

# Retention actions triggered by the strongest risk factors
ACTIONS = {
    "Contract": "Offer a discounted 1- or 2-year contract (month-to-month customers churn ~4x more).",
    "tenure": "Enrol in an early-life onboarding programme: first 6 months are the highest-risk period.",
    "TenureGroup": "Enrol in an early-life onboarding programme: first 6 months are the highest-risk period.",
    "InternetService": "Check fibre service quality and offer a speed or price review.",
    "MonthlyCharges": "Review the plan: suggest a better-value bundle for this usage.",
    "AvgMonthlySpend": "Review the plan: suggest a better-value bundle for this usage.",
    "ChargeIncrease": "Recent bill increase: proactively explain it or apply a loyalty credit.",
    "OnlineSecurity": "Offer a free trial of Online Security.",
    "TechSupport": "Offer a free trial of Tech Support.",
    "PaymentMethod": "Encourage automatic payment (bank transfer / credit card).",
    "PaperlessBilling": "Send a clear monthly bill summary to reduce billing confusion.",
}


def load_model():
    """Return (pipeline, threshold, model_name). Trains a model if none has been saved yet."""
    if MODEL_PATH.exists():
        b = joblib.load(MODEL_PATH)
        return b["model"], b["threshold"], b["model_name"]
    meta = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    name = meta.get("selected_model", "Logistic Regression")
    threshold = meta.get("threshold", 0.5)
    X, y = split_xy(add_features(load_raw()))
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    model = build_models()[name].fit(X_tr, y_tr)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "threshold": threshold, "model_name": name}, MODEL_PATH)
    return model, threshold, name


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Raw customer rows (original dataset columns) -> model input."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    return add_features(df)[CATEGORICAL + NUMERIC]


def score(model, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


def risk_band(p: float, threshold: float) -> str:
    if p >= max(threshold, 0.7):
        return "High"
    if p >= threshold:
        return "Elevated"
    if p >= threshold * 0.6:
        return "Medium"
    return "Low"


def explain(model, row: pd.DataFrame, reference: pd.DataFrame, top: int = 6) -> pd.DataFrame:
    """Per-feature contribution to the churn log-odds for one customer.

    For a linear model, contribution = coefficient x (customer value - average customer value),
    summed over the one-hot columns of each original feature. Positive = pushes towards churn.
    For non-linear models it falls back to a what-if test: replace each feature with the
    reference customer's typical value and measure the change in predicted probability.
    """
    X_row, X_ref = prepare(row), prepare(reference)
    clf, prep = model.named_steps["clf"], model.named_steps["prep"]

    if hasattr(clf, "coef_"):
        diff = prep.transform(X_row)[0] - prep.transform(X_ref).mean(axis=0)
        contrib = clf.coef_[0] * diff
        names = prep.get_feature_names_out()
        totals = {}
        for n, c in zip(names, contrib):
            kind, rest = n.split("__", 1)
            feat = rest if kind == "num" else next(f for f in CATEGORICAL if rest.startswith(f + "_"))
            totals[feat] = totals.get(feat, 0.0) + c
        s = pd.Series(totals)
    else:
        base = model.predict_proba(X_row)[:, 1][0]
        typical = X_ref.mode().iloc[0]
        s = pd.Series({
            f: base - model.predict_proba(X_row.assign(**{f: typical[f]}))[:, 1][0]
            for f in X_row.columns
        })

    out = s.reindex(s.abs().sort_values(ascending=False).index).head(top).to_frame("impact")
    out["factor"] = [LABELS.get(f, f) for f in out.index]
    out["value"] = [X_row.iloc[0][f] for f in out.index]
    out["feature"] = out.index
    return out.reset_index(drop=True)


def recommend(explanation: pd.DataFrame, k: int = 3) -> list:
    actions = []
    for _, r in explanation[explanation["impact"] > 0].iterrows():
        a = ACTIONS.get(r["feature"])
        if a and a not in actions:
            actions.append(a)
    return actions[:k]
