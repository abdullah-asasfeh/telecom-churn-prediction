"""Data loading, feature engineering and model definitions for telecom churn prediction."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "telco_churn.csv"
RANDOM_STATE = 42

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "TenureGroup",
]
NUMERIC = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "NumServices", "AvgMonthlySpend", "ChargeIncrease",
]


DATA_URL = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
            "master/data/Telco-Customer-Churn.csv")


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        # first run: download the public IBM Telco dataset and cache it locally
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(DATA_URL).to_csv(path, index=False)
    df = pd.read_csv(path)
    # TotalCharges is blank for brand-new customers (tenure 0): they have paid nothing yet.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["NumServices"] = (df[SERVICE_COLS] == "Yes").sum(axis=1)
    df["AvgMonthlySpend"] = np.where(df["tenure"] > 0, df["TotalCharges"] / df["tenure"].clip(lower=1), df["MonthlyCharges"])
    # >0 means the customer now pays more per month than their historical average
    df["ChargeIncrease"] = df["MonthlyCharges"] - df["AvgMonthlySpend"]
    df["TenureGroup"] = pd.cut(
        df["tenure"], bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6m", "7-12m", "13-24m", "25-48m", "49-72m"],
    ).astype(str)
    return df


def split_xy(df: pd.DataFrame):
    return df[CATEGORICAL + NUMERIC], df["Churn"]


def _preprocessor(scale: bool) -> ColumnTransformer:
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("num", num, NUMERIC),
    ])


def build_models() -> dict:
    """Candidate models. class_weight='balanced' compensates for the ~26% churn rate."""
    return {
        "Logistic Regression": Pipeline([
            ("prep", _preprocessor(scale=True)),
            ("clf", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")),
        ]),
        "Random Forest": Pipeline([
            ("prep", _preprocessor(scale=False)),
            ("clf", RandomForestClassifier(
                n_estimators=400, min_samples_leaf=5, max_features="sqrt",
                class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)),
        ]),
        "Gradient Boosting": Pipeline([
            ("prep", _preprocessor(scale=False)),
            ("clf", HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                min_samples_leaf=40, l2_regularization=1.0,
                class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
    }
