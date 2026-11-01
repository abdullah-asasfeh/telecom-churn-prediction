"""Train, compare and evaluate churn models.

Usage:  python train.py
Outputs: models/churn_model.joblib, reports/metrics.json, images/*.svg
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from churn.pipeline import RANDOM_STATE, ROOT, add_features, build_models, load_raw, split_xy

IMG = ROOT / "images"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
for d in (IMG, MODELS, REPORTS):
    d.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "svg.fonttype": "none", "figure.dpi": 100,
})
ORANGE, GREY, DARK = "#FF7900", "#9E9E9E", "#333333"


def best_f1_threshold(y, proba):
    p, r, t = precision_recall_curve(y, proba)
    f1 = 2 * p * r / np.clip(p + r, 1e-9, None)
    i = int(np.nanargmax(f1[:-1]))
    return float(t[i])


def main():
    df = add_features(load_raw())
    X, y = split_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # 1. Compare models with 5-fold cross-validation on the training set only
    cv_results, oof = {}, {}
    for name, model in build_models().items():
        proba = cross_val_predict(model, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        oof[name] = proba
        cv_results[name] = {
            "roc_auc": roc_auc_score(y_tr, proba),
            "pr_auc": average_precision_score(y_tr, proba),
        }
        print(f"{name:22s} CV ROC-AUC {cv_results[name]['roc_auc']:.3f}  PR-AUC {cv_results[name]['pr_auc']:.3f}")

    best_name = max(cv_results, key=lambda n: cv_results[n]["roc_auc"])
    # decision threshold chosen on out-of-fold predictions, never on the test set
    threshold = best_f1_threshold(y_tr, oof[best_name])
    print(f"Selected: {best_name}  (threshold {threshold:.2f})")

    # 2. Fit every model on the full training set and evaluate on the held-out test set
    fitted, test_proba = {}, {}
    for name, model in build_models().items():
        fitted[name] = model.fit(X_tr, y_tr)
        test_proba[name] = model.predict_proba(X_te)[:, 1]

    model, proba = fitted[best_name], test_proba[best_name]
    pred = (proba >= threshold).astype(int)
    base_rate = float(y_te.mean())
    top = np.argsort(-proba)[: int(0.2 * len(proba))]
    top20_rate = float(y_te.iloc[top].mean())

    metrics = {
        "dataset": {"customers": int(len(df)), "churn_rate": float(y.mean()),
                    "train_size": int(len(X_tr)), "test_size": int(len(X_te))},
        "cv": cv_results,
        "test": {n: {"roc_auc": roc_auc_score(y_te, p), "pr_auc": average_precision_score(y_te, p)}
                 for n, p in test_proba.items()},
        "selected_model": best_name,
        "threshold": threshold,
        "selected_test": {
            "roc_auc": roc_auc_score(y_te, proba),
            "pr_auc": average_precision_score(y_te, proba),
            "precision": precision_score(y_te, pred),
            "recall": recall_score(y_te, pred),
            "f1": f1_score(y_te, pred),
            "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
            "top20_churn_rate": top20_rate,
            "top20_share_of_churners": float(y_te.iloc[top].sum() / y_te.sum()),
            "lift_top20": top20_rate / base_rate,
        },
    }

    # 3. Permutation importance on the test set (on the original columns)
    pi = permutation_importance(model, X_te, y_te, scoring="roc_auc", n_repeats=10,
                                random_state=RANDOM_STATE, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=X_te.columns).sort_values(ascending=False)
    metrics["feature_importance"] = imp.round(4).to_dict()

    joblib.dump({"model": model, "threshold": threshold, "model_name": best_name}, MODELS / "churn_model.joblib")
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---------- charts ----------
    # ROC curves
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p in test_proba.items():
        fpr, tpr, _ = roc_curve(y_te, p)
        ax.plot(fpr, tpr, lw=2.4 if name == best_name else 1.4,
                color=ORANGE if name == best_name else (GREY if "Forest" in name else DARK),
                label=f"{name} (AUC {roc_auc_score(y_te, p):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="#CCCCCC", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve — held-out test set")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(IMG / "roc_curves.svg"); plt.close(fig)

    # Feature importance (top 10)
    top_imp = imp.head(10)[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.barh(top_imp.index, top_imp.values, color=ORANGE)
    ax.set(xlabel="Drop in ROC-AUC when the feature is shuffled", title="What drives churn (permutation importance)")
    fig.tight_layout(); fig.savefig(IMG / "feature_importance.svg"); plt.close(fig)

    # Confusion matrix
    cm = np.array(metrics["selected_test"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Oranges")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=14,
                color="white" if v > cm.max() / 2 else DARK)
    ax.set_xticks([0, 1], ["Stays", "Churns"]); ax.set_yticks([0, 1], ["Stays", "Churns"])
    ax.set(xlabel="Predicted", ylabel="Actual", title=f"Confusion matrix (threshold {threshold:.2f})")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(); fig.savefig(IMG / "confusion_matrix.svg"); plt.close(fig)

    # Churn rate by contract and tenure (EDA)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    by_contract = df.groupby("Contract")["Churn"].mean().sort_values(ascending=False) * 100
    axes[0].bar(by_contract.index, by_contract.values, color=[ORANGE, GREY, GREY])
    axes[0].set(ylabel="Churn rate (%)", title="Churn rate by contract type")
    order = ["0-6m", "7-12m", "13-24m", "25-48m", "49-72m"]
    by_ten = df.groupby("TenureGroup")["Churn"].mean().reindex(order) * 100
    axes[1].bar(by_ten.index, by_ten.values, color=[ORANGE] + [GREY] * 4)
    axes[1].set(xlabel="Tenure", title="Churn rate by customer tenure")
    for ax_ in axes:
        for b in ax_.patches:
            ax_.annotate(f"{b.get_height():.0f}%", (b.get_x() + b.get_width() / 2, b.get_height()),
                         ha="center", va="bottom", fontsize=10)
    fig.tight_layout(); fig.savefig(IMG / "churn_drivers_eda.svg"); plt.close(fig)

    print(json.dumps(metrics["selected_test"], indent=2))


if __name__ == "__main__":
    main()
