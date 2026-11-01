"""Telecom Customer Churn Predictor — Streamlit app.

Run:  streamlit run app.py
"""
import json

import pandas as pd
import streamlit as st

from churn.pipeline import ROOT, load_raw
from churn.predict import METRICS_PATH, explain, load_model, recommend, risk_band, score

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📡", layout="wide")

ORANGE = "#FF7900"
st.markdown(f"""
<style>
.hero {{ padding: 1.3rem 1.6rem; border-radius: 14px; margin-bottom: 1rem;
         background: linear-gradient(120deg, #111 0%, #333 100%); border-left: 8px solid {ORANGE}; }}
.hero h1 {{ color: #fff; margin: 0; font-size: 1.9rem; }}
.hero p {{ color: #ddd; margin: .35rem 0 0; }}
.band {{ display: inline-block; padding: .2rem .8rem; border-radius: 999px; font-weight: 700; color: #fff; }}
.factor {{ display: flex; justify-content: space-between; padding: .35rem 0;
           border-bottom: 1px solid rgba(128,128,128,.2); }}
</style>
<div class="hero"><h1>📡 Customer Churn Predictor</h1>
<p>Predicts which telecom customers are likely to leave, explains why, and suggests a retention action.</p></div>
""", unsafe_allow_html=True)

BAND_COLOR = {"High": "#D32F2F", "Elevated": ORANGE, "Medium": "#F9A825", "Low": "#2E7D32"}


@st.cache_resource(show_spinner="Loading model…")
def get_model():
    return load_model()


@st.cache_data
def get_reference():
    return load_raw().drop(columns=["Churn"])


model, threshold, model_name = get_model()
reference = get_reference()

tab_one, tab_batch, tab_perf = st.tabs(["🔍 Single customer", "📂 Batch scoring", "📊 Model performance"])

# ---------------------------------------------------------------- single customer
with tab_one:
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.subheader("Customer profile")
        c1, c2 = st.columns(2)
        tenure = c1.slider("Months with the company", 0, 72, 4)
        contract = c2.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet = c1.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        monthly = c2.number_input("Monthly charges ($)", 18.0, 120.0, 85.0, step=1.0)
        payment = c1.selectbox("Payment method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        paperless = c2.selectbox("Paperless billing", ["Yes", "No"])
        senior = c1.selectbox("Senior citizen", ["No", "Yes"])
        partner = c2.selectbox("Has partner", ["No", "Yes"])
        dependents = c1.selectbox("Has dependents", ["No", "Yes"])
        phone = c2.selectbox("Phone service", ["Yes", "No"])

        no_net = internet == "No"
        addon_opts = ["No internet service"] if no_net else ["No", "Yes"]
        st.caption("Add-on services")
        a1, a2, a3 = st.columns(3)
        security = a1.selectbox("Online security", addon_opts)
        backup = a2.selectbox("Online backup", addon_opts)
        protection = a3.selectbox("Device protection", addon_opts)
        support = a1.selectbox("Tech support", addon_opts)
        tv = a2.selectbox("Streaming TV", addon_opts)
        movies = a3.selectbox("Streaming movies", addon_opts)
        lines = ("No phone service" if phone == "No"
                 else st.selectbox("Multiple lines", ["No", "Yes"]))

    customer = pd.DataFrame([{
        "gender": "Male", "SeniorCitizen": int(senior == "Yes"), "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone, "MultipleLines": lines,
        "InternetService": internet, "OnlineSecurity": security, "OnlineBackup": backup,
        "DeviceProtection": protection, "TechSupport": support, "StreamingTV": tv,
        "StreamingMovies": movies, "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly,
        "TotalCharges": monthly * max(tenure, 0),
    }])

    with right:
        p = float(score(model, customer)[0])
        band = risk_band(p, threshold)
        st.subheader("Churn risk")
        m1, m2 = st.columns(2)
        m1.metric("Probability of leaving", f"{p:.0%}")
        m2.markdown(f"<br><span class='band' style='background:{BAND_COLOR[band]}'>{band} risk</span>",
                    unsafe_allow_html=True)
        st.progress(min(p, 1.0))
        st.caption(f"Flagged for retention when probability ≥ {threshold:.0%} (threshold tuned on cross-validation).")

        exp = explain(model, customer, reference)
        st.subheader("Why?")
        for _, r in exp.iterrows():
            arrow, color = ("▲ raises risk", "#D32F2F") if r["impact"] > 0 else ("▼ lowers risk", "#2E7D32")
            st.markdown(
                f"<div class='factor'><span><b>{r['factor']}</b> = {r['value']}</span>"
                f"<span style='color:{color}'>{arrow}</span></div>", unsafe_allow_html=True)

        actions = recommend(exp)
        if p >= threshold and actions:
            st.subheader("Suggested retention actions")
            for a in actions:
                st.markdown(f"- {a}")
        elif p < threshold:
            st.success("Low priority for retention: no action needed right now.")

# ---------------------------------------------------------------- batch scoring
with tab_batch:
    st.subheader("Score a customer list")
    st.write("Upload a CSV with the same columns as the Telco dataset (the `Churn` column is optional). "
             "Customers are ranked so the retention team can call the riskiest first.")
    up = st.file_uploader("Customer CSV", type="csv")
    use_sample = st.button("…or try it on 500 sample customers")
    data = pd.read_csv(up) if up is not None else (reference.sample(500, random_state=1) if use_sample else None)
    if data is not None:
        try:
            out = data.copy()
            out["churn_probability"] = score(model, data).round(3)
            out["risk"] = [risk_band(v, threshold) for v in out["churn_probability"]]
            out = out.sort_values("churn_probability", ascending=False)
            k1, k2, k3 = st.columns(3)
            k1.metric("Customers scored", f"{len(out):,}")
            k2.metric("Flagged for retention", f"{(out['churn_probability'] >= threshold).sum():,}")
            k3.metric("Average risk", f"{out['churn_probability'].mean():.0%}")
            cols = [c for c in ["customerID", "churn_probability", "risk", "Contract", "tenure", "MonthlyCharges"]
                    if c in out.columns]
            st.dataframe(out[cols], use_container_width=True, hide_index=True)
            st.download_button("Download scored list", out.to_csv(index=False), "scored_customers.csv", "text/csv")
        except KeyError as e:
            st.error(f"Missing column in the uploaded file: {e}")

# ---------------------------------------------------------------- model performance
with tab_perf:
    if METRICS_PATH.exists():
        m = json.loads(METRICS_PATH.read_text())
        t = m["selected_test"]
        st.subheader(f"Selected model: {m['selected_model']}")
        c = st.columns(4)
        c[0].metric("ROC-AUC (test)", f"{t['roc_auc']:.3f}")
        c[1].metric("Recall (churners caught)", f"{t['recall']:.0%}")
        c[2].metric("Precision", f"{t['precision']:.0%}")
        c[3].metric("Lift in top 20%", f"{t['lift_top20']:.1f}x")
        st.caption(f"Trained on {m['dataset']['train_size']:,} customers, evaluated on "
                   f"{m['dataset']['test_size']:,} held-out customers.")
        comp = pd.DataFrame(m["test"]).T.rename(columns={"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"}).round(3)
        st.dataframe(comp, use_container_width=True)
    for name in ["roc_curves", "feature_importance", "confusion_matrix", "churn_drivers_eda"]:
        path = ROOT / "images" / f"{name}.svg"
        if path.exists():
            st.markdown(f"<div style='max-width:760px'>{path.read_text()}</div>", unsafe_allow_html=True)

st.caption("Built by Abdullah Asasfeh · Data: IBM Telco Customer Churn (7,043 customers) · scikit-learn + Streamlit")
