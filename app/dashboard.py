"""FraudShield dashboard. Run: streamlit run app/dashboard.py"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))            # lets Python find the src package
from src.metrics import net_savings      # noqa: E402

MODELS, REPORTS = ROOT / "models", ROOT / "reports"
st.set_page_config(page_title="FraudShield", layout="wide")


@st.cache_data
def load_csv(name):
    return pd.read_csv(REPORTS / name)


@st.cache_data
def load_json(path):
    return json.loads(Path(path).read_text())


@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model(MODELS / "xgb_model.json")
    return model, shap.TreeExplainer(model)


config = load_json(MODELS / "config.json")
metrics = load_json(REPORTS / "metrics.json")
feats = config["features"]

page = st.sidebar.radio("Page", [
    "Overview", "Model comparison", "Cost & threshold simulator",
    "Transaction scorer", "Analyst review queue"])
st.sidebar.caption("Time-based split: train on the past, test on the future.")

# ---------------------------------------------------------------- Overview
if page == "Overview":
    st.title("FraudShield: credit card fraud detection")
    at = metrics["at_threshold"]
    xgb_row = next(m for m in metrics["models"] if m["model"] == "xgboost")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test PR-AUC", f"{xgb_row['test_pr_auc']:.3f}")
    c2.metric("Recall at 90% precision", f"{xgb_row['test_recall_at_precision_90']:.1%}")
    c3.metric("Frauds caught / missed", f"{at['frauds_caught']} / {at['frauds_missed']}")
    c4.metric("Net savings (test)", f"{at['net_savings']:,.0f}")
    hourly = load_csv("hourly.csv")
    fig = go.Figure()
    fig.add_bar(x=hourly["hour_index"], y=hourly["transactions"], name="Transactions")
    fig.add_scatter(x=hourly["hour_index"], y=hourly["frauds"], name="Frauds", yaxis="y2", mode="lines+markers")
    fig.update_layout(title="Transactions and frauds per hour since the first transaction",
                      xaxis_title="Hour", yaxis_title="Transactions",
                      yaxis2=dict(title="Frauds", overlaying="y", side="right"))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Deployed model: XGBoost, imbalance strategy '{config['best_imbalance_strategy']}', "
               f"alert threshold {config['threshold']:.2f}. Features are anonymised PCA components, "
               "so feature engineering is limited.")

# ------------------------------------------------------- Model comparison
elif page == "Model comparison":
    st.title("Model comparison (same time-based test set)")
    table = pd.DataFrame(metrics["models"])
    st.dataframe(table.round(4), width="stretch")
    curves = load_csv("pr_curves.csv")
    st.plotly_chart(px.line(curves, x="recall", y="precision", color="model",
                            title="Precision-recall curves (test)"), width="stretch")
    st.subheader("Imbalance strategies (XGBoost, validation set)")
    st.dataframe(pd.DataFrame(metrics["imbalance_study"]).round(4), width="stretch")
    st.subheader("Calibration (test)")
    cal = load_csv("calibration.csv")
    fig = px.line(cal, x="mean_predicted", y="fraction_fraud", markers=True,
                  title="Reliability curve: predicted vs observed fraud rate")
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
    st.plotly_chart(fig, width="stretch")
    st.write(f"Brier score (lower is better): raw {metrics['brier']['raw']:.6f}, "
             f"calibrated {metrics['brier']['calibrated']:.6f}")
    st.subheader("Global feature importance (mean |SHAP|)")
    st.bar_chart(load_csv("shap_importance.csv").head(15).set_index("feature"))

# ------------------------------------------------ Cost & threshold simulator
elif page == "Cost & threshold simulator":
    st.title("Cost and threshold simulator")
    table = load_csv("threshold_table.csv")
    c1, c2, c3 = st.columns(3)
    review_cost = c1.slider("Cost of reviewing one alert", 0.0, 100.0, float(config["review_cost"]), 0.5)
    multiplier = c2.slider("Loss multiplier for a missed fraud", 1.0, 5.0, 1.0, 0.1,
                           help="1 = you lose the transaction amount; higher includes fees and chargebacks")
    capacity = c3.slider("Max alerts the team can review per day", 1, 2000, 500, 1)
    table = table.assign(savings=net_savings(table, review_cost, multiplier),
                         alerts_per_day=table["flagged"] / max(config["test_days"], 1e-6))
    allowed = table[table["alerts_per_day"] <= capacity]
    if allowed.empty:
        st.warning("No threshold fits that review capacity; raise the capacity.")
    else:
        best = allowed.loc[allowed["savings"].idxmax()]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Best threshold", f"{best['threshold']:.3f}")
        m2.metric("Frauds caught / missed", f"{int(best['caught'])} / {int(best['missed'])}")
        m3.metric("Alerts per day", f"{best['alerts_per_day']:.0f}")
        m4.metric("Net savings", f"{best['savings']:,.0f}")
        fig = px.line(table, x="threshold", y="savings", title="Net savings by alert threshold")
        fig.add_vline(x=float(best["threshold"]), line_dash="dash")
        st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------- Transaction scorer
elif page == "Transaction scorer":
    st.title("Score one transaction and see why")
    sample = load_csv("sample_transactions.csv").reset_index(drop=True)
    model, explainer = load_model()
    choice = st.selectbox(
        "Pick a transaction from the test period", sample.index,
        format_func=lambda i: f"#{i} | amount {sample.loc[i, 'Amount']:.2f} | hour {sample.loc[i, 'hour']}")
    row = sample.loc[[choice]]
    score = float(row["score"].iloc[0])
    decision = "FLAG for review" if score >= config["threshold"] else "Allow"
    c1, c2, c3 = st.columns(3)
    c1.metric("Fraud probability", f"{score:.3f}")
    c2.metric("Decision", decision)
    c3.metric("Actual label (demo only)", "Fraud" if int(row["Class"].iloc[0]) == 1 else "Legitimate")
    values = explainer(row[feats])
    shap.plots.waterfall(values[0], show=False)
    st.pyplot(plt.gcf(), clear_figure=True)
    st.caption("Bars show how each feature pushed this transaction's raw score up (red) or down (blue).")

# ------------------------------------------------- Analyst review queue
elif page == "Analyst review queue":
    st.title("Analyst review queue")
    queue = load_csv("top_alerts.csv").reset_index(drop=True)
    n = st.slider("Show top N alerts", 5, 50, 15)
    view = queue.head(n)[["Time", "Amount", "hour", "score", "Class"]].rename(
        columns={"Class": "ground_truth (demo only)"})
    st.dataframe(view, width="stretch")
    st.subheader("Record a decision")
    idx = st.selectbox("Alert #", list(range(n)))
    decision = st.radio("Analyst decision", ["Confirmed fraud", "Not fraud"], horizontal=True)
    if st.button("Save decision"):
        log = REPORTS / "feedback_log.csv"
        entry = pd.DataFrame([{"Time": queue.loc[idx, "Time"], "Amount": queue.loc[idx, "Amount"],
                               "score": queue.loc[idx, "score"], "decision": decision}])
        entry.to_csv(log, mode="a", header=not log.exists(), index=False)
        st.success("Saved. These labels can feed the next retraining run.")
    if (REPORTS / "feedback_log.csv").exists():
        st.dataframe(pd.read_csv(REPORTS / "feedback_log.csv"), width="stretch")