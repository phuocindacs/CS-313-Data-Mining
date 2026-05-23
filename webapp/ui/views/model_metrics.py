"""Page 4 — Model Metrics: real evaluation metrics (AUC, F1, Precision, Recall)
plus system status overview."""

import streamlit as st
import pandas as pd

from ui.utils import api_client

WEEKS = [4, 8, 12, 16, 20, 24]


@st.cache_data(ttl=60, show_spinner=False)
def _load_metadata():
    return api_client.get_metadata()


@st.cache_data(ttl=300, show_spinner=False)
def _load_metrics():
    return api_client.get_metrics()


def render():
    st.title("📋 Model Metrics")
    st.caption("Evaluation metrics on the test set and system overview.")

    meta = _load_metadata()
    if "error" in meta:
        st.error(f"Cannot reach API: {meta['error']}")
        return

    # --- Load metrics ---
    with st.spinner("Computing evaluation metrics on test set (first load may take a minute)…"):
        metrics = _load_metrics()

    if "error" in metrics:
        st.error(f"Metrics error: {metrics['error']}")
        return

    # --- Class distribution + system stats ---
    dist = metrics.get("class_distribution", {})
    ml_loaded = meta.get("ml_models_loaded", [])
    dl_loaded = meta.get("dl_models_loaded", [])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Test students", f"{dist.get('total', 0):,}")
    col2.metric("At-Risk", f"{dist.get('at_risk', 0):,} ({dist.get('at_risk_pct', 0)}%)")
    col3.metric("Not At-Risk", f"{dist.get('not_at_risk', 0):,} ({dist.get('not_at_risk_pct', 0)}%)")
    col4.metric("Models loaded", f"{len(ml_loaded)} ML + {len(dl_loaded)} DL")

    st.divider()

    # --- ML Metrics Table ---
    st.subheader("📦 ML Models — Per-Week Metrics")

    ml_data = metrics.get("ml", {})
    if ml_data:
        rows = []
        for key, m in sorted(ml_data.items(), key=lambda x: (x[1]["model"], x[1]["week"])):
            cm = m.get("confusion_matrix", {})
            rows.append({
                "Model": m["model"],
                "Week": m["week"],
                "AUC-ROC": m["auc_roc"],
                "F1": m["f1"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "Accuracy": m["accuracy"],
                "TP": cm.get("TP", 0),
                "FP": cm.get("FP", 0),
                "TN": cm.get("TN", 0),
                "FN": cm.get("FN", 0),
            })
        df_ml = pd.DataFrame(rows)
        st.dataframe(
            df_ml.style.format({
                "AUC-ROC": "{:.4f}", "F1": "{:.4f}",
                "Precision": "{:.4f}", "Recall": "{:.4f}", "Accuracy": "{:.4f}",
            }).highlight_max(subset=["AUC-ROC", "F1"], color="#d4edda"),
            use_container_width=True, hide_index=True, height=300,
        )
    else:
        st.info("No ML metrics available.")

    st.divider()

    # --- DL Metrics Table ---
    st.subheader("🧠 DL Models — Per-Week Metrics")
    st.caption("LSTM + Transformer evaluated at each cutoff week (sequence masked beyond cutoff)")

    dl_data = metrics.get("dl", {})
    if dl_data:
        rows = []
        for key, m in sorted(dl_data.items(), key=lambda x: (x[1]["model"], x[1]["week"])):
            cm = m.get("confusion_matrix", {})
            rows.append({
                "Model": m["model"],
                "Week": m["week"],
                "AUC-ROC": m["auc_roc"],
                "F1": m["f1"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "Accuracy": m["accuracy"],
                "TP": cm.get("TP", 0),
                "FP": cm.get("FP", 0),
                "TN": cm.get("TN", 0),
                "FN": cm.get("FN", 0),
            })
        df_dl = pd.DataFrame(rows)
        st.dataframe(
            df_dl.style.format({
                "AUC-ROC": "{:.4f}", "F1": "{:.4f}",
                "Precision": "{:.4f}", "Recall": "{:.4f}", "Accuracy": "{:.4f}",
            }).highlight_max(subset=["AUC-ROC", "F1"], color="#d4edda"),
            use_container_width=True, hide_index=True, height=300,
        )
    else:
        st.info("No DL metrics available.")

    st.divider()

    # --- Feature breakdown (collapsed) ---
    st.subheader("🔢 Feature breakdown")

    dyn = meta.get("dynamic_features", [])
    sta = meta.get("static_features", [])

    col_d, col_s = st.columns(2)
    with col_d:
        st.markdown(f"**Dynamic features** ({len(dyn)}) — cumulative weekly aggregates")
        if dyn:
            with st.expander("Show feature list", expanded=False):
                for f in dyn:
                    st.text(f"  • {f}")

    with col_s:
        st.markdown(f"**Static features** ({len(sta)}) — student/course demographics")
        if sta:
            with st.expander("Show feature list", expanded=False):
                for f in sta:
                    st.text(f"  • {f}")
