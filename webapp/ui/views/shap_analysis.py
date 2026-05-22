"""Page 3 — Explainability: feature-level explanation per model.

ML models (XGBoost, LightGBM) use SHAP (TreeExplainer); DL models (LSTM, Transformer)
use Integrated Gradients. This module is the page router; rendering lives in
views/explain_ml.py and views/explain_dl.py.
"""

import streamlit as st

from ui.utils import api_client
from ui.views import explain_ml, explain_dl

WEEKS = [4, 8, 12, 16, 20, 24]
ML_MODELS = ["XGBoost", "LightGBM"]
DL_MODELS = ["LSTM", "Transformer"]
ALL_MODELS = ML_MODELS + DL_MODELS


@st.cache_data(ttl=300, show_spinner=False)
def _load_students():
    data = api_client.get_students()
    return data.get("students", [])


def render():
    st.title("📊 Explainability")
    st.caption("Why a model flags a student — SHAP for XGBoost/LightGBM, Integrated Gradients for LSTM/Transformer.")

    students = _load_students()
    if not students:
        st.error("No student data. Make sure the API is running.")
        return

    # --- Controls ---
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        options = {
            f"#{s['index']}  •  ID {s['student_id']}": s["index"]
            for s in students
        }
        chosen_label = st.selectbox("Select student", list(options.keys()), key="shap_student")
        student_idx = options[chosen_label]

    with col2:
        week = st.selectbox("Cutoff week", WEEKS, index=2, key="shap_week")

    with col3:
        model = st.selectbox("Model", ALL_MODELS, key="shap_model")

    # --- Dispatch by model type ---
    if model in ML_MODELS:
        explain_ml.render(student_idx, week, model)
    else:
        explain_dl.render(student_idx, week, model)
