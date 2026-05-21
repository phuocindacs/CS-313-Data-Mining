"""Page 3 — SHAP Analysis: waterfall + top-feature bar chart for XGBoost or LightGBM."""

import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import shap

from ui.utils import api_client

matplotlib.use("Agg")  # non-interactive backend for Streamlit

WEEKS = [4, 8, 12, 16, 20, 24]
ML_MODELS = ["XGBoost", "LightGBM"]


@st.cache_data(ttl=300, show_spinner=False)
def _load_students():
    data = api_client.get_students()
    return data.get("students", [])


@st.cache_data(ttl=60, show_spinner=False)
def _load_shap(student_idx: int, week: int, model: str):
    return api_client.shap_explain(student_idx, week, model)


def _render_waterfall(shap_data: dict, max_display: int = 15):
    """Render SHAP waterfall chart using shap + matplotlib."""
    sv = np.array(shap_data["shap_values"])
    bv = float(shap_data["base_value"])
    fv = np.array(shap_data["feature_values"])
    names = shap_data["feature_names"]

    explanation = shap.Explanation(
        values=sv,
        base_values=bv,
        data=fv,
        feature_names=names,
    )

    fig, _ = plt.subplots(figsize=(10, 7))
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _render_bar(shap_data: dict, top_n: int = 15):
    """Bar chart of top-N features by |SHAP value|."""
    sv = np.array(shap_data["shap_values"])
    names = shap_data["feature_names"]

    abs_sv = np.abs(sv)
    top_idx = np.argsort(abs_sv)[::-1][:top_n]
    top_sv = sv[top_idx]
    top_names = [names[i] for i in top_idx]

    colors = ["#c5221f" if v > 0 else "#137333" for v in top_sv]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(range(len(top_names)), top_sv[::-1], color=colors[::-1])
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.axvline(0, color="#888", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on At-Risk probability)")
    ax.set_title(f"Top {top_n} features by |SHAP|")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render():
    st.title("📊 SHAP Analysis")
    st.caption("Feature-level explanation for XGBoost or LightGBM at a specific cutoff week.")

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
        model = st.selectbox("Model", ML_MODELS, key="shap_model")

    # --- Load SHAP ---
    cache_key = (student_idx, week, model)
    if cache_key != st.session_state.get("shap_last"):
        with st.spinner(f"Computing SHAP for {model} at week {week}…"):
            shap_data = _load_shap(student_idx, week, model)
        st.session_state["shap_data"] = shap_data
        st.session_state["shap_last"] = cache_key
    else:
        shap_data = st.session_state.get("shap_data", {})

    if "error" in shap_data:
        st.error(f"SHAP error: {shap_data['error']}")
        return

    if not shap_data:
        st.info("Select a student and model above to compute SHAP values.")
        return

    # --- Student header ---
    student = shap_data.get("student", {})
    actual = student.get("actual_label", "")
    target = student.get("target", -1)
    badge_cls = "badge-risk" if target == 1 else "badge-safe"
    bv = shap_data.get("base_value", 0.0)

    st.markdown(f"""
    <div style="background:#f8f9fa; border-radius:10px; padding:12px 18px; margin:10px 0;">
        <span style="color:#80868b;">Student {student.get('student_id', '—')} &nbsp;·&nbsp;
        <b>{model}</b> at <b>Week {week}</b> &nbsp;·&nbsp;
        Base value: <b>{bv:.3f}</b></span>
        &nbsp;&nbsp;
        <span class="badge {badge_cls}">Actual: {actual}</span>
    </div>
    """, unsafe_allow_html=True)

    # --- Tabs: waterfall vs bar ---
    tab_wf, tab_bar, tab_vals = st.tabs(["🌊 Waterfall", "📊 Top features", "📋 Feature values"])

    with tab_wf:
        _render_waterfall(shap_data)

    with tab_bar:
        top_n = st.slider("Top N features", 5, 30, 15, key="shap_top_n")
        _render_bar(shap_data, top_n=top_n)

    with tab_vals:
        import pandas as pd
        sv = np.array(shap_data["shap_values"])
        fv = np.array(shap_data["feature_values"])
        names = shap_data["feature_names"]
        df = pd.DataFrame({
            "Feature": names,
            "Value": fv,
            "SHAP": sv,
            "|SHAP|": np.abs(sv),
        }).sort_values("|SHAP|", ascending=False).reset_index(drop=True)
        st.dataframe(df.style.format({"Value": "{:.4f}", "SHAP": "{:.4f}", "|SHAP|": "{:.4f}"}),
                     use_container_width=True, height=420)
