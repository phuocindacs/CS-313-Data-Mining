"""ML explainability (SHAP TreeExplainer) — waterfall, top-feature bar, values table.

Rendered by views/shap_analysis.py when an XGBoost / LightGBM model is selected.
"""

import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from ui.utils import api_client

matplotlib.use("Agg")  # non-interactive backend for Streamlit


@st.cache_data(ttl=60, show_spinner=False)
def _load_shap(student_idx: int, week: int, model: str):
    return api_client.shap_explain(student_idx, week, model)


def _render_waterfall(shap_data: dict, max_display: int = 15):
    """Render SHAP waterfall chart using shap + matplotlib."""
    sv = np.array(shap_data["shap_values"])
    bv = float(shap_data["base_value"])
    fv = np.array(shap_data["feature_values"])
    names = shap_data["feature_names"]

    explanation = shap.Explanation(values=sv, base_values=bv, data=fv, feature_names=names)

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
    ax.barh(range(len(top_names)), top_sv[::-1], color=colors[::-1])
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.axvline(0, color="#888", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on At-Risk probability)")
    ax.set_title(f"Top {top_n} features by |SHAP|")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render(student_idx: int, week: int, model: str):
    """Load SHAP for the chosen ML model and render header + tabs."""
    with st.spinner(f"Computing SHAP for {model} at week {week}…"):
        shap_data = _load_shap(student_idx, week, model)

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
        Method: <b>SHAP (TreeExplainer)</b> &nbsp;·&nbsp;
        Base value: <b>{bv:.3f}</b></span>
        &nbsp;&nbsp;
        <span class="badge {badge_cls}">Actual: {actual}</span>
    </div>
    """, unsafe_allow_html=True)

    # --- Tabs: waterfall / bar / values ---
    tab_wf, tab_bar, tab_vals = st.tabs(["🌊 Waterfall", "📊 Top features", "📋 Feature values"])

    with tab_wf:
        _render_waterfall(shap_data)

    with tab_bar:
        top_n = st.slider("Top N features", 5, 30, 15, key="shap_top_n")
        _render_bar(shap_data, top_n=top_n)

    with tab_vals:
        sv = np.array(shap_data["shap_values"])
        fv = np.array(shap_data["feature_values"])
        names = shap_data["feature_names"]
        df = pd.DataFrame({
            "Feature": names,
            "Value": fv,
            "SHAP": sv,
            "|SHAP|": np.abs(sv),
        }).sort_values("|SHAP|", ascending=False).reset_index(drop=True)
        st.dataframe(
            df.style.format({"Value": "{:.4f}", "SHAP": "{:.4f}", "|SHAP|": "{:.4f}"}),
            use_container_width=True, height=420,
        )
