"""ML explainability (SHAP TreeExplainer) — waterfall, top-feature bar, values table.

Rendered by views/shap_analysis.py when an XGBoost / LightGBM model is selected.
"""

import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
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

    st.markdown(f"""
    <div style="background:#f8f9fa; border-radius:10px; padding:12px 18px; margin:10px 0;">
        <span style="color:#80868b;">Student {student.get('student_id', '—')} &nbsp;·&nbsp;
        <b>{model}</b> at <b>Week {week}</b></span>
        &nbsp;&nbsp;
        <span class="badge {badge_cls}">Actual: {actual}</span>
    </div>
    """, unsafe_allow_html=True)

    # --- Waterfall ---
    _render_waterfall(shap_data)
