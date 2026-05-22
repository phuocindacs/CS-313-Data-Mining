"""DL explainability (Integrated Gradients) — per-week importance + top-feature attribution.

Rendered by views/shap_analysis.py when an LSTM / Transformer model is selected.
DL attribution is per week × per feature, so we show which weeks drove the risk
(pairs with the Risk Timeline) plus the aggregated top features.
"""

import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ui.utils import api_client

matplotlib.use("Agg")  # non-interactive backend for Streamlit


@st.cache_data(ttl=60, show_spinner=False)
def _load_explain(student_idx: int, week: int, model: str):
    return api_client.explain_dl(student_idx, week, model)


def _render_week_importance(data: dict):
    """Plotly bar of attribution magnitude per valid week."""
    weeks = data["weeks_axis"]
    imp = data["week_importance"]

    fig = go.Figure(go.Bar(
        x=weeks, y=imp,
        marker_color="#34a853",
        hovertemplate="Week %{x}<br>Importance: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(title="Week", tickvals=weeks, ticktext=[f"W{w}" for w in weeks], gridcolor="#f0f0f0"),
        yaxis=dict(title="Total |attribution|", gridcolor="#f0f0f0"),
        plot_bgcolor="#fff", paper_bgcolor="#fff",
        margin=dict(l=60, r=20, t=20, b=50), height=360,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Higher bars = weeks that contributed most to this prediction.")


def _render_feature_bar(data: dict, top_n: int = 15):
    """Matplotlib horizontal bar of top-N features by |attribution| (signed)."""
    attr = np.array(data["feature_attributions"])
    names = data["feature_names"]

    top_idx = np.argsort(np.abs(attr))[::-1][:top_n]
    top_attr = attr[top_idx]
    top_names = [names[i] for i in top_idx]

    colors = ["#c5221f" if v > 0 else "#137333" for v in top_attr]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(len(top_names)), top_attr[::-1], color=colors[::-1])
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.axvline(0, color="#888", linewidth=0.8)
    ax.set_xlabel("Attribution (impact on At-Risk probability)")
    ax.set_title(f"Top {top_n} features by |attribution|")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render(student_idx: int, week: int, model: str):
    """Load IG attribution for the chosen DL model and render header + tabs."""
    with st.spinner(f"Computing Integrated Gradients for {model} at week {week}…"):
        data = _load_explain(student_idx, week, model)

    if "error" in data:
        st.error(f"Explain error: {data['error']}")
        return
    if not data:
        st.info("Select a student and model above to compute attributions.")
        return

    # --- Student header ---
    student = data.get("student", {})
    actual = student.get("actual_label", "")
    target = student.get("target", -1)
    badge_cls = "badge-risk" if target == 1 else "badge-safe"
    pred = data.get("prediction", 0.0)

    st.markdown(f"""
    <div style="background:#f8f9fa; border-radius:10px; padding:12px 18px; margin:10px 0;">
        <span style="color:#80868b;">Student {student.get('student_id', '—')} &nbsp;·&nbsp;
        <b>{model}</b> at <b>Week {week}</b> &nbsp;·&nbsp;
        Method: <b>Integrated Gradients (Captum)</b> &nbsp;·&nbsp;
        Predicted risk: <b>{pred:.1%}</b></span>
        &nbsp;&nbsp;
        <span class="badge {badge_cls}">Actual: {actual}</span>
    </div>
    """, unsafe_allow_html=True)

    # --- Tabs: per-week / top features / values ---
    tab_week, tab_bar, tab_vals = st.tabs(["📅 Per-week importance", "📊 Top features", "📋 Attribution table"])

    with tab_week:
        _render_week_importance(data)

    with tab_bar:
        top_n = st.slider("Top N features", 5, 30, 15, key="ig_top_n")
        _render_feature_bar(data, top_n=top_n)

    with tab_vals:
        attr = np.array(data["feature_attributions"])
        names = data["feature_names"]
        df = pd.DataFrame({
            "Feature": names,
            "Attribution": attr,
            "|Attribution|": np.abs(attr),
        }).sort_values("|Attribution|", ascending=False).reset_index(drop=True)
        st.dataframe(
            df.style.format({"Attribution": "{:.4f}", "|Attribution|": "{:.4f}"}),
            use_container_width=True, height=420,
        )
