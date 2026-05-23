"""DL explainability (Integrated Gradients) — per-week importance + top-feature attribution.

Rendered by views/shap_analysis.py when an LSTM / Transformer model is selected.
DL attribution is per week × per feature, so we show which weeks drove the risk
(pairs with the Risk Timeline) plus the aggregated top features.
"""

import streamlit as st
import plotly.graph_objects as go

from ui.utils import api_client


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

    # --- Per-week importance ---
    _render_week_importance(data)
