"""Page 1 — Risk Timeline: probability trajectory across all weeks for all 4 models."""

import streamlit as st
import plotly.graph_objects as go

from ui.utils import api_client

# Model display config: key → (display name, color, dash style)
MODEL_STYLE = {
    "XGBoost":     ("XGBoost",     "#1a73e8", "solid"),
    "LightGBM":    ("LightGBM",    "#fbbc04", "solid"),
    "LSTM":        ("LSTM",        "#34a853", "dash"),
    "TRANSFORMER": ("Transformer", "#ea4335", "dash"),
}


@st.cache_data(ttl=300, show_spinner=False)
def _load_students():
    data = api_client.get_students()
    return data.get("students", [])


@st.cache_data(ttl=60, show_spinner=False)
def _load_timeline(student_idx: int):
    return api_client.get_timeline(student_idx)


def render():
    st.title("📈 Risk Timeline")
    st.caption("At-Risk probability trajectory across all cutoff weeks — select a student to explore.")

    students = _load_students()
    if not students:
        st.error("No student data available. Make sure the API is running and data is loaded.")
        return

    # --- Student selector ---
    col_sel, col_info = st.columns([2, 3])
    with col_sel:
        options = {
            f"#{s['index']}  •  ID {s['student_id']}": s["index"]
            for s in students
        }
        chosen_label = st.selectbox(
            "Select student",
            list(options.keys()),
            key="timeline_student",
        )
        student_idx = options[chosen_label]

    student = next((s for s in students if s["index"] == student_idx), {})

    with col_info:
        actual = student.get("actual_label", "")
        target = student.get("target", -1)
        badge_cls = "badge-risk" if target == 1 else "badge-safe"
        st.markdown(f"""
        <div style='margin-top:24px;'>
            <span class='section-title'>Student ID</span><br>
            <span style='font-size:18px; font-weight:600;'>{student.get('student_id', '—')}</span>
            &nbsp;&nbsp;
            <span class='badge {badge_cls}'>{actual}</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # --- Load timeline ---
    with st.spinner("Computing predictions across all weeks…"):
        tl = _load_timeline(student_idx)

    if "error" in tl:
        st.error(f"Timeline error: {tl['error']}")
        return

    weeks = tl.get("weeks", [])
    timeline = tl.get("timeline", {})

    if not weeks or not timeline:
        st.warning("No timeline data returned.")
        return

    # --- Build Plotly chart ---
    fig = go.Figure()

    # Threshold line
    fig.add_hline(
        y=0.5, line_dash="dot", line_color="#888",
        annotation_text="Threshold 0.5", annotation_position="right",
    )

    for model_key, (display_name, color, dash) in MODEL_STYLE.items():
        probs = timeline.get(model_key)
        if probs is None:
            continue

        fig.add_trace(go.Scatter(
            x=weeks,
            y=probs,
            mode="lines+markers",
            name=display_name,
            line=dict(color=color, width=2.5, dash=dash),
            marker=dict(size=8, symbol="circle"),
            hovertemplate=(
                f"<b>{display_name}</b><br>"
                "Week: %{x}<br>"
                "Risk prob: %{y:.1%}<extra></extra>"
            ),
        ))

    fig.update_layout(
        xaxis=dict(
            title="Cutoff week",
            tickvals=weeks,
            ticktext=[f"Week {w}" for w in weeks],
            gridcolor="#f0f0f0",
        ),
        yaxis=dict(
            title="At-Risk probability",
            range=[-0.05, 1.05],
            tickformat=".0%",
            gridcolor="#f0f0f0",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#fff",
        paper_bgcolor="#fff",
        margin=dict(l=60, r=20, t=40, b=60),
        height=440,
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- Summary table ---
    st.subheader("Probability table")
    rows = []
    for model_key, (display_name, _, _) in MODEL_STYLE.items():
        probs = timeline.get(model_key)
        if probs is None:
            continue
        row = {"Model": display_name}
        for i, w in enumerate(weeks):
            row[f"Week {w}"] = f"{probs[i]:.1%}" if probs[i] is not None else "—"
        rows.append(row)

    if rows:
        import pandas as pd
        df = pd.DataFrame(rows).set_index("Model")
        st.dataframe(df, use_container_width=True)
