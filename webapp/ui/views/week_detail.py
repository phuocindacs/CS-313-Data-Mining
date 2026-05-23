"""Page 2 — Week Detail: all 4 model predictions for a student at a specific week."""

import streamlit as st

from ui.utils import api_client

WEEKS = [4, 8, 12, 16, 20, 24]

ML_MODELS = ["XGBoost", "LightGBM"]
DL_MODELS = ["LSTM", "TRANSFORMER"]

MODEL_COLORS = {
    "XGBoost":     "#1a73e8",
    "LightGBM":    "#f9ab00",
    "LSTM":        "#34a853",
    "TRANSFORMER": "#ea4335",
}

MODEL_LABELS = {
    "LSTM": "LSTM",
    "TRANSFORMER": "Transformer",
}


@st.cache_data(ttl=300, show_spinner=False)
def _load_students():
    data = api_client.get_students()
    return data.get("students", [])


@st.cache_data(ttl=30, show_spinner=False)
def _predict(student_idx: int, week: int):
    return api_client.predict(student_idx, week)


def _pred_card(model_name: str, pred: dict, model_type: str = "ML"):
    prob = pred.get("probability", 0.0)
    label = pred.get("label", "—")
    is_risk = label == "At-Risk"
    badge_cls = "badge-risk" if is_risk else "badge-safe"
    color = MODEL_COLORS.get(model_name, "#888")
    display = MODEL_LABELS.get(model_name, model_name)
    pct = f"{prob:.1%}"

    st.markdown(f"""
    <div class="pred-card" style="border-left: 4px solid {color};">
        <div class="section-title">{model_type} · {display}</div>
        <div style="display:flex; align-items:center; gap:14px; margin-top:8px;">
            <span style="font-size:26px; font-weight:700; color:{color};">{pct}</span>
            <span class="badge {badge_cls}">{label}</span>
        </div>
        <div style="margin-top:10px; background:#f5f5f5; border-radius:6px; height:8px; overflow:hidden;">
            <div style="width:{pct}; height:100%;
                        background:{'#c5221f' if is_risk else '#137333'};
                        border-radius:6px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render():
    st.title("🔍 Week Detail")
    st.caption("Pick a student and a cutoff week — see what each model predicts at that snapshot.")

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
        chosen_label = st.selectbox("Select student", list(options.keys()), key="detail_student")
        student_idx = options[chosen_label]

    with col2:
        week = st.selectbox("Cutoff week", WEEKS, index=2, key="detail_week")

    with col3:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        run = st.button("🔄 Predict", use_container_width=True)

    # Only predict when button is clicked
    if run:
        with st.spinner("Running models…"):
            result = _predict(student_idx, week)
        st.session_state["detail_result"] = result
        st.session_state["detail_last"] = (student_idx, week)
    elif "detail_result" not in st.session_state:
        return
    else:
        result = st.session_state["detail_result"]

    if "error" in result:
        st.error(f"Prediction error: {result['error']}")
        return

    student = result.get("student", {})
    ml_preds = result.get("ml_predictions", {})
    dl_preds = result.get("dl_predictions", {})

    # --- Student header ---
    actual = student.get("actual_label", "")
    target = student.get("target", -1)
    badge_cls = "badge-risk" if target == 1 else "badge-safe"

    st.markdown(f"""
    <div style="background:#f8f9fa; border-radius:10px; padding:12px 18px; margin:12px 0;">
        <span style="font-size:13px; color:#80868b;">Student {student.get('student_id', '—')} &nbsp;·&nbsp;
        {student.get('code_module', '')} {student.get('code_presentation', '')} &nbsp;·&nbsp;
        Cutoff: <b>Week {week}</b></span>
        &nbsp;&nbsp;
        <span class="badge {badge_cls}">Actual: {actual}</span>
    </div>
    """, unsafe_allow_html=True)

    # --- Prediction cards ---
    st.subheader("Model predictions")
    col_ml, col_dl = st.columns(2)

    with col_ml:
        st.markdown("**📦 ML models** *(week-specific snapshot)*")
        for m in ML_MODELS:
            if m in ml_preds:
                _pred_card(m, ml_preds[m], "ML")
            else:
                st.info(f"{m}: not loaded")

    with col_dl:
        st.markdown("**🧠 DL models** *(sequence masked to week)*")
        for m in DL_MODELS:
            if m in dl_preds:
                _pred_card(m, dl_preds[m], "DL")
            else:
                st.info(f"{m}: not loaded")

    # --- Consensus (majority vote) ---
    all_labels = (
        [p["label"] for p in ml_preds.values()] +
        [p["label"] for p in dl_preds.values()]
    )
    if all_labels:
        n_risk = sum(1 for l in all_labels if l == "At-Risk")
        n_total = len(all_labels)
        consensus = "At-Risk" if n_risk > n_total / 2 else "Not At-Risk"
        badge_cls2 = "badge-risk" if consensus == "At-Risk" else "badge-safe"
        st.divider()
        st.markdown(f"""
        <div style="text-align:center; padding:12px;">
            <span class="section-title">Majority vote</span><br>
            <span style="font-size:26px; font-weight:700;">{n_risk}/{n_total} models predict At-Risk</span>
            &nbsp;&nbsp;
            <span class="badge {badge_cls2}" style="font-size:15px;">{consensus}</span>
        </div>
        """, unsafe_allow_html=True)
