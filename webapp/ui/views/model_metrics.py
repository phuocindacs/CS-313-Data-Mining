"""Page 4 — Model Metrics: loaded model status, feature info, and system overview."""

import streamlit as st

from ui.utils import api_client


@st.cache_data(ttl=60, show_spinner=False)
def _load_metadata():
    return api_client.get_metadata()


def render():
    st.title("📋 Model Metrics")
    st.caption("System overview — loaded models, feature info, and dataset stats.")

    meta = _load_metadata()

    if "error" in meta:
        st.error(f"Cannot reach API: {meta['error']}")
        return

    # --- Top-level stats ---
    col1, col2, col3, col4 = st.columns(4)
    weeks = meta.get("weeks", [])
    ml_loaded = meta.get("ml_models_loaded", [])
    dl_loaded = meta.get("dl_models_loaded", [])

    col1.metric("Students (test set)", f"{meta.get('n_students', 0):,}")
    col2.metric("ML models loaded", len(ml_loaded))
    col3.metric("DL models loaded", len(dl_loaded))
    col4.metric("Feature count", meta.get("feature_count", 0))

    st.divider()

    # --- ML models ---
    col_ml, col_dl = st.columns(2)

    with col_ml:
        st.subheader("📦 ML Weekly Models")
        st.caption("XGBoost + LightGBM trained at fixed week cutoffs")

        expected_ml = [
            f"{name}_w{w}"
            for name in ["XGBoost", "LGBM"]
            for w in weeks
        ]
        rows = []
        for key in expected_ml:
            name_part, week_part = key.rsplit("_w", 1)
            display = "XGBoost" if name_part == "XGBoost" else "LightGBM"
            status = "✅ Loaded" if key in ml_loaded else "❌ Missing"
            rows.append({"Model": display, "Week": int(week_part), "Status": status})

        import pandas as pd
        df_ml = pd.DataFrame(rows)
        st.dataframe(df_ml, use_container_width=True, hide_index=True, height=260)

    with col_dl:
        st.subheader("🧠 DL Sequence Models")
        st.caption("LSTM + Transformer trained on full 25-week sequence")

        expected_dl = ["LSTM", "TRANSFORMER"]
        dl_rows = []
        for name in expected_dl:
            display = "Transformer" if name == "TRANSFORMER" else name
            status = "✅ Loaded" if name in dl_loaded else "❌ Missing"
            dl_rows.append({"Model": display, "Type": "Sequence", "Status": status})

        df_dl = pd.DataFrame(dl_rows)
        st.dataframe(df_dl, use_container_width=True, hide_index=True, height=130)

        st.divider()
        st.subheader("🗓️ Prediction weeks")
        st.write(f"Weeks: {weeks}")
        st.caption("ML uses week-specific snapshots; DL masks data beyond each cutoff.")

    # --- Features ---
    st.divider()
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

    # --- Labels ---
    st.divider()
    labels = meta.get("labels", {"0": "Not At-Risk", "1": "At-Risk"})
    st.subheader("🏷️ Label mapping")
    for k, v in labels.items():
        badge_cls = "badge-risk" if k == "1" else "badge-safe"
        st.markdown(
            f"<span style='font-weight:600;'>Class {k}:</span> "
            f"<span class='badge {badge_cls}'>{v}</span>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.info(
        "💡 **Tip:** Detailed training metrics (AUC, F1, accuracy per week) are available "
        "in the training notebooks under `ketqua/OULAD ML/` and `ketqua/OULAD sequence/`."
    )
