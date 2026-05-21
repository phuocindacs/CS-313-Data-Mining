"""OULAD Early Warning System — Streamlit main app.
4 pages: Risk Timeline, Week Detail, SHAP Analysis, Model Metrics.
"""

# Fix Python 3.12+ / Windows asyncio WebSocket issue with Streamlit
import asyncio
import sys
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st

st.set_page_config(
    page_title="OULAD EWS Demo",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    html, body, [class*="css"] { font-family: 'Segoe UI', Roboto, sans-serif; }
    h1, h2, h3 { font-weight: 600; }

    /* Prediction card — adapts to dark/light mode */
    .pred-card {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 12px;
        padding: 18px 20px;
        margin: 6px 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    }
    .pred-card:hover { box-shadow: 0 3px 10px rgba(0,0,0,0.15); }

    /* Risk badge */
    .badge-risk { background:#fce8e6; color:#c5221f; border:1px solid #f5c6cb; }
    .badge-safe { background:#e6f4ea; color:#137333; border:1px solid #b7e1cd; }
    .badge {
        display:inline-block; padding:5px 14px; border-radius:20px;
        font-weight:600; font-size:13px;
    }

    /* Section title */
    .section-title {
        font-size:11px; text-transform:uppercase; letter-spacing:1px;
        color:#80868b; margin-bottom:4px;
    }

    /* Hide branding & deploy button */
    #MainMenu { visibility:hidden; }
    footer { visibility:hidden; }
    [data-testid="stToolbar"] { visibility:hidden; }
    [data-testid="stDecoration"] { display:none; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎓 OULAD EWS")
    st.caption("CS-313 · Data Mining Demo")
    st.divider()

    page = st.radio(
        "Page",
        ["📈 Risk Timeline", "🔍 Week Detail", "📊 SHAP Analysis", "📋 Model Metrics"],
        label_visibility="collapsed",
    )

    st.divider()

    # API health check
    from ui.utils import api_client
    h = api_client.health()
    if "error" in h:
        st.error("⚠️ API offline\nRun `run.bat` first")
    else:
        ml = h.get("ml_models", 0)
        dl = h.get("dl_models", 0)
        ready = h.get("data_ready", False)
        st.success(f"✅ API online\n{ml} ML · {dl} DL models")
        if not ready:
            st.warning("⚠️ Data not ready yet")

# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------
if "📈" in page:
    from ui.views import risk_timeline
    risk_timeline.render()

elif "🔍" in page:
    from ui.views import week_detail
    week_detail.render()

elif "📊" in page:
    from ui.views import shap_analysis
    shap_analysis.render()

elif "📋" in page:
    from ui.views import model_metrics
    model_metrics.render()
