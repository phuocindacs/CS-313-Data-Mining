"""HTTP client helpers for calling the FastAPI backend from Streamlit."""

import requests

API_BASE = "http://localhost:8000"
TIMEOUT_SHORT = 15   # seconds — health, metadata, students
TIMEOUT_LONG = 120   # seconds — timeline (runs 6 weeks × 4 models)
TIMEOUT_SHAP = 60    # seconds — SHAP computation


def _get(endpoint: str, timeout: int = TIMEOUT_SHORT) -> dict:
    try:
        r = requests.get(f"{API_BASE}{endpoint}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        return {"error": str(e)}


def _post(endpoint: str, data: dict, timeout: int = TIMEOUT_SHORT) -> dict:
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=data, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        return {"error": str(e)}


# ------------------------------------------------------------------
# Endpoint wrappers
# ------------------------------------------------------------------

def health() -> dict:
    return _get("/health")


def get_students() -> dict:
    """Return {count, students: [{index, student_id, target, actual_label}]}."""
    return _get("/students", timeout=TIMEOUT_LONG)


def get_student(index: int) -> dict:
    return _get(f"/students/{index}")


def get_metadata() -> dict:
    return _get("/metadata")


def get_timeline(student_idx: int) -> dict:
    """Return risk probability across all 6 weeks for all 4 models.

    Shape: {weeks: [4,8,12,16,20,24], timeline: {XGBoost: [...], LightGBM: [...], LSTM: [...], TRANSFORMER: [...]}}
    """
    return _get(f"/timeline/{student_idx}", timeout=TIMEOUT_LONG)


def predict(student_idx: int, week: int) -> dict:
    """Predict risk for one student at a specific week.

    Returns ml_predictions and dl_predictions dicts.
    """
    return _post("/predict", {"student_index": student_idx, "week": week}, timeout=TIMEOUT_SHORT)


def shap_explain(student_idx: int, week: int, model: str = "XGBoost") -> dict:
    """Compute SHAP waterfall data for XGBoost or LightGBM.

    Returns shap_values, base_value, feature_values, feature_names.
    """
    return _post("/shap", {"student_index": student_idx, "week": week, "model": model}, timeout=TIMEOUT_SHAP)


def explain_dl(student_idx: int, week: int, model: str = "LSTM") -> dict:
    """Integrated Gradients attribution for a DL model (LSTM or Transformer).

    Returns feature_attributions, feature_names, week_importance, weeks_axis, prediction.
    """
    return _post("/explain/dl", {"student_index": student_idx, "week": week, "model": model}, timeout=TIMEOUT_SHAP)
