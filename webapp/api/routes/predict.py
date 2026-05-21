"""POST /predict — run ML + DL models on a student at a specific week.
POST /shap  — compute SHAP waterfall for XGBoost or LightGBM.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.model_loader import registry
from api.data_manager import data_manager
from api.shap_explainer import ShapExplainer

router = APIRouter()

THRESHOLD = 0.5  # risk probability threshold


def _label(prob: float) -> str:
    return "At-Risk" if prob >= THRESHOLD else "Not At-Risk"


class PredictRequest(BaseModel):
    student_index: int
    week: int = 4  # one of [4, 8, 12, 16, 20, 24]


class ShapRequest(BaseModel):
    student_index: int
    week: int = 4
    model: str = "XGBoost"  # "XGBoost" or "LightGBM"


@router.post("/predict")
def predict(req: PredictRequest):
    """Predict At-Risk probability for one student at a given week.

    Returns ML predictions (XGBoost, LightGBM) and DL predictions (LSTM, TRANSFORMER).
    """
    if not data_manager.ready:
        raise HTTPException(503, "Data not loaded yet")

    week = req.week
    if week not in registry.weeks:
        raise HTTPException(400, f"Invalid week {week}. Choose from {registry.weeks}")

    student_idx = req.student_index
    if student_idx < 0 or student_idx >= data_manager.n_students:
        raise HTTPException(404, f"Student index {student_idx} out of range")

    # --- ML predictions ---
    X_row = data_manager.get_ml_row(student_idx, week)
    ml_preds = {}
    if X_row is not None:
        raw = registry.predict_ml(week, X_row)
        for model_name, probs in raw.items():
            p = float(probs[0])
            ml_preds[model_name] = {"probability": round(p, 4), "label": _label(p)}

    # --- DL predictions (simulate cutoff at given week) ---
    dl_preds = {}
    raw_dl = registry.predict_dl_at_week(
        student_idx, week,
        data_manager.x_seq, data_manager.x_static, data_manager.mask,
    )
    for model_name, prob in raw_dl.items():
        p = float(prob)
        dl_preds[model_name] = {"probability": round(p, 4), "label": _label(p)}

    return {
        "student": data_manager.get_student_info(student_idx),
        "week": week,
        "ml_predictions": ml_preds,
        "dl_predictions": dl_preds,
    }


@router.post("/shap")
def shap_explain(req: ShapRequest):
    """Compute SHAP waterfall data for XGBoost or LightGBM at a given week."""
    if not data_manager.ready:
        raise HTTPException(503, "Data not loaded yet")

    week = req.week
    if week not in registry.weeks:
        raise HTTPException(400, f"Invalid week {week}. Choose from {registry.weeks}")

    model_name = req.model
    if model_name not in ("XGBoost", "LightGBM"):
        raise HTTPException(400, "model must be 'XGBoost' or 'LightGBM'")

    # Map UI name → internal key
    key_map = {"XGBoost": "XGBoost", "LightGBM": "LGBM"}
    model_key = f"{key_map[model_name]}_w{week}"
    model = registry.ml_models.get(model_key)
    if model is None:
        raise HTTPException(404, f"Model {model_key} not loaded")

    student_idx = req.student_index
    X_row = data_manager.get_ml_row(student_idx, week)
    if X_row is None:
        raise HTTPException(404, f"No data for student {student_idx} at week {week}")

    feature_names = data_manager.get_ml_feature_names()
    result = ShapExplainer.explain(model_name, week, student_idx, model, X_row, feature_names)

    if "error" in result:
        raise HTTPException(500, f"SHAP error: {result['error']}")

    result["student"] = data_manager.get_student_info(student_idx)
    return result
