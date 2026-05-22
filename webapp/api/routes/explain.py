"""POST /explain/dl — Integrated Gradients attribution for a DL model (LSTM, Transformer).

Tree models keep using POST /shap (SHAP TreeExplainer); this route covers the
sequence models, which need gradient-based attribution instead.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.model_loader import registry
from api.data_manager import data_manager
from api.dl_explainer import DLExplainer

router = APIRouter()

DL_MODELS = ("LSTM", "TRANSFORMER")


class ExplainDLRequest(BaseModel):
    student_index: int
    week: int = 4  # one of [4, 8, 12, 16, 20, 24]
    model: str = "LSTM"  # "LSTM" or "TRANSFORMER"


@router.post("/explain/dl")
def explain_dl(req: ExplainDLRequest):
    """Compute per-week + per-feature attribution for one student at a cutoff week."""
    if not data_manager.ready:
        raise HTTPException(503, "Data not loaded yet")

    week = req.week
    if week not in registry.weeks:
        raise HTTPException(400, f"Invalid week {week}. Choose from {registry.weeks}")

    model_name = req.model.upper()
    if model_name not in DL_MODELS:
        raise HTTPException(400, f"model must be one of {DL_MODELS}")

    model = registry.dl_models.get(model_name)
    if model is None:
        raise HTTPException(404, f"DL model {model_name} not loaded")

    student_idx = req.student_index
    if student_idx < 0 or student_idx >= data_manager.n_students:
        raise HTTPException(404, f"Student index {student_idx} out of range")

    result = DLExplainer.explain(
        model_name=model_name,
        week=week,
        student_idx=student_idx,
        model=model,
        x_seq_all=data_manager.x_seq,
        x_static_all=data_manager.x_static,
        mask_all=data_manager.mask,
        feature_names=data_manager.get_ml_feature_names(),
    )

    if "error" in result:
        raise HTTPException(500, f"Explain error: {result['error']}")

    result["student"] = data_manager.get_student_info(student_idx)
    return result
