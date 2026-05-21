"""GET /timeline/{student_idx} — risk probability across all weeks for all 4 models."""

from fastapi import APIRouter, HTTPException

from api.model_loader import registry
from api.data_manager import data_manager

router = APIRouter()


@router.get("/timeline/{student_idx}")
def get_timeline(student_idx: int):
    """Return At-Risk probability for every week, for all 4 models.

    ML models (XGBoost, LightGBM): predict at fixed weeks using week-specific pkl.
    DL models (LSTM, TRANSFORMER): predict with data masked to each week cutoff.

    Response shape:
    {
      "student": {...},
      "weeks": [4, 8, 12, 16, 20, 24],
      "timeline": {
        "XGBoost":     [0.12, 0.25, 0.48, 0.61, 0.72, 0.80],
        "LightGBM":    [...],
        "LSTM":        [...],
        "TRANSFORMER": [...]
      }
    }
    """
    if not data_manager.ready:
        raise HTTPException(503, "Data not loaded yet")
    if student_idx < 0 or student_idx >= data_manager.n_students:
        raise HTTPException(404, f"Student index {student_idx} out of range")

    weeks = registry.weeks
    timeline: dict[str, list[float]] = {
        "XGBoost": [],
        "LightGBM": [],
        "LSTM": [],
        "TRANSFORMER": [],
    }

    for week in weeks:
        # --- ML ---
        X_row = data_manager.get_ml_row(student_idx, week)
        if X_row is not None:
            ml_raw = registry.predict_ml(week, X_row)
            timeline["XGBoost"].append(round(float(ml_raw.get("XGBoost", [0.0])[0]), 4))
            timeline["LightGBM"].append(round(float(ml_raw.get("LightGBM", [0.0])[0]), 4))
        else:
            timeline["XGBoost"].append(None)
            timeline["LightGBM"].append(None)

        # --- DL ---
        dl_raw = registry.predict_dl_at_week(
            student_idx, week,
            data_manager.x_seq, data_manager.x_static, data_manager.mask,
        )
        timeline["LSTM"].append(round(float(dl_raw.get("LSTM", 0.0)), 4))
        timeline["TRANSFORMER"].append(round(float(dl_raw.get("TRANSFORMER", 0.0)), 4))

    # Remove models that have no data at all
    timeline = {k: v for k, v in timeline.items() if any(x is not None for x in v)}

    return {
        "student": data_manager.get_student_info(student_idx),
        "weeks": weeks,
        "timeline": timeline,
    }
