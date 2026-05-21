"""GET /metadata — model info, feature names, available weeks."""

from fastapi import APIRouter

from api.model_loader import registry
from api.data_manager import data_manager

router = APIRouter()


@router.get("/metadata")
def get_metadata():
    """Return model info, feature names, and weeks."""
    ml_loaded = [k for k in registry.ml_models]
    dl_loaded = list(registry.dl_models.keys())

    return {
        "weeks": registry.weeks,
        "ml_models": ["XGBoost", "LightGBM"],
        "dl_models": [name for name in dl_loaded],
        "ml_models_loaded": ml_loaded,
        "dl_models_loaded": dl_loaded,
        "feature_names": data_manager.get_ml_feature_names(),
        "feature_count": len(data_manager.get_ml_feature_names()),
        "dynamic_features": data_manager.dynamic_features,
        "static_features": data_manager.static_features,
        "n_students": data_manager.n_students if data_manager.ready else 0,
        "data_ready": data_manager.ready,
        "labels": {"0": "Not At-Risk", "1": "At-Risk"},
    }
