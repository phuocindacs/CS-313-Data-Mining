"""GET /metrics — evaluation metrics (AUC, F1, Precision, Recall, Accuracy, Confusion Matrix)
for each model at each week, computed on the test set."""

from fastapi import APIRouter
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix,
)

from api.model_loader import registry
from api.data_manager import data_manager

router = APIRouter()

_cache: dict | None = None


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "auc_roc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "confusion_matrix": {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)},
    }


def _build_metrics() -> dict:
    global _cache
    if _cache is not None:
        return _cache

    y_true = data_manager.y_test.astype(int)
    n = len(y_true)
    results = {"ml": {}, "dl": {}}

    for w in registry.weeks:
        snap = data_manager._week_snapshots.get(w)
        if snap is None:
            continue

        X = snap[data_manager.ml_feature_cols].values.astype(np.float32)
        ml_preds = registry.predict_ml(w, X)

        for model_name, probs in ml_preds.items():
            key = f"{model_name}_w{w}"
            results["ml"][key] = {
                "model": model_name,
                "week": w,
                **_compute_metrics(y_true, probs),
            }

    for model_name in registry.dl_models:
        for w in registry.weeks:
            mask_w = data_manager.mask.copy()
            mask_w[:, w:] = True
            dl_preds = registry.predict_dl(
                data_manager.x_seq, data_manager.x_static, mask_w,
            )
            if model_name in dl_preds:
                display = "Transformer" if model_name == "TRANSFORMER" else model_name
                key = f"{display}_w{w}"
                results["dl"][key] = {
                    "model": display,
                    "week": w,
                    **_compute_metrics(y_true, dl_preds[model_name]),
                }

    at_risk = int(np.sum(y_true == 1))
    not_at_risk = int(np.sum(y_true == 0))
    results["class_distribution"] = {
        "total": n,
        "at_risk": at_risk,
        "not_at_risk": not_at_risk,
        "at_risk_pct": round(at_risk / n * 100, 1),
        "not_at_risk_pct": round(not_at_risk / n * 100, 1),
    }

    _cache = results
    return results


@router.get("/metrics")
def get_metrics():
    if not data_manager.ready:
        return {"error": "Data not loaded yet"}
    try:
        return _build_metrics()
    except Exception as e:
        return {"error": str(e)}
