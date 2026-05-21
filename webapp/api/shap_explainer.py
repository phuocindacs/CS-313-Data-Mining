"""SHAP TreeExplainer for ML weekly models (XGBoost, LightGBM).
DL models are not supported — DeepExplainer is unreliable for inference demos.
"""

import numpy as np
import shap


class ShapExplainer:
    """Compute SHAP values for a single student using TreeExplainer."""

    # Cache explainers to avoid re-building on every call: {model_key: explainer}
    _cache: dict = {}

    @staticmethod
    def _get_explainer(model_key: str, model):
        if model_key not in ShapExplainer._cache:
            ShapExplainer._cache[model_key] = shap.TreeExplainer(model)
        return ShapExplainer._cache[model_key]

    @staticmethod
    def explain(model_name: str, week: int, student_idx: int,
                model, X_row: np.ndarray, feature_names: list) -> dict:
        """Compute SHAP waterfall data for one student.

        Args:
            model_name: "XGBoost" or "LightGBM"
            week: cutoff week (e.g. 4, 8, ...)
            student_idx: position in test set
            model: fitted sklearn-compatible model
            X_row: (1, n_features) scaled feature array
            feature_names: list of feature name strings

        Returns:
            dict with shap_values, base_value, feature_values, feature_names
        """
        model_key = f"{model_name}_w{week}"
        explainer = ShapExplainer._get_explainer(model_key, model)

        # Explanation object for class 1 (At-Risk)
        try:
            explanation = explainer(X_row)
            # explanation.values shape: (1, n_features, 2) for binary, or (1, n_features)
            sv = explanation.values
            bv = explanation.base_values

            # Handle multi-output (binary gives shape (1, n_feat, 2))
            if sv.ndim == 3:
                sv = sv[0, :, 1]       # class 1 (At-Risk)
                bv = float(bv[0, 1])
            else:
                sv = sv[0]
                bv = float(bv[0]) if hasattr(bv, "__len__") else float(bv)

            return {
                "model": model_name,
                "week": week,
                "student_index": student_idx,
                "shap_values": sv.tolist(),
                "base_value": bv,
                "feature_values": X_row[0].tolist(),
                "feature_names": feature_names,
            }
        except Exception as e:
            return {"error": str(e)}
