"""Captum Integrated Gradients explainer for the DL sequence models (LSTM, Transformer).

TreeExplainer is tree-only and DeepExplainer is unreliable here: these models take a
(T, n_dynamic) sequence + a static vector + a boolean pad/cutoff mask, and output a
single risk logit. Integrated Gradients is deterministic, fast for a single student,
and is run on the SAME masked-cutoff input used for prediction — so the attribution
explains exactly the probability shown for that cutoff week.
"""

import numpy as np
import torch
from captum.attr import IntegratedGradients


class DLExplainer:
    """Compute Integrated Gradients attributions for one student at a cutoff week."""

    # One IntegratedGradients per model object (cheap to build, mirrors ShapExplainer cache).
    _cache: dict = {}

    @staticmethod
    def _forward_prob(x_seq, x_static, mask, model):
        """At-Risk probability per sample — the scalar quantity we attribute."""
        return torch.sigmoid(model(x_seq, x_static, mask))

    @staticmethod
    def _get_ig(model_name: str, model):
        if model_name not in DLExplainer._cache:
            DLExplainer._cache[model_name] = IntegratedGradients(
                lambda xs, xt, m: DLExplainer._forward_prob(xs, xt, m, model)
            )
        return DLExplainer._cache[model_name]

    @staticmethod
    def explain(model_name, week, student_idx, model,
                x_seq_all, x_static_all, mask_all,
                feature_names, n_steps: int = 50) -> dict:
        """Return per-week importance + signed per-feature attributions for one student.

        Args mirror registry.predict_dl_at_week so the explained input is the same
        masked sequence the model predicts on at this cutoff week.

        Returns dict with feature_attributions (n_dynamic + n_static, signed),
        week_importance (one per valid week), weeks_axis, and the prediction.
        """
        try:
            # Cutoff mask: weeks >= week are hidden (matches model_loader masking).
            cutoff_mask = mask_all[student_idx:student_idx + 1].copy()
            cutoff_mask[:, week:] = True

            x_seq = torch.tensor(x_seq_all[student_idx:student_idx + 1], dtype=torch.float32)
            x_static = torch.tensor(x_static_all[student_idx:student_idx + 1], dtype=torch.float32)
            t_mask = torch.tensor(cutoff_mask, dtype=torch.bool)

            model.eval()  # dropout off; gradients still flow (no torch.no_grad here)

            # Prediction at the real input — for the header and as a sanity check.
            with torch.no_grad():
                prob = float(DLExplainer._forward_prob(x_seq, x_static, t_mask, model)[0])

            ig = DLExplainer._get_ig(model_name, model)
            # Zero baseline ≈ population mean in StandardScaler-scaled space ("average student").
            attr_seq, attr_static = ig.attribute(
                inputs=(x_seq, x_static),
                baselines=(torch.zeros_like(x_seq), torch.zeros_like(x_static)),
                additional_forward_args=(t_mask,),
                n_steps=n_steps,
            )

            attr_seq = attr_seq.detach().numpy()[0]        # (T, n_dynamic)
            attr_static = attr_static.detach().numpy()[0]  # (n_static,)

            # Per-week importance over valid (unmasked) weeks only.
            week_imp = np.abs(attr_seq[:week]).sum(axis=1)        # (week,)
            # Signed per-feature attribution: dynamic summed over valid weeks, then static.
            dyn_attr = attr_seq[:week].sum(axis=0)               # (n_dynamic,)
            feat_attr = np.concatenate([dyn_attr, attr_static])  # (n_dynamic + n_static,)

            return {
                "model": model_name,
                "week": week,
                "method": "Integrated Gradients (Captum)",
                "student_index": student_idx,
                "feature_names": feature_names,
                "feature_attributions": feat_attr.tolist(),
                "week_importance": week_imp.tolist(),
                "weeks_axis": list(range(1, week + 1)),
                "prediction": prob,
            }
        except Exception as e:
            return {"error": str(e)}
