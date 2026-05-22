"""Load and manage ML weekly models (XGBoost, LGBM) and DL sequence models (LSTM, Transformer)."""

import json
import joblib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# PyTorch model class definitions (must match training architecture exactly)
# ---------------------------------------------------------------------------

class EWSLSTM(nn.Module):
    """LSTM-based EWS model: takes sequence + static features, outputs risk logit."""

    def __init__(self, n_dynamic, n_static, hidden=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_dynamic, hidden_size=hidden,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.static_proj = nn.Sequential(
            nn.Linear(n_static, hidden), nn.ReLU(), nn.Dropout(dropout)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, x_seq, x_static, mask):
        out, _ = self.lstm(x_seq)
        lengths = (~mask).sum(dim=1).clamp(min=1) - 1
        idx = lengths.view(-1, 1, 1).expand(-1, 1, out.size(-1))
        h_last = out.gather(1, idx).squeeze(1)
        s = self.static_proj(x_static)
        logits = self.head(torch.cat([h_last, s], dim=1)).squeeze(-1)
        return logits


class EWSTransformer(nn.Module):
    """Transformer-based EWS model: takes sequence + static features, outputs risk logit."""

    def __init__(self, n_dynamic, n_static, d_model=64, nhead=4, nlayers=2,
                 dropout=0.3, max_len=64):
        super().__init__()
        self.input_proj = nn.Linear(n_dynamic, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.static_proj = nn.Sequential(
            nn.Linear(n_static, d_model), nn.ReLU(), nn.Dropout(dropout)
        )
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def forward(self, x_seq, x_static, mask):
        B, T, _ = x_seq.shape
        pos = torch.arange(T, device=x_seq.device).unsqueeze(0).expand(B, T)
        h = self.input_proj(x_seq) + self.pos_emb(pos)
        h = self.encoder(h, src_key_padding_mask=mask)
        lengths = (~mask).sum(dim=1).clamp(min=1) - 1
        idx = lengths.view(-1, 1, 1).expand(-1, 1, h.size(-1))
        h_last = h.gather(1, idx).squeeze(1)
        s = self.static_proj(x_static)
        logits = self.head(torch.cat([h_last, s], dim=1)).squeeze(-1)
        return logits


# ---------------------------------------------------------------------------
# ModelRegistry: loads and holds all models
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Central registry holding all loaded ML and DL models."""

    def __init__(self):
        self.ml_models: dict = {}   # {"XGBoost_w4": model, "LGBM_w4": model, ...}
        self.dl_models: dict = {}   # {"LSTM": model, "TRANSFORMER": model}
        self.device = torch.device("cpu")  # always CPU for inference stability
        self.n_dynamic: int = 0
        self.n_static: int = 0
        self.t_max: int = 25
        self.weeks: list = [4, 8, 12, 16, 20, 24]
        self.dynamic_features: list = []
        self.static_features: list = []

    def load_all(self, cfg: dict, base_dir: Path):
        """Load all models from config. base_dir = webapp/ directory."""
        # Read feature dimensions from feature_meta.json — single source of truth.
        # Run run_sequence_fe.py to regenerate if dimensions change after retraining.
        meta_path = base_dir / "data" / "feature_meta.json"
        with open(meta_path) as f:
            _meta = json.load(f)
        self.dynamic_features = _meta["DYNAMIC_FEATURES"]
        self.static_features  = _meta["STATIC_FEATURES"]
        self.n_dynamic = len(self.dynamic_features)
        self.n_static  = len(self.static_features)
        print(f"  Features: {self.n_dynamic} dynamic + {self.n_static} static (from feature_meta.json)")

        self._load_ml_models(cfg["ml"], base_dir)
        self._load_dl_models(cfg["dl"], base_dir)

    def _load_ml_models(self, ml_cfg: dict, base_dir: Path):
        self.weeks = ml_cfg["weeks"]
        xgb_pat = ml_cfg["xgboost_pattern"]
        lgbm_pat = ml_cfg["lgbm_pattern"]

        for w in self.weeks:
            xgb_path = (base_dir / xgb_pat.replace("{w}", str(w))).resolve()
            lgbm_path = (base_dir / lgbm_pat.replace("{w}", str(w))).resolve()

            try:
                self.ml_models[f"XGBoost_w{w}"] = joblib.load(xgb_path)
                print(f"  ✓ XGBoost week {w}")
            except Exception as e:
                print(f"  ✗ XGBoost week {w}: {e}")

            try:
                self.ml_models[f"LGBM_w{w}"] = joblib.load(lgbm_path)
                print(f"  ✓ LGBM week {w}")
            except Exception as e:
                print(f"  ✗ LGBM week {w}: {e}")

    def _load_dl_models(self, dl_cfg: dict, base_dir: Path):
        for model_name, info in dl_cfg.items():
            params_path = (base_dir / info["params"]).resolve()
            model_path = (base_dir / info["path"]).resolve()

            try:
                with open(params_path) as f:
                    params = json.load(f)

                # Load state dict first — needed for max_len inference and sanity check.
                state = torch.load(model_path, map_location=self.device, weights_only=True)

                # Sanity check: verify state_dict input dimension matches feature_meta.json.
                # Mismatches mean data files are stale — run run_sequence_fe.py to fix.
                if model_name == "lstm":
                    expected_dyn = state["lstm.weight_ih_l0"].shape[1]
                elif model_name == "transformer":
                    expected_dyn = state["input_proj.weight"].shape[1]
                else:
                    expected_dyn = self.n_dynamic
                if expected_dyn != self.n_dynamic:
                    raise RuntimeError(
                        f"{model_name.upper()} expects {expected_dyn} dynamic features but "
                        f"feature_meta.json has {self.n_dynamic}. Run run_sequence_fe.py to regenerate data."
                    )

                if model_name == "lstm":
                    model = EWSLSTM(
                        n_dynamic=self.n_dynamic,
                        n_static=self.n_static,
                        hidden=params["hidden"],
                        num_layers=params["num_layers"],
                        dropout=params["dropout"],
                    )
                elif model_name == "transformer":
                    max_len = state.get("pos_emb.weight", torch.empty(self.t_max + 8, 0)).shape[0]
                    model = EWSTransformer(
                        n_dynamic=self.n_dynamic,
                        n_static=self.n_static,
                        d_model=params["d_model"],
                        nhead=params["nhead"],
                        nlayers=params["nlayers"],
                        dropout=params["dropout"],
                        max_len=max_len,
                    )
                else:
                    print(f"  ✗ Unknown DL model type: {model_name}")
                    continue
                model.load_state_dict(state)
                model.eval()
                model.to(self.device)
                self.dl_models[model_name.upper()] = model  # "LSTM", "TRANSFORMER"
                print(f"  ✓ {model_name.upper()} loaded")

            except Exception as e:
                print(f"  ✗ {model_name}: {e}")

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def predict_ml(self, week: int, X: np.ndarray) -> dict:
        """Predict with XGBoost and LGBM at given week. X shape: (N, 58)."""
        results = {}
        for name_key, label in [("XGBoost", "XGBoost"), ("LGBM", "LightGBM")]:
            key = f"{name_key}_w{week}"
            model = self.ml_models.get(key)
            if model is None:
                continue
            try:
                probs = model.predict_proba(X)[:, 1]  # P(At-Risk)
                results[label] = probs
            except Exception as e:
                print(f"  predict_ml {key}: {e}")
        return results

    def predict_dl(self, x_seq: np.ndarray, x_static: np.ndarray,
                   mask: np.ndarray) -> dict:
        """Predict with all DL models. Returns {name: prob_array}."""
        results = {}
        with torch.no_grad():
            t_seq = torch.FloatTensor(x_seq).to(self.device)
            t_static = torch.FloatTensor(x_static).to(self.device)
            t_mask = torch.BoolTensor(mask).to(self.device)

            for name, model in self.dl_models.items():
                try:
                    logits = model(t_seq, t_static, t_mask)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    results[name] = probs
                except Exception as e:
                    print(f"  predict_dl {name}: {e}")
        return results

    def predict_dl_at_week(self, student_idx: int, week: int,
                           x_seq_all: np.ndarray, x_static_all: np.ndarray,
                           mask_all: np.ndarray) -> dict:
        """Predict DL for single student, masking data beyond given week (1-indexed cutoff).
        week=4 means show only weeks 0..3 (first 4 weeks), mask 4..24.
        """
        partial_mask = mask_all[student_idx:student_idx + 1].copy()
        partial_mask[:, week:] = True  # mask weeks >= week (0-indexed)

        x_seq = x_seq_all[student_idx:student_idx + 1]
        x_static = x_static_all[student_idx:student_idx + 1]

        raw = self.predict_dl(x_seq, x_static, partial_mask)
        return {name: float(probs[0]) for name, probs in raw.items()}


# Singleton registry — populated at API startup
registry = ModelRegistry()
