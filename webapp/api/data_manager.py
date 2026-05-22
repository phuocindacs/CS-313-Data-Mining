"""Load and cache test data: long-format CSV for ML weekly prediction, tensors for DL."""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


class DataManager:
    """Holds all test data in memory. Loaded once at startup."""

    def __init__(self):
        self.df_long: pd.DataFrame = None       # (N*T, 63) — all weeks
        self.df_display: pd.DataFrame = None    # (N, ~20) — unscaled, for UI display
        self.x_seq: np.ndarray = None           # (N, 25, 33) — DL input
        self.x_static: np.ndarray = None        # (N, 25) — DL static input
        self.mask: np.ndarray = None            # (N, 25) bool — padded weeks
        self.y_test: np.ndarray = None          # (N,) — ground truth labels

        self.dynamic_features: list = []
        self.static_features: list = []
        self.ml_feature_cols: list = []         # 58 features for ML weekly models
        self.weeks: list = [4, 8, 12, 16, 20, 24]

        # StandardScaler objects (fit on train) used to undo the scaling baked into
        # df_test_long so the tree models receive raw features, as they were trained.
        self.scaler_seq = None                  # 33 dynamic features
        self.scaler_static = None               # 25 static features

        # Pre-filtered snapshots per week: {week: DataFrame of (N, 58)}
        self._week_snapshots: dict = {}
        self.ready: bool = False

    def load(self, cfg: dict, base_dir: Path):
        """Load all data from config. base_dir = webapp/ directory."""
        data_cfg = cfg["data"]

        # Feature metadata
        detail_path = (base_dir / data_cfg["feature_detail"]).resolve()
        with open(detail_path) as f:
            meta = json.load(f)
        self.dynamic_features = meta["dynamic_features"]
        self.static_features = meta["static_features"]
        self.ml_feature_cols = self.dynamic_features + self.static_features

        # Scalers to convert the (scaled) long CSV back to raw for the tree models.
        self.scaler_seq = self._load_scaler(
            base_dir, data_cfg.get("scaler_seq"), len(self.dynamic_features), "scaler_seq")
        self.scaler_static = self._load_scaler(
            base_dir, data_cfg.get("scaler_static"), len(self.static_features), "scaler_static")

        # Long format CSV — used for ML weekly prediction
        long_path = (base_dir / data_cfg["test_long"]).resolve()
        print(f"  Loading {long_path.name} ...")
        self.df_long = pd.read_csv(long_path)

        # Display CSV — unscaled, for UI
        display_path = (base_dir / data_cfg["test_display"]).resolve()
        print(f"  Loading {display_path.name} ...")
        self.df_display = pd.read_csv(display_path)

        # Numpy tensors — used for DL prediction
        print("  Loading tensors ...")
        self.x_seq = np.load((base_dir / data_cfg["x_seq_test"]).resolve())
        self.x_static = np.load((base_dir / data_cfg["x_static_test"]).resolve())
        self.mask = np.load((base_dir / data_cfg["mask_test"]).resolve())
        self.y_test = np.load((base_dir / data_cfg["y_test"]).resolve())

        # The DL tensors (x_seq/x_static/mask/y_test) were saved sorted by these keys,
        # but the CSVs are sorted by id_student first — a different order. Reorder the
        # CSVs to the tensor order so that student index i points to the SAME student in
        # the ML features, the displayed id, the DL tensors, and the ground-truth label.
        order_keys = ["code_module", "code_presentation", "id_student"]
        if all(k in self.df_long.columns for k in order_keys):
            self.df_long = self.df_long.sort_values(order_keys + ["week"]).reset_index(drop=True)
        if all(k in self.df_display.columns for k in order_keys):
            self.df_display = self.df_display.sort_values(order_keys).reset_index(drop=True)
        # Guard: ground-truth order must match the reordered CSV, else indices are off.
        if "target" in self.df_long.columns:
            csv_targets = self.df_long.drop_duplicates(order_keys)["target"].to_numpy()
            if np.array_equal(csv_targets, self.y_test):
                print("  ✓ ML/DL student order aligned")
            else:
                print("  ! WARNING: CSV order != y_test.npy — student alignment may be off")

        # Pre-build week snapshots for fast ML inference. df_test_long is StandardScaled
        # (from the sequence FE pipeline), but the tree models were trained on raw
        # features — so undo the scaling here before they reach the models.
        print("  Pre-building week snapshots ...")
        for w in cfg["ml"]["weeks"]:
            snap = self.df_long[self.df_long["week"] == w].copy()
            # Drop non-feature columns
            drop = [c for c in ["id_student", "code_module", "code_presentation", "week", "target"]
                    if c in snap.columns]
            snap = snap.drop(columns=drop).reset_index(drop=True)
            if self.scaler_seq is not None:
                snap[self.dynamic_features] = self.scaler_seq.inverse_transform(
                    snap[self.dynamic_features].values)
            if self.scaler_static is not None:
                snap[self.static_features] = self.scaler_static.inverse_transform(
                    snap[self.static_features].values)
            self._week_snapshots[w] = snap

        self.n_students = len(self.x_seq)
        self.ready = True
        print(f"  Data ready — {self.n_students} students")

    @staticmethod
    def _load_scaler(base_dir: Path, rel_path: str, expected_n_features: int, name: str):
        """Load a saved StandardScaler, returning None (with a warning) if missing or
        mismatched so startup never crashes — features are then left as-is."""
        if not rel_path:
            print(f"  ! {name} not configured — ML features left scaled")
            return None
        path = (base_dir / rel_path).resolve()
        try:
            with open(path, "rb") as f:
                scaler = pickle.load(f)
        except Exception as e:
            print(f"  ! could not load {name} ({path.name}): {e} — ML features left scaled")
            return None
        n = getattr(scaler, "n_features_in_", None)
        if n is not None and n != expected_n_features:
            print(f"  ! {name} expects {n} features but {expected_n_features} configured "
                  f"— skipping inverse-transform")
            return None
        print(f"  ✓ {name} loaded")
        return scaler

    # ------------------------------------------------------------------
    # Student info helpers
    # ------------------------------------------------------------------

    def get_student_list(self) -> list:
        """Return list of {index, student_id, target, actual_label} for dropdown."""
        df = self.df_display
        students = []
        for i in range(self.n_students):
            row = df.iloc[i] if i < len(df) else {}
            sid = int(row.get("id_student", i))
            target = int(self.y_test[i])
            # Ground-truth outcome shown with the same risk vocabulary as predictions
            # (UI prefixes "Actual:" so it stays distinguishable from a model prediction)
            label = "At-Risk" if target == 1 else "Not At-Risk"
            students.append({
                "index": i,
                "student_id": sid,
                "target": target,
                "actual_label": label,
            })
        return students

    def get_student_info(self, idx: int) -> dict:
        """Return display info for a single student."""
        if idx >= self.n_students:
            return {}
        row = self.df_display.iloc[idx] if idx < len(self.df_display) else {}
        target = int(self.y_test[idx])
        return {
            "index": idx,
            "student_id": int(row.get("id_student", idx)),
            "code_module": str(row.get("code_module", "")),
            "code_presentation": str(row.get("code_presentation", "")),
            "target": target,
            "actual_label": "At-Risk" if target == 1 else "Not At-Risk",
        }

    # ------------------------------------------------------------------
    # Data access for prediction
    # ------------------------------------------------------------------

    def get_ml_row(self, student_idx: int, week: int) -> np.ndarray:
        """Return (1, 58) feature array for ML prediction at given week."""
        snap = self._week_snapshots.get(week)
        if snap is None or student_idx >= len(snap):
            return None
        return snap.iloc[student_idx][self.ml_feature_cols].values.reshape(1, -1).astype(np.float32)

    def get_ml_feature_names(self) -> list:
        return self.ml_feature_cols


# Singleton — populated at API startup
data_manager = DataManager()
