"""Load and cache test data: long-format CSV for ML weekly prediction, tensors for DL."""

import json
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

        # Pre-build week snapshots for fast ML inference
        print("  Pre-building week snapshots ...")
        for w in cfg["ml"]["weeks"]:
            snap = self.df_long[self.df_long["week"] == w].copy()
            # Drop non-feature columns
            drop = [c for c in ["id_student", "code_module", "code_presentation", "week", "target"]
                    if c in snap.columns]
            self._week_snapshots[w] = snap.drop(columns=drop).reset_index(drop=True)

        self.n_students = len(self.x_seq)
        self.ready = True
        print(f"  Data ready — {self.n_students} students")

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
            label = "Fail/Withdrawn" if target == 1 else "Pass/Distinction"
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
            "actual_label": "Fail/Withdrawn" if target == 1 else "Pass/Distinction",
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
