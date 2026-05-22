"""FastAPI app for OULAD Early Warning System.
Loads ML weekly models (XGBoost/LightGBM) + DL sequence models (LSTM/Transformer)
at startup, then serves prediction, SHAP, timeline, and metadata endpoints.
"""

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure webapp/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.model_loader import registry
from api.data_manager import data_manager
from api.routes import predict, students, metadata, timeline, explain

BASE_DIR = Path(__file__).resolve().parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models and data at startup."""
    print("=" * 55)
    print("OULAD Early Warning System — Starting up")
    print("=" * 55)

    config_path = BASE_DIR / "config.json"
    with open(config_path) as f:
        cfg = json.load(f)

    print("\n[1/2] Loading models...")
    registry.load_all(cfg, BASE_DIR)

    print("\n[2/2] Loading data...")
    data_manager.load(cfg, BASE_DIR)

    ml_ok = len(registry.ml_models)
    dl_ok = len(registry.dl_models)
    print("\n" + "=" * 55)
    print(f"✓ Ready — {ml_ok} ML models, {dl_ok} DL models, {data_manager.n_students} students")
    print("=" * 55)

    yield

    print("\nShutting down...")


app = FastAPI(
    title="OULAD Early Warning System",
    description="Predict student At-Risk probability using ML weekly + DL sequence models",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, tags=["Prediction"])
app.include_router(explain.router, tags=["Explainability"])
app.include_router(students.router, tags=["Students"])
app.include_router(metadata.router, tags=["Metadata"])
app.include_router(timeline.router, tags=["Timeline"])


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "ml_models": len(registry.ml_models),
        "dl_models": len(registry.dl_models),
        "data_ready": data_manager.ready,
        "n_students": data_manager.n_students if data_manager.ready else 0,
    }
