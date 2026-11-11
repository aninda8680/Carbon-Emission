"""
FastAPI REST service for the Carbon-Aware AutoML Framework.

Endpoints:
    POST /predict                Run inference with a registered model + energy estimate
    POST /benchmark              Trigger an asynchronous AutoML benchmark run
    GET  /models                 List all registered models
    GET  /models/{model_id}      Get metadata and metrics for a specific model
    GET  /carbon-report/{model_id} Get the Carbon Passport JSON for a model
    GET  /health                 Health check endpoint
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from carbon_automl import __version__
from carbon_automl.artifacts.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
_DEFAULT_EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
_JOULES_PER_KWH = 3_600_000.0
_DEFAULT_CARBON_INTENSITY = 0.7  # kg CO2e / kWh (India national average)
_ASSUMED_WATTS_PER_CORE = 10.0

# Track background benchmark runs in memory
_BENCHMARK_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# App & Middleware
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Carbon-Aware AutoML API",
    description="REST API for model inference, registry inspection, and Carbon Passports.",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_models_dir(request: Request | None = None) -> Path:
    """Resolve the models registry directory (env / app.state / default)."""
    if request is not None and hasattr(request.app.state, "models_dir"):
        return Path(request.app.state.models_dir)
    env_dir = os.environ.get("CARBON_AUTOML_MODELS_DIR")
    if env_dir:
        return Path(env_dir)
    return _DEFAULT_MODELS_DIR


def _get_registry(request: Request | None = None) -> ModelRegistry:
    return ModelRegistry(str(_get_models_dir(request)))


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _to_json_scalar(value: Any) -> Any:
    """Convert NumPy scalars / arrays to plain JSON-serialisable Python types."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _run_model_prediction(artifact: Any, features: dict[str, Any], metadata: dict[str, Any]) -> Any:
    """Run prediction supporting pipelines, wrapped dicts, and raw estimators."""
    df = pd.DataFrame([features])

    # Handle wrapped dict artifacts: {"preprocessor": ..., "model": ..., "feature_names": ...}
    if isinstance(artifact, dict) and "model" in artifact:
        model = artifact["model"]
        preprocessor = artifact.get("preprocessor")
        feature_names = artifact.get("feature_names")
        if feature_names:
            missing = [c for c in feature_names if c not in df.columns]
            if missing:
                raise HTTPException(
                    status_code=422,
                    detail=f"Missing required feature(s): {missing}",
                )
            df = df[list(feature_names)]
        X_in = preprocessor.transform(df) if preprocessor is not None else df
        preds = model.predict(X_in)
        le = artifact.get("label_encoder")
        if le is not None:
            preds = le.inverse_transform(preds)
        return _to_json_scalar(preds[0])

    # Standard sklearn estimator or Pipeline
    feature_names = metadata.get("feature_names")
    if feature_names and isinstance(feature_names, list):
        missing = [c for c in feature_names if c not in df.columns]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Missing required feature(s): {missing}",
            )
        df = df[feature_names]

    try:
        preds = artifact.predict(df)
    except Exception:
        # Fallback for estimators fitted on positional NumPy arrays without column names
        preds = artifact.predict(df.to_numpy())

    return _to_json_scalar(preds[0])


def _launch_benchmark_worker(experiment_id: str, config_path: Path) -> None:
    """Execute the training pipeline in a background daemon thread."""
    with _JOBS_LOCK:
        _BENCHMARK_JOBS[experiment_id]["status"] = "running"

    try:
        import sys
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from main import load_config, run_pipeline

        cfg = load_config(str(config_path))
        results = run_pipeline(cfg, benchmark_mode=True)
        with _JOBS_LOCK:
            _BENCHMARK_JOBS[experiment_id].update({
                "status": "completed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "model_id": results.get("model_id"),
            })
    except Exception as exc:
        logger.exception("Background benchmark %s failed: %s", experiment_id, exc)
        with _JOBS_LOCK:
            _BENCHMARK_JOBS[experiment_id].update({
                "status": "failed",
                "error": str(exc),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            })


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    model_id: str = Field(..., min_length=1, description="Registered model UUID.")
    features: dict[str, Any] = Field(..., description="Mapping of feature_name -> value.")

    @field_validator("features")
    @classmethod
    def _non_empty_features(cls, v: dict[str, Any]) -> dict[str, Any]:
        if not v:
            raise ValueError("'features' dictionary must not be empty.")
        return v


class PredictResponse(BaseModel):
    prediction: Any
    model_id: str
    estimated_inference_energy_joules: float
    estimated_co2_kg: float
    methodology: str
    uncertainty_pct: float
    is_real_time_hardware_measurement: bool = False


class BenchmarkRequest(BaseModel):
    config_path: str = Field(..., min_length=1, description="Path to YAML config file.")


class BenchmarkResponse(BaseModel):
    experiment_id: str
    status: str


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Service health check."""
    return HealthResponse(
        status="ok",
        version=__version__,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/models")
def list_models(request: Request) -> list[dict[str, Any]]:
    """Return all registered models with metadata."""
    registry = _get_registry(request)
    return registry.list_models()


@app.get("/models/{model_id}")
def get_model(model_id: str, request: Request) -> dict[str, Any]:
    """Return full metadata and metrics for a specific registered model."""
    registry = _get_registry(request)
    try:
        return registry.get_model(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.") from exc


@app.get("/carbon-report/{model_id}")
def get_carbon_report(model_id: str, request: Request) -> dict[str, Any]:
    """Return the Carbon Passport JSON document for a registered model."""
    registry = _get_registry(request)
    try:
        meta = registry.get_model(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.") from exc

    passport_path_str = meta.get("carbon_passport_path", "")
    if not passport_path_str:
        raise HTTPException(
            status_code=404,
            detail=f"No Carbon Passport path recorded for model '{model_id}'.",
        )

    passport_path = _resolve_path(passport_path_str)
    if not passport_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Carbon Passport file not found at '{passport_path}'.",
        )

    try:
        return json.loads(passport_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read Carbon Passport: {exc}",
        ) from exc


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    """Run inference with a registered model and return an honest energy estimate."""
    registry = _get_registry(request)
    try:
        meta = registry.get_model(payload.model_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{payload.model_id}' not found.",
        ) from exc

    artifact_path = _resolve_path(meta.get("artifact_path", ""))
    if not artifact_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model artifact file missing on disk: '{artifact_path}'.",
        )

    try:
        artifact = joblib.load(artifact_path)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model artifact: {exc}",
        ) from exc

    t0 = time.perf_counter()
    try:
        prediction = _run_model_prediction(artifact, payload.features, meta)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Inference failed for provided features: {exc}",
        ) from exc
    elapsed_s = max(time.perf_counter() - t0, 1e-6)

    # Duration-based order-of-magnitude estimate (NOT a real-time hardware measurement).
    cpu_count = os.cpu_count() or 1
    estimated_watts = _ASSUMED_WATTS_PER_CORE * cpu_count
    estimated_energy_joules = estimated_watts * elapsed_s
    estimated_energy_kwh = estimated_energy_joules / _JOULES_PER_KWH

    carbon_intensity = _DEFAULT_CARBON_INTENSITY
    passport_path_str = meta.get("carbon_passport_path", "")
    if passport_path_str:
        pp = _resolve_path(passport_path_str)
        if pp.exists():
            try:
                passport = json.loads(pp.read_text(encoding="utf-8"))
                carbon_intensity = float(
                    passport.get("carbon_intensity_assumption", {}).get(
                        "value_kg_per_kwh", _DEFAULT_CARBON_INTENSITY
                    )
                )
            except Exception:
                pass

    estimated_co2_kg = estimated_energy_kwh * carbon_intensity

    methodology = (
        f"Duration-based order-of-magnitude estimate: elapsed_time ({elapsed_s:.6f}s) "
        f"x assumed CPU power ({estimated_watts:.1f}W across {cpu_count} cores) "
        f"x grid carbon intensity ({carbon_intensity} kg CO2e/kWh). "
        "DISCLAIMER: This energy estimate is NOT a real-time hardware power measurement; "
        "single-sample inference is dominated by Python/OS overhead (±100% uncertainty)."
    )

    return PredictResponse(
        prediction=prediction,
        model_id=payload.model_id,
        estimated_inference_energy_joules=estimated_energy_joules,
        estimated_co2_kg=estimated_co2_kg,
        methodology=methodology,
        uncertainty_pct=100.0,
        is_real_time_hardware_measurement=False,
    )


@app.post("/benchmark", response_model=BenchmarkResponse)
def start_benchmark(payload: BenchmarkRequest) -> BenchmarkResponse:
    """Start an asynchronous AutoML benchmark run for the given config file."""
    config_path = _resolve_path(payload.config_path)
    if not config_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Config file not found: '{config_path}'.",
        )

    experiment_id = str(uuid.uuid4())
    with _JOBS_LOCK:
        _BENCHMARK_JOBS[experiment_id] = {
            "experiment_id": experiment_id,
            "config_path": str(config_path),
            "status": "started",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

    thread = threading.Thread(
        target=_launch_benchmark_worker,
        args=(experiment_id, config_path),
        daemon=True,
    )
    thread.start()

    return BenchmarkResponse(experiment_id=experiment_id, status="started")
