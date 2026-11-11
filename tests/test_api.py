"""Tests for the FastAPI REST endpoints in carbon_automl.api.main."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier

from carbon_automl.artifacts.carbon_passport import CarbonPassport
from carbon_automl.artifacts.model_registry import ModelRegistry


@pytest.fixture
def api_client_with_model(tmp_path: Path):
    """Provide a TestClient backed by an isolated tmp_path ModelRegistry with 1 model."""
    from carbon_automl.api.main import app

    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Train & save a small deterministic model
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y = np.array([0, 0, 1, 1])
    clf = DummyClassifier(strategy="most_frequent").fit(X, y)
    artifact_path = models_dir / "dummy_model.joblib"
    joblib.dump(clf, artifact_path)

    # Create & save CarbonPassport
    cp = CarbonPassport()
    doc = cp.generate(
        model_name="dummy",
        dataset_fingerprint={"sha256": "test_sha", "rows": 4, "cols": 2},
        feature_schema={"f1": "float64", "f2": "float64"},
        training_config={"model_type": "dummy"},
        hardware_info={"cpu_count": 2, "ram_gb": 8.0},
        training_measurements={
            "training_duration_seconds": 0.01,
            "training_energy_kwh": 1e-6,
            "training_co2_kg": 7e-7,
            "is_estimated": True,
            "carbon_intensity_kg_per_kwh": 0.7,
        },
        inference_measurements={
            "inference_energy_per_1k_predictions_kwh": 1e-7,
            "inference_latency_ms_p50": 0.1,
            "inference_latency_ms_p99": 0.2,
        },
        predictive_metrics={"macro_f1": 0.67},
        validation_methodology="holdout",
        known_limitations=["Test fixture model"],
    )
    passport_path = Path(cp.export_json(doc, str(models_dir / "dummy_passport.json")))

    registry = ModelRegistry(str(models_dir))
    model_id = registry.register(
        model_name="dummy",
        artifact_path=str(artifact_path),
        metrics={"macro_f1": 0.67},
        carbon_passport_path=str(passport_path),
        experiment_hash="api_test_hash",
    )

    prev_dir = getattr(app.state, "models_dir", None)
    app.state.models_dir = str(models_dir)
    try:
        with TestClient(app) as client:
            yield client, model_id
    finally:
        if prev_dir is None:
            if hasattr(app.state, "models_dir"):
                delattr(app.state, "models_dir")
        else:
            app.state.models_dir = prev_dir


def test_health_endpoint(api_client_with_model) -> None:
    """GET /health returns status ok, version, and timestamp."""
    client, _ = api_client_with_model
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "timestamp" in data


def test_list_and_get_models(api_client_with_model) -> None:
    """GET /models and GET /models/{model_id} return registered model metadata."""
    client, model_id = api_client_with_model

    resp_list = client.get("/models")
    assert resp_list.status_code == 200
    models = resp_list.json()
    assert len(models) == 1
    assert models[0]["registry_id"] == model_id

    resp_one = client.get(f"/models/{model_id}")
    assert resp_one.status_code == 200
    assert resp_one.json()["model_name"] == "dummy"

    resp_missing = client.get("/models/does-not-exist")
    assert resp_missing.status_code == 404


def test_carbon_report_endpoint(api_client_with_model) -> None:
    """GET /carbon-report/{model_id} returns the Carbon Passport JSON."""
    client, model_id = api_client_with_model
    resp = client.get(f"/carbon-report/{model_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_name"] == "dummy"
    assert data["dataset_hash"] == "test_sha"

    assert client.get("/carbon-report/unknown-id").status_code == 404


def test_predict_endpoint_and_honesty_disclaimer(api_client_with_model) -> None:
    """POST /predict returns prediction and explicitly marks energy as an estimate."""
    client, model_id = api_client_with_model
    resp = client.post(
        "/predict",
        json={"model_id": model_id, "features": {"f1": 1.5, "f2": 2.5}},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_id"] == model_id
    assert data["prediction"] in (0, 1)
    assert data["estimated_inference_energy_joules"] >= 0
    assert data["estimated_co2_kg"] >= 0
    assert data["is_real_time_hardware_measurement"] is False
    assert "not a real-time hardware power measurement" in data["methodology"].lower()


def test_predict_error_handling(api_client_with_model) -> None:
    """POST /predict returns 404 for unknown model and 422 for invalid empty features."""
    client, model_id = api_client_with_model

    resp_404 = client.post(
        "/predict",
        json={"model_id": "nonexistent", "features": {"f1": 1.0}},
    )
    assert resp_404.status_code == 404

    resp_422 = client.post(
        "/predict",
        json={"model_id": model_id, "features": {}},
    )
    assert resp_422.status_code == 422


def test_benchmark_missing_config_returns_404(api_client_with_model, tmp_path: Path) -> None:
    """POST /benchmark returns 404 when the requested config_path does not exist."""
    client, _ = api_client_with_model
    resp = client.post(
        "/benchmark",
        json={"config_path": str(tmp_path / "missing_config.yaml")},
    )
    assert resp.status_code == 404


def test_project_root_resolves_to_repo_root() -> None:
    """carbon_automl.api.main.PROJECT_ROOT points to the repository root containing config.yaml."""
    from carbon_automl.api.main import PROJECT_ROOT

    assert PROJECT_ROOT == Path(__file__).resolve().parent.parent
    assert (PROJECT_ROOT / "config.yaml").exists()


def test_predict_wrapped_dict_with_label_encoder_and_422(tmp_path: Path) -> None:
    """Wrapped dict artifact applies label_encoder.inverse_transform and validates missing features with 422."""
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from carbon_automl.api.main import app
    from carbon_automl.core.preprocessing import UniversalPreprocessor

    models_dir = tmp_path / "wrapped_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "num_1": [1.0, 2.0, 10.0, 11.0, 20.0, 21.0],
        "cat_1": ["north", "south", "north", "south", "east", "east"],
    })
    labels = ["Normal", "Normal", "Moderate", "Moderate", "Severe", "Severe"]
    le = LabelEncoder()
    y_enc = le.fit_transform(labels)

    prep = UniversalPreprocessor()
    X_arr = prep.fit_transform(df, {"num_1": "numerical", "cat_1": "categorical"})
    clf = DummyClassifier(strategy="most_frequent").fit(X_arr, y_enc)

    artifact_path = models_dir / "wrapped.joblib"
    joblib.dump(
        {
            "model": clf,
            "preprocessor": prep,
            "feature_names": ["num_1", "cat_1"],
            "label_encoder": le,
        },
        artifact_path,
    )

    registry = ModelRegistry(str(models_dir))
    model_id = registry.register(
        model_name="wrapped_dummy",
        artifact_path=str(artifact_path),
        metrics={"macro_f1": 0.5},
        carbon_passport_path="",
        experiment_hash="wrapped_hash",
    )

    prev_dir = getattr(app.state, "models_dir", None)
    app.state.models_dir = str(models_dir)
    try:
        with TestClient(app) as client:
            ok_resp = client.post(
                "/predict",
                json={"model_id": model_id, "features": {"num_1": 1.5, "cat_1": "north"}},
            )
            assert ok_resp.status_code == 200
            assert ok_resp.json()["prediction"] in {"Normal", "Moderate", "Severe"}

            missing_resp = client.post(
                "/predict",
                json={"model_id": model_id, "features": {"num_1": 1.5}},
            )
            assert missing_resp.status_code == 422
            assert "cat_1" in missing_resp.json()["detail"]
    finally:
        if prev_dir is None:
            if hasattr(app.state, "models_dir"):
                delattr(app.state, "models_dir")
        else:
            app.state.models_dir = prev_dir
