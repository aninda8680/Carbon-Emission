"""Tests for ModelRegistry, RetrainingManager caching/versioning, and CarbonPassport."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from carbon_automl.artifacts.carbon_passport import CarbonPassport
from carbon_automl.artifacts.model_registry import ModelRegistry
from carbon_automl.intelligence.retraining_manager import RetrainingManager


def _build_sample_passport() -> dict:
    return CarbonPassport().generate(
        model_name="random_forest",
        dataset_fingerprint={"sha256": "abc123", "rows": 100, "cols": 5},
        feature_schema={"num_1": "float64", "cat_1": "object"},
        training_config={"model_type": "random_forest", "n_estimators": 50},
        hardware_info={"cpu_count": 4, "ram_gb": 16.0},
        training_measurements={
            "training_duration_seconds": 0.25,
            "training_energy_kwh": 1e-5,
            "training_co2_kg": 7e-6,
            "is_estimated": True,
            "carbon_intensity_kg_per_kwh": 0.7,
        },
        inference_measurements={
            "inference_energy_per_1k_predictions_kwh": 1e-6,
            "inference_latency_ms_p50": 0.5,
            "inference_latency_ms_p99": 1.2,
        },
        predictive_metrics={"macro_f1": {"value": 0.88, "ci_lower": 0.84, "ci_upper": 0.92}},
        validation_methodology="5-fold stratified cross-validation",
        known_limitations=["Small dataset; sub-milligram CO2 estimates carry high uncertainty."],
    )


def test_register_model(tmp_path: Path) -> None:
    """Registered model appears in list_models and is retrievable by ID."""
    registry = ModelRegistry(str(tmp_path / "registry"))
    reg_id = registry.register(
        model_name="random_forest",
        artifact_path=str(tmp_path / "rf.joblib"),
        metrics={"macro_f1": 0.89},
        carbon_passport_path=str(tmp_path / "rf_passport.json"),
        experiment_hash="hash_001",
    )

    models = registry.list_models()
    assert len(models) == 1
    assert models[0]["registry_id"] == reg_id
    fetched = registry.get_model(reg_id)
    assert fetched["model_name"] == "random_forest"
    assert fetched["metrics"]["macro_f1"] == 0.89


def test_no_overwrite(tmp_path: Path) -> None:
    """Saving a model under an existing valid experiment_hash creates a versioned copy."""
    mgr = RetrainingManager(str(tmp_path / "models"))
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    model_v1 = DummyClassifier(strategy="most_frequent").fit(X, y)
    model_v2 = DummyClassifier(strategy="constant", constant=1).fit(X, y)

    path_1 = Path(mgr.save_model(model_v1, "dummy", "deadbeef12345678", {"version": 1}))
    path_2 = Path(mgr.save_model(model_v2, "dummy", "deadbeef12345678", {"version": 2}))

    assert path_1.exists()
    assert path_2.exists()
    assert path_1 != path_2
    assert "_v2" in path_2.name

    loaded_1, meta_1 = mgr.load_model(str(path_1))
    loaded_2, meta_2 = mgr.load_model(str(path_2))
    assert meta_1["version"] == 1
    assert meta_2["version"] == 2
    assert loaded_1.strategy == "most_frequent"
    assert loaded_2.strategy == "constant"


def test_cache_hit(tmp_path: Path) -> None:
    """Same experiment hash results in a cache hit without retraining."""
    mgr = RetrainingManager(str(tmp_path / "models"))
    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1])
    model = DummyClassifier().fit(X, y)

    exp_hash = mgr.compute_experiment_hash(
        dataset_hash="sha_abc",
        config={"target_column": "label", "exclude_columns": [], "task_type": "classification", "random_seed": 42},
        feature_schema={"f1": "float64"},
    )
    mgr.save_model(model, "dummy", exp_hash, {"macro_f1": 0.75})

    cached = mgr.check_cache(exp_hash, "dummy")
    should_train, reason = mgr.should_retrain(exp_hash, "dummy")

    assert cached is not None
    assert cached["experiment_hash"] == exp_hash
    assert should_train is False
    assert "valid cached model" in reason


def test_cache_miss(tmp_path: Path) -> None:
    """Unseen experiment hash returns None on check_cache and True for should_retrain."""
    mgr = RetrainingManager(str(tmp_path / "models"))

    cached = mgr.check_cache("nonexistent_hash_999", "random_forest")
    should_train, reason = mgr.should_retrain("nonexistent_hash_999", "random_forest")

    assert cached is None
    assert should_train is True
    assert "no cached model" in reason


def test_registry_persists(tmp_path: Path) -> None:
    """Registry entries persist to disk and reload in a fresh ModelRegistry instance."""
    reg_dir = tmp_path / "registry"
    reg_1 = ModelRegistry(str(reg_dir))
    reg_id = reg_1.register(
        model_name="lightgbm",
        artifact_path="models/lgbm.joblib",
        metrics={"macro_f1": 0.91},
        carbon_passport_path="models/lgbm_passport.json",
        experiment_hash="hash_persist",
    )

    reg_2 = ModelRegistry(str(reg_dir))
    assert len(reg_2.list_models()) == 1
    assert reg_2.get_model(reg_id)["model_name"] == "lightgbm"
    best = reg_2.get_best_model("macro_f1")
    assert best is not None
    assert best["registry_id"] == reg_id


def test_passport_json_valid(tmp_path: Path) -> None:
    """Exported CarbonPassport JSON is valid and round-trips cleanly."""
    cp = CarbonPassport()
    doc = _build_sample_passport()
    out_path = cp.export_json(doc, str(tmp_path / "passport.json"))

    loaded = json.loads(Path(out_path).read_text(encoding="utf-8"))
    assert loaded["model_name"] == "random_forest"
    assert loaded["dataset_hash"] == "abc123"
    assert loaded["predictive_metrics"]["macro_f1"] == pytest.approx(0.88)


def test_passport_has_required_fields() -> None:
    """Generated CarbonPassport contains all mandatory transparency fields."""
    doc = _build_sample_passport()
    required_fields = {
        "model_id",
        "dataset_hash",
        "feature_schema",
        "training_config",
        "hardware",
        "training_duration_seconds",
        "training_energy_kwh",
        "training_co2_kg_estimate",
        "carbon_intensity_assumption",
        "inference_energy_per_1k_predictions_kwh",
        "predictive_metrics",
        "validation_methodology",
        "confidence_intervals",
        "known_limitations",
        "generated_at",
        "framework_version",
    }
    assert required_fields.issubset(doc.keys())


def test_passport_html_no_external_deps(tmp_path: Path) -> None:
    """Exported CarbonPassport HTML has inline styles and no external CDN/URL links."""
    cp = CarbonPassport()
    doc = _build_sample_passport()
    out_path = cp.export_html(doc, str(tmp_path / "passport.html"))
    html = Path(out_path).read_text(encoding="utf-8")

    assert "<style>" in html
    assert "Carbon Passport" in html
    assert not re.search(r'<link[^>]+href=["\']https?://', html, flags=re.IGNORECASE)
    assert not re.search(r'<script[^>]+src=["\']https?://', html, flags=re.IGNORECASE)
