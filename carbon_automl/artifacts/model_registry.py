"""
ModelRegistry: Persistent catalogue of registered model artifacts with their
metrics, carbon passports, and experiment hashes.

Concurrent-read safety is provided by writing to a temporary file and
performing an atomic rename, so readers never see a partial write.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ModelRegistry:
    """Thread-safe (single-process) model registry backed by a JSON file.

    The registry JSON file is never partially written — all updates use a
    write-to-tmp-then-rename pattern under an ``RLock`` to guarantee
    atomicity and safe concurrent reads.

    Parameters
    ----------
    registry_dir:
        Directory where ``registry.json`` is stored.
    """

    _REGISTRY_FILENAME = "registry.json"

    def __init__(self, registry_dir: str | Path) -> None:
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._registry_path = self.registry_dir / self._REGISTRY_FILENAME
        self._lock = threading.RLock()
        self._models: dict[str, dict] = {}  # registry_id -> model metadata
        self.load_registry()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        model_name: str,
        artifact_path: str,
        metrics: dict,
        carbon_passport_path: str,
        experiment_hash: str,
    ) -> str:
        """Register a trained model in the registry.

        Parameters
        ----------
        model_name:
            Human-readable model name (e.g. ``"random_forest"``).
        artifact_path:
            Absolute path to the saved joblib artifact.
        metrics:
            Dict of evaluation metrics (e.g. ``{"macro_f1": 0.92, "accuracy": 0.93}``).
        carbon_passport_path:
            Path to the JSON carbon passport for this model.
        experiment_hash:
            Hash returned by ``RetrainingManager.compute_experiment_hash``.

        Returns
        -------
        str
            UUID registry identifier (``registry_id``).
        """
        registry_id = str(uuid.uuid4())
        entry: dict[str, Any] = {
            "registry_id": registry_id,
            "model_name": model_name,
            "artifact_path": str(artifact_path),
            "metrics": metrics,
            "carbon_passport_path": str(carbon_passport_path),
            "experiment_hash": experiment_hash,
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            self._models[registry_id] = entry
            self.save_registry()
        return registry_id

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def list_models(self) -> list[dict]:
        """Return a list of all registered model metadata dicts."""
        with self._lock:
            return [dict(m) for m in self._models.values()]

    def get_model(self, registry_id: str) -> dict:
        """Return metadata for a specific registry entry.

        Raises
        ------
        KeyError
            If *registry_id* is not found.
        """
        with self._lock:
            if registry_id not in self._models:
                raise KeyError(f"No model with registry_id '{registry_id}'")
            return dict(self._models[registry_id])

    def get_best_model(self, metric: str = "macro_f1") -> dict | None:
        """Return the model entry with the highest value of *metric*.

        Returns ``None`` if the registry is empty or no model has the metric.
        """
        with self._lock:
            candidates = [
                m for m in self._models.values()
                if metric in m.get("metrics", {})
                and m["metrics"][metric] is not None
            ]
            if not candidates:
                return None
            return dict(max(candidates, key=lambda m: float(m["metrics"][metric])))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_registry(self) -> None:
        """Atomically persist the registry to ``registry.json``.

        Writes to a temporary file first, then renames it over the target —
        preventing readers from ever seeing a partial write.
        """
        with self._lock:
            payload = {
                "schema_version": "1.0",
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "models": self._models,
            }
            tmp_path = self.registry_dir / f".registry.{uuid.uuid4().hex}.tmp"
            tmp_path.write_text(
                json.dumps(payload, indent=2, default=str), encoding="utf-8"
            )
            tmp_path.replace(self._registry_path)  # atomic on most filesystems

    def load_registry(self) -> None:
        """Load the registry from ``registry.json`` if it exists.

        Does nothing (starts empty) if the file does not exist yet.
        """
        with self._lock:
            if not self._registry_path.exists():
                self._models = {}
                return
            try:
                payload = json.loads(self._registry_path.read_text(encoding="utf-8"))
                self._models = payload.get("models", {})
            except (json.JSONDecodeError, OSError):
                # Corrupted registry — start fresh but preserve the bad file
                corrupt_path = self._registry_path.with_suffix(".corrupt")
                self._registry_path.replace(corrupt_path)
                self._models = {}
