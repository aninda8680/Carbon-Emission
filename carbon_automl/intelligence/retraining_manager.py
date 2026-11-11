"""
RetrainingManager: Manages model caching to avoid redundant retraining.

Key principle: Never retrain if a valid cached model exists for the same
dataset + config hash.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import joblib


class RetrainingManager:
    """Manages model caching using experiment hashes to avoid redundant retraining.

    Parameters
    ----------
    models_dir:
        Directory where model artifacts (joblib files + metadata JSONs) are stored.
    """

    _METADATA_SUFFIX = ".meta.json"
    _MODEL_SUFFIX = ".joblib"

    def __init__(self, models_dir: str) -> None:
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------------

    def compute_experiment_hash(
        self,
        dataset_hash: str,
        config: dict,
        feature_schema: dict,
    ) -> str:
        """Compute a deterministic SHA-256 hex digest for an experiment.

        Only the fields that affect the learned model are included:
        ``dataset_hash``, ``target_column``, ``exclude_columns``,
        ``task_type``, ``random_seed``, and ``preprocessing_config``
        (taken from *config* if present).

        Parameters
        ----------
        dataset_hash:
            SHA-256 hex digest of the raw dataset file.
        config:
            Experiment configuration dict (AutoMLConfig or plain dict).
        feature_schema:
            Mapping of column name → dtype string.

        Returns
        -------
        str
            64-character hex string.
        """
        # Extract only the fields that influence training
        config_dict = config if isinstance(config, dict) else config.model_dump()

        canonical = {
            "dataset_hash": dataset_hash,
            "target_column": config_dict.get("target_column", ""),
            "exclude_columns": sorted(config_dict.get("exclude_columns") or []),
            "task_type": config_dict.get("task_type", "auto"),
            "random_seed": config_dict.get("random_seed", 42),
            "preprocessing_config": config_dict.get("preprocessing_config", {}),
            "feature_schema": {k: str(v) for k, v in sorted(feature_schema.items())},
        }
        serialised = json.dumps(canonical, sort_keys=True, default=str)
        return hashlib.sha256(serialised.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Cache checks
    # ------------------------------------------------------------------

    def check_cache(self, experiment_hash: str, model_name: str) -> dict | None:
        """Return the cached result dict if a valid artifact exists, else None.

        Validity checks:
        1. Model file exists.
        2. Metadata file exists and its ``experiment_hash`` matches.
        3. Model file is not corrupted (can be loaded by joblib).

        Parameters
        ----------
        experiment_hash:
            Hash returned by :meth:`compute_experiment_hash`.
        model_name:
            Logical model name (e.g. ``"random_forest"``).

        Returns
        -------
        dict or None
            Cached metadata dict (including ``artifact_path``) on a cache hit,
            ``None`` otherwise.
        """
        artifact_path = self._artifact_path(experiment_hash, model_name)
        meta_path = self._meta_path(experiment_hash, model_name)

        if not artifact_path.exists() or not meta_path.exists():
            return None

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

        # Verify hash in metadata matches what we were asked about
        if meta.get("experiment_hash") != experiment_hash:
            return None

        # Light corruption check
        try:
            joblib.load(artifact_path)
        except Exception:
            return None

        meta["artifact_path"] = str(artifact_path)
        return meta

    def should_retrain(
        self, experiment_hash: str, model_name: str
    ) -> tuple[bool, str]:
        """Decide whether a model needs to be retrained.

        Returns
        -------
        (True, "no cached model")
            when no valid cache exists.
        (False, "valid cached model found at <path>")
            when a valid cached artifact is found.
        """
        cached = self.check_cache(experiment_hash, model_name)
        if cached is None:
            return True, "no cached model"
        return False, f"valid cached model found at {cached['artifact_path']}"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(
        self,
        model: Any,
        model_name: str,
        experiment_hash: str,
        metadata: dict,
    ) -> str:
        """Persist a trained model and its metadata.

        Never overwrites an existing valid artifact — creates a versioned copy
        with a numeric suffix (``_v2``, ``_v3``, …) instead.

        Parameters
        ----------
        model:
            Fitted sklearn-compatible estimator.
        model_name:
            Logical model name.
        experiment_hash:
            Hash for this experiment.
        metadata:
            Arbitrary serialisable dict (metrics, config, timestamps, …).

        Returns
        -------
        str
            Absolute path to the saved model artifact.
        """
        artifact_path = self._resolve_no_overwrite(experiment_hash, model_name)
        meta_path = artifact_path.with_suffix(self._METADATA_SUFFIX)

        # Always record the hash and path in metadata
        metadata = dict(metadata)
        metadata["experiment_hash"] = experiment_hash
        metadata["model_name"] = model_name
        metadata["artifact_path"] = str(artifact_path)

        # Write model first, then metadata (atomic-ish order: metadata acts as
        # the "commit marker")
        joblib.dump(model, artifact_path)

        tmp_meta = meta_path.with_suffix(".tmp")
        tmp_meta.write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8"
        )
        tmp_meta.replace(meta_path)  # atomic rename on most filesystems

        return str(artifact_path)

    def load_model(self, artifact_path: str) -> tuple[Any, dict]:
        """Load a model and its associated metadata.

        Parameters
        ----------
        artifact_path:
            Path returned by :meth:`save_model`.

        Returns
        -------
        (model, metadata)
        """
        artifact_path_obj = Path(artifact_path)
        if not artifact_path_obj.exists():
            raise FileNotFoundError(f"Model artifact not found: {artifact_path_obj}")

        meta_path = artifact_path_obj.with_suffix(self._METADATA_SUFFIX)
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        model = joblib.load(artifact_path_obj)
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        return model, metadata

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _artifact_path(self, experiment_hash: str, model_name: str) -> Path:
        safe_name = self._safe(model_name)
        return self.models_dir / f"{safe_name}_{experiment_hash[:12]}{self._MODEL_SUFFIX}"

    def _meta_path(self, experiment_hash: str, model_name: str) -> Path:
        safe_name = self._safe(model_name)
        return self.models_dir / f"{safe_name}_{experiment_hash[:12]}{self._METADATA_SUFFIX}"

    def _resolve_no_overwrite(self, experiment_hash: str, model_name: str) -> Path:
        """Return a path that does not collide with an existing valid artifact."""
        base = self._artifact_path(experiment_hash, model_name)
        # Check if the existing artifact is still valid; if so create a versioned path
        cached = self.check_cache(experiment_hash, model_name)
        if cached is None:
            # Either doesn't exist or is corrupted — safe to use base path
            return base
        # Existing valid cache — bump version suffix to avoid overwriting
        stem = base.stem  # e.g. random_forest_abc123def456
        version = 2
        while True:
            candidate = base.with_name(f"{stem}_v{version}{self._MODEL_SUFFIX}")
            if not candidate.exists():
                return candidate
            version += 1

    @staticmethod
    def _safe(name: str) -> str:
        """Strip characters that are problematic in file names."""
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
