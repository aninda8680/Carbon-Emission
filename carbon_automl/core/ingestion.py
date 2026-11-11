"""
Data ingestion engine for the Carbon-Aware AutoML framework.

Loads tabular datasets (CSV / Parquet), validates the requested target and
exclusion columns, records a cryptographic fingerprint of the source file,
and returns features, target, and rich metadata — all without ever mutating
the original DataFrame.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from carbon_automl.core.integrity_manager import DataIntegrityManager


# --------------------------------------------------------------------------- #
# Engine                                                                       #
# --------------------------------------------------------------------------- #


class DataIngestionEngine:
    """
    Load, validate, and fingerprint tabular datasets.

    Supported formats
    -----------------
    * ``.csv`` — loaded with ``pandas.read_csv``
    * ``.parquet`` — loaded with ``pandas.read_parquet``

    The engine never modifies the loaded DataFrame. All column exclusions are
    performed on a copy so that the original in-memory object remains intact.

    Examples
    --------
    >>> engine = DataIngestionEngine()
    >>> X, y, meta = engine.load("data/train.csv", target_column="label")
    """

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def load(
        self,
        path: str,
        target_column: str,
        exclude_columns: list[str] | None = None,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
        """
        Load a dataset from *path* and return features, target, and metadata.

        Parameters
        ----------
        path:
            Absolute or relative path to the dataset file.
        target_column:
            Name of the column to use as the prediction target.
        exclude_columns:
            Columns to drop from the feature matrix (e.g. identifiers).
            Must not contain *target_column*.
        **kwargs:
            Extra keyword arguments forwarded to ``pd.read_csv`` /
            ``pd.read_parquet``.

        Returns
        -------
        X : pd.DataFrame
            Feature matrix (copy of the raw data minus target and excluded cols).
        y : pd.Series
            Target series with its original name preserved.
        metadata : dict
            Provenance information with keys:
            ``source_path``, ``format``, ``shape``, ``dtypes``,
            ``file_hash``, ``load_timestamp``.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the format is unsupported, *target_column* is absent, an entry
            in *exclude_columns* is absent, or *target_column* appears in
            *exclude_columns*.
        """
        if exclude_columns is None:
            exclude_columns = []

        file_path = Path(path)

        # ---- existence check ----------------------------------------- #
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # ---- format detection ---------------------------------------- #
        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            raw_df: pd.DataFrame = pd.read_csv(file_path, **kwargs)
            fmt = "csv"
        elif suffix in {".parquet", ".pq"}:
            raw_df = pd.read_parquet(file_path, **kwargs)
            fmt = "parquet"
        else:
            raise ValueError(
                f"Unsupported file format '{suffix}'. Supported: .csv, .parquet"
            )

        # ---- fingerprint the source file ------------------------------ #
        file_hash = self._hash_file(file_path)
        integrity_manager = DataIntegrityManager(str(file_path))
        integrity_manager.record_original(raw_df)

        # ---- column validation --------------------------------------- #
        all_columns: set[str] = set(raw_df.columns)

        if target_column not in all_columns:
            raise ValueError(
                f"target_column '{target_column}' not found in dataset. "
                f"Available columns: {sorted(all_columns)}"
            )

        invalid_excludes = [c for c in exclude_columns if c not in all_columns]
        if invalid_excludes:
            raise ValueError(
                f"The following exclude_columns were not found in the dataset: "
                f"{invalid_excludes}. Available columns: {sorted(all_columns)}"
            )

        if target_column in exclude_columns:
            raise ValueError(
                f"target_column '{target_column}' must not appear in "
                f"exclude_columns. Remove it from exclude_columns."
            )

        # ---- build X and y on a COPY --------------------------------- #
        working_df = raw_df.copy()

        y: pd.Series = working_df[target_column].copy()

        drop_cols = [target_column] + [c for c in exclude_columns if c in working_df.columns]
        X: pd.DataFrame = working_df.drop(columns=drop_cols)

        # ---- metadata ------------------------------------------------ #
        metadata: dict[str, Any] = {
            "source_path": str(file_path.resolve()),
            "format": fmt,
            "shape": raw_df.shape,
            "feature_shape": X.shape,
            "dtypes": {col: str(dtype) for col, dtype in raw_df.dtypes.items()},
            "file_hash": file_hash,
            "file_hash_algorithm": "sha256",
            "load_timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "target_column": target_column,
            "exclude_columns": list(exclude_columns),
            "integrity_fingerprint": integrity_manager.get_fingerprint(),
        }

        return X, y, metadata

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _hash_file(file_path: Path, algorithm: str = "sha256") -> str:
        """Return the hex digest of *file_path* using *algorithm*."""
        block_size = 1 << 20  # 1 MiB
        h = hashlib.new(algorithm)
        with file_path.open("rb") as fh:
            for block in iter(lambda: fh.read(block_size), b""):
                h.update(block)
        return h.hexdigest()
