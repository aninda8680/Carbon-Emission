"""
Data integrity management for the Carbon-Aware AutoML framework.

Computes and verifies cryptographic fingerprints of datasets so that
any accidental mutation is detected before it can silently corrupt results.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd


# --------------------------------------------------------------------------- #
# Exceptions                                                                   #
# --------------------------------------------------------------------------- #


class DataIntegrityError(Exception):
    """Raised when a DataFrame no longer matches its recorded fingerprint."""


# --------------------------------------------------------------------------- #
# Manager                                                                      #
# --------------------------------------------------------------------------- #


class DataIntegrityManager:
    """
    Computes, records, and verifies cryptographic fingerprints of datasets.

    Usage
    -----
    >>> manager = DataIntegrityManager("data/train.csv")
    >>> fingerprint = manager.record_original(df)
    >>> manager.verify_unchanged(df)  # True  — raises DataIntegrityError otherwise
    """

    # Block size for streaming file hashing (1 MiB)
    _BLOCK_SIZE: int = 1 << 20

    def __init__(self, source_path: str) -> None:
        """
        Initialise the manager and immediately hash the source file.

        Parameters
        ----------
        source_path:
            Path to the original dataset file. The file must exist on disk.

        Raises
        ------
        FileNotFoundError
            If *source_path* does not exist.
        """
        self._source_path: Path = Path(source_path)
        if not self._source_path.exists():
            raise FileNotFoundError(f"Source file not found: {self._source_path}")

        self._sha256: str = self._hash_file("sha256")
        self._md5: str = self._hash_file("md5")

        # Will be populated by record_original()
        self.row_count: int = 0
        self.col_count: int = 0
        self.schema: dict[str, str] = {}
        self.dtypes: dict[str, str] = {}
        self._fingerprint: dict[str, Any] | None = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def record_original(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Record the shape and schema of *df* as the authoritative fingerprint.

        The DataFrame is **never** modified.

        Parameters
        ----------
        df:
            The freshly-loaded DataFrame that is to be treated as the reference.

        Returns
        -------
        dict
            Fingerprint with keys: hash, sha256, md5, row_count, col_count,
            columns (list), schema (dict), dtypes (dict col -> dtype string).
        """
        dtypes_map = {col: str(dtype) for col, dtype in df.dtypes.items()}
        self.row_count = len(df)
        self.col_count = len(df.columns)
        self.schema = dict(dtypes_map)
        self.dtypes = dict(dtypes_map)

        fingerprint: dict[str, Any] = {
            "hash": self._sha256,
            "sha256": self._sha256,
            "md5": self._md5,
            "source_path": str(self._source_path),
            "row_count": self.row_count,
            "col_count": self.col_count,
            "columns": list(df.columns),
            "schema": dict(dtypes_map),
            "dtypes": dtypes_map,
        }
        self._fingerprint = fingerprint
        return fingerprint

    def verify_unchanged(self, df: pd.DataFrame) -> bool:
        """
        Verify that *df* still matches the recorded fingerprint.

        Parameters
        ----------
        df:
            DataFrame to verify — not modified.

        Returns
        -------
        bool
            ``True`` if the DataFrame matches; ``False`` is never returned —
            the method raises ``DataIntegrityError`` on any mismatch.

        Raises
        ------
        RuntimeError
            If ``record_original`` has not yet been called.
        DataIntegrityError
            If the DataFrame's row count, column count, or column dtypes differ
            from the recorded fingerprint.
        """
        if self._fingerprint is None:
            raise RuntimeError(
                "Call record_original() before verify_unchanged()."
            )

        fp = self._fingerprint
        mismatches: list[str] = []

        if len(df) != fp["row_count"]:
            mismatches.append(
                f"row_count: expected {fp['row_count']}, got {len(df)}"
            )

        if len(df.columns) != fp["col_count"]:
            mismatches.append(
                f"col_count: expected {fp['col_count']}, got {len(df.columns)}"
            )

        # Only check dtypes for columns present in both
        expected_dtypes: dict[str, str] = fp["dtypes"]
        for col, expected_dtype in expected_dtypes.items():
            if col not in df.columns:
                mismatches.append(f"column '{col}' missing from DataFrame")
            elif str(df[col].dtype) != expected_dtype:
                mismatches.append(
                    f"column '{col}' dtype: expected {expected_dtype!r}, "
                    f"got {str(df[col].dtype)!r}"
                )

        if mismatches:
            detail = "; ".join(mismatches)
            raise DataIntegrityError(
                f"DataFrame no longer matches recorded fingerprint — {detail}"
            )

        return True

    def get_fingerprint(self) -> dict[str, Any]:
        """
        Return the stored fingerprint.

        Raises
        ------
        RuntimeError
            If ``record_original`` has not yet been called.
        """
        if self._fingerprint is None:
            raise RuntimeError(
                "No fingerprint recorded yet. Call record_original() first."
            )
        return dict(self._fingerprint)

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def sha256(self) -> str:
        """SHA-256 hex digest of the source file."""
        return self._sha256

    @property
    def md5(self) -> str:
        """MD5 hex digest of the source file."""
        return self._md5

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _hash_file(self, algorithm: str) -> str:
        """Stream-hash the source file using *algorithm* and return the hex digest."""
        h = hashlib.new(algorithm)
        with self._source_path.open("rb") as fh:
            for block in iter(lambda: fh.read(self._BLOCK_SIZE), b""):
                h.update(block)
        return h.hexdigest()
