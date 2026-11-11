"""Tests for DataIntegrityManager and dataset fingerprinting."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from carbon_automl.core.integrity_manager import (
    DataIntegrityError,
    DataIntegrityManager,
)

EXPECTED_MALNUTRITION_SHA256 = (
    "c68a778e13cc257f80f9edb8c44ea2b42eb9c30fcbbcbb9cce3d4c4bbb1d7ca8"
)


def test_hash_computation(sample_csv_path: Path) -> None:
    """SHA256 and MD5 digests match hashlib computation on the raw bytes."""
    raw = sample_csv_path.read_bytes()
    expected_sha256 = hashlib.sha256(raw).hexdigest()
    expected_md5 = hashlib.md5(raw).hexdigest()

    manager = DataIntegrityManager(str(sample_csv_path))

    assert manager.sha256 == expected_sha256
    assert manager.md5 == expected_md5


def test_fingerprint_recorded(
    sample_csv_path: Path, sample_classification_df: pd.DataFrame
) -> None:
    """Recorded fingerprint contains shape, dtypes, columns, and hashes."""
    manager = DataIntegrityManager(str(sample_csv_path))
    fp = manager.record_original(sample_classification_df)

    assert fp["row_count"] == len(sample_classification_df)
    assert fp["col_count"] == len(sample_classification_df.columns)
    assert fp["columns"] == list(sample_classification_df.columns)
    assert fp["dtypes"] == {
        col: str(dt) for col, dt in sample_classification_df.dtypes.items()
    }
    assert fp["sha256"] == manager.sha256
    assert fp["md5"] == manager.md5
    assert manager.get_fingerprint() == fp


def test_verify_unchanged_passes(
    sample_csv_path: Path, sample_classification_df: pd.DataFrame
) -> None:
    """Verification succeeds when the DataFrame is unmodified."""
    manager = DataIntegrityManager(str(sample_csv_path))
    manager.record_original(sample_classification_df)

    assert manager.verify_unchanged(sample_classification_df.copy()) is True


def test_verify_changed_fails(
    sample_csv_path: Path, sample_classification_df: pd.DataFrame
) -> None:
    """Adding a row causes verify_unchanged to raise DataIntegrityError."""
    manager = DataIntegrityManager(str(sample_csv_path))
    manager.record_original(sample_classification_df)

    modified = pd.concat(
        [sample_classification_df, sample_classification_df.iloc[[0]]],
        ignore_index=True,
    )
    with pytest.raises(DataIntegrityError, match="row_count"):
        manager.verify_unchanged(modified)


def test_verify_column_removed_fails(
    sample_csv_path: Path, sample_classification_df: pd.DataFrame
) -> None:
    """Removing a column causes verify_unchanged to raise DataIntegrityError."""
    manager = DataIntegrityManager(str(sample_csv_path))
    manager.record_original(sample_classification_df)

    modified = sample_classification_df.drop(columns=["num_1"])
    with pytest.raises(DataIntegrityError, match="col_count"):
        manager.verify_unchanged(modified)


def test_verify_dtype_changed_fails(
    sample_csv_path: Path, sample_classification_df: pd.DataFrame
) -> None:
    """Changing a column dtype causes verify_unchanged to raise DataIntegrityError."""
    manager = DataIntegrityManager(str(sample_csv_path))
    manager.record_original(sample_classification_df)

    modified = sample_classification_df.copy()
    modified["num_1"] = modified["num_1"].astype("float32")
    with pytest.raises(DataIntegrityError, match="dtype"):
        manager.verify_unchanged(modified)


def test_original_dataset_hash(malnutrition_dataset_path: Path) -> None:
    """Actual child malnutrition dataset matches its canonical SHA256 digest."""
    manager = DataIntegrityManager(str(malnutrition_dataset_path))
    assert manager.sha256 == EXPECTED_MALNUTRITION_SHA256
