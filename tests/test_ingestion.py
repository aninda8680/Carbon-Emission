"""Tests for DataIngestionEngine."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from carbon_automl.core.ingestion import DataIngestionEngine
from carbon_automl.evaluation.validation import ValidationEngine


def test_load_csv(sample_csv_path: Path) -> None:
    """DataIngestionEngine loads CSV and returns expected X, y shapes and metadata."""
    engine = DataIngestionEngine()
    X, y, meta = engine.load(str(sample_csv_path), target_column="label")

    assert X.shape == (100, 5)
    assert y.shape == (100,)
    assert "label" not in X.columns
    assert y.name == "label"
    assert meta["format"] == "csv"
    assert meta["shape"] == (100, 6)
    assert meta["feature_shape"] == (100, 5)


def test_load_parquet(
    sample_parquet_path: Path, sample_classification_df: pd.DataFrame
) -> None:
    """DataIngestionEngine loads Parquet accurately with matching values."""
    engine = DataIngestionEngine()
    X, y, meta = engine.load(str(sample_parquet_path), target_column="label")

    assert X.shape == (100, 5)
    assert y.shape == (100,)
    assert meta["format"] == "parquet"
    assert list(X.columns) == [c for c in sample_classification_df.columns if c != "label"]
    pd.testing.assert_series_equal(y, sample_classification_df["label"])


def test_missing_target_raises(sample_csv_path: Path) -> None:
    """Loading fails with ValueError when target_column does not exist."""
    engine = DataIngestionEngine()
    with pytest.raises(ValueError, match="target_column"):
        engine.load(str(sample_csv_path), target_column="nonexistent_target")


def test_exclude_columns(sample_csv_path: Path) -> None:
    """Excluded columns are removed from X without affecting y."""
    engine = DataIngestionEngine()
    X, y, meta = engine.load(
        str(sample_csv_path),
        target_column="label",
        exclude_columns=["cat_1", "bin_1"],
    )

    assert "cat_1" not in X.columns
    assert "bin_1" not in X.columns
    assert X.shape == (100, 3)
    assert y.shape == (100,)
    assert meta["exclude_columns"] == ["cat_1", "bin_1"]

    with pytest.raises(ValueError, match="exclude_columns"):
        engine.load(
            str(sample_csv_path),
            target_column="label",
            exclude_columns=["does_not_exist"],
        )


def test_exclude_target_raises(sample_csv_path: Path) -> None:
    """Excluding the target column itself raises ValueError."""
    engine = DataIngestionEngine()
    with pytest.raises(ValueError, match="must not appear in exclude_columns"):
        engine.load(
            str(sample_csv_path),
            target_column="label",
            exclude_columns=["label"],
        )


def test_original_df_unchanged(sample_csv_path: Path) -> None:
    """Source file on disk and subsequent loads are unchanged after exclusions/edits."""
    before_bytes = sample_csv_path.read_bytes()
    engine = DataIngestionEngine()

    X1, y1, _ = engine.load(
        str(sample_csv_path),
        target_column="label",
        exclude_columns=["cat_1"],
    )
    # Mutate returned X1 and y1 in memory
    X1.iloc[0, 0] = -999999.0
    y1.iloc[0] = -1

    after_bytes = sample_csv_path.read_bytes()
    assert before_bytes == after_bytes

    X2, y2, _ = engine.load(str(sample_csv_path), target_column="label")
    assert "cat_1" in X2.columns
    assert X2.shape == (100, 5)
    assert y2.iloc[0] != -1


def test_file_hash_recorded(sample_csv_path: Path) -> None:
    """Metadata records accurate SHA256 file_hash and load_timestamp."""
    expected_sha256 = hashlib.sha256(sample_csv_path.read_bytes()).hexdigest()
    engine = DataIngestionEngine()
    _, _, meta = engine.load(str(sample_csv_path), target_column="label")

    assert "file_hash" in meta
    assert meta["file_hash"] == expected_sha256
    assert meta["file_hash_algorithm"] == "sha256"
    assert "load_timestamp" in meta and meta["load_timestamp"]


def test_unsupported_format_raises(tmp_path: Path) -> None:
    """Loading unsupported formats (e.g. .xlsx) raises ValueError."""
    xlsx_path = tmp_path / "unsupported.xlsx"
    xlsx_path.write_bytes(b"placeholder")

    engine = DataIngestionEngine()
    with pytest.raises(ValueError, match="Unsupported file format"):
        engine.load(str(xlsx_path), target_column="label")

    with pytest.raises(FileNotFoundError):
        engine.load(str(tmp_path / "missing.csv"), target_column="label")


def test_row_count_preserved(sample_csv_path: Path) -> None:
    """Sum of split row counts equals the original ingested dataset row count."""
    engine = DataIngestionEngine()
    X, y, meta = engine.load(str(sample_csv_path), target_column="label")

    validator = ValidationEngine()
    X_train, X_test, y_train, y_test = validator.create_holdout_split(
        X, y, test_size=0.2, random_state=42, stratify=True
    )

    assert X_train.shape[0] + X_test.shape[0] == meta["shape"][0]
    assert y_train.shape[0] + y_test.shape[0] == meta["shape"][0]
