"""Tests for DataProfiler."""

from __future__ import annotations

import numpy as np
import pandas as pd

from carbon_automl.core.data_profiler import DataProfiler


def test_profile_shape(sample_classification_df: pd.DataFrame) -> None:
    """Profile shape matches the input feature DataFrame shape."""
    X = sample_classification_df.drop(columns=["label"])
    y = sample_classification_df["label"]

    profile = DataProfiler().profile(X, y)

    assert profile["shape"] == X.shape
    assert profile["memory_mb"] > 0


def test_numerical_detection(sample_classification_df: pd.DataFrame) -> None:
    """Numerical columns are classified as 'numerical' with summary stats."""
    X = sample_classification_df.drop(columns=["label"])
    y = sample_classification_df["label"]

    profile = DataProfiler().profile(X, y)

    assert profile["feature_types"]["num_1"] == "numerical"
    assert profile["feature_types"]["num_2"] == "numerical"
    assert "num_1" in profile["numerical_stats"]
    for stat_key in ("min", "max", "mean", "std"):
        assert stat_key in profile["numerical_stats"]["num_1"]


def test_categorical_detection(sample_classification_df: pd.DataFrame) -> None:
    """Categorical and binary columns are classified properly."""
    X = sample_classification_df.drop(columns=["label"])
    y = sample_classification_df["label"]

    profile = DataProfiler().profile(X, y)

    assert profile["feature_types"]["cat_1"] == "categorical"
    assert profile["feature_types"]["cat_2"] == "categorical"
    assert profile["feature_types"]["bin_1"] == "binary"


def test_high_cardinality_detection(high_cardinality_df: pd.DataFrame) -> None:
    """Object columns with >50 unique values (and < n_rows) are flagged."""
    X = high_cardinality_df.drop(columns=["label"])
    y = high_cardinality_df["label"]

    profile = DataProfiler().profile(X, y)

    assert profile["feature_types"]["high_card"] == "high_cardinality_categorical"
    assert any("high_card" in rec for rec in profile["recommendations"])


def test_identifier_detection() -> None:
    """String column with unique_count == n_rows is flagged as identifier_like."""
    rng = np.random.default_rng(7)
    n = 40
    X = pd.DataFrame(
        {
            "patient_id": [f"PAT-{i:04d}" for i in range(n)],
            "value": rng.normal(0.0, 1.0, size=n),
        }
    )
    y = pd.Series(rng.integers(0, 2, size=n), name="label")

    profile = DataProfiler().profile(X, y)

    assert profile["feature_types"]["patient_id"] == "identifier_like"
    assert any("patient_id" in rec for rec in profile["recommendations"])


def test_constant_detection() -> None:
    """Column with a single unique value is flagged as 'constant'."""
    rng = np.random.default_rng(11)
    n = 30
    X = pd.DataFrame(
        {
            "const_col": ["same_value"] * n,
            "num_col": rng.normal(0.0, 1.0, size=n),
        }
    )
    y = pd.Series(rng.integers(0, 2, size=n), name="label")

    profile = DataProfiler().profile(X, y)

    assert profile["feature_types"]["const_col"] == "constant"
    assert any("const_col" in rec for rec in profile["recommendations"])


def test_missing_values_counted(missing_values_df: pd.DataFrame) -> None:
    """Missing value counts per column match exact NaN counts."""
    X = missing_values_df.drop(columns=["label"])
    y = missing_values_df["label"]

    profile = DataProfiler().profile(X, y)

    assert profile["missing_values"]["miss_num"] == 30
    assert profile["missing_values"]["miss_cat"] == 30
    assert profile["missing_values"]["clean_num"] == 0
    assert profile["missing_values"]["clean_cat"] == 0


def test_duplicate_rows_counted(sample_classification_df: pd.DataFrame) -> None:
    """Duplicate feature rows are counted accurately."""
    X = sample_classification_df.drop(columns=["label"]).iloc[:20].copy()
    # Append 5 exact duplicates of the first 5 rows
    X_with_dups = pd.concat([X, X.iloc[:5]], ignore_index=True)
    y = pd.Series([0] * 25, name="label")

    profile = DataProfiler().profile(X_with_dups, y)

    assert profile["duplicate_rows"] == 5


def test_class_imbalance_detected(imbalanced_df: pd.DataFrame) -> None:
    """Imbalanced classification target computes exact class counts and imbalance ratio."""
    X = imbalanced_df.drop(columns=["label"])
    y = imbalanced_df["label"]

    profile = DataProfiler().profile(X, y)

    class_dist = profile["class_distribution"]
    assert class_dist is not None
    assert class_dist["counts"]["0"] == 90
    assert class_dist["counts"]["1"] == 10
    assert class_dist["imbalance_ratio"] == 9.0


def test_task_suggestion_classification(sample_classification_df: pd.DataFrame) -> None:
    """Target with <= 20 unique values yields 'classification' suggestion."""
    X = sample_classification_df.drop(columns=["label"])
    y = sample_classification_df["label"]

    profile = DataProfiler().profile(X, y)

    assert profile["task_type_suggestion"] == "classification"


def test_task_suggestion_regression(regression_df: pd.DataFrame) -> None:
    """Continuous target with > 20 unique values yields 'regression' suggestion."""
    X = regression_df.drop(columns=["value"])
    y = regression_df["value"]

    profile = DataProfiler().profile(X, y)

    assert profile["task_type_suggestion"] == "regression"
