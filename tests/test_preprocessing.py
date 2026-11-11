"""Tests for UniversalPreprocessor, safe downcasting, and schema validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from carbon_automl.core.data_profiler import DataProfiler
from carbon_automl.core.preprocessing import UniversalPreprocessor, safe_downcast
from carbon_automl.core.schema_validator import SchemaValidator


def test_numerical_scaled(sample_classification_df: pd.DataFrame) -> None:
    """Numerical features are standardized to zero mean and unit variance on train."""
    X = sample_classification_df.drop(columns=["label"])
    y = sample_classification_df["label"]
    feature_types = DataProfiler().profile(X, y)["feature_types"]

    prep = UniversalPreprocessor(scale_numerical=True)
    arr = prep.fit_transform(X, feature_types)
    names = prep.get_feature_names()

    num_1_idx = names.index("num_1")
    num_2_idx = names.index("num_2")

    assert np.isclose(arr[:, num_1_idx].mean(), 0.0, atol=1e-7)
    assert np.isclose(arr[:, num_1_idx].std(ddof=0), 1.0, atol=1e-7)
    assert np.isclose(arr[:, num_2_idx].mean(), 0.0, atol=1e-7)
    assert np.isclose(arr[:, num_2_idx].std(ddof=0), 1.0, atol=1e-7)


def test_categorical_encoded(sample_classification_df: pd.DataFrame) -> None:
    """Categorical features are encoded into numeric values without strings."""
    X = sample_classification_df.drop(columns=["label"])
    y = sample_classification_df["label"]
    feature_types = DataProfiler().profile(X, y)["feature_types"]

    prep = UniversalPreprocessor()
    arr = prep.fit_transform(X, feature_types)

    assert isinstance(arr, np.ndarray)
    assert np.issubdtype(arr.dtype, np.number)
    names = prep.get_feature_names()
    cat_1_idx = names.index("cat_1")
    assert set(np.unique(arr[:, cat_1_idx])).issubset({0.0, 1.0, 2.0, 3.0})


def test_no_test_data_in_fit() -> None:
    """Preprocessor fit statistics depend exclusively on X_train, never X_test."""
    rng = np.random.default_rng(123)
    X_train = pd.DataFrame({"x": rng.normal(loc=0.0, scale=1.0, size=60)})
    X_test = pd.DataFrame({"x": rng.normal(loc=1000.0, scale=1.0, size=20)})
    ftypes = {"x": "numerical"}

    prep = UniversalPreprocessor(scale_numerical=True)
    prep.fit(X_train, ftypes)

    train_mean = X_train["x"].mean()
    train_std = X_train["x"].std(ddof=0)
    expected_test_scaled = (X_test["x"].to_numpy() - train_mean) / train_std

    actual_test_scaled = prep.transform(X_test)[:, 0]
    np.testing.assert_allclose(actual_test_scaled, expected_test_scaled, rtol=1e-6)
    assert actual_test_scaled.mean() > 500.0


def test_unknown_category_handled(sample_classification_df: pd.DataFrame) -> None:
    """Unseen categories during transform map to -1 without raising an error."""
    X = sample_classification_df.drop(columns=["label"])
    y = sample_classification_df["label"]
    feature_types = DataProfiler().profile(X, y)["feature_types"]

    X_train = X.iloc[:80].copy()
    X_test = X.iloc[80:].copy()
    X_test.loc[X_test.index[0], "cat_1"] = "UNSEEN_CATEGORY_XYZ"

    prep = UniversalPreprocessor()
    prep.fit(X_train, feature_types)
    arr_test = prep.transform(X_test)

    names = prep.get_feature_names()
    cat_1_idx = names.index("cat_1")
    assert arr_test[0, cat_1_idx] == -1.0


def test_missing_values_imputed(missing_values_df: pd.DataFrame) -> None:
    """Missing numerical and categorical values are imputed cleanly with no NaNs."""
    X = missing_values_df.drop(columns=["label"])
    y = missing_values_df["label"]
    feature_types = DataProfiler().profile(X, y)["feature_types"]

    prep = UniversalPreprocessor()
    arr = prep.fit_transform(X, feature_types)

    assert not np.isnan(arr).any()
    assert arr.shape[0] == len(X)


def test_feature_names_preserved(sample_classification_df: pd.DataFrame) -> None:
    """Feature names are accessible after fit and match output column count."""
    X = sample_classification_df.drop(columns=["label"])
    y = sample_classification_df["label"]
    feature_types = DataProfiler().profile(X, y)["feature_types"]

    prep = UniversalPreprocessor()
    arr = prep.fit_transform(X, feature_types)
    names = prep.get_feature_names()

    assert len(names) == arr.shape[1]
    assert set(names) == set(X.columns)


def test_safe_dtype_conversion_float64_float32() -> None:
    """Exact float32-representable values pass both safe_downcast and SchemaValidator."""
    exact_series = pd.Series([0.5, 1.25, -3.5, 4.0], dtype="float64", name="exact")

    assert SchemaValidator.validate_safe_dtype_conversion(exact_series, np.float32) is True
    downcasted = safe_downcast(exact_series, np.float32)
    assert downcasted.dtype == np.float32
    np.testing.assert_array_equal(
        downcasted.astype("float64").to_numpy(), exact_series.to_numpy()
    )


def test_unsafe_dtype_rejected() -> None:
    """High-precision float64 values that lose precision in float32 are rejected."""
    high_prec = pd.Series(
        [1.123456789012345, 2.987654321098765], dtype="float64", name="precise"
    )

    assert SchemaValidator.validate_safe_dtype_conversion(high_prec, np.float32) is False
    with pytest.raises(ValueError, match="precision loss"):
        safe_downcast(high_prec, np.float32)


def test_schema_valid_after_transform(sample_classification_df: pd.DataFrame) -> None:
    """Transformed train/test arrays have consistent shape, finite float values, and valid DataFrame schema."""
    X = sample_classification_df.drop(columns=["label"])
    y = sample_classification_df["label"]
    feature_types = DataProfiler().profile(X, y)["feature_types"]

    X_train, X_test = X.iloc[:70], X.iloc[70:]
    prep = UniversalPreprocessor()
    arr_train = prep.fit_transform(X_train, feature_types)
    arr_test = prep.transform(X_test)
    names = prep.get_feature_names()

    assert arr_train.shape[1] == arr_test.shape[1] == len(names)
    assert np.isfinite(arr_train).all()
    assert np.isfinite(arr_test).all()

    df_train_out = pd.DataFrame(arr_train, columns=names)
    df_test_out = pd.DataFrame(arr_test, columns=names)
    expected_schema = {col: str(dt) for col, dt in df_train_out.dtypes.items()}
    assert SchemaValidator.validate_dataframe(df_test_out, expected_schema) is True
