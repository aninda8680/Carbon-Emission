"""
Universal preprocessing pipeline for the Carbon-Aware AutoML framework.

Builds a scikit-learn ColumnTransformer that handles numerical, categorical,
and high-cardinality categorical features uniformly, while respecting the
"fit only on training data" principle.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def safe_downcast(series: pd.Series, target_dtype: Any) -> pd.Series:
    """
    Downcast *series* to *target_dtype* only if every value survives a
    round-trip conversion without precision loss.

    Parameters
    ----------
    series:
        Input series.
    target_dtype:
        NumPy or pandas dtype to try (e.g. ``np.float32``).

    Returns
    -------
    pd.Series
        Downcast series if round-trip is lossless.

    Raises
    ------
    ValueError
        If ANY value does not survive the round-trip exactly or conversion fails.
    """
    try:
        converted = series.astype(target_dtype)
        round_tripped = converted.astype(series.dtype)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"Cannot safely downcast column '{series.name}' to {target_dtype}: {exc}"
        ) from exc

    orig_na = series.isna().to_numpy()
    conv_na = converted.isna().to_numpy()
    rt_na = round_tripped.isna().to_numpy()
    if not (np.array_equal(orig_na, conv_na) and np.array_equal(orig_na, rt_na)):
        raise ValueError(
            f"Cannot safely downcast column '{series.name}' to {target_dtype}: "
            "missing-value positions changed during round-trip conversion."
        )

    valid_mask = ~orig_na
    if np.any(valid_mask):
        orig_vals = series.iloc[valid_mask].to_numpy()
        rt_vals = round_tripped.iloc[valid_mask].to_numpy()
        try:
            is_float = np.issubdtype(orig_vals.dtype, np.floating) or np.issubdtype(
                orig_vals.dtype, np.complexfloating
            )
        except TypeError:
            is_float = False

        equal = (
            np.array_equal(orig_vals, rt_vals, equal_nan=True)
            if is_float
            else np.array_equal(orig_vals, rt_vals)
        )
        if not equal:
            raise ValueError(
                f"Cannot safely downcast column '{series.name}' to {target_dtype}: "
                "precision loss detected in round-trip conversion."
            )

    return converted


# --------------------------------------------------------------------------- #
# Preprocessor                                                                 #
# --------------------------------------------------------------------------- #


class UniversalPreprocessor:
    """
    Fit-once, transform-many preprocessing pipeline.

    Applies:
    * **Numerical** columns: ``SimpleImputer(strategy='median')`` →
      optional ``StandardScaler`` (enabled by default).
    * **Categorical** columns: ``SimpleImputer(strategy='most_frequent')`` →
      ``OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)``.
    * **High-cardinality categorical** columns: same as categorical.

    All other column types (identifier_like, constant, datetime, text_like,
    binary) are excluded from the transformer and dropped — the caller should
    exclude such columns before calling :meth:`fit`.

    Parameters
    ----------
    scale_numerical:
        If ``True`` (default), apply ``StandardScaler`` after imputation for
        numerical features.

    Examples
    --------
    >>> preprocessor = UniversalPreprocessor()
    >>> X_train_arr = preprocessor.fit_transform(X_train, feature_types)
    >>> X_test_arr  = preprocessor.transform(X_test)
    """

    _NUMERICAL_TYPES = {"numerical"}
    _CATEGORICAL_TYPES = {"categorical", "binary"}
    _HIGH_CARD_TYPES = {"high_cardinality_categorical"}

    def __init__(self, scale_numerical: bool = True) -> None:
        self._scale_numerical = scale_numerical
        self._transformer: ColumnTransformer | None = None
        self._feature_names_in: list[str] = []
        self._is_fitted: bool = False

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def fit(self, X_train: pd.DataFrame, feature_types: dict[str, str]) -> None:
        """
        Build and fit the ColumnTransformer on *X_train*.

        Parameters
        ----------
        X_train:
            Training feature matrix. Must not contain the target column.
        feature_types:
            Mapping of column name → feature type string as produced by
            :class:`~carbon_automl.core.data_profiler.DataProfiler`.

        Raises
        ------
        ValueError
            If *X_train* is empty or no processable columns are found.
        """
        if X_train.empty:
            raise ValueError("X_train is empty — cannot fit preprocessor.")

        self._feature_names_in = list(X_train.columns)

        numerical_cols = self._select_columns(X_train, feature_types, self._NUMERICAL_TYPES)
        categorical_cols = self._select_columns(X_train, feature_types, self._CATEGORICAL_TYPES)
        high_card_cols = self._select_columns(X_train, feature_types, self._HIGH_CARD_TYPES)

        transformers: list[tuple[str, Any, list[str]]] = []

        if numerical_cols:
            steps: list[tuple[str, Any]] = [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ]
            if self._scale_numerical:
                steps.append(("scaler", StandardScaler()))
            transformers.append(("numerical", Pipeline(steps), numerical_cols))

        if categorical_cols:
            transformers.append(
                ("categorical", self._build_categorical_pipeline(), categorical_cols)
            )

        if high_card_cols:
            transformers.append(
                ("high_cardinality", self._build_categorical_pipeline(), high_card_cols)
            )

        if not transformers:
            raise ValueError(
                "No processable columns found (numerical / categorical / "
                "high_cardinality_categorical). Check feature_types mapping."
            )

        self._transformer = ColumnTransformer(
            transformers=transformers,
            remainder="drop",          # drop unknown / excluded column types
            sparse_threshold=0.0,
            verbose_feature_names_out=False,
        )

        # Fit ONLY on training data
        self._transformer.fit(X_train)
        self._is_fitted = True

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Apply the fitted transformer to *X*.

        Parameters
        ----------
        X:
            Feature DataFrame (may be training, validation, or test split).

        Returns
        -------
        np.ndarray
            Dense float64 array of transformed features.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called yet.
        """
        self._check_fitted()
        return self._transformer.transform(X)  # type: ignore[union-attr]

    def fit_transform(self, X_train: pd.DataFrame, feature_types: dict[str, str]) -> np.ndarray:
        """
        Fit on *X_train* then immediately transform it.

        Equivalent to calling :meth:`fit` then :meth:`transform`, but slightly
        more efficient because scikit-learn reuses the fitted state.

        Parameters
        ----------
        X_train:
            Training feature matrix.
        feature_types:
            Mapping of column name → feature type string.

        Returns
        -------
        np.ndarray
            Transformed training array.
        """
        self.fit(X_train, feature_types)
        return self.transform(X_train)

    def get_feature_names(self) -> list[str]:
        """
        Return the feature names produced by the fitted transformer.

        Returns
        -------
        list[str]
            Ordered list of output feature names, mirroring the column order
            of arrays returned by :meth:`transform`.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called yet.
        """
        self._check_fitted()
        try:
            names: list[str] = list(
                self._transformer.get_feature_names_out()  # type: ignore[union-attr]
            )
        except AttributeError:
            # Fallback for older scikit-learn versions
            names = []
            for _, _, cols in self._transformer.transformers_:  # type: ignore[union-attr]
                if isinstance(cols, list):
                    names.extend(cols)
        return names

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _check_fitted(self) -> None:
        if not self._is_fitted or self._transformer is None:
            raise RuntimeError(
                "UniversalPreprocessor is not fitted. Call fit() or fit_transform() first."
            )

    @staticmethod
    def _build_categorical_pipeline() -> Pipeline:
        """Build the shared imputation + ordinal encoding pipeline for categorical columns."""
        return Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ])

    @staticmethod
    def _select_columns(
        df: pd.DataFrame,
        feature_types: dict[str, str],
        type_set: set[str],
    ) -> list[str]:
        """Return columns in *df* whose feature_type is in *type_set*."""
        return [
            col
            for col in df.columns
            if feature_types.get(col, "") in type_set
        ]
