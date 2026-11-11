"""
Data profiling for the Carbon-Aware AutoML framework.

Analyses an input DataFrame and its target series to produce a comprehensive
profile that informs preprocessing decisions, leakage detection, and the
experiment report.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

_HIGH_CARDINALITY_THRESHOLD = 50   # unique values to call object col "high_cardinality"
_TEXT_LIKE_MEAN_LEN = 50           # mean char length above which a col is "text_like"
_MAX_UNIQUE_CLASSIFICATION = 20    # unique target values → classification vs regression
_OUTLIER_IQR_FACTOR = 3.0          # IQR multiplier for outlier detection


# --------------------------------------------------------------------------- #
# Profiler                                                                     #
# --------------------------------------------------------------------------- #


class DataProfiler:
    """
    Compute a rich statistical and structural profile of a dataset.

    The profiler is stateless — call :meth:`profile` as many times as needed.

    Examples
    --------
    >>> profiler = DataProfiler()
    >>> report = profiler.profile(X, y)
    >>> print(report["task_type_suggestion"])
    'classification'
    """

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def profile(self, df: pd.DataFrame, target: pd.Series) -> dict[str, Any]:
        """
        Return a comprehensive profile of *df* and *target*.

        Parameters
        ----------
        df:
            Feature DataFrame. Must not contain the target column.
        target:
            Target series (may be categorical or numerical).

        Returns
        -------
        dict with keys:
            shape, memory_mb, feature_types, missing_values, duplicate_rows,
            class_distribution (classification only), target_distribution,
            outliers, numerical_stats, task_type_suggestion, recommendations.
        """
        # --- guard: empty dataframe ----------------------------------- #
        if df.empty:
            return self._empty_profile(df)

        feature_types = self._detect_feature_types(df)
        missing_values = self._missing_values(df)
        duplicate_rows = int(df.duplicated().sum())
        outliers = self._detect_outliers(df, feature_types)
        numerical_stats = self._numerical_stats(df, feature_types)
        task_type = self._suggest_task_type(target)
        target_dist = self._target_distribution(target, task_type)
        class_dist = self._class_distribution(target, task_type)
        recommendations = self._build_recommendations(df, feature_types, missing_values, duplicate_rows)

        return {
            "shape": df.shape,
            "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 ** 2), 4),
            "feature_types": feature_types,
            "missing_values": missing_values,
            "duplicate_rows": duplicate_rows,
            "class_distribution": class_dist,
            "target_distribution": target_dist,
            "outliers": outliers,
            "numerical_stats": numerical_stats,
            "task_type_suggestion": task_type,
            "recommendations": recommendations,
        }

    # ------------------------------------------------------------------ #
    # Feature-type detection                                               #
    # ------------------------------------------------------------------ #

    def _detect_feature_types(self, df: pd.DataFrame) -> dict[str, str]:
        """
        Classify each column into one of:
        numerical, categorical, high_cardinality_categorical,
        datetime, text_like, identifier_like, constant, binary.
        """
        types: dict[str, str] = {}
        n_rows = len(df)

        for col in df.columns:
            series = df[col]
            n_unique = series.nunique(dropna=True)

            # Constant
            if n_unique <= 1:
                types[col] = "constant"
                continue

            # Datetime
            if pd.api.types.is_datetime64_any_dtype(series):
                types[col] = "datetime"
                continue

            # Numerical
            if pd.api.types.is_numeric_dtype(series):
                if n_unique == 2:
                    types[col] = "binary"
                else:
                    types[col] = "numerical"
                continue

            # Object / string columns
            if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                str_series = series.dropna().astype(str)

                # Text-like: mean string length > threshold
                if len(str_series) > 0:
                    mean_len = str_series.str.len().mean()
                    if mean_len > _TEXT_LIKE_MEAN_LEN:
                        types[col] = "text_like"
                        continue

                # Identifier-like: every non-null value is unique
                if n_unique == n_rows and n_rows > 2:
                    types[col] = "identifier_like"
                    continue

                # High cardinality
                if n_unique > _HIGH_CARDINALITY_THRESHOLD:
                    types[col] = "high_cardinality_categorical"
                    continue

                # Binary categorical
                if n_unique == 2:
                    types[col] = "binary"
                    continue

                types[col] = "categorical"
                continue

            # Categorical dtype
            if hasattr(series, "cat"):
                if n_unique == 2:
                    types[col] = "binary"
                else:
                    types[col] = "categorical"
                continue

            # Fallback
            types[col] = "categorical"

        return types

    # ------------------------------------------------------------------ #
    # Missing values                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _missing_values(df: pd.DataFrame) -> dict[str, int]:
        """Return count of missing values per column."""
        return {col: int(df[col].isna().sum()) for col in df.columns}

    # ------------------------------------------------------------------ #
    # Outlier detection                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _detect_outliers(df: pd.DataFrame, feature_types: dict[str, str]) -> dict[str, int]:
        """
        Count values beyond ±3 IQR from Q1/Q3 for each numerical column.
        """
        outliers: dict[str, int] = {}
        for col, ftype in feature_types.items():
            if ftype != "numerical":
                continue
            if not pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
                continue
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(series) < 4:
                outliers[col] = 0
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                outliers[col] = 0
                continue
            lower = q1 - _OUTLIER_IQR_FACTOR * iqr
            upper = q3 + _OUTLIER_IQR_FACTOR * iqr
            outliers[col] = int(((series < lower) | (series > upper)).sum())
        return outliers

    # ------------------------------------------------------------------ #
    # Numerical statistics                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _numerical_stats(df: pd.DataFrame, feature_types: dict[str, str]) -> dict[str, dict[str, float]]:
        """Return min/max/mean/std for each numerical column."""
        stats: dict[str, dict[str, float]] = {}
        for col, ftype in feature_types.items():
            if ftype != "numerical":
                continue
            series = df[col].dropna()
            if len(series) == 0:
                stats[col] = {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
            else:
                stats[col] = {
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "mean": float(series.mean()),
                    "std": float(series.std()) if len(series) > 1 else 0.0,
                }
        return stats

    # ------------------------------------------------------------------ #
    # Target / class distribution                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _suggest_task_type(target: pd.Series) -> str:
        """Suggest 'classification' or 'regression' based on target cardinality."""
        if (
            pd.api.types.is_object_dtype(target)
            or pd.api.types.is_string_dtype(target)
            or pd.api.types.is_bool_dtype(target)
            or hasattr(target, "cat")
        ):
            return "classification"
        n_unique = target.nunique(dropna=True)
        return "classification" if n_unique <= _MAX_UNIQUE_CLASSIFICATION else "regression"

    @staticmethod
    def _target_distribution(target: pd.Series, task_type: str) -> dict[str, Any]:
        """Return basic descriptive statistics of the target."""
        clean = target.dropna()
        if len(clean) == 0:
            return {"n_valid": 0, "n_missing": int(target.isna().sum())}

        base: dict[str, Any] = {
            "n_valid": int(len(clean)),
            "n_missing": int(target.isna().sum()),
            "n_unique": int(clean.nunique()),
        }

        if task_type == "regression":
            base.update({
                "min": float(clean.min()),
                "max": float(clean.max()),
                "mean": float(clean.mean()),
                "std": float(clean.std()) if len(clean) > 1 else 0.0,
                "median": float(clean.median()),
            })
        else:
            vc = clean.value_counts()
            base["value_counts"] = {str(k): int(v) for k, v in vc.items()}

        return base

    @staticmethod
    def _class_distribution(target: pd.Series, task_type: str) -> dict[str, Any] | None:
        """Return class counts and imbalance ratio for classification tasks."""
        if task_type != "classification":
            return None

        vc = target.value_counts()
        if len(vc) == 0:
            return {"counts": {}, "imbalance_ratio": None}

        counts = {str(k): int(v) for k, v in vc.items()}
        max_count = int(vc.max())
        min_count = int(vc.min())
        imbalance_ratio = round(max_count / min_count, 4) if min_count > 0 else None

        return {"counts": counts, "imbalance_ratio": imbalance_ratio}

    # ------------------------------------------------------------------ #
    # Recommendations                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_recommendations(
        df: pd.DataFrame,
        feature_types: dict[str, str],
        missing_values: dict[str, int],
        duplicate_rows: int,
    ) -> list[str]:
        """Generate actionable string recommendations."""
        recs: list[str] = []
        n_rows = len(df)

        for col, ftype in feature_types.items():
            if ftype == "identifier_like":
                recs.append(
                    f"Column '{col}' appears identifier-like (all unique values) — "
                    "consider excluding it to prevent leakage."
                )
            elif ftype == "constant":
                recs.append(
                    f"Column '{col}' is constant (single unique value) — "
                    "it carries no predictive signal and should be dropped."
                )
            elif ftype == "text_like":
                recs.append(
                    f"Column '{col}' appears text-like (mean string length > {_TEXT_LIKE_MEAN_LEN}) — "
                    "consider NLP feature extraction or exclusion."
                )
            elif ftype == "high_cardinality_categorical":
                recs.append(
                    f"Column '{col}' is high-cardinality categorical (>{_HIGH_CARDINALITY_THRESHOLD} unique) — "
                    "OrdinalEncoder applied; consider target encoding or frequency encoding."
                )

            # High missingness
            miss = missing_values.get(col, 0)
            if n_rows > 0 and (miss / n_rows) > 0.5:
                recs.append(
                    f"Column '{col}' has {miss}/{n_rows} ({miss/n_rows:.1%}) missing values — "
                    "consider imputation strategy or dropping."
                )

        if duplicate_rows > 0:
            recs.append(
                f"Dataset contains {duplicate_rows} duplicate rows — "
                "review whether they are legitimate replicates or data-entry errors."
            )

        return recs

    # ------------------------------------------------------------------ #
    # Edge case: empty dataframe                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _empty_profile(df: pd.DataFrame | None = None) -> dict[str, Any]:
        """Return a minimal profile for an empty DataFrame."""
        shape = df.shape if df is not None else (0, 0)
        missing = {col: 0 for col in df.columns} if df is not None else {}
        return {
            "shape": shape,
            "memory_mb": 0.0,
            "feature_types": {},
            "missing_values": missing,
            "duplicate_rows": 0,
            "class_distribution": None,
            "target_distribution": {"n_valid": 0, "n_missing": 0},
            "outliers": {},
            "numerical_stats": {},
            "task_type_suggestion": "classification",
            "recommendations": ["Input DataFrame is empty — no profiling performed."],
        }
