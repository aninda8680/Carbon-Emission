"""
Automated data-leakage detection for the Carbon-Aware AutoML framework.

Implements seven complementary leakage checks covering single-feature predictive
power, target correlation, mutual-information concentration, perfect per-class
range separation (e.g. WHO MUAC diagnostic criteria), unexcluded identifiers,
duplicate observations, and constant features.

The detector NEVER automatically deletes or modifies features — it only emits
structured warnings for human review.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from carbon_automl.core.data_profiler import DataProfiler


class LeakageDetector:
    """
    Detect potential target and split leakage across tabular features.

    All checks are read-only: ``X`` and ``y`` are never mutated and no columns
    are ever dropped automatically.

    Examples
    --------
    >>> detector = LeakageDetector()
    >>> warnings = detector.detect(X, y, feature_types, task_type="classification")
    """

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def detect(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_types: dict[str, str] | None = None,
        task_type: str = "auto",
        single_feature_threshold: float = 0.95,
        correlation_threshold: float = 0.95,
    ) -> list[dict[str, Any]]:
        """
        Run all leakage checks on *(X, y)* and return a list of warning dicts.

        Parameters
        ----------
        X:
            Feature matrix (must not contain the target column).
        y:
            Target series.
        feature_types:
            Mapping of column name -> feature type from :class:`DataProfiler`.
            Inferred automatically if omitted or incomplete.
        task_type:
            ``"classification"``, ``"regression"``, or ``"auto"``.
        single_feature_threshold:
            Score threshold (accuracy for classification, R² for regression)
            above which a single-feature DecisionTree flags critical leakage.
        correlation_threshold:
            Absolute Pearson correlation threshold above which a numerical
            feature is flagged.

        Returns
        -------
        list[dict]
            Each warning dict contains:
            ``feature``, ``detection_method``, ``evidence``, ``severity``,
            ``recommended_investigation``.
        """
        if X.empty or len(X.columns) == 0 or len(y) == 0:
            return []

        profiler = DataProfiler()
        inferred_types = profiler._detect_feature_types(X)
        resolved_types: dict[str, str] = {**inferred_types, **(feature_types or {})}

        resolved_task = (
            profiler._suggest_task_type(y)
            if task_type == "auto"
            else task_type
        )

        warnings: list[dict[str, Any]] = []

        # ---- Unsupervised structural checks (always safe to run) ----- #
        warnings.extend(self._check_constant_features(X, resolved_types))
        warnings.extend(self._check_identifier_leakage(X, resolved_types))
        warnings.extend(self._check_duplicate_observations(X))

        # ---- Supervised checks require >= 2 unique non-null targets --- #
        valid_y_mask = ~y.isna()
        y_valid = y.loc[valid_y_mask]
        if len(y_valid) < 4 or y_valid.nunique() <= 1:
            return warnings

        X_valid = X.loc[valid_y_mask]

        warnings.extend(
            self._check_single_feature_performance(
                X_valid,
                y_valid,
                resolved_types,
                resolved_task,
                single_feature_threshold,
            )
        )
        warnings.extend(
            self._check_target_correlation(
                X_valid,
                y_valid,
                resolved_types,
                resolved_task,
                correlation_threshold,
            )
        )
        warnings.extend(
            self._check_mutual_information_anomaly(
                X_valid,
                y_valid,
                resolved_types,
                resolved_task,
            )
        )
        if resolved_task == "classification":
            warnings.extend(
                self._check_perfect_range_separation(X_valid, y_valid, resolved_types)
            )

        return warnings

    # ------------------------------------------------------------------ #
    # Check 1: Single-feature performance                                  #
    # ------------------------------------------------------------------ #

    def _check_single_feature_performance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_types: dict[str, str],
        task_type: str,
        threshold: float,
    ) -> list[dict[str, Any]]:
        """Train Dummy + DecisionTree(max_depth=3) on each feature individually."""
        warnings: list[dict[str, Any]] = []
        skip_types = {"constant", "identifier_like", "text_like", "datetime"}

        y_encoded, is_clf = self._prepare_target(y, task_type)
        if y_encoded is None:
            return warnings

        cv = self._build_cv(y_encoded, is_clf)
        if cv is None:
            return warnings

        for col in X.columns:
            ftype = feature_types.get(col, "")
            if ftype in skip_types:
                continue

            col_series = X[col]
            if col_series.dropna().empty or col_series.nunique(dropna=True) <= 1:
                continue

            X_col, is_numeric = self._prepare_single_column(col_series)
            dt_pipe, dummy_pipe = self._build_single_feature_pipelines(
                is_numeric=is_numeric,
                is_clf=is_clf,
            )
            scoring = "accuracy" if is_clf else "r2"

            try:
                dummy_scores = cross_val_score(
                    dummy_pipe, X_col, y_encoded, cv=cv, scoring=scoring
                )
                dt_scores = cross_val_score(
                    dt_pipe, X_col, y_encoded, cv=cv, scoring=scoring
                )
            except Exception:
                continue

            dummy_score = float(np.mean(dummy_scores))
            dt_score = float(np.mean(dt_scores))

            if dt_score > threshold and dt_score > dummy_score + 1e-6:
                metric_name = "accuracy" if is_clf else "R²"
                baseline_name = "DummyClassifier" if is_clf else "DummyRegressor"
                model_name = "DecisionTree" if is_clf else "DecisionTreeRegressor"
                warnings.append(
                    self._make_warning(
                        feature=str(col),
                        detection_method="single_feature_performance",
                        evidence=(
                            f"{col} achieves {dt_score:.4f} {metric_name} alone using "
                            f"{model_name}(max_depth=3) ({baseline_name} baseline: "
                            f"{dummy_score:.4f})."
                        ),
                        severity="critical",
                        recommended_investigation=(
                            f"Verify whether '{col}' is a post-outcome measurement, "
                            "diagnostic criterion, or direct proxy for the target; "
                            "consider adding it to exclude_columns."
                        ),
                    )
                )

        return warnings

    # ------------------------------------------------------------------ #
    # Check 2: Target correlation                                          #
    # ------------------------------------------------------------------ #

    def _check_target_correlation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_types: dict[str, str],
        task_type: str,
        threshold: float,
    ) -> list[dict[str, Any]]:
        """Compute |pearson_r| between each numerical feature and encoded target."""
        warnings: list[dict[str, Any]] = []

        # Build numeric target representations for correlation checks
        target_vectors: list[np.ndarray] = []
        if task_type == "regression" and pd.api.types.is_numeric_dtype(y):
            target_vectors.append(y.to_numpy(dtype=float))
        else:
            le = LabelEncoder()
            y_codes = le.fit_transform(y.astype(str)).astype(float)
            target_vectors.append(y_codes)
            classes = np.unique(y_codes)
            if len(classes) > 2:
                for cls in classes:
                    target_vectors.append((y_codes == cls).astype(float))

        for col in X.columns:
            ftype = feature_types.get(col, "")
            if ftype not in {"numerical", "binary"}:
                continue
            if not pd.api.types.is_numeric_dtype(X[col]):
                continue

            x_vals = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=float)
            valid = ~np.isnan(x_vals)
            if np.sum(valid) < 3:
                continue

            x_clean = x_vals[valid]
            if np.std(x_clean) == 0:
                continue

            max_abs_r = 0.0
            for t_vec in target_vectors:
                t_clean = t_vec[valid]
                if np.std(t_clean) == 0:
                    continue
                r = float(np.corrcoef(x_clean, t_clean)[0, 1])
                if not np.isnan(r):
                    max_abs_r = max(max_abs_r, abs(r))

            if max_abs_r > threshold:
                severity = "critical" if max_abs_r >= 0.99 else "high"
                warnings.append(
                    self._make_warning(
                        feature=str(col),
                        detection_method="target_correlation",
                        evidence=(
                            f"Feature '{col}' has |Pearson r| = {max_abs_r:.4f} with "
                            f"the encoded target (threshold: {threshold:.2f})."
                        ),
                        severity=severity,
                        recommended_investigation=(
                            f"Inspect '{col}' for target leakage or collinearity "
                            "with a post-target variable."
                        ),
                    )
                )

        return warnings

    # ------------------------------------------------------------------ #
    # Check 3: Mutual information anomaly                                  #
    # ------------------------------------------------------------------ #

    def _check_mutual_information_anomaly(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_types: dict[str, str],
        task_type: str,
    ) -> list[dict[str, Any]]:
        """Flag if a single feature holds > 95% of total mutual information."""
        warnings: list[dict[str, Any]] = []
        skip_types = {"constant", "identifier_like", "text_like", "datetime"}

        usable_cols: list[str] = []
        encoded_cols: list[np.ndarray] = []
        discrete_mask: list[bool] = []

        for col in X.columns:
            ftype = feature_types.get(col, "")
            if ftype in skip_types:
                continue
            series = X[col]
            if series.dropna().empty or series.nunique(dropna=True) <= 1:
                continue

            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
                median_val = float(np.nanmedian(vals))
                vals = np.where(np.isnan(vals), median_val, vals)
                encoded_cols.append(vals)
                discrete_mask.append(ftype == "binary")
            else:
                filled = series.astype(object).where(~series.isna(), "__MISSING__").astype(str)
                n_unique = filled.nunique()
                # High-cardinality quasi-unique strings inflate discrete plug-in MI;
                # skip quasi-identifiers (>50% unique) and treat >20-cat as non-discrete.
                if n_unique > 0.5 * len(series):
                    continue
                codes = pd.factorize(filled, sort=False)[0].astype(float)
                encoded_cols.append(codes)
                discrete_mask.append(n_unique <= 20)

            usable_cols.append(str(col))

        # Need at least 2 features for "share of total MI sum" to be meaningful
        if len(usable_cols) < 2 or len(y) < 3:
            return warnings

        X_mi = np.column_stack(encoded_cols)
        n_neighbors = min(3, max(1, len(y) - 1))

        y_encoded, is_clf = self._prepare_target(y, task_type)
        if y_encoded is None:
            return warnings

        try:
            if is_clf:
                mi_scores = mutual_info_classif(
                    X_mi,
                    y_encoded,
                    discrete_features=np.array(discrete_mask, dtype=bool),
                    n_neighbors=n_neighbors,
                    random_state=42,
                )
            else:
                mi_scores = mutual_info_regression(
                    X_mi,
                    y_encoded,
                    discrete_features=np.array(discrete_mask, dtype=bool),
                    n_neighbors=n_neighbors,
                    random_state=42,
                )
        except Exception:
            return warnings

        total_mi = float(np.sum(mi_scores))
        if total_mi <= 1e-9:
            return warnings

        for col_name, mi_val in zip(usable_cols, mi_scores):
            share = float(mi_val) / total_mi
            if share > 0.95:
                warnings.append(
                    self._make_warning(
                        feature=col_name,
                        detection_method="mutual_information_anomaly",
                        evidence=(
                            f"Feature '{col_name}' accounts for {share:.1%} of total "
                            f"mutual information across features "
                            f"(MI={mi_val:.4f} of total {total_mi:.4f})."
                        ),
                        severity="high",
                        recommended_investigation=(
                            f"Investigate why '{col_name}' dominates all other features "
                            "in mutual information; check for deterministic or leaked rules."
                        ),
                    )
                )

        return warnings

    # ------------------------------------------------------------------ #
    # Check 4: Perfect range separation                                    #
    # ------------------------------------------------------------------ #

    def _check_perfect_range_separation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_types: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Check whether per-class feature value ranges are strictly disjoint."""
        warnings: list[dict[str, Any]] = []

        for col in X.columns:
            ftype = feature_types.get(col, "")
            if ftype not in {"numerical", "binary"}:
                continue
            if not pd.api.types.is_numeric_dtype(X[col]):
                continue

            vals = pd.to_numeric(X[col], errors="coerce")
            valid_mask = ~vals.isna()
            if valid_mask.sum() < 2:
                continue

            vals_clean = vals.loc[valid_mask]
            if vals_clean.nunique() <= 1:
                continue

            y_clean = y.loc[valid_mask]
            intervals: list[tuple[float, float, str]] = []
            for cls_val, grp in vals_clean.groupby(y_clean, observed=True):
                if grp.empty:
                    continue
                intervals.append((float(grp.min()), float(grp.max()), str(cls_val)))

            if len(intervals) < 2:
                continue

            intervals.sort(key=lambda item: (item[0], item[1]))
            non_overlapping = all(
                intervals[i][1] < intervals[i + 1][0]
                for i in range(len(intervals) - 1)
            )

            if non_overlapping:
                range_summary = ", ".join(
                    f"{cls}: [{lo:.3g}, {hi:.3g}]" for lo, hi, cls in intervals
                )
                warnings.append(
                    self._make_warning(
                        feature=str(col),
                        detection_method="perfect_range_separation",
                        evidence=(
                            f"Feature '{col}' has completely non-overlapping value "
                            f"ranges per class ({range_summary})."
                        ),
                        severity="critical",
                        recommended_investigation=(
                            f"'{col}' deterministically separates target classes and "
                            "is almost certainly a diagnostic rule or target proxy; "
                            "exclude it unless intentionally evaluating a rule-based baseline."
                        ),
                    )
                )

        return warnings

    # ------------------------------------------------------------------ #
    # Check 5: Identifier leakage                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _check_identifier_leakage(
        X: pd.DataFrame,
        feature_types: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Flag identifier-like columns that remain in the feature matrix."""
        warnings: list[dict[str, Any]] = []
        for col in X.columns:
            if feature_types.get(col) == "identifier_like":
                warnings.append(
                    LeakageDetector._make_warning(
                        feature=str(col),
                        detection_method="identifier_leakage",
                        evidence=(
                            f"Feature '{col}' has all unique values "
                            f"({X[col].nunique(dropna=True)}/{len(X)}) and appears to be an identifier."
                        ),
                        severity="high",
                        recommended_investigation=(
                            f"Add '{col}' to exclude_columns to prevent entity "
                            "memorisation and split leakage."
                        ),
                    )
                )
        return warnings

    # ------------------------------------------------------------------ #
    # Check 6: Duplicate observations                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _check_duplicate_observations(X: pd.DataFrame) -> list[dict[str, Any]]:
        """Flag duplicate feature rows that could leak across CV splits."""
        if len(X) == 0:
            return []

        n_dups = int(X.duplicated().sum())
        if n_dups == 0:
            return []

        dup_ratio = n_dups / len(X)
        severity = "high" if dup_ratio > 0.1 else "medium"
        return [
            LeakageDetector._make_warning(
                feature="__dataset__",
                detection_method="duplicate_observations",
                evidence=(
                    f"Dataset contains {n_dups} duplicate feature row(s) "
                    f"({dup_ratio:.1%} of {len(X)} rows), which can leak between "
                    "training and validation folds."
                ),
                severity=severity,
                recommended_investigation=(
                    "Deduplicate rows before splitting or use group_kfold on an "
                    "entity/subject identifier."
                ),
            )
        ]

    # ------------------------------------------------------------------ #
    # Check 7: Constant features                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _check_constant_features(
        X: pd.DataFrame,
        feature_types: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Flag constant or all-missing columns."""
        warnings: list[dict[str, Any]] = []
        for col in X.columns:
            if feature_types.get(col) == "constant" or X[col].nunique(dropna=True) <= 1:
                n_missing = int(X[col].isna().sum())
                detail = (
                    "all values are missing"
                    if n_missing == len(X)
                    else f"only {X[col].nunique(dropna=True)} unique non-null value"
                )
                warnings.append(
                    LeakageDetector._make_warning(
                        feature=str(col),
                        detection_method="constant_feature",
                        evidence=f"Feature '{col}' is constant ({detail}).",
                        severity="low",
                        recommended_investigation=(
                            f"Consider adding '{col}' to exclude_columns since it "
                            "carries zero predictive variance."
                        ),
                    )
                )
        return warnings

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_warning(
        feature: str,
        detection_method: str,
        evidence: str,
        severity: str,
        recommended_investigation: str,
    ) -> dict[str, Any]:
        return {
            "feature": feature,
            "detection_method": detection_method,
            "evidence": evidence,
            "severity": severity,
            "recommended_investigation": recommended_investigation,
        }

    @staticmethod
    def _prepare_target(y: pd.Series, task_type: str) -> tuple[np.ndarray | None, bool]:
        if task_type == "regression":
            y_num = pd.to_numeric(y, errors="coerce")
            if y_num.isna().any():
                return None, False
            return y_num.to_numpy(dtype=float), False

        le = LabelEncoder()
        y_enc = le.fit_transform(y.astype(str))
        return y_enc, True

    @staticmethod
    def _build_cv(y_encoded: np.ndarray, is_clf: bool) -> Any:
        n_samples = len(y_encoded)
        if n_samples < 4:
            return None

        if is_clf:
            _, counts = np.unique(y_encoded, return_counts=True)
            min_count = int(np.min(counts))
            if min_count < 2:
                return None
            n_splits = min(5, min_count)
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        n_splits = min(5, n_samples)
        if n_splits < 2:
            return None
        return KFold(n_splits=n_splits, shuffle=True, random_state=42)

    @staticmethod
    def _prepare_single_column(series: pd.Series) -> tuple[np.ndarray, bool]:
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float).reshape(-1, 1)
            return arr, True
        if pd.api.types.is_bool_dtype(series):
            arr = series.astype(float).to_numpy().reshape(-1, 1)
            return arr, True

        arr_obj = (
            series.astype(object)
            .where(~series.isna(), "__MISSING__")
            .astype(str)
            .to_numpy()
            .reshape(-1, 1)
        )
        return arr_obj, False

    @staticmethod
    def _build_single_feature_pipelines(
        is_numeric: bool,
        is_clf: bool,
    ) -> tuple[Pipeline, Pipeline]:
        def _prep_steps() -> list[tuple[str, Any]]:
            if is_numeric:
                return [("imputer", SimpleImputer(strategy="median"))]
            return [
                ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]

        if is_clf:
            dummy_model: Any = DummyClassifier(strategy="most_frequent")
            dt_model: Any = DecisionTreeClassifier(max_depth=3, random_state=42)
        else:
            dummy_model = DummyRegressor(strategy="mean")
            dt_model = DecisionTreeRegressor(max_depth=3, random_state=42)

        dt_pipe = Pipeline(_prep_steps() + [("model", dt_model)])
        dummy_pipe = Pipeline(_prep_steps() + [("model", dummy_model)])
        return dt_pipe, dummy_pipe
