"""
validation.py – ValidationEngine for cross-validation, nested CV, and
holdout splits.

Key design constraint: NO test-set leakage. Optuna and cross-validation
NEVER see the held-out test set. Models are reset between folds via
sklearn.base.clone().
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import (
    BaseCrossValidator,
    StratifiedKFold,
    KFold,
    GroupKFold,
    StratifiedGroupKFold,
    TimeSeriesSplit,
    ParameterGrid,
    train_test_split,
)

from .metrics import MetricsCalculator


class ValidationEngine:
    """Provides cross-validation utilities without test-set leakage."""

    # ------------------------------------------------------------------
    # CV splitter factory
    # ------------------------------------------------------------------

    def create_cv_splitter(
        self,
        strategy: str,
        n_folds: int,
        random_state: int,
        groups=None,
    ) -> BaseCrossValidator:
        """Create a cross-validation splitter.

        Parameters
        ----------
        strategy : str
            One of: "stratified_kfold", "kfold", "group_kfold",
            "stratified_group_kfold", "time_series".
        n_folds : int
        random_state : int
        groups : array-like or None
            Required for "group_kfold" and "stratified_group_kfold".

        Returns
        -------
        BaseCrossValidator instance.
        """
        strategy = strategy.lower()

        if strategy == "stratified_kfold":
            return StratifiedKFold(
                n_splits=n_folds, shuffle=True, random_state=random_state
            )
        elif strategy == "kfold":
            return KFold(
                n_splits=n_folds, shuffle=True, random_state=random_state
            )
        elif strategy == "group_kfold":
            if groups is None:
                raise ValueError(
                    "groups must be provided for 'group_kfold' strategy."
                )
            return GroupKFold(n_splits=n_folds)
        elif strategy == "stratified_group_kfold":
            if groups is None:
                raise ValueError(
                    "groups must be provided for 'stratified_group_kfold' strategy."
                )
            return StratifiedGroupKFold(
                n_splits=n_folds, shuffle=True, random_state=random_state
            )
        elif strategy == "time_series":
            return TimeSeriesSplit(n_splits=n_folds)
        else:
            raise ValueError(
                f"Unknown CV strategy '{strategy}'. Expected one of: "
                "stratified_kfold, kfold, group_kfold, stratified_group_kfold, time_series."
            )

    # ------------------------------------------------------------------
    # Cross-validate a model
    # ------------------------------------------------------------------

    def cross_validate_model(
        self,
        model,
        X,
        y,
        cv_splitter: BaseCrossValidator,
        metrics_calculator: MetricsCalculator,
        task_type: str,
        groups=None,
    ) -> dict[str, Any]:
        """Cross-validate a model; never touches a held-out test set.

        The model is cloned before each fold so no state leaks between folds.

        Parameters
        ----------
        model : sklearn estimator
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
        cv_splitter : BaseCrossValidator
        metrics_calculator : MetricsCalculator
        task_type : str – "classification" or "regression"
        groups : array-like or None

        Returns
        -------
        dict with keys:
            fold_scores (list of dicts), mean (dict), std (dict),
            confidence_intervals (dict of metric -> CI dict), n_folds.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        fold_metrics: list[dict] = []

        split_kwargs: dict[str, Any] = {}
        if groups is not None:
            split_kwargs["groups"] = np.asarray(groups)

        for train_idx, val_idx in cv_splitter.split(X, y, **split_kwargs):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            fold_model = clone(model)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fold_model.fit(X_tr, y_tr)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y_pred = fold_model.predict(X_val)

                if task_type == "classification":
                    y_proba = None
                    if hasattr(fold_model, "predict_proba"):
                        try:
                            y_proba = fold_model.predict_proba(X_val)
                        except Exception:
                            y_proba = None
                    fold_m = metrics_calculator.compute_classification_metrics(
                        y_val, y_pred, y_proba=y_proba
                    )
                else:
                    fold_m = metrics_calculator.compute_regression_metrics(
                        y_val, y_pred
                    )

            fold_metrics.append(fold_m)

        if not fold_metrics:
            return {
                "fold_scores": [],
                "mean": {},
                "std": {},
                "confidence_intervals": {},
                "n_folds": 0,
            }

        # Aggregate: collect scalar numeric metrics across all folds (skip bools/dicts/lists)
        scalar_keys: list[str] = []
        seen_keys: set[str] = set()
        for fm in fold_metrics:
            for k, v in fm.items():
                if (
                    k not in seen_keys
                    and v is not None
                    and not isinstance(v, bool)
                    and isinstance(v, (int, float))
                ):
                    scalar_keys.append(k)
                    seen_keys.add(k)

        mean_scores: dict[str, float] = {}
        std_scores: dict[str, float] = {}
        ci_dict: dict[str, dict] = {}

        for key in scalar_keys:
            vals = [
                fm[key]
                for fm in fold_metrics
                if fm.get(key) is not None
            ]
            if not vals:
                continue
            mean_scores[key] = float(np.mean(vals))
            std_scores[key] = float(np.std(vals))
            ci_dict[key] = metrics_calculator.compute_confidence_intervals(vals)

        return {
            "fold_scores": fold_metrics,
            "mean": mean_scores,
            "std": std_scores,
            "confidence_intervals": ci_dict,
            "n_folds": len(fold_metrics),
        }

    # ------------------------------------------------------------------
    # Nested CV
    # ------------------------------------------------------------------

    def nested_cv(
        self,
        model,
        X,
        y,
        outer_cv: BaseCrossValidator,
        inner_cv: BaseCrossValidator,
        param_grid: dict | None = None,
        task_type: str = "classification",
        groups=None,
    ) -> dict[str, Any]:
        """Nested cross-validation for unbiased performance estimation.

        Outer loop: unbiased performance estimate.
        Inner loop: hyperparameter selection (grid search over param_grid).

        Parameters
        ----------
        model : sklearn estimator
        X, y : data arrays
        outer_cv, inner_cv : BaseCrossValidator
        param_grid : dict or None
            If provided, inner loop performs grid search. Otherwise,
            default params are used.
        task_type : str
        groups : array-like or None
            Optional group labels for group-aware splitters.

        Returns
        -------
        dict with outer_scores (list), mean_outer_score, std_outer_score,
        selected_params_per_fold (list of dicts).
        """
        from sklearn.metrics import f1_score, r2_score

        X = np.asarray(X)
        y = np.asarray(y)
        groups_arr = np.asarray(groups) if groups is not None else None

        metrics_calc = MetricsCalculator()
        outer_scores: list[float] = []
        selected_params: list[dict] = []

        primary_metric = "macro_f1" if task_type == "classification" else "r2"

        outer_split_kwargs: dict[str, Any] = {}
        if groups_arr is not None:
            outer_split_kwargs["groups"] = groups_arr

        for outer_train_idx, outer_val_idx in outer_cv.split(
            X, y, **outer_split_kwargs
        ):
            X_outer_tr, X_outer_val = X[outer_train_idx], X[outer_val_idx]
            y_outer_tr, y_outer_val = y[outer_train_idx], y[outer_val_idx]

            inner_split_kwargs: dict[str, Any] = {}
            if groups_arr is not None:
                inner_split_kwargs["groups"] = groups_arr[outer_train_idx]

            # Inner loop: pick best params
            best_inner_score = -np.inf
            best_params: dict = {}

            candidates = list(ParameterGrid(param_grid)) if param_grid else [{}]

            for params in candidates:
                fold_inner_scores: list[float] = []

                for inner_tr_idx, inner_val_idx in inner_cv.split(
                    X_outer_tr, y_outer_tr, **inner_split_kwargs
                ):
                    X_in_tr = X_outer_tr[inner_tr_idx]
                    X_in_val = X_outer_tr[inner_val_idx]
                    y_in_tr = y_outer_tr[inner_tr_idx]
                    y_in_val = y_outer_tr[inner_val_idx]

                    inner_model = clone(model)
                    if params:
                        inner_model.set_params(**params)

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        inner_model.fit(X_in_tr, y_in_tr)
                        y_in_pred = inner_model.predict(X_in_val)

                    if task_type == "classification":
                        score = float(
                            f1_score(
                                y_in_val,
                                y_in_pred,
                                average="macro",
                                zero_division=0,
                            )
                        )
                    else:
                        score = float(r2_score(y_in_val, y_in_pred))

                    fold_inner_scores.append(score)

                mean_inner = float(np.mean(fold_inner_scores))
                if mean_inner > best_inner_score:
                    best_inner_score = mean_inner
                    best_params = params

            # Refit on full outer training set with best params
            outer_model = clone(model)
            if best_params:
                outer_model.set_params(**best_params)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                outer_model.fit(X_outer_tr, y_outer_tr)
                y_outer_pred = outer_model.predict(X_outer_val)

            if task_type == "classification":
                outer_m = metrics_calc.compute_classification_metrics(
                    y_outer_val, y_outer_pred
                )
            else:
                outer_m = metrics_calc.compute_regression_metrics(
                    y_outer_val, y_outer_pred
                )

            outer_scores.append(float(outer_m.get(primary_metric, 0.0)))
            selected_params.append(best_params)

        return {
            "outer_scores": outer_scores,
            "mean_outer_score": float(np.mean(outer_scores)),
            "std_outer_score": float(np.std(outer_scores)),
            "selected_params_per_fold": selected_params,
            "primary_metric": primary_metric,
            "n_outer_folds": len(outer_scores),
        }

    # ------------------------------------------------------------------
    # Holdout split
    # ------------------------------------------------------------------

    def create_holdout_split(
        self,
        X,
        y,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
    ) -> tuple:
        """Create a train/test holdout split.

        Parameters
        ----------
        X, y : array-like, DataFrame, or Series
        test_size : float
            Fraction of data for the test set.
        random_state : int
        stratify : bool
            If True, stratify by y (suitable for classification).

        Returns
        -------
        (X_train, X_test, y_train, y_test)
        """
        stratify_arr = y if stratify else None

        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_arr,
        )
