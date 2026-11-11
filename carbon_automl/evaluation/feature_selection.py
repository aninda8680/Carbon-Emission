"""
feature_selection.py – FeatureSelector using mutual information and
permutation importance.

IMPORTANT: All selection methods must be called only on training data.
No test-set leakage is introduced here.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class FeatureSelector:
    """Feature selection utilities using mutual information and permutation
    importance."""

    # ------------------------------------------------------------------
    # Mutual Information Selection
    # ------------------------------------------------------------------

    def mutual_information_selection(
        self,
        X: np.ndarray,
        y,
        feature_names: list[str],
        k: int | None = None,
        task_type: str = "classification",
        random_state: int = 42,
    ) -> dict[str, Any]:
        """Select top-k features by mutual information with the target.

        MUST be called only on training data to prevent leakage.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : array-like
        feature_names : list[str]
        k : int or None
            Number of features to select. If None, uses
            max(2, int(n_features * 0.7)).
        task_type : str – "classification" or "regression"
        random_state : int

        Returns
        -------
        dict with keys:
            selected_features (list[str]), scores (dict name -> float),
            ranking (list[str] sorted by score desc), k_selected (int).
        """
        from sklearn.feature_selection import (
            mutual_info_classif,
            mutual_info_regression,
        )

        X = np.asarray(X)
        y = np.asarray(y)
        n_features = X.shape[1]

        if len(feature_names) != n_features:
            raise ValueError(
                f"feature_names length ({len(feature_names)}) must match "
                f"X.shape[1] ({n_features})."
            )

        if k is None:
            k = max(2, int(n_features * 0.7))
        k = min(k, n_features)  # clamp

        if task_type.lower() == "classification":
            mi_scores = mutual_info_classif(X, y, random_state=random_state)
        else:
            mi_scores = mutual_info_regression(X, y, random_state=random_state)

        # Sort descending by MI score
        sorted_idx = np.argsort(mi_scores)[::-1]
        ranking = [feature_names[i] for i in sorted_idx]
        scores = {feature_names[i]: float(mi_scores[i]) for i in range(n_features)}

        selected_features = [feature_names[i] for i in sorted_idx[:k]]

        return {
            "selected_features": selected_features,
            "scores": scores,
            "ranking": ranking,
            "k_selected": k,
        }

    # ------------------------------------------------------------------
    # Permutation Importance Selection
    # ------------------------------------------------------------------

    def permutation_importance_selection(
        self,
        model,
        X: np.ndarray,
        y,
        feature_names: list[str],
        n_repeats: int = 5,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """Select features by permutation importance.

        Threshold: mean importance > 0 (remove features with negative or
        zero mean importance, which hurt performance on average).

        Parameters
        ----------
        model : fitted sklearn estimator
        X : np.ndarray – validation/training features
        y : array-like – true labels
        feature_names : list[str]
        n_repeats : int
        random_state : int

        Returns
        -------
        dict with keys:
            importances_mean (dict name -> float),
            importances_std (dict name -> float),
            selected_features (list[str]),
            threshold (float).
        """
        from sklearn.inspection import permutation_importance

        X = np.asarray(X)
        y = np.asarray(y)

        result = permutation_importance(
            model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=1
        )

        threshold = 0.0  # mean importance > 0

        importances_mean = {
            feature_names[i]: float(result.importances_mean[i])
            for i in range(len(feature_names))
        }
        importances_std = {
            feature_names[i]: float(result.importances_std[i])
            for i in range(len(feature_names))
        }

        selected_features = [
            name
            for name, imp in importances_mean.items()
            if imp > threshold
        ]

        # Maintain ranking by mean importance (descending)
        selected_features.sort(key=lambda n: importances_mean[n], reverse=True)

        return {
            "importances_mean": importances_mean,
            "importances_std": importances_std,
            "selected_features": selected_features,
            "threshold": threshold,
        }

    # ------------------------------------------------------------------
    # Compare feature sets
    # ------------------------------------------------------------------

    def compare_feature_sets(
        self,
        full_scores: dict,
        reduced_scores: dict,
        full_energy: float,
        reduced_energy: float,
    ) -> dict[str, Any]:
        """Compare full vs. reduced feature set results.

        Does NOT assume fewer features = lower energy without measuring;
        both energy values are taken as given measurements.

        Parameters
        ----------
        full_scores : dict
            Metric dict for the full-feature model.
        reduced_scores : dict
            Metric dict for the reduced-feature model.
        full_energy : float
            Measured energy (Joules or kWh) for the full model.
        reduced_energy : float
            Measured energy for the reduced model.

        Returns
        -------
        dict with keys:
            performance_delta (dict metric -> delta),
            energy_delta (float, positive = reduced uses less energy),
            energy_reduction_pct (float),
            recommendation (str).
        """
        # Compute per-metric deltas (reduced - full)
        common_keys = set(full_scores) & set(reduced_scores)
        scalar_keys = [
            k
            for k in common_keys
            if isinstance(full_scores.get(k), (int, float))
            and isinstance(reduced_scores.get(k), (int, float))
        ]

        performance_delta = {
            k: float(reduced_scores[k]) - float(full_scores[k])
            for k in scalar_keys
        }

        energy_delta = full_energy - reduced_energy  # positive = savings
        energy_reduction_pct = (
            (energy_delta / full_energy * 100.0) if full_energy > 0 else 0.0
        )

        # Build recommendation
        # Choose a primary metric heuristically (macro_f1 or r2 if present)
        primary = None
        for cand in ("macro_f1", "r2", "accuracy", "balanced_accuracy"):
            if cand in performance_delta:
                primary = cand
                break

        if primary:
            perf_d = performance_delta[primary]
        else:
            perf_d = 0.0

        PERF_TOL = 0.005  # 0.5 pp tolerance

        if energy_reduction_pct > 0 and perf_d >= -PERF_TOL:
            recommendation = (
                f"USE REDUCED features: saves {energy_reduction_pct:.1f}% energy "
                f"with negligible performance change "
                f"({primary}Δ={perf_d:+.4f})."
            )
        elif energy_reduction_pct > 0 and perf_d < -PERF_TOL:
            recommendation = (
                f"PREFER FULL features: reduced set saves {energy_reduction_pct:.1f}% "
                f"energy but drops {primary} by {abs(perf_d):.4f} "
                f"(> {PERF_TOL} tolerance)."
            )
        elif energy_reduction_pct <= 0 and perf_d >= -PERF_TOL:
            recommendation = (
                f"USE FULL features: reduced set uses MORE energy "
                f"({abs(energy_reduction_pct):.1f}% increase) "
                f"with no meaningful performance gain."
            )
        else:
            recommendation = (
                "USE FULL features: reduced set is worse on both energy "
                "and performance."
            )

        return {
            "performance_delta": performance_delta,
            "energy_delta": energy_delta,
            "energy_reduction_pct": round(energy_reduction_pct, 4),
            "recommendation": recommendation,
        }

    # ------------------------------------------------------------------
    # Model-Based Selection (SelectFromModel)
    # ------------------------------------------------------------------

    def model_based_selection(
        self,
        X: np.ndarray,
        y,
        feature_names: list[str],
        task_type: str = "classification",
        estimator=None,
        threshold: str | float = "median",
        max_features: int | None = None,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """Select features using sklearn ``SelectFromModel``.

        MUST be called only on training data to prevent leakage.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : array-like
        feature_names : list[str]
        task_type : str – "classification" or "regression"
        estimator : sklearn estimator or None
            Defaults to RandomForestClassifier/Regressor(n_estimators=50).
        threshold : str or float
            Threshold passed to ``SelectFromModel``.
        max_features : int or None
        random_state : int

        Returns
        -------
        dict with keys:
            selected_features (list[str]), scores (dict name -> float),
            ranking (list[str] sorted desc), k_selected (int),
            threshold (float), method (str).
        """
        from sklearn.base import clone
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.feature_selection import SelectFromModel

        X = np.asarray(X)
        y = np.asarray(y)
        n_features = X.shape[1]

        if len(feature_names) != n_features:
            raise ValueError(
                f"feature_names length ({len(feature_names)}) must match "
                f"X.shape[1] ({n_features})."
            )

        if estimator is None:
            if task_type.lower() == "classification":
                base_est = RandomForestClassifier(
                    n_estimators=50, random_state=random_state, n_jobs=1
                )
            else:
                base_est = RandomForestRegressor(
                    n_estimators=50, random_state=random_state, n_jobs=1
                )
        else:
            base_est = clone(estimator)

        selector = SelectFromModel(
            estimator=base_est,
            threshold=threshold,
            max_features=max_features,
        )
        selector.fit(X, y)

        fitted_est = selector.estimator_
        if hasattr(fitted_est, "feature_importances_"):
            raw_scores = np.asarray(fitted_est.feature_importances_, dtype=float)
        elif hasattr(fitted_est, "coef_"):
            coef = np.asarray(fitted_est.coef_, dtype=float)
            raw_scores = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
        else:
            raw_scores = np.zeros(n_features, dtype=float)

        support = selector.get_support()
        scores = {feature_names[i]: float(raw_scores[i]) for i in range(n_features)}
        sorted_idx = np.argsort(raw_scores)[::-1]
        ranking = [feature_names[i] for i in sorted_idx]
        selected_features = [feature_names[i] for i in sorted_idx if support[i]]

        # Fallback if threshold eliminates every feature
        if not selected_features and n_features > 0:
            selected_features = [ranking[0]]

        return {
            "selected_features": selected_features,
            "scores": scores,
            "ranking": ranking,
            "k_selected": len(selected_features),
            "threshold": float(selector.threshold_),
            "method": "SelectFromModel",
        }

    # ------------------------------------------------------------------
    # Recursive Feature Elimination (RFE)
    # ------------------------------------------------------------------

    def rfe_selection(
        self,
        X: np.ndarray,
        y,
        feature_names: list[str],
        n_features_to_select: int | None = None,
        step: int | float = 1,
        task_type: str = "classification",
        estimator=None,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """Select features using Recursive Feature Elimination (``RFE``).

        MUST be called only on training data to prevent leakage.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : array-like
        feature_names : list[str]
        n_features_to_select : int or None
            Defaults to ``max(2, int(n_features * 0.7))`` clamped to n_features.
        step : int or float
        task_type : str – "classification" or "regression"
        estimator : sklearn estimator or None
            Defaults to RandomForestClassifier/Regressor(n_estimators=50).
        random_state : int

        Returns
        -------
        dict with keys:
            selected_features (list[str]), ranking (dict name -> int rank),
            ordered_features (list[str] sorted asc by rank),
            k_selected (int), method (str).
        """
        from sklearn.base import clone
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.feature_selection import RFE

        X = np.asarray(X)
        y = np.asarray(y)
        n_features = X.shape[1]

        if len(feature_names) != n_features:
            raise ValueError(
                f"feature_names length ({len(feature_names)}) must match "
                f"X.shape[1] ({n_features})."
            )

        if n_features_to_select is None:
            n_features_to_select = max(2, int(n_features * 0.7))
        n_features_to_select = max(1, min(n_features_to_select, n_features))

        if estimator is None:
            if task_type.lower() == "classification":
                base_est = RandomForestClassifier(
                    n_estimators=50, random_state=random_state, n_jobs=1
                )
            else:
                base_est = RandomForestRegressor(
                    n_estimators=50, random_state=random_state, n_jobs=1
                )
        else:
            base_est = clone(estimator)

        rfe = RFE(
            estimator=base_est,
            n_features_to_select=n_features_to_select,
            step=step,
        )
        rfe.fit(X, y)

        support = rfe.support_
        ranks = rfe.ranking_
        ranking_dict = {
            feature_names[i]: int(ranks[i]) for i in range(n_features)
        }
        ordered_idx = np.argsort(ranks)
        ordered_features = [feature_names[i] for i in ordered_idx]
        selected_features = [
            feature_names[i] for i in ordered_idx if support[i]
        ]

        return {
            "selected_features": selected_features,
            "ranking": ranking_dict,
            "ordered_features": ordered_features,
            "k_selected": len(selected_features),
            "method": "RFE",
        }
