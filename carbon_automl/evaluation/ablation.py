"""
ablation.py – AblationAnalyzer for systematic component & optimization studies.

Evaluates which pipeline optimizations genuinely help vs hurt predictive
quality and estimated energy:
- full_features_no_opt (baseline)
- feature_selection_only (MI selection on training data only)
- pca_disabled vs pca_enabled (PCA fitted on training data only)
- early_stopping_disabled (and early stopping comparison when supported)
- hpo_only (Optuna CV HPO on training data only, 10 trials)
- full_pipeline (feature selection + HPO)
- leave_one_feature_out (LOFO) impact analysis
- complexity_comparison (lightweight vs ensemble estimators)

Note: Energy values in ablation use ``training_time * 10.0`` W as an explicit
rough proxy (flagged ``is_estimated_energy=True``), distinct from hardware RAPL
measurements.
"""
from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from .feature_selection import FeatureSelector
from .metrics import MetricsCalculator
from .model_zoo import get_model_zoo
from .optimizer import HyperparameterOptimizer

ROUGH_POWER_WATTS = 10.0
QUALITY_EPS = 0.005


class AblationAnalyzer:
    """Run controlled ablation experiments on training and holdout splits."""

    def __init__(self) -> None:
        self._metrics = MetricsCalculator()
        self._selector = FeatureSelector()
        self._optimizer = HyperparameterOptimizer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_ablation(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        model_name: str,
        task_type: str,
        random_state: int = 42,
        feature_names: list[str] | None = None,
        include_leave_one_out: bool = True,
        include_complexity_comparison: bool = True,
        n_hpo_trials: int = 10,
    ) -> dict[str, Any]:
        """Run ablation conditions on a model and report what helps vs hurts.

        All selection, PCA, and HPO steps are fitted ONLY on ``X_train, y_train``.
        ``X_test, y_test`` are used strictly for final condition evaluation.

        Parameters
        ----------
        X_train, y_train : array-like or DataFrame/Series
        X_test, y_test : array-like or DataFrame/Series
        model_name : str
            Model key in ``get_model_zoo(task_type)``.
        task_type : str – "classification" or "regression"
        random_state : int
        feature_names : list[str] or None
            Optional names; inferred from DataFrame columns or ``f0..fn``.
        include_leave_one_out : bool
        include_complexity_comparison : bool
        n_hpo_trials : int
            Optuna trials for ``hpo_only`` and ``full_pipeline`` (default 10).

        Returns
        -------
        dict containing ``conditions`` (plus top-level condition keys),
        ``helpful_optimizations``, ``harmful_optimizations``,
        ``neutral_optimizations``, ``pca_comparison``,
        ``early_stopping_comparison``, ``leave_one_feature_out``,
        ``complexity_comparison``, and ``summary``.
        """
        task = task_type.lower()
        if feature_names is None:
            if hasattr(X_train, "columns"):
                feature_names = [str(c) for c in X_train.columns]
            else:
                n_cols = np.asarray(X_train).shape[1]
                feature_names = [f"f{i}" for i in range(n_cols)]

        X_tr = np.asarray(X_train, dtype=float)
        X_te = np.asarray(X_test, dtype=float)
        y_tr = np.asarray(y_train)
        y_te = np.asarray(y_test)

        n_train, n_features = X_tr.shape
        primary_metric = "macro_f1" if task == "classification" else "r2"

        enable_xgb = model_name.lower() == "xgboost"
        zoo = get_model_zoo(
            task,
            enable_xgboost=enable_xgb,
            dataset_size=n_train,
            n_features=n_features,
            random_state=random_state,
        )
        if model_name not in zoo:
            raise ValueError(
                f"Model '{model_name}' not found in zoo for task '{task}'."
            )
        base_estimator = zoo[model_name]

        inner_cv = self._make_inner_cv(y_tr, task, random_state)

        conditions: dict[str, dict[str, Any]] = {}

        # 1. full_features_no_opt (Baseline)
        conditions["full_features_no_opt"] = self._evaluate_fit(
            clone(base_estimator),
            X_tr,
            y_tr,
            X_te,
            y_te,
            task,
            primary_metric,
            n_features_used=n_features,
            notes="All features, default hyperparameters",
        )
        baseline_score = conditions["full_features_no_opt"]["primary_score"]
        baseline_energy = conditions["full_features_no_opt"]["estimated_energy_joules"]

        # 2. feature_selection_only (MI fitted ONLY on X_train)
        if n_features > 1:
            mi_res = self._selector.mutual_information_selection(
                X_tr,
                y_tr,
                feature_names=feature_names,
                task_type=task,
                random_state=random_state,
            )
            sel_names = mi_res["selected_features"]
            sel_indices = [feature_names.index(name) for name in sel_names]
            X_tr_fs = X_tr[:, sel_indices]
            X_te_fs = X_te[:, sel_indices]
        else:
            sel_names = list(feature_names)
            sel_indices = list(range(n_features))
            X_tr_fs = X_tr
            X_te_fs = X_te

        conditions["feature_selection_only"] = self._evaluate_fit(
            clone(base_estimator),
            X_tr_fs,
            y_tr,
            X_te_fs,
            y_te,
            task,
            primary_metric,
            n_features_used=len(sel_indices),
            extra={"selected_features": sel_names},
            notes="MI feature selection (train only), default hyperparameters",
        )

        # 3. pca_disabled vs pca_enabled (PCA fitted ONLY on X_train)
        conditions["pca_disabled"] = self._evaluate_fit(
            clone(base_estimator),
            X_tr,
            y_tr,
            X_te,
            y_te,
            task,
            primary_metric,
            n_features_used=n_features,
            extra={"pca_applied": False},
            notes="PCA disabled (original feature space), default hyperparameters",
        )

        if n_features >= 2 and n_train >= 2:
            n_components = min(max(2, int(n_features * 0.7)), n_features, n_train)
            pca = PCA(n_components=n_components, random_state=random_state)
            X_tr_pca = pca.fit_transform(X_tr)
            X_te_pca = pca.transform(X_te)
            explained_var = float(np.sum(pca.explained_variance_ratio_))
        else:
            n_components = n_features
            X_tr_pca = X_tr
            X_te_pca = X_te
            explained_var = 1.0

        conditions["pca_enabled"] = self._evaluate_fit(
            clone(base_estimator),
            X_tr_pca,
            y_tr,
            X_te_pca,
            y_te,
            task,
            primary_metric,
            n_features_used=n_components,
            extra={
                "pca_applied": True,
                "n_components": n_components,
                "explained_variance_ratio": round(explained_var, 4),
            },
            notes=f"PCA enabled (n_components={n_components}, train only)",
        )

        # 4. early_stopping_disabled (and early_stopping_enabled check when applicable)
        es_disabled_est = clone(base_estimator)
        es_disabled_params = es_disabled_est.get_params()
        if "early_stopping" in es_disabled_params:
            es_disabled_est.set_params(early_stopping=False)
        if model_name.lower() == "lightgbm" and "n_estimators" in es_disabled_params:
            es_disabled_est.set_params(n_estimators=max(100, int(es_disabled_params["n_estimators"])))

        conditions["early_stopping_disabled"] = self._evaluate_fit(
            es_disabled_est,
            X_tr,
            y_tr,
            X_te,
            y_te,
            task,
            primary_metric,
            n_features_used=n_features,
            extra={"early_stopping": False},
            notes="Full iteration/estimator budget without early stopping",
        )

        early_stopping_comparison = self._run_early_stopping_check(
            base_estimator,
            model_name,
            X_tr,
            y_tr,
            X_te,
            y_te,
            task,
            primary_metric,
            conditions["early_stopping_disabled"],
            random_state,
        )

        # 5. hpo_only (all features, Optuna HPO on X_train CV ONLY)
        hpo_res = self._optimizer.optimize(
            model_name=model_name,
            X_train=X_tr,
            y_train=y_tr,
            cv_splitter=inner_cv,
            task_type=task,
            n_trials=n_hpo_trials,
            random_state=random_state,
        )
        hpo_model = clone(base_estimator)
        if hpo_res["best_params"]:
            valid_p = hpo_model.get_params()
            hpo_model.set_params(
                **{k: v for k, v in hpo_res["best_params"].items() if k in valid_p}
            )
        conditions["hpo_only"] = self._evaluate_fit(
            hpo_model,
            X_tr,
            y_tr,
            X_te,
            y_te,
            task,
            primary_metric,
            n_features_used=n_features,
            extra={
                "best_params": hpo_res["best_params"],
                "best_cv_score": hpo_res["best_score"],
                "hpo_trials": hpo_res["n_trials"],
                "hpo_duration_seconds": hpo_res["duration_seconds"],
            },
            notes=f"All features, Optuna HPO ({n_hpo_trials} trials on train CV only)",
        )

        # 6. full_pipeline (MI feature selection + Optuna HPO on train only)
        full_hpo_res = self._optimizer.optimize(
            model_name=model_name,
            X_train=X_tr_fs,
            y_train=y_tr,
            cv_splitter=inner_cv,
            task_type=task,
            n_trials=n_hpo_trials,
            random_state=random_state,
        )
        full_model = clone(base_estimator)
        if full_hpo_res["best_params"]:
            valid_p = full_model.get_params()
            full_model.set_params(
                **{k: v for k, v in full_hpo_res["best_params"].items() if k in valid_p}
            )
        conditions["full_pipeline"] = self._evaluate_fit(
            full_model,
            X_tr_fs,
            y_tr,
            X_te_fs,
            y_te,
            task,
            primary_metric,
            n_features_used=len(sel_indices),
            extra={
                "selected_features": sel_names,
                "best_params": full_hpo_res["best_params"],
                "best_cv_score": full_hpo_res["best_score"],
                "hpo_trials": full_hpo_res["n_trials"],
                "hpo_duration_seconds": full_hpo_res["duration_seconds"],
            },
            notes=f"MI feature selection + Optuna HPO ({n_hpo_trials} trials on train CV only)",
        )

        # Annotate deltas and helps/hurts vs baseline
        helpful: list[str] = []
        harmful: list[str] = []
        neutral: list[str] = []
        assessment: dict[str, dict[str, Any]] = {}

        for cond_name, cond in conditions.items():
            q_delta = cond["primary_score"] - baseline_score
            e_delta = cond["estimated_energy_joules"] - baseline_energy
            cond["delta_vs_baseline"] = round(q_delta, 6)
            cond["energy_delta_vs_baseline"] = round(e_delta, 6)

            if cond_name == "full_features_no_opt":
                verdict = "baseline"
            else:
                verdict = self._classify_optimization_impact(
                    q_delta, e_delta, baseline_energy
                )
                if verdict == "helps":
                    helpful.append(cond_name)
                elif verdict == "hurts":
                    harmful.append(cond_name)
                else:
                    neutral.append(cond_name)

            cond["helps_or_hurts"] = verdict
            assessment[cond_name] = {
                "verdict": verdict,
                "quality_delta": round(q_delta, 6),
                "energy_delta_joules": round(e_delta, 6),
            }

        pca_comparison = {
            "pca_disabled_score": conditions["pca_disabled"]["primary_score"],
            "pca_enabled_score": conditions["pca_enabled"]["primary_score"],
            "score_delta": round(
                conditions["pca_enabled"]["primary_score"]
                - conditions["pca_disabled"]["primary_score"],
                6,
            ),
            "pca_disabled_energy_j": conditions["pca_disabled"]["estimated_energy_joules"],
            "pca_enabled_energy_j": conditions["pca_enabled"]["estimated_energy_joules"],
            "pca_helps": conditions["pca_enabled"]["helps_or_hurts"] == "helps",
        }

        # 7. Leave-One-Feature-Out (LOFO)
        lofo_results: dict[str, Any] = {}
        if include_leave_one_out and n_features > 1:
            lofo_results = self._run_leave_one_out(
                base_estimator,
                X_tr,
                y_tr,
                X_te,
                y_te,
                feature_names,
                task,
                primary_metric,
                baseline_score,
                random_state,
            )

        # 8. Complexity comparison
        complexity_comparison: dict[str, Any] = {}
        if include_complexity_comparison:
            complexity_comparison = self._run_complexity_comparison(
                zoo, X_tr, y_tr, X_te, y_te, task, primary_metric
            )

        summary = self._build_summary(
            primary_metric, baseline_score, helpful, harmful, neutral
        )

        return {
            "conditions": conditions,
            **conditions,
            "baseline_condition": "full_features_no_opt",
            "primary_metric": primary_metric,
            "model_name": model_name,
            "task_type": task,
            "helpful_optimizations": helpful,
            "harmful_optimizations": harmful,
            "neutral_optimizations": neutral,
            "optimization_assessment": assessment,
            "pca_comparison": pca_comparison,
            "early_stopping_comparison": early_stopping_comparison,
            "leave_one_feature_out": lofo_results,
            "complexity_comparison": complexity_comparison,
            "summary": summary,
            "energy_estimation_note": (
                f"Estimated energy uses training_time * {ROUGH_POWER_WATTS} W rough proxy; "
                "use EnergyMonitor/CarbonCalculator for measured energy."
            ),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_fit(
        self,
        estimator,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_te: np.ndarray,
        y_te: np.ndarray,
        task: str,
        primary_metric: str,
        n_features_used: int,
        extra: dict[str, Any] | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator.fit(X_tr, y_tr)
        train_time = time.perf_counter() - t0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred = estimator.predict(X_te)

            if task == "classification":
                y_proba = None
                if hasattr(estimator, "predict_proba"):
                    try:
                        y_proba = estimator.predict_proba(X_te)
                    except Exception:
                        y_proba = None
                metrics = self._metrics.compute_classification_metrics(
                    y_te, y_pred, y_proba=y_proba
                )
            else:
                metrics = self._metrics.compute_regression_metrics(y_te, y_pred)

        est_energy_j = train_time * ROUGH_POWER_WATTS
        payload: dict[str, Any] = {
            "metrics": metrics,
            "primary_metric": primary_metric,
            "primary_score": float(metrics.get(primary_metric, 0.0)),
            "training_time": round(train_time, 6),
            "estimated_energy_joules": round(est_energy_j, 6),
            "is_estimated_energy": True,
            "n_features_used": int(n_features_used),
            "notes": notes,
        }
        if extra:
            payload.update(extra)
        return payload

    def _run_early_stopping_check(
        self,
        base_estimator,
        model_name: str,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_te: np.ndarray,
        y_te: np.ndarray,
        task: str,
        primary_metric: str,
        disabled_result: dict[str, Any],
        random_state: int,
    ) -> dict[str, Any]:
        name = model_name.lower()
        if name == "lightgbm" and len(X_tr) >= 30:
            try:
                import lightgbm as lgb

                strat = y_tr if task == "classification" else None
                X_sub_tr, X_sub_val, y_sub_tr, y_sub_val = train_test_split(
                    X_tr,
                    y_tr,
                    test_size=0.15,
                    random_state=random_state,
                    stratify=strat,
                )
                es_est = clone(base_estimator)
                es_est.set_params(n_estimators=200)
                t0 = time.perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    es_est.fit(
                        X_sub_tr,
                        y_sub_tr,
                        eval_set=[(X_sub_val, y_sub_val)],
                        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)],
                    )
                train_time = time.perf_counter() - t0
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    y_pred = es_est.predict(X_te)
                    y_proba = (
                        es_est.predict_proba(X_te)
                        if task == "classification" and hasattr(es_est, "predict_proba")
                        else None
                    )
                m = (
                    self._metrics.compute_classification_metrics(
                        y_te, y_pred, y_proba=y_proba
                    )
                    if task == "classification"
                    else self._metrics.compute_regression_metrics(y_te, y_pred)
                )
                es_score = float(m.get(primary_metric, 0.0))
                es_energy = train_time * ROUGH_POWER_WATTS
                return {
                    "supported": True,
                    "early_stopping_disabled_score": disabled_result["primary_score"],
                    "early_stopping_enabled_score": es_score,
                    "early_stopping_disabled_energy_j": disabled_result[
                        "estimated_energy_joules"
                    ],
                    "early_stopping_enabled_energy_j": round(es_energy, 6),
                    "best_iteration": getattr(es_est, "best_iteration_", None),
                    "early_stopping_helps": (
                        es_score >= disabled_result["primary_score"] - QUALITY_EPS
                        and es_energy < disabled_result["estimated_energy_joules"]
                    ),
                }
            except Exception:
                pass

        return {
            "supported": False,
            "early_stopping_disabled_score": disabled_result["primary_score"],
            "early_stopping_disabled_energy_j": disabled_result[
                "estimated_energy_joules"
            ],
            "note": f"Model '{model_name}' does not support built-in early stopping.",
        }

    def _run_leave_one_out(
        self,
        base_estimator,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_te: np.ndarray,
        y_te: np.ndarray,
        feature_names: list[str],
        task: str,
        primary_metric: str,
        baseline_score: float,
        random_state: int,
    ) -> dict[str, Any]:
        n_features = len(feature_names)
        if n_features <= 15:
            candidate_indices = list(range(n_features))
        else:
            # Cap at top 15 features by MI on X_train to keep ablation fast
            mi_res = self._selector.mutual_information_selection(
                X_tr,
                y_tr,
                feature_names=feature_names,
                k=15,
                task_type=task,
                random_state=random_state,
            )
            candidate_indices = [
                feature_names.index(name) for name in mi_res["selected_features"]
            ]

        per_feature: dict[str, dict[str, Any]] = {}
        for idx in candidate_indices:
            fname = feature_names[idx]
            keep = [j for j in range(n_features) if j != idx]
            res = self._evaluate_fit(
                clone(base_estimator),
                X_tr[:, keep],
                y_tr,
                X_te[:, keep],
                y_te,
                task,
                primary_metric,
                n_features_used=len(keep),
                notes=f"Leave-one-feature-out without '{fname}'",
            )
            delta = res["primary_score"] - baseline_score
            per_feature[fname] = {
                "score_without_feature": res["primary_score"],
                "delta_when_removed": round(delta, 6),
                "feature_contribution": round(-delta, 6),
                "training_time": res["training_time"],
                "estimated_energy_joules": res["estimated_energy_joules"],
                "impact": (
                    "helpful_feature"
                    if delta < -QUALITY_EPS
                    else ("harmful_or_noisy_feature" if delta > QUALITY_EPS else "neutral_feature")
                ),
            }

        return {
            "features_evaluated": len(per_feature),
            "per_feature": per_feature,
        }

    def _run_complexity_comparison(
        self,
        zoo: dict[str, Any],
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_te: np.ndarray,
        y_te: np.ndarray,
        task: str,
        primary_metric: str,
    ) -> dict[str, Any]:
        tiers = {
            "dummy": "trivial_baseline",
            "linear_regression": "low_complexity_linear",
            "ridge": "low_complexity_linear",
            "logistic_regression": "low_complexity_linear",
            "decision_tree": "low_complexity_tree",
            "random_forest": "medium_complexity_ensemble",
            "extra_trees": "medium_complexity_ensemble",
            "lightgbm": "high_efficiency_boosting",
            "xgboost": "high_efficiency_boosting",
        }
        comparison: dict[str, dict[str, Any]] = {}
        for m_name, est in zoo.items():
            res = self._evaluate_fit(
                clone(est),
                X_tr,
                y_tr,
                X_te,
                y_te,
                task,
                primary_metric,
                n_features_used=X_tr.shape[1],
                notes=f"Complexity comparison model '{m_name}'",
            )
            comparison[m_name] = {
                "complexity_tier": tiers.get(m_name, "other"),
                "primary_score": res["primary_score"],
                "training_time": res["training_time"],
                "estimated_energy_joules": res["estimated_energy_joules"],
            }
        return comparison

    @staticmethod
    def _make_inner_cv(y_tr: np.ndarray, task: str, random_state: int):
        if task == "classification":
            _, counts = np.unique(y_tr, return_counts=True)
            min_count = int(np.min(counts)) if len(counts) > 0 else 0
            if min_count >= 2:
                n_splits = max(2, min(3, min_count))
                return StratifiedKFold(
                    n_splits=n_splits, shuffle=True, random_state=random_state
                )
        n_splits = max(2, min(3, len(y_tr)))
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    @staticmethod
    def _classify_optimization_impact(
        quality_delta: float, energy_delta: float, baseline_energy: float
    ) -> str:
        rel_energy_change = (
            energy_delta / baseline_energy if baseline_energy > 1e-12 else 0.0
        )
        if quality_delta > QUALITY_EPS:
            return "helps"
        if quality_delta < -QUALITY_EPS:
            return "hurts"
        # Quality roughly unchanged (within +-QUALITY_EPS): check energy
        if rel_energy_change < -0.10:
            return "helps"
        if rel_energy_change > 0.25:
            return "hurts"
        return "neutral"

    @staticmethod
    def _build_summary(
        primary_metric: str,
        baseline_score: float,
        helpful: list[str],
        harmful: list[str],
        neutral: list[str],
    ) -> str:
        parts = [
            f"Baseline ({primary_metric}={baseline_score:.4f}).",
            f"Helpful optimizations: {helpful or 'none'}.",
            f"Harmful optimizations: {harmful or 'none'}.",
            f"Neutral optimizations: {neutral or 'none'}.",
        ]
        return " ".join(parts)
