"""
metrics.py – MetricsCalculator for classification and regression tasks.

Design goals:
- Handle edge cases: single-class folds, zero-division.
- Bootstrap confidence intervals (1000 resamples).
- Quality-constraint checking between baseline and candidate models.
- ASCII confusion-matrix rendering.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np


class MetricsCalculator:
    """Compute, compare, and format evaluation metrics."""

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def compute_classification_metrics(
        self,
        y_true,
        y_pred,
        y_proba=None,
    ) -> dict[str, Any]:
        """Compute classification metrics.

        Parameters
        ----------
        y_true : array-like
        y_pred : array-like
        y_proba : array-like or None
            Class probability matrix (n_samples, n_classes). Used for
            ROC-AUC when more than two classes are present.

        Returns
        -------
        dict with keys:
            accuracy, macro_f1, weighted_f1, balanced_accuracy,
            macro_precision, macro_recall, per_class_recall (dict),
            confusion_matrix (list of lists),
            roc_auc_ovr (optional, float).
        """
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            balanced_accuracy_score,
            precision_score,
            recall_score,
            confusion_matrix,
            roc_auc_score,
        )

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        classes = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(classes)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            accuracy = float(accuracy_score(y_true, y_pred))
            macro_f1 = float(
                f1_score(y_true, y_pred, average="macro", zero_division=0)
            )
            weighted_f1 = float(
                f1_score(y_true, y_pred, average="weighted", zero_division=0)
            )
            balanced_acc = float(
                balanced_accuracy_score(y_true, y_pred)
            )
            macro_precision = float(
                precision_score(y_true, y_pred, average="macro", zero_division=0)
            )
            macro_recall = float(
                recall_score(y_true, y_pred, average="macro", zero_division=0)
            )

            # Per-class recall
            recall_per_class = recall_score(
                y_true, y_pred, average=None, zero_division=0, labels=classes
            )
            per_class_recall = {
                str(cls): float(r)
                for cls, r in zip(classes, recall_per_class)
            }

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=classes)
            cm_list = cm.tolist()

        result: dict[str, Any] = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "balanced_accuracy": balanced_acc,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "per_class_recall": per_class_recall,
            "confusion_matrix": cm_list,
        }

        # ROC-AUC and calibration metrics (only if proba provided and meaningful)
        if y_proba is not None:
            y_proba = np.asarray(y_proba, dtype=float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    if y_proba.ndim == 1 or (y_proba.ndim == 2 and y_proba.shape[1] == 2):
                        if len(np.unique(y_true)) == 2:
                            proba_col = (
                                y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                            )
                            result["roc_auc_ovr"] = float(
                                roc_auc_score(y_true, proba_col)
                            )
                        else:
                            result["roc_auc_ovr"] = None
                    elif y_proba.ndim == 2 and y_proba.shape[1] > 2:
                        if len(np.unique(y_true)) == y_proba.shape[1]:
                            result["roc_auc_ovr"] = float(
                                roc_auc_score(
                                    y_true,
                                    y_proba,
                                    multi_class="ovr",
                                    average="macro",
                                    labels=np.unique(y_true),
                                )
                            )
                        else:
                            result["roc_auc_ovr"] = None
                    else:
                        result["roc_auc_ovr"] = None
                except (ValueError, TypeError):
                    # Edge case: single class or label mismatch during CV
                    result["roc_auc_ovr"] = None

            cal_metrics = self._compute_calibration_metrics(y_true, y_pred, y_proba)
            result.update(cal_metrics)

        return result

    @staticmethod
    def _compute_calibration_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
    ) -> dict[str, float | None]:
        """Compute brier_score, log_loss, and Expected Calibration Error (ece)."""
        from sklearn.metrics import brier_score_loss, log_loss

        brier: float | None = None
        ll: float | None = None
        ece: float | None = None

        unique_true = np.unique(y_true)
        if len(y_true) == 0:
            return {"brier_score": None, "log_loss": None, "ece": None}

        # Normalize probabilities to 2D (n_samples, n_classes_proba)
        if y_proba.ndim == 1:
            p_clipped = np.clip(y_proba, 0.0, 1.0)
            y_proba_2d = np.column_stack([1.0 - p_clipped, p_clipped])
        else:
            y_proba_2d = np.clip(y_proba, 0.0, 1.0)

        n_proba_cols = y_proba_2d.shape[1]

        # Expected Calibration Error (ECE): max-confidence vs prediction correctness
        try:
            confidences = np.max(y_proba_2d, axis=1)
            accuracies = (y_pred == y_true).astype(float)
            bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
            ece_val = 0.0
            n_samples = len(y_true)
            for idx in range(n_bins):
                low, high = bin_edges[idx], bin_edges[idx + 1]
                if idx == 0:
                    mask = (confidences >= low) & (confidences <= high)
                else:
                    mask = (confidences > low) & (confidences <= high)
                if np.any(mask):
                    bin_acc = float(np.mean(accuracies[mask]))
                    bin_conf = float(np.mean(confidences[mask]))
                    ece_val += (np.sum(mask) / n_samples) * abs(bin_acc - bin_conf)
            ece = float(ece_val)
        except Exception:
            ece = None

        # Log loss and Brier score
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                if n_proba_cols == 2 and len(unique_true) <= 2:
                    if len(unique_true) == 2:
                        ll = float(log_loss(y_true, y_proba_2d, labels=unique_true))
                        # Map positive class to 1 for binary brier_score_loss
                        pos_label = unique_true[1]
                        y_bin = (y_true == pos_label).astype(int)
                        brier = float(brier_score_loss(y_bin, y_proba_2d[:, 1]))
                elif n_proba_cols > 2 and len(unique_true) == n_proba_cols:
                    labels_sorted = np.sort(unique_true)
                    ll = float(log_loss(y_true, y_proba_2d, labels=labels_sorted))
                    try:
                        brier = float(
                            brier_score_loss(y_true, y_proba_2d, labels=labels_sorted)
                        )
                    except (TypeError, ValueError):
                        # Fallback for sklearn < 1.7 where brier_score_loss is binary-only
                        label_to_idx = {lbl: i for i, lbl in enumerate(labels_sorted)}
                        y_onehot = np.zeros_like(y_proba_2d)
                        for row_i, lbl in enumerate(y_true):
                            y_onehot[row_i, label_to_idx[lbl]] = 1.0
                        brier = float(
                            np.mean(np.sum((y_proba_2d - y_onehot) ** 2, axis=1))
                        )
            except (ValueError, TypeError):
                ll = None
                brier = None

        return {"brier_score": brier, "log_loss": ll, "ece": ece}

    # ------------------------------------------------------------------
    # Regression
    # ------------------------------------------------------------------

    def compute_regression_metrics(self, y_true, y_pred) -> dict[str, Any]:
        """Compute regression metrics.

        Computes MAPE ONLY when ``np.all(np.abs(y_true) > 1e-8)``; otherwise
        sets ``mape: None`` with an explanatory ``mape_note``.

        Returns
        -------
        dict with keys: r2, rmse, mae, residuals_stats (mean, std, max_abs),
        mape (float or None), mape_note (str).
        """
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        r2 = float(r2_score(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))

        residuals = y_pred - y_true
        residuals_stats = {
            "mean": float(np.mean(residuals)),
            "std": float(np.std(residuals)),
            "max_abs": float(np.max(np.abs(residuals))),
        }

        if len(y_true) > 0 and np.all(np.abs(y_true) > 1e-8):
            mape: float | None = float(
                np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0
            )
            mape_note = (
                "MAPE computed in percentage (0-100 scale); all |y_true| > 1e-8."
            )
        else:
            mape = None
            mape_note = (
                "MAPE omitted: y_true contains values <= 1e-8 in magnitude "
                "(division by near-zero target is undefined/unstable)."
            )

        return {
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "mape_note": mape_note,
            "residuals_stats": residuals_stats,
        }

    # ------------------------------------------------------------------
    # Confidence intervals
    # ------------------------------------------------------------------

    def compute_confidence_intervals(
        self,
        scores: list[float],
        confidence: float = 0.95,
    ) -> dict[str, Any]:
        """Bootstrap confidence interval for a list of scores.

        Parameters
        ----------
        scores : list[float]
            Observed scores (e.g., per-fold CV scores).
        confidence : float
            Confidence level, default 0.95.

        Returns
        -------
        dict with keys: mean, std, ci_lower, ci_upper, n_samples.
        """
        rng = np.random.default_rng(seed=42)
        arr = np.asarray(scores, dtype=float)
        n = len(arr)

        boot_means = np.array(
            [
                rng.choice(arr, size=n, replace=True).mean()
                for _ in range(1000)
            ]
        )

        alpha = (1.0 - confidence) / 2.0
        ci_lower = float(np.percentile(boot_means, 100 * alpha))
        ci_upper = float(np.percentile(boot_means, 100 * (1.0 - alpha)))

        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_samples": n,
        }

    # ------------------------------------------------------------------
    # Quality constraint check
    # ------------------------------------------------------------------

    def check_quality_constraint(
        self,
        baseline_metrics: dict,
        candidate_metrics: dict,
        tolerance: float,
        task_type: str,
        primary_metric: str = "",
        critical_class_recall_constraints: dict[str, float] | None = None,
        regression_tolerances: dict[str, float] | None = None,
    ) -> tuple[bool, str]:
        """Check whether a candidate model is within tolerance of the baseline.

        Parameters
        ----------
        baseline_metrics : dict
        candidate_metrics : dict
        tolerance : float
            Acceptable degradation. E.g., 0.005 = 0.5 percentage points.
        task_type : str
            "classification" or "regression".
        primary_metric : str
            Override for the primary metric key. Defaults to "macro_f1"
            for classification and "r2" for regression.
        critical_class_recall_constraints : dict[str, float] or None
            Optional per-class minimum recall thresholds for classification
            (e.g., ``{"Severe": 0.85}``).
        regression_tolerances : dict[str, float] or None
            Optional tolerances for regression metrics. Supports ``"r2"``
            (max drop), ``"rmse"`` / ``"mae"`` (max absolute increase), and
            ``"rmse_rel"`` / ``"mae_rel"`` (max relative increase, e.g. 0.05).

        Returns
        -------
        (passes: bool, reason: str)
        """
        task = task_type.lower()
        if not primary_metric:
            primary_metric = "macro_f1" if task == "classification" else "r2"

        baseline_val = baseline_metrics.get(primary_metric)
        candidate_val = candidate_metrics.get(primary_metric)

        if baseline_val is None:
            return False, (
                f"Primary metric '{primary_metric}' missing from baseline_metrics."
            )
        if candidate_val is None:
            return False, (
                f"Primary metric '{primary_metric}' missing from candidate_metrics."
            )

        lower_is_better = primary_metric in ("rmse", "mae", "mape", "log_loss", "brier_score", "ece")
        reasons: list[str] = []
        violations: list[str] = []

        if lower_is_better:
            threshold = float(baseline_val) + float(tolerance)
            if float(candidate_val) <= threshold:
                delta = float(candidate_val) - float(baseline_val)
                reasons.append(
                    f"candidate {primary_metric}={candidate_val:.4f} "
                    f"({delta:+.4f} vs baseline {baseline_val:.4f}, tolerance={tolerance})"
                )
            else:
                increase = float(candidate_val) - float(baseline_val)
                violations.append(
                    f"candidate {primary_metric}={candidate_val:.4f} increases by "
                    f"{increase:.4f} above baseline {baseline_val:.4f} (tolerance={tolerance})"
                )
        else:
            threshold = float(baseline_val) - float(tolerance)
            if float(candidate_val) >= threshold:
                delta = float(candidate_val) - float(baseline_val)
                reasons.append(
                    f"candidate {primary_metric}={candidate_val:.4f} "
                    f"({delta:+.4f} vs baseline {baseline_val:.4f}, tolerance={tolerance})"
                )
            else:
                drop = float(baseline_val) - float(candidate_val)
                violations.append(
                    f"candidate {primary_metric}={candidate_val:.4f} drops "
                    f"{drop:.4f} below baseline {baseline_val:.4f} (tolerance={tolerance})"
                )

        # Optional per-class critical recall constraints
        if task == "classification" and critical_class_recall_constraints:
            recall_map = candidate_metrics.get("per_class_recall", {}) or {}
            for cls_key, min_recall in critical_class_recall_constraints.items():
                actual_recall = recall_map.get(cls_key)
                if actual_recall is None:
                    actual_recall = recall_map.get(str(cls_key))
                if actual_recall is None:
                    violations.append(
                        f"critical class '{cls_key}' missing from per_class_recall"
                    )
                elif float(actual_recall) < float(min_recall):
                    violations.append(
                        f"critical class '{cls_key}' recall={float(actual_recall):.4f} "
                        f"< minimum {float(min_recall):.4f}"
                    )
                else:
                    reasons.append(
                        f"critical class '{cls_key}' recall={float(actual_recall):.4f} >= {float(min_recall):.4f}"
                    )

        # Optional regression tolerance checks
        if task == "regression" and regression_tolerances:
            if "r2" in regression_tolerances and primary_metric != "r2":
                b_r2 = baseline_metrics.get("r2")
                c_r2 = candidate_metrics.get("r2")
                tol_r2 = float(regression_tolerances["r2"])
                if b_r2 is not None and c_r2 is not None and float(c_r2) < float(b_r2) - tol_r2:
                    violations.append(
                        f"r2={float(c_r2):.4f} drops below baseline {float(b_r2):.4f} - {tol_r2}"
                    )

            for err_key in ("rmse", "mae"):
                b_err = baseline_metrics.get(err_key)
                c_err = candidate_metrics.get(err_key)
                if b_err is None or c_err is None:
                    continue
                if err_key in regression_tolerances:
                    tol_abs = float(regression_tolerances[err_key])
                    if float(c_err) > float(b_err) + tol_abs:
                        violations.append(
                            f"{err_key}={float(c_err):.4f} exceeds baseline {float(b_err):.4f} + {tol_abs}"
                        )
                rel_key = f"{err_key}_rel"
                if rel_key in regression_tolerances:
                    tol_rel = float(regression_tolerances[rel_key])
                    if float(b_err) > 1e-12:
                        max_allowed = float(b_err) * (1.0 + tol_rel)
                        if float(c_err) > max_allowed:
                            violations.append(
                                f"{err_key}={float(c_err):.4f} exceeds {1.0 + tol_rel:.2f}x baseline "
                                f"({max_allowed:.4f})"
                            )

        if violations:
            return False, "FAIL – " + "; ".join(violations) + "."
        return True, "PASS – " + "; ".join(reasons) + "."

    # ------------------------------------------------------------------
    # ASCII confusion matrix
    # ------------------------------------------------------------------

    def format_confusion_matrix(self, cm: list, class_names: list) -> str:
        """Render confusion matrix as ASCII art.

        Parameters
        ----------
        cm : list of lists
            Confusion matrix values (rows = true, cols = predicted).
        class_names : list of str

        Returns
        -------
        str – multi-line ASCII table.
        """
        cm_arr = np.asarray(cm)
        n = len(class_names)
        if n == 0 or cm_arr.size == 0:
            return "Confusion Matrix: <empty>"

        # Column width: max of class name length and cell value width
        cell_w = max(
            max(len(str(c)) for c in class_names),
            max(len(str(v)) for v in cm_arr.flatten()),
            4,
        )
        name_w = max(len(str(c)) for c in class_names) + 2

        header_prefix = " " * (name_w + 2)
        header_cols = "  ".join(str(c).center(cell_w) for c in class_names)
        lines = [
            f"Confusion Matrix (rows=True, cols=Predicted)",
            f"{header_prefix}{header_cols}",
            f"{header_prefix}" + "-" * (cell_w * n + 2 * (n - 1)),
        ]
        for i, row in enumerate(cm_arr):
            row_label = str(class_names[i]).ljust(name_w)
            row_vals = "  ".join(str(v).center(cell_w) for v in row)
            lines.append(f"{row_label}| {row_vals}")

        return "\n".join(lines)
