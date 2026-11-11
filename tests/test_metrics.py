"""Tests for MetricsCalculator."""

from __future__ import annotations

import numpy as np
import pytest

from carbon_automl.evaluation.metrics import MetricsCalculator


def test_accuracy_correct() -> None:
    """Accuracy matches manually calculated proportion of exact predictions."""
    y_true = [0, 1, 2, 0, 1]
    y_pred = [0, 1, 1, 0, 2]
    # 3 correct out of 5 -> 0.6
    calc = MetricsCalculator()
    metrics = calc.compute_classification_metrics(y_true, y_pred)

    assert metrics["accuracy"] == pytest.approx(0.6)


def test_macro_f1_correct() -> None:
    """Macro F1 matches manually computed unweighted average of per-class F1."""
    # Class 0: TP=2, FP=0, FN=0 -> F1 = 1.0
    # Class 1: TP=1, FP=1, FN=1 -> P=0.5, R=0.5, F1 = 0.5
    # Class 2: TP=0, FP=1, FN=1 -> P=0.0, R=0.0, F1 = 0.0
    # Macro F1 = (1.0 + 0.5 + 0.0) / 3 = 0.5
    y_true = [0, 0, 1, 1, 2]
    y_pred = [0, 0, 1, 2, 1]
    calc = MetricsCalculator()
    metrics = calc.compute_classification_metrics(y_true, y_pred)

    assert metrics["macro_f1"] == pytest.approx(0.5)


def test_balanced_accuracy_correct() -> None:
    """Balanced accuracy properly reflects poor minority recall under imbalance."""
    y_true = [0] * 9 + [1] * 1
    y_pred = [0] * 10  # Always predicts majority class
    calc = MetricsCalculator()
    metrics = calc.compute_classification_metrics(y_true, y_pred)

    assert metrics["accuracy"] == pytest.approx(0.9)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)


def test_per_class_recall() -> None:
    """per_class_recall returns a dict keyed by string class labels."""
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]
    # Class 0: 1/2 = 0.5, Class 1: 2/2 = 1.0, Class 2: 1/2 = 0.5
    calc = MetricsCalculator()
    metrics = calc.compute_classification_metrics(y_true, y_pred)

    pcr = metrics["per_class_recall"]
    assert isinstance(pcr, dict)
    assert pcr["0"] == pytest.approx(0.5)
    assert pcr["1"] == pytest.approx(1.0)
    assert pcr["2"] == pytest.approx(0.5)


def test_regression_r2() -> None:
    """R² matches manual computation 1 - SS_res / SS_tot."""
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [1.0, 2.0, 3.0, 5.0]
    # mean=2.5; SS_tot = 2.25+0.25+0.25+2.25 = 5.0; SS_res = 1.0 -> R2 = 0.8
    calc = MetricsCalculator()
    metrics = calc.compute_regression_metrics(y_true, y_pred)

    assert metrics["r2"] == pytest.approx(0.8)
    assert metrics["mae"] == pytest.approx(0.25)


def test_regression_rmse() -> None:
    """RMSE matches sqrt(mean((y_pred - y_true)^2))."""
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [1.0, 2.0, 3.0, 5.0]
    # MSE = 1/4 = 0.25 -> RMSE = 0.5
    calc = MetricsCalculator()
    metrics = calc.compute_regression_metrics(y_true, y_pred)

    assert metrics["rmse"] == pytest.approx(0.5)


def test_bootstrap_ci() -> None:
    """Bootstrap 95% CI brackets the true population mean for large sample n."""
    rng = np.random.default_rng(42)
    samples = rng.normal(loc=5.0, scale=1.0, size=200).tolist()
    calc = MetricsCalculator()
    ci = calc.compute_confidence_intervals(samples, confidence=0.95)

    assert ci["n_samples"] == 200
    assert ci["ci_lower"] <= 5.0 <= ci["ci_upper"]
    assert ci["ci_lower"] < ci["mean"] < ci["ci_upper"]


def test_quality_constraint_pass() -> None:
    """Candidate within quality tolerance passes constraint check."""
    calc = MetricsCalculator()
    baseline = {"macro_f1": 0.850}
    candidate = {"macro_f1": 0.846}  # Drop of 0.004 <= tolerance 0.005

    passes, reason = calc.check_quality_constraint(
        baseline_metrics=baseline,
        candidate_metrics=candidate,
        tolerance=0.005,
        task_type="classification",
    )

    assert passes is True
    assert "PASS" in reason


def test_quality_constraint_fail() -> None:
    """Candidate dropping below baseline - tolerance fails constraint check."""
    calc = MetricsCalculator()
    baseline = {"macro_f1": 0.850}
    candidate = {"macro_f1": 0.830}  # Drop of 0.020 > tolerance 0.005

    passes, _ = calc.check_quality_constraint(
        baseline_metrics=baseline,
        candidate_metrics=candidate,
        tolerance=0.005,
        task_type="classification",
    )

    assert passes is False


def test_quality_constraint_violation_message() -> None:
    """Failed quality constraint returns an informative diagnostic message."""
    calc = MetricsCalculator()
    baseline = {"r2": 0.900}
    candidate = {"r2": 0.850}

    passes, reason = calc.check_quality_constraint(
        baseline_metrics=baseline,
        candidate_metrics=candidate,
        tolerance=0.005,
        task_type="regression",
    )

    assert passes is False
    assert "FAIL" in reason
    assert "r2" in reason
    assert "0.8500" in reason
    assert "0.9000" in reason
