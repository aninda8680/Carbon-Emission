"""Tests for LeakageDetector."""

from __future__ import annotations

import numpy as np
import pandas as pd

from carbon_automl.core.data_profiler import DataProfiler
from carbon_automl.leakage.detector import LeakageDetector


REQUIRED_WARNING_KEYS = {
    "feature",
    "detection_method",
    "evidence",
    "severity",
    "recommended_investigation",
}


def test_perfect_feature_flagged(leakage_df: pd.DataFrame) -> None:
    """A feature achieving >0.95 single-feature accuracy is flagged as critical."""
    X = leakage_df.drop(columns=["label"])
    y = leakage_df["label"]
    feature_types = DataProfiler().profile(X, y)["feature_types"]

    warnings = LeakageDetector().detect(
        X, y, feature_types=feature_types, task_type="classification"
    )

    single_feat_hits = [
        w
        for w in warnings
        if w["feature"] == "leaky_feature"
        and w["detection_method"] == "single_feature_performance"
    ]
    assert len(single_feat_hits) == 1
    assert single_feat_hits[0]["severity"] == "critical"


def test_muac_leakage_flagged(malnutrition_df: pd.DataFrame) -> None:
    """On the actual malnutrition dataset, MUAC is flagged as critical leakage."""
    target_col = "Malnutrition Status"
    X = malnutrition_df.drop(columns=[target_col])
    y = malnutrition_df[target_col]
    feature_types = DataProfiler().profile(X, y)["feature_types"]

    warnings = LeakageDetector().detect(
        X, y, feature_types=feature_types, task_type="classification"
    )

    muac_critical = [
        w
        for w in warnings
        if "Mid-Upper Arm Circumference" in w["feature"]
        and w["severity"] == "critical"
    ]
    assert len(muac_critical) >= 1


def test_no_false_positive_on_clean_data() -> None:
    """Multiple noisy informative features without leakage trigger zero leakage warnings."""
    rng = np.random.default_rng(777)
    n = 150
    labels = np.array([0] * 50 + [1] * 50 + [2] * 50)
    rng.shuffle(labels)

    # All features carry moderate genuine signal (<0.75 single-feature accuracy)
    # and overlapping ranges, so no single feature monopolizes MI or predicts >0.95.
    X = pd.DataFrame(
        {
            "f1": rng.normal(0.0, 1.0, size=n) + labels * 1.1,
            "f2": rng.normal(5.0, 1.0, size=n) - labels * 1.1,
            "f3": rng.normal(10.0, 1.2, size=n) + (labels == 1) * 1.5,
            "cat_clean": np.where(labels == 0, rng.choice(["A", "B"], size=n), rng.choice(["B", "C"], size=n)),
        }
    )
    y = pd.Series(labels, name="label")
    feature_types = DataProfiler().profile(X, y)["feature_types"]

    warnings = LeakageDetector().detect(
        X, y, feature_types=feature_types, task_type="classification"
    )

    assert warnings == []


def test_constant_feature_flagged(sample_classification_df: pd.DataFrame) -> None:
    """Constant feature columns are flagged with detection_method='constant_feature'."""
    X = sample_classification_df.drop(columns=["label"]).copy()
    X["const_col"] = 42
    y = sample_classification_df["label"]
    feature_types = DataProfiler().profile(X, y)["feature_types"]

    warnings = LeakageDetector().detect(
        X, y, feature_types=feature_types, task_type="classification"
    )

    const_hits = [
        w
        for w in warnings
        if w["feature"] == "const_col" and w["detection_method"] == "constant_feature"
    ]
    assert len(const_hits) == 1


def test_identifier_feature_flagged(sample_classification_df: pd.DataFrame) -> None:
    """Identifier-like columns (all unique strings) are flagged as identifier_leakage."""
    X = sample_classification_df.drop(columns=["label"]).copy()
    X["record_id"] = [f"REC-{i:04d}" for i in range(len(X))]
    y = sample_classification_df["label"]
    feature_types = DataProfiler().profile(X, y)["feature_types"]

    warnings = LeakageDetector().detect(
        X, y, feature_types=feature_types, task_type="classification"
    )

    id_hits = [
        w
        for w in warnings
        if w["feature"] == "record_id" and w["detection_method"] == "identifier_leakage"
    ]
    assert len(id_hits) == 1


def test_leakage_warning_has_required_fields(leakage_df: pd.DataFrame) -> None:
    """Every leakage warning dict includes all required schema keys."""
    X = leakage_df.drop(columns=["label"])
    y = leakage_df["label"]
    feature_types = DataProfiler().profile(X, y)["feature_types"]

    warnings = LeakageDetector().detect(
        X, y, feature_types=feature_types, task_type="classification"
    )

    assert len(warnings) > 0
    for warning in warnings:
        assert REQUIRED_WARNING_KEYS.issubset(warning.keys())
        for key in REQUIRED_WARNING_KEYS:
            assert isinstance(warning[key], str) and len(warning[key]) > 0


def test_leakage_does_not_remove_features(leakage_df: pd.DataFrame) -> None:
    """LeakageDetector never mutates X or drops flagged features."""
    X = leakage_df.drop(columns=["label"]).copy()
    y = leakage_df["label"].copy()
    original_shape = X.shape
    original_columns = list(X.columns)

    warnings = LeakageDetector().detect(X, y, task_type="classification")

    assert len(warnings) > 0
    assert X.shape == original_shape
    assert list(X.columns) == original_columns
    assert "leaky_feature" in X.columns
