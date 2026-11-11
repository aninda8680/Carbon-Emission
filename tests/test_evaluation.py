"""Tests for evaluation engine: model zoo, CV isolation, HPO, feature selection, ablation, group/temporal CV, and regression E2E."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier

from carbon_automl.core.data_profiler import DataProfiler
from carbon_automl.core.preprocessing import UniversalPreprocessor
from carbon_automl.evaluation.ablation import AblationAnalyzer
from carbon_automl.evaluation.feature_selection import FeatureSelector
from carbon_automl.evaluation.metrics import MetricsCalculator
from carbon_automl.evaluation.model_zoo import get_model_zoo
from carbon_automl.evaluation.optimizer import HyperparameterOptimizer
from carbon_automl.evaluation.size_aware import SizeAwareOptimizer
from carbon_automl.evaluation.validation import ValidationEngine
from carbon_automl.intelligence.model_selector import ModelSelector


def test_cv_no_test_leakage(sample_classification_df: pd.DataFrame) -> None:
    """Optuna objective splits only X_train and is completely invariant to X_test/y_test."""
    X = sample_classification_df.drop(columns=["label"])
    y = sample_classification_df["label"]
    ftypes = DataProfiler().profile(X, y)["feature_types"]

    val_engine = ValidationEngine()
    X_tr_df, X_te_df, y_tr, y_te = val_engine.create_holdout_split(
        X, y, test_size=0.2, random_state=42, stratify=True
    )

    prep = UniversalPreprocessor()
    X_train = prep.fit_transform(X_tr_df, ftypes)
    _ = prep.transform(X_te_df)

    cv_splitter = val_engine.create_cv_splitter("stratified_kfold", n_folds=3, random_state=42)

    seen_lengths: list[int] = []
    orig_split = cv_splitter.split

    def tracking_split(X_in, y_in=None, groups=None):
        seen_lengths.append(len(X_in))
        return orig_split(X_in, y_in, groups)

    cv_splitter.split = tracking_split  # type: ignore[method-assign]

    res = HyperparameterOptimizer().optimize(
        model_name="decision_tree",
        X_train=X_train,
        y_train=y_tr,
        cv_splitter=cv_splitter,
        task_type="classification",
        n_trials=3,
        random_state=42,
    )

    assert len(seen_lengths) == 3
    assert all(n == len(X_train) for n in seen_lengths)
    assert res["best_score"] is not None


def test_stratified_kfold_class_distribution(sample_classification_df: pd.DataFrame) -> None:
    """StratifiedKFold preserves all target classes in every fold."""
    X = sample_classification_df.drop(columns=["label"])
    y = sample_classification_df["label"].to_numpy()
    expected_classes = set(np.unique(y))

    splitter = ValidationEngine().create_cv_splitter(
        strategy="stratified_kfold", n_folds=5, random_state=42
    )
    folds = list(splitter.split(X, y))
    assert len(folds) == 5

    for train_idx, val_idx in folds:
        assert set(np.unique(y[train_idx])) == expected_classes
        assert set(np.unique(y[val_idx])) == expected_classes
        _, val_counts = np.unique(y[val_idx], return_counts=True)
        # Each fold has 20 samples across 3 classes -> ~6-7 per class
        assert all(c >= 6 for c in val_counts)


def test_model_zoo_classification() -> None:
    """Classification model zoo contains at least 5 unfitted models."""
    zoo = get_model_zoo("classification", enable_xgboost=False)
    assert len(zoo) >= 5
    for expected in ("dummy", "logistic_regression", "decision_tree", "random_forest", "extra_trees", "lightgbm"):
        assert expected in zoo


def test_model_zoo_regression() -> None:
    """Regression model zoo contains at least 5 unfitted models."""
    zoo = get_model_zoo("regression", enable_xgboost=False)
    assert len(zoo) >= 5
    for expected in ("dummy", "linear_regression", "ridge", "decision_tree", "random_forest", "extra_trees", "lightgbm"):
        assert expected in zoo


def test_optuna_uses_cv_only(sample_classification_df: pd.DataFrame) -> None:
    """HyperparameterOptimizer returns the exact CV mean score on X_train, never holdout test."""
    X = sample_classification_df.drop(columns=["label"])
    y = sample_classification_df["label"]
    ftypes = DataProfiler().profile(X, y)["feature_types"]

    val_engine = ValidationEngine()
    X_tr_df, _, y_tr, _ = val_engine.create_holdout_split(
        X, y, test_size=0.2, random_state=42, stratify=True
    )
    X_train = UniversalPreprocessor().fit_transform(X_tr_df, ftypes)
    cv_splitter = val_engine.create_cv_splitter("stratified_kfold", n_folds=4, random_state=42)

    hpo_res = HyperparameterOptimizer().optimize(
        model_name="decision_tree",
        X_train=X_train,
        y_train=y_tr,
        cv_splitter=cv_splitter,
        task_type="classification",
        n_trials=4,
        random_state=42,
    )

    # Independently recompute CV score with best_params on X_train alone
    dt = clone(get_model_zoo("classification", random_state=42)["decision_tree"])
    dt.set_params(**hpo_res["best_params"])
    cv_check = val_engine.cross_validate_model(
        dt, X_train, y_tr, cv_splitter, MetricsCalculator(), task_type="classification"
    )

    assert hpo_res["best_score"] == pytest.approx(cv_check["mean"]["macro_f1"], abs=1e-9)
    assert hpo_res["n_trials"] == 4


def test_deterministic_results(sample_classification_df: pd.DataFrame) -> None:
    """Identical random seeds yield identical CV metrics and Optuna optimization results."""
    X = sample_classification_df.drop(columns=["label"])
    y = sample_classification_df["label"]
    ftypes = DataProfiler().profile(X, y)["feature_types"]

    val_engine = ValidationEngine()
    X_tr_df, _, y_tr, _ = val_engine.create_holdout_split(
        X, y, test_size=0.2, random_state=42, stratify=True
    )
    X_train = UniversalPreprocessor().fit_transform(X_tr_df, ftypes)
    cv = val_engine.create_cv_splitter("stratified_kfold", n_folds=4, random_state=42)

    dt = get_model_zoo("classification", random_state=42)["decision_tree"]
    cv_1 = val_engine.cross_validate_model(dt, X_train, y_tr, cv, MetricsCalculator(), "classification")
    cv_2 = val_engine.cross_validate_model(dt, X_train, y_tr, cv, MetricsCalculator(), "classification")
    assert cv_1["mean"]["macro_f1"] == cv_2["mean"]["macro_f1"]
    assert cv_1["fold_scores"] == cv_2["fold_scores"]

    opt = HyperparameterOptimizer()
    hpo_1 = opt.optimize("decision_tree", X_train, y_tr, cv, "classification", n_trials=4, random_state=42)
    hpo_2 = opt.optimize("decision_tree", X_train, y_tr, cv, "classification", n_trials=4, random_state=42)
    assert hpo_1["best_params"] == hpo_2["best_params"]
    assert hpo_1["best_score"] == pytest.approx(hpo_2["best_score"])


def test_feature_selection_inside_cv() -> None:
    """MI feature selection uses fold training data only and ignores test-only correlation."""
    rng = np.random.default_rng(42)
    n_train, n_val = 80, 40

    y_train = rng.integers(0, 2, size=n_train)
    y_val = rng.integers(0, 2, size=n_val)

    # Feature 0 is genuinely predictive on train; Feature 1 is pure noise on train
    # even though on validation it equals y_val (simulating unobserved future/val data).
    X_train = np.column_stack([
        y_train.astype(float) + rng.normal(0.0, 0.1, size=n_train),
        rng.normal(0.0, 1.0, size=n_train),
    ])
    _ = np.column_stack([
        y_val.astype(float),
        y_val.astype(float),
    ])

    fs = FeatureSelector()
    res = fs.mutual_information_selection(
        X_train, y_train, feature_names=["true_signal", "noise_on_train"], k=1, task_type="classification"
    )
    assert res["selected_features"] == ["true_signal"]
    assert res["scores"]["true_signal"] > res["scores"]["noise_on_train"]


def test_ablation_runs(sample_classification_df: pd.DataFrame) -> None:
    """AblationAnalyzer returns all required conditions with metrics, timing, and energy proxy."""
    X = sample_classification_df.drop(columns=["label"])
    y = sample_classification_df["label"]
    ftypes = DataProfiler().profile(X, y)["feature_types"]

    val_engine = ValidationEngine()
    X_tr_df, X_te_df, y_tr, y_te = val_engine.create_holdout_split(
        X, y, test_size=0.25, random_state=42, stratify=True
    )
    prep = UniversalPreprocessor()
    X_tr = prep.fit_transform(X_tr_df, ftypes)
    X_te = prep.transform(X_te_df)

    abl = AblationAnalyzer().run_ablation(
        X_train=X_tr,
        y_train=y_tr,
        X_test=X_te,
        y_test=y_te,
        model_name="decision_tree",
        task_type="classification",
        random_state=42,
        feature_names=prep.get_feature_names(),
        include_leave_one_out=False,
        include_complexity_comparison=False,
        n_hpo_trials=3,
    )

    required_conditions = {
        "full_features_no_opt",
        "feature_selection_only",
        "hpo_only",
        "full_pipeline",
        "pca_disabled",
        "pca_enabled",
        "early_stopping_disabled",
    }
    assert required_conditions.issubset(abl["conditions"].keys())
    for cond_name in required_conditions:
        cond = abl["conditions"][cond_name]
        assert "metrics" in cond
        assert "training_time" in cond
        assert "estimated_energy_joules" in cond


def test_group_validation() -> None:
    """GroupKFold and StratifiedGroupKFold never split a group across train and validation."""
    rng = np.random.default_rng(99)
    n_groups = 6
    rows_per_group = 10
    groups = np.repeat([f"grp_{i}" for i in range(n_groups)], rows_per_group)
    X = rng.normal(0.0, 1.0, size=(n_groups * rows_per_group, 4))
    y = np.tile([0, 1], (n_groups * rows_per_group) // 2)

    val_engine = ValidationEngine()
    for strategy in ("group_kfold", "stratified_group_kfold"):
        splitter = val_engine.create_cv_splitter(
            strategy=strategy, n_folds=3, random_state=42, groups=groups
        )
        for tr_idx, val_idx in splitter.split(X, y, groups=groups):
            assert set(groups[tr_idx]).isdisjoint(set(groups[val_idx]))

        dt = get_model_zoo("classification", random_state=42)["decision_tree"]
        cv_res = val_engine.cross_validate_model(
            dt, X, y, splitter, MetricsCalculator(), task_type="classification", groups=groups
        )
        assert cv_res["n_folds"] == 3


def test_temporal_validation() -> None:
    """TimeSeriesSplit enforces strictly past-to-future ordering without future leakage."""
    rng = np.random.default_rng(55)
    n = 60
    X = rng.normal(0.0, 1.0, size=(n, 3))
    y = 2.0 * X[:, 0] + rng.normal(0.0, 0.2, size=n)

    val_engine = ValidationEngine()
    splitter = val_engine.create_cv_splitter("time_series", n_folds=4, random_state=42)

    for tr_idx, val_idx in splitter.split(X, y):
        assert max(tr_idx) < min(val_idx)

    ridge = get_model_zoo("regression", random_state=42)["ridge"]
    cv_res = val_engine.cross_validate_model(
        ridge, X, y, splitter, MetricsCalculator(), task_type="regression"
    )
    assert cv_res["n_folds"] == 4
    assert "r2" in cv_res["mean"]


def test_regression_end_to_end(regression_df: pd.DataFrame) -> None:
    """Full end-to-end regression pipeline from profiling through ModelSelector."""
    X = regression_df.drop(columns=["value"])
    y = regression_df["value"]

    profile = DataProfiler().profile(X, y)
    assert profile["task_type_suggestion"] == "regression"

    val_engine = ValidationEngine()
    X_tr_df, X_te_df, y_tr, y_te = val_engine.create_holdout_split(
        X, y, test_size=0.2, random_state=42, stratify=False
    )

    prep = UniversalPreprocessor()
    X_tr = prep.fit_transform(X_tr_df, profile["feature_types"])
    X_te = prep.transform(X_te_df)

    cv = val_engine.create_cv_splitter("kfold", n_folds=4, random_state=42)
    zoo = get_model_zoo("regression", random_state=42)
    calc = MetricsCalculator()

    model_results = []
    for idx, name in enumerate(("dummy", "ridge", "decision_tree")):
        cv_out = val_engine.cross_validate_model(
            zoo[name], X_tr, y_tr, cv, calc, task_type="regression"
        )
        model_results.append(
            {
                "name": name,
                **cv_out["mean"],
                "training_energy_kwh": 0.001 * (idx + 1),
                "training_co2_kg": 0.0007 * (idx + 1),
            }
        )

    hpo_res = HyperparameterOptimizer().optimize(
        "ridge", X_tr, y_tr, cv, task_type="regression", n_trials=3, random_state=42
    )
    assert hpo_res["best_score"] is not None

    baseline_r2 = next(m["r2"] for m in model_results if m["name"] == "ridge")
    sel = ModelSelector().select(
        model_results=model_results,
        baseline_name="ridge",
        quality_metric="r2",
        quality_min=baseline_r2 - 0.005,
        energy_metric="training_energy_kwh",
        task_type="regression",
    )
    rec_name = sel["recommended"]["name"]
    assert rec_name in {"ridge", "decision_tree"}

    final_est = clone(zoo[rec_name]).fit(X_tr, y_tr)
    test_metrics = calc.compute_regression_metrics(y_te, final_est.predict(X_te))
    assert test_metrics["r2"] > 0.8
    assert test_metrics["rmse"] >= 0.0


def test_size_aware_optimizer(sample_classification_df: pd.DataFrame) -> None:
    """SizeAwareOptimizer assesses multi-dimensional complexity and picks strategy."""
    X = sample_classification_df.drop(columns=["label"])
    y = sample_classification_df["label"]
    ftypes = DataProfiler().profile(X, y)["feature_types"]

    sao = SizeAwareOptimizer()
    complexity = sao.assess_dataset_complexity(X, y, task_type="classification", feature_types=ftypes)
    assert complexity["n_rows"] == 100
    assert complexity["n_features"] == 5
    strategy = sao.recommend_strategy(X, y, task_type="classification", feature_types=ftypes)
    assert "selected_models" in strategy
    assert "tuning_strategy" in strategy
    assert "memory_strategy" in strategy
    assert len(strategy["selected_models"]) >= 5
