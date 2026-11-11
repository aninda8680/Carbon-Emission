"""Shared pytest fixtures for the Carbon-Aware AutoML test suite.

All synthetic datasets use fixed random seeds for complete determinism,
require no network access, and write any temporary files exclusively under
pytest's ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_classification_df() -> pd.DataFrame:
    """100-row DataFrame with 5 features (2 num, 2 cat, 1 binary), 3-class target 'label'."""
    rng = np.random.default_rng(42)
    n = 100

    # Balanced 3-class target (34, 33, 33)
    labels = np.array([0] * 34 + [1] * 33 + [2] * 33)
    rng.shuffle(labels)

    # Moderate class-correlated signal without deterministic leakage
    num_1 = rng.normal(loc=0.0, scale=1.0, size=n) + labels * 1.2
    num_2 = rng.normal(loc=50.0, scale=10.0, size=n) - labels * 4.0

    cats_a = np.array(["north", "south", "east", "west"])
    cats_b = np.array(["low", "medium", "high"])
    cat_1 = rng.choice(cats_a, size=n)
    cat_2 = rng.choice(cats_b, size=n)

    bin_1 = rng.integers(0, 2, size=n)

    return pd.DataFrame(
        {
            "num_1": np.round(num_1, 4),
            "num_2": np.round(num_2, 4),
            "cat_1": cat_1,
            "cat_2": cat_2,
            "bin_1": bin_1,
            "label": labels,
        }
    )


@pytest.fixture
def imbalanced_df() -> pd.DataFrame:
    """100 rows, 2 classes (90/10 imbalance), 4 numerical features."""
    rng = np.random.default_rng(101)
    n = 100
    labels = np.array([0] * 90 + [1] * 10)
    rng.shuffle(labels)

    return pd.DataFrame(
        {
            "feat_1": rng.normal(0, 1, size=n) + labels * 1.5,
            "feat_2": rng.normal(10, 2, size=n),
            "feat_3": rng.uniform(0, 100, size=n),
            "feat_4": rng.normal(-5, 3, size=n) - labels * 2.0,
            "label": labels,
        }
    )


@pytest.fixture
def regression_df() -> pd.DataFrame:
    """100 rows, 5 numerical features, continuous target 'value'."""
    rng = np.random.default_rng(202)
    n = 100
    X_data = {
        f"feat_{i}": rng.normal(loc=float(i), scale=2.0, size=n)
        for i in range(1, 6)
    }
    noise = rng.normal(0.0, 0.5, size=n)
    value = (
        2.5 * X_data["feat_1"]
        - 1.3 * X_data["feat_2"]
        + 0.8 * X_data["feat_3"]
        + noise
    )
    X_data["value"] = np.round(value, 5)
    return pd.DataFrame(X_data)


@pytest.fixture
def missing_values_df() -> pd.DataFrame:
    """100 rows with exactly 30% (30 rows) missing in 2 feature columns."""
    rng = np.random.default_rng(303)
    n = 100
    labels = np.array([0] * 34 + [1] * 33 + [2] * 33)
    rng.shuffle(labels)

    miss_num = rng.normal(10.0, 3.0, size=n).astype(float)
    miss_idx_1 = rng.choice(n, size=30, replace=False)
    miss_num[miss_idx_1] = np.nan

    miss_cat = rng.choice(["A", "B", "C"], size=n).astype(object)
    miss_idx_2 = rng.choice(n, size=30, replace=False)
    miss_cat[miss_idx_2] = np.nan

    return pd.DataFrame(
        {
            "miss_num": miss_num,
            "miss_cat": miss_cat,
            "clean_num": rng.normal(0.0, 1.0, size=n),
            "clean_cat": rng.choice(["X", "Y"], size=n),
            "label": labels,
        }
    )


@pytest.fixture
def high_cardinality_df() -> pd.DataFrame:
    """100 rows, 1 categorical feature with 80 unique categories (< n_rows)."""
    rng = np.random.default_rng(404)
    n = 100
    # 80 distinct categories across 100 rows so n_unique > 50 and n_unique < n_rows
    base_cats = [f"cat_{i:03d}" for i in range(80)]
    extra_cats = list(rng.choice(base_cats, size=20, replace=True))
    high_card = np.array(base_cats + extra_cats, dtype=object)
    rng.shuffle(high_card)

    return pd.DataFrame(
        {
            "high_card": high_card,
            "num_feat": rng.normal(0.0, 1.0, size=n),
            "label": rng.integers(0, 3, size=n),
        }
    )


@pytest.fixture
def leakage_df() -> pd.DataFrame:
    """DataFrame where 'leaky_feature' deterministically predicts 'label' (like MUAC)."""
    rng = np.random.default_rng(505)
    n = 120
    labels = np.array([0] * 40 + [1] * 40 + [2] * 40)
    rng.shuffle(labels)

    # Non-overlapping ranges per class:
    # class 0 -> [8.0, 11.4], class 1 -> [11.6, 12.4], class 2 -> [12.6, 16.0]
    leaky = np.empty(n, dtype=float)
    leaky[labels == 0] = rng.uniform(8.0, 11.4, size=(labels == 0).sum())
    leaky[labels == 1] = rng.uniform(11.6, 12.4, size=(labels == 1).sum())
    leaky[labels == 2] = rng.uniform(12.6, 16.0, size=(labels == 2).sum())

    return pd.DataFrame(
        {
            "clean_num_1": rng.normal(24.0, 8.0, size=n),
            "clean_num_2": rng.normal(85.0, 10.0, size=n),
            "clean_cat": rng.choice(["Urban", "Rural"], size=n),
            "leaky_feature": np.round(leaky, 2),
            "label": labels,
        }
    )


@pytest.fixture
def sample_csv_path(
    tmp_path: Path, sample_classification_df: pd.DataFrame
) -> Path:
    """Save sample_classification_df to a temporary CSV and return its Path."""
    csv_path = tmp_path / "sample_classification.csv"
    sample_classification_df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_parquet_path(
    tmp_path: Path, sample_classification_df: pd.DataFrame
) -> Path:
    """Save sample_classification_df to a temporary Parquet file and return its Path."""
    parquet_path = tmp_path / "sample_classification.parquet"
    sample_classification_df.to_parquet(parquet_path, index=False)
    return parquet_path


@pytest.fixture
def malnutrition_dataset_path() -> Path:
    """Locate the actual malnutrition CSV in datasets/ or project root, else skip."""
    candidates = [
        PROJECT_ROOT / "datasets" / "child_health_westbengal_1000.csv",
        PROJECT_ROOT / "child_health_westbengal_1000.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    pytest.skip("child_health_westbengal_1000.csv not found in datasets/ or project root")


@pytest.fixture
def malnutrition_df(malnutrition_dataset_path: Path) -> pd.DataFrame:
    """Load the actual child malnutrition dataset, skipping if unavailable."""
    return pd.read_csv(malnutrition_dataset_path)


@pytest.fixture
def config_dict(sample_csv_path: Path, tmp_path: Path) -> dict[str, Any]:
    """Return a valid CarbonAutoMLConfig dictionary for classification."""
    return {
        "dataset_path": str(sample_csv_path),
        "target_column": "label",
        "task_type": "classification",
        "exclude_columns": [],
        "group_column": None,
        "timestamp_column": None,
        "random_seed": 42,
        "validation_strategy": "stratified_kfold",
        "n_folds": 5,
        "n_repeats": 3,
        "trial_budget": 10,
        "quality_tolerance": 0.005,
        "carbon_intensity_kg_per_kwh": 0.7,
        "carbon_budget_kg": None,
        "output_dir": str(tmp_path / "experiments"),
        "models_dir": str(tmp_path / "models"),
        "enable_xgboost": False,
        "measurement_repetitions": 3,
        "leakage_single_feature_threshold": 0.95,
        "leakage_correlation_threshold": 0.95,
    }
