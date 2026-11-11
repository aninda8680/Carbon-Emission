"""
model_zoo.py – Returns a dict of model_name -> unfitted sklearn estimator.

Supports both classification and regression task types, with optional
XGBoost inclusion and size-aware defaults.
"""
from __future__ import annotations

from typing import Any

SEED = 42


def get_model_zoo(
    task_type: str,
    enable_xgboost: bool = False,
    dataset_size: int = 1000,
    n_features: int = 10,
    random_state: int = SEED,
) -> dict[str, Any]:
    """Return a dict of model_name -> unfitted sklearn estimator.

    Parameters
    ----------
    task_type : str
        "classification" or "regression".
    enable_xgboost : bool
        Whether to include XGBoost when installed.
    dataset_size : int
        Number of rows in the dataset (kept for size-aware consumers).
    n_features : int
        Number of features (kept for size-aware consumers).
    random_state : int
        Random seed threaded to stochastic estimators.

    Returns
    -------
    dict[str, Any]
        Mapping of model name to an unfitted estimator instance.
    """
    seed = random_state
    task = task_type.lower()

    if task == "classification":
        from sklearn.dummy import DummyClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
        from lightgbm import LGBMClassifier

        zoo: dict[str, Any] = {
            "dummy": DummyClassifier(strategy="most_frequent"),
            "logistic_regression": LogisticRegression(
                max_iter=1000, random_state=seed
            ),
            "decision_tree": DecisionTreeClassifier(random_state=seed),
            "random_forest": RandomForestClassifier(
                n_estimators=100, random_state=seed, n_jobs=1
            ),
            "extra_trees": ExtraTreesClassifier(
                n_estimators=100, random_state=seed, n_jobs=1
            ),
            "lightgbm": LGBMClassifier(n_jobs=1, verbose=-1, random_state=seed),
        }

        if enable_xgboost:
            try:
                from xgboost import XGBClassifier

                zoo["xgboost"] = XGBClassifier(
                    n_jobs=1,
                    random_state=seed,
                    verbosity=0,
                    eval_metric="mlogloss",
                )
            except ImportError:
                pass  # XGBoost not installed – skip silently

        return zoo

    elif task == "regression":
        from sklearn.dummy import DummyRegressor
        from sklearn.linear_model import LinearRegression, Ridge
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
        from lightgbm import LGBMRegressor

        reg_zoo: dict[str, Any] = {
            "dummy": DummyRegressor(),
            "linear_regression": LinearRegression(),
            "ridge": Ridge(random_state=seed),
            "decision_tree": DecisionTreeRegressor(random_state=seed),
            "random_forest": RandomForestRegressor(
                n_estimators=100, random_state=seed, n_jobs=1
            ),
            "extra_trees": ExtraTreesRegressor(
                n_estimators=100, random_state=seed, n_jobs=1
            ),
            "lightgbm": LGBMRegressor(n_jobs=1, verbose=-1, random_state=seed),
        }

        if enable_xgboost:
            try:
                from xgboost import XGBRegressor

                reg_zoo["xgboost"] = XGBRegressor(
                    n_jobs=1,
                    random_state=seed,
                    verbosity=0,
                )
            except ImportError:
                pass  # XGBoost not installed – skip silently

        return reg_zoo

    else:
        raise ValueError(
            f"Unsupported task_type '{task_type}'. "
            "Expected 'classification' or 'regression'."
        )
