"""
optimizer.py – HyperparameterOptimizer using Optuna TPE + MedianPruner.

CRITICAL: The objective function uses CROSS-VALIDATION on X_train ONLY.
The held-out test set is NEVER touched during optimization.
"""
from __future__ import annotations

import logging
import time
import warnings
from typing import Any

import numpy as np
import optuna
from sklearn.base import clone
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold, KFold

# Silence Optuna's default INFO logs globally
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("optuna").setLevel(logging.WARNING)


class HyperparameterOptimizer:
    """Optimize hyperparameters via Optuna TPE with cross-validation scoring.

    The objective function evaluates candidates using ONLY the training data
    passed to ``optimize()``. The held-out test set is never seen.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        model_name: str,
        X_train,
        y_train,
        cv_splitter: BaseCrossValidator,
        task_type: str,
        n_trials: int = 30,
        random_state: int = 42,
        study_name: str | None = None,
        groups=None,
    ) -> dict[str, Any]:
        """Run Optuna optimization on the training set via cross-validation.

        Parameters
        ----------
        model_name : str
            Key in the model zoo (e.g., "random_forest", "lightgbm").
        X_train : array-like, shape (n_samples, n_features)
            Training features. NEVER includes test data.
        y_train : array-like, shape (n_samples,)
            Training labels.
        cv_splitter : BaseCrossValidator
            CV splitter that defines inner CV folds on X_train.
        task_type : str – "classification" or "regression"
        n_trials : int
            Optuna trial budget.
        random_state : int
        study_name : str or None
            If given, an SQLite-backed study is created (pass a path or name).
            Otherwise, the study is stored in memory.
        groups : array-like or None
            Optional group labels for group-aware CV splitters.

        Returns
        -------
        dict with keys: best_params, best_score, n_trials, duration_seconds,
        study_name.
        """
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        groups_arr = np.asarray(groups) if groups is not None else None

        from .model_zoo import get_model_zoo

        enable_xgb = model_name.lower() == "xgboost"
        zoo = get_model_zoo(
            task_type,
            enable_xgboost=enable_xgb,
            dataset_size=len(X_train),
            n_features=X_train.shape[1] if X_train.ndim == 2 else 1,
            random_state=random_state,
        )
        if model_name not in zoo:
            raise ValueError(
                f"Model '{model_name}' not in zoo for task '{task_type}'."
            )

        # Build the model factory once so the objective closure is clean
        def _build_model(trial: optuna.Trial):
            params = self.get_param_space(model_name, trial)
            base_model = clone(zoo[model_name])
            valid_params = base_model.get_params()
            safe_params = {k: v for k, v in params.items() if k in valid_params}
            base_model.set_params(**safe_params)
            return base_model

        def objective(trial: optuna.Trial) -> float:
            model = _build_model(trial)

            cv_scores: list[float] = []
            split_kwargs = {"groups": groups_arr} if groups_arr is not None else {}
            split_gen = cv_splitter.split(X_train, y_train, **split_kwargs)

            for train_idx, val_idx in split_gen:
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                fold_model = clone(model)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fold_model.fit(X_tr, y_tr)
                    y_pred = fold_model.predict(X_val)

                if task_type.lower() == "classification":
                    from sklearn.metrics import f1_score
                    score = float(
                        f1_score(y_val, y_pred, average="macro", zero_division=0)
                    )
                else:
                    from sklearn.metrics import r2_score
                    score = float(r2_score(y_val, y_pred))

                cv_scores.append(score)

                # Report intermediate value for pruning
                trial.report(float(np.mean(cv_scores)), step=len(cv_scores))
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return float(np.mean(cv_scores))

        # Build study
        sampler = optuna.samplers.TPESampler(seed=random_state)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)

        storage = None
        resolved_name = study_name or f"study_{model_name}_{task_type}"

        if study_name and not study_name.startswith(":memory:"):
            if study_name.startswith("sqlite:///"):
                storage = study_name
            elif study_name.endswith(".db"):
                storage = f"sqlite:///{study_name}"
            else:
                storage = f"sqlite:///{study_name}.db"

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name=resolved_name,
            storage=storage,
            load_if_exists=True,
        )

        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        duration = time.perf_counter() - t0

        completed_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        best_trial = study.best_trial if completed_trials else None

        best_params: dict[str, Any] = {}
        if best_trial is not None:
            raw_params = dict(best_trial.params)
            # Clean helper flags so returned params can be passed directly to set_params()
            if raw_params.pop("rf_max_depth_none", False):
                raw_params["max_depth"] = None
            if raw_params.pop("et_max_depth_none", False):
                raw_params["max_depth"] = None
            valid_estimator_params = zoo[model_name].get_params()
            best_params = {
                k: v for k, v in raw_params.items() if k in valid_estimator_params
            }

        return {
            "best_params": best_params,
            "best_score": best_trial.value if best_trial else None,
            "n_trials": len(study.trials),
            "duration_seconds": round(duration, 4),
            "study_name": resolved_name,
        }

    # ------------------------------------------------------------------
    # Search space definitions
    # ------------------------------------------------------------------

    @staticmethod
    def get_param_space(model_name: str, trial: optuna.Trial) -> dict[str, Any]:
        """Define Optuna search space for each model.

        Parameters
        ----------
        model_name : str
        trial : optuna.Trial

        Returns
        -------
        dict of hyperparameter name -> value suggested by trial.
        """
        name = model_name.lower()

        if name in ("random_forest", "randomforest"):
            max_depth_flag = trial.suggest_categorical(
                "rf_max_depth_none", [True, False]
            )
            max_depth = (
                None
                if max_depth_flag
                else trial.suggest_int("max_depth", 3, 20)
            )
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": max_depth,
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            }

        elif name in ("lightgbm", "lgbm", "lgbmclassifier", "lgbmregressor"):
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            }

        elif name in ("decision_tree", "decisiontree"):
            return {
                "max_depth": trial.suggest_int("max_depth", 2, 15),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
            }

        elif name in ("logistic_regression", "logisticregression"):
            return {
                "C": trial.suggest_float("C", 0.001, 100.0, log=True),
                "solver": trial.suggest_categorical("solver", ["lbfgs", "saga"]),
            }

        elif name in ("extra_trees", "extratrees"):
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
            }

        elif name in ("xgboost", "xgb"):
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            }

        elif name in ("ridge",):
            return {
                "alpha": trial.suggest_float("alpha", 0.001, 100.0, log=True),
            }

        else:
            # No hyperparameters defined for this model (e.g., dummy, linear)
            return {}
