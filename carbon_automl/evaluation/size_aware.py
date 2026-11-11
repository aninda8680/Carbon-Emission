"""
size_aware.py – SizeAwareOptimizer for multi-dimensional dataset adaptation and
synthetic scaling benchmarks.

Selects model zoo subsets, HPO / tuning strategies, and memory-aware chunking
based on a composite profile of:
- row count (n_rows)
- feature dimensionality (n_features & n_features / n_rows ratio)
- feature cardinality (high-cardinality categoricals)
- dtypes (float64 vs float32/int/object memory footprint)
- memory footprint (MB)
- task complexity (classification class count / target structure)

Never relies on a fixed row-count threshold alone.

Also provides ``run_synthetic_scale_benchmark(sizes=[10000, 100000, 1000000])``
explicitly tagged as synthetic scaling measurements, kept completely separate
from real-world evaluation results.
"""
from __future__ import annotations

import gc
import os
import time
import tracemalloc
import warnings
from typing import Any, Iterator

import numpy as np
import pandas as pd
from sklearn.base import clone

from .model_zoo import get_model_zoo

ROUGH_POWER_WATTS = 10.0
JOULES_PER_KWH = 3_600_000.0


class SizeAwareOptimizer:
    """Adapt model selection, tuning strategy, and chunking to data complexity."""

    # ------------------------------------------------------------------
    # Complexity assessment (multi-dimensional, never row-count alone)
    # ------------------------------------------------------------------

    def assess_dataset_complexity(
        self,
        X=None,
        y=None,
        task_type: str = "classification",
        feature_types: dict[str, str] | None = None,
        *,
        n_rows: int | None = None,
        n_features: int | None = None,
        memory_mb: float | None = None,
        max_cardinality: int | None = None,
        high_cardinality_count: int | None = None,
        n_classes: int | None = None,
        dtypes: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Compute a multi-dimensional dataset complexity profile.

        Accepts either an actual feature matrix/DataFrame ``X`` (and optional
        ``y``) or summary metadata kwargs.
        """
        task = task_type.lower()
        dtype_breakdown: dict[str, int] = {
            "float64": 0,
            "float32": 0,
            "int": 0,
            "object_or_category": 0,
            "other": 0,
        }

        if X is not None:
            if isinstance(X, pd.DataFrame):
                resolved_rows, resolved_cols = X.shape
                if memory_mb is None:
                    memory_mb = float(X.memory_usage(deep=True).sum() / (1024.0 ** 2))
                for col, dt in X.dtypes.items():
                    dt_str = str(dt)
                    if "float64" in dt_str:
                        dtype_breakdown["float64"] += 1
                    elif "float32" in dt_str:
                        dtype_breakdown["float32"] += 1
                    elif "int" in dt_str or "uint" in dt_str:
                        dtype_breakdown["int"] += 1
                    elif "object" in dt_str or "category" in dt_str or "string" in dt_str:
                        dtype_breakdown["object_or_category"] += 1
                    else:
                        dtype_breakdown["other"] += 1

                if max_cardinality is None or high_cardinality_count is None:
                    card_vals: list[int] = []
                    hc_count = 0
                    for col in X.columns:
                        ftype = feature_types.get(str(col)) if feature_types else None
                        if (
                            ftype in ("categorical", "high_cardinality_categorical")
                            or X[col].dtype == object
                            or isinstance(X[col].dtype, pd.CategoricalDtype)
                        ):
                            nunique = int(X[col].nunique(dropna=True))
                            card_vals.append(nunique)
                            if nunique > 50 or ftype == "high_cardinality_categorical":
                                hc_count += 1
                    if max_cardinality is None:
                        max_cardinality = max(card_vals) if card_vals else 0
                    if high_cardinality_count is None:
                        high_cardinality_count = hc_count
            else:
                arr = np.asarray(X)
                if arr.ndim == 1:
                    resolved_rows, resolved_cols = arr.shape[0], 1
                else:
                    resolved_rows, resolved_cols = arr.shape[0], arr.shape[1]
                if memory_mb is None:
                    memory_mb = float(arr.nbytes / (1024.0 ** 2))
                dt_str = str(arr.dtype)
                if "float64" in dt_str:
                    dtype_breakdown["float64"] = resolved_cols
                elif "float32" in dt_str:
                    dtype_breakdown["float32"] = resolved_cols
                elif "int" in dt_str or "uint" in dt_str:
                    dtype_breakdown["int"] = resolved_cols
                else:
                    dtype_breakdown["other"] = resolved_cols
        else:
            if n_rows is None or n_features is None:
                raise ValueError(
                    "Provide either X or both n_rows and n_features."
                )
            resolved_rows = int(n_rows)
            resolved_cols = int(n_features)
            if dtypes:
                for dt_str in dtypes.values():
                    s = str(dt_str).lower()
                    if "float64" in s:
                        dtype_breakdown["float64"] += 1
                    elif "float32" in s:
                        dtype_breakdown["float32"] += 1
                    elif "int" in s:
                        dtype_breakdown["int"] += 1
                    elif "object" in s or "category" in s or "str" in s:
                        dtype_breakdown["object_or_category"] += 1
                    else:
                        dtype_breakdown["other"] += 1
            else:
                dtype_breakdown["float64"] = resolved_cols

            if memory_mb is None:
                # Estimate 8 bytes per float64/object cell, 4 bytes per float32
                avg_bytes = 4.0 if dtype_breakdown["float32"] == resolved_cols else 8.0
                memory_mb = float(resolved_rows * resolved_cols * avg_bytes / (1024.0 ** 2))

        if high_cardinality_count is None and feature_types:
            high_cardinality_count = sum(
                1 for v in feature_types.values() if v == "high_cardinality_categorical"
            )
        high_cardinality_count = int(high_cardinality_count or 0)
        max_cardinality = int(max_cardinality or 0)

        if y is not None and n_classes is None and task == "classification":
            n_classes = int(len(np.unique(np.asarray(y))))
        elif n_classes is None:
            n_classes = 2 if task == "classification" else 1

        n_cells = int(resolved_rows * resolved_cols)
        dim_ratio = float(resolved_cols / max(1, resolved_rows))
        is_wide = resolved_cols >= 100 or dim_ratio > 0.1
        float64_heavy = (
            dtype_breakdown["float64"] >= max(1, resolved_cols * 0.5)
        )

        # Composite complexity score (multi-factor, NOT row-count alone)
        # Factors: memory footprint, cell volume, dimensionality, cardinality, task classes
        mem_factor = np.log10(max(memory_mb, 0.01) + 1.0) / 2.5        # ~0..1+
        cell_factor = np.log10(max(n_cells, 100) / 100.0) / 5.5        # ~0..1+
        dim_factor = min(1.0, resolved_cols / 150.0) * 0.45
        card_factor = min(1.0, (max_cardinality / 500.0) + 0.15 * high_cardinality_count) * 0.35
        class_factor = min(0.5, max(0, (n_classes - 2)) * 0.05) if task == "classification" else 0.15
        dtype_factor = 0.15 if (float64_heavy and memory_mb > 10.0) else 0.0

        complexity_score = float(
            0.35 * mem_factor
            + 0.35 * cell_factor
            + dim_factor
            + card_factor
            + class_factor
            + dtype_factor
        )

        # Tier determined by composite profile (memory, dimensionality, cardinality, cells)
        if complexity_score >= 0.95 or memory_mb >= 100.0 or n_cells >= 10_000_000:
            tier = "large_scale"
        elif complexity_score >= 0.55 or memory_mb >= 20.0 or n_cells >= 1_000_000 or is_wide:
            tier = "heavy"
        elif complexity_score >= 0.28 or n_cells >= 50_000 or high_cardinality_count > 0:
            tier = "moderate"
        else:
            tier = "light"

        return {
            "n_rows": int(resolved_rows),
            "n_features": int(resolved_cols),
            "n_cells": n_cells,
            "memory_mb": round(float(memory_mb), 4),
            "dimensionality_ratio": round(dim_ratio, 6),
            "is_wide_dataset": bool(is_wide),
            "dtype_breakdown": dtype_breakdown,
            "float64_heavy": bool(float64_heavy),
            "max_cardinality": max_cardinality,
            "high_cardinality_count": high_cardinality_count,
            "n_classes": int(n_classes),
            "task_type": task,
            "complexity_score": round(complexity_score, 4),
            "tier": tier,
        }

    # ------------------------------------------------------------------
    # Strategy recommendation
    # ------------------------------------------------------------------

    def recommend_strategy(
        self,
        X=None,
        y=None,
        task_type: str = "classification",
        feature_types: dict[str, str] | None = None,
        enable_xgboost: bool = False,
        random_state: int = 42,
        **complexity_kwargs,
    ) -> dict[str, Any]:
        """Select model zoo, tuning strategy, and chunking from dataset complexity."""
        task = task_type.lower()
        profile = self.assess_dataset_complexity(
            X=X,
            y=y,
            task_type=task,
            feature_types=feature_types,
            **complexity_kwargs,
        )

        tier = profile["tier"]
        n_rows = profile["n_rows"]
        n_features = profile["n_features"]
        memory_mb = profile["memory_mb"]
        is_wide = profile["is_wide_dataset"]
        high_card = profile["high_cardinality_count"]
        max_card = profile["max_cardinality"]
        float64_heavy = profile["float64_heavy"]

        full_zoo = get_model_zoo(
            task,
            enable_xgboost=enable_xgboost,
            dataset_size=n_rows,
            n_features=n_features,
            random_state=random_state,
        )

        rationale: list[str] = []

        # 1. Model zoo selection & tuning configuration
        if tier == "light":
            selected_names = list(full_zoo.keys())
            tuning_strategy = {
                "mode": "full_optuna_cv",
                "suggested_trials": 30,
                "cv_folds": 5,
                "use_pruning": True,
                "subsample_for_hpo": False,
                "subsample_fraction": 1.0,
                "early_stopping_recommended": False,
            }
            rationale.append(
                f"Tier 'light' ({n_rows}x{n_features}, {memory_mb:.2f} MB): "
                "all candidate models and full 5-fold Optuna HPO enabled."
            )

        elif tier == "moderate":
            selected_names = list(full_zoo.keys())
            if is_wide and task == "regression" and "linear_regression" in selected_names:
                # Unregularized OLS can be ill-conditioned on wide datasets; keep Ridge
                selected_names.remove("linear_regression")
                rationale.append(
                    "Wide feature space detected: preferring regularized 'ridge' over unregularized 'linear_regression'."
                )
            tuning_strategy = {
                "mode": "adaptive_pruned_optuna",
                "suggested_trials": 20,
                "cv_folds": 5,
                "use_pruning": True,
                "subsample_for_hpo": False,
                "subsample_fraction": 1.0,
                "early_stopping_recommended": True,
            }
            rationale.append(
                f"Tier 'moderate' (score={profile['complexity_score']}): "
                "using pruned Optuna CV (20 trials) and early stopping for boosting."
            )

        elif tier == "heavy":
            # Drop slow unregularized / deep ensemble models if wide or high-memory
            preferred = (
                ["dummy", "logistic_regression", "decision_tree", "lightgbm", "random_forest"]
                if task == "classification"
                else ["dummy", "ridge", "decision_tree", "lightgbm", "random_forest"]
            )
            if enable_xgboost and "xgboost" in full_zoo:
                preferred.append("xgboost")
            selected_names = [m for m in preferred if m in full_zoo]
            tuning_strategy = {
                "mode": "adaptive_subsample_hpo",
                "suggested_trials": 15,
                "cv_folds": 3,
                "use_pruning": True,
                "subsample_for_hpo": n_rows > 25_000,
                "subsample_fraction": 0.4 if n_rows > 25_000 else 1.0,
                "early_stopping_recommended": True,
            }
            rationale.append(
                f"Tier 'heavy' ({n_rows}x{n_features}, {memory_mb:.2f} MB, wide={is_wide}, "
                f"high_card={high_card}): narrowed zoo to scalable models with 3-fold pruned HPO."
            )

        else:  # large_scale
            preferred = (
                ["dummy", "logistic_regression", "decision_tree", "lightgbm"]
                if task == "classification"
                else ["dummy", "ridge", "decision_tree", "lightgbm"]
            )
            if enable_xgboost and "xgboost" in full_zoo:
                preferred.append("xgboost")
            selected_names = [m for m in preferred if m in full_zoo]
            tuning_strategy = {
                "mode": "fast_subsample_hpo",
                "suggested_trials": 10,
                "cv_folds": 3,
                "use_pruning": True,
                "subsample_for_hpo": True,
                "subsample_fraction": min(1.0, max(0.05, 50_000.0 / max(1, n_rows))),
                "early_stopping_recommended": True,
            }
            rationale.append(
                f"Tier 'large_scale' ({n_rows}x{n_features}, {memory_mb:.2f} MB): "
                "selecting O(N) / histogram-based learners (linear/ridge, decision_tree, lightgbm) "
                "with subsampled HPO and early stopping."
            )

        # Cardinality & multiclass adjustments
        if high_card > 0 or max_card > 100:
            rationale.append(
                f"High cardinality detected (max_cardinality={max_card}, count={high_card}): "
                "avoiding naive one-hot explosion; LightGBM/ordinal encoding preferred."
            )
        if task == "classification" and profile["n_classes"] > 10:
            rationale.append(
                f"High class count ({profile['n_classes']} classes): "
                "increasing boosting efficiency priority."
            )

        # Build adjusted estimator instances
        tailored_zoo: dict[str, Any] = {}
        for name in selected_names:
            est = clone(full_zoo[name])
            params = est.get_params()
            if tier in ("heavy", "large_scale"):
                if name in ("random_forest", "extra_trees") and "n_estimators" in params:
                    est.set_params(n_estimators=50, max_depth=12)
                elif name == "lightgbm":
                    est.set_params(
                        n_estimators=50 if tier == "large_scale" else 80,
                        max_depth=6 if tier == "large_scale" else 8,
                    )
                elif name == "decision_tree" and "max_depth" in params:
                    est.set_params(max_depth=10)
            tailored_zoo[name] = est

        # 2. Memory-aware chunking & dtype downcasting advice
        use_chunking = bool(
            memory_mb >= 50.0
            or profile["n_cells"] >= 2_000_000
            or (n_rows >= 50_000 and n_features >= 30)
        )
        if use_chunking:
            # Target ~16 MB per chunk for prediction / streaming transforms
            bytes_per_row = max(1.0, (memory_mb * 1024.0 * 1024.0) / max(1, n_rows))
            target_chunk_bytes = 16.0 * 1024.0 * 1024.0
            chunk_size = int(max(1000, min(50_000, target_chunk_bytes / bytes_per_row)))
        else:
            chunk_size = None

        safe_downcast_recommended = bool(float64_heavy and memory_mb >= 10.0)
        potential_savings_mb = (
            round(memory_mb * 0.45, 3) if safe_downcast_recommended else 0.0
        )
        if safe_downcast_recommended:
            rationale.append(
                f"Float64-heavy matrix ({memory_mb:.2f} MB): safe float64->float32 downcast "
                f"(verified lossless round-trip) can save ~{potential_savings_mb:.2f} MB."
            )
        if use_chunking:
            rationale.append(
                f"Memory-aware chunking enabled (chunk_size={chunk_size} rows) for batch inference/transforms."
            )

        memory_strategy = {
            "use_chunking": use_chunking,
            "chunk_size": chunk_size,
            "safe_downcast_recommended": safe_downcast_recommended,
            "estimated_downcast_savings_mb": potential_savings_mb,
            "estimated_peak_memory_mb": round(memory_mb * 2.5, 3),
        }

        return {
            "complexity_profile": profile,
            "selected_models": selected_names,
            "model_zoo": tailored_zoo,
            "tuning_strategy": tuning_strategy,
            "memory_strategy": memory_strategy,
            "rationale": rationale,
        }

    # ------------------------------------------------------------------
    # Memory-aware chunk iterator
    # ------------------------------------------------------------------

    def iter_chunks(
        self,
        X,
        y=None,
        chunk_size: int | None = None,
    ) -> Iterator[Any]:
        """Yield slices of ``X`` (and ``y`` if provided) of size ``chunk_size``."""
        n_rows = len(X)
        if chunk_size is None or chunk_size <= 0 or chunk_size >= n_rows:
            yield (X, y) if y is not None else X
            return

        is_df = isinstance(X, pd.DataFrame)
        is_y_pd = isinstance(y, (pd.Series, pd.DataFrame))

        for start in range(0, n_rows, chunk_size):
            end = min(start + chunk_size, n_rows)
            X_chunk = X.iloc[start:end] if is_df else X[start:end]
            if y is not None:
                y_chunk = y.iloc[start:end] if is_y_pd else y[start:end]
                yield X_chunk, y_chunk
            else:
                yield X_chunk

    # ------------------------------------------------------------------
    # Synthetic scale benchmark (explicitly isolated from real-world results)
    # ------------------------------------------------------------------

    def run_synthetic_scale_benchmark(
        self,
        sizes: list[int] | None = None,
        n_features: int = 20,
        task_type: str = "classification",
        model_name: str = "lightgbm",
        random_state: int = 42,
    ) -> dict[str, Any]:
        """Benchmark scaling across synthetic dataset sizes.

        Default sizes: ``[10_000, 100_000, 1_000_000]``.
        Results are explicitly flagged ``is_synthetic=True`` and kept strictly
        separate from real-world evaluation results.

        Measures per size:
        - runtime_seconds (data_generation, fit, predict, total)
        - peak_memory_mb (via tracemalloc + psutil RSS delta when available)
        - estimated_energy_joules (runtime * 10W rough proxy) and kWh
        - selected strategy tier & chunking decision
        """
        if sizes is None:
            sizes = [10_000, 100_000, 1_000_000]

        task = task_type.lower()
        rng = np.random.default_rng(seed=random_state)

        try:
            import psutil

            proc = psutil.Process(os.getpid())
        except Exception:
            proc = None

        size_results: list[dict[str, Any]] = []

        for n_rows in sizes:
            gc.collect()
            rss_before_mb = (
                proc.memory_info().rss / (1024.0 ** 2) if proc is not None else None
            )

            tracemalloc.start()
            t_total_start = time.perf_counter()

            # 1. Synthetic data generation (float32 to remain memory-efficient at 1M)
            t_gen0 = time.perf_counter()
            X_syn = rng.standard_normal((n_rows, n_features), dtype=np.float32)
            signal = 0.7 * X_syn[:, 0] - 0.5 * X_syn[:, 1] + 0.3 * X_syn[:, 2]
            if task == "classification":
                # 3-class balanced synthetic target
                q1, q2 = np.quantile(signal, [0.33, 0.67])
                y_syn = np.where(signal < q1, 0, np.where(signal < q2, 1, 2)).astype(
                    np.int32
                )
            else:
                y_syn = (signal + 0.1 * rng.standard_normal(n_rows)).astype(np.float32)
            gen_time = time.perf_counter() - t_gen0

            # 2. Size-aware strategy selection
            strategy = self.recommend_strategy(
                n_rows=n_rows,
                n_features=n_features,
                memory_mb=float(X_syn.nbytes / (1024.0 ** 2)),
                task_type=task,
                n_classes=3 if task == "classification" else 1,
                dtypes={f"f{i}": "float32" for i in range(n_features)},
                random_state=random_state,
            )

            zoo = strategy["model_zoo"]
            if model_name in zoo:
                estimator = clone(zoo[model_name])
            else:
                # Fallback to fastest available scalable model in strategy zoo
                fallback_zoo = get_model_zoo(
                    task,
                    dataset_size=n_rows,
                    n_features=n_features,
                    random_state=random_state,
                )
                estimator = clone(fallback_zoo.get(model_name, next(iter(zoo.values()))))

            # Ensure bounded single-run iteration budget at large scales
            est_params = estimator.get_params()
            if "n_estimators" in est_params and n_rows >= 100_000:
                estimator.set_params(n_estimators=min(30, int(est_params["n_estimators"])))
            if "max_depth" in est_params and n_rows >= 100_000 and est_params["max_depth"] is None:
                estimator.set_params(max_depth=8)

            # 3. Fit
            t_fit0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator.fit(X_syn, y_syn)
            fit_time = time.perf_counter() - t_fit0

            # 4. Predict (chunked if memory_strategy enables chunking)
            t_pred0 = time.perf_counter()
            chunk_sz = strategy["memory_strategy"]["chunk_size"]
            preds_chunks: list[np.ndarray] = []
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for X_c in self.iter_chunks(X_syn, chunk_size=chunk_sz):
                    preds_chunks.append(estimator.predict(X_c))
            _ = np.concatenate(preds_chunks) if len(preds_chunks) > 1 else preds_chunks[0]
            pred_time = time.perf_counter() - t_pred0

            total_runtime = time.perf_counter() - t_total_start
            _, tracemalloc_peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            rss_after_mb = (
                proc.memory_info().rss / (1024.0 ** 2) if proc is not None else None
            )
            tracemalloc_peak_mb = tracemalloc_peak_bytes / (1024.0 ** 2)
            rss_delta_mb = (
                max(0.0, rss_after_mb - rss_before_mb)
                if (rss_before_mb is not None and rss_after_mb is not None)
                else None
            )
            peak_memory_mb = max(
                tracemalloc_peak_mb,
                rss_delta_mb if rss_delta_mb is not None else 0.0,
            )

            # Estimated energy (explicitly flagged as estimate)
            compute_time = fit_time + pred_time
            est_energy_j = compute_time * ROUGH_POWER_WATTS
            est_energy_kwh = est_energy_j / JOULES_PER_KWH

            size_results.append(
                {
                    "n_rows": int(n_rows),
                    "n_features": int(n_features),
                    "data_memory_mb": round(float(X_syn.nbytes / (1024.0 ** 2)), 3),
                    "strategy_tier": strategy["complexity_profile"]["tier"],
                    "use_chunking": strategy["memory_strategy"]["use_chunking"],
                    "chunk_size": chunk_sz,
                    "data_gen_time_s": round(gen_time, 4),
                    "fit_time_s": round(fit_time, 4),
                    "predict_time_s": round(pred_time, 4),
                    "compute_runtime_s": round(compute_time, 4),
                    "total_runtime_s": round(total_runtime, 4),
                    "peak_memory_mb": round(float(peak_memory_mb), 3),
                    "tracemalloc_peak_mb": round(float(tracemalloc_peak_mb), 3),
                    "rss_delta_mb": round(float(rss_delta_mb), 3) if rss_delta_mb is not None else None,
                    "estimated_energy_joules": round(float(est_energy_j), 4),
                    "estimated_energy_kwh": float(est_energy_kwh),
                    "is_estimated_energy": True,
                    "is_synthetic": True,
                }
            )

            del X_syn, y_syn, preds_chunks, estimator
            gc.collect()

        return {
            "is_synthetic": True,
            "benchmark_scope": "synthetic_scaling_only",
            "real_world_separation_notice": (
                "Synthetic scale benchmark results characterize algorithmic scaling only "
                "and are strictly separate from real-world dataset evaluations."
            ),
            "task_type": task,
            "model_name": model_name,
            "n_features": int(n_features),
            "sizes": list(sizes),
            "results": size_results,
        }
