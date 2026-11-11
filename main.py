"""
main.py – Carbon-Aware AutoML Framework CLI entry point.

Usage:
    python main.py --config config.yaml [subcommand] [options]

Subcommands:
    profile    Profile dataset and report leakage warnings.
    train      Full training pipeline (default).
    benchmark  Repeated-CV training for statistical comparison.
    report     Generate HTML report from saved experiment results.
    predict    Run predictions with a saved model on a new CSV.
    compare    Compare two experiment result JSON files.
    validate   Validate config + dataset without training.

If no subcommand is given, the full training pipeline runs (same as `train`).
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Logging & project paths
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("carbon_automl")

ROOT = Path(__file__).parent


def _resolve_path(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else ROOT / p


def _import(module: str, attr: str | None = None) -> Any:
    """Safely import a module or attribute, returning None on ImportError."""
    try:
        mod = importlib.import_module(module)
        return getattr(mod, attr) if attr else mod
    except ImportError as exc:
        logger.warning("Optional module unavailable (%s): %s", module, exc)
        return None


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str):
    """Load and validate YAML config using Pydantic v2 CarbonAutoMLConfig."""
    schema_cls = (
        _import("carbon_automl.config.schema", "CarbonAutoMLConfig")
        or _import("carbon_automl.config.schema", "AutoMLConfig")
    )
    if schema_cls is None:
        raise RuntimeError("carbon_automl.config.schema is unavailable")

    path = Path(config_path)
    if not path.is_absolute() and not path.exists():
        path = ROOT / config_path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")

    # Preserve carbon_intensity_source even if schema version omits it
    extra_source = None
    if (
        "carbon_intensity_source" in raw
        and hasattr(schema_cls, "model_fields")
        and "carbon_intensity_source" not in schema_cls.model_fields
    ):
        extra_source = raw.pop("carbon_intensity_source")

    validator_cls = _import("carbon_automl.core.schema_validator", "SchemaValidator")
    if validator_cls is not None:
        cfg = validator_cls.validate_config(raw)
    else:
        cfg = schema_cls(**raw)

    if extra_source is not None and not hasattr(cfg, "carbon_intensity_source"):
        object.__setattr__(cfg, "_carbon_intensity_source_fallback", extra_source)

    logger.info("Configuration loaded and validated from %s", path)
    return cfg


# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------

def _print_banner(title: str) -> None:
    width = 72
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _print_leakage_warnings(warnings: list[dict[str, Any]]) -> bool:
    """Print leakage warnings prominently; return True if critical leakage found."""
    if not warnings:
        print("  ✓ No data leakage warnings detected.")
        return False

    critical = [w for w in warnings if w.get("severity") == "critical"]
    high = [w for w in warnings if w.get("severity") == "high"]
    others = [w for w in warnings if w.get("severity") not in ("critical", "high")]

    for w in critical:
        print(f"\n  [CRITICAL LEAKAGE] Feature: {w.get('feature')}")
        print(f"    Method : {w.get('detection_method')}")
        print(f"    Evidence: {w.get('evidence')}")
        print(f"    Action  : {w.get('recommended_investigation')}")

    for w in high:
        print(f"\n  [HIGH LEAKAGE] Feature: {w.get('feature')}")
        print(f"    Method : {w.get('detection_method')}")
        print(f"    Evidence: {w.get('evidence')}")

    for w in others:
        print(f"\n  [{str(w.get('severity', 'info')).upper()}] {w.get('feature')}: {w.get('evidence')}")

    if critical:
        print(
            "\n  " + "!" * 68 + "\n"
            "  CRITICAL TARGET LEAKAGE DETECTED!\n"
            "  The feature(s) above deterministically or near-deterministically\n"
            "  predict the target. Results obtained with them included do NOT\n"
            "  reflect generalisable predictive performance.\n"
            "  To exclude them, add their names to `exclude_columns` in config.yaml.\n"
            + "  " + "!" * 68
        )
        return True
    return False


# ---------------------------------------------------------------------------
# Subcommand: profile
# ---------------------------------------------------------------------------

def cmd_profile(cfg, args) -> None:
    _print_banner("DATASET PROFILING")
    DataIngestionEngine = _import("carbon_automl.core.ingestion", "DataIngestionEngine")
    DataProfiler = _import("carbon_automl.core.data_profiler", "DataProfiler")
    LeakageDetector = _import("carbon_automl.leakage.detector", "LeakageDetector")

    if not all([DataIngestionEngine, DataProfiler]):
        logger.error("Required core modules unavailable for profiling.")
        sys.exit(1)

    engine = DataIngestionEngine()
    X, y, metadata = engine.load(
        str(_resolve_path(cfg.dataset_path)),
        cfg.target_column,
        cfg.exclude_columns,
    )
    print(f"  Dataset : {metadata['source_path']}")
    print(f"  Shape   : {metadata['shape']} (features={X.shape})")
    print(f"  SHA-256 : {metadata.get('file_hash', 'n/a')}")

    profile = DataProfiler().profile(X, y)
    task_type = cfg.task_type if cfg.task_type != "auto" else profile.get("task_type_suggestion", "classification")

    print(f"  Task    : {task_type}")
    print(f"  Memory  : {profile.get('memory_mb', 0):.4f} MB")
    print(f"  Dup rows: {profile.get('duplicate_rows', 0)}")

    print("\n  Feature types:")
    for col, ftype in profile.get("feature_types", {}).items():
        missing = profile.get("missing_values", {}).get(col, 0)
        print(f"    {col:<40} {ftype:<28} missing={missing}")

    if profile.get("recommendations"):
        print("\n  Recommendations:")
        for rec in profile["recommendations"]:
            print(f"    - {rec}")

    _print_banner("LEAKAGE ANALYSIS")
    warnings: list[dict[str, Any]] = []
    if LeakageDetector is not None:
        detector = LeakageDetector()
        warnings = detector.detect(
            X,
            y,
            feature_types=profile.get("feature_types", {}),
            task_type=task_type,
            single_feature_threshold=cfg.leakage_single_feature_threshold,
            correlation_threshold=cfg.leakage_correlation_threshold,
        )
        _print_leakage_warnings(warnings)
    else:
        logger.warning("LeakageDetector unavailable — skipping leakage analysis.")

    out_path_str = getattr(args, "output", None)
    if out_path_str:
        out = Path(out_path_str)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"metadata": metadata, "profile": profile, "leakage_warnings": warnings}
        out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"\n  Profile report saved -> {out}")


# ---------------------------------------------------------------------------
# Subcommand: validate
# ---------------------------------------------------------------------------

def cmd_validate(cfg, args) -> None:
    _print_banner("CONFIGURATION & DATASET VALIDATION")
    print(f"  Config path         : {args.config}")
    print(f"  Dataset path        : {cfg.dataset_path}")
    print(f"  Target column       : {cfg.target_column}")
    print(f"  Task type           : {cfg.task_type}")
    print(f"  Validation strategy : {cfg.validation_strategy}")
    print(f"  Carbon intensity    : {cfg.carbon_intensity_kg_per_kwh} kg CO2e/kWh")

    dataset_path = _resolve_path(cfg.dataset_path)
    if not dataset_path.exists():
        print(f"\n  [ERROR] Dataset not found: {dataset_path}")
        sys.exit(1)
    print(f"  [OK] Dataset exists : {dataset_path}")

    DataIngestionEngine = _import("carbon_automl.core.ingestion", "DataIngestionEngine")
    if DataIngestionEngine is not None:
        try:
            X, y, meta = DataIngestionEngine().load(
                str(dataset_path), cfg.target_column, cfg.exclude_columns
            )
            print(f"  [OK] Dataset loaded : shape={meta['shape']}, sha256={meta['file_hash'][:16]}...")
        except Exception as exc:
            print(f"  [ERROR] Dataset validation failed: {exc}")
            sys.exit(1)

    if getattr(args, "print_defaults", False):
        schema_cls = _import("carbon_automl.config.schema", "CarbonAutoMLConfig")
        if schema_cls is not None:
            defaults = {
                name: (
                    None
                    if field.is_required()
                    else field.get_default(call_default_factory=True)
                )
                for name, field in schema_cls.model_fields.items()
            }
            print("\nDefault config schema fields:")
            print(yaml.dump(defaults, sort_keys=False, default_flow_style=False))

    print("\n  ✓ Validation passed.")


# ---------------------------------------------------------------------------
# Full pipeline (train & benchmark)
# ---------------------------------------------------------------------------

def run_pipeline(
    cfg,
    *,
    benchmark_mode: bool = False,
    interactive_pause: bool = False,
    force_retrain: bool = False,
) -> dict[str, Any]:
    """Execute the full Carbon-Aware AutoML pipeline with zero test-set leakage."""
    intensity_source = getattr(
        cfg,
        "carbon_intensity_source",
        getattr(cfg, "_carbon_intensity_source_fallback", "user_configured"),
    )

    results: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark_mode": benchmark_mode,
        "reused_cached_model": False,
        "config": cfg.model_dump(),
        "phases": {},
    }

    # Graceful imports
    DataIngestionEngine = _import("carbon_automl.core.ingestion", "DataIngestionEngine")
    DataProfiler = _import("carbon_automl.core.data_profiler", "DataProfiler")
    LeakageDetector = _import("carbon_automl.leakage.detector", "LeakageDetector")
    UniversalPreprocessor = _import("carbon_automl.core.preprocessing", "UniversalPreprocessor")
    ValidationEngine = _import("carbon_automl.evaluation.validation", "ValidationEngine")
    MetricsCalculator = _import("carbon_automl.evaluation.metrics", "MetricsCalculator")
    get_model_zoo = _import("carbon_automl.evaluation.model_zoo", "get_model_zoo")
    HyperparameterOptimizer = _import("carbon_automl.evaluation.optimizer", "HyperparameterOptimizer")
    FeatureSelector = _import("carbon_automl.evaluation.feature_selection", "FeatureSelector")
    AblationAnalyzer = _import("carbon_automl.evaluation.ablation", "AblationAnalyzer")
    SizeAwareOptimizer = _import("carbon_automl.evaluation.size_aware", "SizeAwareOptimizer")
    EnergyMonitor = _import("carbon_automl.sustainability.energy_monitor", "EnergyMonitor")
    CarbonCalculator = _import("carbon_automl.sustainability.carbon_calculator", "CarbonCalculator")
    StatisticalAnalyzer = _import("carbon_automl.sustainability.statistical_analysis", "StatisticalAnalyzer")
    ParetoAnalyzer = _import("carbon_automl.sustainability.pareto", "ParetoAnalyzer")
    CarbonBudgetOptimizer = _import("carbon_automl.sustainability.carbon_budget", "CarbonBudgetOptimizer")
    ModelSelector = _import("carbon_automl.intelligence.model_selector", "ModelSelector")
    RetrainingManager = _import("carbon_automl.intelligence.retraining_manager", "RetrainingManager")
    CarbonPassport = _import("carbon_automl.artifacts.carbon_passport", "CarbonPassport")
    ModelRegistry = _import("carbon_automl.artifacts.model_registry", "ModelRegistry")
    HTMLReportGenerator = _import("carbon_automl.reporting.html_report", "HTMLReportGenerator")
    PDFReportGenerator = _import("carbon_automl.reporting.pdf_report", "PDFReportGenerator")
    Visualizer = _import("carbon_automl.reporting.visualizations", "Visualizer")

    if not DataIngestionEngine:
        raise RuntimeError("DataIngestionEngine is required to run the pipeline.")

    monitor = EnergyMonitor(carbon_intensity_kg_per_kwh=cfg.carbon_intensity_kg_per_kwh) if EnergyMonitor else None
    calc = CarbonCalculator(
        carbon_intensity_kg_per_kwh=cfg.carbon_intensity_kg_per_kwh,
        intensity_source=intensity_source,
    ) if CarbonCalculator else None

    def _record_phase(phase_name: str, m: Any) -> None:
        if m is None:
            return
        results["phases"][phase_name] = {
            "phase_name": m.phase_name,
            "duration_seconds": m.duration_seconds,
            "energy_kwh": m.energy_kwh,
            "energy_joules": (m.energy_kwh * 3_600_000.0) if m.energy_kwh is not None else None,
            "co2_kg": m.co2_kg,
            "measurement_method": m.measurement_method,
            "is_estimated": m.is_estimated,
            "uncertainty_pct": m.uncertainty_pct,
        }

    # ── Step 1: Data Ingestion & Integrity ────────────────────────────────
    _print_banner("Step 1 · Data Ingestion & Integrity")
    dataset_file = _resolve_path(cfg.dataset_path)
    if monitor:
        monitor.start("ingestion")
    X, y, metadata = DataIngestionEngine().load(
        str(dataset_file),
        cfg.target_column,
        cfg.exclude_columns,
    )
    if monitor:
        _record_phase("ingestion", monitor.stop())
    results["metadata"] = metadata
    results["dataset_fingerprint"] = metadata.get("integrity_fingerprint", {})
    print(f"  Loaded {X.shape[0]} rows × {X.shape[1]} features (SHA-256: {metadata['file_hash'][:12]}...)")

    # Extract group / timestamp columns safely without modifying raw file
    def _extract_auxiliary_column(col_name: str | None) -> pd.Series | None:
        if not col_name:
            return None
        if col_name in X.columns:
            return X[col_name].copy()
        raw_df = (
            pd.read_parquet(dataset_file, columns=[col_name])
            if dataset_file.suffix.lower() in {".parquet", ".pq"}
            else pd.read_csv(dataset_file, usecols=[col_name])
        )
        return raw_df[col_name].copy()

    groups_series = _extract_auxiliary_column(cfg.group_column)
    timestamp_series = _extract_auxiliary_column(cfg.timestamp_column)
    if cfg.group_column and cfg.group_column in X.columns:
        X = X.drop(columns=[cfg.group_column])
    if cfg.validation_strategy == "time_series" and cfg.timestamp_column and cfg.timestamp_column in X.columns:
        X = X.drop(columns=[cfg.timestamp_column])

    # ── Step 2: Profiling & Size-Aware Strategy ───────────────────────────
    _print_banner("Step 2 · Dataset Profiling")
    profile: dict[str, Any] = {}
    if DataProfiler:
        profile = DataProfiler().profile(X, y)
        results["profile"] = profile
    task_type = (
        profile.get("task_type_suggestion", "classification")
        if cfg.task_type == "auto"
        else cfg.task_type
    )
    results["task_type"] = task_type
    print(f"  Task type: {task_type}")

    size_strategy: dict[str, Any] | None = None
    if SizeAwareOptimizer:
        try:
            size_strategy = SizeAwareOptimizer().recommend_strategy(
                X=X,
                y=y,
                task_type=task_type,
                feature_types=profile.get("feature_types", {}),
                enable_xgboost=cfg.enable_xgboost,
                random_state=cfg.random_seed,
            )
            results["size_aware_strategy"] = size_strategy
            print(f"  Size-aware tier: {size_strategy.get('complexity_profile', size_strategy.get('complexity', {})).get('tier', 'n/a')}")
        except Exception as exc:
            logger.warning("SizeAwareOptimizer skipped: %s", exc)

    # ── Step 3: Leakage Detection ─────────────────────────────────────────
    _print_banner("Step 3 · Leakage Detection")
    leakage_warnings: list[dict[str, Any]] = []
    if LeakageDetector:
        leakage_warnings = LeakageDetector().detect(
            X,
            y,
            feature_types=profile.get("feature_types", {}),
            task_type=task_type,
            single_feature_threshold=cfg.leakage_single_feature_threshold,
            correlation_threshold=cfg.leakage_correlation_threshold,
        )
        results["leakage_warnings"] = leakage_warnings
        has_critical = _print_leakage_warnings(leakage_warnings)
        if has_critical and interactive_pause and sys.stdin.isatty():
            input("\n  Press Enter to acknowledge critical leakage warning and continue (or Ctrl+C to abort)... ")
    else:
        logger.warning("LeakageDetector unavailable — skipping leakage check.")

    # ── Step 3b: Retraining Cache Check ───────────────────────────────────
    _print_banner("Step 3b · Retraining Cache Check")
    models_dir = _resolve_path(cfg.models_dir)
    out_dir = _resolve_path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    retrainer = RetrainingManager(str(models_dir)) if RetrainingManager else None
    registry = ModelRegistry(str(models_dir)) if ModelRegistry else None

    exp_hash = ""
    if retrainer:
        exp_hash = retrainer.compute_experiment_hash(
            dataset_hash=metadata["file_hash"],
            config=cfg.model_dump(),
            feature_schema=metadata.get("dtypes", {}),
        )
        results["experiment_hash"] = exp_hash
        print(f"  Experiment hash: {exp_hash[:16]}...")

        if not benchmark_mode and not force_retrain and registry:
            for reg_entry in reversed(registry.list_models()):
                if reg_entry.get("experiment_hash") != exp_hash:
                    continue
                cached = retrainer.check_cache(exp_hash, reg_entry["model_name"])
                if cached is not None:
                    results["reused_cached_model"] = True
                    results["model_id"] = reg_entry.get("registry_id")
                    reg_metrics = dict(reg_entry.get("metrics", {}))
                    primary_m = "macro_f1" if task_type == "classification" else "r2"
                    results["best_model"] = {
                        "name": reg_entry["model_name"],
                        **reg_metrics,
                        "mean_score": reg_metrics.get("cv_score", reg_metrics.get(primary_m)),
                        "cached": True,
                        "artifact_path": cached.get("artifact_path"),
                        "justification": (
                            f"Reused valid cached model '{reg_entry['model_name']}' "
                            f"(experiment_hash={exp_hash[:12]}...) without retraining."
                        ),
                    }
                    results["holdout_test_metrics"] = reg_metrics
                    pp_path_str = reg_entry.get("carbon_passport_path", "")
                    if pp_path_str and Path(pp_path_str).exists():
                        try:
                            results["carbon_passport"] = json.loads(
                                Path(pp_path_str).read_text(encoding="utf-8")
                            )
                        except Exception:
                            pass
                    ts_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                    exp_file = out_dir / f"experiment_{ts_tag}.json"
                    exp_file.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
                    results["experiment_file"] = str(exp_file)
                    if HTMLReportGenerator:
                        try:
                            HTMLReportGenerator().generate(results, str(out_dir / f"report_{ts_tag}.html"))
                        except Exception as exc:
                            logger.warning("HTML report generation failed: %s", exc)
                    if PDFReportGenerator:
                        try:
                            pdf_path = PDFReportGenerator().generate(results, str(out_dir / f"report_{ts_tag}.pdf"))
                            results["pdf_report"] = pdf_path
                        except Exception as exc:
                            logger.warning("PDF report generation failed: %s", exc)
                    print(
                        f"  ✓ Cache hit: reusing model '{reg_entry['model_name']}' "
                        f"(registry_id={results['model_id']}) — skipping retraining."
                    )
                    return results

    # ── Step 4: Isolated Train/Holdout-Test Split ─────────────────────────
    _print_banner("Step 4 · Isolated Train / Holdout-Test Split")
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    indices = np.arange(len(X))
    if cfg.validation_strategy == "time_series" and timestamp_series is not None:
        ts_parsed = pd.to_datetime(timestamp_series, errors="coerce")
        if not ts_parsed.isna().all():
            order = np.argsort(ts_parsed.to_numpy(), kind="mergesort")
        else:
            order = np.argsort(timestamp_series.to_numpy(), kind="mergesort")
        split_idx = max(1, min(len(order) - 1, int(len(order) * 0.8)))
        train_idx = order[:split_idx]
        test_idx = order[split_idx:]
    elif cfg.group_column and groups_series is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=cfg.random_seed)
        train_idx, test_idx = next(gss.split(indices, y, groups=groups_series.to_numpy()))
    else:
        stratify_y = None
        if task_type == "classification":
            vc = y.value_counts()
            if len(vc) >= 2 and int(vc.min()) >= 2:
                stratify_y = y

        if ValidationEngine is not None and stratify_y is not None:
            train_idx, test_idx, _, _ = ValidationEngine().create_holdout_split(
                indices, y, test_size=0.2, random_state=cfg.random_seed, stratify=True
            )
        else:
            train_idx, test_idx = train_test_split(
                indices, test_size=0.2, random_state=cfg.random_seed, stratify=stratify_y
            )

    X_train_df = X.iloc[train_idx].copy()
    X_test_df = X.iloc[test_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()
    groups_train = groups_series.iloc[train_idx].to_numpy() if groups_series is not None else None
    print(f"  Train split: {len(X_train_df)} rows | Isolated Holdout Test split: {len(X_test_df)} rows")

    # ── Step 5: Preprocessing (FIT ONLY ON TRAINING SPLIT) ────────────────
    _print_banner("Step 5 · Preprocessing (Fit on Train Only)")
    preprocessor = None
    feature_names = list(X_train_df.columns)
    if UniversalPreprocessor:
        if monitor:
            monitor.start("preprocessing")
        preprocessor = UniversalPreprocessor()
        feature_types = profile.get("feature_types", {})
        X_train_proc = preprocessor.fit_transform(X_train_df, feature_types)
        X_test_proc = preprocessor.transform(X_test_df)
        feature_names = preprocessor.get_feature_names()
        if monitor:
            _record_phase("preprocessing", monitor.stop())
        prep_info: dict[str, Any] = {
            "feature_names_in": list(X_train_df.columns),
            "feature_names_out": feature_names,
            "train_shape": list(X_train_proc.shape),
            "test_shape": list(X_test_proc.shape),
        }
        if hasattr(preprocessor, "get_feature_mapping"):
            prep_info["feature_mapping"] = preprocessor.get_feature_mapping()
        if hasattr(preprocessor, "export_transformation_history"):
            prep_info["transformation_history"] = preprocessor.export_transformation_history()
        results["preprocessing"] = prep_info
        print(f"  Transformed shape: train={X_train_proc.shape}, test={X_test_proc.shape}")
    else:
        X_train_proc = X_train_df.to_numpy()
        X_test_proc = X_test_df.to_numpy()

    # Encode classification targets cleanly with LabelEncoder fitted on train only
    label_encoder = None
    if task_type == "classification":
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y_train_arr = label_encoder.fit_transform(y_train)
        y_test_arr = label_encoder.transform(y_test)
        class_names = [str(c) for c in label_encoder.classes_]
    else:
        y_train_arr = np.asarray(y_train, dtype=float)
        y_test_arr = np.asarray(y_test, dtype=float)
        class_names = []

    # ── Step 7: Cross-Validation across Model Zoo (Train Split Only) ──────
    _print_banner("Step 7 · Model Zoo Cross-Validation")
    primary_metric = "macro_f1" if task_type == "classification" else "r2"
    model_results: list[dict[str, Any]] = []
    model_fold_scores: dict[str, list[float]] = {}

    if not (ValidationEngine and MetricsCalculator and get_model_zoo):
        logger.error("Evaluation modules unavailable — cannot train models.")
        return results

    val_engine = ValidationEngine()
    metrics_calc = MetricsCalculator()
    if getattr(cfg, "adaptive_optimization", True) and size_strategy and size_strategy.get("model_zoo"):
        from sklearn.base import clone as _clone_est
        zoo = {}
        tier_name = size_strategy.get("complexity_profile", {}).get("tier", "light")
        for _mname, _mest in size_strategy["model_zoo"].items():
            est_copy = _clone_est(_mest)
            # For medium/larger datasets (>=5k rows), prevent unbounded tree depth in single-thread CV
            if len(X_train_proc) >= 5000 and _mname in ("random_forest", "extra_trees"):
                _p = est_copy.get_params()
                if "n_estimators" in _p:
                    est_copy.set_params(n_estimators=min(50, int(_p["n_estimators"])))
                if "max_depth" in _p and _p["max_depth"] is None:
                    est_copy.set_params(max_depth=12)
            zoo[_mname] = est_copy
    else:
        zoo = get_model_zoo(
            task_type=task_type,
            enable_xgboost=cfg.enable_xgboost,
            dataset_size=len(X_train_proc),
            n_features=X_train_proc.shape[1],
            random_state=cfg.random_seed,
        )

    n_folds = cfg.n_folds
    if groups_train is not None and cfg.validation_strategy in ("group_kfold", "stratified_group_kfold"):
        n_folds = max(2, min(n_folds, int(len(np.unique(groups_train)))))
    elif task_type == "classification":
        _, counts = np.unique(y_train_arr, return_counts=True)
        n_folds = max(2, min(n_folds, int(counts.min())))
    else:
        n_folds = max(2, min(n_folds, len(X_train_proc)))

    strategy = cfg.validation_strategy
    if strategy in ("group_kfold", "stratified_group_kfold") and groups_train is None:
        strategy = "stratified_kfold" if task_type == "classification" else "kfold"
    if task_type == "regression" and strategy == "stratified_kfold":
        strategy = "kfold"
    if task_type == "regression" and strategy == "stratified_group_kfold":
        strategy = "group_kfold"

    cv_splitter = val_engine.create_cv_splitter(
        strategy=strategy,
        n_folds=n_folds,
        random_state=cfg.random_seed,
        groups=groups_train,
    )

    class _GroupedSplitterAdapter:
        """Bind groups for HyperparameterOptimizer.optimize when using GroupKFold."""
        def __init__(self, inner, groups):
            self._inner = inner
            self._groups = groups

        def split(self, X_in, y_in=None, groups=None):
            return self._inner.split(X_in, y_in, groups=self._groups if groups is None else groups)

    hpo_cv_splitter = (
        _GroupedSplitterAdapter(cv_splitter, groups_train)
        if groups_train is not None and strategy in ("group_kfold", "stratified_group_kfold")
        else cv_splitter
    )

    n_reps = cfg.measurement_repetitions if benchmark_mode else 1

    for name, estimator in zoo.items():
        rep_energies: list[float] = []
        rep_durations: list[float] = []
        cv_out: dict[str, Any] = {}

        for rep in range(n_reps):
            if monitor:
                monitor.start(f"train_{name}_rep{rep + 1}")
            cv_out = val_engine.cross_validate_model(
                estimator,
                X_train_proc,
                y_train_arr,
                cv_splitter,
                metrics_calc,
                task_type=task_type,
                groups=groups_train,
            )
            if monitor:
                m = monitor.stop()
                if m.energy_kwh is not None:
                    rep_energies.append(m.energy_kwh)
                rep_durations.append(m.duration_seconds)

        mean_metrics = cv_out.get("mean", {})
        fold_primary = [
            float(fs.get(primary_metric, 0.0))
            for fs in cv_out.get("fold_scores", [])
        ]
        model_fold_scores[name] = fold_primary

        mean_energy_kwh = float(np.mean(rep_energies)) if rep_energies else 0.0
        mean_duration_s = float(np.mean(rep_durations)) if rep_durations else 0.0
        co2_kg = calc.compute_co2(mean_energy_kwh) if calc else mean_energy_kwh * cfg.carbon_intensity_kg_per_kwh

        entry: dict[str, Any] = {
            "name": name,
            "model_name": name,
            **mean_metrics,
            "mean_score": mean_metrics.get(primary_metric, 0.0),
            "std_score": cv_out.get("std", {}).get(primary_metric, 0.0),
            "confidence_intervals": cv_out.get("confidence_intervals", {}),
            "training_duration_seconds": mean_duration_s,
            "training_energy_kwh": mean_energy_kwh,
            "training_energy_joules": mean_energy_kwh * 3_600_000.0,
            "training_co2_kg": co2_kg,
            "inference_energy_kwh": 0.0,
            "memory_mb": 0.0,
            "energy_measurements_kwh": rep_energies,
            "source": "cv_baseline",
        }
        model_results.append(entry)
        print(
            f"  {name:<22} {primary_metric}={entry['mean_score']:.4f} "
            f"± {entry['std_score']:.4f} | energy={mean_energy_kwh:.3e} kWh"
        )

    # ── Step 8: Hyperparameter Optimisation (CV on Train Split Only) ──────
    _print_banner("Step 8 · Hyperparameter Optimisation (Fair Trial Budget)")
    best_hpo_params: dict[str, dict[str, Any]] = {}
    if HyperparameterOptimizer:
        hpo = HyperparameterOptimizer()
        tunable = [n for n in zoo.keys() if n not in ("dummy", "linear_regression")]
        # Use controlled tuning subset on large datasets (>=5000 rows) per Phase 5, then evaluate on full CV
        X_hpo, y_hpo, hpo_cv = X_train_proc, y_train_arr, hpo_cv_splitter
        if getattr(cfg, "adaptive_optimization", True) and len(X_train_proc) >= 5000 and groups_train is None:
            rng_sub = np.random.default_rng(cfg.random_seed)
            sub_n = min(3000, len(X_train_proc))
            sub_idx = rng_sub.choice(len(X_train_proc), size=sub_n, replace=False)
            X_hpo = X_train_proc[sub_idx]
            y_hpo = y_train_arr[sub_idx]
            hpo_strat = "stratified_kfold" if task_type == "classification" else "kfold"
            hpo_cv = val_engine.create_cv_splitter(strategy=hpo_strat, n_folds=min(3, n_folds), random_state=cfg.random_seed)
            print(f"  Size-aware HPO: using controlled tuning subset ({sub_n}/{len(X_train_proc)} rows, {min(3, n_folds)}-fold CV).")
        per_model_trials = max(2, cfg.trial_budget // max(1, len(tunable)))
        print(f"  Allocating {per_model_trials} trials to each of {len(tunable)} tunable models.")

        for name in tunable:
            if monitor:
                monitor.start(f"hpo_{name}")
            try:
                hpo_out = hpo.optimize(
                    model_name=name,
                    X_train=X_hpo,
                    y_train=y_hpo,
                    cv_splitter=hpo_cv,
                    task_type=task_type,
                    n_trials=per_model_trials,
                    random_state=cfg.random_seed,
                )
            except Exception as exc:
                logger.warning("HPO skipped for %s: %s", name, exc)
                if monitor:
                    monitor.stop()
                continue
            m_hpo = monitor.stop() if monitor else None
            if m_hpo:
                _record_phase(f"hpo_{name}", m_hpo)

            params = hpo_out.get("best_params", {})
            best_hpo_params[name] = params
            # Evaluate tuned estimator on CV
            from sklearn.base import clone
            tuned_est = clone(zoo[name])
            valid_params = tuned_est.get_params()
            safe_params = {k: v for k, v in params.items() if k in valid_params}
            if safe_params:
                tuned_est.set_params(**safe_params)

            if monitor:
                monitor.start(f"cv_tuned_{name}")
            tuned_cv = val_engine.cross_validate_model(
                tuned_est, X_train_proc, y_train_arr, cv_splitter, metrics_calc, task_type=task_type, groups=groups_train
            )
            m_tuned = monitor.stop() if monitor else None
            tuned_energy_kwh = (m_tuned.energy_kwh or 0.0) if m_tuned else 0.0
            tuned_dur = m_tuned.duration_seconds if m_tuned else 0.0
            tuned_co2 = calc.compute_co2(tuned_energy_kwh) if calc else tuned_energy_kwh * cfg.carbon_intensity_kg_per_kwh

            tuned_name = f"{name}_tuned"
            tuned_means = tuned_cv.get("mean", {})
            model_fold_scores[tuned_name] = [
                float(fs.get(primary_metric, 0.0)) for fs in tuned_cv.get("fold_scores", [])
            ]
            tuned_entry = {
                "name": tuned_name,
                "model_name": tuned_name,
                "base_model_name": name,
                "hyperparameters": safe_params,
                **tuned_means,
                "mean_score": tuned_means.get(primary_metric, 0.0),
                "std_score": tuned_cv.get("std", {}).get(primary_metric, 0.0),
                "confidence_intervals": tuned_cv.get("confidence_intervals", {}),
                "training_duration_seconds": tuned_dur,
                "training_energy_kwh": tuned_energy_kwh,
                "training_energy_joules": tuned_energy_kwh * 3_600_000.0,
                "training_co2_kg": tuned_co2,
                "inference_energy_kwh": 0.0,
                "memory_mb": 0.0,
                "source": "hpo_tuned",
            }
            model_results.append(tuned_entry)
            print(f"  {tuned_name:<22} {primary_metric}={tuned_entry['mean_score']:.4f}")
    else:
        logger.warning("HyperparameterOptimizer unavailable — skipping HPO.")

    results["models"] = model_results

    # ── Step 9: Statistical, Pareto, Budget & Model Selection ─────────────
    _print_banner("Step 9 · Statistical Analysis, Pareto & Model Selection")
    stats_results: dict[str, Any] = {}
    if StatisticalAnalyzer and len(model_fold_scores) >= 2:
        stats_results = StatisticalAnalyzer().compare_models_statistically(model_fold_scores)
        results["statistics"] = stats_results
        print(f"  Pairwise comparisons: {stats_results.get('n_comparisons', 0)}")

    baseline_name = "random_forest" if any(m["name"] == "random_forest" for m in model_results) else model_results[0]["name"]
    baseline_entry = next((m for m in model_results if m["name"] == baseline_name), model_results[0])
    baseline_score = float(baseline_entry.get("mean_score", 0.0))
    quality_min = baseline_score - cfg.quality_tolerance

    if ParetoAnalyzer and model_results:
        pa = ParetoAnalyzer()
        pareto_results = pa.compute_pareto_frontier(
            model_results,
            objectives=[primary_metric, "training_energy_kwh"],
            minimize=[False, True],
        )
        results["pareto_frontier"] = pareto_results
        try:
            results["efficiency_metrics"] = pa.compute_efficiency_metrics(
                model_results,
                baseline_metrics=baseline_entry,
                quality_metric=primary_metric,
                energy_metric="training_energy_kwh",
                carbon_metric="training_co2_kg",
            )
        except Exception as exc:
            logger.warning("Efficiency metrics computation skipped: %s", exc)
        print(f"  Pareto-optimal models: {[m['name'] for m in pareto_results]}")

    if CarbonBudgetOptimizer and cfg.carbon_budget_kg is not None:
        budget_res = CarbonBudgetOptimizer(cfg.carbon_budget_kg, calc).find_models_within_budget(
            model_results, quality_min=quality_min, quality_metric=primary_metric
        )
        results["carbon_budget"] = budget_res
        if budget_res.get("conflict_explanation"):
            print(f"  [BUDGET CONFLICT] {budget_res['conflict_explanation']}")

    selection: dict[str, Any] = {}
    if ModelSelector and model_results:
        selection = ModelSelector().select(
            model_results=model_results,
            baseline_name=baseline_name,
            quality_metric=primary_metric,
            quality_min=quality_min,
            energy_metric="training_energy_kwh",
            task_type=task_type,
            critical_class_recall_constraints=getattr(cfg, "critical_class_recall_constraints", None),
        )
        results["selection"] = selection
        rec_name = selection["recommended"]["name"]
        rec_metrics = selection["recommended"].get("metrics", {})
        results["best_model"] = {"name": rec_name, **rec_metrics, "justification": selection["recommended"]["justification"]}
        print(f"  Recommended : {rec_name}")
        print(f"  Justification: {selection['recommended']['justification']}")
    else:
        best_fallback = max(model_results, key=lambda m: m["mean_score"])
        results["best_model"] = best_fallback
        rec_name = best_fallback["name"]

    # ── Step 10: Final Fit on Full Train Split & Single Holdout Test Eval ──
    _print_banner("Step 10 · Final Fit & Isolated Holdout Test Evaluation")
    from sklearn.base import clone
    base_key = results["best_model"].get("base_model_name", rec_name.replace("_tuned", ""))
    final_est = clone(zoo.get(base_key, list(zoo.values())[0]))
    hp = results["best_model"].get("hyperparameters", {})
    if hp:
        final_est.set_params(**{k: v for k, v in hp.items() if k in final_est.get_params()})

    if monitor:
        monitor.start("final_training")
    final_est.fit(X_train_proc, y_train_arr)
    m_final_train = monitor.stop() if monitor else None
    if m_final_train:
        _record_phase("final_training", m_final_train)

    y_test_pred = final_est.predict(X_test_proc)
    if task_type == "classification":
        y_test_proba = final_est.predict_proba(X_test_proc) if hasattr(final_est, "predict_proba") else None
        test_metrics = metrics_calc.compute_classification_metrics(y_test_arr, y_test_pred, y_proba=y_test_proba)
    else:
        test_metrics = metrics_calc.compute_regression_metrics(y_test_arr, y_test_pred)
    results["holdout_test_metrics"] = test_metrics
    print(f"  Holdout test {primary_metric}: {test_metrics.get(primary_metric, 0.0):.4f}")

    # Inference energy estimate
    infer_metrics: dict[str, Any] = {}
    if calc:
        infer_metrics = calc.compute_inference_emissions(
            final_est, X_test_proc, n_predictions=1000, repetitions=cfg.measurement_repetitions
        )
        results["inference_measurements"] = infer_metrics

    # Carbon totals across session
    if monitor:
        session_sum = monitor.session_summary()
        total_energy_kwh = sum(
            (v.get("total_energy_kwh") or 0.0) for v in session_sum.values()
        )
        total_co2_kg = calc.compute_co2(total_energy_kwh) if calc else total_energy_kwh * cfg.carbon_intensity_kg_per_kwh
        results["carbon"] = {
            "total_energy_kwh": total_energy_kwh,
            "total_energy_joules": total_energy_kwh * 3_600_000.0,
            "total_co2_kg": total_co2_kg,
            "training_co2_kg": results["best_model"].get("training_co2_kg", 0.0),
            "inference_co2_kg": infer_metrics.get("co2_per_1k_predictions_kg") or 0.0,
            "carbon_intensity_kg_per_kwh": cfg.carbon_intensity_kg_per_kwh,
            "carbon_intensity_source": intensity_source,
            "session_summary": session_sum,
        }

    # ── Step 11: Feature Selection & Ablation Analysis ────────────────────
    _print_banner("Step 11 · Feature Selection & Ablation Analysis")
    if FeatureSelector:
        fs = FeatureSelector()
        mi_res = fs.mutual_information_selection(
            X_train_proc, y_train_arr, feature_names=feature_names, task_type=task_type
        )
        results["feature_selection"] = mi_res
        print(f"  Top MI features: {mi_res.get('ranking', [])[:5]}")

    if AblationAnalyzer:
        try:
            ablation = AblationAnalyzer()
            is_large_tr = len(X_train_proc) >= 5000
            abl_res = ablation.run_ablation(
                X_train_proc, y_train_arr, X_test_proc, y_test_arr,
                model_name=base_key, task_type=task_type, random_state=cfg.random_seed,
                feature_names=feature_names,
                include_leave_one_out=(not is_large_tr),
                n_hpo_trials=(3 if is_large_tr else 10),
            )
            results["ablation"] = abl_res
            print("  Ablation analysis complete.")
        except Exception as exc:
            logger.warning("AblationAnalyzer skipped: %s", exc)

    # ── Step 12: Artifacts, Carbon Passport, Registry & HTML Report ──────
    _print_banner("Step 12 · Saving Artifacts, Carbon Passport & Report")
    artifact_payload = {
        "model": final_est,
        "preprocessor": preprocessor,
        "feature_names": list(X.columns),
        "processed_feature_names": feature_names,
        "label_encoder": label_encoder,
        "task_type": task_type,
    }
    artifact_path = ""
    if retrainer:
        artifact_path = retrainer.save_model(
            artifact_payload,
            model_name=rec_name,
            experiment_hash=exp_hash,
            metadata={"metrics": test_metrics, "config": cfg.model_dump()},
        )
        print(f"  Model saved -> {artifact_path}")

    ts_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    passport_json_path = ""
    if CarbonPassport:
        cp = CarbonPassport()
        known_lims = [
            f"Grid carbon intensity ({cfg.carbon_intensity_kg_per_kwh} kg CO2e/kWh, {intensity_source}) is static.",
            "Sub-second CPU energy measurements carry ±50–100% uncertainty from background noise.",
        ]
        if any(w.get("severity") == "critical" for w in leakage_warnings):
            known_lims.append("Critical target leakage warning(s) were active during this run.")

        passport_doc = cp.generate(
            model_name=rec_name,
            dataset_fingerprint=results["dataset_fingerprint"],
            feature_schema=metadata.get("dtypes", {}),
            training_config={"model_type": base_key, **hp},
            hardware_info={"cpu_count": os.cpu_count() or 1, "os": platform.platform()},
            training_measurements={
                "training_duration_seconds": results["best_model"].get("training_duration_seconds", 0.0),
                "training_energy_kwh": results["best_model"].get("training_energy_kwh", 0.0),
                "training_co2_kg": results["best_model"].get("training_co2_kg", 0.0),
                "is_estimated": True,
                "carbon_intensity_kg_per_kwh": cfg.carbon_intensity_kg_per_kwh,
                "carbon_intensity_source": intensity_source,
            },
            inference_measurements={
                "inference_energy_per_1k_predictions_kwh": (
                    (infer_metrics.get("energy_per_prediction_joules") or 0.0) * 1000.0 / 3_600_000.0
                ),
                "inference_latency_ms_p50": (infer_metrics.get("mean_duration_seconds") or 0.0) * 1000.0,
                "inference_latency_ms_p99": None,
            },
            predictive_metrics={**test_metrics, "cv_mean": results["best_model"].get("mean_score")},
            validation_methodology=f"{n_folds}-fold {strategy} on 80% train split; evaluated once on 20% holdout test split.",
            known_limitations=known_lims,
        )
        passport_file = out_dir / f"carbon_passport_{rec_name}_{ts_tag}.json"
        passport_json_path = cp.export_json(passport_doc, str(passport_file))
        cp.export_html(passport_doc, str(out_dir / f"carbon_passport_{rec_name}_{ts_tag}.html"))
        results["carbon_passport"] = passport_doc
        print(f"  Carbon Passport -> {passport_json_path}")

    if registry and artifact_path:
        reg_id = registry.register(
            model_name=rec_name,
            artifact_path=artifact_path,
            metrics={**{k: v for k, v in test_metrics.items() if isinstance(v, (int, float))}, "cv_score": results["best_model"].get("mean_score")},
            carbon_passport_path=passport_json_path,
            experiment_hash=exp_hash,
        )
        results["model_id"] = reg_id
        print(f"  Registered model_id: {reg_id}")

    exp_file = out_dir / f"experiment_{ts_tag}.json"
    exp_file.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    results["experiment_file"] = str(exp_file)
    print(f"  Experiment JSON -> {exp_file}")

    if HTMLReportGenerator:
        try:
            report_file = out_dir / f"report_{ts_tag}.html"
            HTMLReportGenerator().generate(results, str(report_file))
            results["html_report"] = str(report_file)
            print(f"  HTML report     -> {report_file}")
        except Exception as exc:
            logger.warning("HTML report generation failed: %s", exc)

    if PDFReportGenerator:
        try:
            pdf_file = out_dir / f"report_{ts_tag}.pdf"
            PDFReportGenerator().generate(results, str(pdf_file))
            results["pdf_report"] = str(pdf_file)
            print(f"  PDF report      -> {pdf_file}")
        except Exception as exc:
            logger.warning("PDF report generation failed: %s", exc)

    if Visualizer:
        try:
            charts_dir = out_dir / f"charts_{ts_tag}"
            charts_dir.mkdir(parents=True, exist_ok=True)
            vis = Visualizer(charts_dir)
            vis.plot_model_comparison(
                model_results, [primary_metric, "training_energy_kwh"], str(charts_dir / "model_comparison.png")
            )
            vis.plot_energy_by_phase(results.get("phases", {}), str(charts_dir / "energy_by_phase.png"))
            vis.plot_pareto_frontier(
                model_results, primary_metric, "training_energy_kwh", str(charts_dir / "pareto_frontier.png")
            )
            if task_type == "classification" and "confusion_matrix" in test_metrics:
                vis.plot_confusion_matrix(
                    test_metrics["confusion_matrix"], class_names, str(charts_dir / "confusion_matrix.png")
                )
            if "feature_selection" in results:
                vis.plot_feature_importance(results["feature_selection"], str(charts_dir / "feature_importance.png"))
            if "ablation" in results:
                vis.plot_ablation_results(results["ablation"], str(charts_dir / "ablation_results.png"))
            results["charts_dir"] = str(charts_dir)
            print(f"  Charts          -> {charts_dir}")
        except Exception as exc:
            logger.warning("Visualizer charts skipped: %s", exc)

    # Refresh experiment JSON with report & chart paths
    exp_file.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    return results


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_train(cfg, args) -> None:
    results = run_pipeline(
        cfg,
        benchmark_mode=False,
        force_retrain=getattr(args, "force_retrain", False),
    )
    _print_banner("TRAINING SUMMARY")
    bm = results.get("best_model", {})
    print(f"  Recommended model : {bm.get('name', 'n/a')}")
    print(f"  CV Score          : {bm.get('mean_score', 'n/a')}")
    carbon = results.get("carbon", {})
    print(f"  Total CO2e        : {carbon.get('total_co2_kg', 'n/a')} kg")
    print(f"  Model ID          : {results.get('model_id', 'n/a')}")


def cmd_benchmark(cfg, args) -> None:
    _print_banner("BENCHMARK MODE (Repeated Measurements)")
    results = run_pipeline(cfg, benchmark_mode=True)
    stats = results.get("statistics", {})
    sig_diffs = stats.get("significant_differences", [])
    print(f"\n  Significant pairwise differences (Bonferroni-corrected): {len(sig_diffs)}")
    for diff in sig_diffs:
        print(
            f"    {diff['model_a']} vs {diff['model_b']}: "
            f"p_bonf={diff['bonferroni_p_value']:.4f}, r={diff['effect_size_r']:.3f}"
        )


def cmd_report(cfg, args) -> None:
    _print_banner("HTML REPORT GENERATION")
    HTMLReportGenerator = _import("carbon_automl.reporting.html_report", "HTMLReportGenerator")
    if not HTMLReportGenerator:
        logger.error("HTMLReportGenerator unavailable.")
        sys.exit(1)

    exp_path = Path(args.experiment)
    if not exp_path.exists():
        print(f"  [ERROR] Experiment file not found: {exp_path}")
        sys.exit(1)

    results = json.loads(exp_path.read_text(encoding="utf-8"))
    out_target = Path(args.output) if args.output else _resolve_path(cfg.output_dir) / f"{exp_path.stem}_report.html"
    if out_target.is_dir():
        out_target = out_target / f"{exp_path.stem}_report.html"

    generated = HTMLReportGenerator().generate(results, str(out_target))
    print(f"  HTML report saved -> {generated}")


def cmd_predict(cfg, args) -> None:
    _print_banner("PREDICTION")
    ModelRegistry = _import("carbon_automl.artifacts.model_registry", "ModelRegistry")
    if not ModelRegistry:
        logger.error("ModelRegistry unavailable.")
        sys.exit(1)

    registry = ModelRegistry(str(_resolve_path(cfg.models_dir)))
    try:
        model_meta = registry.get_model(args.model_id)
    except KeyError:
        print(f"  [ERROR] Model ID not found in registry: {args.model_id}")
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"  [ERROR] Input CSV not found: {input_path}")
        sys.exit(1)

    artifact_path = Path(model_meta["artifact_path"])
    if not artifact_path.exists():
        print(f"  [ERROR] Model artifact missing: {artifact_path}")
        sys.exit(1)

    df = pd.read_csv(input_path)
    artifact = joblib.load(artifact_path)

    if isinstance(artifact, dict) and "model" in artifact:
        preprocessor = artifact.get("preprocessor")
        feature_names = artifact.get("feature_names")
        X_df = df[feature_names] if feature_names else df
        X_in = preprocessor.transform(X_df) if preprocessor is not None else X_df
        raw_preds = artifact["model"].predict(X_in)
        le = artifact.get("label_encoder")
        preds = le.inverse_transform(raw_preds) if le is not None else raw_preds
    else:
        preds = artifact.predict(df)

    out_path = Path(args.output) if args.output else input_path.with_suffix(".predictions.csv")
    out_df = df.copy()
    out_df["prediction"] = preds
    out_df.to_csv(out_path, index=False)
    print(f"  Predictions ({len(preds)} rows) saved -> {out_path}")


def cmd_compare(cfg, args) -> None:
    _print_banner("EXPERIMENT COMPARISON")
    path_a, path_b = Path(args.experiment_a), Path(args.experiment_b)
    for p in (path_a, path_b):
        if not p.exists():
            print(f"  [ERROR] Experiment file not found: {p}")
            sys.exit(1)

    exp_a = json.loads(path_a.read_text(encoding="utf-8"))
    exp_b = json.loads(path_b.read_text(encoding="utf-8"))

    bm_a = exp_a.get("best_model", {})
    bm_b = exp_b.get("best_model", {})
    co2_a = float(exp_a.get("carbon", {}).get("total_co2_kg", 0.0) or 0.0)
    co2_b = float(exp_b.get("carbon", {}).get("total_co2_kg", 0.0) or 0.0)

    print(f"\n  {'Metric':<28} {'Experiment A':<24} {'Experiment B':<24}")
    print("  " + "-" * 76)
    print(f"  {'model_name':<28} {str(bm_a.get('name', 'n/a')):<24} {str(bm_b.get('name', 'n/a')):<24}")
    score_a = float(bm_a.get("mean_score", 0.0) or 0.0)
    score_b = float(bm_b.get("mean_score", 0.0) or 0.0)
    print(f"  {'cv_score':<28} {score_a:<24.4f} {score_b:<24.4f}")
    print(f"  {'total_co2_kg':<28} {co2_a:<24.6g} {co2_b:<24.6g}")
    print(f"\n  Score delta (B - A): {score_b - score_a:+.4f}")
    print(f"  CO2 delta   (B - A): {co2_b - co2_a:+.6g} kg")


# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="carbon-automl",
        description="Carbon-Aware AutoML Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )

    sub = parser.add_subparsers(dest="subcommand")

    p_profile = sub.add_parser("profile", help="Profile dataset and report leakage warnings")
    p_profile.add_argument("--output", "-o", default=None, help="Save profile JSON to this path")

    p_train = sub.add_parser("train", help="Run full training pipeline")
    p_train.add_argument(
        "--force-retrain",
        action="store_true",
        help="Ignore cached models and force retraining from scratch",
    )
    sub.add_parser("benchmark", help="Repeated-CV training for statistical comparison")

    p_report = sub.add_parser("report", help="Generate HTML report from experiment results")
    p_report.add_argument("--experiment", "-e", required=True, help="Path to experiment JSON")
    p_report.add_argument("--output", "-o", default=None, help="Output HTML path")

    p_predict = sub.add_parser("predict", help="Run predictions with a saved model")
    p_predict.add_argument("--model-id", required=True, help="Model registry UUID")
    p_predict.add_argument("--input", "-i", required=True, help="Path to input CSV")
    p_predict.add_argument("--output", "-o", default=None, help="Output predictions CSV path")

    p_compare = sub.add_parser("compare", help="Compare two experiment result files")
    p_compare.add_argument("experiment_a", help="Path to first experiment JSON")
    p_compare.add_argument("experiment_b", help="Path to second experiment JSON")

    p_validate = sub.add_parser("validate", help="Validate config + dataset without training")
    p_validate.add_argument(
        "--print-defaults",
        action="store_true",
        help="Print default config schema values",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.getLogger().setLevel(getattr(logging, args.log_level, logging.INFO))

    try:
        cfg = load_config(args.config)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("Config validation failed: %s", exc)
        sys.exit(1)

    dispatch = {
        None: cmd_train,
        "train": cmd_train,
        "benchmark": cmd_benchmark,
        "profile": cmd_profile,
        "report": cmd_report,
        "predict": cmd_predict,
        "compare": cmd_compare,
        "validate": cmd_validate,
    }
    handler = dispatch.get(args.subcommand)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    try:
        handler(cfg, args)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(0)
    except Exception as exc:
        logger.error("Pipeline error: %s", exc)
        if args.log_level == "DEBUG":
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
