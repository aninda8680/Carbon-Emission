# Carbon-Efficient Machine Learning for Sustainable Model Training — Phase 2 Upgrade Summary

Upgraded the project from dataset-specific scripts into a dataset-agnostic, reproducible, carbon-aware AutoML framework (`carbon_automl`), preserving `Phase_1/` and `child_health_westbengal_1000.csv` intact (pre-upgrade git snapshot at `7cdab80`, original SHA-256 `c68a778e13cc257f80f9edb8c44ea2b42eb9c30fcbbcbb9cce3d4c4bbb1d7ca8` verified unchanged).

---

## 1. Phase 0 Audit & Investigation of the 100% Accuracy

The original [Phase_1/](file:///d:/MYSELF/GRP/Carbon-Emission/Phase_1) results (`1.0000` accuracy for both baseline RandomForest and LightGBM+FS) were confirmed to be **deterministic target leakage**, not generalizable ML performance:
- **`Mid-Upper Arm Circumference (cm)` (MUAC)** separates all three `Malnutrition Status` classes with zero overlap (`Severe`: `[8.4, 11.4]`, `Moderate`: `[11.5, 12.4]`, `Normal`: `[12.5, 19.1]`), matching a 3-threshold WHO diagnostic rule (`1.0000` accuracy from MUAC alone, `97.8%` of total mutual information).
- **Critical flaws resolved**: hardcoded `'Malnutrition Status'` target and test-set leakage inside Optuna's objective (`Phase_1/optimized/Optimized.py`), unsafe `float64 → float32` downcasting without round-trip checks (`Phase_1/utils.py`), single holdout split without cross-validation or statistical tests, dimensionally incoherent `CEGI = accuracy / CO2`, and sub-milligram CodeCarbon jitter presented as savings (`Minimal Complexity` actually recorded `0.000017 kg` CO₂ vs baseline `0.000012 kg` in `Phase_1/results/`).

See the full Phase 0 report in [phase0_audit.md](file:///C:/Users/user/.gemini/antigravity/brain/8ea95e3b-6c5c-4901-a305-2c7a4ea0a461/phase0_audit.md).

---

## 2. Architecture & Implemented Modules

- **Configuration & Validation**: [`CarbonAutoMLConfig`](file:///d:/MYSELF/GRP/Carbon-Emission/carbon_automl/config/schema.py#L16) (`config.yaml`), [`SchemaValidator`](file:///d:/MYSELF/GRP/Carbon-Emission/carbon_automl/core/schema_validator.py#L23)
- **Data Integrity & Ingestion (Phases 1–2)**: [`DataIntegrityManager`](file:///d:/MYSELF/GRP/Carbon-Emission/carbon_automl/core/integrity_manager.py#L31) (SHA-256/MD5 fingerprinting), [`DataIngestionEngine`](file:///d:/MYSELF/GRP/Carbon-Emission/carbon_automl/core/ingestion.py#L27) (`.csv` and `.parquet`, copy-only exclusions), [`DataProfiler`](file:///d:/MYSELF/GRP/Carbon-Emission/carbon_automl/core/data_profiler.py#L32), [`UniversalPreprocessor`](file:///d:/MYSELF/GRP/Carbon-Emission/carbon_automl/core/preprocessing.py#L68) + [`safe_downcast`](file:///d:/MYSELF/GRP/Carbon-Emission/carbon_automl/core/preprocessing.py#L26) (lossless round-trip validation, training-only fit), [`LeakageDetector`](file:///d:/MYSELF/GRP/Carbon-Emission/carbon_automl/leakage/detector.py#L30) (flags without auto-deleting)
- **Rigorous Evaluation & Size-Aware Scaling (Phases 2, 5)**: [`get_model_zoo`](file:///d:/MYSELF/GRP/Carbon-Emission/carbon_automl/evaluation/model_zoo.py#L14), [`MetricsCalculator`](file:///d:/MYSELF/GRP/Carbon-Emission/carbon_automl/evaluation/metrics.py#L18) (macro-F1, balanced accuracy, per-class recall, ROC-AUC, Brier/log-loss/ECE calibration, guarded MAPE, bootstrap CIs, quality/recall gates), [`ValidationEngine`](file:///d:/MYSELF/GRP/Carbon-Emission/carbon_automl/evaluation/validation.py#L28) (`stratified_kfold`, `kfold`, `group_kfold`, `stratified_group_kfold`, `time_series`, `nested_cv`, isolated holdout), [`HyperparameterOptimizer`](file:///d:/MYSELF/GRP/Carbon-Emission/carbon_automl/evaluation/optimizer.py#L24) (Optuna TPE + `MedianPruner`, CV-only objective, fair budget), [`FeatureSelector`](file:///d:/MYSELF/GRP/Carbon-Emission/carbon_automl/evaluation/feature_selection.py), [`AblationAnalyzer`](file:///d:/MYSELF/GRP/Carbon-Emission/carbon_automl/evaluation/ablation.py#L39), [`SizeAwareOptimizer`](file:///d:/MYSELF/GRP/Carbon-Emission/carbon_automl/evaluation/size_aware.py)
- **Green AI & Carbon Accounting (Phase 3)**: [`EnergyMonitor`](file:///d:/MYSELF/GRP/Carbon-Emission/carbon_automl/sustainability/energy_monitor.py) (CodeCarbon per-phase + duration fallback, kWh & Joules, explicit ±50–100% uncertainty on sub-second runs), [`CarbonCalculator`](file:///d:/MYSELF/GRP/Carbon-Emission/carbon_automl/sustainability/carbon_calculator.py#L39) (configurable `0.7 kg CO₂e/kWh` default, inference & amortization), [`StatisticalAnalyzer`](file:///d:/MYSELF/GRP/Carbon-Emission/carbon_automl/sustainability/statistical_analysis.py#L46) (bootstrap CIs, Bonferroni-corrected Wilcoxon tests, `is_demonstrably_greener`), [`ParetoAnalyzer`](file:///d:/MYSELF/GRP/Carbon-Emission/carbon_automl/sustainability/pareto.py#L58) (replaces raw CEGI with quality retention, energy/carbon efficiency, and reduction % alongside raw values), [`CarbonBudgetOptimizer`](file:///d:/MYSELF/GRP/Carbon-Emission/carbon_automl/sustainability/carbon_budget.py#L52), [`CarbonScheduler`](file:///d:/MYSELF/GRP/Carbon-Emission/carbon_automl/sustainability/carbon_scheduler.py) (experimental; reports `unavailable` without live grid forecast API)
- **Model Intelligence, Artifacts & Reporting (Phases 4, 6, 8)**: [`ModelSelector`](file:///d:/MYSELF/GRP/Carbon-Emission/carbon_automl/intelligence/model_selector.py#L12) (enforces `quality_tolerance` and `critical_class_recall_constraints`), [`RetrainingManager`](file:///d:/MYSELF/GRP/Carbon-Emission/carbon_automl/intelligence/retraining_manager.py#L19) (hash-based cache reuse across file renames without overwriting artifacts), [`DriftDetector`](file:///d:/MYSELF/GRP/Carbon-Emission/carbon_automl/intelligence/drift_detector.py), [`ModelRegistry`](file:///d:/MYSELF/GRP/Carbon-Emission/carbon_automl/artifacts/model_registry.py#L18), [`CarbonPassport`](file:///d:/MYSELF/GRP/Carbon-Emission/carbon_automl/artifacts/carbon_passport.py#L39) (JSON & offline HTML), [`HTMLReportGenerator`](file:///d:/MYSELF/GRP/Carbon-Emission/carbon_automl/reporting/html_report.py#L19) (24 sections, offline CSS, `MEASURED`/`ESTIMATED`/`CALCULATED` badges), [`PDFReportGenerator`](file:///d:/MYSELF/GRP/Carbon-Emission/carbon_automl/reporting/pdf_report.py#L22), [`Visualizer`](file:///d:/MYSELF/GRP/Carbon-Emission/carbon_automl/reporting/visualizations.py)
- **Interfaces & Deployment (Phase 6)**:
  - CLI: [main.py](file:///d:/MYSELF/GRP/Carbon-Emission/main.py) (`python main.py --config config.yaml [profile|train|benchmark|report|predict|compare|validate]`)
  - FastAPI service: [carbon_automl/api/main.py](file:///d:/MYSELF/GRP/Carbon-Emission/carbon_automl/api/main.py) (`/predict`, `/benchmark`, `/models`, `/models/{model_id}`, `/carbon-report/{model_id}`, `/health`, returning explicit `is_real_time_hardware_measurement: false` disclaimer)
  - Streamlit 12-page dashboard: [carbon_automl/dashboard/app.py](file:///d:/MYSELF/GRP/Carbon-Emission/carbon_automl/dashboard/app.py)
  - Packaging & CI: [requirements.txt](file:///d:/MYSELF/GRP/Carbon-Emission/requirements.txt), [pyproject.toml](file:///d:/MYSELF/GRP/Carbon-Emission/pyproject.toml), [Dockerfile](file:///d:/MYSELF/GRP/Carbon-Emission/Dockerfile), [docker-compose.yml](file:///d:/MYSELF/GRP/Carbon-Emission/docker-compose.yml), [.github/workflows/ci.yml](file:///d:/MYSELF/GRP/Carbon-Emission/.github/workflows/ci.yml)

---

## 3. Experimental Findings (Real Data & Honest Reporting)

### A. Original Malnutrition Dataset: With vs. Without MUAC
| Condition | Leakage Warnings | Recommended Model | CV `macro_f1` | Holdout `macro_f1` | Holdout Balanced Acc / ROC-AUC | Pipeline Total CO₂e (est.) |
|---|---|---|---|---|---|---|
| **With MUAC** ([experiment_20261111_193047.json](file:///d:/MYSELF/GRP/Carbon-Emission/experiments/experiment_20261111_193047.json)) | 3 (2 critical, 1 high) | `decision_tree` | `1.0000 ± 0.0000` | `1.0000` | `1.0000` / `1.0000` | `0.000617 kg` |
| **Without MUAC** ([experiment_20261111_193208.json](file:///d:/MYSELF/GRP/Carbon-Emission/experiments/cross_dataset/malnutrition_no_muac/experiment_20261111_193208.json)) | 0 | `decision_tree` | `0.3150 ± 0.0213` | `0.3804` | `0.3809` / `0.5365` | `0.000532 kg` |

- **Failed Hypothesis & Scientific Finding**: Once MUAC is removed, predictive quality drops to near-chance (`ROC-AUC = 0.5365`, `dummy` accuracy `0.6550` vs `random_forest` `0.6262`). Controlled ablation confirmed that Mutual Information selection, PCA, HPO, and the full optimization pipeline all performed **worse** than the raw baseline (`0.3804` holdout `macro_f1`) on these 6 covariates. Consequently, this dataset cannot support clinical claims without external clinical validation and genuine anthropometric ground truth.

### B. Cross-Dataset Evaluation Across 5 Public Datasets ([cross_dataset_summary.json](file:///d:/MYSELF/GRP/Carbon-Emission/experiments/cross_dataset_summary.json))
| Dataset | Format & Shape | Task / Tier | Recommended Model | CV Score | Holdout Score | Total Pipeline Energy / CO₂e (est.) |
|---|---|---|---|---|---|---|
| Child Health West Bengal (no MUAC) | CSV `[1000, 8]` | Multi-class (`light`) | `decision_tree` | `0.3150` (`macro_f1`) | `0.3804` | `2,738.4 J` / `0.000532 kg` |
| UCI Breast Cancer Wisconsin | CSV `[569, 31]` | Imbalanced binary (`light`) | `logistic_regression_tuned` | `0.9787` (`macro_f1`) | `0.9716` (`AUC 0.9960`) | `3,048.7 J` / `0.000593 kg` |
| OpenML Titanic (`boat/body/name` excl.) | Parquet `[1309, 14]` | High-card + missing (`moderate`) | `random_forest` | `0.7850` (`macro_f1`) | `0.8015` (`AUC 0.9011`) | `4,675.2 J` / `0.000909 kg` |
| Diabetes Progression | Parquet `[442, 11]` | Regression (`moderate`) | `ridge` | `0.4808` (`R²`) | `0.4542` (`RMSE 53.78`) | `3,679.3 J` / `0.000715 kg` |
| California Housing | CSV `[20640, 9]` | Medium/large regression (`moderate`) | `lightgbm` | `0.8338` (`R²`) | `0.8386` (`RMSE 0.4599`) | `18,670.1 J` / `0.003630 kg` |

- On **California Housing (`20,640` rows)**, `lightgbm` achieved higher CV `R²` (`0.8338 ± 0.0059`) than `random_forest` (`0.7935 ± 0.0066`) at ~13.5× lower estimated training energy (`2.502e-05` vs `3.385e-04 kWh`). On small datasets (`Breast Cancer`, `Diabetes`), linear models (`logistic_regression_tuned`, `ridge`) beat tree ensembles on both accuracy/R² and compute.

### C. Phase 5 Synthetic Scaling Benchmark ([synthetic_scale_benchmark.json](file:///d:/MYSELF/GRP/Carbon-Emission/experiments/synthetic_scale_benchmark.json))
Strictly separated from real-world runs (`is_synthetic: true`, LightGBM, 10 `float32` features):
- `10,000` rows (`moderate`, `chunk=False`): `fit=0.987s`, `pred=0.341s`, `peak_mem=3.61 MB`, estimated `13.28 J`
- `100,000` rows (`heavy`, `chunk=False`): `fit=1.497s`, `pred=0.653s`, `peak_mem=8.63 MB`, estimated `21.51 J`
- `1,000,000` rows (`large_scale`, `chunk_size=50,000`): `fit=12.377s`, `pred=6.713s`, `peak_mem=61.07 MB`, estimated `190.90 J`

---

## 4. Verification Results & Remaining Limitations

- **Test Suite** (run after final edit via `python -m pytest tests/ -v --cov=carbon_automl --tb=short`): **`104 passed, 1 warning in 183.73s`** (`67%` total statement coverage across `carbon_automl`), covering CSV/Parquet ingestion, schema/dtype safety, original immutability, leakage detection (including MUAC), group & temporal CV, Optuna CV isolation, quality/recall gates, carbon accounting & budget conflicts, model registry/cache reuse, FastAPI endpoints, CLI workflows, reporting, and Streamlit `AppTest`.
- **Docker Compose**: `docker compose -f docker-compose.yml config --quiet` exited `0` (full container image build via `docker compose up --build` was not executed).
- **Remaining Limitations**: On Windows without hardware RAPL/GPU power sensors, sub-second CodeCarbon readings carry ±50–100% background-noise uncertainty (duration-based `10 W/core` estimates are used as fallback); `CarbonScheduler` remains unavailable until an external real-time grid carbon intensity API is configured; `XGBoost` is optional (`enable_xgboost: false` by default).

---

## 5. Quick Run Commands

```bash
# Default end-to-end pipeline
python main.py --config config.yaml

# Run full test suite
python -m pytest tests/ -v --cov=carbon_automl

# Start FastAPI / Streamlit Dashboard
uvicorn carbon_automl.api.main:app --host 0.0.0.0 --port 8000
streamlit run carbon_automl/dashboard/app.py
```

