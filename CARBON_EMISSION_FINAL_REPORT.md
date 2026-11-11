# Final Report: Carbon-Efficient Machine Learning for Sustainable Model Training

**Project Title:** Carbon-Efficient Machine Learning for Sustainable Model Training (`carbon_automl` v1.0.0)  
**Institution:** Department of Computer Science & Engineering, Adamas University, Kolkata  
**Original Benchmark Dataset:** Child Health West Bengal Malnutrition Dataset (`1,000` records, `7` features, `3` classes)  
**Status:** Phase 2 Upgrade Complete — Dataset-Agnostic, Leakage-Aware, Reproducible Carbon-Aware AutoML Framework (Verified via 104 Pytest Tests & 5-Dataset Evaluation)

---

## 1. Executive Summary

Machine learning workflows commonly optimize predictive accuracy through brute-force model complexity and exhaustive hyperparameter search while ignoring computational energy, carbon footprint, and statistical validity.

Building upon the preliminary [Phase_1/](file:///d:/MYSELF/GRP/Carbon-Emission/Phase_1) prototype, this project delivers a **dataset-agnostic, scientifically defensible, reproducible, carbon-aware AutoML framework** ([`carbon_automl`](file:///d:/MYSELF/GRP/Carbon-Emission/carbon_automl/__init__.py)) supporting both **classification** and **regression** over unseen **CSV** and **Parquet** tabular datasets without code changes.

Key engineering and scientific outcomes of the upgrade:
* **Root-Cause Audit of Preliminary 100% Accuracy:** Independent verification proved that the Phase 1 `100.0%` accuracy was caused by deterministic target leakage through `Mid-Upper Arm Circumference (cm)` (MUAC), a clinical diagnostic threshold variable, rather than generalizable predictive learning.
* **Non-Negotiable Data Integrity & Leakage Detection:** Cryptographic source fingerprinting (SHA-256/MD5), immutable raw data preservation, lossless round-trip `safe_downcast` validation, training-only preprocessing pipelines, and 7 automated leakage heuristics that flag suspicious predictors without silently deleting data.
* **Scientifically Rigorous Evaluation:** Isolated 20% holdout test sets, stratified/grouped/time-aware cross-validation, nested CV, fair-budget Optuna TPE optimization strictly inside training folds, quality & critical-class recall protection gates, controlled ablation studies, and complexity-aware scaling (`SizeAwareOptimizer`).
* **Transparent Carbon Accounting:** Per-phase energy tracking (kWh and Joules) via CodeCarbon with duration-based fallbacks, explicit sub-second measurement uncertainty (±50–100%), configurable regional carbon intensity (`0.7 kg CO₂e/kWh` default for India), Bonferroni-corrected Wilcoxon statistical validation, Pareto frontier selection replacing raw CEGI, Carbon Budget Optimization, Model Carbon Passports (JSON/HTML), and hash-based retraining avoidance.
* **Production Interfaces & Verification:** Unified CLI ([`main.py`](file:///d:/MYSELF/GRP/Carbon-Emission/main.py)), FastAPI REST service ([`carbon_automl/api/main.py`](file:///d:/MYSELF/GRP/Carbon-Emission/carbon_automl/api/main.py)), 12-page Streamlit dashboard ([`carbon_automl/dashboard/app.py`](file:///d:/MYSELF/GRP/Carbon-Emission/carbon_automl/dashboard/app.py)), offline 24-section HTML & PDF reporting, Docker Compose deployment, GitHub Actions CI, and **104 passing pytest tests** (`67%` statement coverage).

---

## 2. Phase 0 Audit: Investigation of Phase 1 Results & MUAC Target Leakage

Before modifying any code, the original Phase 1 implementation (`commit 7cdab80`) was audited end-to-end:

### 2.1 Critical Finding: MUAC Diagnostic Leakage
In [`child_health_westbengal_1000.csv`](file:///d:/MYSELF/GRP/Carbon-Emission/child_health_westbengal_1000.csv) (`1,000 × 8`, SHA-256 `c68a778e13cc257f80f9edb8c44ea2b42eb9c30fcbbcbb9cce3d4c4bbb1d7ca8`), the feature `Mid-Upper Arm Circumference (cm)` deterministically defines `Malnutrition Status` with **zero overlap across classes**:

| Malnutrition Status | Sample Count | Observed MUAC Range (cm) | WHO Diagnostic Rule Match |
|---|---:|:---:|:---:|
| `Severe` | 136 | `8.4 – 11.4` | `< 11.5 cm` (0 mismatches) |
| `Moderate` | 209 | `11.5 – 12.4` | `11.5 – 12.4 cm` (0 mismatches) |
| `Normal` | 655 | `12.5 – 19.1` | `>= 12.5 cm` (0 mismatches) |

* A single-feature `DecisionTree(max_depth=3)` on MUAC alone achieves **`1.0000` accuracy** (vs. `0.6550` majority-class baseline), and MUAC accounts for **`97.8%` of total mutual information** (`MI = 0.8767` out of `0.8962`).
* All other features (`Age`, `Gender`, `Height`, `Weight`, `Subdivision`, `Mother's Education`) yield near-chance individual predictive accuracy (`0.4850 – 0.6550`).

### 2.2 Methodological Flaws Resolved from Phase 1
1. **Hardcoded Dataset Logic:** `Phase_1/utils.py` hardcoded `'Malnutrition Status'` and `Phase_1/` scripts failed on arbitrary schemas or regression tasks.
2. **Test-Set Leakage in HPO:** `Phase_1/optimized/Optimized.py` evaluated Optuna trial models directly on `X_test_selected, y_test`.
3. **Unsafe Memory Downcasting:** `float64 → float32` conversion was applied without verifying round-trip precision preservation.
4. **No Cross-Validation or Uncertainty:** Single 80/20 split with sub-milligram CodeCarbon noise (`0.000010–0.000017 kg` CO₂) interpreted as environmental savings without statistical testing.
5. **Dimensionally Incoherent CEGI:** Dividing dimensionless accuracy by kg CO₂ (`Accuracy / CO₂`) mixed incomparable scales and masked quality degradation.

---

## 3. Upgraded Framework Architecture (`carbon_automl`)

```
Carbon-Emission/
├── main.py                                   # Unified CLI (validate, profile, train, benchmark, report, predict, compare)
├── config.yaml                               # Dataset-agnostic YAML configuration
├── carbon_automl/
│   ├── config/schema.py                      # Pydantic v2 config validation (CarbonAutoMLConfig)
│   ├── core/
│   │   ├── integrity_manager.py              # SHA-256/MD5 hashing & DataIntegrityError checks
│   │   ├── ingestion.py                      # Immutable CSV & Parquet loader
│   │   ├── data_profiler.py                  # Feature types, missingness, outliers, imbalance, sensitive cols
│   │   ├── schema_validator.py               # Config, DataFrame schema & dtype round-trip validation
│   │   └── preprocessing.py                  # Training-only ColumnTransformer, safe_downcast, mappings
│   ├── leakage/detector.py                   # 7+ leakage heuristics (flags without deleting)
│   ├── evaluation/
│   │   ├── model_zoo.py                      # Classification & regression model catalogs (optional XGBoost)
│   │   ├── validation.py                     # StratifiedKFold, KFold, GroupKFold, StratifiedGroupKFold, TimeSeriesSplit, Nested CV
│   │   ├── metrics.py                        # Multi-metric scoring, calibration (Brier/log-loss/ECE), bootstrap CIs, quality/recall gates
│   │   ├── optimizer.py                      # Optuna TPE + MedianPruner with CV-only objective & fair budgets
│   │   ├── feature_selection.py              # MI, permutation importance, model-based (SelectFromModel), RFE
│   │   ├── ablation.py                       # Controlled ablations (FS, PCA, HPO, early stopping, LOFO, complexity)
│   │   └── size_aware.py                     # Multi-factor complexity tiers, chunked inference & synthetic scaling
│   ├── sustainability/
│   │   ├── energy_monitor.py                 # CodeCarbon per-phase wrapper + duration fallback & uncertainty bounds
│   │   ├── carbon_calculator.py              # CO2e = kWh * intensity, inference energy & break-even amortization
│   │   ├── statistical_analysis.py           # Bootstrap CIs, Bonferroni-corrected Wilcoxon signed-rank tests
│   │   ├── pareto.py                         # Pareto frontier, quality-guarded recommendation, transparent efficiency metrics
│   │   ├── carbon_budget.py                  # Hard carbon budget feasibility & explicit conflict reporting
│   │   └── carbon_scheduler.py               # Experimental deferred-training advisor (unavailable without live grid API)
│   ├── intelligence/
│   │   ├── model_selector.py                 # Quality- and recall-guarded multi-objective selection
│   │   ├── retraining_manager.py             # Content-hash cache reuse & non-overwriting model versioning
│   │   └── drift_detector.py                 # PSI and mean-shift drift detection
│   ├── artifacts/
│   │   ├── model_registry.py                 # Atomic JSON model registry
│   │   └── carbon_passport.py                # Machine- & human-readable Carbon Passports (JSON & offline HTML)
│   ├── reporting/
│   │   ├── html_report.py                    # 24-section self-contained offline HTML report
│   │   ├── pdf_report.py                     # Multi-page PDF report via matplotlib PdfPages
│   │   └── visualizations.py                 # 150 DPI comparison, Pareto, phase energy, CM, FI & ablation charts
│   ├── api/main.py                           # FastAPI REST endpoints
│   └── dashboard/app.py                      # 12-page minimal Streamlit research dashboard
├── datasets/                                 # Reference datasets (read-only original copy)
├── experiments/                              # Experiment JSONs, HTML/PDF reports, Carbon Passports & charts
├── models/                                   # Serialized joblib artifacts, metadata & registry.json
├── tests/                                    # 104 pytest unit, integration, API, CLI & Streamlit smoke tests
├── Phase_1/                                  # Preserved original implementation (untouched)
├── Dockerfile & docker-compose.yml           # Containerized FastAPI + Streamlit services
└── .github/workflows/ci.yml                  # GitHub Actions CI (Python 3.10 & 3.11)
```

---

## 4. Green AI & Carbon Accounting Methodology

### 4.1 Operational Carbon Accounting
Operational emissions are calculated transparently using:

$$\text{CO}_2\text{e (kg)} = \text{Energy (kWh)} \times \text{Carbon Intensity (kg CO}_2\text{e/kWh)}$$

* **Default Assumption:** `0.7 kg CO₂e/kWh` (India national average, `user_configured`), surfaced explicitly in every report, Carbon Passport, and dashboard view alongside its boundary caveat (operational compute energy only; excludes embodied hardware manufacturing and data-center PUE cooling unless separately measured).
* **Per-Phase Accounting:** Measures data ingestion, preprocessing, cross-validation, HPO, final training, and inference (`1,000`-prediction amortized energy in Joules and kWh).
* **Honest Uncertainty Reporting:** On small datasets where training completes in `< 1s`, CodeCarbon CPU-load readings are heavily affected by OS background noise; [`EnergyMeasurement`](file:///d:/MYSELF/GRP/Carbon-Emission/carbon_automl/sustainability/energy_monitor.py) explicitly attaches `uncertainty_pct` (`±50–100%`) and `is_estimated` flags, and [`StatisticalAnalyzer.is_demonstrably_greener()`](file:///d:/MYSELF/GRP/Carbon-Emission/carbon_automl/sustainability/statistical_analysis.py) prohibits claiming environmental savings without `p < 0.05` statistical significance (`n >= 5`).

### 4.2 Replacing CEGI with Transparent Quality–Efficiency Metrics
Instead of an opaque `Accuracy / CO₂` ratio, [`ParetoAnalyzer.compute_efficiency_metrics()`](file:///d:/MYSELF/GRP/Carbon-Emission/carbon_automl/sustainability/pareto.py) reports:
1. Raw predictive quality (`macro_f1` or `R²`), raw energy (`kWh` & `J`), and raw `CO₂e (kg)`
2. **Energy Efficiency** (`quality / kWh`) and **Carbon Efficiency** (`quality / kg CO₂e`)
3. **Quality Retention (%)**, **Energy Reduction (%)**, and **Carbon Reduction (%)** relative to the baseline model
4. **Strict Quality Protection:** By default (`quality_tolerance: 0.005`), any model dropping more than 0.5 percentage points below the baseline — or violating any configured `critical_class_recall_constraints` — is disqualified regardless of energy savings.

---

## 5. Verified Experimental Results

### 5.1 Malnutrition Dataset: With vs. Without MUAC (Phase 1 vs. Upgraded Framework)

| Experiment Condition | Leakage Warnings | Recommended Model | CV `macro_f1` | Holdout `macro_f1` | Holdout Balanced Acc / ROC-AUC | Total Pipeline CO₂e (est.) |
|---|---|---|---:|---:|---:|---:|
| **Phase 1 Legacy** (`Baseline.py` / `Optimized.py`) | None detected | `LightGBM + FS` | N/A (single split) | `1.0000` (weighted F1) | N/A | `0.000010 kg` (training only) |
| **Upgraded — With MUAC** (`config.yaml`) | **3** (2 critical, 1 high) | `decision_tree` | `1.0000 ± 0.0000` | `1.0000` | `1.0000` / `1.0000` | `0.000617 kg` |
| **Upgraded — Without MUAC** (`exclude_columns`) | **0** | `decision_tree` | `0.3150 ± 0.0213` | `0.3804` | `0.3809` / `0.5365` | `0.000532 kg` |

* **Scientific Finding & Failed Hypothesis:** Excluding the leaky `MUAC` diagnostic rule causes predictive performance to collapse to near-random (`ROC-AUC = 0.5365`, `dummy` accuracy `0.6550` vs. `random_forest` `0.6262`). Controlled ablation (`AblationAnalyzer`) confirmed that Mutual Information selection, PCA, HPO, and the combined optimization pipeline all degraded holdout `macro_f1` compared to the unoptimized baseline (`0.3804`). **No clinical viability claims can be made from this dataset** without external clinical validation and non-leaky ground-truth measurements.
* Even when MUAC is kept for demonstration, Pareto selection chooses a single `decision_tree` (`1.393e-05 kWh`) over `random_forest` (`2.791e-05 kWh`) at identical `1.0000` `macro_f1`.

### 5.2 Cross-Dataset Evaluation Across 5 Public Datasets

All 5 datasets were executed end-to-end through the identical dataset-agnostic `main.py` pipeline across CSV and Parquet formats:

| Dataset | Format & Shape | Task / Size Tier | Recommended Model | CV Primary Score | Holdout Score | Total Pipeline Energy / CO₂e (est.) |
|---|---|---|---|---:|---:|---|
| **Child Health West Bengal (no MUAC)** | CSV `[1000, 8]` | 3-class (`light`) | `decision_tree` | `0.3150` (`macro_f1`) | `0.3804` | `2,738.4 J` / `0.000532 kg` |
| **UCI Breast Cancer Wisconsin** | CSV `[569, 31]` | Imbalanced binary (`light`) | `logistic_regression_tuned` | `0.9787` (`macro_f1`) | `0.9716` (`AUC 0.9960`) | `3,048.7 J` / `0.000593 kg` |
| **OpenML Titanic** (`boat/body/name` excl.) | Parquet `[1309, 14]` | High-card + missing (`moderate`) | `random_forest` | `0.7850` (`macro_f1`) | `0.8015` (`AUC 0.9011`) | `4,675.2 J` / `0.000909 kg` |
| **Diabetes Progression** | Parquet `[442, 11]` | Regression (`moderate`) | `ridge` | `0.4808` (`R²`) | `0.4542` (`RMSE 53.78`) | `3,679.3 J` / `0.000715 kg` |
| **California Housing** | CSV `[20640, 9]` | Medium/large regression (`moderate`) | `lightgbm` | `0.8338` (`R²`) | `0.8386` (`RMSE 0.4599`) | `18,670.1 J` / `0.003630 kg` |

* **Key Cross-Dataset Insights:**
  * On small tabular problems (`Breast Cancer`, `Diabetes`), simple linear models (`logistic_regression_tuned`, `ridge`) outperformed heavy tree ensembles in both predictive quality and energy.
  * On **California Housing (`20,640` rows)**, histogram-based `lightgbm` achieved substantially higher CV `R²` (`0.8338 ± 0.0059`) than `random_forest` (`0.7935 ± 0.0066`) at **~13.5× lower estimated training energy** (`2.502e-05` vs. `3.385e-04 kWh`).

### 5.3 Phase 5 Synthetic Scaling Benchmark (Isolated from Real-World Data)

Executed via [`SizeAwareOptimizer.run_synthetic_scale_benchmark()`](file:///d:/MYSELF/GRP/Carbon-Emission/carbon_automl/evaluation/size_aware.py#L456) (`is_synthetic=True`, LightGBM, 10 `float32` features):

| Synthetic Rows | Complexity Tier | Chunked Inference | Fit Time (s) | Predict Time (s) | Peak Memory (MB) | Estimated Energy (J) |
|---:|---|---|---:|---:|---:|---:|
| `10,000` | `moderate` | `False` | `0.987` | `0.341` | `3.61` | `13.28` |
| `100,000` | `heavy` | `False` | `1.497` | `0.653` | `8.63` | `21.51` |
| `1,000,000` | `large_scale` | `True` (`chunk=50,000`) | `12.377` | `6.713` | `61.07` | `190.90` |

---

## 6. Verification & Known Limitations

* **Automated Test Suite:** `python -m pytest tests/ -v --cov=carbon_automl --tb=short` → **`104 passed, 1 warning in 183.73s`** (`67%` total statement coverage across `carbon_automl`), covering ingestion, integrity, profiling, leakage detection, lossless downcasting, CV/Optuna isolation, quality/recall gates, carbon accounting, registry/cache reuse, FastAPI endpoints, CLI workflows, reporting, and Streamlit `AppTest` smoke checks.
* **Container Validation:** `docker compose -f docker-compose.yml config --quiet` exited `0` (full container image build via `docker compose up --build` was not executed).
* **Known Limitations:**
  * On Windows hosts without hardware RAPL/NVML counters, sub-second phase measurements carry ±50–100% OS background-noise uncertainty and rely on duration-based `10 W/core` estimates as fallback.
  * [`CarbonScheduler`](file:///d:/MYSELF/GRP/Carbon-Emission/carbon_automl/sustainability/carbon_scheduler.py) remains experimental and returns `"unavailable"` unless an external real-time carbon intensity API is configured.
  * XGBoost is optional (`enable_xgboost: false` by default).

---

## 7. Quick Execution Commands

```bash
# 1. Validate configuration and dataset
python main.py --config config.yaml validate

# 2. Profile dataset and run leakage diagnostics
python main.py --config config.yaml profile

# 3. Run full end-to-end training, Carbon Passport, and HTML/PDF reporting
python main.py --config config.yaml

# 4. Run full automated test suite
python -m pytest tests/ -v --cov=carbon_automl

# 5. Launch FastAPI service & Streamlit research dashboard
uvicorn carbon_automl.api.main:app --host 0.0.0.0 --port 8000
streamlit run carbon_automl/dashboard/app.py
```
