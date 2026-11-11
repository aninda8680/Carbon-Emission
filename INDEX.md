# Carbon-Efficient Machine Learning: Documentation Index

Welcome to the project documentation hub for **Carbon-Efficient Machine Learning for Sustainable Model Training**.

---

## 🎯 Quick Navigation

| Document | Description | Target Audience |
| :--- | :--- | :--- |
| **[`report/report.md`](report/report.md)** | 🌿 **Dedicated Final Report** with implementation fixes, benchmarks & analysis | Evaluation & Submission |
| **[`CARBON_EMISSION_FINAL_REPORT.md`](CARBON_EMISSION_FINAL_REPORT.md)** | Flagship Final Report on carbon emission reduction & benchmarks | Evaluation & Submission |
| **[`FINAL_SUMMARY.md`](FINAL_SUMMARY.md)** | Master project completion report and executive overview | Evaluators & General Review |
| **[`prompt.md`](prompt.md)** | Official 15-slide academic presentation deck outline | Presentation Prep |
| **[`PRESENTATION_GUIDE.md`](PRESENTATION_GUIDE.md)** | Full 15-minute speaking script and section-by-section guide | Speakers & Viva Defense |
| **[`COMPARISON_RESULTS.md`](COMPARISON_RESULTS.md)** | Detailed side-by-side metric comparison and analysis | Technical Review |
| **[`QUICK_REFERENCE.md`](QUICK_REFERENCE.md)** | 2-page quick cheat sheet and results summary | Quick Reference |
| **[`SETUP_GUIDE.md`](SETUP_GUIDE.md)** | Installation, prerequisites, and execution commands | Developers & Setup |
| **[`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md)** | Full project directory tree and future phase roadmaps | Architecture |
| **[`Phase_1/documentation/OPTIMIZATION.md`](Phase_1/documentation/OPTIMIZATION.md)** | 70+ page deep-dive into the 8 optimization techniques & research | Researchers |
| **[`Phase_1/documentation/RESULTS_ANALYSIS.md`](Phase_1/documentation/RESULTS_ANALYSIS.md)** | In-depth breakdown of metrics, trade-offs, and scaling | Analysis |

---

## 🚀 Quick Execution Guide

To reproduce all experiments, benchmarks, and visualizations end-to-end:

```bash
# 1. Install dependencies
pip install pandas scikit-learn lightgbm codecarbon matplotlib numpy optuna

# 2. Run complete pipeline
cd Phase_1
python run_all.py
```

### Outputs Generated:
* **Metrics:** [`Phase_1/results/baseline_results.txt`](Phase_1/results/baseline_results.txt) and [`Phase_1/results/optimized_results.txt`](Phase_1/results/optimized_results.txt)
* **Visualizations:**
  1. `Phase_1/visualizations/accuracy_comparison.png`
  2. `Phase_1/visualizations/emissions_time_comparison.png`
  3. `Phase_1/visualizations/cegi_comparison.png`
  4. `Phase_1/visualizations/accuracy_vs_emissions.png`
  5. `Phase_1/visualizations/comprehensive_dashboard.png`

---

## 📊 Summary of Implemented Models

1. **Baseline Model (`RandomForest`):** Default unoptimized benchmark (100 trees, full float64 precision).
2. **Tier 1 - Production Best (`LightGBM + Feature Selection`):** 100% accuracy, ~5–6x faster, 14.3% fewer features, highest CEGI score.
3. **Tier 2 - Balanced (`PCA + LightGBM`):** 98.5% accuracy, 4 features (43% reduction), rapid execution.
4. **Tier 3 - Maximum Speed (`Minimal Complexity`):** 98.5% accuracy, ~25x faster training, ultra-compact (20 shallow trees).

