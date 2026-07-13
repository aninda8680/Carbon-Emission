# Phase 1 - Organized Folder Structure

## 📁 Folder Organization

```
Phase_1/
├── baseline/                    # Baseline (unoptimized) implementation
│   └── Baseline.py             # RandomForest baseline model
│
├── optimized/                   # Optimized implementation
│   └── Optimized.py            # 8 optimization techniques
│
├── results/                     # Generated results
│   ├── baseline_results.txt    # Baseline metrics
│   └── optimized_results.txt   # Optimized metrics
│
├── visualizations/              # Generated charts
│   ├── accuracy_comparison.png
│   ├── emissions_time_comparison.png
│   ├── cegi_comparison.png
│   ├── accuracy_vs_emissions.png
│   └── comprehensive_dashboard.png
│
├── documentation/               # Project documentation
│   ├── README.md               # Quick start guide
│   ├── OPTIMIZATION.md         # 70+ page technical guide
│   └── RESULTS_ANALYSIS.md     # Detailed results analysis
│
├── utils.py                     # Shared utility functions
└── plot_result.py              # Visualization generator
```

---

## 🚀 How to Run

### Step 1: Run Baseline

```bash
cd Phase_1/baseline
python Baseline.py
```

**Output:** `../results/baseline_results.txt`

---

### Step 2: Run Optimized

```bash
cd Phase_1/optimized
python Optimized.py
```

**Output:** `../results/optimized_results.txt`

---

### Step 3: Generate Visualizations

```bash
cd Phase_1
python plot_result.py
```

**Output:** 5 PNG files in `visualizations/` folder

---

## 📊 Expected Results

### Baseline Results
- **Location:** `results/baseline_results.txt`
- **Metrics:** Accuracy, F1, Precision, Recall, CO2, Time
- **Model:** RandomForest (100 estimators)

### Optimized Results
- **Location:** `results/optimized_results.txt`
- **Metrics:** 3 models with comprehensive metrics
- **Models:** 
  1. LightGBM + Feature Selection
  2. PCA + LightGBM
  3. Minimal Complexity

### Visualizations
- **Location:** `visualizations/` folder
- **Files:** 5 professional charts
- **Format:** PNG (300 DPI)

---

## 📚 Documentation

### Quick Start
**File:** `documentation/README.md`
- Installation instructions
- Running the project
- Expected results
- Troubleshooting

### Technical Guide
**File:** `documentation/OPTIMIZATION.md`
- Research foundation (30+ citations)
- 8 optimization techniques explained
- Implementation details
- Tools and technologies
- Future improvements

### Results Analysis
**File:** `documentation/RESULTS_ANALYSIS.md`
- Experimental setup
- Detailed results
- Comparative analysis
- Recommendations
- Lessons learned

---

## 🔧 Folder Benefits

### 1. Clear Separation
- Baseline and optimized code separated
- Easy to compare implementations
- No file naming confusion

### 2. Organized Results
- All results in one place
- Easy to find and compare
- Clean project root

### 3. Professional Structure
- Industry-standard organization
- Easy for others to navigate
- Scalable for future phases

### 4. Easy Maintenance
- Update baseline without affecting optimized
- Add new optimizations easily
- Clear documentation location

---

## 🎯 Quick Commands

### Run Everything
```bash
# From Phase_1 directory
cd baseline && python Baseline.py && cd ..
cd optimized && python Optimized.py && cd ..
python plot_result.py
```

### View Results
```bash
# View baseline results
cat results/baseline_results.txt

# View optimized results
cat results/optimized_results.txt
```

### View Visualizations
```bash
# Open visualizations folder
start visualizations/  # Windows
open visualizations/   # Mac
xdg-open visualizations/  # Linux
```

---

## 📝 File Descriptions

### baseline/Baseline.py
- **Purpose:** Establish unoptimized reference point
- **Algorithm:** RandomForest (100 estimators)
- **Features:** All 7 features
- **Optimizations:** None
- **Output:** `../results/baseline_results.txt`

### optimized/Optimized.py
- **Purpose:** Demonstrate carbon reduction techniques
- **Algorithms:** LightGBM (3 variants)
- **Features:** 6 (selected) or 4 (PCA)
- **Optimizations:** 8 techniques
- **Output:** `../results/optimized_results.txt`

### utils.py
- **Purpose:** Shared utility functions
- **Functions:**
  - `load_and_prepare_data()` - Data loading
  - `optimize_dataframe_memory()` - Memory optimization
- **Used by:** Both baseline and optimized scripts

### plot_result.py
- **Purpose:** Generate visualizations
- **Input:** Results from `results/` folder
- **Output:** 5 PNG files in `visualizations/` folder
- **Charts:**
  1. Accuracy comparison
  2. Emissions & time comparison
  3. CEGI comparison
  4. Accuracy vs emissions trade-off
  5. Comprehensive dashboard

---

## 🔄 Workflow

```
1. Run baseline/Baseline.py
   ↓
   Creates results/baseline_results.txt

2. Run optimized/Optimized.py
   ↓
   Creates results/optimized_results.txt

3. Run plot_result.py
   ↓
   Reads results/*.txt
   ↓
   Creates visualizations/*.png
```

---

## ✅ Checklist

Before running:
- [ ] Install dependencies: `pip install pandas scikit-learn lightgbm codecarbon matplotlib numpy`
- [ ] Ensure dataset exists: `../child_health_westbengal_1000.csv`
- [ ] Check Python version: 3.8+

After running:
- [ ] Check `results/baseline_results.txt` exists
- [ ] Check `results/optimized_results.txt` exists
- [ ] Check 5 PNG files in `visualizations/` folder
- [ ] Review metrics in results files
- [ ] View visualizations

---

## 🎓 For Presentation

**Show this structure to demonstrate:**
1. **Professional organization** - Industry-standard folder structure
2. **Clear separation** - Baseline vs optimized clearly separated
3. **Comprehensive results** - All outputs organized
4. **Easy navigation** - Anyone can understand the structure
5. **Scalable design** - Ready for Phase 2 additions

---

## 📞 Need Help?

**Setup issues:** See `documentation/README.md`
**Understanding code:** See `documentation/OPTIMIZATION.md`
**Results questions:** See `documentation/RESULTS_ANALYSIS.md`

---

**Structure Version:** 2.0  
**Last Updated:** May 9, 2026  
**Status:** Organized and ready to use
