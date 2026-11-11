# Setup Guide - Carbon Emission Reduction Project

## Quick Setup (5 minutes)

### Step 1: Install Python Dependencies

```bash
pip install pandas scikit-learn lightgbm codecarbon matplotlib numpy
```

**What each package does:**
- `pandas` - Data loading and manipulation
- `scikit-learn` - Feature selection, PCA, metrics
- `lightgbm` - Energy-efficient gradient boosting
- `codecarbon` - Carbon emission tracking
- `matplotlib` - Visualization
- `numpy` - Numerical operations

---

### Step 2: Verify Installation

```bash
python -c "import pandas, sklearn, lightgbm, codecarbon, matplotlib, numpy; print('All packages installed successfully!')"
```

**Expected output:**
```
All packages installed successfully!
```

---

### Step 3: Run the Project

```bash
# Navigate to Phase 1 directory
cd Phase_1

# Option A: Run everything with the master pipeline (Recommended)
python run_all.py

# Option B: Run step-by-step
python baseline/Baseline.py
python optimized/Optimized.py
python plot_result.py
```

---

## Detailed Setup Instructions

### Option 1: Using pip (Recommended)

```bash
# Install all dependencies at once
pip install pandas scikit-learn lightgbm codecarbon matplotlib numpy

# Or install with specific versions (for reproducibility)
pip install pandas==2.0.0 scikit-learn==1.3.0 lightgbm==4.0.0 codecarbon==2.3.0 matplotlib==3.7.0 numpy==1.24.0
```

---

### Option 2: Using conda

```bash
# Create a new conda environment
conda create -n carbon-ml python=3.10

# Activate the environment
conda activate carbon-ml

# Install packages
conda install pandas scikit-learn matplotlib numpy
pip install lightgbm codecarbon
```

---

### Option 3: Using requirements.txt

Create a file named `requirements.txt`:

```
pandas>=2.0.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
codecarbon>=2.3.0
matplotlib>=3.7.0
numpy>=1.24.0
```

Then install:

```bash
pip install -r requirements.txt
```

---

## Troubleshooting

### Issue 1: pip not found

**Problem:** `pip: command not found`

**Solution:**
```bash
# On Windows
python -m pip install pandas scikit-learn lightgbm codecarbon matplotlib numpy

# On Linux/Mac
python3 -m pip install pandas scikit-learn lightgbm codecarbon matplotlib numpy
```

---

### Issue 2: Permission denied

**Problem:** `Permission denied` or `Access denied`

**Solution:**
```bash
# Install for current user only
pip install --user pandas scikit-learn lightgbm codecarbon matplotlib numpy

# Or use sudo (Linux/Mac only)
sudo pip install pandas scikit-learn lightgbm codecarbon matplotlib numpy
```

---

### Issue 3: LightGBM installation fails

**Problem:** LightGBM fails to install (especially on Windows)

**Solution:**

**Windows:**
```bash
# Install Visual C++ Build Tools first
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Then install LightGBM
pip install lightgbm
```

**Linux:**
```bash
# Install build dependencies
sudo apt-get install build-essential

# Then install LightGBM
pip install lightgbm
```

**Mac:**
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Then install LightGBM
pip install lightgbm
```

---

### Issue 4: CodeCarbon warnings

**Problem:** CodeCarbon shows warnings about CPU/GPU detection

**Solution:** These warnings are informational and can be ignored. The tracker still works correctly. To suppress them:

```python
# In the code, use:
tracker = EmissionsTracker(log_level='error', save_to_file=False)
```

This is already implemented in the updated code.

---

### Issue 5: Module not found after installation

**Problem:** `ModuleNotFoundError` even after installation

**Solution:**
```bash
# Check which Python you're using
which python  # Linux/Mac
where python  # Windows

# Check which pip you're using
which pip  # Linux/Mac
where pip  # Windows

# Make sure they match
# If not, use:
python -m pip install [package]
```

---

### Issue 6: Matplotlib display issues

**Problem:** Plots don't display or show errors

**Solution:**

**If running on a server without display:**
```python
# Add this at the top of plot_result.py
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
```

**If running in Jupyter:**
```python
# Add this at the top
%matplotlib inline
```

---

## Verification Steps

### 1. Check Python Version

```bash
python --version
```

**Required:** Python 3.8 or higher

---

### 2. Check Package Versions

```bash
python -c "import pandas; print('pandas:', pandas.__version__)"
python -c "import sklearn; print('scikit-learn:', sklearn.__version__)"
python -c "import lightgbm; print('lightgbm:', lightgbm.__version__)"
python -c "import codecarbon; print('codecarbon:', codecarbon.__version__)"
python -c "import matplotlib; print('matplotlib:', matplotlib.__version__)"
python -c "import numpy; print('numpy:', numpy.__version__)"
```

---

### 3. Test Data Loading

```bash
cd Phase_1
python -c "from utils import load_and_prepare_data; X_train, X_test, y_train, y_test = load_and_prepare_data('../child_health_westbengal_1000.csv'); print('Data loaded successfully!')"
```

---

### 4. Test CodeCarbon

```bash
python -c "from codecarbon import EmissionsTracker; tracker = EmissionsTracker(log_level='error'); tracker.start(); tracker.stop(); print('CodeCarbon working!')"
```

---

## Running the Project

### Full Workflow

```bash
# 1. Navigate to project directory
cd /path/to/project

# 2. Navigate to Phase 1
cd Phase_1

# 3. Run baseline (takes ~10-30 seconds)
python Baseline.py

# Expected output:
# ======================================================================
# BASELINE ML TRAINING (UNOPTIMIZED)
# ======================================================================
# [1/3] Loading data without optimization...
# ✓ Data loaded in X.XXs
# [2/3] Training baseline RandomForest model...
# ✓ Model trained in X.XXs
# ======================================================================
# BASELINE RESULTS
# ======================================================================
# ...

# 4. Run optimized version (takes ~20-60 seconds)
python Optimized.py

# Expected output:
# ======================================================================
# CARBON-OPTIMIZED ML TRAINING PIPELINE
# ======================================================================
# [1/6] Loading data with memory optimization...
# ✓ Data loaded in X.XXs
# [2/6] Applying intelligent feature selection...
# ✓ Features reduced from X to Y
# ...

# 5. Generate visualizations (takes ~5-10 seconds)
python plot_result.py

# Expected output:
# ======================================================================
# GENERATING VISUALIZATIONS
# ======================================================================
# ✓ Saved: accuracy_comparison.png
# ✓ Saved: emissions_time_comparison.png
# ✓ Saved: cegi_comparison.png
# ✓ Saved: accuracy_vs_emissions.png
# ✓ Saved: comprehensive_dashboard.png
```

---

### Individual Scripts

**Run only baseline:**
```bash
cd Phase_1
python Baseline.py
```

**Run only optimized:**
```bash
cd Phase_1
python Optimized.py
```

**Run only visualizations:**
```bash
cd Phase_1
python plot_result.py
```

**Note:** Visualizations require both baseline and optimized results to exist.

---

## Expected Output Files

After running all scripts, you should have:

```
Phase_1/
├── baseline_results.txt              # Baseline metrics
├── optimized_results.txt             # Optimized metrics
├── accuracy_comparison.png           # Accuracy & F1 charts
├── emissions_time_comparison.png     # CO2 & time charts
├── cegi_comparison.png               # CEGI metric chart
├── accuracy_vs_emissions.png         # Trade-off analysis
└── comprehensive_dashboard.png       # Complete overview
```

---

## System Requirements

### Minimum Requirements
- **OS:** Windows 10, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python:** 3.8 or higher
- **RAM:** 2 GB
- **Disk Space:** 500 MB (for packages)
- **CPU:** Any modern CPU (2+ cores recommended)

### Recommended Requirements
- **OS:** Windows 11, macOS 12+, or Linux (Ubuntu 20.04+)
- **Python:** 3.10 or higher
- **RAM:** 4 GB or more
- **Disk Space:** 1 GB
- **CPU:** 4+ cores for faster training

### Optional
- **GPU:** Not required (CPU-only is fine)
- **Internet:** Required for initial package installation only

---

## Development Environment Setup

### Using VS Code (Recommended)

1. **Install VS Code:** https://code.visualstudio.com/

2. **Install Python Extension:**
   - Open VS Code
   - Go to Extensions (Ctrl+Shift+X)
   - Search for "Python"
   - Install the official Python extension by Microsoft

3. **Open Project:**
   ```bash
   cd /path/to/project
   code .
   ```

4. **Select Python Interpreter:**
   - Press Ctrl+Shift+P
   - Type "Python: Select Interpreter"
   - Choose your Python installation

5. **Run Scripts:**
   - Open a Python file
   - Press F5 to run with debugging
   - Or right-click and select "Run Python File in Terminal"

---

### Using PyCharm

1. **Install PyCharm:** https://www.jetbrains.com/pycharm/

2. **Open Project:**
   - File → Open
   - Select project directory

3. **Configure Interpreter:**
   - File → Settings → Project → Python Interpreter
   - Add interpreter if needed

4. **Run Scripts:**
   - Right-click on Python file
   - Select "Run 'filename.py'"

---

### Using Jupyter Notebook (Optional)

```bash
# Install Jupyter
pip install jupyter

# Start Jupyter
jupyter notebook

# Create a new notebook and run:
%run Phase_1/Baseline.py
%run Phase_1/Optimized.py
%run Phase_1/plot_result.py
```

---

## Docker Setup (Advanced)

If you prefer using Docker:

**Create Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["bash"]
```

**Build and run:**
```bash
# Build image
docker build -t carbon-ml .

# Run container
docker run -it -v $(pwd):/app carbon-ml

# Inside container
cd Phase_1
python Baseline.py
python Optimized.py
python plot_result.py
```

---

## Cloud Setup (Google Colab)

If you want to run in Google Colab:

```python
# In a Colab notebook

# Install packages
!pip install pandas scikit-learn lightgbm codecarbon matplotlib numpy

# Upload your dataset
from google.colab import files
uploaded = files.upload()  # Upload child_health_westbengal_1000.csv

# Clone or upload your code files
# Then run:
!python Phase_1/Baseline.py
!python Phase_1/Optimized.py
!python Phase_1/plot_result.py

# Download results
files.download('Phase_1/baseline_results.txt')
files.download('Phase_1/optimized_results.txt')
files.download('Phase_1/comprehensive_dashboard.png')
```

---

## Testing Your Setup

### Quick Test Script

Create a file named `test_setup.py`:

```python
"""Test script to verify setup"""

print("Testing imports...")

try:
    import pandas as pd
    print("✓ pandas")
except ImportError:
    print("✗ pandas - Run: pip install pandas")

try:
    import sklearn
    print("✓ scikit-learn")
except ImportError:
    print("✗ scikit-learn - Run: pip install scikit-learn")

try:
    import lightgbm
    print("✓ lightgbm")
except ImportError:
    print("✗ lightgbm - Run: pip install lightgbm")

try:
    import codecarbon
    print("✓ codecarbon")
except ImportError:
    print("✗ codecarbon - Run: pip install codecarbon")

try:
    import matplotlib
    print("✓ matplotlib")
except ImportError:
    print("✗ matplotlib - Run: pip install matplotlib")

try:
    import numpy
    print("✓ numpy")
except ImportError:
    print("✗ numpy - Run: pip install numpy")

print("\nTesting data loading...")
try:
    from Phase_1.utils import load_and_prepare_data
    X_train, X_test, y_train, y_test = load_and_prepare_data(
        "child_health_westbengal_1000.csv"
    )
    print(f"✓ Data loaded: {X_train.shape[0]} training samples")
except Exception as e:
    print(f"✗ Data loading failed: {e}")

print("\nTesting CodeCarbon...")
try:
    from codecarbon import EmissionsTracker
    tracker = EmissionsTracker(log_level='error', save_to_file=False)
    tracker.start()
    tracker.stop()
    print("✓ CodeCarbon working")
except Exception as e:
    print(f"✗ CodeCarbon failed: {e}")

print("\n" + "="*50)
print("Setup test complete!")
print("="*50)
```

Run it:
```bash
python test_setup.py
```

---

## Getting Help

### If you encounter issues:

1. **Check this guide** for common solutions
2. **Read error messages carefully** - they often tell you what's wrong
3. **Check package documentation:**
   - pandas: https://pandas.pydata.org/docs/
   - scikit-learn: https://scikit-learn.org/stable/
   - lightgbm: https://lightgbm.readthedocs.io/
   - codecarbon: https://docs.codecarbon.io/

4. **Search for the error message** on Google or Stack Overflow
5. **Check Python version compatibility** - use Python 3.8+

---

## Next Steps

After successful setup:

1. ✅ Read `Phase_1/README.md` for project overview
2. ✅ Run `Baseline.py` to establish baseline
3. ✅ Run `Optimized.py` to see optimizations
4. ✅ Run `plot_result.py` to visualize results
5. ✅ Read `Phase_1/OPTIMIZATION.md` for detailed explanations
6. ✅ Review `PRESENTATION_GUIDE.md` for presentation tips

---

## Summary

**Quick Setup:**
```bash
pip install pandas scikit-learn lightgbm codecarbon matplotlib numpy
cd Phase_1
python Baseline.py
python Optimized.py
python plot_result.py
```

**That's it!** You should now have:
- All dependencies installed
- Baseline results generated
- Optimized results generated
- 5 visualization files created

---

**Setup Guide Version:** 1.0  
**Last Updated:** May 9, 2026  
**Estimated Setup Time:** 5-10 minutes
