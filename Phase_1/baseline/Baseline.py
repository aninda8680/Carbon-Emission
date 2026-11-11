"""
BASELINE ML TRAINING - UNOPTIMIZED APPROACH
============================================

This baseline implementation uses standard practices without any carbon
optimization techniques. It serves as a reference point to measure the
effectiveness of optimization strategies.

Characteristics:
- Standard RandomForest with default parameters (100 estimators)
- No memory optimization (float64)
- No early stopping
- No feature selection
- No hyperparameter tuning
- Full feature set
"""

import time
import warnings
from pathlib import Path
import sys

warnings.filterwarnings('ignore')

# Dynamically resolve directories so script can be run from anywhere
SCRIPT_DIR = Path(__file__).resolve().parent
PHASE1_DIR = SCRIPT_DIR.parent
ROOT_DIR = PHASE1_DIR.parent
DATA_PATH = ROOT_DIR / "child_health_westbengal_1000.csv"
RESULTS_DIR = PHASE1_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Add Phase_1 to sys.path to import utils
if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from codecarbon import EmissionsTracker
from utils import load_and_prepare_data

print("=" * 70)
print("BASELINE ML TRAINING (UNOPTIMIZED)")
print("=" * 70)

# Load data WITHOUT memory optimization
print("\n[1/3] Loading data without optimization...")
start_time = time.time()
X_train, X_test, y_train, y_test = load_and_prepare_data(
    DATA_PATH,
    optimize_memory=False  # Using float64 (higher memory usage)
)
load_time = time.time() - start_time
print(f"✓ Data loaded in {load_time:.2f}s")
print(f"  Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")

# Start tracking
print("\n[2/3] Training baseline RandomForest model...")
tracker = EmissionsTracker(log_level='error', save_to_file=False)
tracker.start()
start_time = time.time()

# Train model with default parameters (computationally expensive)
model = RandomForestClassifier(
    n_estimators=100,      # Default: many trees
    max_depth=None,        # No depth limit (deep trees)
    min_samples_split=2,   # Default splitting
    min_samples_leaf=1,    # Default leaf size
    random_state=42,
    n_jobs=1               # Single thread for fair comparison
)
model.fit(X_train, y_train)

# Stop tracking
training_time = time.time() - start_time
emissions = tracker.stop()

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted')

print(f"✓ Model trained in {training_time:.2f}s")

# Display results
print("\n" + "=" * 70)
print("BASELINE RESULTS")
print("=" * 70)
print(f"\nModel: RandomForest (100 estimators, no optimization)")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  F1 Score:  {f1:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  CO2:       {emissions:.6f} kg")
print(f"  Time:      {training_time:.2f} seconds")
print(f"  Features:  {X_train.shape[1]} (all features)")

# Save results
print("\n[3/3] Saving results...")
with open(RESULTS_DIR / "baseline_results.txt", "w") as f:
    f.write("=" * 70 + "\n")
    f.write("BASELINE ML TRAINING RESULTS (UNOPTIMIZED)\n")
    f.write("=" * 70 + "\n\n")
    f.write("RandomForest:\n")
    f.write(f"  Accuracy: {accuracy:.4f}\n")
    f.write(f"  F1 Score: {f1:.4f}\n")
    f.write(f"  Precision: {precision:.4f}\n")
    f.write(f"  Recall: {recall:.4f}\n")
    f.write(f"  CO2: {emissions:.6f} kg\n")
    f.write(f"  Training Time: {training_time:.2f} seconds\n")
    f.write(f"  Features Used: {X_train.shape[1]}\n")
    f.write("  Memory Optimization: No\n")
    f.write("  Early Stopping: No\n")
    f.write("  Feature Selection: No\n")
    f.write("  Hyperparameter Tuning: No\n")

print(f"✓ Results saved to '{RESULTS_DIR / 'baseline_results.txt'}'")
print("\n" + "=" * 70)
print("BASELINE TRAINING COMPLETE!")
print("=" * 70)
print("\nNote: This baseline uses no optimization techniques.")
print("Run Optimized.py to see carbon reduction strategies in action.")