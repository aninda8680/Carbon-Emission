"""
Master script to run all experiments in sequence
================================================

This script runs:
1. Baseline model (baseline/Baseline.py)
2. Optimized models (optimized/Optimized.py)
3. Visualization generation (plot_result.py)

Usage:
    python run_all.py
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, cwd=None):
    """Run a command and return success status"""
    try:
        print(f"\n{'='*70}")
        print(f"Running: {command}")
        print(f"Directory: {cwd if cwd else 'current'}")
        print('='*70)
        
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print(f"\n✓ Success: {command}")
            return True
        else:
            print(f"\n✗ Failed: {command}")
            return False
            
    except Exception as e:
        print(f"\n✗ Error running {command}: {e}")
        return False

def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("CARBON EMISSION REDUCTION - COMPLETE PIPELINE")
    print("="*70)
    print("\nThis script will run:")
    print("  1. Baseline model (RandomForest)")
    print("  2. Optimized models (LightGBM variants)")
    print("  3. Visualization generation")
    print("\nEstimated time: 1-2 minutes")
    print("="*70)
    
    input("\nPress Enter to continue...")
    
    # Get the Phase_1 directory
    phase1_dir = Path(__file__).parent
    
    # Step 1: Run Baseline
    print("\n\n" + "="*70)
    print("STEP 1/3: RUNNING BASELINE MODEL")
    print("="*70)
    
    baseline_dir = phase1_dir / "baseline"
    if not run_command("python Baseline.py", cwd=baseline_dir):
        print("\n✗ Baseline failed. Stopping.")
        sys.exit(1)
    
    # Step 2: Run Optimized
    print("\n\n" + "="*70)
    print("STEP 2/3: RUNNING OPTIMIZED MODELS")
    print("="*70)
    
    optimized_dir = phase1_dir / "optimized"
    if not run_command("python Optimized.py", cwd=optimized_dir):
        print("\n✗ Optimized failed. Stopping.")
        sys.exit(1)
    
    # Step 3: Generate Visualizations
    print("\n\n" + "="*70)
    print("STEP 3/3: GENERATING VISUALIZATIONS")
    print("="*70)
    
    if not run_command("python plot_result.py", cwd=phase1_dir):
        print("\n✗ Visualization failed. Stopping.")
        sys.exit(1)
    
    # Success!
    print("\n\n" + "="*70)
    print("✓ ALL STEPS COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    print("\n📊 Generated Files:")
    print("\nResults:")
    print("  - results/baseline_results.txt")
    print("  - results/optimized_results.txt")
    
    print("\nVisualizations:")
    print("  - visualizations/accuracy_comparison.png")
    print("  - visualizations/emissions_time_comparison.png")
    print("  - visualizations/cegi_comparison.png")
    print("  - visualizations/accuracy_vs_emissions.png")
    print("  - visualizations/comprehensive_dashboard.png")
    
    print("\n📚 Next Steps:")
    print("  1. View results: cat results/*.txt")
    print("  2. View visualizations: open visualizations/")
    print("  3. Read analysis: documentation/RESULTS_ANALYSIS.md")
    print("  4. Prepare presentation: ../PRESENTATION_GUIDE.md")
    
    print("\n" + "="*70)
    print("Project complete! Ready for presentation. 🎉")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
