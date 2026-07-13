"""
VISUALIZATION SCRIPT FOR CARBON OPTIMIZATION RESULTS
====================================================

This script creates comprehensive visualizations comparing baseline and
optimized models across multiple metrics:
- Accuracy comparison
- CO2 emissions comparison
- Training time comparison
- Carbon Efficiency Gain Index (CEGI)
- Multi-dimensional trade-off analysis
"""

import matplotlib.pyplot as plt
import numpy as np
import re

def parse_results_file(filename):
    """Parse results file and extract metrics"""
    results = []
    
    # Adjust path to look in results folder
    filepath = f"results/{filename}"
    
    with open(filepath, "r") as f:
        content = f.read()
    
    # Split by model sections
    sections = content.split('\n\n')
    
    for section in sections:
        if 'Accuracy:' in section and 'CO2:' in section:
            model_data = {}
            
            # Extract model name
            lines = section.strip().split('\n')
            if lines[0].endswith(':'):
                model_data['name'] = lines[0][:-1]
            else:
                continue
            
            # Extract metrics
            for line in lines[1:]:
                if 'Accuracy:' in line:
                    model_data['accuracy'] = float(line.split(':')[1].strip())
                elif 'F1 Score:' in line:
                    model_data['f1'] = float(line.split(':')[1].strip())
                elif 'CO2:' in line:
                    co2_str = line.split(':')[1].strip().replace(' kg', '')
                    model_data['co2'] = float(co2_str)
                elif 'Training Time:' in line:
                    time_str = line.split(':')[1].strip().replace(' seconds', '')
                    model_data['time'] = float(time_str)
                elif 'CEGI:' in line:
                    model_data['cegi'] = float(line.split(':')[1].strip())
            
            if 'accuracy' in model_data and 'co2' in model_data:
                results.append(model_data)
    
    return results

# Parse baseline results
try:
    baseline_results = parse_results_file("baseline_results.txt")
    baseline = baseline_results[0] if baseline_results else None
except FileNotFoundError:
    print("Warning: results/baseline_results.txt not found. Run baseline/Baseline.py first.")
    baseline = None

# Parse optimized results
try:
    optimized_results = parse_results_file("optimized_results.txt")
except FileNotFoundError:
    print("Error: results/optimized_results.txt not found. Run optimized/Optimized.py first.")
    exit(1)

# Combine all results
all_results = []
if baseline:
    all_results.append({
        'name': 'Baseline\n(RandomForest)',
        'accuracy': baseline['accuracy'],
        'f1': baseline.get('f1', baseline['accuracy']),
        'co2': baseline['co2'],
        'time': baseline.get('time', 0),
        'cegi': baseline.get('cegi', baseline['accuracy'] / baseline['co2'])
    })

for opt in optimized_results:
    # Shorten names for better visualization
    name = opt['name'].replace('LightGBM + Feature Selection', 'LGBM+FS')
    name = name.replace('PCA + LightGBM', 'PCA+LGBM')
    name = name.replace('Minimal Complexity', 'Minimal')
    all_results.append({
        'name': name,
        'accuracy': opt['accuracy'],
        'f1': opt.get('f1', opt['accuracy']),
        'co2': opt['co2'],
        'time': opt.get('time', 0),
        'cegi': opt.get('cegi', opt['accuracy'] / opt['co2'])
    })

# Extract data for plotting
models = [r['name'] for r in all_results]
accuracies = [r['accuracy'] for r in all_results]
f1_scores = [r['f1'] for r in all_results]
emissions = [r['co2'] for r in all_results]
times = [r['time'] for r in all_results]
cegis = [r['cegi'] for r in all_results]

# Color scheme
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
colors = colors[:len(models)]

print("=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

# ============================================================================
# FIGURE 1: Accuracy and F1 Score Comparison
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy comparison
bars1 = ax1.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_ylim(min(accuracies) * 0.95, 1.0)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.tick_params(axis='x', rotation=15)

# Add value labels on bars
for bar, acc in zip(bars1, accuracies):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{acc:.3f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# F1 Score comparison
bars2 = ax2.bar(models, f1_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_title('Model F1 Score Comparison', fontsize=14, fontweight='bold')
ax2.set_ylabel('F1 Score', fontsize=12)
ax2.set_ylim(min(f1_scores) * 0.95, 1.0)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.tick_params(axis='x', rotation=15)

# Add value labels on bars
for bar, f1 in zip(bars2, f1_scores):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{f1:.3f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/accuracy_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/accuracy_comparison.png")

# ============================================================================
# FIGURE 2: CO2 Emissions and Training Time
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# CO2 emissions comparison
bars1 = ax1.bar(models, emissions, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_title('CO2 Emissions Comparison', fontsize=14, fontweight='bold')
ax1.set_ylabel('CO2 Emissions (kg)', fontsize=12)
ax1.set_yscale('log')
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.tick_params(axis='x', rotation=15)

# Add value labels
for bar, co2 in zip(bars1, emissions):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{co2:.6f}',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# Calculate reduction percentages
if baseline:
    baseline_co2 = emissions[0]
    for i, (bar, co2) in enumerate(zip(bars1, emissions)):
        if i > 0:  # Skip baseline
            reduction = ((baseline_co2 - co2) / baseline_co2) * 100
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 0.5,
                     f'-{reduction:.1f}%',
                     ha='center', va='center', fontsize=9, 
                     color='white', fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='green', alpha=0.8))

# Training time comparison
if any(t > 0 for t in times):
    bars2 = ax2.bar(models, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Training Time (seconds)', fontsize=12)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.tick_params(axis='x', rotation=15)
    
    # Add value labels
    for bar, time in zip(bars2, times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{time:.2f}s',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
else:
    ax2.text(0.5, 0.5, 'Training time data not available',
             ha='center', va='center', transform=ax2.transAxes, fontsize=12)
    ax2.set_xticks([])
    ax2.set_yticks([])

plt.tight_layout()
plt.savefig('visualizations/emissions_time_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/emissions_time_comparison.png")

# ============================================================================
# FIGURE 3: Carbon Efficiency Gain Index (CEGI)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(models, cegis, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_title('Carbon Efficiency Gain Index (CEGI)\nHigher = Better', 
             fontsize=14, fontweight='bold')
ax.set_ylabel('CEGI (Accuracy / CO2)', fontsize=12)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.tick_params(axis='x', rotation=15)

# Add value labels
for bar, cegi in zip(bars, cegis):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{cegi:.0f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Highlight best model
best_idx = cegis.index(max(cegis))
bars[best_idx].set_color('#FFD700')
bars[best_idx].set_edgecolor('darkgoldenrod')
bars[best_idx].set_linewidth(3)

plt.tight_layout()
plt.savefig('visualizations/cegi_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/cegi_comparison.png")

# ============================================================================
# FIGURE 4: Accuracy vs CO2 Emissions (Trade-off Analysis)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 7))

# Scatter plot
scatter = ax.scatter(emissions, accuracies, c=range(len(models)), 
                     cmap='viridis', s=300, alpha=0.7, 
                     edgecolors='black', linewidth=2)

ax.set_xlabel('CO2 Emissions (kg)', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Accuracy vs CO2 Emissions Trade-off\n(Top-right = Best)', 
             fontsize=14, fontweight='bold')
ax.set_xscale('log')
ax.grid(alpha=0.3, linestyle='--')

# Add model labels with arrows
for i, (model, co2, acc) in enumerate(zip(models, emissions, accuracies)):
    ax.annotate(model, 
                xy=(co2, acc), 
                xytext=(15, 15 if i % 2 == 0 else -15),
                textcoords='offset points',
                fontsize=10,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[i], alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=2))

# Add ideal region annotation
ax.axhline(y=max(accuracies) * 0.95, color='green', linestyle='--', 
           alpha=0.5, linewidth=2, label='95% of best accuracy')
ax.legend(loc='lower left', fontsize=10)

plt.tight_layout()
plt.savefig('visualizations/accuracy_vs_emissions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/accuracy_vs_emissions.png")

# ============================================================================
# FIGURE 5: Comprehensive Dashboard
# ============================================================================
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Accuracy
ax1 = fig.add_subplot(gs[0, :2])
bars = ax1.barh(models, accuracies, color=colors, alpha=0.8, edgecolor='black')
ax1.set_xlabel('Accuracy', fontweight='bold')
ax1.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)
for bar, acc in zip(bars, accuracies):
    width = bar.get_width()
    ax1.text(width, bar.get_y() + bar.get_height()/2., f'{acc:.3f}',
             ha='left', va='center', fontsize=10, fontweight='bold')

# CO2 Emissions
ax2 = fig.add_subplot(gs[1, :2])
bars = ax2.barh(models, emissions, color=colors, alpha=0.8, edgecolor='black')
ax2.set_xlabel('CO2 Emissions (kg)', fontweight='bold')
ax2.set_xscale('log')
ax2.grid(axis='x', alpha=0.3)
for bar, co2 in zip(bars, emissions):
    width = bar.get_width()
    ax2.text(width, bar.get_y() + bar.get_height()/2., f'{co2:.6f}',
             ha='left', va='center', fontsize=9, fontweight='bold')

# CEGI
ax3 = fig.add_subplot(gs[2, :2])
bars = ax3.barh(models, cegis, color=colors, alpha=0.8, edgecolor='black')
ax3.set_xlabel('CEGI (Higher = Better)', fontweight='bold')
ax3.grid(axis='x', alpha=0.3)
for bar, cegi in zip(bars, cegis):
    width = bar.get_width()
    ax3.text(width, bar.get_y() + bar.get_height()/2., f'{cegi:.0f}',
             ha='left', va='center', fontsize=10, fontweight='bold')

# Summary statistics
ax4 = fig.add_subplot(gs[:, 2])
ax4.axis('off')

summary_text = "OPTIMIZATION SUMMARY\n" + "="*30 + "\n\n"

if baseline:
    best_opt = all_results[1] if len(all_results) > 1 else all_results[0]
    
    co2_reduction = ((baseline['co2'] - best_opt['co2']) / baseline['co2']) * 100
    acc_change = ((best_opt['accuracy'] - baseline['accuracy']) / baseline['accuracy']) * 100
    
    summary_text += f"Baseline Model:\n"
    summary_text += f"  Accuracy: {baseline['accuracy']:.4f}\n"
    summary_text += f"  CO2: {baseline['co2']:.6f} kg\n\n"
    
    summary_text += f"Best Optimized:\n"
    summary_text += f"  {best_opt['name']}\n"
    summary_text += f"  Accuracy: {best_opt['accuracy']:.4f}\n"
    summary_text += f"  CO2: {best_opt['co2']:.6f} kg\n\n"
    
    summary_text += f"Improvements:\n"
    summary_text += f"  CO2 Reduction: {co2_reduction:.1f}%\n"
    summary_text += f"  Accuracy Change: {acc_change:+.1f}%\n\n"
    
    summary_text += f"Best CEGI Model:\n"
    best_cegi_idx = cegis.index(max(cegis))
    summary_text += f"  {models[best_cegi_idx]}\n"
    summary_text += f"  CEGI: {max(cegis):.0f}\n\n"

summary_text += "Optimization Techniques:\n"
summary_text += "  ✓ Memory optimization\n"
summary_text += "  ✓ Feature selection\n"
summary_text += "  ✓ Early stopping\n"
summary_text += "  ✓ LightGBM algorithm\n"
summary_text += "  ✓ PCA reduction\n"
summary_text += "  ✓ Hyperparameter tuning\n"
summary_text += "  ✓ Batch optimization\n"
summary_text += "  ✓ Model simplification\n"

ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Carbon-Optimized ML Training Dashboard', 
             fontsize=16, fontweight='bold', y=0.98)
plt.savefig('visualizations/comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/comprehensive_dashboard.png")

print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE!")
print("=" * 70)
print("\nGenerated files:")
print("  1. visualizations/accuracy_comparison.png")
print("  2. visualizations/emissions_time_comparison.png")
print("  3. visualizations/cegi_comparison.png")
print("  4. visualizations/accuracy_vs_emissions.png")
print("  5. visualizations/comprehensive_dashboard.png")
print("\n" + "=" * 70)