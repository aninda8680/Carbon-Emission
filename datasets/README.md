# Child Health & Malnutrition Dataset (West Bengal)

## Overview
- **File**: `child_health_westbengal_1000.csv`
- **Rows**: 1,000
- **Columns**: 8 (7 features + 1 target)
- **Target Column**: `Malnutrition Status` (`Normal`, `Moderate`, `Severe`)
- **SHA-256**: `c68a778e13cc257f80f9edb8c44ea2b42eb9c30fcbbcbb9cce3d4c4bbb1d7ca8`

## Schema
| Column | Type | Description |
|---|---|---|
| `Age (months)` | int64 | Child age in months |
| `Gender` | object | `Male` or `Female` |
| `Height (cm)` | float64 | Standing/recumbent height in cm |
| `Weight (kg)` | float64 | Body weight in kg |
| `Subdivision (West Bengal)` | object | Administrative subdivision in West Bengal |
| `Mother's Education` | object | Maternal education level (`Illiterate`, `Primary`, `Secondary`, etc.) |
| `Mid-Upper Arm Circumference (cm)` | float64 | MUAC measurement in cm |
| `Malnutrition Status` | object | Target label (`Normal`, `Moderate`, `Severe`) |

## CRITICAL FINDING: Target Leakage via MUAC
`Mid-Upper Arm Circumference (cm)` (MUAC) is a WHO anthropometric diagnostic criterion used directly to define `Malnutrition Status` in this dataset:
- MUAC value ranges separate the target classes deterministically (non-overlapping thresholds).
- Training a model with MUAC included yields ~100% accuracy purely from diagnostic label leakage, masking true predictive generalization from demographic and anthropometric covariates (`Age`, `Height`, `Weight`, etc.).
- The framework's `LeakageDetector` intentionally flags `Mid-Upper Arm Circumference (cm)` as **critical** leakage. Users should decide whether to add `"Mid-Upper Arm Circumference (cm)"` to `exclude_columns` in `config.yaml` depending on their scientific objective.

## License
Assumed Public Domain / CC0 for research and benchmarking purposes until an upstream dataset license is explicitly confirmed. The original file at project root (`../child_health_westbengal_1000.csv`) must never be modified.
