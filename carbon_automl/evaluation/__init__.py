"""
Carbon AutoML - Evaluation Engine
Provides model zoo, metrics, validation, hyperparameter optimization,
feature selection, ablation analysis, and size-aware optimization capabilities.
"""

from .model_zoo import get_model_zoo
from .metrics import MetricsCalculator
from .validation import ValidationEngine
from .optimizer import HyperparameterOptimizer
from .feature_selection import FeatureSelector
from .ablation import AblationAnalyzer
from .size_aware import SizeAwareOptimizer

__all__ = [
    "get_model_zoo",
    "MetricsCalculator",
    "ValidationEngine",
    "HyperparameterOptimizer",
    "FeatureSelector",
    "AblationAnalyzer",
    "SizeAwareOptimizer",
]
