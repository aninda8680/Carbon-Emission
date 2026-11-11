"""
Sustainability & Carbon Accounting Engine for carbon-aware AutoML.

Modules:
    energy_monitor      - Wraps CodeCarbon for per-phase energy measurement
    carbon_calculator   - CO₂e computation, inference emissions, amortization
    statistical_analysis - Bootstrap CI, Wilcoxon tests, greener-than comparisons
    pareto              - Pareto frontier analysis and model ranking
    carbon_budget       - Budget-constrained model selection
    carbon_scheduler    - (Experimental) Time-shifting recommendations

NOTE: At small dataset scale (~1000 rows), training completes in < 1 second.
CodeCarbon measurements at this scale are dominated by background process
noise. Treat all sub-milligram CO₂ figures as ORDER-OF-MAGNITUDE ESTIMATES
with ≥ 50% uncertainty, not as precise environmental measurements.
"""

from .energy_monitor import (
    EnergyMonitor,
    EnergyMeasurement,
    JOULES_PER_KWH,
    STANDARD_PHASES,
    get_hardware_specs,
)
from .carbon_calculator import CarbonCalculator
from .statistical_analysis import StatisticalAnalyzer
from .pareto import ParetoAnalyzer, ParetoRecommendation
from .carbon_budget import CarbonBudgetOptimizer, BudgetAnalysisResult
from .carbon_scheduler import CarbonScheduler, ScheduleRecommendation

__all__ = [
    "EnergyMonitor",
    "EnergyMeasurement",
    "JOULES_PER_KWH",
    "STANDARD_PHASES",
    "get_hardware_specs",
    "CarbonCalculator",
    "StatisticalAnalyzer",
    "ParetoAnalyzer",
    "ParetoRecommendation",
    "CarbonBudgetOptimizer",
    "BudgetAnalysisResult",
    "CarbonScheduler",
    "ScheduleRecommendation",
]
