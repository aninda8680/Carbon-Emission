"""Tests for CarbonCalculator, EnergyMonitor, StatisticalAnalyzer, CarbonBudgetOptimizer, and CarbonScheduler."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from carbon_automl.sustainability.carbon_budget import CarbonBudgetOptimizer
from carbon_automl.sustainability.carbon_calculator import CarbonCalculator
from carbon_automl.sustainability.carbon_scheduler import CarbonScheduler
from carbon_automl.sustainability.energy_monitor import EnergyMonitor
from carbon_automl.sustainability.statistical_analysis import StatisticalAnalyzer


def test_co2_calculation() -> None:
    """CO2 equals energy_kwh * carbon_intensity_kg_per_kwh."""
    calc = CarbonCalculator(carbon_intensity_kg_per_kwh=0.7)
    assert calc.compute_co2(0.0) == pytest.approx(0.0)
    assert calc.compute_co2(2.5) == pytest.approx(1.75)


def test_carbon_intensity_configurable() -> None:
    """Changing grid carbon intensity changes computed CO2 and assumption statement."""
    calc_low = CarbonCalculator(carbon_intensity_kg_per_kwh=0.2, intensity_source="hydro_grid")
    calc_high = CarbonCalculator(carbon_intensity_kg_per_kwh=0.9, intensity_source="coal_grid")

    assert calc_low.compute_co2(1.0) == pytest.approx(0.2)
    assert calc_high.compute_co2(1.0) == pytest.approx(0.9)
    assert "0.2000" in calc_low.assumption_statement
    assert "hydro_grid" in calc_low.assumption_statement
    assert "0.9000" in calc_high.assumption_statement


def test_uncertainty_documented() -> None:
    """EnergyMonitor and inference emissions measurements document uncertainty_pct."""
    # Force deterministic fallback inside EnergyMonitor so test is fast & offline-safe
    monitor = EnergyMonitor(phase_name="unit_test", carbon_intensity_kg_per_kwh=0.7)
    with patch.object(monitor, "_check_codecarbon", return_value=False):
        with monitor:
            _ = sum(range(1000))
    measurement = monitor.last_measurement

    assert measurement is not None
    assert measurement.phase_name == "unit_test"
    assert measurement.uncertainty_pct >= 50.0
    assert measurement.energy_kwh is not None
    assert measurement.co2_kg == pytest.approx(measurement.energy_kwh * 0.7)

    # Also test inference emissions uncertainty reporting
    X = np.zeros((20, 3))
    y = np.array([0, 1] * 10)
    clf = DummyClassifier(strategy="most_frequent").fit(X, y)

    calc = CarbonCalculator(carbon_intensity_kg_per_kwh=0.7)
    with patch(
        "carbon_automl.sustainability.energy_monitor.EnergyMonitor._check_codecarbon",
        return_value=False,
    ):
        inf = calc.compute_inference_emissions(clf, X, n_predictions=100, repetitions=3)

    assert "measurement_uncertainty_pct" in inf
    assert inf["measurement_uncertainty_pct"] >= 50.0
    assert "warning" in inf and len(inf["warning"]) > 0


def test_no_unsupported_savings_claims() -> None:
    """StatisticalAnalyzer.is_demonstrably_greener returns False for n < 5."""
    analyzer = StatisticalAnalyzer()

    # Even with 10x lower values, n=3 is statistically insufficient for Wilcoxon
    energy_a = [0.001, 0.001, 0.001]
    energy_b = [0.010, 0.010, 0.010]
    greener, reason = analyzer.is_demonstrably_greener(energy_a, energy_b)

    assert greener is False
    assert "Insufficient samples" in reason or "not statistically significant" in reason

    # Also verify non-significant difference with n=5 returns False
    energy_c = [1.0, 2.0, 1.5, 2.5, 1.2]
    energy_d = [1.1, 1.9, 1.6, 2.4, 1.3]
    greener_ns, _ = analyzer.is_demonstrably_greener(energy_c, energy_d)
    assert greener_ns is False


def test_amortization_calculation() -> None:
    """Training + inference emissions sum and break-even predictions are exact."""
    calc = CarbonCalculator(carbon_intensity_kg_per_kwh=0.7)
    res = calc.amortize_training(
        training_co2_kg=0.010,
        inference_co2_per_prediction_kg=1e-6,
        prediction_volume=5000,
    )

    assert res["training_co2_kg"] == pytest.approx(0.010)
    assert res["inference_co2_kg"] == pytest.approx(0.005)
    assert res["total_co2_kg"] == pytest.approx(0.015)
    assert res["training_fraction"] + res["inference_fraction"] == pytest.approx(1.0)
    assert res["break_even_predictions"] == 10000


def test_carbon_budget_violated_reported() -> None:
    """Budget optimizer explains budget/quality conflicts rather than silently relaxing."""
    calc = CarbonCalculator(carbon_intensity_kg_per_kwh=0.7)
    optimizer = CarbonBudgetOptimizer(budget_kg=0.005, carbon_calculator=calc)

    candidates = [
        {"name": "model_a", "training_co2_kg": 0.010, "macro_f1": 0.90},
        {"name": "model_b", "training_co2_kg": 0.020, "macro_f1": 0.93},
    ]
    res = optimizer.find_models_within_budget(candidates)

    assert res["within_budget"] == []
    assert len(res["over_budget"]) == 2
    assert res["best_within_budget"] is None
    assert res["conflict_explanation"] is not None
    assert "conflict" in res["conflict_explanation"].lower()
    assert "not silently relaxed" in res["conflict_explanation"].lower()


def test_carbon_scheduler_unavailable_without_external_data() -> None:
    """Experimental CarbonScheduler never invents grid forecasts when unconfigured."""
    scheduler = CarbonScheduler(expected_duration_minutes=5.0, expected_energy_kwh=0.05)
    rec = scheduler.get_recommendation()

    assert rec["experimental"] is True
    assert rec["recommendation"] == "unavailable"
    assert rec["estimated_savings_kg"] is None
    assert "No real-time carbon intensity source" in rec["reason"]
