"""Tests for quality constraints, baseline fallback, justification, and Pareto optimality."""

from __future__ import annotations

from carbon_automl.intelligence.model_selector import ModelSelector
from carbon_automl.sustainability.pareto import ParetoAnalyzer


def test_quality_violation_model_not_recommended() -> None:
    """A low-energy model below quality_min MUST NOT be recommended."""
    candidates = [
        {
            "name": "tiny_bad_model",
            "macro_f1": 0.60,
            "training_energy_kwh": 0.0001,
            "inference_energy_kwh": 0.00001,
            "training_co2_kg": 0.00007,
        },
        {
            "name": "good_model",
            "macro_f1": 0.88,
            "training_energy_kwh": 0.0020,
            "inference_energy_kwh": 0.0001,
            "training_co2_kg": 0.0014,
        },
    ]

    selector_res = ModelSelector().select(
        model_results=candidates,
        baseline_name="good_model",
        quality_metric="macro_f1",
        quality_min=0.80,
        energy_metric="training_energy_kwh",
        task_type="classification",
    )
    assert "tiny_bad_model" in selector_res["quality_constraint_violations"]
    assert selector_res["recommended"]["name"] == "good_model"

    pareto_res = ParetoAnalyzer().recommend_model(
        models=candidates,
        quality_metric="macro_f1",
        energy_metric="training_energy_kwh",
        quality_min=0.80,
        baseline_metrics=candidates[1],
    )
    assert pareto_res["recommended_model"]["name"] == "good_model"
    assert "tiny_bad_model" in pareto_res["tradeoffs"]["rejected_for_quality_violation"]


def test_baseline_recommended_when_all_fail() -> None:
    """If all optimized models fail quality_min, baseline is recommended."""
    candidates = [
        {
            "name": "baseline_rf",
            "macro_f1": 0.78,
            "training_energy_kwh": 0.005,
            "training_co2_kg": 0.0035,
        },
        {
            "name": "optimized_lgbm",
            "macro_f1": 0.76,
            "training_energy_kwh": 0.002,
            "training_co2_kg": 0.0014,
        },
    ]

    selector_res = ModelSelector().select(
        model_results=candidates,
        baseline_name="baseline_rf",
        quality_metric="macro_f1",
        quality_min=0.85,
        energy_metric="training_energy_kwh",
        task_type="classification",
    )
    assert selector_res["recommended"]["name"] == "baseline_rf"
    assert "baseline_rf" in selector_res["quality_constraint_violations"]
    assert "optimized_lgbm" in selector_res["quality_constraint_violations"]

    pareto_res = ParetoAnalyzer().recommend_model(
        models=candidates,
        quality_metric="macro_f1",
        energy_metric="training_energy_kwh",
        quality_min=0.85,
        baseline_metrics=candidates[0],
    )
    assert pareto_res["recommended_model"]["name"] == "baseline_rf"
    assert "fallback" in pareto_res["reason"].lower()


def test_recommendation_has_justification() -> None:
    """Recommendation response contains a non-empty justification string."""
    candidates = [
        {
            "name": "baseline_rf",
            "macro_f1": 0.85,
            "training_energy_kwh": 0.005,
            "training_co2_kg": 0.0035,
        },
        {
            "name": "efficient_dt",
            "macro_f1": 0.86,
            "training_energy_kwh": 0.001,
            "training_co2_kg": 0.0007,
        },
    ]

    selector_res = ModelSelector().select(
        model_results=candidates,
        baseline_name="baseline_rf",
        quality_metric="macro_f1",
        quality_min=0.84,
        energy_metric="training_energy_kwh",
        task_type="classification",
    )
    justification = selector_res["recommended"].get("justification")
    assert isinstance(justification, str)
    assert len(justification) > 20
    assert "efficient_dt" in justification


def test_pareto_frontier_correct() -> None:
    """Manually verified Pareto optimality for 2-model dominance and non-dominance."""
    # Case A: model_best strictly dominates model_worse (higher quality AND lower energy)
    two_models_dominated = [
        {"name": "model_best", "macro_f1": 0.90, "training_energy_kwh": 0.001},
        {"name": "model_worse", "macro_f1": 0.80, "training_energy_kwh": 0.005},
    ]
    frontier_a = ParetoAnalyzer().compute_pareto_frontier(
        two_models_dominated,
        objectives=["macro_f1", "training_energy_kwh"],
        minimize=[False, True],
    )
    assert [m["name"] for m in frontier_a] == ["model_best"]

    # Case B: Trade-off pair — neither dominates the other
    two_models_tradeoff = [
        {"name": "high_quality", "macro_f1": 0.92, "training_energy_kwh": 0.010},
        {"name": "low_energy", "macro_f1": 0.85, "training_energy_kwh": 0.001},
    ]
    frontier_b = ParetoAnalyzer().compute_pareto_frontier(
        two_models_tradeoff,
        objectives=["macro_f1", "training_energy_kwh"],
        minimize=[False, True],
    )
    assert {m["name"] for m in frontier_b} == {"high_quality", "low_energy"}

    # Verify ModelSelector agrees on the same Pareto frontier
    sel_b = ModelSelector().select(
        model_results=[
            {**m, "training_co2_kg": m["training_energy_kwh"] * 0.7}
            for m in two_models_tradeoff
        ],
        baseline_name="high_quality",
        quality_metric="macro_f1",
        quality_min=0.80,
        energy_metric="training_energy_kwh",
        task_type="classification",
    )
    assert set(sel_b["pareto_optimal"]) == {"high_quality", "low_energy"}
