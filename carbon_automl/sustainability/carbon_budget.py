"""
carbon_budget.py — Carbon-budget-constrained model selection and feasibility
analysis.

Design constraints
------------------
* Never silently relax a carbon budget or a predictive quality floor.
* When no candidate satisfies both constraints simultaneously, clearly report
  the conflict with concrete numbers so the user can make an informed choice.
* Use dataclasses for structured internal representations and return clean dicts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .carbon_calculator import CarbonCalculator


@dataclass
class BudgetAnalysisResult:
    """Structured result for budget-constrained model selection."""

    budget_kg: float
    within_budget: List[Dict[str, Any]]
    over_budget: List[Dict[str, Any]]
    best_within_budget: Optional[Dict[str, Any]]
    conflict_explanation: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "budget_kg": self.budget_kg,
            "within_budget": self.within_budget,
            "over_budget": self.over_budget,
            "best_within_budget": self.best_within_budget,
            "conflict_explanation": self.conflict_explanation,
        }


def _is_valid_number(value: Any) -> bool:
    if value is None:
        return False
    try:
        f = float(value)
        return not (math.isnan(f) or math.isinf(f))
    except (TypeError, ValueError):
        return False


class CarbonBudgetOptimizer:
    """Evaluate candidate models against a hard CO₂e budget.

    Parameters
    ----------
    budget_kg:
        Maximum allowed carbon emissions in kg CO₂e.
    carbon_calculator:
        Calculator used to convert energy (kWh) to CO₂e if a candidate entry
        provides energy without precomputed CO₂e.
    """

    def __init__(
        self,
        budget_kg: float,
        carbon_calculator: Optional[CarbonCalculator] = None,
    ) -> None:
        if budget_kg < 0:
            raise ValueError("budget_kg must be non-negative.")
        self.budget_kg = float(budget_kg)
        self.carbon_calculator = carbon_calculator or CarbonCalculator()

    # ------------------------------------------------------------------
    # Internal extraction helpers
    # ------------------------------------------------------------------

    def _extract_co2_kg(self, candidate: Dict[str, Any]) -> Optional[float]:
        """Extract pipeline/training CO₂e (kg), falling back to energy conversion."""
        for key in (
            "pipeline_co2_kg",
            "total_co2_kg",
            "training_co2_kg",
            "co2_kg",
        ):
            if _is_valid_number(candidate.get(key)):
                return float(candidate[key])

        for energy_key in (
            "pipeline_energy_kwh",
            "total_energy_kwh",
            "training_energy_kwh",
            "energy_kwh",
        ):
            if _is_valid_number(candidate.get(energy_key)):
                return float(self.carbon_calculator.compute_co2(float(candidate[energy_key])))

        return None

    @staticmethod
    def _infer_quality_metric(
        candidates: List[Dict[str, Any]],
        preferred: Optional[str] = None,
    ) -> str:
        if preferred:
            return preferred
        for key in ("macro_f1", "accuracy", "balanced_accuracy", "r2", "quality"):
            if any(_is_valid_number(c.get(key)) for c in candidates):
                return key
        return "macro_f1"

    @staticmethod
    def _extract_quality(candidate: Dict[str, Any], quality_metric: str) -> Optional[float]:
        if _is_valid_number(candidate.get(quality_metric)):
            return float(candidate[quality_metric])
        for fallback in ("macro_f1", "r2", "accuracy", "quality"):
            if _is_valid_number(candidate.get(fallback)):
                return float(candidate[fallback])
        return None

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def find_models_within_budget(
        self,
        candidate_results: List[Dict[str, Any]],
        quality_min: Optional[float] = None,
        quality_metric: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Partition candidates by carbon budget and select the best compliant model.

        If no model satisfies both ``budget_kg`` and ``quality_min`` (when
        specified), ``best_within_budget`` is ``None`` and
        ``conflict_explanation`` describes the exact trade-off instead of
        silently relaxing either constraint.
        """
        if not candidate_results:
            return BudgetAnalysisResult(
                budget_kg=self.budget_kg,
                within_budget=[],
                over_budget=[],
                best_within_budget=None,
                conflict_explanation="No candidate models were provided for budget evaluation.",
            ).to_dict()

        q_metric = self._infer_quality_metric(candidate_results, quality_metric)
        within_budget: List[Dict[str, Any]] = []
        over_budget: List[Dict[str, Any]] = []

        for cand in candidate_results:
            entry = dict(cand)
            co2_kg = self._extract_co2_kg(entry)
            entry["_evaluated_co2_kg"] = co2_kg

            if co2_kg is None:
                entry["budget_status"] = "unknown_emissions"
                over_budget.append(entry)
            elif co2_kg <= self.budget_kg:
                entry["budget_status"] = "within_budget"
                entry["budget_headroom_kg"] = self.budget_kg - co2_kg
                within_budget.append(entry)
            else:
                entry["budget_status"] = "over_budget"
                entry["budget_excess_kg"] = co2_kg - self.budget_kg
                over_budget.append(entry)

        # Case 1: Nothing fits within the carbon budget at all.
        if not within_budget:
            known_co2 = [
                (c.get("name", "unknown"), c["_evaluated_co2_kg"])
                for c in over_budget
                if c.get("_evaluated_co2_kg") is not None
            ]
            if known_co2:
                cheapest_name, cheapest_co2 = min(known_co2, key=lambda item: item[1])
                explanation = (
                    f"Carbon budget conflict: 0 of {len(candidate_results)} candidate model(s) "
                    f"fit within the budget of {self.budget_kg:.6g} kg CO₂e. "
                    f"Lowest-emission candidate '{cheapest_name}' requires "
                    f"{cheapest_co2:.6g} kg CO₂e (exceeds budget by "
                    f"{cheapest_co2 - self.budget_kg:.6g} kg CO₂e). "
                    "Budget was NOT silently relaxed."
                )
            else:
                explanation = (
                    f"Carbon budget conflict: no candidate model had measurable CO₂e "
                    f"emissions to verify compliance with budget {self.budget_kg:.6g} kg CO₂e."
                )
            return BudgetAnalysisResult(
                budget_kg=self.budget_kg,
                within_budget=within_budget,
                over_budget=over_budget,
                best_within_budget=None,
                conflict_explanation=explanation,
            ).to_dict()

        # Filter within-budget candidates by quality constraint if present.
        feasible: List[Dict[str, Any]] = []
        failed_quality_within_budget: List[Dict[str, Any]] = []
        for cand in within_budget:
            q_val = self._extract_quality(cand, q_metric)
            explicit_pass = cand.get("passes_quality", cand.get("quality_constraint_passed", True))
            passes_floor = True if quality_min is None else (q_val is not None and q_val >= quality_min)
            if bool(explicit_pass) and passes_floor:
                feasible.append(cand)
            else:
                failed_quality_within_budget.append(cand)

        # Case 2: Some models fit the budget, but none satisfy quality constraints.
        if not feasible:
            within_names = [c.get("name", "unknown") for c in failed_quality_within_budget]
            over_quality_pass = [
                c for c in over_budget
                if (
                    bool(c.get("passes_quality", c.get("quality_constraint_passed", True)))
                    and (quality_min is None or (self._extract_quality(c, q_metric) or float("-inf")) >= quality_min)
                )
            ]
            if over_quality_pass:
                cheapest_quality = min(
                    over_quality_pass,
                    key=lambda c: c["_evaluated_co2_kg"] if c["_evaluated_co2_kg"] is not None else float("inf"),
                )
                extra = (
                    f"Lowest-carbon model meeting quality is '{cheapest_quality.get('name')}' "
                    f"at {cheapest_quality.get('_evaluated_co2_kg', 0.0):.6g} kg CO₂e "
                    f"(minimum budget needed: {cheapest_quality.get('_evaluated_co2_kg', 0.0):.6g} kg CO₂e)."
                )
            else:
                extra = "No candidate model in the experiment achieved the required quality floor."

            q_desc = f"{q_metric} >= {quality_min}" if quality_min is not None else "quality gate"
            explanation = (
                f"Budget vs. quality conflict: {len(within_budget)} model(s) "
                f"({within_names}) fit within the carbon budget ({self.budget_kg:.6g} kg CO₂e), "
                f"but ALL failed the quality constraint ({q_desc}). {extra} "
                "Neither constraint was silently relaxed."
            )
            return BudgetAnalysisResult(
                budget_kg=self.budget_kg,
                within_budget=within_budget,
                over_budget=over_budget,
                best_within_budget=None,
                conflict_explanation=explanation,
            ).to_dict()

        # Case 3: At least one candidate satisfies both budget and quality.
        best = max(
            feasible,
            key=lambda c: (
                self._extract_quality(c, q_metric) if self._extract_quality(c, q_metric) is not None else float("-inf"),
                -(c["_evaluated_co2_kg"] if c["_evaluated_co2_kg"] is not None else float("inf")),
            ),
        )

        return BudgetAnalysisResult(
            budget_kg=self.budget_kg,
            within_budget=within_budget,
            over_budget=over_budget,
            best_within_budget=best,
            conflict_explanation=None,
        ).to_dict()

    def estimate_budget_for_quality(
        self,
        target_quality: float,
        candidate_results: List[Dict[str, Any]],
        quality_metric: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Estimate the minimum CO₂e budget required to achieve *target_quality*.

        Returns
        -------
        dict with keys:
            target_quality, quality_metric, feasible, minimum_budget_kg,
            recommended_model, budget_sufficient_for_configured, explanation
        """
        if not candidate_results:
            return {
                "target_quality": target_quality,
                "quality_metric": quality_metric or "macro_f1",
                "feasible": False,
                "minimum_budget_kg": None,
                "recommended_model": None,
                "budget_sufficient_for_configured": False,
                "explanation": "No candidate results provided.",
            }

        q_metric = self._infer_quality_metric(candidate_results, quality_metric)
        qualifying: List[tuple[Dict[str, Any], float, float]] = []

        for cand in candidate_results:
            q_val = self._extract_quality(cand, q_metric)
            co2_kg = self._extract_co2_kg(cand)
            if q_val is not None and q_val >= target_quality and co2_kg is not None:
                qualifying.append((dict(cand), q_val, co2_kg))

        if not qualifying:
            best_q_cand = max(
                candidate_results,
                key=lambda c: self._extract_quality(c, q_metric) or float("-inf"),
            )
            best_q = self._extract_quality(best_q_cand, q_metric)
            return {
                "target_quality": target_quality,
                "quality_metric": q_metric,
                "feasible": False,
                "minimum_budget_kg": None,
                "recommended_model": None,
                "budget_sufficient_for_configured": False,
                "explanation": (
                    f"No candidate achieved target quality {q_metric} >= {target_quality:.4f}. "
                    f"Highest observed was '{best_q_cand.get('name', 'unknown')}' at "
                    f"{best_q if best_q is not None else 'N/A'}."
                ),
            }

        # Pick the qualifying model with lowest CO₂e (tie-break on highest quality).
        best_cand, best_q, min_budget = min(qualifying, key=lambda item: (item[2], -item[1]))
        sufficient = min_budget <= self.budget_kg

        return {
            "target_quality": target_quality,
            "quality_metric": q_metric,
            "feasible": True,
            "minimum_budget_kg": min_budget,
            "recommended_model": best_cand,
            "budget_sufficient_for_configured": sufficient,
            "explanation": (
                f"Minimum CO₂e budget to achieve {q_metric} >= {target_quality:.4f} is "
                f"{min_budget:.6g} kg CO₂e using '{best_cand.get('name', 'unknown')}' "
                f"({q_metric}={best_q:.4f}). Configured budget ({self.budget_kg:.6g} kg CO₂e) "
                f"is {'sufficient' if sufficient else 'INSUFFICIENT'}."
            ),
        }
