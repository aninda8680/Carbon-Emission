"""
ModelSelector: Recommends the best model from a list of evaluated models based on
quality, energy, and carbon constraints using Pareto optimality analysis.
"""

from __future__ import annotations

import math
from typing import Any


class ModelSelector:
    """Selects the recommended model from a list of model evaluation results.

    Applies Pareto optimality analysis across quality and energy dimensions,
    enforcing a minimum quality constraint before ranking by balanced score.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(
        self,
        model_results: list[dict],
        baseline_name: str,
        quality_metric: str,
        quality_min: float,
        energy_metric: str,
        task_type: str,
        critical_class_recall_constraints: dict[str, float] | None = None,
    ) -> dict:
        """Select the recommended model from evaluated results.

        Parameters
        ----------
        model_results:
            List of dicts, each containing at minimum:
            ``name``, ``macro_f1`` or ``r2``, ``training_energy_kwh``,
            ``inference_energy_kwh``, ``training_co2_kg``,
            ``training_duration_seconds``, ``memory_mb``.
        baseline_name:
            The name of the baseline model (fallback if no optimised model
            passes quality gate).
        quality_metric:
            Metric key used to judge quality (e.g. ``"macro_f1"`` or ``"r2"``).
        quality_min:
            Minimum acceptable value for *quality_metric*.
        energy_metric:
            Metric key used to judge energy (e.g. ``"training_energy_kwh"``).
        task_type:
            ``"classification"`` or ``"regression"``.
        critical_class_recall_constraints:
            Optional mapping of ``{class_label: min_recall}`` (for
            classification tasks). Models failing any per-class recall floor
            are flagged as violations and excluded from recommendation.

        Returns
        -------
        dict with keys:
            highest_quality, lowest_energy, lowest_carbon, best_balanced,
            recommended, pareto_optimal, quality_constraint_violations,
            critical_class_recall_violations.
        """
        if not model_results:
            raise ValueError("model_results must not be empty")

        def _name(m: dict) -> str:
            return str(m.get("name") or m.get("model_name") or "unknown")

        # ---- quality / energy helpers --------------------------------
        def _quality(m: dict) -> float:
            val = m.get(quality_metric)
            if val is None:
                val = m.get("macro_f1") if m.get("macro_f1") is not None else m.get("r2")
            return float(val) if val is not None else 0.0

        def _energy(m: dict) -> float:
            val = m.get(energy_metric)
            if val is None:
                val = m.get("training_energy_kwh")
            return float(val) if val is not None else 0.0

        def _carbon(m: dict) -> float:
            val = m.get("training_co2_kg")
            return float(val) if val is not None else 0.0

        def _recall_violation_reasons(m: dict) -> list[str]:
            if not critical_class_recall_constraints:
                return []
            pcr = m.get("per_class_recall")
            if pcr is None and isinstance(m.get("metrics"), dict):
                pcr = m["metrics"].get("per_class_recall")
            if not isinstance(pcr, dict):
                return [
                    "missing per_class_recall for critical_class_recall_constraints"
                ]
            reasons: list[str] = []
            for cls_name, min_rec in critical_class_recall_constraints.items():
                if cls_name in pcr:
                    val = float(pcr[cls_name])
                elif str(cls_name) in pcr:
                    val = float(pcr[str(cls_name)])
                else:
                    val = 0.0
                if val < float(min_rec):
                    reasons.append(
                        f"class '{cls_name}' recall {val:.4f} < {float(min_rec):.4f}"
                    )
            return reasons

        # ---- violations -------------------------------------------
        violations: list[str] = []
        recall_violations: dict[str, list[str]] = {}
        for m in model_results:
            m_name = _name(m)
            failed = _quality(m) < quality_min
            rec_reasons = _recall_violation_reasons(m)
            if rec_reasons:
                recall_violations[m_name] = rec_reasons
                failed = True
            if failed and m_name not in violations:
                violations.append(m_name)

        # ---- single-dimension winners ------------------------------
        highest_quality_model = max(model_results, key=_quality)
        lowest_energy_model = min(model_results, key=_energy)
        lowest_carbon_model = min(model_results, key=_carbon)

        def _summarise(m: dict) -> dict:
            m_name = _name(m)
            return {
                "name": m_name,
                "metrics": {k: v for k, v in m.items() if k not in ("name", "model_name")},
            }

        # ---- Pareto frontier (maximise quality, minimise energy) ---
        pareto_names = self._pareto_optimal(model_results, _quality, _energy, _name)

        # ---- best balanced (Pareto + normalised score) -------------
        pareto_models = [m for m in model_results if _name(m) in pareto_names]
        best_balanced_model = self._best_balanced(pareto_models, _quality, _energy)

        # ---- recommended ------------------------------------------
        # Must pass quality gate AND critical-class recall gate
        passing = [m for m in model_results if _name(m) not in violations]

        if passing:
            # Recompute Pareto over passing models only
            passing_pareto_names = self._pareto_optimal(passing, _quality, _energy, _name)
            passing_pareto = [m for m in passing if _name(m) in passing_pareto_names]
            recommended_model = self._best_balanced(passing_pareto, _quality, _energy)
            rec_name = _name(recommended_model)
            justification = (
                f"'{rec_name}' achieves the best normalised "
                f"balance of {quality_metric} ({_quality(recommended_model):.4f}) "
                f"and {energy_metric} ({_energy(recommended_model):.6f} kWh) "
                f"among Pareto-optimal models that pass the quality gate "
                f"(minimum {quality_metric} = {quality_min}"
                + (
                    f" and critical-class recall constraints"
                    if critical_class_recall_constraints
                    else ""
                )
                + ")."
            )
        else:
            # Fall back to baseline
            baseline_candidates = [m for m in model_results if _name(m) == baseline_name]
            if baseline_candidates:
                recommended_model = baseline_candidates[0]
            else:
                # Last resort: pick highest quality even if below gate
                recommended_model = highest_quality_model
            rec_name = _name(recommended_model)
            constraint_desc = f"minimum {quality_metric} threshold of {quality_min}"
            if critical_class_recall_constraints:
                constraint_desc += " and critical-class recall constraints"
            justification = (
                f"No optimised model met the {constraint_desc}. "
                f"Falling back to '{rec_name}' "
                f"(baseline). Consider relaxing constraints or providing more "
                f"training data / better features."
            )

        recommended_entry = _summarise(recommended_model)
        recommended_entry["justification"] = justification

        return {
            "highest_quality": _summarise(highest_quality_model),
            "lowest_energy": _summarise(lowest_energy_model),
            "lowest_carbon": _summarise(lowest_carbon_model),
            "best_balanced": _summarise(best_balanced_model),
            "recommended": recommended_entry,
            "pareto_optimal": list(pareto_names),
            "quality_constraint_violations": violations,
            "critical_class_recall_violations": recall_violations,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pareto_optimal(
        models: list[dict],
        quality_fn: Any,
        energy_fn: Any,
        name_fn: Any = None,
    ) -> set[str]:
        """Return names of Pareto-optimal models (max quality, min energy)."""
        if name_fn is None:
            name_fn = lambda m: str(m.get("name") or m.get("model_name") or "unknown")
        pareto: set[str] = set()
        for candidate in models:
            q_c = quality_fn(candidate)
            e_c = energy_fn(candidate)
            c_name = name_fn(candidate)
            dominated = False
            for other in models:
                if name_fn(other) == c_name:
                    continue
                q_o = quality_fn(other)
                e_o = energy_fn(other)
                # other dominates candidate if other is at least as good on
                # both dimensions and strictly better on at least one
                if q_o >= q_c and e_o <= e_c and (q_o > q_c or e_o < e_c):
                    dominated = True
                    break
            if not dominated:
                pareto.add(c_name)
        return pareto

    @staticmethod
    def _best_balanced(
        models: list[dict],
        quality_fn: Any,
        energy_fn: Any,
    ) -> dict:
        """Pick the model with the best normalised (quality, −energy) score."""
        if not models:
            raise ValueError("Cannot select best balanced from empty list")
        if len(models) == 1:
            return models[0]

        qualities = [quality_fn(m) for m in models]
        energies = [energy_fn(m) for m in models]

        q_min, q_max = min(qualities), max(qualities)
        e_min, e_max = min(energies), max(energies)

        def _normalise_quality(q: float) -> float:
            if math.isclose(q_max, q_min):
                return 1.0
            return (q - q_min) / (q_max - q_min)

        def _normalise_energy(e: float) -> float:
            # Lower energy → higher score (invert)
            if math.isclose(e_max, e_min):
                return 1.0
            return 1.0 - (e - e_min) / (e_max - e_min)

        def _score(m: dict) -> float:
            return _normalise_quality(quality_fn(m)) + _normalise_energy(energy_fn(m))

        return max(models, key=_score)
