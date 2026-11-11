"""
pareto.py — Multi-objective Pareto frontier analysis and quality-guarded
model recommendation.

Design constraints
------------------
* Models that violate ``quality_min`` MUST NEVER be recommended, even if they
  consume the lowest energy.
* If no candidate satisfies the quality constraint, the baseline model is
  recommended with a clear explanation.
* Uses dataclasses where structured metadata is helpful while returning the
  exact dict/list contracts expected by downstream consumers.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ParetoRecommendation:
    """Structured representation of a quality-guarded model recommendation."""

    recommended_model: Dict[str, Any]
    reason: str
    tradeoffs: Dict[str, Any] = field(default_factory=dict)
    pareto_optimal: List[Dict[str, Any]] = field(default_factory=list)
    alternatives: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommended_model": self.recommended_model,
            "recommended_model_name": self.recommended_model.get("name", "unknown"),
            "reason": self.reason,
            "justification": self.reason,
            "tradeoffs": self.tradeoffs,
            "pareto_optimal": self.pareto_optimal,
            "alternatives": self.alternatives,
        }


def _is_valid_number(value: Any) -> bool:
    if value is None:
        return False
    try:
        f = float(value)
        return not (math.isnan(f) or math.isinf(f))
    except (TypeError, ValueError):
        return False


class ParetoAnalyzer:
    """Compute Pareto frontiers, normalize/rank models, and recommend models
    subject to strict predictive-quality floors.
    """

    # ------------------------------------------------------------------
    # Pareto frontier
    # ------------------------------------------------------------------

    def compute_pareto_frontier(
        self,
        models: List[Dict[str, Any]],
        objectives: List[str],
        minimize: List[bool],
    ) -> List[Dict[str, Any]]:
        """Identify non-dominated (Pareto-optimal) models.

        Parameters
        ----------
        models:
            List of model dicts, each containing ``"name"`` and objective keys.
        objectives:
            Metric names to optimise, e.g. ``["macro_f1", "training_energy_kwh"]``.
        minimize:
            Boolean flags aligned with *objectives*: ``True`` means lower is
            better, ``False`` means higher is better.

        Returns
        -------
        list[dict]
            Subset of *models* on the Pareto frontier (in original order).
        """
        if len(objectives) != len(minimize):
            raise ValueError("objectives and minimize must have the same length.")
        if not models or not objectives:
            return []

        valid_models = [
            m
            for m in models
            if all(_is_valid_number(m.get(obj)) for obj in objectives)
        ]
        if not valid_models:
            return []

        frontier: List[Dict[str, Any]] = []
        for idx, candidate in enumerate(valid_models):
            dominated = False
            for jdx, other in enumerate(valid_models):
                if idx == jdx:
                    continue
                if self._dominates(other, candidate, objectives, minimize):
                    dominated = True
                    break
            if not dominated:
                frontier.append(dict(candidate))

        return frontier

    @staticmethod
    def _dominates(
        a: Dict[str, Any],
        b: Dict[str, Any],
        objectives: List[str],
        minimize: List[bool],
    ) -> bool:
        """Return True if *a* Pareto-dominates *b*."""
        strictly_better_in_one = False
        for obj, is_min in zip(objectives, minimize):
            va = float(a[obj])
            vb = float(b[obj])
            if is_min:
                if va > vb:
                    return False
                if va < vb:
                    strictly_better_in_one = True
            else:
                if va < vb:
                    return False
                if va > vb:
                    strictly_better_in_one = True
        return strictly_better_in_one

    # ------------------------------------------------------------------
    # Normalization & ranking
    # ------------------------------------------------------------------

    def rank_models(
        self,
        models: List[Dict[str, Any]],
        quality_metric: str,
        energy_metric: str,
        quality_weight: float = 0.6,
    ) -> List[Dict[str, Any]]:
        """Rank models by weighted normalized quality and energy.

        Both metrics are min-max normalized to ``[0, 1]``:
        * ``normalized_quality``: 1.0 = highest quality, 0.0 = lowest quality.
        * ``normalized_energy``: 0.0 = lowest energy (best), 1.0 = highest energy.
        * ``combined_score = quality_weight * normalized_quality + (1 - quality_weight) * (1 - normalized_energy)``.

        Returns a list sorted descending by ``combined_score``, where each dict
        includes ``rank``, ``normalized_quality``, ``normalized_energy``, and
        ``combined_score`` alongside the original model fields.
        """
        if not 0.0 <= quality_weight <= 1.0:
            raise ValueError("quality_weight must be between 0.0 and 1.0.")
        if not models:
            return []

        energy_weight = 1.0 - quality_weight
        qual_vals = [
            float(m[quality_metric])
            for m in models
            if _is_valid_number(m.get(quality_metric))
        ]
        ener_vals = [
            float(m[energy_metric])
            for m in models
            if _is_valid_number(m.get(energy_metric))
        ]

        q_min = min(qual_vals) if qual_vals else 0.0
        q_max = max(qual_vals) if qual_vals else 0.0
        e_min = min(ener_vals) if ener_vals else 0.0
        e_max = max(ener_vals) if ener_vals else 0.0

        scored: List[Dict[str, Any]] = []
        for m in models:
            entry = dict(m)
            q_raw = m.get(quality_metric)
            e_raw = m.get(energy_metric)

            if _is_valid_number(q_raw):
                if q_max > q_min:
                    norm_q = (float(q_raw) - q_min) / (q_max - q_min)
                else:
                    norm_q = 1.0
            else:
                norm_q = 0.0

            if _is_valid_number(e_raw):
                if e_max > e_min:
                    norm_e = (float(e_raw) - e_min) / (e_max - e_min)
                else:
                    norm_e = 0.0
            else:
                # Unknown energy cannot claim lowest energy; penalize conservatively.
                norm_e = 1.0

            combined = quality_weight * norm_q + energy_weight * (1.0 - norm_e)
            entry["normalized_quality"] = float(norm_q)
            entry["normalized_energy"] = float(norm_e)
            entry["combined_score"] = float(combined)
            scored.append(entry)

        scored.sort(
            key=lambda x: (
                x["combined_score"],
                x.get(quality_metric) if _is_valid_number(x.get(quality_metric)) else float("-inf"),
            ),
            reverse=True,
        )

        for rank_idx, entry in enumerate(scored, start=1):
            entry["rank"] = rank_idx

        return scored

    # ------------------------------------------------------------------
    # Quality-guarded model recommendation
    # ------------------------------------------------------------------

    def recommend_model(
        self,
        models: List[Dict[str, Any]],
        quality_metric: str,
        energy_metric: str,
        quality_min: float,
        baseline_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Recommend a model while strictly enforcing ``quality_min``.

        A model with ``model[quality_metric] < quality_min`` MUST NOT be
        recommended even if it has the lowest energy.  If no candidate model
        satisfies ``quality_min``, the baseline is recommended with a clear
        explanation.
        """
        baseline_entry = dict(baseline_metrics)
        baseline_entry.setdefault("name", "baseline")
        baseline_name = baseline_entry["name"]

        # Identify candidates passing the quality floor.
        passing_models: List[Dict[str, Any]] = []
        rejected_models: List[Dict[str, Any]] = []

        for m in models:
            q_val = m.get(quality_metric)
            if _is_valid_number(q_val) and float(q_val) >= quality_min:
                passing_models.append(dict(m))
            else:
                rejected_models.append(dict(m))

        # Check if any non-baseline candidate passes (or if models only had baseline).
        optimized_passing = [
            m for m in passing_models if m.get("name") != baseline_name
        ]

        all_pareto = self.compute_pareto_frontier(
            models,
            objectives=[quality_metric, energy_metric],
            minimize=[False, True],
        )

        if not passing_models or (not optimized_passing and any(m.get("name") == baseline_name for m in models)):
            # Either zero models passed quality_min, or only the baseline itself passed.
            rejected_names = [m.get("name", "unknown") for m in rejected_models]
            if not passing_models:
                reason = (
                    f"No candidate model satisfied the minimum quality constraint "
                    f"({quality_metric} >= {quality_min:.4f}). "
                    f"Rejected models: {rejected_names}. "
                    f"Recommending baseline '{baseline_name}' as the safe fallback."
                )
            else:
                reason = (
                    f"No optimized candidate satisfied the minimum quality constraint "
                    f"({quality_metric} >= {quality_min:.4f}); only baseline "
                    f"'{baseline_name}' met the threshold. Recommending baseline."
                )

            rec = ParetoRecommendation(
                recommended_model=baseline_entry,
                reason=reason,
                tradeoffs={
                    "quality_floor": quality_min,
                    "quality_metric": quality_metric,
                    "energy_metric": energy_metric,
                    "rejected_for_quality_violation": rejected_names,
                    "baseline_quality": baseline_entry.get(quality_metric),
                    "baseline_energy": baseline_entry.get(energy_metric),
                },
                pareto_optimal=all_pareto,
                alternatives=[],
            )
            return rec.to_dict()

        # Among models passing quality_min, compute Pareto frontier & rank.
        passing_pareto = self.compute_pareto_frontier(
            passing_models,
            objectives=[quality_metric, energy_metric],
            minimize=[False, True],
        )
        pool_for_ranking = passing_pareto if passing_pareto else passing_models
        ranked = self.rank_models(
            pool_for_ranking,
            quality_metric=quality_metric,
            energy_metric=energy_metric,
        )

        winner = ranked[0]
        alternatives = ranked[1:]

        winner_q = winner.get(quality_metric)
        winner_e = winner.get(energy_metric)
        base_q = baseline_entry.get(quality_metric)
        base_e = baseline_entry.get(energy_metric)

        q_delta = (
            float(winner_q) - float(base_q)
            if _is_valid_number(winner_q) and _is_valid_number(base_q)
            else None
        )
        e_ratio = (
            float(winner_e) / float(base_e)
            if _is_valid_number(winner_e) and _is_valid_number(base_e) and float(base_e) > 0
            else None
        )

        rejected_names = [m.get("name", "unknown") for m in rejected_models]
        reason_parts = [
            f"Recommended '{winner.get('name', 'unknown')}' (Pareto-optimal among "
            f"{len(passing_models)} quality-compliant models with {quality_metric}="
            f"{float(winner_q):.4f} >= {quality_min:.4f}, "
            f"combined_score={winner['combined_score']:.4f})."
        ]
        if _is_valid_number(winner_e):
            reason_parts.append(f"Estimated {energy_metric}={float(winner_e):.6g}.")
        if q_delta is not None:
            reason_parts.append(
                f"Quality delta vs baseline '{baseline_name}': {q_delta:+.4f}."
            )
        if rejected_names:
            reason_parts.append(
                f"Excluded {len(rejected_names)} model(s) for violating quality floor: "
                f"{rejected_names}."
            )

        rec = ParetoRecommendation(
            recommended_model=winner,
            reason=" ".join(reason_parts),
            tradeoffs={
                "quality_floor": quality_min,
                "quality_metric": quality_metric,
                "energy_metric": energy_metric,
                "quality_delta_vs_baseline": q_delta,
                "energy_ratio_vs_baseline": e_ratio,
                "rejected_for_quality_violation": rejected_names,
            },
            pareto_optimal=passing_pareto,
            alternatives=alternatives,
        )
        return rec.to_dict()

    # ------------------------------------------------------------------
    # Efficiency metrics (replacing opaque composite CEGI ratios)
    # ------------------------------------------------------------------

    def compute_efficiency_metrics(
        self,
        models: List[Dict[str, Any]] | Dict[str, Any],
        baseline_metrics: Optional[Dict[str, Any]] = None,
        quality_metric: str = "macro_f1",
        energy_metric: str = "training_energy_kwh",
        carbon_metric: str = "training_co2_kg",
    ) -> List[Dict[str, Any]] | Dict[str, Any]:
        """Compute transparent efficiency metrics without hiding raw measurements.

        Replaces opaque composite CEGI scores with explicit components:
        - ``quality_score`` (raw predictive quality)
        - ``energy_efficiency`` (quality per kWh) and ``quality_per_joule``
        - ``carbon_efficiency`` (quality per kg CO₂e)
        - ``quality_retention_pct`` (% of baseline quality retained)
        - ``energy_reduction_pct`` (% energy change vs baseline; positive = reduction)
        - ``carbon_reduction_pct`` (% CO₂e change vs baseline; positive = reduction)
        - ``raw_measurements`` and ``baseline_measurements`` kept visible.
        """
        single_input = isinstance(models, dict)
        model_list = [models] if single_input else list(models)

        base = baseline_metrics or {}
        base_q = float(base[quality_metric]) if _is_valid_number(base.get(quality_metric)) else None
        base_e = float(base[energy_metric]) if _is_valid_number(base.get(energy_metric)) else None
        base_c = float(base[carbon_metric]) if _is_valid_number(base.get(carbon_metric)) else None

        results: List[Dict[str, Any]] = []
        for m in model_list:
            q_val = float(m[quality_metric]) if _is_valid_number(m.get(quality_metric)) else None
            e_val = float(m[energy_metric]) if _is_valid_number(m.get(energy_metric)) else None
            c_val = float(m[carbon_metric]) if _is_valid_number(m.get(carbon_metric)) else None
            e_joules = e_val * 3_600_000.0 if e_val is not None else None

            energy_eff = (q_val / e_val) if (q_val is not None and e_val is not None and e_val > 0) else None
            quality_per_j = (q_val / e_joules) if (q_val is not None and e_joules is not None and e_joules > 0) else None
            carbon_eff = (q_val / c_val) if (q_val is not None and c_val is not None and c_val > 0) else None

            q_ret_pct = (
                (q_val / base_q) * 100.0
                if (q_val is not None and base_q is not None and abs(base_q) > 1e-12)
                else None
            )
            e_red_pct = (
                ((base_e - e_val) / base_e) * 100.0
                if (e_val is not None and base_e is not None and base_e > 0)
                else None
            )
            c_red_pct = (
                ((base_c - c_val) / base_c) * 100.0
                if (c_val is not None and base_c is not None and base_c > 0)
                else None
            )

            results.append(
                {
                    "name": m.get("name", "unknown"),
                    "quality_metric": quality_metric,
                    "quality_score": q_val,
                    "energy_kwh": e_val,
                    "energy_joules": e_joules,
                    "co2_kg": c_val,
                    "energy_efficiency": energy_eff,
                    "quality_per_joule": quality_per_j,
                    "carbon_efficiency": carbon_eff,
                    "quality_retention_pct": q_ret_pct,
                    "energy_reduction_pct": e_red_pct,
                    "carbon_reduction_pct": c_red_pct,
                    "raw_measurements": {
                        quality_metric: q_val,
                        energy_metric: e_val,
                        "energy_joules": e_joules,
                        carbon_metric: c_val,
                    },
                    "baseline_measurements": {
                        "name": base.get("name", "baseline") if base else None,
                        quality_metric: base_q,
                        energy_metric: base_e,
                        "energy_joules": base_e * 3_600_000.0 if base_e is not None else None,
                        carbon_metric: base_c,
                    },
                    "note": (
                        "Efficiency metrics are reported alongside raw measurements "
                        "(replacing single opaque CEGI ratios). Negative reduction % "
                        "indicates higher cost than baseline. Do not claim reductions "
                        "without statistical significance."
                    ),
                }
            )

        return results[0] if single_input else results

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot_pareto_frontier(
        self,
        models: List[Dict[str, Any]],
        quality_metric: str,
        energy_metric: str,
        output_path: Optional[str] = None,
    ) -> None:
        """Render a scatter plot of energy vs quality with the Pareto frontier highlighted."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        valid = [
            m
            for m in models
            if _is_valid_number(m.get(quality_metric))
            and _is_valid_number(m.get(energy_metric))
        ]
        if not valid:
            logger.warning("No valid models with both %s and %s to plot.", quality_metric, energy_metric)
            return

        pareto = self.compute_pareto_frontier(
            valid,
            objectives=[quality_metric, energy_metric],
            minimize=[False, True],
        )
        pareto_names = {m.get("name") for m in pareto}

        fig, ax = plt.subplots(figsize=(8, 5))

        non_pareto = [m for m in valid if m.get("name") not in pareto_names]
        if non_pareto:
            ax.scatter(
                [float(m[energy_metric]) for m in non_pareto],
                [float(m[quality_metric]) for m in non_pareto],
                label="Dominated models",
                alpha=0.65,
                s=70,
            )

        if pareto:
            pareto_sorted = sorted(pareto, key=lambda m: float(m[energy_metric]))
            px = [float(m[energy_metric]) for m in pareto_sorted]
            py = [float(m[quality_metric]) for m in pareto_sorted]
            ax.scatter(px, py, label="Pareto frontier", s=100, zorder=5)
            ax.plot(px, py, linestyle="--", linewidth=1.5, zorder=4)

        for m in valid:
            ax.annotate(
                str(m.get("name", "")),
                (float(m[energy_metric]), float(m[quality_metric])),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

        ax.set_xlabel(f"{energy_metric} (lower is better)")
        ax.set_ylabel(f"{quality_metric} (higher is better)")
        ax.set_title("Quality vs. Energy Pareto Frontier")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend()
        fig.tight_layout()

        if output_path is not None:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=150)

        plt.close(fig)
