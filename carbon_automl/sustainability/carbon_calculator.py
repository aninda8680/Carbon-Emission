"""
carbon_calculator.py — CO₂e computation, inference emissions, and reporting.

Design constraints
------------------
* All carbon-intensity figures are user-supplied assumptions, not live data.
* The module CLEARLY distinguishes measured / estimated / calculated values.
* Energy is always reported in BOTH kWh and Joules (1 kWh = 3_600_000 J).
* Inference energy is measured via repeated runs; the result carries
  explicit uncertainty because individual runs are extremely short.
* The ``amortize_training`` helper allows break-even analysis only —
  it does NOT claim environmental benefit without evidence.
"""

from __future__ import annotations

import logging
import textwrap
from typing import Any, Dict, List, Optional

import numpy as np

from .energy_monitor import (
    EnergyMeasurement,
    EnergyMonitor,
    JOULES_PER_KWH,
    get_hardware_specs,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_JOULES_PER_KWH = JOULES_PER_KWH


# ---------------------------------------------------------------------------
# CarbonCalculator
# ---------------------------------------------------------------------------


class CarbonCalculator:
    """Compute and report CO₂-equivalent emissions from energy measurements.

    Parameters
    ----------
    carbon_intensity_kg_per_kwh:
        Grid carbon intensity to use when converting kWh → kg CO₂e.
        Defaults to 0.7 (India national average).  This is a user-supplied
        assumption; the class surfaces it prominently in every report.
    intensity_source:
        Human-readable label for where the intensity figure came from,
        e.g. ``"user_configured"``, ``"India national average"``.
    """

    JOULES_PER_KWH = _JOULES_PER_KWH

    def __init__(
        self,
        carbon_intensity_kg_per_kwh: float = 0.7,
        intensity_source: str = "user_configured",
    ) -> None:
        if carbon_intensity_kg_per_kwh <= 0:
            raise ValueError("carbon_intensity_kg_per_kwh must be positive.")
        self.carbon_intensity_kg_per_kwh = carbon_intensity_kg_per_kwh
        self.intensity_source = intensity_source

    # ------------------------------------------------------------------
    # Unit conversion & core computation
    # ------------------------------------------------------------------

    @staticmethod
    def kwh_to_joules(energy_kwh: float) -> float:
        """Convert kilowatt-hours to Joules (1 kWh = 3,600,000 J)."""
        return float(energy_kwh) * _JOULES_PER_KWH

    @staticmethod
    def joules_to_kwh(energy_joules: float) -> float:
        """Convert Joules to kilowatt-hours."""
        return float(energy_joules) / _JOULES_PER_KWH

    def compute_co2(self, energy_kwh: float) -> float:
        """Convert energy (kWh) to kg CO₂e using the configured intensity."""
        return float(energy_kwh) * self.carbon_intensity_kg_per_kwh

    # ------------------------------------------------------------------
    # Inference emissions
    # ------------------------------------------------------------------

    def compute_inference_emissions(
        self,
        model: Any,
        X_inference: np.ndarray,
        n_predictions: int = 1000,
        repetitions: int = 3,
    ) -> dict:
        """Measure energy required to serve *n_predictions* from *model*.

        Returns energy in BOTH Joules and kWh, along with median/CI/outlier
        statistics, hardware metadata, and explicit uncertainty caveats.
        """
        hw_specs = get_hardware_specs()

        if not hasattr(model, "predict"):
            return {
                "energy_per_prediction_joules": None,
                "energy_per_prediction_kwh": None,
                "energy_per_1k_predictions_joules": None,
                "energy_per_1k_predictions_kwh": None,
                "co2_per_prediction_kg": None,
                "co2_per_1k_predictions_kg": None,
                "measurement_uncertainty_pct": 100.0,
                "n_repetitions": 0,
                "mean_duration_seconds": 0.0,
                "median_duration_seconds": 0.0,
                "energy_stats_joules": None,
                "outlier_count": 0,
                "hardware_specs": hw_specs,
                "method": "unavailable",
                "warning": "Model does not expose a .predict() method.",
            }

        n_rows = len(X_inference)
        if n_rows == 0:
            return {
                "energy_per_prediction_joules": None,
                "energy_per_prediction_kwh": None,
                "energy_per_1k_predictions_joules": None,
                "energy_per_1k_predictions_kwh": None,
                "co2_per_prediction_kg": None,
                "co2_per_1k_predictions_kg": None,
                "measurement_uncertainty_pct": 100.0,
                "n_repetitions": 0,
                "mean_duration_seconds": 0.0,
                "median_duration_seconds": 0.0,
                "energy_stats_joules": None,
                "outlier_count": 0,
                "hardware_specs": hw_specs,
                "method": "unavailable",
                "warning": "X_inference is empty.",
            }

        if n_rows < n_predictions:
            repeats = int(np.ceil(n_predictions / n_rows))
            X_batch = np.tile(X_inference, (repeats, 1))[:n_predictions]
        else:
            X_batch = X_inference[:n_predictions]

        monitor = EnergyMonitor(
            carbon_intensity_kg_per_kwh=self.carbon_intensity_kg_per_kwh,
        )

        def _infer() -> None:
            model.predict(X_batch)

        measurements: List[EnergyMeasurement] = monitor.measure_repeated(
            _infer, n_repeats=repetitions, phase_name="inference"
        )

        stats = EnergyMonitor.compute_measurement_stats(measurements)
        durations = [m.duration_seconds for m in measurements]
        energies = [m.energy_kwh for m in measurements if m.energy_kwh is not None]
        mean_duration = float(np.mean(durations))
        median_duration = float(np.median(durations))

        if energies:
            mean_energy_kwh = float(np.mean(energies))
            energy_per_pred_kwh = mean_energy_kwh / n_predictions
            energy_per_pred_joules = self.kwh_to_joules(energy_per_pred_kwh)
            energy_per_1k_kwh = energy_per_pred_kwh * 1_000.0
            energy_per_1k_joules = energy_per_pred_joules * 1_000.0
            co2_per_pred = self.compute_co2(energy_per_pred_kwh)
            co2_per_1k = co2_per_pred * 1_000.0
            any_estimated = any(m.is_estimated for m in measurements)
            uncertainty = 100.0 if mean_duration < 0.01 else 50.0
            method = "codecarbon_cpu_load" if not any_estimated else "estimated_from_duration"

            per_pred_joules_samples = [
                self.kwh_to_joules(e) / n_predictions for e in energies
            ]
            j_arr = np.asarray(per_pred_joules_samples, dtype=float)
            energy_stats_joules = {
                "mean": float(np.mean(j_arr)),
                "median": float(np.median(j_arr)),
                "std": float(np.std(j_arr, ddof=1)) if len(j_arr) > 1 else 0.0,
                "ci_lower": (
                    stats["energy_joules"]["ci_lower"] / n_predictions
                    if stats["energy_joules"]["ci_lower"] is not None
                    else None
                ),
                "ci_upper": (
                    stats["energy_joules"]["ci_upper"] / n_predictions
                    if stats["energy_joules"]["ci_upper"] is not None
                    else None
                ),
            }
            outlier_count = stats["energy_kwh"]["outlier_count"]
        else:
            energy_per_pred_kwh = None
            energy_per_pred_joules = None
            energy_per_1k_kwh = None
            energy_per_1k_joules = None
            co2_per_pred = None
            co2_per_1k = None
            energy_stats_joules = None
            outlier_count = 0
            uncertainty = 100.0
            method = "unavailable"

        warning = (
            "Inference energy measurements at this scale (< 10 ms per batch) are "
            "dominated by OS and measurement overhead. "
            f"Report values carry ±{uncertainty:.0f}% uncertainty and should NOT "
            "be treated as precise environmental measurements."
        )

        return {
            "energy_per_prediction_joules": energy_per_pred_joules,
            "energy_per_prediction_kwh": energy_per_pred_kwh,
            "energy_per_1k_predictions_joules": energy_per_1k_joules,
            "energy_per_1k_predictions_kwh": energy_per_1k_kwh,
            "co2_per_prediction_kg": co2_per_pred,
            "co2_per_1k_predictions_kg": co2_per_1k,
            "measurement_uncertainty_pct": uncertainty,
            "n_repetitions": repetitions,
            "mean_duration_seconds": mean_duration,
            "median_duration_seconds": median_duration,
            "energy_stats_joules": energy_stats_joules,
            "outlier_count": outlier_count,
            "hardware_specs": measurements[-1].hardware_specs if measurements else hw_specs,
            "method": method,
            "warning": warning,
        }

    # ------------------------------------------------------------------
    # Training amortization
    # ------------------------------------------------------------------

    def amortize_training(
        self,
        training_co2_kg: float,
        inference_co2_per_prediction_kg: float,
        prediction_volume: int,
    ) -> dict:
        """Allocate total lifetime CO₂ across training + inference."""
        inference_total = inference_co2_per_prediction_kg * prediction_volume
        total = training_co2_kg + inference_total

        training_fraction = training_co2_kg / total if total > 0 else 1.0
        inference_fraction = inference_total / total if total > 0 else 0.0

        if inference_co2_per_prediction_kg > 0:
            break_even = int(np.ceil(training_co2_kg / inference_co2_per_prediction_kg))
        else:
            break_even = None

        return {
            "total_co2_kg": total,
            "training_co2_kg": training_co2_kg,
            "inference_co2_kg": inference_total,
            "training_fraction": training_fraction,
            "inference_fraction": inference_fraction,
            "break_even_predictions": break_even,
        }

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_phase_entry(m: Any) -> Dict[str, Any]:
        """Normalise an EnergyMeasurement, list[EnergyMeasurement], or dict."""
        if isinstance(m, list):
            if not m:
                return {
                    "energy_kwh": None,
                    "energy_joules": None,
                    "co2_kg": None,
                    "is_estimated": True,
                    "measurement_method": "unavailable",
                    "uncertainty_pct": 100.0,
                    "duration_seconds": 0.0,
                    "median_energy_kwh": None,
                    "ci_95_energy_kwh": None,
                    "outlier_count": 0,
                    "hardware_specs": None,
                }
            stats = EnergyMonitor.compute_measurement_stats(m)
            energies = [x.energy_kwh for x in m if x.energy_kwh is not None]
            co2s = [x.co2_kg for x in m if x.co2_kg is not None]
            e_sum = sum(energies) if energies else None
            return {
                "energy_kwh": e_sum,
                "energy_joules": e_sum * _JOULES_PER_KWH if e_sum is not None else None,
                "co2_kg": sum(co2s) if co2s else None,
                "is_estimated": any(x.is_estimated for x in m),
                "measurement_method": m[-1].measurement_method,
                "uncertainty_pct": max(x.uncertainty_pct for x in m),
                "duration_seconds": sum(x.duration_seconds for x in m),
                "median_energy_kwh": stats["energy_kwh"]["median"],
                "ci_95_energy_kwh": (
                    stats["energy_kwh"]["ci_lower"],
                    stats["energy_kwh"]["ci_upper"],
                ),
                "outlier_count": stats["energy_kwh"]["outlier_count"],
                "hardware_specs": m[-1].hardware_specs,
            }

        if hasattr(m, "__dataclass_fields__"):
            e_kwh = m.energy_kwh
            e_j = getattr(m, "energy_joules", None)
            if e_j is None and e_kwh is not None:
                e_j = e_kwh * _JOULES_PER_KWH
            return {
                "energy_kwh": e_kwh,
                "energy_joules": e_j,
                "co2_kg": m.co2_kg,
                "is_estimated": m.is_estimated,
                "measurement_method": m.measurement_method,
                "uncertainty_pct": m.uncertainty_pct,
                "duration_seconds": m.duration_seconds,
                "median_energy_kwh": e_kwh,
                "ci_95_energy_kwh": None,
                "outlier_count": 0,
                "hardware_specs": getattr(m, "hardware_specs", None),
            }

        # Dict (either single measurement dict or session_summary entry)
        e_kwh = m.get("energy_kwh", m.get("total_energy_kwh"))
        e_j = m.get("energy_joules", m.get("total_energy_joules"))
        if e_j is None and e_kwh is not None:
            e_j = float(e_kwh) * _JOULES_PER_KWH
        return {
            "energy_kwh": e_kwh,
            "energy_joules": e_j,
            "co2_kg": m.get("co2_kg", m.get("total_co2_kg")),
            "is_estimated": m.get("is_estimated", m.get("any_estimated", True)),
            "measurement_method": m.get("measurement_method", "unknown"),
            "uncertainty_pct": m.get("uncertainty_pct", m.get("max_uncertainty_pct", 50.0)),
            "duration_seconds": m.get("duration_seconds", m.get("total_duration_s", 0.0)),
            "median_energy_kwh": m.get("median_energy_kwh"),
            "ci_95_energy_kwh": m.get("ci_95_energy_kwh"),
            "outlier_count": m.get("outlier_count", 0),
            "hardware_specs": m.get("hardware_specs"),
        }

    def format_carbon_report(self, measurements: dict) -> str:
        """Format a human-readable carbon accounting report.

        Reports energy in BOTH kWh and Joules (``1 kWh = 3,600,000 J``),
        median/CI/outlier stats where available, hardware specs, carbon
        intensity assumptions, and explicit Measured / Estimated / Calculated
        provenance tags.
        """
        lines: list[str] = []
        sep = "=" * 70

        lines.append(sep)
        lines.append("  CARBON ACCOUNTING REPORT")
        lines.append(sep)
        lines.append("")
        lines.append("ASSUMPTIONS")
        lines.append("-" * 40)
        lines.append(self.assumption_statement)
        lines.append("")
        lines.append("MEASUREMENT BOUNDARY")
        lines.append("-" * 40)
        lines.append(
            "  Scope: CPU/RAM compute energy during active execution only.\n"
            "  Excluded: manufacturing, data-centre cooling (PUE), network,\n"
            "            storage, monitor, idle background processes.\n"
            "  Figures represent ESTIMATES of DIRECT compute energy only."
        )
        lines.append("")

        hw_reported = None
        lines.append("PER-PHASE MEASUREMENTS")
        lines.append("-" * 40)

        for phase_name, raw_m in measurements.items():
            entry = self._normalize_phase_entry(raw_m)
            if hw_reported is None and entry.get("hardware_specs"):
                hw_reported = entry["hardware_specs"]

            label = "ESTIMATED" if entry["is_estimated"] else "MEASURED"
            energy_kwh = entry["energy_kwh"]
            energy_joules = entry["energy_joules"]
            co2_kg = entry["co2_kg"]

            if energy_kwh is not None:
                energy_str = (
                    f"{energy_kwh:.6e} kWh ({energy_joules:.4f} J)"
                    if energy_joules is not None
                    else f"{energy_kwh:.6e} kWh"
                )
            else:
                energy_str = "unavailable"

            if co2_kg is not None:
                co2_str = f"{co2_kg:.6e} kg ({co2_kg * 1e9:.4f} µg CO₂e)"
            else:
                co2_str = "unavailable"

            lines.append(f"  Phase: {phase_name}")
            lines.append(f"    Duration       : {entry['duration_seconds']:.4f} s")
            lines.append(f"    Energy         : {energy_str}  [{label}]")
            lines.append(f"    CO₂e           : {co2_str}  [CALCULATED from energy]")
            if entry.get("median_energy_kwh") is not None:
                med_kwh = entry["median_energy_kwh"]
                lines.append(
                    f"    Median Energy  : {med_kwh:.6e} kWh ({med_kwh * _JOULES_PER_KWH:.4f} J)"
                )
            ci = entry.get("ci_95_energy_kwh")
            if ci and ci[0] is not None and ci[1] is not None:
                lines.append(f"    95% CI (kWh)   : [{ci[0]:.6e}, {ci[1]:.6e}]")
            if entry.get("outlier_count", 0) > 0:
                lines.append(f"    Outliers (IQR) : {entry['outlier_count']}")
            lines.append(f"    Method         : {entry['measurement_method']}")
            lines.append(f"    Uncertainty    : ±{entry['uncertainty_pct']:.0f}%")
            lines.append("")

        if hw_reported:
            lines.append("HARDWARE SPECIFICATIONS")
            lines.append("-" * 40)
            for k, v in hw_reported.items():
                lines.append(f"  {k}: {v}")
            lines.append("")

        lines.append("IMPORTANT CAVEATS")
        lines.append("-" * 40)
        lines.append(
            textwrap.fill(
                "At small dataset scale (~1000 rows), ML training completes in "
                "< 1 second. CodeCarbon CPU-load sampling at this timescale is "
                "dominated by background process noise. The sub-milligram CO₂ "
                "figures above are ORDER-OF-MAGNITUDE ESTIMATES only. "
                "Do NOT present micro-gram differences as meaningful "
                "environmental improvements without statistically significant "
                "evidence (see StatisticalAnalyzer.is_demonstrably_greener).",
                width=70,
                initial_indent="  ",
                subsequent_indent="  ",
            )
        )
        lines.append("")
        lines.append(sep)

        return "\n".join(lines)

    @property
    def assumption_statement(self) -> str:
        """Human-readable statement of the carbon-intensity assumption."""
        return (
            f"Carbon intensity: {self.carbon_intensity_kg_per_kwh:.4f} kg CO₂e/kWh "
            f"(source: {self.intensity_source}). "
            "This estimate does NOT include manufacturing emissions, data-centre "
            "cooling overhead (PUE), network transmission, or storage. "
            "It reflects direct CPU/RAM compute energy only, converted to CO₂e "
            "using a static grid-average intensity figure."
        )
