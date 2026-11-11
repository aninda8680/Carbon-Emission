"""
energy_monitor.py — Per-phase energy measurement wrapper around CodeCarbon.

Design note
-----------
At small dataset scales (~1000 rows), ML training completes in well under one
second.  At that timescale, CodeCarbon's CPU-load sampling (which typically
samples every few seconds) produces measurements that are dominated by
background process noise — the energy attributed to the target workload can
easily be off by 50–100 %.  This module surfaces that uncertainty explicitly
in every EnergyMeasurement it returns, so callers can communicate it to users
rather than presenting micro-gram CO₂ figures as meaningful.
"""

from __future__ import annotations

import logging
import math
import os
import platform
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

JOULES_PER_KWH: float = 3_600_000.0

STANDARD_PHASES: Tuple[str, ...] = (
    "ingestion",
    "preprocessing",
    "feature_selection",
    "hpo",
    "training",
    "inference",
    "total_pipeline",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ASSUMED_WATTS_PER_CPU_CORE = 10.0  # conservative estimate used as fallback


def _get_cpu_count() -> int:
    try:
        return os.cpu_count() or 1
    except Exception:
        return 1


def _get_ram_gb() -> float:
    try:
        import psutil  # type: ignore

        return float(psutil.virtual_memory().total) / (1024 ** 3)
    except Exception:
        return 0.0


def get_hardware_specs() -> Dict[str, Any]:
    """Collect host hardware and runtime environment specifications."""
    return {
        "cpu_count": _get_cpu_count(),
        "cpu_model": platform.processor() or "unknown",
        "ram_gb": round(_get_ram_gb(), 2),
        "os": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
    }


def _compute_numeric_stats(
    values: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 500,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Compute mean, median, std, CI, min, max, and 1.5-IQR outlier counts."""
    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "ci_lower": None,
            "ci_upper": None,
            "outlier_count": 0,
        }

    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean_v = float(np.mean(arr))
    median_v = float(np.median(arr))
    std_v = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))

    if n >= 2:
        rng = np.random.default_rng(random_state)
        boot = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            boot[i] = rng.choice(arr, size=n, replace=True).mean()
        alpha = 1.0 - confidence
        ci_lower = float(np.percentile(boot, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    else:
        ci_lower = mean_v
        ci_upper = mean_v

    # Tukey 1.5 * IQR outlier count (only meaningful when n >= 4)
    if n >= 4:
        q25, q75 = np.percentile(arr, [25, 75])
        iqr = q75 - q25
        lower_fence = q25 - 1.5 * iqr
        upper_fence = q75 + 1.5 * iqr
        outlier_count = int(np.sum((arr < lower_fence) | (arr > upper_fence)))
    else:
        outlier_count = 0

    return {
        "n": n,
        "mean": mean_v,
        "median": median_v,
        "std": std_v,
        "min": min_v,
        "max": max_v,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "outlier_count": outlier_count,
    }


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class EnergyMeasurement:
    """Structured result of one energy-measurement interval.

    Fields
    ------
    phase_name : str
        Human-readable label for the measured phase (e.g. ``"training"``).
    duration_seconds : float
        Wall-clock duration of the measurement.
    energy_kwh : float | None
        Energy consumed during the phase in kilowatt-hours, or None when
        measurement is unavailable.
    co2_kg : float | None
        Estimated CO₂-equivalent in kg, or None.
    cpu_count : int
        Number of logical CPUs on the host.
    ram_gb : float
        Total RAM on the host in GB.
    measurement_method : str
        Short identifier such as ``"codecarbon_cpu_load"`` or
        ``"estimated_from_duration"``.
    is_estimated : bool
        True when the value is a rough estimate rather than an instrument
        reading.
    uncertainty_pct : float
        Conservative uncertainty as a percentage of the reported value.
        Defaults to 50 % for CPU-load estimates at sub-second workloads;
        may be as high as 100 % for extremely short runs.
    energy_joules : float | None
        Energy consumed in Joules (``energy_kwh * 3_600_000``). Populated
        automatically if ``energy_kwh`` is provided.
    hardware_specs : dict
        Snapshot of host hardware metadata (CPU count/model, RAM, OS, Python).
    """

    phase_name: str
    duration_seconds: float
    energy_kwh: Optional[float]
    co2_kg: Optional[float]
    cpu_count: int
    ram_gb: float
    measurement_method: str
    is_estimated: bool
    uncertainty_pct: float = 50.0
    energy_joules: Optional[float] = None
    hardware_specs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.energy_joules is None and self.energy_kwh is not None:
            self.energy_joules = float(self.energy_kwh) * JOULES_PER_KWH
        elif self.energy_kwh is None and self.energy_joules is not None:
            self.energy_kwh = float(self.energy_joules) / JOULES_PER_KWH
        if not self.hardware_specs:
            self.hardware_specs = {
                "cpu_count": self.cpu_count,
                "ram_gb": round(float(self.ram_gb), 2),
                "os": platform.platform(),
                "python_version": platform.python_version(),
            }

    # ------------------------------------------------------------------
    def human_summary(self) -> str:
        """Return a plain-English one-liner suitable for logs / reports."""
        if self.energy_kwh is None:
            energy_str = "energy unavailable"
            co2_str = "CO₂ unavailable"
        else:
            joules = self.energy_joules if self.energy_joules is not None else self.energy_kwh * JOULES_PER_KWH
            energy_str = f"{self.energy_kwh:.6e} kWh ({joules:.4f} J)"
            co2_str = (
                f"{self.co2_kg * 1e9:.4f} µg CO₂e"
                if self.co2_kg is not None
                else "CO₂ unavailable"
            )

        flag = "ESTIMATED" if self.is_estimated else "MEASURED"
        return (
            f"[{flag}] Phase '{self.phase_name}': "
            f"{self.duration_seconds:.4f}s | "
            f"{energy_str} | {co2_str} "
            f"(uncertainty ±{self.uncertainty_pct:.0f}%, method: {self.measurement_method})"
        )


def _build_fallback_measurement(
    phase_name: str,
    duration_seconds: float,
    carbon_intensity_kg_per_kwh: float = 0.7,
) -> EnergyMeasurement:
    """Estimate energy from elapsed time and a fixed watts-per-core assumption."""
    specs = get_hardware_specs()
    cpu_count = int(specs["cpu_count"])
    ram_gb = float(specs["ram_gb"])

    # Very coarse: assume all CPUs at the stated wattage during the interval.
    total_watts = _ASSUMED_WATTS_PER_CPU_CORE * cpu_count
    energy_kwh = (total_watts * duration_seconds) / JOULES_PER_KWH  # W·s → kWh
    co2_kg = energy_kwh * carbon_intensity_kg_per_kwh

    # For sub-second workloads the uncertainty is at least 100 %.
    uncertainty = 100.0 if duration_seconds < 1.0 else 50.0

    return EnergyMeasurement(
        phase_name=phase_name,
        duration_seconds=duration_seconds,
        energy_kwh=energy_kwh,
        energy_joules=energy_kwh * JOULES_PER_KWH,
        co2_kg=co2_kg,
        cpu_count=cpu_count,
        ram_gb=ram_gb,
        measurement_method="estimated_from_duration",
        is_estimated=True,
        uncertainty_pct=uncertainty,
        hardware_specs=specs,
    )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class EnergyMonitor:
    """Wraps CodeCarbon's EmissionsTracker (offline-safe) with graceful fallback.

    Supports all standard AutoML pipeline phases:
    ``ingestion``, ``preprocessing``, ``feature_selection``, ``hpo``,
    ``training``, ``inference``, and ``total_pipeline``.

    Usage (context-manager)::

        with EnergyMonitor(phase_name="training") as em:
            train_model(...)
        measurement = em.last_measurement

    Usage (explicit)::

        em = EnergyMonitor()
        em.start("preprocessing")
        preprocess(...)
        m = em.stop()
    """

    STANDARD_PHASES = STANDARD_PHASES

    def __init__(
        self,
        phase_name: str = "",
        carbon_intensity_kg_per_kwh: float = 0.7,
        codecarbon_log_level: str = "ERROR",
        output_dir: str = ".",
        country_iso_code: str = "IND",
    ) -> None:
        self.phase_name = phase_name
        self.carbon_intensity_kg_per_kwh = carbon_intensity_kg_per_kwh
        self.codecarbon_log_level = codecarbon_log_level
        self.output_dir = output_dir
        self.country_iso_code = country_iso_code

        self._tracker = None  # Active CodeCarbon tracker for current interval
        self._reusable_tracker = None  # Cached tracker supporting start_task/stop_task
        self._using_task_mode: bool = False
        self._active_task_name: Optional[str] = None
        self._task_seq: int = 0
        self._codecarbon_available: Optional[bool] = None  # None = not yet checked
        self._start_time: Optional[float] = None
        self._active_phase: Optional[str] = None

        self.session: Dict[str, List[EnergyMeasurement]] = {}
        self.last_measurement: Optional[EnergyMeasurement] = None
        self.hardware_specs: Dict[str, Any] = get_hardware_specs()

    # ------------------------------------------------------------------
    # CodeCarbon availability & offline-safe tracker creation
    # ------------------------------------------------------------------

    def _check_codecarbon(self) -> bool:
        if self._codecarbon_available is not None:
            return self._codecarbon_available
        try:
            import codecarbon  # noqa: F401

            self._codecarbon_available = True
        except ImportError:
            logger.warning(
                "codecarbon not installed — falling back to time-based energy estimate. "
                "Install with: pip install codecarbon"
            )
            self._codecarbon_available = False
        return self._codecarbon_available

    def _build_tracker(self, project: str) -> Any:
        """Instantiate an offline-safe CodeCarbon tracker."""
        import codecarbon  # type: ignore

        common_kwargs = {
            "project_name": project,
            "log_level": self.codecarbon_log_level,
            "save_to_file": False,
            "save_to_api": False,
            "allow_multiple_runs": True,
        }

        offline_cls = getattr(codecarbon, "OfflineEmissionsTracker", None)
        if offline_cls is not None:
            try:
                return offline_cls(
                    country_iso_code=self.country_iso_code,
                    **common_kwargs,
                )
            except TypeError:
                return offline_cls(
                    country_iso_code=self.country_iso_code,
                    project_name=project,
                    log_level=self.codecarbon_log_level,
                    save_to_file=False,
                )

        tracker_cls = getattr(codecarbon, "EmissionsTracker")
        try:
            return tracker_cls(**common_kwargs)
        except TypeError:
            return tracker_cls(
                project_name=project,
                log_level=self.codecarbon_log_level,
                save_to_file=False,
            )

    def _extract_codecarbon_measurement(
        self,
        phase: str,
        duration: float,
        raw: Any,
        emissions_data: Any = None,
    ) -> EnergyMeasurement:
        """Build an EnergyMeasurement from CodeCarbon's EmissionsData / float return."""
        energy_kwh: Optional[float] = None
        co2_kg: Optional[float] = None
        specs = dict(self.hardware_specs)
        cpu_count = int(specs["cpu_count"])
        ram_gb = float(specs["ram_gb"])

        if raw is not None:
            raw_energy = getattr(raw, "energy_consumed", None)
            if raw_energy is not None and not math.isnan(float(raw_energy)):
                energy_kwh = float(raw_energy)
            raw_co2 = getattr(raw, "emissions", None)
            if raw_co2 is not None and not math.isnan(float(raw_co2)):
                co2_kg = float(raw_co2)
            elif isinstance(emissions_data, float) and not math.isnan(emissions_data):
                co2_kg = float(emissions_data)

            cpu_count = int(getattr(raw, "cpu_count", cpu_count) or cpu_count)
            ram_gb = float(getattr(raw, "ram_total_size", ram_gb) or ram_gb)
            if getattr(raw, "cpu_model", None):
                specs["cpu_model"] = raw.cpu_model
            if getattr(raw, "gpu_model", None):
                specs["gpu_model"] = raw.gpu_model
            if getattr(raw, "gpu_count", None) is not None:
                specs["gpu_count"] = raw.gpu_count
            specs["cpu_count"] = cpu_count
            specs["ram_gb"] = round(ram_gb, 2)
        elif isinstance(emissions_data, float) and not math.isnan(emissions_data):
            co2_kg = float(emissions_data)
            energy_kwh = (
                co2_kg / self.carbon_intensity_kg_per_kwh
                if self.carbon_intensity_kg_per_kwh > 0
                else None
            )

        # Ensure CO₂e always honours the configured carbon intensity when energy is known
        if energy_kwh is not None and self.carbon_intensity_kg_per_kwh > 0:
            co2_kg = energy_kwh * self.carbon_intensity_kg_per_kwh

        # At sub-second duration CodeCarbon sampling is dominated by background noise.
        uncertainty = 100.0 if duration < 1.0 else 50.0

        return EnergyMeasurement(
            phase_name=phase,
            duration_seconds=duration,
            energy_kwh=energy_kwh,
            energy_joules=(energy_kwh * JOULES_PER_KWH) if energy_kwh is not None else None,
            co2_kg=co2_kg,
            cpu_count=cpu_count,
            ram_gb=ram_gb,
            measurement_method="codecarbon_cpu_load",
            is_estimated=False,
            uncertainty_pct=uncertainty,
            hardware_specs=specs,
        )

    # ------------------------------------------------------------------
    # start / stop
    # ------------------------------------------------------------------

    def start(self, phase_name: str = "") -> None:
        """Begin measuring energy for *phase_name*."""
        self._active_phase = phase_name or self.phase_name or "unnamed"
        self._using_task_mode = False
        self._active_task_name = None

        if self._check_codecarbon():
            try:
                if self._reusable_tracker is None:
                    tracker = self._build_tracker(self._active_phase)
                    if hasattr(tracker, "start_task") and hasattr(tracker, "stop_task"):
                        self._reusable_tracker = tracker
                    self._tracker = tracker
                else:
                    self._tracker = self._reusable_tracker

                if hasattr(self._tracker, "start_task") and hasattr(self._tracker, "stop_task"):
                    self._task_seq += 1
                    self._active_task_name = f"{self._active_phase}__{self._task_seq}"
                    self._using_task_mode = True
                    self._tracker.start_task(self._active_task_name)
                else:
                    self._tracker.start()
            except Exception as exc:
                logger.warning("CodeCarbon start failed (%s); using fallback.", exc)
                self._tracker = None
                self._reusable_tracker = None
                self._using_task_mode = False
                self._active_task_name = None
                self._codecarbon_available = False

        # Start workload stopwatch AFTER tracker initialization/start so cold-start
        # hardware probing does not inflate the workload's reported duration.
        self._start_time = time.perf_counter()

    def stop(self) -> EnergyMeasurement:
        """Stop measuring and return an :class:`EnergyMeasurement`."""
        end_time = time.perf_counter()
        duration = (end_time - self._start_time) if self._start_time is not None else 0.0
        phase = self._active_phase or self.phase_name or "unnamed"

        measurement: EnergyMeasurement

        if self._tracker is not None:
            try:
                if self._using_task_mode and self._active_task_name is not None:
                    raw = self._tracker.stop_task(self._active_task_name)
                    emissions_data = getattr(raw, "emissions", None) if raw is not None else None
                else:
                    emissions_data = self._tracker.stop()
                    raw = getattr(self._tracker, "final_emissions_data", None)

                measurement = self._extract_codecarbon_measurement(
                    phase=phase,
                    duration=duration,
                    raw=raw,
                    emissions_data=emissions_data,
                )
            except Exception as exc:
                logger.warning("CodeCarbon stop failed (%s); using fallback.", exc)
                measurement = _build_fallback_measurement(
                    phase, duration, self.carbon_intensity_kg_per_kwh
                )
        else:
            measurement = _build_fallback_measurement(
                phase, duration, self.carbon_intensity_kg_per_kwh
            )

        # Store in session
        self.session.setdefault(phase, []).append(measurement)
        self.last_measurement = measurement

        # Reset active interval state (keep _reusable_tracker cached for fast subsequent runs)
        self._start_time = None
        self._active_phase = None
        self._tracker = None
        self._using_task_mode = False
        self._active_task_name = None

        return measurement

    def close(self) -> None:
        """Release any cached CodeCarbon tracker resources."""
        if self._reusable_tracker is not None:
            try:
                if hasattr(self._reusable_tracker, "stop"):
                    self._reusable_tracker.stop()
            except Exception:
                pass
            self._reusable_tracker = None

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "EnergyMonitor":
        self.start(self.phase_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.stop()
        return False

    # ------------------------------------------------------------------
    # Repeated measurement helper
    # ------------------------------------------------------------------

    def measure_repeated(
        self,
        func: Callable,
        n_repeats: int = 3,
        phase_name: str = "",
    ) -> List[EnergyMeasurement]:
        """Run *func* exactly *n_repeats* times, measuring energy each time."""
        base_name = phase_name or self.phase_name or "repeated"
        results: List[EnergyMeasurement] = []

        for i in range(n_repeats):
            rep_name = f"{base_name}_rep{i + 1}"
            self.start(rep_name)
            try:
                func()
            except Exception:
                self.stop()
                raise
            m = self.stop()
            results.append(m)

        return results

    # ------------------------------------------------------------------
    # Statistics & session summary
    # ------------------------------------------------------------------

    @staticmethod
    def compute_measurement_stats(
        measurements: List[EnergyMeasurement],
    ) -> Dict[str, Any]:
        """Compute median, mean, CI, and outlier statistics for a list of measurements."""
        durations = [m.duration_seconds for m in measurements]
        energies_kwh = [m.energy_kwh for m in measurements if m.energy_kwh is not None]
        energies_j = [
            m.energy_joules if m.energy_joules is not None else (m.energy_kwh * JOULES_PER_KWH)
            for m in measurements
            if m.energy_kwh is not None or m.energy_joules is not None
        ]
        co2s = [m.co2_kg for m in measurements if m.co2_kg is not None]

        return {
            "n_measurements": len(measurements),
            "duration_seconds": _compute_numeric_stats(durations),
            "energy_kwh": _compute_numeric_stats(energies_kwh),
            "energy_joules": _compute_numeric_stats(energies_j),
            "co2_kg": _compute_numeric_stats(co2s),
            "any_estimated": any(m.is_estimated for m in measurements),
            "max_uncertainty_pct": max((m.uncertainty_pct for m in measurements), default=100.0),
        }

    def get_phase_statistics(self, phase_name: str) -> Dict[str, Any]:
        """Return detailed stats for a phase (or its ``<phase>_rep*`` runs)."""
        if phase_name in self.session:
            items = self.session[phase_name]
        else:
            prefix = f"{phase_name}_rep"
            items = [
                m
                for key, mlist in self.session.items()
                if key.startswith(prefix)
                for m in mlist
            ]
        stats = self.compute_measurement_stats(items)
        stats["phase_name"] = phase_name
        stats["hardware_specs"] = self.hardware_specs
        return stats

    def session_summary(self) -> Dict[str, dict]:
        """Return a dict summarising all phases measured in this session,
        including both kWh and Joules, median/CI/outlier stats, and hardware specs.
        """
        summary: Dict[str, dict] = {}
        for phase_name, measurements in self.session.items():
            stats = self.compute_measurement_stats(measurements)
            energies = [m.energy_kwh for m in measurements if m.energy_kwh is not None]
            energies_j = [
                m.energy_joules if m.energy_joules is not None else (m.energy_kwh * JOULES_PER_KWH)
                for m in measurements
                if m.energy_kwh is not None or m.energy_joules is not None
            ]
            co2s = [m.co2_kg for m in measurements if m.co2_kg is not None]
            durations = [m.duration_seconds for m in measurements]

            total_kwh = sum(energies) if energies else None
            total_j = sum(energies_j) if energies_j else None

            summary[phase_name] = {
                "n_measurements": len(measurements),
                "total_duration_s": sum(durations),
                "mean_duration_s": stats["duration_seconds"]["mean"],
                "median_duration_s": stats["duration_seconds"]["median"],
                "total_energy_kwh": total_kwh,
                "total_energy_joules": total_j,
                "mean_energy_kwh": stats["energy_kwh"]["mean"],
                "median_energy_kwh": stats["energy_kwh"]["median"],
                "ci_95_energy_kwh": (
                    stats["energy_kwh"]["ci_lower"],
                    stats["energy_kwh"]["ci_upper"],
                ),
                "mean_energy_joules": stats["energy_joules"]["mean"],
                "median_energy_joules": stats["energy_joules"]["median"],
                "outlier_count": stats["energy_kwh"]["outlier_count"],
                "total_co2_kg": sum(co2s) if co2s else None,
                "mean_co2_kg": stats["co2_kg"]["mean"],
                "median_co2_kg": stats["co2_kg"]["median"],
                "any_estimated": stats["any_estimated"],
                "max_uncertainty_pct": stats["max_uncertainty_pct"],
                "hardware_specs": measurements[-1].hardware_specs if measurements else self.hardware_specs,
            }

        # Add aggregated total_pipeline if sub-phases exist and total_pipeline wasn't explicitly recorded
        if self.session and "total_pipeline" not in summary:
            all_sub = [
                m
                for phase_key, mlist in self.session.items()
                if phase_key != "total_pipeline"
                for m in mlist
            ]
            if all_sub:
                sub_stats = self.compute_measurement_stats(all_sub)
                e_sub = [m.energy_kwh for m in all_sub if m.energy_kwh is not None]
                j_sub = [
                    m.energy_joules if m.energy_joules is not None else (m.energy_kwh * JOULES_PER_KWH)
                    for m in all_sub
                    if m.energy_kwh is not None or m.energy_joules is not None
                ]
                c_sub = [m.co2_kg for m in all_sub if m.co2_kg is not None]
                summary["total_pipeline"] = {
                    "n_measurements": len(all_sub),
                    "total_duration_s": sum(m.duration_seconds for m in all_sub),
                    "mean_duration_s": sub_stats["duration_seconds"]["mean"],
                    "median_duration_s": sub_stats["duration_seconds"]["median"],
                    "total_energy_kwh": sum(e_sub) if e_sub else None,
                    "total_energy_joules": sum(j_sub) if j_sub else None,
                    "mean_energy_kwh": sub_stats["energy_kwh"]["mean"],
                    "median_energy_kwh": sub_stats["energy_kwh"]["median"],
                    "ci_95_energy_kwh": (
                        sub_stats["energy_kwh"]["ci_lower"],
                        sub_stats["energy_kwh"]["ci_upper"],
                    ),
                    "mean_energy_joules": sub_stats["energy_joules"]["mean"],
                    "median_energy_joules": sub_stats["energy_joules"]["median"],
                    "outlier_count": sub_stats["energy_kwh"]["outlier_count"],
                    "total_co2_kg": sum(c_sub) if c_sub else None,
                    "mean_co2_kg": sub_stats["co2_kg"]["mean"],
                    "median_co2_kg": sub_stats["co2_kg"]["median"],
                    "any_estimated": sub_stats["any_estimated"],
                    "max_uncertainty_pct": sub_stats["max_uncertainty_pct"],
                    "hardware_specs": self.hardware_specs,
                    "is_aggregated_from_phases": True,
                }

        return summary
