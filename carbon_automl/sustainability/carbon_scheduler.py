"""
carbon_scheduler.py — [EXPERIMENTAL] Carbon-aware workload scheduling helper.

Status: EXPERIMENTAL
--------------------
Grid-aware temporal shifting ("train when the grid is cleaner") requires
reliable, location-specific, real-time or forecasted marginal carbon-intensity
data (e.g., from Electricity Maps, WattTime, or a regional grid operator).

Design constraints
------------------
* NEVER invent renewable-energy availability or fabricated grid forecasts.
* If no external carbon-intensity forecast/source is configured, gracefully
  return ``recommendation="unavailable"`` with an explicit explanation.
* Marked explicitly as experimental in both documentation and return payloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class ScheduleRecommendation:
    """Structured recommendation from :class:`CarbonScheduler` (experimental)."""

    recommendation: str  # "train_now" | "defer" | "unavailable"
    reason: str
    expected_co2_kg: float
    estimated_savings_kg: Optional[float] = None
    experimental: bool = True
    recommended_start_time: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommendation": self.recommendation,
            "reason": self.reason,
            "expected_co2_kg": self.expected_co2_kg,
            "estimated_savings_kg": self.estimated_savings_kg,
            "experimental": self.experimental,
            "recommended_start_time": self.recommended_start_time,
        }


class CarbonScheduler:
    """[EXPERIMENTAL] Advise whether to run training now or defer for lower grid carbon.

    Parameters
    ----------
    expected_duration_minutes:
        Estimated wall-clock training duration in minutes.
    expected_energy_kwh:
        Estimated compute energy required in kilowatt-hours.
    carbon_intensity_kg_per_kwh:
        Fallback static carbon intensity assumption (default 0.7 kg CO₂e/kWh, India).
    intensity_forecast:
        Optional list of user-provided forecast slots, each a dict containing
        ``"timestamp"`` (``datetime`` or ISO-8601 string) and
        ``"carbon_intensity_kg_per_kwh"`` (``float``). No forecast is ever
        synthesised internally.
    real_time_carbon_api_url:
        Optional external API endpoint reference. Note: this offline-safe
        scheduler does NOT fabricate data or scrape endpoints; if a live URL
        is supplied without pre-fetched ``intensity_forecast`` slots, it
        reports ``"unavailable"``.
    """

    EXPERIMENTAL: bool = True

    def __init__(
        self,
        expected_duration_minutes: float,
        expected_energy_kwh: float,
        carbon_intensity_kg_per_kwh: float = 0.7,
        intensity_forecast: Optional[List[Dict[str, Any]]] = None,
        real_time_carbon_api_url: Optional[str] = None,
    ) -> None:
        if expected_duration_minutes < 0:
            raise ValueError("expected_duration_minutes must be non-negative.")
        if expected_energy_kwh < 0:
            raise ValueError("expected_energy_kwh must be non-negative.")
        if carbon_intensity_kg_per_kwh <= 0:
            raise ValueError("carbon_intensity_kg_per_kwh must be positive.")

        self.expected_duration_minutes = float(expected_duration_minutes)
        self.expected_energy_kwh = float(expected_energy_kwh)
        self.carbon_intensity_kg_per_kwh = float(carbon_intensity_kg_per_kwh)
        self.intensity_forecast = intensity_forecast
        self.real_time_carbon_api_url = real_time_carbon_api_url

    def get_recommendation(self, deadline: Optional[datetime] = None) -> Dict[str, Any]:
        """Evaluate whether to train now, defer, or report unavailability.

        Parameters
        ----------
        deadline:
            Optional latest acceptable completion/start timestamp.

        Returns
        -------
        dict with keys:
            recommendation ("train_now" | "defer" | "unavailable"),
            reason, expected_co2_kg, estimated_savings_kg,
            experimental (True), recommended_start_time.
        """
        baseline_co2_kg = self.expected_energy_kwh * self.carbon_intensity_kg_per_kwh

        # Without real external forecast data, never invent renewable availability.
        if not self.intensity_forecast:
            if self.real_time_carbon_api_url:
                reason = (
                    "No pre-fetched real-time carbon intensity data available from "
                    f"configured source ({self.real_time_carbon_api_url}). "
                    "Deferral without verified data would be guesswork."
                )
            else:
                reason = (
                    "No real-time carbon intensity source configured. "
                    "Deferral without data would be guesswork."
                )
            return ScheduleRecommendation(
                recommendation="unavailable",
                reason=reason,
                expected_co2_kg=baseline_co2_kg,
                estimated_savings_kg=None,
                experimental=True,
                recommended_start_time=None,
            ).to_dict()

        # Filter forecast slots to those valid before deadline (if provided).
        valid_slots: List[tuple[datetime, float]] = []
        for slot in self.intensity_forecast:
            ts_raw = slot.get("timestamp")
            intensity_raw = slot.get("carbon_intensity_kg_per_kwh")
            if ts_raw is None or intensity_raw is None:
                continue
            if isinstance(ts_raw, str):
                ts = datetime.fromisoformat(ts_raw)
            elif isinstance(ts_raw, datetime):
                ts = ts_raw
            else:
                continue

            if deadline is not None:
                # Normalise timezone awareness if comparing naive and aware datetimes
                ts_cmp = ts.replace(tzinfo=None) if ts.tzinfo and not deadline.tzinfo else ts
                dl_cmp = deadline.replace(tzinfo=None) if deadline.tzinfo and not ts.tzinfo else deadline
                if ts_cmp > dl_cmp:
                    continue

            valid_slots.append((ts, float(intensity_raw)))

        if not valid_slots:
            return ScheduleRecommendation(
                recommendation="unavailable",
                reason=(
                    "Configured carbon intensity forecast contains no valid slots "
                    "before the specified deadline. Deferral without data would be guesswork."
                ),
                expected_co2_kg=baseline_co2_kg,
                estimated_savings_kg=None,
                experimental=True,
                recommended_start_time=None,
            ).to_dict()

        # Sort chronologically; first slot represents current intensity, lowest slot is optimal.
        valid_slots.sort(key=lambda item: item[0])
        now_ts, now_intensity = valid_slots[0]
        best_ts, best_intensity = min(valid_slots, key=lambda item: item[1])

        now_co2 = self.expected_energy_kwh * now_intensity
        best_co2 = self.expected_energy_kwh * best_intensity
        savings = now_co2 - best_co2

        # Require a clear > 5% relative improvement from external forecast data to advise deferral.
        if best_ts > now_ts and now_co2 > 0 and (savings / now_co2) > 0.05:
            return ScheduleRecommendation(
                recommendation="defer",
                reason=(
                    f"[EXPERIMENTAL] External forecast shows lower grid intensity "
                    f"({best_intensity:.4f} vs {now_intensity:.4f} kg CO₂e/kWh) at "
                    f"{best_ts.isoformat()}."
                ),
                expected_co2_kg=best_co2,
                estimated_savings_kg=savings,
                experimental=True,
                recommended_start_time=best_ts.isoformat(),
            ).to_dict()

        return ScheduleRecommendation(
            recommendation="train_now",
            reason=(
                "[EXPERIMENTAL] Current external forecast slot is already at or near "
                f"the minimum carbon intensity before deadline ({now_intensity:.4f} kg CO₂e/kWh)."
            ),
            expected_co2_kg=now_co2,
            estimated_savings_kg=0.0,
            experimental=True,
            recommended_start_time=now_ts.isoformat(),
        ).to_dict()
