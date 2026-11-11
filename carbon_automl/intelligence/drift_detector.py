"""
DriftDetector: Detects data distribution drift between a reference dataset
and new incoming data using Population Stability Index (PSI) and fallback
mean-shift detection.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


class DriftDetector:
    """Detects feature distribution drift using PSI and mean-shift fallback.

    Usage::

        detector = DriftDetector()
        ref_stats = detector.compute_reference_stats(X_train)
        drift = detector.detect_drift(X_new, ref_stats)
        print(detector.format_drift_report(drift))
    """

    # PSI thresholds (industry standard)
    _PSI_LOW = 0.1    # < 0.1  → no significant change
    _PSI_MED = 0.2    # 0.1–0.2 → moderate change
    # > 0.2            → significant drift

    _N_BINS = 10  # number of PSI bins

    # ------------------------------------------------------------------
    # Reference stats
    # ------------------------------------------------------------------

    def compute_reference_stats(self, X: pd.DataFrame) -> dict:
        """Compute per-feature reference statistics from the training set.

        Parameters
        ----------
        X:
            Training feature DataFrame.

        Returns
        -------
        dict
            Mapping ``{column_name: {mean, std, min, max, q25, q75,
            unique_count, is_numerical, bin_edges, bin_proportions}}``.
            ``bin_edges`` / ``bin_proportions`` are populated only for
            numerical features (used by PSI later).
        """
        stats: dict[str, dict] = {}
        for col in X.columns:
            series = X[col]
            is_num = pd.api.types.is_numeric_dtype(series)
            col_stats: dict[str, Any] = {
                "is_numerical": is_num,
                "unique_count": int(series.nunique()),
            }
            if is_num:
                col_stats["mean"] = float(series.mean()) if not series.isna().all() else float("nan")
                col_stats["std"] = float(series.std()) if not series.isna().all() else float("nan")
                col_stats["min"] = float(series.min()) if not series.isna().all() else float("nan")
                col_stats["max"] = float(series.max()) if not series.isna().all() else float("nan")
                col_stats["q25"] = float(series.quantile(0.25)) if not series.isna().all() else float("nan")
                col_stats["q75"] = float(series.quantile(0.75)) if not series.isna().all() else float("nan")

                # Pre-compute PSI bins from the reference distribution
                clean = series.dropna()
                if len(clean) > 0 and clean.nunique() > 1:
                    _, bin_edges = np.histogram(clean, bins=self._N_BINS)
                    ref_counts, _ = np.histogram(clean, bins=bin_edges)
                    ref_props = (ref_counts + 1e-8) / (ref_counts.sum() + 1e-8 * len(ref_counts))
                    col_stats["bin_edges"] = bin_edges.tolist()
                    col_stats["bin_proportions"] = ref_props.tolist()
                else:
                    col_stats["bin_edges"] = None
                    col_stats["bin_proportions"] = None
            else:
                # Categorical: store value frequencies as reference
                vc = series.value_counts(normalize=True)
                col_stats["mean"] = float("nan")
                col_stats["std"] = float("nan")
                col_stats["min"] = float("nan")
                col_stats["max"] = float("nan")
                col_stats["q25"] = float("nan")
                col_stats["q75"] = float("nan")
                col_stats["bin_edges"] = None
                col_stats["bin_proportions"] = vc.to_dict()

            stats[col] = col_stats
        return stats

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------

    def detect_drift(
        self, X_new: pd.DataFrame, reference_stats: dict
    ) -> dict:
        """Detect drift in *X_new* relative to *reference_stats*.

        For each numerical feature, PSI is computed. If PSI is unavailable
        (constant reference feature), mean shift > 2 std is used as fallback.
        Categorical features use a simple frequency-shift heuristic.

        Parameters
        ----------
        X_new:
            New incoming data (same schema as reference).
        reference_stats:
            Output of :meth:`compute_reference_stats`.

        Returns
        -------
        dict
            ``{feature: {psi, drift_detected, severity, method}}``.
        """
        results: dict[str, dict] = {}

        for col, ref in reference_stats.items():
            if col not in X_new.columns:
                # Feature is missing in new data — flag as drift
                results[col] = {
                    "psi": None,
                    "drift_detected": True,
                    "severity": "high",
                    "method": "missing_feature",
                    "note": "Feature absent in new data",
                }
                continue

            new_series = X_new[col]
            is_num = ref["is_numerical"]

            if is_num:
                result = self._check_numerical(col, new_series, ref)
            else:
                result = self._check_categorical(col, new_series, ref)

            results[col] = result

        return results

    # ------------------------------------------------------------------
    # Report formatting
    # ------------------------------------------------------------------

    def format_drift_report(self, drift_results: dict) -> str:
        """Return a human-readable plain-text drift report.

        Parameters
        ----------
        drift_results:
            Output of :meth:`detect_drift`.

        Returns
        -------
        str
        """
        lines = ["=" * 72, "DRIFT DETECTION REPORT", "=" * 72]

        drifted = {f: r for f, r in drift_results.items() if r.get("drift_detected")}
        stable = {f: r for f, r in drift_results.items() if not r.get("drift_detected")}

        # Summary header
        lines.append(
            f"Features checked : {len(drift_results)}\n"
            f"Drifted          : {len(drifted)}  "
            f"({'⚠ action recommended' if drifted else 'none'})\n"
            f"Stable           : {len(stable)}\n"
        )

        if drifted:
            lines.append("── DRIFTED FEATURES " + "─" * 52)
            for feat, info in sorted(drifted.items(), key=lambda x: _severity_rank(x[1].get("severity", "low")), reverse=True):
                psi_str = f"{info['psi']:.4f}" if info.get("psi") is not None else "N/A"
                sev = info.get("severity", "unknown").upper()
                method = info.get("method", "unknown")
                note = f"  ({info['note']})" if info.get("note") else ""
                lines.append(
                    f"  [{sev:6s}] {feat}\n"
                    f"           PSI={psi_str}  method={method}{note}"
                )

        if stable:
            lines.append("\n── STABLE FEATURES " + "─" * 53)
            for feat, info in sorted(stable.items()):
                psi_str = f"{info['psi']:.4f}" if info.get("psi") is not None else "N/A"
                method = info.get("method", "unknown")
                lines.append(f"  [OK    ] {feat}   PSI={psi_str}  method={method}")

        lines.append("=" * 72)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_numerical(
        self, col: str, new_series: pd.Series, ref: dict
    ) -> dict:
        """Return drift info for a numerical feature."""
        clean_new = new_series.dropna()

        # Try PSI first
        bin_edges = ref.get("bin_edges")
        ref_props = ref.get("bin_proportions")

        if (
            bin_edges is not None
            and ref_props is not None
            and len(clean_new) > 0
        ):
            bin_edges_arr = np.array(bin_edges)
            ref_props_arr = np.array(ref_props)
            new_counts, _ = np.histogram(clean_new, bins=bin_edges_arr)
            new_props = (new_counts + 1e-8) / (new_counts.sum() + 1e-8 * len(new_counts))
            psi = float(
                np.sum((new_props - ref_props_arr) * np.log(new_props / ref_props_arr))
            )
            drift_detected = psi > self._PSI_MED
            severity = self._psi_severity(psi)
            return {
                "psi": psi,
                "drift_detected": drift_detected,
                "severity": severity,
                "method": "psi",
            }

        # Fallback: mean shift > 2 std
        ref_mean = ref.get("mean", float("nan"))
        ref_std = ref.get("std", float("nan"))
        if math.isnan(ref_mean) or math.isnan(ref_std) or ref_std == 0 or len(clean_new) == 0:
            return {
                "psi": None,
                "drift_detected": False,
                "severity": "unknown",
                "method": "insufficient_data",
                "note": "Cannot compute drift — constant or empty reference feature",
            }

        new_mean = float(clean_new.mean())
        shift = abs(new_mean - ref_mean) / ref_std
        drift_detected = shift > 2.0
        severity = "high" if shift > 4 else ("medium" if shift > 2 else "low")
        return {
            "psi": None,
            "drift_detected": drift_detected,
            "severity": severity,
            "method": "mean_shift_fallback",
            "mean_shift_std_units": round(shift, 4),
        }

    @staticmethod
    def _check_categorical(col: str, new_series: pd.Series, ref: dict) -> dict:
        """Heuristic drift check for categorical features using JS divergence."""
        ref_probs: dict = ref.get("bin_proportions", {})
        if not ref_probs:
            return {
                "psi": None,
                "drift_detected": False,
                "severity": "unknown",
                "method": "no_reference_distribution",
            }

        new_vc = new_series.value_counts(normalize=True).to_dict()
        # Build aligned probability vectors
        all_cats = set(ref_probs) | set(new_vc)
        eps = 1e-8
        p = np.array([ref_probs.get(c, eps) for c in all_cats])
        q = np.array([new_vc.get(c, eps) for c in all_cats])
        p = p / p.sum()
        q = q / q.sum()

        # Compute PSI-equivalent for categorical
        psi = float(np.sum((q - p) * np.log((q + eps) / (p + eps))))
        drift_detected = psi > 0.2
        severity = "high" if psi > 0.5 else ("medium" if psi > 0.2 else "low")
        return {
            "psi": psi,
            "drift_detected": drift_detected,
            "severity": severity,
            "method": "categorical_psi",
        }

    @staticmethod
    def _psi_severity(psi: float) -> str:
        if psi < 0.1:
            return "low"
        if psi < 0.2:
            return "medium"
        return "high"


def _severity_rank(severity: str) -> int:
    return {"high": 3, "medium": 2, "low": 1, "unknown": 0}.get(severity, 0)
