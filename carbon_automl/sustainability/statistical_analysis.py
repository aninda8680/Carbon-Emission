"""
statistical_analysis.py — Rigorous statistical comparisons for ML results.

Design constraints
------------------
* NEVER claims one model is "greener" without statistically significant
  evidence at p < 0.05 (Bonferroni-corrected for multiple comparisons).
* All public methods return plain dicts — no hidden state.
* At n < 5 samples per group, the Wilcoxon signed-rank test is unreliable;
  the module returns None with an explanation rather than a spurious p-value.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try importing scipy; fall back to pure-NumPy implementations where needed.
# ---------------------------------------------------------------------------

try:
    from scipy.stats import wilcoxon as _scipy_wilcoxon  # type: ignore

    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SCIPY_AVAILABLE = False
    logger.warning(
        "scipy not installed — Wilcoxon tests unavailable. "
        "Install with: pip install scipy"
    )

_MIN_SAMPLES_WILCOXON = 5  # Minimum group size for a reliable Wilcoxon test.


# ---------------------------------------------------------------------------
# StatisticalAnalyzer
# ---------------------------------------------------------------------------


class StatisticalAnalyzer:
    """Statistical utilities for comparing model quality and energy usage.

    All methods are stateless — no internal caching or side effects.
    """

    # ------------------------------------------------------------------
    # Bootstrap confidence interval
    # ------------------------------------------------------------------

    def bootstrap_ci(
        self,
        scores: List[float],
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        random_state: int = 42,
    ) -> dict:
        """Estimate a confidence interval via the percentile bootstrap.

        Parameters
        ----------
        scores:
            Observed values (e.g. cross-validation F1 scores).
        n_bootstrap:
            Number of bootstrap resamples.
        confidence:
            Coverage probability for the interval (e.g. 0.95 for 95 %).
        random_state:
            Seed for reproducibility.

        Returns
        -------
        dict with keys:
            mean, std, ci_lower, ci_upper, n_samples, method
        """
        scores_arr = np.asarray(scores, dtype=float)
        n = len(scores_arr)
        if n == 0:
            return {
                "mean": None,
                "std": None,
                "ci_lower": None,
                "ci_upper": None,
                "n_samples": 0,
                "method": "bootstrap_percentile",
            }

        rng = np.random.default_rng(random_state)
        boot_means = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            sample = rng.choice(scores_arr, size=n, replace=True)
            boot_means[i] = sample.mean()

        alpha = 1.0 - confidence
        ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

        return {
            "mean": float(np.mean(scores_arr)),
            "std": float(np.std(scores_arr, ddof=1)) if n > 1 else 0.0,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_samples": n,
            "method": "bootstrap_percentile",
        }

    # ------------------------------------------------------------------
    # Wilcoxon signed-rank test
    # ------------------------------------------------------------------

    def wilcoxon_test(
        self,
        scores_a: List[float],
        scores_b: List[float],
    ) -> Optional[dict]:
        """Wilcoxon signed-rank test comparing two paired score lists.

        Returns ``None`` with an explanation when the test cannot be run
        (e.g. insufficient samples, identical distributions).

        Parameters
        ----------
        scores_a, scores_b:
            Paired observations (must have the same length).

        Returns
        -------
        dict | None
            ``None`` if the test cannot be performed.  Otherwise a dict with:
            statistic, p_value, significant (bool, p < 0.05),
            effect_size_r, interpretation.
        """
        a = np.asarray(scores_a, dtype=float)
        b = np.asarray(scores_b, dtype=float)
        n = len(a)

        if len(a) != len(b):
            return {
                "statistic": None,
                "p_value": None,
                "significant": False,
                "effect_size_r": None,
                "interpretation": (
                    f"Cannot run Wilcoxon test: score lists have different lengths "
                    f"({len(a)} vs {len(b)}). Paired test requires equal-length inputs."
                ),
            }

        if n < _MIN_SAMPLES_WILCOXON:
            return {
                "statistic": None,
                "p_value": None,
                "significant": False,
                "effect_size_r": None,
                "interpretation": (
                    f"Insufficient samples for Wilcoxon test (n={n}, "
                    f"minimum required: {_MIN_SAMPLES_WILCOXON}). "
                    "No statistical comparison is possible at this sample size."
                ),
            }

        if not _SCIPY_AVAILABLE:
            return {
                "statistic": None,
                "p_value": None,
                "significant": False,
                "effect_size_r": None,
                "interpretation": (
                    "scipy is not installed. "
                    "Install with `pip install scipy` to enable Wilcoxon tests."
                ),
            }

        differences = a - b
        if np.all(differences == 0):
            return {
                "statistic": None,
                "p_value": 1.0,
                "significant": False,
                "effect_size_r": 0.0,
                "interpretation": (
                    "All paired differences are zero — the two distributions are "
                    "identical at this precision. No difference detected."
                ),
            }

        try:
            result = _scipy_wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
            stat = float(result.statistic)
            p_val = float(result.pvalue)
        except Exception as exc:
            return {
                "statistic": None,
                "p_value": None,
                "significant": False,
                "effect_size_r": None,
                "interpretation": f"Wilcoxon test raised an exception: {exc}",
            }

        # Effect size r = z / sqrt(N), approximated via normal approximation.
        # For the signed-rank test: z ≈ Φ⁻¹(1 - p/2) * sign(stat - expected)
        # We use a simpler formula: r = abs(z) / sqrt(n).
        # z is approximated from the p-value via the inverse normal CDF.
        try:
            from scipy.stats import norm  # type: ignore

            z = abs(norm.ppf(max(p_val / 2, 1e-16)))
        except Exception:
            z = 0.0
        effect_r = z / math.sqrt(n) if n > 0 else 0.0

        significant = p_val < 0.05
        if significant:
            interp = (
                f"Statistically significant difference (p={p_val:.4f} < 0.05, "
                f"effect size r={effect_r:.3f})."
            )
        else:
            interp = (
                f"No statistically significant difference detected "
                f"(p={p_val:.4f} ≥ 0.05, n={n}, effect size r={effect_r:.3f}). "
                "Cannot reject the null hypothesis of equal performance."
            )

        return {
            "statistic": stat,
            "p_value": p_val,
            "significant": significant,
            "effect_size_r": effect_r,
            "interpretation": interp,
        }

    # ------------------------------------------------------------------
    # Multi-model pairwise comparison
    # ------------------------------------------------------------------

    def compare_models_statistically(
        self,
        model_scores: Dict[str, List[float]],
    ) -> dict:
        """Pairwise Wilcoxon tests with Bonferroni correction.

        Parameters
        ----------
        model_scores:
            Mapping of model name → list of cross-validated scores.

        Returns
        -------
        dict with keys:
            comparison_matrix : dict[str, dict[str, dict]]
                Nested dict: matrix[model_a][model_b] = test result dict.
            significant_differences : list[dict]
                Each entry: {model_a, model_b, p_value, bonferroni_p_value,
                significant_after_correction, effect_size_r, mean_a, mean_b}.
            n_comparisons : int
            bonferroni_alpha : float
        """
        model_names = list(model_scores.keys())
        k = len(model_names)
        n_comparisons = k * (k - 1) // 2
        bonferroni_alpha = 0.05 / n_comparisons if n_comparisons > 0 else 0.05

        matrix: Dict[str, Dict[str, dict]] = {m: {} for m in model_names}
        significant: list[dict] = []

        for i, name_a in enumerate(model_names):
            for j, name_b in enumerate(model_names):
                if i >= j:
                    continue
                scores_a = model_scores[name_a]
                scores_b = model_scores[name_b]
                raw = self.wilcoxon_test(scores_a, scores_b)

                if raw is None:
                    result = {
                        "statistic": None,
                        "p_value": None,
                        "significant": False,
                        "effect_size_r": None,
                        "bonferroni_p_value": None,
                        "significant_after_correction": False,
                        "interpretation": "Test could not be run.",
                    }
                else:
                    bonf_p = (
                        min(raw["p_value"] * n_comparisons, 1.0)
                        if raw["p_value"] is not None
                        else None
                    )
                    sig_corrected = bonf_p is not None and bonf_p < 0.05
                    result = {**raw, "bonferroni_p_value": bonf_p, "significant_after_correction": sig_corrected}

                matrix[name_a][name_b] = result
                matrix[name_b][name_a] = result  # symmetric

                if result.get("significant_after_correction"):
                    a_arr = np.asarray(scores_a)
                    b_arr = np.asarray(scores_b)
                    significant.append({
                        "model_a": name_a,
                        "model_b": name_b,
                        "p_value": result["p_value"],
                        "bonferroni_p_value": result["bonferroni_p_value"],
                        "significant_after_correction": True,
                        "effect_size_r": result["effect_size_r"],
                        "mean_a": float(np.mean(a_arr)),
                        "mean_b": float(np.mean(b_arr)),
                    })

        return {
            "comparison_matrix": matrix,
            "significant_differences": significant,
            "n_comparisons": n_comparisons,
            "bonferroni_alpha": bonferroni_alpha,
        }

    # ------------------------------------------------------------------
    # Greenness comparison
    # ------------------------------------------------------------------

    def is_demonstrably_greener(
        self,
        energy_a: List[float],
        energy_b: List[float],
    ) -> Tuple[bool, str]:
        """Test whether model A uses demonstrably less energy than model B.

        This method NEVER claims greener without statistical evidence.

        Parameters
        ----------
        energy_a:
            Repeated energy measurements for model A (kWh or any consistent unit).
        energy_b:
            Repeated energy measurements for model B.

        Returns
        -------
        (bool, str)
            True  + explanation if the difference is statistically significant
                    at p < 0.05.
            False + explanation otherwise, naming the reason (not enough data,
                    p too large, etc.).
        """
        a_arr = np.asarray(energy_a, dtype=float)
        b_arr = np.asarray(energy_b, dtype=float)

        n_a = len(a_arr)
        n_b = len(b_arr)
        mean_a = float(np.mean(a_arr)) if n_a > 0 else float("nan")
        mean_b = float(np.mean(b_arr)) if n_b > 0 else float("nan")

        result = self.wilcoxon_test(list(energy_a), list(energy_b))

        if result is None or result["p_value"] is None:
            reason = result["interpretation"] if result else "Test could not be run."
            return (
                False,
                f"Cannot determine which model is greener: {reason} "
                f"(mean energy A={mean_a:.6g}, B={mean_b:.6g}, n_a={n_a}, n_b={n_b})",
            )

        if result["significant"] and mean_a < mean_b:
            return (
                True,
                f"statistically significant at p={result['p_value']:.4f} < 0.05 "
                f"with effect size r={result['effect_size_r']:.3f} "
                f"(mean energy A={mean_a:.6g} < B={mean_b:.6g}, n={n_a} measurements)",
            )

        if result["significant"] and mean_a >= mean_b:
            # B is actually greener
            return (
                False,
                f"Model B appears greener (mean energy B={mean_b:.6g} < A={mean_a:.6g}), "
                f"and the difference IS statistically significant "
                f"(p={result['p_value']:.4f}, r={result['effect_size_r']:.3f}). "
                "Model A is NOT demonstrably greener.",
            )

        # Not significant
        direction = "lower" if mean_a < mean_b else "higher"
        return (
            False,
            f"{direction} estimated energy in this experiment for model A "
            f"(mean A={mean_a:.6g} vs B={mean_b:.6g}), but difference not "
            f"statistically significant at n={n_a} measurements "
            f"(p={result['p_value']:.4f} ≥ 0.05). "
            "More repeated measurements or a larger workload are needed to "
            "make a confident claim about relative energy consumption.",
        )
