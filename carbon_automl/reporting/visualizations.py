"""
Visualizer: Clean, minimal matplotlib visualizations for Carbon-Aware AutoML.

All charts save at 150 DPI using pathlib.Path and avoid excessive gradients
or decorative clutter.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib
if "matplotlib.pyplot" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _get_clean_style() -> str:
    """Return seaborn whitegrid style name if available, else 'default'."""
    for candidate in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
        if candidate in plt.style.available:
            return candidate
    return "default"


def _prep_output_path(output_path: str | Path) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _apply_minimal_axes(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)


class Visualizer:
    """Generates publication-ready minimal visualizations at 150 DPI."""

    _DPI = 150

    def __init__(self, output_dir: str | Path | None = None) -> None:
        self.output_dir = Path(output_dir) if output_dir is not None else None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Model Comparison
    # ------------------------------------------------------------------

    @staticmethod
    def plot_model_comparison(
        model_results: list[dict],
        metrics: list[str],
        output_path: str,
    ) -> str:
        """Plot grouped or multi-panel bar chart comparing models across metrics."""
        out = _prep_output_path(output_path)
        style = _get_clean_style()

        with plt.style.context(style):
            if not model_results or not metrics:
                fig, ax = plt.subplots(figsize=(7, 4))
                try:
                    ax.text(0.5, 0.5, "No model comparison data available", ha="center", va="center")
                    ax.axis("off")
                    fig.savefig(out, dpi=Visualizer._DPI, bbox_inches="tight")
                finally:
                    plt.close(fig)
                return str(out.resolve())

            names = [str(m.get("name") or m.get("model_name") or f"model_{i}") for i, m in enumerate(model_results)]
            n_metrics = len(metrics)
            n_models = len(names)

            fig, axes = plt.subplots(
                1,
                n_metrics,
                figsize=(max(5.5, 3.8 * n_metrics), max(4.0, 0.55 * n_models + 2.2)),
                squeeze=False,
            )
            try:
                palette = ["#2c5f8a", "#2e8b57", "#d97706", "#6b7280", "#8b5cf6"]
                for idx, metric in enumerate(metrics):
                    ax = axes[0, idx]
                    vals: list[float] = []
                    for m in model_results:
                        v = m.get(metric)
                        if v is None and isinstance(m.get("metrics"), dict):
                            v = m["metrics"].get(metric)
                        vals.append(float(v) if v is not None else 0.0)

                    y_pos = np.arange(n_models)
                    color = palette[idx % len(palette)]
                    bars = ax.barh(y_pos, vals, color=color, edgecolor="none", height=0.6)
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(names if idx == 0 else [])
                    ax.invert_yaxis()
                    ax.set_xlabel(metric)
                    ax.set_title(metric.replace("_", " ").title(), fontsize=10, fontweight="bold")
                    _apply_minimal_axes(ax)

                    for bar, val in zip(bars, vals):
                        label = f"{val:.4g}"
                        ax.text(
                            bar.get_width(),
                            bar.get_y() + bar.get_height() / 2.0,
                            f" {label}",
                            va="center",
                            ha="left",
                            fontsize=8,
                        )

                fig.suptitle("Model Comparison", fontsize=12, fontweight="bold", y=1.02)
                fig.tight_layout()
                fig.savefig(out, dpi=Visualizer._DPI, bbox_inches="tight")
            finally:
                plt.close(fig)

        return str(out.resolve())

    # ------------------------------------------------------------------
    # 2. Energy by Phase
    # ------------------------------------------------------------------

    @staticmethod
    def plot_energy_by_phase(
        phase_measurements: dict,
        output_path: str,
    ) -> str:
        """Plot horizontal bar chart of energy consumption (kWh) per pipeline phase."""
        out = _prep_output_path(output_path)
        style = _get_clean_style()

        with plt.style.context(style):
            fig, ax = plt.subplots(figsize=(7, 4.2))
            try:
                if not phase_measurements:
                    ax.text(0.5, 0.5, "No phase energy measurements available", ha="center", va="center")
                    ax.axis("off")
                    fig.savefig(out, dpi=Visualizer._DPI, bbox_inches="tight")
                    return str(out.resolve())

                phases: list[str] = []
                energies: list[float] = []
                estimated_flags: list[bool] = []

                for phase_name, m in phase_measurements.items():
                    phases.append(str(phase_name))
                    if isinstance(m, (int, float)):
                        energies.append(float(m))
                        estimated_flags.append(True)
                    elif hasattr(m, "energy_kwh"):
                        energies.append(float(m.energy_kwh or 0.0))
                        estimated_flags.append(bool(getattr(m, "is_estimated", True)))
                    elif isinstance(m, dict):
                        val = m.get("energy_kwh", m.get("training_energy_kwh", 0.0))
                        energies.append(float(val or 0.0))
                        estimated_flags.append(bool(m.get("is_estimated", True)))
                    else:
                        energies.append(0.0)
                        estimated_flags.append(True)

                y_pos = np.arange(len(phases))
                colors = ["#d97706" if est else "#2e8b57" for est in estimated_flags]
                bars = ax.barh(y_pos, energies, color=colors, height=0.55)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(phases)
                ax.invert_yaxis()
                ax.set_xlabel("Energy (kWh)")
                ax.set_title("Energy Consumption by Phase (Orange = Estimated, Green = Measured)", fontsize=10, fontweight="bold")
                _apply_minimal_axes(ax)

                for bar, val in zip(bars, energies):
                    ax.text(
                        bar.get_width(),
                        bar.get_y() + bar.get_height() / 2.0,
                        f" {val:.3e} kWh",
                        va="center",
                        ha="left",
                        fontsize=8,
                    )

                fig.tight_layout()
                fig.savefig(out, dpi=Visualizer._DPI, bbox_inches="tight")
            finally:
                plt.close(fig)

        return str(out.resolve())

    # ------------------------------------------------------------------
    # 3. Pareto Frontier
    # ------------------------------------------------------------------

    @staticmethod
    def plot_pareto_frontier(
        models: list[dict],
        quality_metric: str,
        energy_metric: str,
        output_path: str,
    ) -> str:
        """Scatter plot of quality vs energy with Pareto-optimal frontier highlighted."""
        out = _prep_output_path(output_path)
        style = _get_clean_style()

        with plt.style.context(style):
            fig, ax = plt.subplots(figsize=(7, 5))
            try:
                if not models:
                    ax.text(0.5, 0.5, "No model data for Pareto plot", ha="center", va="center")
                    ax.axis("off")
                    fig.savefig(out, dpi=Visualizer._DPI, bbox_inches="tight")
                    return str(out.resolve())

                def _get_val(m: dict, key: str, fallback: float = 0.0) -> float:
                    val = m.get(key)
                    if val is None and isinstance(m.get("metrics"), dict):
                        val = m["metrics"].get(key)
                    return float(val) if val is not None else fallback

                names = [str(m.get("name") or m.get("model_name") or f"model_{i}") for i, m in enumerate(models)]
                qualities = [_get_val(m, quality_metric) for m in models]
                energies = [_get_val(m, energy_metric) for m in models]

                # Determine Pareto optimality (maximize quality, minimize energy)
                is_pareto: list[bool] = []
                for i in range(len(models)):
                    dominated = False
                    for j in range(len(models)):
                        if i == j:
                            continue
                        if (
                            qualities[j] >= qualities[i]
                            and energies[j] <= energies[i]
                            and (qualities[j] > qualities[i] or energies[j] < energies[i])
                        ):
                            dominated = True
                            break
                    is_pareto.append(not dominated)

                # Non-Pareto points
                non_p_e = [e for e, p in zip(energies, is_pareto) if not p]
                non_p_q = [q for q, p in zip(qualities, is_pareto) if not p]
                if non_p_e:
                    ax.scatter(non_p_e, non_p_q, color="#64748b", s=70, label="Dominated", zorder=3)

                # Pareto points sorted by energy ascending
                p_pts = sorted(
                    [(e, q, n) for e, q, n, p in zip(energies, qualities, names, is_pareto) if p],
                    key=lambda t: (t[0], -t[1]),
                )
                if p_pts:
                    pe, pq, _ = zip(*p_pts)
                    ax.plot(pe, pq, color="#15803d", linestyle="--", linewidth=1.5, zorder=2)
                    ax.scatter(pe, pq, color="#16a34a", s=100, edgecolor="#14532d", linewidth=1.2, label="Pareto Optimal", zorder=4)

                # Labels
                for e, q, name in zip(energies, qualities, names):
                    ax.annotate(
                        name,
                        (e, q),
                        textcoords="offset points",
                        xytext=(6, 4),
                        fontsize=8,
                    )

                ax.set_xlabel(energy_metric.replace("_", " "))
                ax.set_ylabel(quality_metric.replace("_", " "))
                ax.set_title("Quality vs. Energy Pareto Frontier", fontsize=11, fontweight="bold")
                ax.legend(frameon=True)
                _apply_minimal_axes(ax)
                fig.tight_layout()
                fig.savefig(out, dpi=Visualizer._DPI, bbox_inches="tight")
            finally:
                plt.close(fig)

        return str(out.resolve())

    # ------------------------------------------------------------------
    # 4. Confusion Matrix
    # ------------------------------------------------------------------

    @staticmethod
    def plot_confusion_matrix(
        cm: list[list],
        class_names: list[str],
        output_path: str,
    ) -> str:
        """Plot a clean heatmap for a confusion matrix."""
        out = _prep_output_path(output_path)
        style = _get_clean_style()

        with plt.style.context(style):
            cm_arr = np.asarray(cm)
            fig, ax = plt.subplots(figsize=(6, 5))
            try:
                if cm_arr.size == 0:
                    ax.text(0.5, 0.5, "No confusion matrix available", ha="center", va="center")
                    ax.axis("off")
                    fig.savefig(out, dpi=Visualizer._DPI, bbox_inches="tight")
                    return str(out.resolve())

                n_classes = cm_arr.shape[0]
                labels = [str(c) for c in class_names] if class_names else [str(i) for i in range(n_classes)]
                if len(labels) < n_classes:
                    labels.extend(str(i) for i in range(len(labels), n_classes))
                labels = labels[:n_classes]

                im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
                ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set(
                    xticks=np.arange(n_classes),
                    yticks=np.arange(n_classes),
                    xticklabels=labels,
                    yticklabels=labels,
                    ylabel="True Label",
                    xlabel="Predicted Label",
                    title="Confusion Matrix",
                )
                ax.grid(False)
                plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

                thresh = cm_arr.max() / 2.0 if cm_arr.size else 0.0
                for i in range(cm_arr.shape[0]):
                    for j in range(cm_arr.shape[1]):
                        ax.text(
                            j,
                            i,
                            f"{cm_arr[i, j]}",
                            ha="center",
                            va="center",
                            color="white" if cm_arr[i, j] > thresh else "black",
                            fontsize=9,
                        )

                fig.tight_layout()
                fig.savefig(out, dpi=Visualizer._DPI, bbox_inches="tight")
            finally:
                plt.close(fig)

        return str(out.resolve())

    # ------------------------------------------------------------------
    # 5. Feature Importance
    # ------------------------------------------------------------------

    @staticmethod
    def plot_feature_importance(
        importances: dict,
        output_path: str,
    ) -> str:
        """Plot horizontal bar chart of feature importances (top 20)."""
        out = _prep_output_path(output_path)
        style = _get_clean_style()

        # Unwrap nested dicts if caller passed FeatureSelector output directly
        if "importances_mean" in importances and isinstance(importances["importances_mean"], dict):
            imp_map = importances["importances_mean"]
        elif "scores" in importances and isinstance(importances["scores"], dict):
            imp_map = importances["scores"]
        else:
            imp_map = importances

        with plt.style.context(style):
            items = [
                (str(k), float(v))
                for k, v in imp_map.items()
                if isinstance(v, (int, float, np.number))
            ]
            items.sort(key=lambda t: abs(t[1]), reverse=True)
            items = items[:20]

            fig, ax = plt.subplots(figsize=(7, max(4.0, 0.35 * len(items) + 1.5)))
            try:
                if not items:
                    ax.text(0.5, 0.5, "No feature importance data available", ha="center", va="center")
                    ax.axis("off")
                    fig.savefig(out, dpi=Visualizer._DPI, bbox_inches="tight")
                    return str(out.resolve())

                names, values = zip(*items)
                y_pos = np.arange(len(names))
                colors = ["#2c5f8a" if v >= 0 else "#dc2626" for v in values]
                ax.barh(y_pos, values, color=colors, height=0.6)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(names)
                ax.invert_yaxis()
                ax.set_xlabel("Importance")
                ax.set_title("Feature Importances", fontsize=11, fontweight="bold")
                _apply_minimal_axes(ax)
                fig.tight_layout()
                fig.savefig(out, dpi=Visualizer._DPI, bbox_inches="tight")
            finally:
                plt.close(fig)

        return str(out.resolve())

    # ------------------------------------------------------------------
    # 6. Ablation Results
    # ------------------------------------------------------------------

    @staticmethod
    def plot_ablation_results(
        ablation_results: dict,
        output_path: str,
    ) -> str:
        """Plot bar chart comparing ablation conditions."""
        out = _prep_output_path(output_path)
        style = _get_clean_style()

        conditions = ablation_results.get("conditions", ablation_results)
        labels: list[str] = []
        scores: list[float] = []

        if isinstance(conditions, dict):
            for cond_name, val in conditions.items():
                if isinstance(val, (int, float, np.number)):
                    labels.append(str(cond_name))
                    scores.append(float(val))
                elif isinstance(val, dict):
                    score = (
                        val.get("score")
                        if val.get("score") is not None
                        else val.get("macro_f1")
                        if val.get("macro_f1") is not None
                        else val.get("r2")
                        if val.get("r2") is not None
                        else val.get("mean_score")
                    )
                    if score is not None:
                        labels.append(str(cond_name))
                        scores.append(float(score))

        with plt.style.context(style):
            fig, ax = plt.subplots(figsize=(7.5, max(4.0, 0.5 * len(labels) + 1.5)))
            try:
                if not labels:
                    ax.text(0.5, 0.5, "No ablation data available", ha="center", va="center")
                    ax.axis("off")
                    fig.savefig(out, dpi=Visualizer._DPI, bbox_inches="tight")
                    return str(out.resolve())

                y_pos = np.arange(len(labels))
                bars = ax.barh(y_pos, scores, color="#3b82f6", height=0.55)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels)
                ax.invert_yaxis()
                ax.set_xlabel("Validation Score")
                ax.set_title("Ablation Study Results", fontsize=11, fontweight="bold")
                _apply_minimal_axes(ax)

                for bar, val in zip(bars, scores):
                    ax.text(
                        bar.get_width(),
                        bar.get_y() + bar.get_height() / 2.0,
                        f" {val:.4f}",
                        va="center",
                        ha="left",
                        fontsize=8,
                    )

                fig.tight_layout()
                fig.savefig(out, dpi=Visualizer._DPI, bbox_inches="tight")
            finally:
                plt.close(fig)

        return str(out.resolve())
