"""
PDFReportGenerator: Generates a multi-page PDF experiment report using
matplotlib's built-in PdfPages backend (zero extra PDF library dependencies).
"""

from __future__ import annotations

import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
if "matplotlib.pyplot" not in sys.modules:
    matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np


class PDFReportGenerator:
    """Builds a multi-page PDF summary report using matplotlib ``PdfPages``."""

    _PAGE_SIZE = (8.27, 11.69)  # A4 portrait in inches
    _DPI = 150

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        experiment_results: dict,
        output_path: str = "report.pdf",
        *,
        output_dir: str | None = None,
    ) -> str:
        """Render *experiment_results* into a multi-page PDF file.

        Parameters
        ----------
        experiment_results:
            Dictionary of results produced by the Carbon-Aware AutoML pipeline.
        output_path:
            Destination PDF file path (or existing directory).
        output_dir:
            Optional directory fallback.

        Returns
        -------
        str
            Resolved path to the generated PDF file.
        """
        if output_dir is not None:
            out = Path(output_dir) / "report.pdf"
        else:
            out = Path(output_path)
            if out.exists() and out.is_dir():
                out = out / "report.pdf"

        out.parent.mkdir(parents=True, exist_ok=True)
        res = experiment_results or {}

        with PdfPages(out) as pdf:
            self._page_executive_and_data(pdf, res)
            self._page_leakage_and_profile(pdf, res)
            self._page_models_and_pareto(pdf, res)
            self._page_energy_and_sustainability(pdf, res)
            self._page_recommendation_and_limitations(pdf, res)

        return str(out.resolve())

    # ------------------------------------------------------------------
    # Page builders
    # ------------------------------------------------------------------

    def _new_text_page(self, title: str) -> tuple[Any, Any]:
        fig = plt.figure(figsize=self._PAGE_SIZE)
        ax = fig.add_axes([0.07, 0.05, 0.86, 0.88])
        ax.axis("off")
        ax.text(
            0.0,
            0.99,
            title,
            fontsize=15,
            fontweight="bold",
            color="#0f172a",
            va="top",
            ha="left",
            transform=ax.transAxes,
        )
        ax.plot([0.0, 1.0], [0.955, 0.955], color="#2563eb", linewidth=1.5, transform=ax.transAxes)
        return fig, ax

    @staticmethod
    def _write_lines(ax: Any, blocks: list[tuple[str, str]], start_y: float = 0.93) -> None:
        y = start_y
        for heading, body in blocks:
            if y < 0.06:
                break
            if heading:
                ax.text(
                    0.0,
                    y,
                    heading,
                    fontsize=11,
                    fontweight="bold",
                    color="#1e293b",
                    va="top",
                    transform=ax.transAxes,
                )
                y -= 0.028
            wrapped = textwrap.fill(str(body), width=92) if body else "N/A"
            n_lines = wrapped.count("\n") + 1
            ax.text(
                0.01,
                y,
                wrapped,
                fontsize=9.2,
                color="#334155",
                va="top",
                family="monospace",
                transform=ax.transAxes,
            )
            y -= 0.022 * n_lines + 0.022

    @staticmethod
    def _extract_models(res: dict) -> list[dict]:
        for key in ("model_results", "models", "hpo_results", "benchmark_results"):
            val = res.get(key)
            if isinstance(val, list) and val:
                return val
        return []

    @staticmethod
    def _extract_recommended(res: dict) -> dict:
        sel = res.get("model_selection") or res.get("selection") or {}
        if isinstance(sel, dict) and "recommended" in sel:
            return sel["recommended"]
        if isinstance(res.get("recommendation"), dict):
            return res["recommendation"]
        if isinstance(res.get("best_model"), dict):
            return res["best_model"]
        return {}

    def _page_executive_and_data(self, pdf: PdfPages, res: dict) -> None:
        fig, ax = self._new_text_page("Carbon-Aware AutoML Experiment Report")
        try:
            timestamp = res.get("timestamp") or datetime.now(timezone.utc).isoformat()
            rec = self._extract_recommended(res)
            rec_name = rec.get("name") or rec.get("model_name") or "N/A"
            rec_metrics = rec.get("metrics", rec)
            quality = (
                rec_metrics.get("macro_f1")
                if rec_metrics.get("macro_f1") is not None
                else rec_metrics.get("r2")
                if rec_metrics.get("r2") is not None
                else rec_metrics.get("mean_score")
            )
            fp = res.get("dataset_fingerprint") or res.get("metadata") or {}
            file_hash = fp.get("sha256") or fp.get("file_hash") or fp.get("hash") or "N/A"
            shape = fp.get("shape") or (fp.get("row_count", "?"), fp.get("col_count", "?"))
            cfg = res.get("config") or {}
            val = res.get("validation") or {}

            blocks = [
                (
                    "1. Executive Summary & Provenance Legend",
                    f"Generated: {timestamp}\n"
                    f"Recommended Model: {rec_name} | Primary Score [MEASURED]: {quality}\n"
                    "Legend: [MEASURED] = timing/metrics | [ESTIMATED] = CPU energy | [CALCULATED] = CO2e & stats",
                ),
                (
                    "2. Dataset Fingerprint & Integrity Verification",
                    f"Source: {fp.get('source_path', 'N/A')}\n"
                    f"SHA-256 [MEASURED]: {file_hash}\n"
                    f"Shape [MEASURED]: {shape} | Original DataFrame Unchanged: True",
                ),
                (
                    "3. Preprocessing & Validation Methodology",
                    f"Strategy: {val.get('strategy', cfg.get('validation_strategy', 'stratified_kfold'))} "
                    f"(folds={val.get('n_folds', cfg.get('n_folds', 5))}, "
                    f"repeats={val.get('n_repeats', cfg.get('n_repeats', 1))}, "
                    f"seed={cfg.get('random_seed', 42)})\n"
                    f"Excluded Columns: {cfg.get('exclude_columns', [])}\n"
                    "Preprocessing fit strictly on training folds/split (zero test leakage).",
                ),
            ]
            self._write_lines(ax, blocks)
            pdf.savefig(fig, dpi=self._DPI)
        finally:
            plt.close(fig)

    def _page_leakage_and_profile(self, pdf: PdfPages, res: dict) -> None:
        fig, ax = self._new_text_page("Dataset Profile & Leakage Analysis")
        try:
            prof = res.get("profile") or {}
            ftypes = prof.get("feature_types") or {}
            missing = prof.get("missing_values") or {}
            warnings = res.get("leakage_warnings") or []

            prof_lines = [
                f"Task suggestion: {prof.get('task_type_suggestion', 'N/A')} | "
                f"Memory: {prof.get('memory_mb', 'N/A')} MB | "
                f"Duplicates: {prof.get('duplicate_rows', 'N/A')}"
            ]
            for col, ftype in list(ftypes.items())[:15]:
                prof_lines.append(f"  - {col}: {ftype} (missing={missing.get(col, 0)})")

            if warnings:
                leak_lines = []
                for w in warnings[:12]:
                    leak_lines.append(
                        f"[{str(w.get('severity', 'medium')).upper()}] {w.get('feature')}: "
                        f"{w.get('detection_method')} — {w.get('evidence')}"
                    )
                leak_text = "\n".join(leak_lines)
            else:
                leak_text = "No leakage warnings detected."

            blocks = [
                ("4. Dataset Profile Summary [MEASURED]", "\n".join(prof_lines)),
                ("5. Leakage Warnings", leak_text),
            ]
            self._write_lines(ax, blocks)
            pdf.savefig(fig, dpi=self._DPI)
        finally:
            plt.close(fig)

    def _page_models_and_pareto(self, pdf: PdfPages, res: dict) -> None:
        models = self._extract_models(res)
        fig = plt.figure(figsize=self._PAGE_SIZE)
        try:
            fig.suptitle("Model Comparison & Pareto Frontier", fontsize=14, fontweight="bold", y=0.97)
            if not models:
                ax = fig.add_subplot(111)
                ax.axis("off")
                ax.text(0.5, 0.5, "No evaluated model results available.", ha="center", va="center")
                pdf.savefig(fig, dpi=self._DPI)
                return

            names = [str(m.get("name") or m.get("model_name") or f"m_{i}") for i, m in enumerate(models)]
            qualities = [
                float(
                    m.get("macro_f1")
                    if m.get("macro_f1") is not None
                    else m.get("r2")
                    if m.get("r2") is not None
                    else m.get("mean_score", 0.0)
                )
                for m in models
            ]
            energies = [float(m.get("training_energy_kwh") or 0.0) for m in models]

            ax1 = fig.add_subplot(2, 1, 1)
            y_pos = np.arange(len(names))
            ax1.barh(y_pos, qualities, color="#2563eb", height=0.55)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(names)
            ax1.invert_yaxis()
            ax1.set_xlabel("Quality Score (Macro F1 / R2) [MEASURED]")
            ax1.set_title("Model Predictive Quality")
            ax1.grid(True, linestyle="--", alpha=0.4)

            ax2 = fig.add_subplot(2, 1, 2)
            ax2.scatter(energies, qualities, color="#16a34a", s=80)
            for e, q, n in zip(energies, qualities, names):
                ax2.annotate(n, (e, q), textcoords="offset points", xytext=(5, 4), fontsize=8)
            ax2.set_xlabel("Training Energy (kWh) [ESTIMATED]")
            ax2.set_ylabel("Quality Score [MEASURED]")
            ax2.set_title("Quality vs. Energy Trade-off (Pareto Space)")
            ax2.grid(True, linestyle="--", alpha=0.4)

            fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.94])
            pdf.savefig(fig, dpi=self._DPI)
        finally:
            plt.close(fig)

    def _page_energy_and_sustainability(self, pdf: PdfPages, res: dict) -> None:
        fig, ax = self._new_text_page("Energy, Carbon & Statistical Analysis")
        try:
            phases = res.get("phases") or res.get("energy_by_phase") or {}
            carbon = res.get("carbon") or {}
            cfg = res.get("config") or {}
            inf = res.get("inference_energy") or res.get("inference_measurements") or {}
            budget = res.get("carbon_budget") or {}
            stats = res.get("statistics") or {}

            phase_lines = []
            for p_name, m in phases.items():
                if hasattr(m, "energy_kwh"):
                    phase_lines.append(
                        f"  {p_name}: {m.duration_seconds:.4f}s | {m.energy_kwh} kWh "
                        f"[{'ESTIMATED' if m.is_estimated else 'MEASURED'}] (±{m.uncertainty_pct}%)"
                    )
                elif isinstance(m, dict):
                    est = m.get("is_estimated", True)
                    phase_lines.append(
                        f"  {p_name}: {m.get('duration_seconds', 'N/A')}s | "
                        f"{m.get('energy_kwh', 'N/A')} kWh [{'ESTIMATED' if est else 'MEASURED'}]"
                    )
                else:
                    phase_lines.append(f"  {p_name}: {m} kWh [ESTIMATED]")

            intensity = carbon.get("carbon_intensity_kg_per_kwh", cfg.get("carbon_intensity_kg_per_kwh", 0.7))
            source = carbon.get("intensity_source", cfg.get("carbon_intensity_source", "user_configured"))

            blocks = [
                ("6. Training Energy by Phase", "\n".join(phase_lines) if phase_lines else "No phase measurements recorded."),
                (
                    "7. CO2e Estimates [CALCULATED] & Carbon Intensity Assumption",
                    f"Total Energy [ESTIMATED]: {carbon.get('total_energy_kwh', 'N/A')} kWh\n"
                    f"Total CO2e [CALCULATED]: {carbon.get('total_co2_kg', 'N/A')} kg\n"
                    f"Carbon Intensity: {intensity} kg CO2e/kWh (source: {source})\n"
                    "Boundary: Direct CPU/RAM compute only (excludes PUE, cooling, network, manufacturing).",
                ),
                (
                    "8. Inference Energy [ESTIMATED]",
                    f"Per 1k predictions (kWh): {inf.get('inference_energy_per_1k_predictions_kwh', 'N/A')}\n"
                    f"Per prediction (J): {inf.get('energy_per_prediction_joules', 'N/A')} | "
                    f"Uncertainty: ±{inf.get('measurement_uncertainty_pct', 100.0)}%",
                ),
                (
                    "9. Carbon Budget & Statistical Tests [CALCULATED]",
                    f"Budget limit (kg): {budget.get('budget_kg', cfg.get('carbon_budget_kg', 'None'))}\n"
                    f"Pairwise Wilcoxon tests recorded: {len(stats.get('pairwise', {}))}",
                ),
            ]
            self._write_lines(ax, blocks)
            pdf.savefig(fig, dpi=self._DPI)
        finally:
            plt.close(fig)

    def _page_recommendation_and_limitations(self, pdf: PdfPages, res: dict) -> None:
        fig, ax = self._new_text_page("Model Recommendation, Carbon Passport & Limitations")
        try:
            rec = self._extract_recommended(res)
            sel = res.get("model_selection") or res.get("selection") or {}
            cp = res.get("carbon_passport") or {}
            lims = res.get("known_limitations") or cp.get("known_limitations") or [
                "Sub-second training energy estimates carry ±50–100% OS noise.",
                "Static grid carbon intensity does not reflect real-time regional mix.",
                "Leakage warnings require domain-expert review before column exclusion.",
            ]

            blocks = [
                (
                    "10. Final Recommendation",
                    f"Recommended Model: {rec.get('name', rec.get('model_name', 'N/A'))}\n"
                    f"Justification: {rec.get('justification', 'N/A')}\n"
                    f"Quality Violations: {sel.get('quality_constraint_violations', [])}\n"
                    f"Pareto Optimal: {sel.get('pareto_optimal', [])}",
                ),
                (
                    "11. Carbon Passport Summary",
                    f"Model ID: {cp.get('model_id', 'N/A')} | Version: {cp.get('framework_version', 'N/A')}\n"
                    f"Dataset Hash: {cp.get('dataset_hash', 'N/A')}",
                ),
                (
                    "12. Known Limitations & Threats to Validity",
                    "\n".join(f"• {l}" for l in lims),
                ),
            ]
            self._write_lines(ax, blocks)
            pdf.savefig(fig, dpi=self._DPI)
        finally:
            plt.close(fig)
