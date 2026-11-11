"""Tests for HTMLReportGenerator, PDFReportGenerator, Visualizer, and DriftDetector."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from carbon_automl.intelligence.drift_detector import DriftDetector
from carbon_automl.reporting.html_report import HTMLReportGenerator
from carbon_automl.reporting.pdf_report import PDFReportGenerator
from carbon_automl.reporting.visualizations import Visualizer


def _sample_experiment_results() -> dict:
    return {
        "title": "Test AutoML Experiment",
        "timestamp": "2026-11-12T00:00:00Z",
        "dataset_fingerprint": {"sha256": "deadbeef", "shape": (100, 5)},
        "profile": {
            "shape": (100, 5),
            "memory_mb": 0.01,
            "duplicate_rows": 0,
            "feature_types": {"num_1": "numerical", "cat_1": "categorical"},
            "missing_values": {"num_1": 0, "cat_1": 0},
            "outliers": {"num_1": 0},
            "task_type_suggestion": "classification",
        },
        "leakage_warnings": [
            {
                "feature": "leaky_col",
                "severity": "critical",
                "detection_method": "single_feature_performance",
                "evidence": "Accuracy 1.0000 alone",
                "recommended_investigation": "Exclude column",
            }
        ],
        "model_results": [
            {
                "name": "dummy",
                "macro_f1": 0.33,
                "accuracy": 0.34,
                "balanced_accuracy": 0.33,
                "training_duration_seconds": 0.01,
                "training_energy_kwh": 1e-6,
                "training_co2_kg": 7e-7,
            },
            {
                "name": "random_forest",
                "macro_f1": 0.89,
                "accuracy": 0.90,
                "balanced_accuracy": 0.88,
                "training_duration_seconds": 0.2,
                "training_energy_kwh": 2e-5,
                "training_co2_kg": 1.4e-5,
            },
        ],
        "confusion_matrix": [[10, 1], [2, 12]],
        "class_names": ["0", "1"],
        "per_class_recall": {"0": 0.909, "1": 0.857},
        "phases": {
            "training": {
                "duration_seconds": 0.2,
                "energy_kwh": 2e-5,
                "is_estimated": True,
                "measurement_method": "estimated_from_duration",
                "uncertainty_pct": 100.0,
            }
        },
        "selection": {
            "recommended": {
                "name": "random_forest",
                "metrics": {"macro_f1": 0.89},
                "justification": "Best quality within energy constraints.",
            }
        },
    }


def test_html_report_self_contained(tmp_path: Path) -> None:
    """HTMLReportGenerator produces valid offline HTML with all sections and no CDN deps."""
    res = _sample_experiment_results()
    out_path = HTMLReportGenerator().generate(res, str(tmp_path / "report.html"))

    html_text = Path(out_path).read_text(encoding="utf-8")
    assert Path(out_path).exists()
    assert "<style>" in html_text
    assert "MEASURED" in html_text
    assert "ESTIMATED" in html_text
    assert "CALCULATED" in html_text
    assert "1. Executive Summary" in html_text
    assert "24. Known Limitations" in html_text
    assert not re.search(r'<link[^>]+href=["\']https?://', html_text, flags=re.IGNORECASE)
    assert not re.search(r'<script[^>]+src=["\']https?://', html_text, flags=re.IGNORECASE)


def test_pdf_report_generated(tmp_path: Path) -> None:
    """PDFReportGenerator produces a non-empty valid PDF file."""
    res = _sample_experiment_results()
    out_path = PDFReportGenerator().generate(res, str(tmp_path / "report.pdf"))

    pdf_bytes = Path(out_path).read_bytes()
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 1000


def test_visualizer_outputs(tmp_path: Path) -> None:
    """Visualizer generates clean PNG plots in tmp_path."""
    vis = Visualizer()
    models = _sample_experiment_results()["model_results"]

    p1 = vis.plot_model_comparison(models, ["macro_f1", "accuracy"], str(tmp_path / "comp.png"))
    p2 = vis.plot_pareto_frontier(models, "macro_f1", "training_energy_kwh", str(tmp_path / "pareto.png"))
    p3 = vis.plot_confusion_matrix([[10, 1], [2, 12]], ["A", "B"], str(tmp_path / "cm.png"))
    p4 = vis.plot_feature_importance({"f1": 0.7, "f2": 0.3}, str(tmp_path / "fi.png"))

    for p in (p1, p2, p3, p4):
        assert Path(p).exists()
        assert Path(p).stat().st_size > 500


def test_drift_detector() -> None:
    """DriftDetector flags shifted numerical distributions via PSI and keeps stable data clear."""
    rng = np.random.default_rng(42)
    X_ref = pd.DataFrame({
        "num_stable": rng.normal(0.0, 1.0, size=200),
        "num_shifted": rng.normal(0.0, 1.0, size=200),
    })
    X_new = pd.DataFrame({
        "num_stable": rng.normal(0.0, 1.0, size=200),
        "num_shifted": rng.normal(5.0, 1.0, size=200),
    })

    dd = DriftDetector()
    ref_stats = dd.compute_reference_stats(X_ref)
    drift = dd.detect_drift(X_new, ref_stats)

    assert drift["num_stable"]["drift_detected"] is False
    assert drift["num_shifted"]["drift_detected"] is True
    assert "DRIFT DETECTION REPORT" in dd.format_drift_report(drift)
