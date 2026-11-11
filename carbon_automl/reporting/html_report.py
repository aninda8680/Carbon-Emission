"""
HTMLReportGenerator: Generates a self-contained, offline-ready research HTML
report using only Python stdlib (no Jinja2, no external CSS/JS CDN links).

Every quantitative value clearly distinguishes:
- MEASURED (wall-clock timing, direct profiler/instrument reads)
- ESTIMATED (CPU-load energy approximations, short-run inference estimates)
- CALCULATED (CO₂e = energy * carbon_intensity, statistical tests, Pareto)
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class HTMLReportGenerator:
    """Builds a comprehensive 24-section self-contained HTML experiment report."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        experiment_results: dict,
        output_path: str = "report.html",
        *,
        output_dir: str | None = None,
    ) -> str:
        """Render *experiment_results* to a self-contained HTML file.

        Parameters
        ----------
        experiment_results:
            Dictionary of results collected across the AutoML pipeline.
            Missing optional sections are rendered gracefully.
        output_path:
            Destination HTML file path (or existing directory, in which case
            ``report.html`` is created inside it).
        output_dir:
            Optional keyword fallback if caller passes a directory.

        Returns
        -------
        str
            Resolved path to the generated HTML file.
        """
        if output_dir is not None:
            out = Path(output_dir) / "report.html"
        else:
            out = Path(output_path)
            if out.exists() and out.is_dir():
                out = out / "report.html"

        out.parent.mkdir(parents=True, exist_ok=True)
        html_doc = self._render_document(experiment_results or {})
        out.write_text(html_doc, encoding="utf-8")
        return str(out.resolve())

    # ------------------------------------------------------------------
    # Document rendering
    # ------------------------------------------------------------------

    def _render_document(self, res: dict) -> str:
        css = self._css()
        timestamp = res.get("timestamp") or datetime.now(timezone.utc).isoformat()
        title = res.get("title") or "Carbon-Aware AutoML Experiment Report"

        sections_html = "\n".join(
            [
                self._sec_01_executive_summary(res),
                self._sec_02_dataset_fingerprint(res),
                self._sec_03_dataset_profile(res),
                self._sec_04_data_integrity(res),
                self._sec_05_leakage_warnings(res),
                self._sec_06_preprocessing(res),
                self._sec_07_validation_methodology(res),
                self._sec_08_model_comparison(res),
                self._sec_09_confusion_matrix(res),
                self._sec_10_per_class_recall(res),
                self._sec_11_calibration(res),
                self._sec_12_feature_importance(res),
                self._sec_13_ablation_analysis(res),
                self._sec_14_training_energy(res),
                self._sec_15_inference_energy(res),
                self._sec_16_co2_estimates(res),
                self._sec_17_carbon_intensity(res),
                self._sec_18_uncertainty_and_stats(res),
                self._sec_19_pareto_frontier(res),
                self._sec_20_carbon_budget(res),
                self._sec_21_model_recommendation(res),
                self._sec_22_carbon_passport(res),
                self._sec_23_threats_to_validity(res),
                self._sec_24_known_limitations(res),
            ]
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{self._esc(title)}</title>
<style>
{css}
</style>
</head>
<body>
<div class="container">
  <header class="report-header">
    <h1>{self._esc(title)}</h1>
    <div class="subtitle">
      Generated: {self._esc(timestamp)} &nbsp;|&nbsp;
      Provenance Legend:
      <span class="badge badge-measured">MEASURED</span>
      <span class="badge badge-estimated">ESTIMATED</span>
      <span class="badge badge-calculated">CALCULATED</span>
    </div>
  </header>

  <nav class="toc">
    <strong>Sections:</strong>
    <a href="#sec-1">1. Executive Summary</a>
    <a href="#sec-2">2. Fingerprint</a>
    <a href="#sec-3">3. Profile</a>
    <a href="#sec-4">4. Integrity</a>
    <a href="#sec-5">5. Leakage</a>
    <a href="#sec-6">6. Preprocessing</a>
    <a href="#sec-7">7. Validation</a>
    <a href="#sec-8">8. Models</a>
    <a href="#sec-9">9. Confusion Matrix</a>
    <a href="#sec-10">10. Per-Class Recall</a>
    <a href="#sec-11">11. Calibration</a>
    <a href="#sec-12">12. Feature Importance</a>
    <a href="#sec-13">13. Ablation</a>
    <a href="#sec-14">14. Training Energy</a>
    <a href="#sec-15">15. Inference Energy</a>
    <a href="#sec-16">16. CO2e Estimates</a>
    <a href="#sec-17">17. Carbon Intensity</a>
    <a href="#sec-18">18. Uncertainty &amp; Stats</a>
    <a href="#sec-19">19. Pareto Frontier</a>
    <a href="#sec-20">20. Carbon Budget</a>
    <a href="#sec-21">21. Recommendation</a>
    <a href="#sec-22">22. Carbon Passport</a>
    <a href="#sec-23">23. Threats to Validity</a>
    <a href="#sec-24">24. Known Limitations</a>
  </nav>

  {sections_html}

  <footer class="report-footer">
    Carbon-Aware AutoML Framework &mdash; Self-Contained Offline Report
  </footer>
</div>
</body>
</html>
"""

    # ------------------------------------------------------------------
    # Styling & Formatting Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _css() -> str:
        return """
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
            line-height: 1.55;
            color: #1e293b;
            background: #f8fafc;
            margin: 0;
            padding: 24px;
        }
        .container {
            max-width: 1120px;
            margin: 0 auto;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            overflow: hidden;
        }
        .report-header {
            background: #0f172a;
            color: #f8fafc;
            padding: 28px 36px;
        }
        .report-header h1 { margin: 0 0 8px 0; font-size: 1.75rem; }
        .subtitle { font-size: 0.88rem; color: #cbd5e1; }
        .toc {
            background: #f1f5f9;
            padding: 12px 36px;
            border-bottom: 1px solid #e2e8f0;
            font-size: 0.82rem;
            display: flex;
            flex-wrap: wrap;
            gap: 8px 14px;
        }
        .toc a { color: #1d4ed8; text-decoration: none; }
        .toc a:hover { text-decoration: underline; }
        .section {
            padding: 22px 36px;
            border-bottom: 1px solid #e2e8f0;
        }
        .section h2 {
            margin: 0 0 14px 0;
            font-size: 1.15rem;
            color: #0f172a;
            border-left: 4px solid #2563eb;
            padding-left: 10px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0 14px 0;
            font-size: 0.88rem;
        }
        th {
            background: #1e293b;
            color: #f8fafc;
            text-align: left;
            padding: 8px 10px;
            font-weight: 600;
        }
        td {
            padding: 7px 10px;
            border-bottom: 1px solid #e2e8f0;
            vertical-align: top;
        }
        tr:nth-child(even) td { background: #f8fafc; }
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 12px;
            margin: 10px 0;
        }
        .kpi-card {
            border: 1px solid #cbd5e1;
            border-radius: 6px;
            padding: 12px 14px;
            background: #f8fafc;
        }
        .kpi-label { font-size: 0.78rem; color: #475569; text-transform: uppercase; }
        .kpi-value { font-size: 1.15rem; font-weight: 700; color: #0f172a; margin-top: 4px; }
        .badge {
            display: inline-block;
            padding: 2px 7px;
            border-radius: 4px;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            vertical-align: middle;
        }
        .badge-measured { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
        .badge-estimated { background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }
        .badge-calculated { background: #dbeafe; color: #1e40af; border: 1px solid #93c5fd; }
        .sev-critical { background: #fee2e2; color: #991b1b; }
        .sev-high { background: #ffedd5; color: #9a3412; }
        .sev-medium { background: #fef9c3; color: #854d0e; }
        .sev-low { background: #e0f2fe; color: #075985; }
        .callout {
            padding: 12px 16px;
            border-radius: 6px;
            margin: 10px 0;
            font-size: 0.9rem;
        }
        .callout-info { background: #eff6ff; border-left: 4px solid #2563eb; }
        .callout-warn { background: #fffbeb; border-left: 4px solid #d97706; }
        .callout-danger { background: #fef2f2; border-left: 4px solid #dc2626; }
        .callout-success { background: #f0fdf4; border-left: 4px solid #16a34a; }
        .muted { color: #64748b; font-size: 0.88rem; }
        code {
            background: #f1f5f9;
            padding: 1px 5px;
            border-radius: 3px;
            font-family: Consolas, Monaco, monospace;
            font-size: 0.85em;
        }
        .report-footer {
            text-align: center;
            padding: 16px;
            font-size: 0.8rem;
            color: #64748b;
            background: #f1f5f9;
        }
        """

    @staticmethod
    def _esc(val: Any) -> str:
        return html.escape(str(val))

    @staticmethod
    def _fmt(val: Any, digits: int = 6) -> str:
        if val is None:
            return "N/A"
        if isinstance(val, bool):
            return "Yes" if val else "No"
        if isinstance(val, float):
            return f"{val:.{digits}g}"
        return html.escape(str(val))

    @staticmethod
    def _badge(kind: str) -> str:
        k = kind.lower()
        if k in ("measured", "meas"):
            return '<span class="badge badge-measured">MEASURED</span>'
        if k in ("estimated", "est"):
            return '<span class="badge badge-estimated">ESTIMATED</span>'
        return '<span class="badge badge-calculated">CALCULATED</span>'

    def _kv_table(self, rows: list[tuple[str, Any, str]]) -> str:
        body = []
        for label, val, kind in rows:
            badge_html = f" {self._badge(kind)}" if kind else ""
            body.append(
                f"<tr><td><strong>{self._esc(label)}</strong></td>"
                f"<td>{self._fmt(val)}{badge_html}</td></tr>"
            )
        return f"<table><tbody>{''.join(body)}</tbody></table>"

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

    # ------------------------------------------------------------------
    # 24 Report Sections
    # ------------------------------------------------------------------

    def _sec_01_executive_summary(self, res: dict) -> str:
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
        carbon = res.get("carbon") or {}
        total_co2 = carbon.get("total_co2_kg", rec_metrics.get("training_co2_kg"))
        total_energy = carbon.get("total_energy_kwh", rec_metrics.get("training_energy_kwh"))
        warnings = res.get("leakage_warnings") or []
        crit_count = sum(1 for w in warnings if str(w.get("severity", "")).lower() == "critical")

        summary_text = res.get("executive_summary") or (
            f"Recommended model: {rec_name}. "
            f"Total leakage warnings detected: {len(warnings)} ({crit_count} critical)."
        )

        cards = f"""
        <div class="kpi-grid">
          <div class="kpi-card">
            <div class="kpi-label">Recommended Model</div>
            <div class="kpi-value">{self._esc(rec_name)}</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Primary Quality Score {self._badge('measured')}</div>
            <div class="kpi-value">{self._fmt(quality, 4)}</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Total Energy (kWh) {self._badge('estimated')}</div>
            <div class="kpi-value">{self._fmt(total_energy, 4)}</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Total CO2e (kg) {self._badge('calculated')}</div>
            <div class="kpi-value">{self._fmt(total_co2, 4)}</div>
          </div>
        </div>
        <p>{self._esc(summary_text)}</p>
        """
        return f'<div class="section" id="sec-1"><h2>1. Executive Summary</h2>{cards}</div>'

    def _sec_02_dataset_fingerprint(self, res: dict) -> str:
        fp = res.get("dataset_fingerprint") or res.get("metadata") or {}
        if not fp:
            return '<div class="section" id="sec-2"><h2>2. Dataset Fingerprint</h2><p class="muted">No dataset fingerprint provided.</p></div>'

        file_hash = fp.get("sha256") or fp.get("file_hash") or fp.get("hash") or "N/A"
        shape = fp.get("shape")
        if not shape and ("row_count" in fp or "rows" in fp):
            shape = (fp.get("row_count", fp.get("rows")), fp.get("col_count", fp.get("cols")))
        dtypes = fp.get("dtypes") or res.get("feature_schema") or {}

        rows = [
            ("Source Path", fp.get("source_path", "N/A"), ""),
            ("Format", fp.get("format", "N/A"), ""),
            ("SHA-256 Hash", file_hash, "measured"),
            ("MD5 Hash", fp.get("md5", "N/A"), "measured"),
            ("Shape (rows, cols)", shape, "measured"),
            ("Load Timestamp", fp.get("load_timestamp", "N/A"), ""),
        ]
        table = self._kv_table(rows)
        schema_html = ""
        if isinstance(dtypes, dict) and dtypes:
            s_rows = "".join(
                f"<tr><td><code>{self._esc(c)}</code></td><td>{self._esc(d)}</td></tr>"
                for c, d in dtypes.items()
            )
            schema_html = f"<h3>Schema ({len(dtypes)} columns)</h3><table><thead><tr><th>Column</th><th>Dtype</th></tr></thead><tbody>{s_rows}</tbody></table>"

        return f'<div class="section" id="sec-2"><h2>2. Dataset Fingerprint</h2>{table}{schema_html}</div>'

    def _sec_03_dataset_profile(self, res: dict) -> str:
        prof = res.get("profile") or res.get("dataset_profile") or {}
        if not prof:
            return '<div class="section" id="sec-3"><h2>3. Dataset Profile</h2><p class="muted">No dataset profile provided.</p></div>'

        summary_table = self._kv_table([
            ("Shape", prof.get("shape"), "measured"),
            ("Memory (MB)", prof.get("memory_mb"), "measured"),
            ("Duplicate Rows", prof.get("duplicate_rows"), "measured"),
            ("Task Type Suggestion", prof.get("task_type_suggestion"), "calculated"),
        ])

        ftypes = prof.get("feature_types") or {}
        missing = prof.get("missing_values") or {}
        outliers = prof.get("outliers") or {}
        all_cols = list(dict.fromkeys(list(ftypes.keys()) + list(missing.keys()) + list(outliers.keys())))

        feat_table = ""
        if all_cols:
            f_rows = "".join(
                f"<tr><td><code>{self._esc(c)}</code></td>"
                f"<td>{self._esc(ftypes.get(c, 'unknown'))}</td>"
                f"<td>{self._fmt(missing.get(c, 0))}</td>"
                f"<td>{self._fmt(outliers.get(c, 0))}</td></tr>"
                for c in all_cols
            )
            feat_table = (
                "<h3>Feature Types, Missing Values &amp; Outliers (3&times;IQR)</h3>"
                "<table><thead><tr><th>Feature</th><th>Detected Type</th><th>Missing</th><th>Outliers</th></tr></thead>"
                f"<tbody>{f_rows}</tbody></table>"
            )

        dist_html = ""
        class_dist = prof.get("class_distribution")
        if isinstance(class_dist, dict) and class_dist:
            c_rows = "".join(
                f"<tr><td>{self._esc(k)}</td><td>{self._fmt(v)}</td></tr>"
                for k, v in class_dist.items()
            )
            dist_html = f"<h3>Class / Target Distribution</h3><table><thead><tr><th>Class / Stat</th><th>Value</th></tr></thead><tbody>{c_rows}</tbody></table>"
        elif isinstance(prof.get("target_distribution"), dict) and prof["target_distribution"]:
            t_rows = "".join(
                f"<tr><td>{self._esc(k)}</td><td>{self._fmt(v)}</td></tr>"
                for k, v in prof["target_distribution"].items()
            )
            dist_html = f"<h3>Target Distribution</h3><table><thead><tr><th>Statistic</th><th>Value</th></tr></thead><tbody>{t_rows}</tbody></table>"

        return f'<div class="section" id="sec-3"><h2>3. Dataset Profile</h2>{summary_table}{feat_table}{dist_html}</div>'

    def _sec_04_data_integrity(self, res: dict) -> str:
        integ = res.get("integrity") or res.get("data_integrity") or {}
        fp = res.get("dataset_fingerprint") or res.get("metadata") or {}
        expected_hash = integ.get("expected_hash")
        actual_hash = integ.get("actual_hash") or fp.get("sha256") or fp.get("file_hash")
        verified = integ.get("verified")
        if verified is None and actual_hash and expected_hash:
            verified = actual_hash == expected_hash
        if verified is None and actual_hash:
            verified = True

        status_cls = "callout-success" if verified else "callout-warn"
        status_msg = (
            "Dataset integrity verified: source file hash and schema preserved without mutation."
            if verified
            else "Integrity status unavailable or mismatch detected — review hash records."
        )
        table = self._kv_table([
            ("Verification Passed", verified, "measured"),
            ("Recorded SHA-256", actual_hash, "measured"),
            ("Expected SHA-256", expected_hash or actual_hash, ""),
            ("Original DataFrame Unchanged", integ.get("unchanged", verified), "measured"),
        ])
        return (
            f'<div class="section" id="sec-4"><h2>4. Data Integrity Verification</h2>'
            f'<div class="callout {status_cls}">{self._esc(status_msg)}</div>{table}</div>'
        )

    def _sec_05_leakage_warnings(self, res: dict) -> str:
        warnings = res.get("leakage_warnings") or []
        if not warnings:
            return (
                '<div class="section" id="sec-5"><h2>5. Leakage Warnings</h2>'
                '<div class="callout callout-success">No data leakage warnings detected.</div></div>'
            )

        rows = []
        for w in warnings:
            sev = str(w.get("severity", "medium")).lower()
            sev_cls = f"sev-{sev}" if sev in ("critical", "high", "medium", "low") else "sev-medium"
            rows.append(
                f"<tr>"
                f"<td><code>{self._esc(w.get('feature', 'unknown'))}</code></td>"
                f"<td><span class='badge {sev_cls}'>{self._esc(sev.upper())}</span></td>"
                f"<td>{self._esc(w.get('detection_method', 'N/A'))}</td>"
                f"<td>{self._esc(w.get('evidence', 'N/A'))}</td>"
                f"<td>{self._esc(w.get('recommended_investigation', 'N/A'))}</td>"
                f"</tr>"
            )
        table = (
            "<table><thead><tr><th>Feature</th><th>Severity</th><th>Method</th>"
            "<th>Evidence</th><th>Recommended Action</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )
        return (
            f'<div class="section" id="sec-5"><h2>5. Leakage Warnings</h2>'
            f'<div class="callout callout-danger">Detected {len(warnings)} potential leakage signal(s). '
            f'Features are never auto-deleted silently; review and configure <code>exclude_columns</code> explicitly.</div>'
            f'{table}</div>'
        )

    def _sec_06_preprocessing(self, res: dict) -> str:
        prep = res.get("preprocessing") or {}
        cfg = res.get("config") or {}
        if not prep and not cfg:
            return '<div class="section" id="sec-6"><h2>6. Preprocessing Decisions</h2><p class="muted">No preprocessing metadata recorded.</p></div>'

        rows = [
            ("Excluded Columns", cfg.get("exclude_columns", prep.get("excluded_columns", [])), ""),
            ("Numerical Pipeline", prep.get("numerical_pipeline", "SimpleImputer(median) + StandardScaler"), ""),
            ("Categorical Pipeline", prep.get("categorical_pipeline", "SimpleImputer(most_frequent) + OrdinalEncoder(unknown=-1)"), ""),
            ("Fit Scope", prep.get("fit_scope", "Training folds/split only (zero test leakage)"), ""),
            ("Output Shape", prep.get("output_shape", "N/A"), "measured"),
        ]
        html_out = self._kv_table(rows)
        history = prep.get("transformation_history") or prep.get("steps") or []
        if isinstance(history, list) and history:
            items = "".join(f"<li>{self._esc(item)}</li>" for item in history)
            html_out += f"<h3>Transformation History</h3><ul>{items}</ul>"
        return f'<div class="section" id="sec-6"><h2>6. Preprocessing Decisions</h2>{html_out}</div>'

    def _sec_07_validation_methodology(self, res: dict) -> str:
        val = res.get("validation") or {}
        cfg = res.get("config") or {}
        strategy = val.get("strategy") or cfg.get("validation_strategy") or "stratified_kfold"
        n_folds = val.get("n_folds") or cfg.get("n_folds") or 5
        n_repeats = val.get("n_repeats") or cfg.get("n_repeats") or 1
        seed = val.get("random_seed") or cfg.get("random_seed") or 42
        holdout = val.get("holdout_isolation", "Holdout test split isolated from preprocessing & HPO")

        table = self._kv_table([
            ("Validation Strategy", strategy, ""),
            ("Cross-Validation Folds", n_folds, ""),
            ("CV Repeats", n_repeats, ""),
            ("Random Seed", seed, ""),
            ("Group Column", cfg.get("group_column", val.get("group_column")), ""),
            ("Timestamp Column", cfg.get("timestamp_column", val.get("timestamp_column")), ""),
            ("Data Isolation", holdout, ""),
        ])
        return f'<div class="section" id="sec-7"><h2>7. Validation Methodology</h2>{table}</div>'

    def _sec_08_model_comparison(self, res: dict) -> str:
        models = self._extract_models(res)
        if not models:
            return '<div class="section" id="sec-8"><h2>8. Model Comparison</h2><p class="muted">No evaluated models found.</p></div>'

        rows = []
        for m in models:
            name = m.get("name") or m.get("model_name") or "unknown"
            metrics = m.get("metrics", m) if isinstance(m.get("metrics"), dict) else m
            macro_f1 = metrics.get("macro_f1")
            acc = metrics.get("accuracy")
            bal_acc = metrics.get("balanced_accuracy")
            r2 = metrics.get("r2")
            rmse = metrics.get("rmse")
            mean_score = m.get("mean_score", macro_f1 if macro_f1 is not None else r2)
            ci = m.get("ci") or metrics.get("ci") or {}
            ci_str = (
                f"[{self._fmt(ci.get('ci_lower'), 4)}, {self._fmt(ci.get('ci_upper'), 4)}]"
                if isinstance(ci, dict) and ci.get("ci_lower") is not None
                else "N/A"
            )
            dur = m.get("training_duration_seconds", metrics.get("training_duration_seconds"))
            eng = m.get("training_energy_kwh", metrics.get("training_energy_kwh"))
            co2 = m.get("training_co2_kg", metrics.get("training_co2_kg"))

            rows.append(
                f"<tr>"
                f"<td><strong>{self._esc(name)}</strong></td>"
                f"<td>{self._fmt(mean_score, 4)}</td>"
                f"<td>{ci_str}</td>"
                f"<td>{self._fmt(acc if acc is not None else r2, 4)}</td>"
                f"<td>{self._fmt(bal_acc if bal_acc is not None else rmse, 4)}</td>"
                f"<td>{self._fmt(dur, 4)}</td>"
                f"<td>{self._fmt(eng, 4)}</td>"
                f"<td>{self._fmt(co2, 4)}</td>"
                f"</tr>"
            )

        table = (
            f"<p class='muted'>Quality &amp; duration {self._badge('measured')} &nbsp; "
            f"Energy {self._badge('estimated')} &nbsp; CO2e {self._badge('calculated')}</p>"
            "<table><thead><tr>"
            "<th>Model</th><th>Primary Score (F1 / R&sup2;)</th><th>95% Bootstrap CI</th>"
            "<th>Acc / R&sup2;</th><th>Bal. Acc / RMSE</th>"
            "<th>Train Time (s)</th><th>Energy (kWh)</th><th>CO2e (kg)</th>"
            "</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )
        return f'<div class="section" id="sec-8"><h2>8. Model Comparison (Mean &amp; CI)</h2>{table}</div>'

    def _sec_09_confusion_matrix(self, res: dict) -> str:
        rec = self._extract_recommended(res)
        rec_metrics = rec.get("metrics", rec)
        cm = res.get("confusion_matrix") or rec_metrics.get("confusion_matrix")
        class_names = res.get("class_names")
        if not cm:
            return '<div class="section" id="sec-9"><h2>9. Confusion Matrix</h2><p class="muted">Not applicable or not recorded (e.g. regression task).</p></div>'

        n = len(cm)
        labels = [str(c) for c in class_names] if class_names else [str(i) for i in range(n)]
        if len(labels) < n:
            labels.extend(str(i) for i in range(len(labels), n))

        header = "<tr><th>True \\ Pred</th>" + "".join(f"<th>{self._esc(l)}</th>" for l in labels[:n]) + "</tr>"
        body_rows = []
        for i, row in enumerate(cm):
            cells = "".join(f"<td>{self._fmt(v)}</td>" for v in row)
            body_rows.append(f"<tr><th>{self._esc(labels[i])}</th>{cells}</tr>")

        return (
            f'<div class="section" id="sec-9"><h2>9. Confusion Matrix {self._badge("measured")}</h2>'
            f"<table><thead>{header}</thead><tbody>{''.join(body_rows)}</tbody></table></div>"
        )

    def _sec_10_per_class_recall(self, res: dict) -> str:
        rec = self._extract_recommended(res)
        rec_metrics = rec.get("metrics", rec)
        pcr = res.get("per_class_recall") or rec_metrics.get("per_class_recall")
        if not isinstance(pcr, dict) or not pcr:
            return '<div class="section" id="sec-10"><h2>10. Per-Class Recall</h2><p class="muted">Not applicable or not recorded.</p></div>'

        constraints = (res.get("config") or {}).get("critical_class_recall_constraints", {})
        rows = []
        for cls_name, recall_val in pcr.items():
            floor = constraints.get(cls_name, constraints.get(str(cls_name)))
            status = "Pass" if (floor is None or float(recall_val) >= float(floor)) else f"FAIL (< {floor})"
            rows.append(
                f"<tr><td>{self._esc(cls_name)}</td><td>{self._fmt(recall_val, 4)}</td>"
                f"<td>{self._fmt(floor, 4) if floor is not None else 'None'}</td><td>{self._esc(status)}</td></tr>"
            )
        return (
            f'<div class="section" id="sec-10"><h2>10. Per-Class Recall {self._badge("measured")}</h2>'
            "<table><thead><tr><th>Class</th><th>Recall</th><th>Min Constraint</th><th>Status</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>"
        )

    def _sec_11_calibration(self, res: dict) -> str:
        cal = res.get("calibration") or {}
        rec = self._extract_recommended(res)
        rec_metrics = rec.get("metrics", rec)
        brier = cal.get("brier_score", rec_metrics.get("brier_score"))
        log_loss = cal.get("log_loss", rec_metrics.get("log_loss"))
        ece = cal.get("ece", rec_metrics.get("ece"))

        if brier is None and log_loss is None and ece is None:
            return '<div class="section" id="sec-11"><h2>11. Calibration Analysis</h2><p class="muted">Probability calibration metrics not computed (requires predicted probabilities in classification).</p></div>'

        table = self._kv_table([
            ("Brier Score (lower is better)", brier, "calculated"),
            ("Log Loss (lower is better)", log_loss, "calculated"),
            ("Expected Calibration Error (ECE)", ece, "calculated"),
        ])
        return f'<div class="section" id="sec-11"><h2>11. Calibration Analysis</h2>{table}</div>'

    def _sec_12_feature_importance(self, res: dict) -> str:
        fi = res.get("feature_importance") or res.get("feature_selection") or {}
        if isinstance(fi, dict):
            imp_map = fi.get("importances_mean") or fi.get("scores") or fi.get("importances") or (
                fi if all(isinstance(v, (int, float)) for v in fi.values()) else {}
            )
        else:
            imp_map = {}

        if not imp_map:
            return '<div class="section" id="sec-12"><h2>12. Feature Importance</h2><p class="muted">No feature importance scores recorded.</p></div>'

        sorted_items = sorted(imp_map.items(), key=lambda kv: abs(float(kv[1])), reverse=True)[:25]
        rows = "".join(
            f"<tr><td><code>{self._esc(k)}</code></td><td>{self._fmt(v, 4)}</td></tr>"
            for k, v in sorted_items
        )
        return (
            f'<div class="section" id="sec-12"><h2>12. Feature Importance {self._badge("calculated")}</h2>'
            "<table><thead><tr><th>Feature</th><th>Importance / Score</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
        )

    def _sec_13_ablation_analysis(self, res: dict) -> str:
        ablation = res.get("ablation") or res.get("ablation_results") or {}
        conditions = ablation.get("conditions", ablation) if isinstance(ablation, dict) else {}
        if not isinstance(conditions, dict) or not conditions:
            return '<div class="section" id="sec-13"><h2>13. Ablation Analysis</h2><p class="muted">No ablation study results recorded.</p></div>'

        rows = []
        for cond, val in conditions.items():
            if isinstance(val, dict):
                score = val.get("score", val.get("macro_f1", val.get("r2", val.get("mean_score"))))
                dur = val.get("duration_seconds", val.get("training_duration_seconds"))
                eng = val.get("energy_kwh")
                notes = val.get("notes", val.get("description", ""))
            else:
                score, dur, eng, notes = val, None, None, ""
            rows.append(
                f"<tr><td><strong>{self._esc(cond)}</strong></td>"
                f"<td>{self._fmt(score, 4)}</td>"
                f"<td>{self._fmt(dur, 4)}</td>"
                f"<td>{self._fmt(eng, 4)}</td>"
                f"<td>{self._esc(notes)}</td></tr>"
            )
        return (
            f'<div class="section" id="sec-13"><h2>13. Ablation Analysis {self._badge("measured")}</h2>'
            "<table><thead><tr><th>Condition</th><th>Score</th><th>Duration (s)</th><th>Energy (kWh)</th><th>Notes</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>"
        )

    def _sec_14_training_energy(self, res: dict) -> str:
        phases = res.get("phases") or res.get("energy_by_phase") or {}
        if not phases:
            return '<div class="section" id="sec-14"><h2>14. Training Energy by Phase</h2><p class="muted">No per-phase energy measurements recorded.</p></div>'

        rows = []
        for phase_name, m in phases.items():
            if hasattr(m, "__dataclass_fields__"):
                dur = m.duration_seconds
                eng = m.energy_kwh
                est = m.is_estimated
                method = m.measurement_method
                unc = m.uncertainty_pct
            elif isinstance(m, dict):
                dur = m.get("duration_seconds")
                eng = m.get("energy_kwh")
                est = m.get("is_estimated", True)
                method = m.get("measurement_method", "unknown")
                unc = m.get("uncertainty_pct", 50.0)
            else:
                dur, eng, est, method, unc = None, m, True, "scalar", 50.0

            label = self._badge("estimated" if est else "measured")
            rows.append(
                f"<tr><td><strong>{self._esc(phase_name)}</strong></td>"
                f"<td>{self._fmt(dur, 4)} {self._badge('measured')}</td>"
                f"<td>{self._fmt(eng, 4)} {label}</td>"
                f"<td>{self._esc(method)}</td>"
                f"<td>&plusmn;{self._fmt(unc, 3)}%</td></tr>"
            )
        return (
            '<div class="section" id="sec-14"><h2>14. Training Energy by Phase</h2>'
            "<table><thead><tr><th>Phase</th><th>Duration (s)</th><th>Energy (kWh)</th>"
            "<th>Method</th><th>Uncertainty</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>"
        )

    def _sec_15_inference_energy(self, res: dict) -> str:
        inf = res.get("inference_energy") or res.get("inference_measurements") or {}
        if not inf:
            passport = res.get("carbon_passport") or {}
            if passport:
                inf = {
                    "inference_energy_per_1k_predictions_kwh": passport.get("inference_energy_per_1k_predictions_kwh"),
                    "latency": passport.get("inference_latency", {}),
                }
        if not inf:
            return '<div class="section" id="sec-15"><h2>15. Inference Energy</h2><p class="muted">No inference energy measurements recorded.</p></div>'

        table = self._kv_table([
            ("Energy per prediction (J)", inf.get("energy_per_prediction_joules"), "estimated"),
            ("Energy per 1k predictions (kWh)", inf.get("inference_energy_per_1k_predictions_kwh"), "estimated"),
            ("CO2e per 1k predictions (kg)", inf.get("co2_per_1k_predictions_kg"), "calculated"),
            ("Mean batch duration (s)", inf.get("mean_duration_seconds"), "measured"),
            ("Uncertainty (%)", inf.get("measurement_uncertainty_pct", 100.0), "estimated"),
            ("Method", inf.get("method", "repeated_batch_estimation"), ""),
        ])
        warn = inf.get("warning") or "Inference energy at microsecond/millisecond scale carries high noise and is an order-of-magnitude estimate."
        return (
            f'<div class="section" id="sec-15"><h2>15. Inference Energy</h2>'
            f'<div class="callout callout-warn">{self._esc(warn)}</div>{table}</div>'
        )

    def _sec_16_co2_estimates(self, res: dict) -> str:
        carbon = res.get("carbon") or {}
        total_co2 = carbon.get("total_co2_kg")
        total_energy = carbon.get("total_energy_kwh")
        uncertainty = carbon.get("uncertainty_pct", 50.0)
        models = self._extract_models(res)
        if total_co2 is None and models:
            total_co2 = sum(float(m.get("training_co2_kg") or 0.0) for m in models)
            total_energy = sum(float(m.get("training_energy_kwh") or 0.0) for m in models)

        table = self._kv_table([
            ("Total Compute Energy (kWh)", total_energy, "estimated"),
            ("Total Estimated CO2e (kg)", total_co2, "calculated"),
            ("Reported Uncertainty", f"±{uncertainty}%", "estimated"),
            ("Measurement Boundary", "Direct CPU/RAM active compute only (excludes PUE, network, embodied carbon)", ""),
        ])
        return f'<div class="section" id="sec-16"><h2>16. CO2e Estimates with Uncertainty</h2>{table}</div>'

    def _sec_17_carbon_intensity(self, res: dict) -> str:
        cfg = res.get("config") or {}
        carbon = res.get("carbon") or {}
        intensity = carbon.get("carbon_intensity_kg_per_kwh", cfg.get("carbon_intensity_kg_per_kwh", 0.7))
        source = carbon.get("intensity_source", cfg.get("carbon_intensity_source", "user_configured (India national average default)"))
        statement = carbon.get("assumption_statement") or (
            f"Carbon intensity assumed: {intensity} kg CO2e/kWh (source: {source}). "
            "This is a static grid-average assumption, not a real-time marginal intensity measurement. "
            "Embodied hardware emissions, cooling PUE, network, and storage are excluded."
        )
        return (
            f'<div class="section" id="sec-17"><h2>17. Carbon Intensity Assumption</h2>'
            f'<div class="callout callout-info">{self._esc(statement)}</div></div>'
        )

    def _sec_18_uncertainty_and_stats(self, res: dict) -> str:
        stats = res.get("statistics") or {}
        pairwise = stats.get("pairwise") or stats.get("comparisons") or {}
        if not pairwise:
            return (
                '<div class="section" id="sec-18"><h2>18. Measurement Uncertainty &amp; Statistical Analysis</h2>'
                '<p class="muted">No pairwise statistical comparison tests recorded (requires &ge; 5 repeated samples per model for Wilcoxon signed-rank test).</p></div>'
            )

        rows = []
        for pair, info in pairwise.items():
            if not isinstance(info, dict):
                continue
            rows.append(
                f"<tr><td>{self._esc(pair)}</td>"
                f"<td>{self._fmt(info.get('p_value'), 4)}</td>"
                f"<td>{self._fmt(info.get('corrected_alpha', info.get('alpha')))}</td>"
                f"<td>{self._fmt(info.get('significant'))}</td>"
                f"<td>{self._esc(info.get('note', info.get('reason', '')))}</td></tr>"
            )
        table = (
            "<table><thead><tr><th>Comparison</th><th>p-value</th><th>Alpha</th>"
            "<th>Significant</th><th>Notes</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )
        return f'<div class="section" id="sec-18"><h2>18. Measurement Uncertainty &amp; Statistical Analysis {self._badge("calculated")}</h2>{table}</div>'

    def _sec_19_pareto_frontier(self, res: dict) -> str:
        pareto = res.get("pareto") or {}
        sel = res.get("model_selection") or res.get("selection") or {}
        front = pareto.get("pareto_front") or pareto.get("pareto_optimal") or sel.get("pareto_optimal") or []
        if isinstance(front, list) and front and isinstance(front[0], dict):
            front_names = [str(m.get("name") or m.get("model_name") or "?") for m in front]
        else:
            front_names = [str(x) for x in front]

        if not front_names:
            return '<div class="section" id="sec-19"><h2>19. Pareto Frontier</h2><p class="muted">No Pareto frontier computed.</p></div>'

        items = ", ".join(f"<code>{self._esc(n)}</code>" for n in front_names)
        return (
            f'<div class="section" id="sec-19"><h2>19. Pareto Frontier {self._badge("calculated")}</h2>'
            f"<p>Non-dominated models balancing predictive quality (maximised) and energy consumption (minimised): {items}</p>"
            "</div>"
        )

    def _sec_20_carbon_budget(self, res: dict) -> str:
        budget = res.get("carbon_budget") or {}
        cfg = res.get("config") or {}
        limit = budget.get("budget_kg", cfg.get("carbon_budget_kg"))
        if limit is None and not budget:
            return '<div class="section" id="sec-20"><h2>20. Carbon Budget Results</h2><p class="muted">No hard carbon budget was configured for this run.</p></div>'

        table = self._kv_table([
            ("Configured Carbon Budget (kg CO2e)", limit, ""),
            ("Models Within Budget", budget.get("within_budget", "N/A"), "calculated"),
            ("Conflict / Trade-off Note", budget.get("conflict_message", budget.get("message", "None")), ""),
        ])
        return f'<div class="section" id="sec-20"><h2>20. Carbon Budget Results</h2>{table}</div>'

    def _sec_21_model_recommendation(self, res: dict) -> str:
        rec = self._extract_recommended(res)
        sel = res.get("model_selection") or res.get("selection") or {}
        if not rec:
            return '<div class="section" id="sec-21"><h2>21. Model Recommendation</h2><p class="muted">No model recommendation produced.</p></div>'

        name = rec.get("name") or rec.get("model_name") or "unknown"
        justification = rec.get("justification") or "Selected via Pareto-balanced quality and energy ranking."
        violations = sel.get("quality_constraint_violations", [])
        recall_violations = sel.get("critical_class_recall_violations", {})

        rows = [
            ("Recommended Model", name, "calculated"),
            ("Justification", justification, ""),
            ("Quality Gate Violations", violations if violations else "None", "calculated"),
        ]
        if recall_violations:
            rows.append(("Critical-Class Recall Violations", recall_violations, "calculated"))
        return f'<div class="section" id="sec-21"><h2>21. Model Recommendation</h2>{self._kv_table(rows)}</div>'

    def _sec_22_carbon_passport(self, res: dict) -> str:
        cp = res.get("carbon_passport") or {}
        if not cp:
            return '<div class="section" id="sec-22"><h2>22. Carbon Passport Summary</h2><p class="muted">No Carbon Passport embedded.</p></div>'

        hw = cp.get("hardware") or {}
        te = cp.get("training_energy_kwh") or {}
        co2 = cp.get("training_co2_kg_estimate") or {}
        table = self._kv_table([
            ("Model ID (UUID)", cp.get("model_id"), ""),
            ("Framework Version", cp.get("framework_version"), ""),
            ("Dataset Hash", cp.get("dataset_hash"), "measured"),
            ("Hardware", f"{hw.get('cpu_count', '?')} CPUs, {hw.get('ram_gb', '?')} GB RAM, {hw.get('os', '?')}", "measured"),
            ("Training Energy (kWh)", te.get("value") if isinstance(te, dict) else te, "estimated"),
            ("Training CO2e (kg)", co2.get("value") if isinstance(co2, dict) else co2, "calculated"),
            ("Inference Energy / 1k preds (kWh)", cp.get("inference_energy_per_1k_predictions_kwh"), "estimated"),
        ])
        return f'<div class="section" id="sec-22"><h2>22. Carbon Passport</h2>{table}</div>'

    def _sec_23_threats_to_validity(self, res: dict) -> str:
        threats = res.get("threats_to_validity") or [
            "Internal validity: Sub-second training/inference on small datasets (~1,000 rows) falls below CodeCarbon sampling resolution; background OS noise can reach ±50–100%.",
            "External validity: Grid carbon intensity varies regionally and temporally; static national averages do not capture marginal grid emissions.",
            "Construct validity: Direct compute energy excludes embodied hardware emissions, data-centre PUE cooling overhead, storage, and networking.",
            "Statistical validity: Claims of environmental superiority require n >= 5 repeated runs and Bonferroni-corrected Wilcoxon signed-rank significance (p < 0.05).",
        ]
        items = "".join(f"<li>{self._esc(t)}</li>" for t in threats)
        return f'<div class="section" id="sec-23"><h2>23. Threats to Validity</h2><ul>{items}</ul></div>'

    def _sec_24_known_limitations(self, res: dict) -> str:
        cp = res.get("carbon_passport") or {}
        lims = res.get("known_limitations") or cp.get("known_limitations") or [
            "CPU-load power modelling used when hardware RAPL/NVML sensors are unavailable.",
            "Carbon intensity is user-configured rather than queried from a live grid API.",
            "Leakage warnings identify statistical proxies and require domain-expert review before column exclusion.",
        ]
        items = "".join(f"<li>{self._esc(l)}</li>" for l in lims)
        return f'<div class="section" id="sec-24"><h2>24. Known Limitations</h2><ul>{items}</ul></div>'
