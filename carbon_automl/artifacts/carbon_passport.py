"""
CarbonPassport: Generates, exports, and renders carbon transparency documents
for trained ML models in JSON and HTML formats.

All values in the passport are plain Python types (str, int, float, list,
dict) so the resulting JSON is parseable without custom deserializers.
"""

from __future__ import annotations

import html as html_lib
import json
import platform
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Optional dependency — gracefully degrade if sklearn is not importable
try:
    import sklearn
    _SKLEARN_VERSION = sklearn.__version__
except ImportError:
    _SKLEARN_VERSION = "unknown"


_FRAMEWORK_VERSION = "carbon_automl 1.0.0"
_DEFAULT_CARBON_INTENSITY = 0.7  # kg CO₂e / kWh — India default
_CARBON_INTENSITY_SOURCE = (
    "India national grid average (CEA 2021–22 Annual Report). "
    "Actual intensity varies by region, time of day, and renewable mix."
)
_CARBON_INTENSITY_CAVEAT = (
    "This is a country-level annual average. Real-time regional values may "
    "differ significantly. For procurement decisions, use regional/real-time "
    "data (e.g., Electricity Maps API)."
)


class CarbonPassport:
    """Generates structured carbon transparency documents for ML models.

    Example::

        passport = CarbonPassport()
        doc = passport.generate(
            model_name="random_forest",
            dataset_fingerprint={"sha256": "...", "rows": 1000, "cols": 15},
            ...
        )
        passport.export_json(doc, "reports/rf_passport.json")
        passport.export_html(doc, "reports/rf_passport.html")
    """

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        model_name: str,
        dataset_fingerprint: dict,
        feature_schema: dict,
        training_config: dict,
        hardware_info: dict,
        training_measurements: dict,
        inference_measurements: dict,
        predictive_metrics: dict,
        validation_methodology: str,
        known_limitations: list[str],
    ) -> dict:
        """Generate a carbon passport for a trained model.

        Parameters
        ----------
        model_name:
            Human-readable model name.
        dataset_fingerprint:
            Dict with at least ``{"sha256": str, "rows": int, "cols": int}``.
        feature_schema:
            Mapping of column name → dtype string.
        training_config:
            Hyperparameters and model type (e.g. ``{"n_estimators": 100, ...}``).
        hardware_info:
            CPU, RAM, OS info. Missing fields are auto-detected.
        training_measurements:
            Dict with keys: ``training_duration_seconds``,
            ``training_energy_kwh``, ``training_co2_kg``,
            ``is_estimated`` (bool), ``carbon_intensity_kg_per_kwh`` (float).
        inference_measurements:
            Dict with keys: ``inference_energy_per_1k_predictions_kwh``,
            ``inference_latency_ms_p50``, ``inference_latency_ms_p99``.
        predictive_metrics:
            Dict of metric name → value or ``{value, ci_lower, ci_upper}``.
        validation_methodology:
            Human-readable string (e.g. ``"5-fold stratified cross-validation"``).
        known_limitations:
            List of plain-text limitation strings.

        Returns
        -------
        dict
            Fully populated passport — all values are JSON-serialisable
            without custom deserializers.
        """
        carbon_intensity_used = float(
            training_measurements.get(
                "carbon_intensity_kg_per_kwh", _DEFAULT_CARBON_INTENSITY
            )
        )

        # Flatten metrics: allow both scalar values and CI dicts
        normalised_metrics: dict[str, Any] = {}
        confidence_intervals: dict[str, dict] = {}
        for key, val in predictive_metrics.items():
            if isinstance(val, dict) and "value" in val:
                normalised_metrics[key] = val["value"]
                confidence_intervals[key] = {
                    "ci_lower": val.get("ci_lower"),
                    "ci_upper": val.get("ci_upper"),
                }
            else:
                normalised_metrics[key] = val

        # Merge auto-detected hardware info with caller-supplied values
        hw = self._hardware_info(hardware_info)

        # Resolve dataset hash and dimensions flexibly across module conventions
        ds_hash = (
            dataset_fingerprint.get("sha256")
            or dataset_fingerprint.get("file_hash")
            or dataset_fingerprint.get("hash")
            or ""
        )
        # Resolve inference energy per 1k predictions (support CarbonCalculator output)
        inf_energy_1k = inference_measurements.get("inference_energy_per_1k_predictions_kwh")
        if inf_energy_1k is None:
            e_per_pred_j = inference_measurements.get("energy_per_prediction_joules")
            if e_per_pred_j is not None:
                inf_energy_1k = (float(e_per_pred_j) * 1000.0) / 3_600_000.0
            else:
                inf_energy_1k = 0.0

        passport: dict[str, Any] = {
            # Identity
            "model_id": str(uuid.uuid4()),
            "model_name": model_name,
            "framework_version": _FRAMEWORK_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),

            # Data
            "dataset_hash": ds_hash,
            "dataset_fingerprint": {
                k: v for k, v in dataset_fingerprint.items()
            },
            "feature_schema": {col: str(dtype) for col, dtype in feature_schema.items()},

            # Training configuration
            "training_config": {
                "model_type": training_config.get("model_type", model_name),
                "hyperparameters": {
                    k: v for k, v in training_config.items() if k != "model_type"
                },
            },

            # Hardware
            "hardware": hw,

            # Training energy & carbon
            "training_duration_seconds": float(
                training_measurements.get("training_duration_seconds", 0.0)
            ),
            "training_energy_kwh": {
                "value": float(training_measurements.get("training_energy_kwh", 0.0)),
                "is_estimated": bool(training_measurements.get("is_estimated", True)),
                "measurement_method": training_measurements.get(
                    "measurement_method", "process_time_estimation"
                ),
            },
            "training_co2_kg_estimate": {
                "value": float(training_measurements.get("training_co2_kg", 0.0)),
                "carbon_intensity_used": carbon_intensity_used,
            },
            "carbon_intensity_assumption": {
                "value_kg_per_kwh": carbon_intensity_used,
                "source": training_measurements.get(
                    "carbon_intensity_source", _CARBON_INTENSITY_SOURCE
                ),
                "caveat": _CARBON_INTENSITY_CAVEAT,
            },

            # Inference energy
            "inference_energy_per_1k_predictions_kwh": float(inf_energy_1k),
            "inference_latency": {
                "p50_ms": inference_measurements.get("inference_latency_ms_p50"),
                "p99_ms": inference_measurements.get("inference_latency_ms_p99"),
            },

            # Quality
            "predictive_metrics": normalised_metrics,
            "confidence_intervals": confidence_intervals,
            "validation_methodology": validation_methodology,

            # Transparency
            "known_limitations": list(known_limitations),
        }
        return passport

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_json(self, passport: dict, output_path: str) -> str:
        """Serialise *passport* to a JSON file.

        Parameters
        ----------
        passport:
            Dict returned by :meth:`generate`.
        output_path:
            Destination path (parent directories are created automatically).

        Returns
        -------
        str
            Absolute path to the written file.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(passport, indent=2, default=str), encoding="utf-8")
        return str(out.resolve())

    def export_html(self, passport: dict, output_path: str) -> str:
        """Render *passport* as a self-contained HTML file.

        No external CSS or JS dependencies — the full style is inlined.

        Parameters
        ----------
        passport:
            Dict returned by :meth:`generate`.
        output_path:
            Destination path (parent directories are created automatically).

        Returns
        -------
        str
            Absolute path to the written file.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        html = self._render_html(passport)
        out.write_text(html, encoding="utf-8")
        return str(out.resolve())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hardware_info(caller_info: dict) -> dict:
        """Merge caller-supplied hardware info with auto-detected values."""
        try:
            import os
            cpu_count = os.cpu_count() or 1
        except Exception:
            cpu_count = 1

        try:
            import psutil
            ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
        except ImportError:
            ram_gb = caller_info.get("ram_gb", "unknown")

        return {
            "cpu_count": caller_info.get("cpu_count", cpu_count),
            "ram_gb": caller_info.get("ram_gb", ram_gb),
            "os": caller_info.get("os", platform.platform()),
            "python_version": caller_info.get("python_version", platform.python_version()),
            "sklearn_version": caller_info.get("sklearn_version", _SKLEARN_VERSION),
            **{k: v for k, v in caller_info.items() if k not in {
                "cpu_count", "ram_gb", "os", "python_version", "sklearn_version"
            }},
        }

    @staticmethod
    def _fmt(val: Any) -> str:
        """Format a passport value safely for HTML display."""
        if val is None:
            return "<em>N/A</em>"
        if isinstance(val, float):
            return f"{val:.6g}"
        return html_lib.escape(str(val))

    def _render_html(self, passport: dict) -> str:
        """Render a passport dict into self-contained HTML."""
        css = """
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px;
                background: #f5f7fa; color: #2c3e50; }
        .passport { max-width: 900px; margin: 0 auto; background: #fff;
                    border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,.1);
                    overflow: hidden; }
        .header { background: linear-gradient(135deg, #1a5276, #2980b9);
                  color: #fff; padding: 30px 40px; }
        .header h1 { margin: 0 0 8px 0; font-size: 1.8em; }
        .header .meta { font-size: 0.85em; opacity: 0.85; }
        .section { padding: 20px 40px; border-bottom: 1px solid #ecf0f1; }
        .section h2 { color: #1a5276; font-size: 1.1em; margin-bottom: 12px;
                       border-left: 4px solid #2980b9; padding-left: 10px; }
        table { border-collapse: collapse; width: 100%; font-size: 0.9em; }
        th { background: #2980b9; color: #fff; padding: 8px 12px; text-align: left; }
        td { padding: 7px 12px; border-bottom: 1px solid #ecf0f1; }
        tr:nth-child(even) td { background: #f8f9fa; }
        .badge { display: inline-block; padding: 2px 8px; border-radius: 12px;
                 font-size: 0.78em; font-weight: bold; }
        .badge-est { background: #fdebd0; color: #c0392b; }
        .badge-meas { background: #d5f5e3; color: #1e8449; }
        .badge-calc { background: #d6eaf8; color: #1a5276; }
        .limitation { background: #fdfefe; border-left: 3px solid #e74c3c;
                       padding: 6px 12px; margin: 4px 0; font-size: 0.88em; }
        .footer { text-align: center; font-size: 0.75em; color: #7f8c8d;
                  padding: 16px; background: #f8f9fa; }
        pre { background: #f8f9fa; padding: 10px; border-radius: 4px;
              font-size: 0.8em; overflow-x: auto; }
        """

        def row(k: str, v: Any, badge: str = "") -> str:
            badge_html = ""
            if badge == "est":
                badge_html = ' <span class="badge badge-est">Estimated</span>'
            elif badge == "meas":
                badge_html = ' <span class="badge badge-meas">Measured</span>'
            elif badge == "calc":
                badge_html = ' <span class="badge badge-calc">Calculated</span>'
            return f"<tr><td><strong>{k}</strong></td><td>{self._fmt(v)}{badge_html}</td></tr>"

        training_energy = passport.get("training_energy_kwh", {})
        is_estimated = training_energy.get("is_estimated", True)
        energy_badge = "est" if is_estimated else "meas"
        co2 = passport.get("training_co2_kg_estimate", {})
        ci_assump = passport.get("carbon_intensity_assumption", {})
        hw = passport.get("hardware", {})
        metrics = passport.get("predictive_metrics", {})
        ci_dict = passport.get("confidence_intervals", {})
        schema = passport.get("feature_schema", {})
        limitations = passport.get("known_limitations", [])
        training_cfg = passport.get("training_config", {})
        infer = passport.get("inference_latency", {})

        # Build metrics table rows with CI
        metrics_rows = ""
        for mname, mval in metrics.items():
            ci = ci_dict.get(mname, {})
            ci_str = ""
            if ci.get("ci_lower") is not None:
                ci_str = f" [CI: {self._fmt(ci['ci_lower'])}–{self._fmt(ci['ci_upper'])}]"
            metrics_rows += f"<tr><td>{mname}</td><td>{self._fmt(mval)}{ci_str}</td></tr>\n"

        # Schema table
        schema_rows = "".join(
            f"<tr><td>{col}</td><td>{dtype}</td></tr>\n"
            for col, dtype in schema.items()
        )

        # Hyperparams
        hp = training_cfg.get("hyperparameters", {})
        hp_rows = "".join(
            f"<tr><td>{k}</td><td>{self._fmt(v)}</td></tr>\n"
            for k, v in hp.items()
        )

        # Limitations
        lim_html = "".join(
            f'<div class="limitation">⚠ {lim}</div>\n' for lim in limitations
        ) if limitations else "<em>None declared</em>"

        ds_fp = passport.get("dataset_fingerprint", {})
        ds_shape = ds_fp.get("shape")
        ds_rows = ds_fp.get("rows", ds_fp.get("row_count", ds_shape[0] if isinstance(ds_shape, (list, tuple)) and len(ds_shape) >= 1 else "N/A"))
        ds_cols = ds_fp.get("cols", ds_fp.get("col_count", ds_shape[1] if isinstance(ds_shape, (list, tuple)) and len(ds_shape) >= 2 else "N/A"))

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Carbon Passport — {html_lib.escape(str(passport.get('model_name', 'Model')))}</title>
<style>{css}</style>
</head>
<body>
<div class="passport">
  <div class="header">
    <h1>🌿 Carbon Passport</h1>
    <div class="meta">
      Model: <strong>{html_lib.escape(str(passport.get('model_name', 'Unknown')))}</strong> &nbsp;|&nbsp;
      ID: {html_lib.escape(str(passport.get('model_id', 'N/A')))} &nbsp;|&nbsp;
      Generated: {html_lib.escape(str(passport.get('generated_at', 'N/A')))} &nbsp;|&nbsp;
      {html_lib.escape(str(passport.get('framework_version', '')))}
    </div>
  </div>

  <div class="section">
    <h2>Dataset</h2>
    <table>
      {row("SHA-256 hash", passport.get('dataset_hash', ''))}
      {row("Rows", ds_rows)}
      {row("Columns", ds_cols)}
    </table>
  </div>

  <div class="section">
    <h2>Feature Schema ({len(schema)} features)</h2>
    <table>
      <tr><th>Feature</th><th>Type</th></tr>
      {schema_rows}
    </table>
  </div>

  <div class="section">
    <h2>Training Configuration</h2>
    <table>
      {row("Model type", training_cfg.get('model_type', 'N/A'))}
    </table>
    {"<table><tr><th>Hyperparameter</th><th>Value</th></tr>" + hp_rows + "</table>" if hp_rows else ""}
  </div>

  <div class="section">
    <h2>Hardware</h2>
    <table>
      {row("CPU cores", hw.get('cpu_count'))}
      {row("RAM (GB)", hw.get('ram_gb'))}
      {row("OS", hw.get('os'))}
      {row("Python version", hw.get('python_version'))}
      {row("scikit-learn version", hw.get('sklearn_version'))}
    </table>
  </div>

  <div class="section">
    <h2>Energy &amp; Carbon</h2>
    <table>
      {row("Training duration (s)", passport.get('training_duration_seconds'), 'meas')}
      {row("Training energy (kWh)", training_energy.get('value'), energy_badge)}
      {row("Training CO₂ estimate (kg)", co2.get('value'), 'calc')}
      {row("Carbon intensity used (kg CO₂e/kWh)", co2.get('carbon_intensity_used'), 'calc')}
      {row("Inference energy / 1k predictions (kWh)",
            passport.get('inference_energy_per_1k_predictions_kwh'), 'est')}
      {row("Inference latency p50 (ms)", infer.get('p50_ms'), 'meas')}
      {row("Inference latency p99 (ms)", infer.get('p99_ms'), 'meas')}
    </table>
    <p style="font-size:0.82em; color:#7f8c8d;">
      <strong>Carbon intensity source:</strong> {ci_assump.get('source', 'N/A')}<br>
      <strong>Caveat:</strong> {ci_assump.get('caveat', 'N/A')}
    </p>
  </div>

  <div class="section">
    <h2>Predictive Performance</h2>
    <p style="font-size:0.82em;color:#7f8c8d;">
      Validation: {passport.get('validation_methodology', 'N/A')}
    </p>
    <table>
      <tr><th>Metric</th><th>Value</th></tr>
      {metrics_rows}
    </table>
  </div>

  <div class="section">
    <h2>Known Limitations</h2>
    {lim_html}
  </div>

  <div class="footer">
    Generated by {passport.get('framework_version', 'carbon_automl')} &nbsp;|&nbsp;
    Model ID: {passport.get('model_id', 'N/A')}
  </div>
</div>
</body>
</html>"""
        return html
