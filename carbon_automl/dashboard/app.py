"""
Streamlit Dashboard for the Carbon-Aware AutoML Framework.

Launch with:
    streamlit run carbon_automl/dashboard/app.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

PAGES = [
    "Overview",
    "Dataset Profiling",
    "Data Integrity",
    "Leakage Analysis",
    "Model Benchmarking",
    "Energy & Carbon Analysis",
    "Pareto Frontier",
    "Carbon Budget",
    "Carbon Passport",
    "Model Registry",
    "Predictions",
    "Experiment History",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml_config() -> dict[str, Any]:
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open(encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    return {
        "dataset_path": "datasets/child_health_westbengal_1000.csv",
        "target_column": "Malnutrition Status",
        "exclude_columns": [],
        "carbon_intensity_kg_per_kwh": 0.7,
        "carbon_intensity_source": "user_configured",
        "output_dir": "experiments",
        "models_dir": "models",
        "leakage_single_feature_threshold": 0.95,
        "leakage_correlation_threshold": 0.95,
        "quality_tolerance": 0.005,
    }


def _resolve_project_path(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _load_experiments(experiments_dir: Path) -> list[dict[str, Any]]:
    if not experiments_dir.exists():
        return []
    files = sorted(experiments_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    loaded: list[dict[str, Any]] = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data["_file_name"] = f.name
                data["_file_path"] = str(f)
                loaded.append(data)
        except Exception:
            continue
    return loaded


def _load_registered_models(models_dir: Path) -> list[dict[str, Any]]:
    try:
        from carbon_automl.artifacts.model_registry import ModelRegistry

        return ModelRegistry(str(models_dir)).list_models()
    except Exception:
        return []


def _extract_model_rows(exp: dict[str, Any], carbon_intensity: float) -> list[dict[str, Any]]:
    """Normalise model result records from an experiment JSON."""
    raw_models = exp.get("models") or exp.get("model_results") or []
    if isinstance(raw_models, dict):
        rows = []
        for name, info in raw_models.items():
            row = dict(info) if isinstance(info, dict) else {}
            row.setdefault("name", name)
            rows.append(row)
        raw_models = rows

    rows_out: list[dict[str, Any]] = []
    for m in raw_models:
        entry = dict(m)
        entry.setdefault("name", entry.get("model_name", "unknown"))
        if "training_energy_kwh" in entry and entry["training_energy_kwh"] is not None:
            e_kwh = float(entry["training_energy_kwh"])
            entry["training_co2_kg_recalc"] = e_kwh * carbon_intensity
            entry["training_energy_joules"] = e_kwh * 3_600_000.0
        rows_out.append(entry)
    return rows_out


def _get_dataset_from_source(uploaded_file, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]] | None:
    """Load dataset either from Streamlit uploader or configured dataset_path."""
    from carbon_automl.core.ingestion import DataIngestionEngine

    engine = DataIngestionEngine()
    target_col = cfg.get("target_column", "Malnutrition Status")
    exclude_cols = cfg.get("exclude_columns", []) or []

    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix.lower() or ".csv"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = Path(tmp.name)
        try:
            raw_preview = pd.read_csv(tmp_path) if suffix == ".csv" else pd.read_parquet(tmp_path)
            target_options = list(raw_preview.columns)
            chosen_target = st.selectbox(
                "Target column for uploaded dataset",
                options=target_options,
                index=target_options.index(target_col) if target_col in target_options else len(target_options) - 1,
            )
            return engine.load(str(tmp_path), target_column=chosen_target, exclude_columns=[])
        finally:
            tmp_path.unlink(missing_ok=True)

    ds_path = _resolve_project_path(cfg.get("dataset_path", "datasets/child_health_westbengal_1000.csv"))
    if not ds_path.exists():
        st.warning(f"Configured dataset not found: `{ds_path}`")
        return None
    return engine.load(str(ds_path), target_column=target_col, exclude_columns=exclude_cols)


# ---------------------------------------------------------------------------
# Page Renderers
# ---------------------------------------------------------------------------

def page_overview(cfg: dict[str, Any], experiments: list[dict], models: list[dict], carbon_intensity: float) -> None:
    st.header("Overview")
    st.write(
        "Carbon-Aware AutoML Framework evaluates predictive quality alongside compute energy "
        "and CO₂e emissions, detects target leakage before training, and communicates "
        "measurement uncertainty explicitly."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Registered Models", len(models))
    col2.metric("Completed Experiments", len(experiments))
    col3.metric("Carbon Intensity (kg CO₂e/kWh)", f"{carbon_intensity:.3f}")

    st.subheader("Carbon Intensity Assumption")
    st.info(
        f"Current grid carbon intensity: **{carbon_intensity:.4f} kg CO₂e/kWh** "
        f"(source: `{cfg.get('carbon_intensity_source', 'user_configured')}`). "
        "Static grid averages exclude embodied hardware emissions, PUE cooling overhead, and regional temporal variation."
    )

    if experiments:
        latest = experiments[0]
        bm = latest.get("best_model", {})
        st.subheader("Latest Experiment Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Recommended Model", str(bm.get("name", bm.get("model_name", "N/A"))))
        score_val = bm.get("macro_f1", bm.get("r2", bm.get("mean_score")))
        c2.metric("Primary Score", f"{score_val:.4f}" if isinstance(score_val, (int, float)) else "N/A")
        total_co2 = latest.get("carbon", {}).get("total_co2_kg")
        c3.metric("Total CO₂e (kg)", f"{total_co2:.6g}" if isinstance(total_co2, (int, float)) else "N/A")
    else:
        st.info("No experiments yet. Run `python main.py --config config.yaml train` to populate experiment results.")


def page_dataset_profiling(cfg: dict[str, Any]) -> None:
    st.header("Dataset Profiling")
    uploaded = st.file_uploader("Upload CSV or Parquet (optional — defaults to config.yaml dataset)", type=["csv", "parquet"])

    try:
        loaded = _get_dataset_from_source(uploaded, cfg)
    except Exception as exc:
        st.error(f"Failed to load dataset: {exc}")
        return
    if loaded is None:
        return

    X, y, metadata = loaded
    from carbon_automl.core.data_profiler import DataProfiler

    profile = DataProfiler().profile(X, y)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", metadata["shape"][0])
    c2.metric("Features", X.shape[1])
    c3.metric("Memory (MB)", f"{profile.get('memory_mb', 0):.3f}")
    c4.metric("Suggested Task", str(profile.get("task_type_suggestion", "unknown")))

    st.subheader("Feature Types & Missingness")
    ft_rows = []
    for col, ftype in profile.get("feature_types", {}).items():
        ft_rows.append({
            "Feature": col,
            "Detected Type": ftype,
            "Missing": profile.get("missing_values", {}).get(col, 0),
            "Outliers (±3 IQR)": profile.get("outliers", {}).get(col, 0),
        })
    st.dataframe(pd.DataFrame(ft_rows), use_container_width=True)

    class_dist = profile.get("class_distribution")
    if class_dist and class_dist.get("counts"):
        st.subheader("Target Class Distribution")
        counts = class_dist["counts"]
        dist_df = pd.DataFrame({"Class": list(counts.keys()), "Count": list(counts.values())})
        fig = px.bar(dist_df, x="Class", y="Count", title="Class Distribution")
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        if class_dist.get("imbalance_ratio") is not None:
            st.caption(f"Imbalance ratio (max/min): {class_dist['imbalance_ratio']}")

    if profile.get("recommendations"):
        st.subheader("Recommendations")
        for rec in profile["recommendations"]:
            st.write(f"- {rec}")


def page_data_integrity(cfg: dict[str, Any]) -> None:
    st.header("Data Integrity")
    ds_path = _resolve_project_path(cfg.get("dataset_path", "datasets/child_health_westbengal_1000.csv"))
    if not ds_path.exists():
        st.warning(f"Dataset file not found at `{ds_path}`.")
        return

    from carbon_automl.core.integrity_manager import DataIntegrityError, DataIntegrityManager

    raw_df = pd.read_parquet(ds_path) if ds_path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(ds_path)
    manager = DataIntegrityManager(str(ds_path))
    fp = manager.record_original(raw_df)
    try:
        verified = manager.verify_unchanged(raw_df)
    except DataIntegrityError as exc:
        verified = False
        st.error(f"Integrity verification failed: {exc}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Verification Status", "PASS" if verified else "FAIL")
    c2.metric("Row Count", fp["row_count"])
    c3.metric("Column Count", fp["col_count"])

    st.text(f"SHA-256: {fp['sha256']}")
    st.text(f"MD5:     {fp['md5']}")

    st.subheader("Recorded Schema")
    schema_df = pd.DataFrame(
        [{"Column": col, "Dtype": dtype} for col, dtype in fp["dtypes"].items()]
    )
    st.dataframe(schema_df, use_container_width=True)


def page_leakage_analysis(cfg: dict[str, Any], experiments: list[dict]) -> None:
    st.header("Leakage Analysis")

    warnings_list: list[dict[str, Any]] = []
    source_label = ""
    if experiments and experiments[0].get("leakage_warnings") is not None:
        warnings_list = experiments[0].get("leakage_warnings", [])
        source_label = f"From latest experiment (`{experiments[0].get('_file_name')}`)"
    else:
        try:
            loaded = _get_dataset_from_source(None, cfg)
            if loaded is not None:
                X, y, _ = loaded
                from carbon_automl.core.data_profiler import DataProfiler
                from carbon_automl.leakage.detector import LeakageDetector

                profile = DataProfiler().profile(X, y)
                detector = LeakageDetector()
                warnings_list = detector.detect(
                    X,
                    y,
                    feature_types=profile["feature_types"],
                    task_type=profile["task_type_suggestion"],
                    single_feature_threshold=cfg.get("leakage_single_feature_threshold", 0.95),
                    correlation_threshold=cfg.get("leakage_correlation_threshold", 0.95),
                )
                source_label = "Live scan on configured dataset"
        except ImportError:
            st.info("LeakageDetector module not available yet.")
            return
        except Exception as exc:
            st.error(f"Leakage scan failed: {exc}")
            return

    if source_label:
        st.caption(source_label)

    if not warnings_list:
        st.success("No leakage warnings detected.")
        return

    def _badge(sev: str) -> str:
        s = str(sev).lower()
        if s == "critical":
            return "🔴 CRITICAL"
        if s == "high":
            return "🟠 HIGH"
        if s == "medium":
            return "🟡 MEDIUM"
        return "⚪ LOW"

    df_warn = pd.DataFrame([
        {
            "Severity": _badge(w.get("severity", "low")),
            "Feature": w.get("feature", ""),
            "Detection Method": w.get("detection_method", ""),
            "Evidence": w.get("evidence", ""),
            "Recommended Investigation": w.get("recommended_investigation", ""),
        }
        for w in warnings_list
    ])
    st.dataframe(df_warn, use_container_width=True)


def page_model_benchmarking(experiments: list[dict], carbon_intensity: float) -> None:
    st.header("Model Benchmarking")
    if not experiments:
        st.info("No experiments yet. Run training or benchmark mode to view model comparisons.")
        return

    rows = _extract_model_rows(experiments[0], carbon_intensity)
    if not rows:
        st.info("No model benchmark rows found in the latest experiment.")
        return

    df = pd.DataFrame(rows)
    sort_col = st.selectbox("Sort by column", options=list(df.columns), index=0)
    ascending = st.checkbox("Ascending order", value=False)
    st.dataframe(df.sort_values(by=sort_col, ascending=ascending), use_container_width=True)


def page_energy_carbon(experiments: list[dict], carbon_intensity: float) -> None:
    st.header("Energy & Carbon Analysis")
    if not experiments:
        st.info("No experiments yet. Run training to collect per-phase energy and CO₂e estimates.")
        return

    exp = experiments[0]
    phases = exp.get("phases", {})
    if phases:
        phase_rows = []
        for phase_name, pdata in phases.items():
            if not isinstance(pdata, dict):
                continue
            energy_kwh = float(pdata.get("energy_kwh") or 0.0)
            phase_rows.append({
                "Phase": phase_name,
                "Energy (kWh)": energy_kwh,
                "Energy (J)": energy_kwh * 3_600_000.0,
                "Duration (s)": float(pdata.get("duration_seconds") or 0.0),
                "CO₂e (kg)": energy_kwh * carbon_intensity,
            })
        if phase_rows:
            pdf = pd.DataFrame(phase_rows)
            fig_bar = px.bar(pdf, x="Phase", y="Energy (J)", title="Energy Consumption per Phase (Joules)")
            fig_bar.update_layout(template="plotly_white")
            st.plotly_chart(fig_bar, use_container_width=True)
            st.dataframe(pdf, use_container_width=True)
    else:
        st.info("No per-phase measurements recorded in this experiment.")

    carbon = exp.get("carbon", {})
    train_co2 = float(carbon.get("training_co2_kg", carbon.get("total_co2_kg", 0.0)) or 0.0)
    infer_co2 = float(carbon.get("inference_co2_kg", 0.0) or 0.0)
    if train_co2 > 0 or infer_co2 > 0:
        pie_df = pd.DataFrame({
            "Stage": ["Training", "Inference (Amortised)"],
            "CO₂e (kg)": [train_co2, infer_co2],
        })
        fig_pie = px.pie(pie_df, names="Stage", values="CO₂e (kg)", title="Training vs Inference Emissions")
        fig_pie.update_layout(template="plotly_white")
        st.plotly_chart(fig_pie, use_container_width=True)

    st.caption(
        "Note: At small dataset sizes (~1,000 rows), sub-second phase measurements are estimates "
        "subject to ±50–100% background noise."
    )


def page_pareto_frontier(experiments: list[dict], carbon_intensity: float) -> None:
    st.header("Pareto Frontier")
    if not experiments:
        st.info("No experiments yet. Run training to inspect the quality-vs-energy Pareto frontier.")
        return

    rows = _extract_model_rows(experiments[0], carbon_intensity)
    if not rows:
        st.info("No model rows available in the latest experiment.")
        return

    quality_metric = "macro_f1" if any("macro_f1" in r for r in rows) else (
        "r2" if any("r2" in r for r in rows) else "mean_score"
    )
    energy_metric = "training_energy_kwh" if any("training_energy_kwh" in r for r in rows) else "training_duration_seconds"

    valid = [r for r in rows if r.get(quality_metric) is not None and r.get(energy_metric) is not None]
    if not valid:
        st.info(f"Models lack both `{quality_metric}` and `{energy_metric}`.")
        return

    try:
        from carbon_automl.sustainability.pareto import ParetoAnalyzer

        pareto_models = ParetoAnalyzer().compute_pareto_frontier(
            valid, objectives=[quality_metric, energy_metric], minimize=[False, True]
        )
        pareto_names = {m.get("name") for m in pareto_models}
    except Exception:
        pareto_names = set()

    df = pd.DataFrame(valid)
    df["Pareto Optimal"] = df["name"].isin(pareto_names)

    fig = go.Figure()
    non_p = df[~df["Pareto Optimal"]]
    if not non_p.empty:
        fig.add_trace(go.Scatter(
            x=non_p[energy_metric],
            y=non_p[quality_metric],
            mode="markers+text",
            text=non_p["name"],
            textposition="top center",
            name="Dominated",
            marker=dict(size=9, color="#7f8c8d"),
        ))

    on_p = df[df["Pareto Optimal"]].sort_values(by=energy_metric)
    if not on_p.empty:
        fig.add_trace(go.Scatter(
            x=on_p[energy_metric],
            y=on_p[quality_metric],
            mode="lines+markers+text",
            text=on_p["name"],
            textposition="top center",
            name="Pareto Frontier",
            marker=dict(size=12, color="#1f77b4"),
            line=dict(dash="dash", color="#1f77b4"),
        ))

    fig.update_layout(
        title="Quality vs Energy Pareto Frontier",
        xaxis_title=f"{energy_metric} (lower is better)",
        yaxis_title=f"{quality_metric} (higher is better)",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)


def page_carbon_budget(experiments: list[dict], carbon_intensity: float) -> None:
    st.header("Carbon Budget")
    if not experiments:
        st.info("No experiments yet. Run training before evaluating carbon budgets.")
        return

    rows = _extract_model_rows(experiments[0], carbon_intensity)
    if not rows:
        st.info("No candidate models available in the latest experiment.")
        return

    budget_kg = st.number_input(
        "Carbon budget (kg CO₂e)",
        min_value=0.0,
        value=0.001,
        step=0.0001,
        format="%.6f",
    )

    try:
        from carbon_automl.sustainability.carbon_budget import CarbonBudgetOptimizer
        from carbon_automl.sustainability.carbon_calculator import CarbonCalculator

        calc = CarbonCalculator(carbon_intensity_kg_per_kwh=carbon_intensity)
        optimizer = CarbonBudgetOptimizer(budget_kg=budget_kg, carbon_calculator=calc)
        res = optimizer.find_models_within_budget(rows)
    except Exception as exc:
        st.error(f"Budget analysis failed: {exc}")
        return

    if res.get("conflict_explanation"):
        st.warning(res["conflict_explanation"])
    elif res.get("best_within_budget"):
        best = res["best_within_budget"]
        st.success(f"Best model within budget: **{best.get('name')}**")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader(f"Within Budget ({len(res.get('within_budget', []))})")
        if res.get("within_budget"):
            st.dataframe(pd.DataFrame(res["within_budget"]), use_container_width=True)
    with c2:
        st.subheader(f"Over Budget ({len(res.get('over_budget', []))})")
        if res.get("over_budget"):
            st.dataframe(pd.DataFrame(res["over_budget"]), use_container_width=True)


def page_carbon_passport(models: list[dict], experiments: list[dict]) -> None:
    st.header("Carbon Passport")

    passport_doc: dict[str, Any] | None = None
    if models:
        options = {f"{m['model_name']} ({m['registry_id'][:8]})": m for m in models}
        selected = st.selectbox("Select registered model", options=list(options.keys()))
        entry = options[selected]
        pp_path = _resolve_project_path(entry.get("carbon_passport_path", ""))
        if pp_path.exists():
            passport_doc = json.loads(pp_path.read_text(encoding="utf-8"))
    elif experiments and experiments[0].get("carbon_passport"):
        passport_doc = experiments[0]["carbon_passport"]

    if not passport_doc:
        st.info("No experiments yet or no Carbon Passports generated.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Model", str(passport_doc.get("model_name", "N/A")))
    c2.metric("Duration (s)", f"{passport_doc.get('training_duration_seconds', 0):.4f}")
    co2_val = passport_doc.get("training_co2_kg_estimate", {}).get("value", 0.0)
    c3.metric("Training CO₂e (kg)", f"{co2_val:.6g}")

    st.json(passport_doc)


def page_model_registry(models: list[dict]) -> None:
    st.header("Model Registry")
    if not models:
        st.info("No registered models yet. Run `python main.py --config config.yaml train` to register a model.")
        return

    flat_rows = []
    for m in models:
        metrics = m.get("metrics", {})
        flat_rows.append({
            "registry_id": m.get("registry_id"),
            "model_name": m.get("model_name"),
            "registered_at": m.get("registered_at"),
            "experiment_hash": str(m.get("experiment_hash", ""))[:12],
            **{f"metric_{k}": v for k, v in metrics.items() if isinstance(v, (int, float, str))},
        })
    st.dataframe(pd.DataFrame(flat_rows), use_container_width=True)


def page_predictions(models: list[dict]) -> None:
    st.header("Predictions")
    if not models:
        st.info("No registered models available for prediction yet.")
        return

    options = {f"{m['model_name']} ({m['registry_id'][:8]})": m for m in models}
    selected = st.selectbox("Select model", options=list(options.keys()))
    model_meta = options[selected]

    uploaded = st.file_uploader("Upload CSV for batch prediction", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV file to run predictions.")
        return

    df = pd.read_csv(uploaded)
    st.dataframe(df.head(5), use_container_width=True)

    if st.button("Run Predictions"):
        artifact_path = _resolve_project_path(model_meta.get("artifact_path", ""))
        if not artifact_path.exists():
            st.error(f"Model artifact not found: `{artifact_path}`")
            return
        try:
            artifact = joblib.load(artifact_path)
            if isinstance(artifact, dict) and "model" in artifact:
                feature_names = artifact.get("feature_names")
                if feature_names:
                    missing = [c for c in feature_names if c not in df.columns]
                    if missing:
                        st.error(
                            f"Uploaded CSV is missing required feature column(s): {missing}. "
                            f"Expected columns: {list(feature_names)}"
                        )
                        return
                    df_in = df[list(feature_names)]
                else:
                    df_in = df
                X_in = artifact["preprocessor"].transform(df_in) if artifact.get("preprocessor") else df_in
                preds = artifact["model"].predict(X_in)
                le = artifact.get("label_encoder")
                if le is not None:
                    preds = le.inverse_transform(preds)
            else:
                preds = artifact.predict(df)
            out_df = df.copy()
            out_df["prediction"] = preds
            st.success(f"Generated {len(preds)} predictions.")
            st.dataframe(out_df, use_container_width=True)
            st.download_button(
                "Download Predictions CSV",
                data=out_df.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


def page_experiment_history(experiments: list[dict]) -> None:
    st.header("Experiment History")
    if not experiments:
        st.info("No experiments yet in `experiments/`.")
        return

    rows = []
    for exp in experiments:
        bm = exp.get("best_model", {})
        rows.append({
            "File": exp.get("_file_name"),
            "Timestamp": exp.get("timestamp"),
            "Best Model": bm.get("name", bm.get("model_name", "N/A")),
            "Score": bm.get("macro_f1", bm.get("r2", bm.get("mean_score"))),
            "Total CO₂ (kg)": exp.get("carbon", {}).get("total_co2_kg"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ---------------------------------------------------------------------------
# App Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Carbon-Aware AutoML", layout="wide")

    cfg = _load_yaml_config()
    experiments_dir = _resolve_project_path(cfg.get("output_dir", "experiments"))
    models_dir = _resolve_project_path(cfg.get("models_dir", "models"))

    st.sidebar.title("Carbon-Aware AutoML")
    page = st.sidebar.radio("Navigation", PAGES)

    st.sidebar.markdown("---")
    carbon_intensity = st.sidebar.number_input(
        "Carbon Intensity Override (kg CO₂e/kWh)",
        min_value=0.001,
        max_value=5.0,
        value=float(cfg.get("carbon_intensity_kg_per_kwh", 0.7)),
        step=0.05,
        format="%.3f",
    )

    experiments = _load_experiments(experiments_dir)
    models = _load_registered_models(models_dir)

    if page == "Overview":
        page_overview(cfg, experiments, models, carbon_intensity)
    elif page == "Dataset Profiling":
        page_dataset_profiling(cfg)
    elif page == "Data Integrity":
        page_data_integrity(cfg)
    elif page == "Leakage Analysis":
        page_leakage_analysis(cfg, experiments)
    elif page == "Model Benchmarking":
        page_model_benchmarking(experiments, carbon_intensity)
    elif page == "Energy & Carbon Analysis":
        page_energy_carbon(experiments, carbon_intensity)
    elif page == "Pareto Frontier":
        page_pareto_frontier(experiments, carbon_intensity)
    elif page == "Carbon Budget":
        page_carbon_budget(experiments, carbon_intensity)
    elif page == "Carbon Passport":
        page_carbon_passport(models, experiments)
    elif page == "Model Registry":
        page_model_registry(models)
    elif page == "Predictions":
        page_predictions(models)
    elif page == "Experiment History":
        page_experiment_history(experiments)


if __name__ == "__main__":
    main()
