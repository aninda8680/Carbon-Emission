"""Tests for CLI entry points in main.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import yaml

from carbon_automl.artifacts.model_registry import ModelRegistry

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main as cli_main  # noqa: E402


def _write_temp_config(tmp_path: Path, config_dict: dict) -> Path:
    cfg = dict(config_dict)
    cfg["n_folds"] = 3
    cfg["n_repeats"] = 1
    cfg["trial_budget"] = 4
    cfg["measurement_repetitions"] = 1
    cfg_path = tmp_path / "test_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


def test_cli_validate(tmp_path: Path, config_dict: dict, capsys) -> None:
    """`main.py --config ... validate --print-defaults` succeeds on valid config."""
    cfg_path = _write_temp_config(tmp_path, config_dict)
    cli_main.main(["--config", str(cfg_path), "validate", "--print-defaults"])
    captured = capsys.readouterr()
    assert "Validation passed" in captured.out
    assert "Default config schema fields" in captured.out


def test_cli_profile(tmp_path: Path, config_dict: dict) -> None:
    """`main.py --config ... profile --output ...` writes profile and leakage JSON."""
    cfg_path = _write_temp_config(tmp_path, config_dict)
    out_json = tmp_path / "profile_out.json"

    cli_main.main(["--config", str(cfg_path), "profile", "--output", str(out_json)])

    assert out_json.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert "profile" in payload
    assert "leakage_warnings" in payload
    assert payload["profile"]["shape"] == [100, 5]


@pytest.mark.slow
def test_cli_train_predict_report_compare(
    tmp_path: Path,
    config_dict: dict,
    sample_classification_df: pd.DataFrame,
    capsys,
) -> None:
    """Full CLI workflow: train -> predict -> report -> compare using tmp_path."""
    cfg_path = _write_temp_config(tmp_path, config_dict)

    # Disable real CodeCarbon daemon overhead for deterministic sub-5s execution
    with patch(
        "carbon_automl.sustainability.energy_monitor.EnergyMonitor._check_codecarbon",
        return_value=False,
    ):
        cli_main.main(["--config", str(cfg_path), "train"])

    exp_dir = tmp_path / "experiments"
    models_dir = tmp_path / "models"
    exp_files = list(exp_dir.glob("experiment_*.json"))
    report_files = list(exp_dir.glob("report_*.html"))
    passport_files = list(exp_dir.glob("carbon_passport_*.json"))

    assert len(exp_files) == 1
    assert len(report_files) == 1
    assert len(passport_files) == 1

    # Verify registered model and run `predict`
    registry = ModelRegistry(str(models_dir))
    registered = registry.list_models()
    assert len(registered) == 1
    model_id = registered[0]["registry_id"]

    pred_input = tmp_path / "new_inputs.csv"
    sample_classification_df.drop(columns=["label"]).head(10).to_csv(pred_input, index=False)
    pred_output = tmp_path / "predictions.csv"

    cli_main.main([
        "--config", str(cfg_path),
        "predict",
        "--model-id", model_id,
        "--input", str(pred_input),
        "--output", str(pred_output),
    ])
    assert pred_output.exists()
    pred_df = pd.read_csv(pred_output)
    assert "prediction" in pred_df.columns
    assert len(pred_df) == 10

    # Verify `report` subcommand regenerates offline HTML
    regen_html = tmp_path / "custom_report.html"
    cli_main.main([
        "--config", str(cfg_path),
        "report",
        "--experiment", str(exp_files[0]),
        "--output", str(regen_html),
    ])
    assert regen_html.exists()
    html_text = regen_html.read_text(encoding="utf-8")
    assert "Carbon-Aware AutoML" in html_text
    assert "<style>" in html_text

    # Verify `compare` subcommand compares two experiment JSONs
    cli_main.main([
        "--config", str(cfg_path),
        "compare",
        str(exp_files[0]),
        str(exp_files[0]),
    ])
    captured = capsys.readouterr()
    assert "EXPERIMENT COMPARISON" in captured.out
    assert "Score delta (B - A)" in captured.out


@pytest.mark.slow
def test_cli_cache_reuse_across_renamed_csv(
    tmp_path: Path,
    config_dict: dict,
    sample_csv_path: Path,
) -> None:
    """Renaming an identical CSV hits cache without retraining; --force-retrain bypasses cache."""
    import shutil

    cfg_path = _write_temp_config(tmp_path, config_dict)
    with patch(
        "carbon_automl.sustainability.energy_monitor.EnergyMonitor._check_codecarbon",
        return_value=False,
    ):
        cli_main.main(["--config", str(cfg_path), "train"])

        models_dir = tmp_path / "models"
        exp_dir = tmp_path / "experiments"
        registry = ModelRegistry(str(models_dir))
        initial_models = registry.list_models()
        assert len(initial_models) == 1
        first_artifact = Path(initial_models[0]["artifact_path"])
        first_bytes = first_artifact.read_bytes()

        # Copy identical bytes to a new filename -> same content SHA256 -> cache hit
        renamed_csv = tmp_path / "renamed_copy.csv"
        shutil.copy2(sample_csv_path, renamed_csv)
        cfg_renamed = dict(config_dict)
        cfg_renamed["dataset_path"] = str(renamed_csv)
        cfg_renamed["n_folds"] = 3
        cfg_renamed["n_repeats"] = 1
        cfg_renamed["trial_budget"] = 4
        cfg_renamed["measurement_repetitions"] = 1
        cfg_renamed_path = tmp_path / "renamed_config.yaml"
        cfg_renamed_path.write_text(yaml.safe_dump(cfg_renamed), encoding="utf-8")

        cli_main.main(["--config", str(cfg_renamed_path), "train"])
        assert len(ModelRegistry(str(models_dir)).list_models()) == 1
        assert first_artifact.read_bytes() == first_bytes

        exp_files = sorted(exp_dir.glob("experiment_*.json"))
        assert len(exp_files) >= 2
        latest_exp = json.loads(exp_files[-1].read_text(encoding="utf-8"))
        assert latest_exp.get("reused_cached_model") is True

        # --force-retrain bypasses cache and trains/registers anew without overwriting first artifact
        cli_main.main(["--config", str(cfg_renamed_path), "train", "--force-retrain"])
        all_models = ModelRegistry(str(models_dir)).list_models()
        assert len(all_models) == 2
        assert first_artifact.read_bytes() == first_bytes

        exp_files_after = sorted(exp_dir.glob("experiment_*.json"))
        force_exp = json.loads(exp_files_after[-1].read_text(encoding="utf-8"))
        assert bool(force_exp.get("reused_cached_model")) is False


@pytest.mark.slow
def test_cli_parquet_regression(tmp_path: Path, regression_df: pd.DataFrame) -> None:
    """CLI `train` runs end-to-end on a Parquet regression dataset and writes reports."""
    pq_path = tmp_path / "regression_data.parquet"
    regression_df.to_parquet(pq_path, index=False)

    exp_dir = tmp_path / "exp_pq_reg"
    mod_dir = tmp_path / "mod_pq_reg"
    cfg_path = tmp_path / "pq_reg_config.yaml"
    cfg_path.write_text(
        yaml.safe_dump({
            "dataset_path": str(pq_path),
            "target_column": "value",
            "task_type": "regression",
            "validation_strategy": "kfold",
            "n_folds": 3,
            "n_repeats": 1,
            "trial_budget": 3,
            "measurement_repetitions": 1,
            "output_dir": str(exp_dir),
            "models_dir": str(mod_dir),
        }),
        encoding="utf-8",
    )

    with patch(
        "carbon_automl.sustainability.energy_monitor.EnergyMonitor._check_codecarbon",
        return_value=False,
    ):
        cli_main.main(["--config", str(cfg_path), "train"])

    exp_files = sorted(exp_dir.glob("experiment_*.json"))
    html_files = list(exp_dir.glob("report_*.html"))
    pdf_files = list(exp_dir.glob("report_*.pdf"))
    passport_files = list(exp_dir.glob("carbon_passport_*.json"))

    assert len(exp_files) == 1
    assert len(html_files) == 1
    assert len(pdf_files) == 1
    assert len(passport_files) == 1

    exp_data = json.loads(exp_files[0].read_text(encoding="utf-8"))
    assert exp_data["task_type"] == "regression"
    holdout = exp_data["holdout_test_metrics"]
    assert {"r2", "rmse", "mae"}.issubset(holdout.keys())


@pytest.mark.slow
def test_cli_group_and_temporal_splits(tmp_path: Path) -> None:
    """CLI `train` supports group_kfold and chronological time_series end-to-end."""
    import numpy as np

    rng = np.random.default_rng(77)
    n = 60

    # 1. GroupKFold dataset via CLI
    grp_csv = tmp_path / "grouped.csv"
    pd.DataFrame({
        "grp_id": np.repeat([f"g{i}" for i in range(6)], 10),
        "x1": rng.normal(0.0, 1.0, size=n),
        "x2": rng.normal(5.0, 2.0, size=n),
        "label": np.tile([0, 1, 2], 20),
    }).to_csv(grp_csv, index=False)

    exp_grp = tmp_path / "exp_grp"
    grp_cfg_path = tmp_path / "grp_config.yaml"
    grp_cfg_path.write_text(
        yaml.safe_dump({
            "dataset_path": str(grp_csv),
            "target_column": "label",
            "task_type": "classification",
            "group_column": "grp_id",
            "validation_strategy": "group_kfold",
            "n_folds": 3,
            "n_repeats": 1,
            "trial_budget": 3,
            "measurement_repetitions": 1,
            "output_dir": str(exp_grp),
            "models_dir": str(tmp_path / "mod_grp"),
        }),
        encoding="utf-8",
    )

    with patch(
        "carbon_automl.sustainability.energy_monitor.EnergyMonitor._check_codecarbon",
        return_value=False,
    ):
        cli_main.main(["--config", str(grp_cfg_path), "train"])

    grp_res = json.loads(sorted(exp_grp.glob("experiment_*.json"))[-1].read_text(encoding="utf-8"))
    assert grp_res["task_type"] == "classification"
    assert "size_aware_strategy" in grp_res
    assert "efficiency_metrics" in grp_res
    assert grp_res["best_model"]["name"]

    # 2. TimeSeries regression dataset via CLI
    ts_csv = tmp_path / "temporal.csv"
    x1 = rng.normal(0.0, 1.0, size=n)
    x2 = rng.normal(2.0, 0.5, size=n)
    pd.DataFrame({
        "ts": pd.date_range("2025-01-01", periods=n, freq="D").astype(str),
        "x1": x1,
        "x2": x2,
        "value": 3.0 * x1 - 1.5 * x2 + rng.normal(0.0, 0.2, size=n),
    }).to_csv(ts_csv, index=False)

    exp_ts = tmp_path / "exp_ts"
    ts_cfg_path = tmp_path / "ts_config.yaml"
    ts_cfg_path.write_text(
        yaml.safe_dump({
            "dataset_path": str(ts_csv),
            "target_column": "value",
            "task_type": "regression",
            "timestamp_column": "ts",
            "validation_strategy": "time_series",
            "n_folds": 3,
            "n_repeats": 1,
            "trial_budget": 3,
            "measurement_repetitions": 1,
            "output_dir": str(exp_ts),
            "models_dir": str(tmp_path / "mod_ts"),
        }),
        encoding="utf-8",
    )

    with patch(
        "carbon_automl.sustainability.energy_monitor.EnergyMonitor._check_codecarbon",
        return_value=False,
    ):
        cli_main.main(["--config", str(ts_cfg_path), "train"])

    ts_res = json.loads(sorted(exp_ts.glob("experiment_*.json"))[-1].read_text(encoding="utf-8"))
    assert ts_res["task_type"] == "regression"
    assert "r2" in ts_res["holdout_test_metrics"]
    assert "pdf_report" in ts_res and Path(ts_res["pdf_report"]).exists()
