"""
Pydantic v2 configuration schema for the Carbon-Aware AutoML framework.

This module defines the validated configuration model that governs all aspects
of a training run: data loading, validation strategy, carbon accounting, and
leakage detection thresholds.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class CarbonAutoMLConfig(BaseModel):
    """
    Validated configuration for a Carbon-Aware AutoML training run.

    All fields have sensible defaults so the user only needs to specify
    ``dataset_path`` and ``target_column`` to get started.
    """

    # ------------------------------------------------------------------ #
    # Data                                                                 #
    # ------------------------------------------------------------------ #
    dataset_path: str = Field(..., description="Absolute or relative path to the dataset file (.csv or .parquet).")
    target_column: str = Field(..., description="Name of the column to predict.")
    task_type: Literal["auto", "classification", "regression"] = Field(
        "auto",
        description="ML task type. 'auto' infers from target cardinality.",
    )
    exclude_columns: list[str] = Field(
        default_factory=list,
        description="Columns to drop before modelling (e.g. identifiers, leaky features).",
    )
    group_column: str | None = Field(
        None,
        description="Column used for group-aware splitting (GroupKFold / group_kfold strategy).",
    )
    timestamp_column: str | None = Field(
        None,
        description="Column used for time-ordered splitting (time_series strategy).",
    )

    # ------------------------------------------------------------------ #
    # Reproducibility                                                      #
    # ------------------------------------------------------------------ #
    random_seed: int = Field(42, description="Global random seed for reproducibility.")

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #
    validation_strategy: Literal[
        "stratified_kfold", "kfold", "group_kfold", "stratified_group_kfold", "time_series"
    ] = Field(
        "stratified_kfold",
        description="Cross-validation strategy.",
    )
    n_folds: int = Field(5, ge=2, description="Number of CV folds.")
    n_repeats: int = Field(3, ge=1, description="Number of repeated CV rounds.")

    # ------------------------------------------------------------------ #
    # Optimisation & Quality Protection                                    #
    # ------------------------------------------------------------------ #
    trial_budget: int = Field(30, ge=1, description="Maximum number of HPO trials.")
    quality_tolerance: float = Field(
        0.005,
        ge=0.0,
        description=(
            "Maximum allowed quality drop from baseline (e.g. 0.005 = 0.5 percentage points)."
        ),
    )
    critical_class_recall_constraints: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Optional per-class minimum recall constraints (e.g. {'Severe': 0.85}). "
            "Models violating any constraint are excluded from recommendation."
        ),
    )
    regression_tolerances: dict[str, float] = Field(
        default_factory=lambda: {"r2": 0.005, "rmse_rel": 0.05, "mae_rel": 0.05},
        description="Configurable absolute/relative tolerances for regression metrics (R², RMSE, MAE).",
    )
    adaptive_optimization: bool = Field(
        True,
        description="Enable size- and complexity-aware adaptive model/tuning strategy.",
    )

    # ------------------------------------------------------------------ #
    # Carbon accounting                                                    #
    # ------------------------------------------------------------------ #
    carbon_intensity_kg_per_kwh: float = Field(
        0.7,
        gt=0.0,
        description="Grid carbon intensity in kg CO₂e / kWh. Default: India national average.",
    )
    carbon_intensity_source: str = Field(
        "user_configured",
        description="Provenance of the grid carbon intensity value (e.g. user_configured, electricitymap, watttime).",
    )
    carbon_budget_kg: float | None = Field(
        None,
        description="Optional hard carbon budget in kg CO₂e. Training stops early if exceeded.",
    )
    real_time_carbon_api_url: str | None = Field(
        None,
        description="Optional external grid carbon-intensity forecast API endpoint for experimental CarbonScheduler.",
    )

    # ------------------------------------------------------------------ #
    # Outputs                                                              #
    # ------------------------------------------------------------------ #
    output_dir: str = Field("experiments", description="Directory for experiment artefacts.")
    models_dir: str = Field("models", description="Directory for saved model artefacts.")

    # ------------------------------------------------------------------ #
    # Models                                                               #
    # ------------------------------------------------------------------ #
    enable_xgboost: bool = Field(
        False,
        description="Include XGBoost in the model search (requires xgboost to be installed).",
    )

    # ------------------------------------------------------------------ #
    # Energy measurement                                                   #
    # ------------------------------------------------------------------ #
    measurement_repetitions: int = Field(
        3,
        ge=1,
        description="Number of times each energy measurement is repeated for averaging.",
    )

    # ------------------------------------------------------------------ #
    # Leakage detection                                                    #
    # ------------------------------------------------------------------ #
    leakage_single_feature_threshold: float = Field(
        0.95,
        gt=0.0,
        le=1.0,
        description=(
            "Single-feature accuracy threshold above which a feature is "
            "flagged as potentially leaky."
        ),
    )
    leakage_correlation_threshold: float = Field(
        0.95,
        gt=0.0,
        le=1.0,
        description=(
            "Pearson |r| threshold with the target above which a numerical "
            "feature is flagged as potentially leaky."
        ),
    )

    # ------------------------------------------------------------------ #
    # Validators                                                           #
    # ------------------------------------------------------------------ #
    @field_validator("exclude_columns", mode="before")
    @classmethod
    def _coerce_exclude_columns(cls, v: object) -> list[str]:
        """Accept None as an empty list."""
        if v is None:
            return []
        return list(v)

    @model_validator(mode="after")
    def _check_group_and_timestamp_consistency(self) -> "CarbonAutoMLConfig":
        """Warn-level consistency: group_kfold requires group_column, time_series requires timestamp_column."""
        if self.validation_strategy in ("group_kfold", "stratified_group_kfold") and self.group_column is None:
            raise ValueError(
                f"validation_strategy='{self.validation_strategy}' requires 'group_column' to be set."
            )
        if self.validation_strategy == "time_series" and self.timestamp_column is None:
            raise ValueError(
                "validation_strategy='time_series' requires 'timestamp_column' to be set."
            )
        return self

    model_config = {"extra": "forbid", "validate_assignment": True}


# Backward-compatible alias
AutoMLConfig = CarbonAutoMLConfig
