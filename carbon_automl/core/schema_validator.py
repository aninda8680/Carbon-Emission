"""
Schema and configuration validation for the Carbon-Aware AutoML framework.

Provides centralised validation for user configs (via Pydantic v2),
DataFrame schemas, and lossless dtype conversions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import ValidationError

from carbon_automl.config.schema import CarbonAutoMLConfig


class SchemaValidationError(ValueError):
    """Raised when configuration or DataFrame schema validation fails."""


class SchemaValidator:
    """
    Validate configurations, DataFrame schemas, and dtype conversions.

    Examples
    --------
    >>> validator = SchemaValidator()
    >>> cfg = validator.validate_config({"dataset_path": "d.csv", "target_column": "y"})
    >>> ok = validator.validate_safe_dtype_conversion(pd.Series([1.0, 2.0]), np.float32)
    """

    # ------------------------------------------------------------------ #
    # Config validation                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def validate_config(config: dict[str, Any] | CarbonAutoMLConfig) -> CarbonAutoMLConfig:
        """
        Validate *config* using :class:`~carbon_automl.config.schema.CarbonAutoMLConfig`.

        Parameters
        ----------
        config:
            Raw configuration dictionary (or existing ``CarbonAutoMLConfig``).

        Returns
        -------
        CarbonAutoMLConfig
            Validated Pydantic v2 model instance.

        Raises
        ------
        SchemaValidationError
            With a descriptive message listing every invalid field if
            validation fails.
        """
        if isinstance(config, CarbonAutoMLConfig):
            return CarbonAutoMLConfig.model_validate(config.model_dump())

        if not isinstance(config, dict):
            raise SchemaValidationError(
                f"Expected config to be a dict or CarbonAutoMLConfig, got {type(config).__name__}."
            )

        try:
            return CarbonAutoMLConfig.model_validate(config)
        except ValidationError as exc:
            details: list[str] = []
            for err in exc.errors():
                loc = ".".join(str(part) for part in err.get("loc", ())) or "<root>"
                msg = err.get("msg", "Invalid value")
                details.append(f"  - {loc}: {msg}")
            formatted = "\n".join(details)
            raise SchemaValidationError(
                f"Configuration validation failed with {len(details)} error(s):\n{formatted}"
            ) from exc

    # ------------------------------------------------------------------ #
    # DataFrame schema validation                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def validate_dataframe(
        df: pd.DataFrame,
        expected_schema: dict[str, Any],
    ) -> bool:
        """
        Check that all columns in *expected_schema* exist in *df* with
        compatible dtypes.

        Parameters
        ----------
        df:
            DataFrame to inspect (not modified).
        expected_schema:
            Mapping of ``column_name -> expected_dtype`` (dtype string or
            NumPy/pandas dtype object).

        Returns
        -------
        bool
            ``True`` if every expected column is present with a compatible
            dtype; ``False`` otherwise.
        """
        if not isinstance(df, pd.DataFrame):
            return False

        for col, expected_dtype in expected_schema.items():
            if col not in df.columns:
                return False
            if not SchemaValidator._are_dtypes_compatible(df[col].dtype, expected_dtype):
                return False

        return True

    # ------------------------------------------------------------------ #
    # Safe dtype conversion validation                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def validate_safe_dtype_conversion(series: pd.Series, target_dtype: Any) -> bool:
        """
        Return ``True`` only if **all** values in *series* survive a
        round-trip conversion to *target_dtype* and back without loss.

        Parameters
        ----------
        series:
            Input pandas Series.
        target_dtype:
            Target NumPy/pandas dtype (e.g. ``np.float32``, ``"int32"``).

        Returns
        -------
        bool
            ``True`` when conversion + round-trip restores all values and
            missing-value positions identically; ``False`` otherwise.
        """
        if not isinstance(series, pd.Series):
            return False

        try:
            converted = series.astype(target_dtype)
            round_tripped = converted.astype(series.dtype)
        except (TypeError, ValueError, OverflowError):
            return False

        # Missing-value mask must remain identical
        orig_na = series.isna().to_numpy()
        conv_na = converted.isna().to_numpy()
        rt_na = round_tripped.isna().to_numpy()
        if not (np.array_equal(orig_na, conv_na) and np.array_equal(orig_na, rt_na)):
            return False

        # Compare non-missing values
        valid_mask = ~orig_na
        if not np.any(valid_mask):
            return True

        orig_vals = series.iloc[valid_mask].to_numpy()
        rt_vals = round_tripped.iloc[valid_mask].to_numpy()

        # Use equal_nan=True only for floating/complex dtypes; fallback to array_equal
        try:
            if np.issubdtype(orig_vals.dtype, np.floating) or np.issubdtype(
                orig_vals.dtype, np.complexfloating
            ):
                return bool(np.array_equal(orig_vals, rt_vals, equal_nan=True))
        except TypeError:
            pass

        return bool(np.array_equal(orig_vals, rt_vals))

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _are_dtypes_compatible(actual_dtype: Any, expected_dtype: Any) -> bool:
        """Check whether *actual_dtype* is compatible with *expected_dtype*."""
        if str(actual_dtype) == str(expected_dtype):
            return True

        try:
            exp_dt = pd.api.types.pandas_dtype(expected_dtype)
            act_dt = pd.api.types.pandas_dtype(actual_dtype)
        except TypeError:
            return False

        if act_dt == exp_dt:
            return True

        # Booleans must match booleans
        act_is_bool = pd.api.types.is_bool_dtype(act_dt)
        exp_is_bool = pd.api.types.is_bool_dtype(exp_dt)
        if act_is_bool or exp_is_bool:
            return act_is_bool and exp_is_bool

        # Datetime must match datetime
        act_is_dt = pd.api.types.is_datetime64_any_dtype(act_dt)
        exp_is_dt = pd.api.types.is_datetime64_any_dtype(exp_dt)
        if act_is_dt or exp_is_dt:
            return act_is_dt and exp_is_dt

        # Numeric (int/float) are compatible within the numeric family
        act_is_num = pd.api.types.is_numeric_dtype(act_dt)
        exp_is_num = pd.api.types.is_numeric_dtype(exp_dt)
        if act_is_num and exp_is_num:
            return True
        if act_is_num or exp_is_num:
            return False

        # Object / string / categorical are mutually compatible
        def _is_text_or_cat(dt: Any) -> bool:
            return bool(
                pd.api.types.is_object_dtype(dt)
                or pd.api.types.is_string_dtype(dt)
                or isinstance(dt, pd.CategoricalDtype)
            )

        if _is_text_or_cat(act_dt) and _is_text_or_cat(exp_dt):
            return True

        return False
