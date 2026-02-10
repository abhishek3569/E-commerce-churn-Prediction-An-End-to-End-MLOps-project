from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class DataQualityResult:
    missing_fraction: Dict[str, float]
    numeric_outliers: Dict[str, int]
    schema_valid: bool
    drift_metrics: Optional[Dict[str, float]] = None


def check_missing_values(df: pd.DataFrame, threshold: float = 0.2) -> Dict[str, float]:
    fractions = df.isna().mean().to_dict()
    severe = {col: frac for col, frac in fractions.items() if frac > threshold}
    return severe


def detect_outliers_iqr(df: pd.DataFrame, factor: float = 1.5) -> Dict[str, int]:
    numeric = df.select_dtypes(include=[np.number])
    outliers: Dict[str, int] = {}
    for col in numeric.columns:
        q1 = numeric[col].quantile(0.25)
        q3 = numeric[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            outliers[col] = 0
            continue
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        mask = (numeric[col] < lower) | (numeric[col] > upper)
        outliers[col] = int(mask.sum())
    return outliers


def validate_schema(df: pd.DataFrame, expected_schema: Dict[str, str]) -> bool:
    """
    Simple schema validation based on presence of columns and dtypes category.
    expected_schema: mapping column -> 'numeric' or 'categorical'
    """
    for col, kind in expected_schema.items():
        if col not in df.columns:
            return False
        if kind == "numeric" and not np.issubdtype(df[col].dtype, np.number):
            return False
        if kind == "categorical" and np.issubdtype(df[col].dtype, np.number):
            return False
    return True


def population_stability_index(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute PSI for numerical feature drift detection."""
    expected = np.asarray(expected)
    actual = np.asarray(actual)

    quantiles = np.linspace(0, 1, n_bins + 1)
    bins = np.unique(np.quantile(expected, quantiles))
    if len(bins) <= 2:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)

    expected_perc = expected_counts / np.clip(expected_counts.sum(), 1, None)
    actual_perc = actual_counts / np.clip(actual_counts.sum(), 1, None)

    mask = (expected_perc > 0) & (actual_perc > 0)
    psi = np.sum((actual_perc[mask] - expected_perc[mask]) * np.log(actual_perc[mask] / expected_perc[mask]))
    return float(psi)


def chi_square_drift(
    expected: np.ndarray,
    actual: np.ndarray,
) -> float:
    """
    Chi-square statistic for categorical drift detection.
    Returns the p-value (small values indicate drift).
    """
    expected_vals, expected_counts = np.unique(expected, return_counts=True)
    actual_vals, actual_counts = np.unique(actual, return_counts=True)
    categories = sorted(set(expected_vals) | set(actual_vals))

    exp_counts_aligned: List[int] = []
    act_counts_aligned: List[int] = []
    for c in categories:
        exp_counts_aligned.append(int(expected_counts[expected_vals == c].sum()))
        act_counts_aligned.append(int(actual_counts[actual_vals == c].sum()))

    chi2, p, _, _ = stats.chi2_contingency([exp_counts_aligned, act_counts_aligned])
    _ = chi2  # not used currently
    return float(p)


def validate_data_quality(
    current_df: pd.DataFrame,
    training_df: Optional[pd.DataFrame] = None,
    expected_schema: Optional[Dict[str, str]] = None,
) -> DataQualityResult:
    missing = check_missing_values(current_df)
    outliers = detect_outliers_iqr(current_df)

    schema_valid = True
    if expected_schema is not None:
        schema_valid = validate_schema(current_df, expected_schema)

    drift_metrics: Dict[str, float] = {}
    if training_df is not None:
        # PSI for numeric
        numeric_cols = current_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            psi = population_stability_index(training_df[col].dropna(), current_df[col].dropna())
            drift_metrics[f"psi__{col}"] = psi

        # Chi-square for categorical
        cat_cols = current_df.select_dtypes(exclude=[np.number]).columns
        for col in cat_cols:
            p_val = chi_square_drift(training_df[col].dropna(), current_df[col].dropna())
            drift_metrics[f"chi2_p__{col}"] = p_val

    return DataQualityResult(
        missing_fraction=missing,
        numeric_outliers=outliers,
        schema_valid=schema_valid,
        drift_metrics=drift_metrics if drift_metrics else None,
    )


__all__ = [
    "DataQualityResult",
    "check_missing_values",
    "detect_outliers_iqr",
    "validate_schema",
    "population_stability_index",
    "chi_square_drift",
    "validate_data_quality",
]

