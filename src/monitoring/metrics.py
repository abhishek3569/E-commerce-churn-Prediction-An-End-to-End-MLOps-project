"""
Monitoring metrics: drift detection, performance tracking, and alerting.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from prometheus_client import Counter, Gauge, Histogram
from scipy import stats

from src.config import get_settings
from src.features.feature_validation import population_stability_index, chi_square_drift

logger = logging.getLogger(__name__)

# Prometheus metrics
DRIFT_PSI_GAUGE = Gauge("drift_psi_score", "PSI drift score per feature", ["feature"])
DRIFT_ALERT_COUNTER = Counter("drift_alerts_total", "Total drift alerts triggered")
LATENCY_GAUGE = Gauge("api_latency_ms", "API prediction latency in ms")
THROUGHPUT_GAUGE = Gauge("api_throughput", "Requests per second")
ERROR_RATE_GAUGE = Gauge("api_error_rate", "Error rate (0-1)")
CHURN_RATE_GAUGE = Gauge("predicted_churn_rate", "Predicted churn rate from recent batch")


@dataclass
class DriftResult:
    """Result of drift detection for a single feature."""
    feature: str
    psi: Optional[float] = None
    chi2_p: Optional[float] = None
    drifted: bool = False
    alert: bool = False


@dataclass
class MonitoringReport:
    """Aggregated monitoring report."""
    drift_results: List[DriftResult] = field(default_factory=list)
    predicted_churn_rate: Optional[float] = None
    baseline_churn_rate: Optional[float] = None
    churn_rate_drift: bool = False
    alerts: List[str] = field(default_factory=list)


def compute_psi_batch(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    numerical_columns: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute PSI for all numerical features between baseline and current."""
    if numerical_columns is None:
        numerical_columns = baseline.select_dtypes(include=[np.number]).columns.tolist()
    psi_scores: Dict[str, float] = {}
    for col in numerical_columns:
        if col not in current.columns:
            continue
        try:
            psi = population_stability_index(
                baseline[col].dropna().values,
                current[col].dropna().values,
            )
            psi_scores[col] = psi
            DRIFT_PSI_GAUGE.labels(feature=col).set(psi)
        except Exception as e:
            logger.warning("PSI failed for %s: %s", col, e)
    return psi_scores


def compute_categorical_drift(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    categorical_columns: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Chi-square p-values for categorical drift (low p = drift)."""
    if categorical_columns is None:
        categorical_columns = baseline.select_dtypes(exclude=[np.number]).columns.tolist()
    results: Dict[str, float] = {}
    for col in categorical_columns:
        if col not in current.columns:
            continue
        try:
            p_val = chi_square_drift(baseline[col].dropna().values, current[col].dropna().values)
            results[col] = p_val
        except Exception as e:
            logger.warning("Chi2 drift failed for %s: %s", col, e)
    return results


def detect_drift(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    psi_threshold: Optional[float] = None,
    chi2_p_threshold: float = 0.05,
) -> MonitoringReport:
    """
    Run full drift detection and populate Prometheus gauges.
    """
    settings = get_settings()
    psi_threshold = psi_threshold or settings.monitoring.drift_psi_threshold

    report = MonitoringReport()

    num_cols = baseline.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = baseline.select_dtypes(exclude=[np.number]).columns.tolist()

    for col, psi in compute_psi_batch(baseline, current, num_cols).items():
        drifted = psi > psi_threshold
        alert = drifted
        report.drift_results.append(
            DriftResult(feature=col, psi=psi, drifted=drifted, alert=alert)
        )
        if alert:
            report.alerts.append(f"PSI drift on {col}: {psi:.4f} > {psi_threshold}")
            DRIFT_ALERT_COUNTER.inc()

    for col, p_val in compute_categorical_drift(baseline, current, cat_cols).items():
        drifted = p_val < chi2_p_threshold
        alert = drifted
        report.drift_results.append(
            DriftResult(feature=col, chi2_p=p_val, drifted=drifted, alert=alert)
        )
        if alert:
            report.alerts.append(f"Categorical drift on {col}: p={p_val:.4f}")
            DRIFT_ALERT_COUNTER.inc()

    return report


def check_churn_rate_drift(
    current_churn_rate: float,
    baseline_churn_rate: float,
    threshold: Optional[float] = None,
) -> bool:
    """Check if predicted churn rate has drifted significantly."""
    settings = get_settings()
    threshold = threshold or settings.monitoring.churn_rate_change_threshold
    delta = abs(current_churn_rate - baseline_churn_rate)
    drifted = delta > threshold
    CHURN_RATE_GAUGE.set(current_churn_rate)
    return drifted


def update_latency_metric(latency_ms: float) -> None:
    """Update latency gauge for alerting."""
    LATENCY_GAUGE.set(latency_ms)


def update_error_rate(rate: float) -> None:
    """Update error rate gauge."""
    ERROR_RATE_GAUGE.set(rate)


def update_throughput(rps: float) -> None:
    """Update throughput gauge."""
    THROUGHPUT_GAUGE.set(rps)


def generate_monitoring_report(
    baseline_path: Path,
    current_path: Path,
    output_path: Optional[Path] = None,
) -> MonitoringReport:
    """
    Generate monitoring report comparing baseline to current data.
    """
    baseline = pd.read_parquet(baseline_path) if baseline_path.suffix == ".parquet" else pd.read_csv(baseline_path)
    current = pd.read_parquet(current_path) if current_path.suffix == ".parquet" else pd.read_csv(current_path)

    report = detect_drift(baseline, current)

    if "churned" in baseline.columns and "churned" in current.columns:
        report.baseline_churn_rate = baseline["churned"].mean()
        report.predicted_churn_rate = current["churned"].mean()
        report.churn_rate_drift = check_churn_rate_drift(
            report.predicted_churn_rate, report.baseline_churn_rate
        )
        if report.churn_rate_drift:
            report.alerts.append(
                f"Churn rate drift: {report.predicted_churn_rate:.2%} vs baseline {report.baseline_churn_rate:.2%}"
            )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for a in report.alerts:
                f.write(a + "\n")
            for r in report.drift_results:
                f.write(f"{r.feature}: psi={r.psi}, chi2_p={r.chi2_p}, drifted={r.drifted}\n")

    return report
