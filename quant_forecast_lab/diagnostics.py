"""Training-scope time-series diagnostics; these do not select a model."""

from __future__ import annotations

import inspect
import json
import warnings
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd


def _legacy_result(function, values, **kwargs):
    # Statsmodels 0.15 introduces result objects; retain the documented tuple
    # interface explicitly while still supporting the 0.14 API.
    if "result_object" in inspect.signature(function).parameters:
        kwargs["result_object"] = False
    return function(values, **kwargs)


def analyze_series(series, *, max_analysis_lag: int = 20, alpha: float = 0.05):
    """Return stationarity results and pointwise ACF/PACF estimates.

    The caller must provide only the training period. Missing observations are
    rejected rather than removed, preserving the meaning of each time step.
    """
    from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

    values = np.asarray(series, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Diagnostics require a one-dimensional finite series.")
    if isinstance(max_analysis_lag, bool) or not isinstance(max_analysis_lag, int) or max_analysis_lag < 1:
        raise ValueError("max_analysis_lag must be a positive integer.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one.")
    result = {
        "n_obs": len(values),
        "alpha": alpha,
        "requested_max_lag": max_analysis_lag,
        "interpretation": "Exploratory training diagnostics; no proof of stationarity or forecasting skill.",
        "multiple_comparisons": "Unadjusted pointwise intervals; candidate lags require temporal validation.",
    }
    columns = [
        "kind",
        "lag",
        "estimate",
        "ci_lower",
        "ci_upper",
        "standard_error",
        "normal_critical_value",
        "pointwise_significant",
        "method",
    ]
    if len(values) < 12 or np.ptp(values) == 0:
        result.update(status="not_applicable", reason="At least 12 nonconstant training observations are required.")
        return result, pd.DataFrame(columns=columns)

    nlags = min(max_analysis_lag, len(values) // 2 - 1)
    result.update(status="completed", max_analysis_lag=nlags)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adf = _legacy_result(adfuller, values, regression="c", autolag="AIC")
        kpss_result = _legacy_result(kpss, values, regression="c", nlags="auto")
        acf_values, acf_ci = _legacy_result(acf, values, nlags=nlags, alpha=alpha, bartlett_confint=True)
        pacf_values, pacf_ci = _legacy_result(pacf, values, nlags=nlags, alpha=alpha, method="ywm")
    result["warnings"] = [f"{w.category.__name__}: {w.message}" for w in caught]
    result["adf"] = {
        "null": "unit root",
        "regression": "c",
        "autolag": "AIC",
        "statistic": float(adf[0]),
        "pvalue": float(adf[1]),
        "usedlag": int(adf[2]),
        "nobs": int(adf[3]),
        "critical_values": adf[4],
        "reject_null": bool(adf[1] < alpha),
    }
    result["kpss"] = {
        "null": "level stationary",
        "regression": "c",
        "nlags_policy": "auto",
        "statistic": float(kpss_result[0]),
        "pvalue": float(kpss_result[1]),
        "usedlag": int(kpss_result[2]),
        "critical_values": kpss_result[3],
        "pvalue_note": "May be a table boundary; consult recorded warnings.",
        "reject_null": bool(kpss_result[1] < alpha),
    }
    z = NormalDist().inv_cdf(1 - alpha / 2)
    rows = []
    for kind, estimates, intervals, method in (
        ("acf", acf_values, acf_ci, "Bartlett"),
        ("pacf", pacf_values, pacf_ci, "ywm; asymptotic 1/sqrt(N)"),
    ):
        for lag in range(1, nlags + 1):
            low, high = intervals[lag]
            rows.append(
                {
                    "kind": kind,
                    "lag": lag,
                    "estimate": float(estimates[lag]),
                    "ci_lower": float(low),
                    "ci_upper": float(high),
                    "standard_error": float((high - low) / (2 * z)),
                    "normal_critical_value": z,
                    "pointwise_significant": bool(low > 0 or high < 0),
                    "method": method,
                }
            )
    table = pd.DataFrame(rows, columns=columns)
    result["exploratory_candidate_lags"] = sorted(table.loc[table.pointwise_significant, "lag"].unique().tolist())
    return result, table


def write_training_diagnostics(
    close: pd.Series, output_dir, *, max_analysis_lag=20, alpha=0.05, label="training"
) -> dict:
    """Write mentor-readable TXT, structured results and interval-aware plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (close <= 0).any():
        raise ValueError("Close must be positive for log-return diagnostics.")
    returns = np.log(close).diff().iloc[1:]
    date = datetime.now(UTC).date().isoformat()
    results, tables = {}, []
    for name, series in (("close", close), ("log_return", returns)):
        result, table = analyze_series(series, max_analysis_lag=max_analysis_lag, alpha=alpha)
        result["scope_label"] = label
        result["period"] = {"first": str(series.index[0]), "last": str(series.index[-1])} if len(series) else None
        results[name] = result
        table.insert(0, "series", name)
        tables.append(table)
        if not table.empty:
            fig, axes = plt.subplots(2, 1, figsize=(9, 6), layout="constrained")
            fig.suptitle(label, fontsize=9)
            for kind, axis in zip(("acf", "pacf"), axes, strict=True):
                subset = table[table.kind == kind]
                axis.vlines(subset.lag, 0, subset.estimate)
                # Intervals in the CSV surround the estimate. Shift them to
                # zero for the plot, as Statsmodels plot_acf does.
                axis.fill_between(
                    subset.lag, subset.ci_lower - subset.estimate, subset.ci_upper - subset.estimate, alpha=0.2
                )
                axis.axhline(0, color="black", linewidth=0.7)
                axis.set(title=f"Training {name}: {kind.upper()} (pointwise intervals)", xlabel="Observation lag")
            try:
                fig.savefig(output / f"{name}_acf_pacf.png", dpi=120)
            finally:
                plt.close(fig)
    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(output / "acf_pacf_values.csv", index=False)
    (output / f"z_table_{date}.txt").write_text(
        "Exploratory pointwise correlation intervals, not ADF/KPSS critical values.\n"
        "No multiple-testing adjustment; not a lag selection rule.\n" + combined.to_string(index=False),
        encoding="utf-8",
    )
    encoded = json.dumps(results, indent=2, allow_nan=False)
    for name in ("stationarity_results.json", "stationary_test.txt", f"ACF_PACF_Analysis_{date}.txt"):
        (output / name).write_text(encoded, encoding="utf-8")
    return results
