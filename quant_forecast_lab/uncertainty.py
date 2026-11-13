"""Distribution-free uncertainty helpers for point forecasts."""

from __future__ import annotations

import math

import numpy as np


def conformal_radius(calibration_actual, calibration_predicted, coverage: float = 0.90) -> float:
    """Return the ceil((n+1)*coverage)-th residual, or infinity if rank > n.

    Coverage guarantees require exchangeable calibration/test scores. They do
    not automatically hold for dependent or shifting financial time series.
    """
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be strictly between 0 and 1.")

    actual = np.asarray(calibration_actual, dtype=float).reshape(-1)
    predicted = np.asarray(calibration_predicted, dtype=float).reshape(-1)
    if len(actual) != len(predicted):
        raise ValueError("Calibration arrays must have the same length.")

    residuals = np.abs(actual - predicted)
    if not np.isfinite(residuals).all():
        raise ValueError("Calibration residuals must all be finite.")
    if len(residuals) == 0:
        raise ValueError("No finite calibration residuals available.")

    rank = math.ceil((len(residuals) + 1) * coverage)
    if rank > len(residuals):
        return float("inf")
    return float(np.partition(residuals, rank - 1)[rank - 1])


def symmetric_conformal_interval(
    point_forecast,
    calibration_actual,
    calibration_predicted,
    coverage: float = 0.90,
):
    """Construct symmetric intervals around one or more point forecasts."""
    radius = conformal_radius(calibration_actual, calibration_predicted, coverage)
    point = np.asarray(point_forecast, dtype=float)
    return point - radius, point + radius


def paired_rmse_skill_interval(
    actual, predicted, baseline=None, *, block_length=20, replicates=2000, coverage=0.95, seed=42
):
    """Exploratory paired circular-block interval for relative RMSE skill.

    Identical resampled time indices are used for all three aligned series.
    Financial dependence and regime shifts can invalidate the approximation;
    this is not a post-selection or multiple-comparison correction.
    """
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    baseline = np.zeros_like(actual) if baseline is None else np.asarray(baseline, dtype=float)
    if any(x.ndim != 1 for x in (actual, predicted, baseline)):
        raise ValueError("Paired series must be one-dimensional.")
    if len(actual) < 2 or not (actual.shape == predicted.shape == baseline.shape):
        raise ValueError("Paired series must have the same length of at least two.")
    if not all(np.isfinite(x).all() for x in (actual, predicted, baseline)):
        raise ValueError("Paired series must be finite; observations cannot be dropped.")
    if isinstance(block_length, bool) or not isinstance(block_length, int) or not 1 <= block_length <= len(actual):
        raise ValueError("block_length must be an integer between 1 and the sample count.")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or not 100 <= replicates <= 10000:
        raise ValueError("replicates must be an integer between 100 and 10000.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or not 0 < coverage < 1:
        raise ValueError("A nonnegative integer seed and coverage between 0 and 1 are required.")
    loss = (actual - predicted) ** 2
    reference = (actual - baseline) ** 2
    if not np.isfinite(loss).all() or not np.isfinite(reference).all() or reference.mean() <= 0:
        raise ValueError("Squared losses must be finite and baseline error must be positive.")
    rng = np.random.default_rng(seed)
    n = len(actual)
    samples = []
    for first in range(0, replicates, 128):
        batch = min(128, replicates - first)
        starts = rng.integers(0, n, size=(batch, math.ceil(n / block_length)))
        indices = ((starts[:, :, None] + np.arange(block_length)) % n).reshape(batch, -1)[:, :n]
        denominators = reference[indices].mean(axis=1)
        if (denominators <= 0).any():
            raise ValueError("A resampled baseline error is zero; relative skill interval is undefined.")
        samples.extend(1 - np.sqrt(loss[indices].mean(axis=1) / denominators))
    tail = (1 - coverage) / 2
    lower, upper = np.quantile(samples, [tail, 1 - tail])
    return {
        "rmse_skill": float(1 - np.sqrt(loss.mean() / reference.mean())),
        "interval_lower": float(lower), "interval_upper": float(upper),
        "coverage": coverage, "observations": n, "block_length": block_length,
        "replicates": replicates, "seed": seed,
        "method": "paired circular moving-block bootstrap",
        "interpretation": "exploratory marginal interval; dependence approximation; no selection or multiplicity correction",
    }
