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
