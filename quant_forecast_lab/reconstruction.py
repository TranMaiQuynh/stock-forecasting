"""Causal inverse transforms shared by historical and recursive forecasts."""

from __future__ import annotations

import numpy as np


def _finite_vector(values, name):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite one-dimensional vector.")
    return array


def rolling_return_prices(log_returns, origin_prices):
    """Invert independent one-step returns using each observed origin price."""
    returns = _finite_vector(log_returns, "log_returns")
    origins = _finite_vector(origin_prices, "origin_prices")
    if returns.shape != origins.shape or (origins <= 0).any():
        raise ValueError("Each return needs a matching positive observed origin price.")
    with np.errstate(over="ignore", under="ignore"):
        prices = origins * np.exp(returns)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Reconstructed prices must be finite and positive.")
    return prices


def causal_trend(history, window):
    """Extrapolate one price step from at most window already-observed prices.

    A single observation uses persistence, matching legacy residual targets.
    The target price itself must never be included in history.
    """
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer.")
    values = _finite_vector(history, "history")[-window:]
    if len(values) == 1:
        return float(values[0])
    slope, intercept = np.polyfit(np.arange(len(values)), values, 1)
    estimate = float(slope * len(values) + intercept)
    if not np.isfinite(estimate):
        raise ValueError("Trend extrapolation must be finite.")
    return estimate


def recursive_residual_prices(residuals, close_history, window):
    """Invert residuals, recomputing the trend from observed/generated history."""
    values = _finite_vector(residuals, "residuals")
    history = _finite_vector(close_history, "close_history").tolist()
    prices = []
    for residual in values:
        price = causal_trend(history, window) + float(residual)
        if not np.isfinite(price) or price <= 0:
            raise ValueError("Reconstructed prices must be finite and positive.")
        prices.append(price)
        history.append(price)
    return np.asarray(prices)
