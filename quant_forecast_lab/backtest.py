"""Small explicit backtesting helpers for forecast diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class StrategyMetrics:
    annualized_return: float
    annualized_volatility: float
    sharpe: float | None
    max_drawdown: float
    average_turnover: float

    def as_dict(self) -> dict[str, float | None]:
        return asdict(self)


def backtest_directional_strategy(
    actual_returns,
    predicted_returns,
    *,
    transaction_cost_bps: float = 0.0,
    periods_per_year: int = 252,
) -> StrategyMetrics:
    """One-period aligned simple-return diagnostic, starting with unit capital.

    Forecasts must be available before their corresponding realized return.
    No order book, financing, liquidation or terminal exit is simulated.
    Annualized return is the arithmetic mean times periods_per_year, not CAGR.
    """
    if not np.isfinite(transaction_cost_bps) or transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be finite and nonnegative.")
    if isinstance(periods_per_year, bool) or not isinstance(periods_per_year, (int, np.integer)) or periods_per_year < 1:
        raise ValueError("periods_per_year must be a positive integer.")
    actual = np.asarray(actual_returns, dtype=float).reshape(-1)
    predicted = np.asarray(predicted_returns, dtype=float).reshape(-1)
    if len(actual) != len(predicted):
        raise ValueError("Return arrays must have the same length.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Return arrays must contain only finite observations; periods cannot be dropped.")
    if len(actual) < 2:
        raise ValueError("At least two finite observations are required.")

    position = np.sign(predicted)
    previous = np.concatenate(([0.0], position[:-1]))
    turnover = np.abs(position - previous)
    costs = turnover * (float(transaction_cost_bps) / 10_000.0)
    strategy_returns = position * actual - costs
    if (actual < -1.0).any() or (strategy_returns <= -1.0).any():
        raise ValueError("Returns exhaust capital or violate the simple-return contract.")

    mean_return = float(np.mean(strategy_returns))
    volatility = float(np.std(strategy_returns, ddof=1))
    annualized_return = mean_return * periods_per_year
    annualized_volatility = volatility * np.sqrt(periods_per_year)
    sharpe = annualized_return / annualized_volatility if annualized_volatility else None

    equity = np.concatenate(([1.0], np.cumprod(1.0 + strategy_returns)))
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0

    return StrategyMetrics(
        annualized_return=float(annualized_return),
        annualized_volatility=float(annualized_volatility),
        sharpe=float(sharpe) if sharpe is not None else None,
        max_drawdown=float(np.min(drawdown)),
        average_turnover=float(np.mean(turnover)),
    )
