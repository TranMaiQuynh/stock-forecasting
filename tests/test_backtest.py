import numpy as np
import pytest

from quant_forecast_lab.backtest import backtest_directional_strategy


def test_transaction_costs_reduce_directional_strategy_return():
    actual = np.array([0.01, -0.01, 0.02, -0.02, 0.01])
    predicted = np.array([0.01, -0.01, 0.01, -0.01, 0.01])

    free = backtest_directional_strategy(actual, predicted, transaction_cost_bps=0)
    costly = backtest_directional_strategy(actual, predicted, transaction_cost_bps=10)

    assert costly.annualized_return < free.annualized_return
    assert costly.average_turnover == free.average_turnover


def test_drawdown_includes_loss_from_initial_capital():
    result = backtest_directional_strategy([-0.1, 0.0], [1.0, 1.0])
    assert result.max_drawdown == pytest.approx(-0.1)


@pytest.mark.parametrize("actual,predicted", [([0.1, np.nan, 0.2], [1, 1, 1]), ([0.1, 0.2, 0.3], [1, np.inf, 1])])
def test_nonfinite_returns_do_not_silently_remove_periods(actual, predicted):
    with pytest.raises(ValueError, match="finite"):
        backtest_directional_strategy(actual, predicted)


@pytest.mark.parametrize("cost", [np.nan, np.inf])
def test_cost_must_be_finite(cost):
    with pytest.raises(ValueError, match="finite"):
        backtest_directional_strategy([0.1, 0.2], [1, 1], transaction_cost_bps=cost)


def test_nonpositive_equity_is_rejected():
    with pytest.raises(ValueError, match="capital"):
        backtest_directional_strategy([1.5, 0.0], [-1, 1])


def test_zero_volatility_has_undefined_sharpe():
    result = backtest_directional_strategy([0., 0.], [1., 1.])
    assert result.sharpe is None
