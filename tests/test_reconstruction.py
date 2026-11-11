import numpy as np
import pytest

from quant_forecast_lab.reconstruction import (
    causal_trend,
    recursive_residual_prices,
    rolling_return_prices,
)


def test_rolling_forecasts_use_each_observed_origin_instead_of_compounding():
    np.testing.assert_allclose(rolling_return_prices([0, 0], [100, 110]), [100, 110])
    np.testing.assert_allclose(rolling_return_prices(np.log([1.1, 0.9]), [100, 110]), [110, 99])


def test_causal_trend_extrapolates_known_linear_history_and_respects_window():
    assert causal_trend([100], 3) == 100
    assert causal_trend([999, 100, 110, 120], 3) == pytest.approx(130)
    # Reconstructing the next step uses the forecasted close in its own history.
    np.testing.assert_allclose(recursive_residual_prices([1, 2], [100, 110, 120], 3), [131, 143 + 1 / 3])
    np.testing.assert_allclose(recursive_residual_prices([0, 0], [100, 110, 120], 3), [130, 140])


@pytest.mark.parametrize("window", [0, -1, 2.5, True])
def test_invalid_trend_windows_are_rejected(window):
    with pytest.raises(ValueError, match="positive integer"):
        causal_trend([100, 110], window)


@pytest.mark.parametrize("values", [[], [[1, 2]], [np.nan], [np.inf]])
def test_invalid_return_vectors_are_rejected(values):
    with pytest.raises(ValueError):
        rolling_return_prices(values, [100])


def test_reconstruction_rejects_mismatched_origins_and_nonpositive_prices():
    with pytest.raises(ValueError, match="matching"):
        rolling_return_prices([0, 0], [100])
    with pytest.raises(ValueError, match="positive"):
        rolling_return_prices([0], [-100])
    with pytest.raises(ValueError, match="finite and positive"):
        rolling_return_prices([1000], [100])
    with pytest.raises(ValueError, match="finite and positive"):
        recursive_residual_prices([-1000], [100], 3)
