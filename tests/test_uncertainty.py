import numpy as np
import pytest

from quant_forecast_lab.uncertainty import conformal_radius, symmetric_conformal_interval


def test_conformal_interval_is_symmetric_and_contains_point():
    actual = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    predicted = np.array([10.1, 10.8, 12.2, 12.8, 14.1])
    point = np.array([15.0, 16.0])

    radius = conformal_radius(actual, predicted, coverage=0.8)
    lower, upper = symmetric_conformal_interval(point, actual, predicted, coverage=0.8)

    assert radius >= 0
    assert np.all(lower <= point)
    assert np.all(upper >= point)
    np.testing.assert_allclose(point - lower, upper - point)


def test_conformal_uses_exact_finite_sample_order_statistic():
    # ceil((9 + 1) * .8) = 8, so choose the eighth ordered residual.
    assert conformal_radius(np.arange(1., 10.), np.zeros(9), coverage=.8) == 8.


def test_insufficient_calibration_for_finite_bound_returns_infinity():
    assert np.isinf(conformal_radius([1.], [0.], coverage=.9))


def test_calibration_cannot_silently_discard_nonfinite_residuals():
    with pytest.raises(ValueError, match="finite"):
        conformal_radius([1., np.nan, 2.], [0., 0., 0.])
