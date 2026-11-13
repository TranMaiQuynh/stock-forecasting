import numpy as np
import pytest

from quant_forecast_lab.uncertainty import paired_rmse_skill_interval


def test_identical_forecasts_have_exactly_zero_skill_and_interval():
    y = np.arange(1.0, 41.0)
    prediction = y / 2
    result = paired_rmse_skill_interval(y, prediction, prediction, replicates=100, block_length=5)
    assert (result["rmse_skill"], result["interval_lower"], result["interval_upper"]) == (0, 0, 0)


def test_perfect_predictions_have_unit_skill_and_interval():
    y = np.arange(1.0, 41.0)
    result = paired_rmse_skill_interval(y, y, replicates=100, block_length=5)
    assert (result["rmse_skill"], result["interval_lower"], result["interval_upper"]) == (1, 1, 1)


def test_pairing_reproducibility_and_scale_invariance():
    rng = np.random.default_rng(42)
    y = rng.normal(0, 1, 120)
    p = y + rng.normal(0, 0.5, 120)
    baseline = y + rng.normal(0, 1, 120)
    first = paired_rmse_skill_interval(y, p, baseline, replicates=200, block_length=10)
    assert first == paired_rmse_skill_interval(y, p, baseline, replicates=200, block_length=10)
    scaled = paired_rmse_skill_interval(y * 3, p * 3, baseline * 3, replicates=200, block_length=10)
    for key in ("rmse_skill", "interval_lower", "interval_upper"):
        assert first[key] == pytest.approx(scaled[key])
    assert first["rmse_skill"] == pytest.approx(1 - np.sqrt(np.mean((y - p) ** 2) / np.mean((y - baseline) ** 2)))


@pytest.mark.parametrize("kwargs", [{"block_length": True}, {"block_length": 6}, {"replicates": 10}, {"seed": -1}, {"coverage": 1}])
def test_invalid_resampling_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        paired_rmse_skill_interval([1, 2, 3], [0, 0, 0], **kwargs)


@pytest.mark.parametrize("actual,predicted,baseline", [
    ([0, 0], [0, 0], [0, 0]), ([1, 2], [0], None),
    ([1, np.nan], [0, 0], None), ([[1, 2]], [[0, 0]], None),
])
def test_invalid_or_undefined_paired_losses_are_rejected(actual, predicted, baseline):
    with pytest.raises(ValueError):
        paired_rmse_skill_interval(actual, predicted, baseline, block_length=1)
