import numpy as np
import pytest

pytest.importorskip("statsmodels")

from quant_forecast_lab.diagnostics import analyze_series


def test_intervals_are_centered_on_estimate_and_include_negative_signals():
    rng = np.random.default_rng(42)
    series = np.zeros(500)
    for t in range(1, len(series)):
        series[t] = -0.8 * series[t - 1] + rng.normal()
    result, table = analyze_series(series, max_analysis_lag=8)
    row = table[(table.kind == "acf") & (table.lag == 1)].iloc[0]
    assert row.estimate < -0.5
    assert row.pointwise_significant
    assert row.ci_upper < 0
    assert 1 in result["exploratory_candidate_lags"]
    assert result["adf"]["null"] == "unit root"
    assert result["kpss"]["null"] == "level stationary"
    assert (table.lag > 0).all()


@pytest.mark.parametrize("series", [np.ones(30), np.arange(4.0)])
def test_inapplicable_series_does_not_fabricate_test_success(series):
    result, table = analyze_series(series)
    assert result["status"] == "not_applicable"
    assert "adf" not in result
    assert table.empty


def test_nonfinite_series_is_not_compressed_before_lag_analysis():
    with pytest.raises(ValueError, match="finite"):
        analyze_series([1, 2, np.nan, 4])


def test_analysis_lag_is_capped_for_sample_size():
    result, table = analyze_series(np.random.default_rng(5).normal(size=20), max_analysis_lag=100)
    assert result["max_analysis_lag"] == 9
    assert table.lag.max() == 9
