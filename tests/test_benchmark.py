import pandas as pd
import pytest

from quant_forecast_lab.benchmark import benchmark_frame


def test_benchmark_ranks_perfect_model_ahead_of_naive_like_model():
    frame = pd.DataFrame(
        {
            "actual": [100.0, 101.0, 99.0, 103.0, 102.0],
            "perfect": [100.0, 101.0, 99.0, 103.0, 102.0],
            "lagged": [100.0, 100.0, 101.0, 99.0, 103.0],
        }
    )
    result = benchmark_frame(
        frame,
        actual_col="actual",
        prediction_cols=["perfect", "lagged"],
    )

    assert result.index[0] == "perfect"
    assert result.loc["perfect", "rmse_skill_vs_naive"] == 1.0
    assert result.loc["lagged", "rmse_skill_vs_naive"] == 0.0


@pytest.mark.parametrize("missing", [float("nan"), float("inf")])
def test_candidate_cannot_omit_difficult_observations(missing):
    frame = pd.DataFrame({"actual": [100., 200., 100.], "origin": [100.] * 3,
                          "complete": [101.] * 3, "missing": [101., missing, 101.]})
    with pytest.raises(ValueError, match="finite"):
        benchmark_frame(frame, actual_col="actual", prediction_cols=["complete", "missing"],
                        previous_actual_col="origin")
