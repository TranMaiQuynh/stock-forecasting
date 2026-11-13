import json

import numpy as np
import pandas as pd
import pytest

from quant_forecast_lab.forecast import run_saved_forecast
from quant_forecast_lab.research import ResearchConfig, run_offline_experiment


def test_saved_baseline_predicts_and_rejects_an_origin_before_its_refit(tmp_path):
    frame = pd.DataFrame({"Date": pd.bdate_range("2020-01-01", periods=160),
                          "Close": 100 * np.exp(np.arange(160) * 0.001)})
    source = tmp_path / "input.csv"
    frame.to_csv(source, index=False)
    config = ResearchConfig(asset="DEMO", lookbacks=(1,), ridge_alphas=(0.1,), include_baselines=True,
                            penalty_mode="per_sample", refit_policy="monthly", max_analysis_lag=3)
    model = run_offline_experiment(source, output_root=tmp_path / "models", config=config, split_boundaries=(60, 95, 160))
    predicted = run_saved_forecast(model, source, asset="DEMO", output_root=tmp_path / "forecasts")
    output = json.loads((predicted / "forecast.json").read_text())
    assert output["estimator_kind"] == "training_mean"
    assert output["predicted_log_return"] == pytest.approx(0.001)
    assert output["actual"] is None and output["target_timestamp"] is None
    # This origin follows selection but precedes the exported model's last fit.
    old_source = tmp_path / "old.csv"
    frame.iloc[:100].to_csv(old_source, index=False)
    with pytest.raises(ValueError, match="selection or fit cutoff"):
        run_saved_forecast(model, old_source, asset="DEMO", output_root=tmp_path / "rejected")
