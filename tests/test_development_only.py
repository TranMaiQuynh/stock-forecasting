import json

import numpy as np
import pandas as pd
import pytest

from quant_forecast_lab.artifacts import verify_run
from quant_forecast_lab.research import ResearchConfig, synthetic_close
from quant_forecast_lab.walk_forward import WalkForwardConfig, run_walk_forward_experiment


def test_development_runner_excludes_known_holdout_and_retains_every_candidate(tmp_path):
    frame = synthetic_close().iloc[:130].copy()
    csv = tmp_path / "input.csv"
    frame.to_csv(csv, index=False)
    cutoff = str(frame.Date.iloc[100])
    research = ResearchConfig(asset="DEMO", lookbacks=(1, 3), ridge_alphas=(0.1,), include_baselines=True,
                              penalty_mode="per_sample", refit_policy="monthly", training_window=30, max_analysis_lag=3)
    walk = WalkForwardConfig(min_train_size=40, validation_size=10, test_size=20,
                             final_holdout_start=cutoff, feature_groups=("returns",), development_only=True)
    first = run_walk_forward_experiment(csv, output_root=tmp_path / "a", config=research, walk_config=walk)
    assert verify_run(first)["model_name"] == "walk_forward_return"
    metrics = json.loads((first / "metrics.json").read_text())
    assert metrics["final"]["status"] == "not_scored_development_only"
    assert metrics["actual_validation_candidates"] == 8
    assert metrics["actual_ridge_trials"] == 4
    assert len(pd.read_csv(first / "input_snapshot.csv")) == 100
    assert not (first / "final").exists()
    assert not (first / "final_predictions.csv").exists()
    predictions = pd.read_csv(first / "development_predictions.csv")
    assert (pd.to_datetime(predictions.target_timestamp, utc=True) < pd.Timestamp(cutoff, tz="UTC")).all()
    error = predictions.actual_log_return - predictions.predicted_log_return
    assert metrics["development"]["returns"]["selected_log_return"]["rmse"] == pytest.approx(np.sqrt(np.mean(error**2)))
    # Even an invalid future close cannot enter development validation or fitting.
    frame.loc[100:, "Close"] = -1
    frame.to_csv(csv, index=False)
    second = run_walk_forward_experiment(csv, output_root=tmp_path / "b", config=research, walk_config=walk)
    assert (first / "selection.json").read_bytes() == (second / "selection.json").read_bytes()
    pd.testing.assert_frame_equal(pd.read_csv(first / "development_predictions.csv").drop(columns=["run_id", "child_run"]),
                                  pd.read_csv(second / "development_predictions.csv").drop(columns=["run_id", "child_run"]))


@pytest.mark.parametrize("settings", [{"development_only": 1}, {"development_only": True}, {"final_holdout_start": "NaT"}])
def test_development_cutoff_must_be_explicit_and_valid(settings):
    with pytest.raises(ValueError):
        WalkForwardConfig(**settings).validate()
