import json

import numpy as np
import pandas as pd
import pytest

from quant_forecast_lab.artifacts import verify_run
from quant_forecast_lab.research import (
    ResearchConfig,
    feature_windows,
    predict_saved_model,
    run_offline_experiment,
    synthetic_close,
)


def test_default_predictions_still_match_the_original_ridge_pipeline(tmp_path):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    frame = synthetic_close().iloc[:120]
    source = tmp_path / "input.csv"
    frame.to_csv(source, index=False)
    config = ResearchConfig(lookbacks=(1, 3), ridge_alphas=(1.0, 10.0), max_analysis_lag=4)
    run = run_offline_experiment(source, output_root=tmp_path / "runs", config=config)
    targets = np.arange(4, 120)
    best = None
    for lookback in config.lookbacks:
        x, y = feature_windows(frame.Close, targets, lookback)
        for alpha in config.ridge_alphas:
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="svd"))
            model.fit(x[targets < 72], y[targets < 72])
            valid = (targets >= 72) & (targets < 96)
            error = np.sqrt(np.mean((model.predict(x[valid]) - y[valid]) ** 2))
            if best is None or error < best[0]:
                best = error, model.predict(x[targets >= 96])
    predicted = pd.read_csv(run / "predictions.csv")
    np.testing.assert_allclose(predicted[predicted.split == "test"].predicted_log_return, best[1], atol=1e-14)


def test_monthly_pipeline_uses_and_verifies_each_origin_checkpoint(tmp_path):
    frame = synthetic_close().iloc[:160]
    source = tmp_path / "input.csv"
    frame.to_csv(source, index=False)
    config = ResearchConfig(asset="DEMO", lookbacks=(1, 3), ridge_alphas=(0.1,), include_baselines=True,
                            penalty_mode="per_sample", refit_policy="monthly", training_window=35, max_analysis_lag=4)
    run = run_offline_experiment(source, output_root=tmp_path / "runs", config=config, split_boundaries=(60, 95, 160))
    assert verify_run(run)["model_name"] == "return_forecast"
    predictions = pd.read_csv(run / "predictions.csv")
    for _, row in predictions.iterrows():
        target = pd.to_datetime(frame.Date, utc=True).searchsorted(pd.Timestamp(row.target_timestamp))
        assert predict_saved_model(run / row.model_artifact, frame.Close.iloc[:target]) == pytest.approx(row.predicted_log_return)
    test = predictions[predictions.split == "test"]
    metrics = json.loads((run / "metrics.json").read_text())["test"]
    assert metrics["selected_log_return"]["rmse"] == pytest.approx(np.sqrt(np.mean((test.actual_log_return - test.predicted_log_return) ** 2)))
    assert metrics["training_mean"]["rmse"] == pytest.approx(np.sqrt(np.mean((test.actual_log_return - test.training_mean_log_return) ** 2)))
    assert metrics["always_up_directional_accuracy"] == pytest.approx(np.mean(test.actual_log_return > 0))
    split = json.loads((run / "split_manifest.json").read_text())
    model = json.loads((run / "artifacts/model.json").read_text())
    assert split["model_fit_cutoff"] == model["fit_cutoff"]
    assert split["refit"] is True


@pytest.mark.parametrize("settings", [
    {"include_baselines": 1}, {"penalty_mode": "bad"}, {"refit_policy": "daily"},
    {"training_window": True}, {"training_window": 5}, {"training_window": 12.5},
])
def test_invalid_new_policy_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        ResearchConfig(**settings).validate()


def test_cli_routes_explicit_selection_and_refit_settings(tmp_path, monkeypatch):
    import quant_forecast_lab.walk_forward as module
    from quant_forecast_lab.cli import main

    captured = {}

    def capture(*args, **kwargs):
        captured.update(kwargs)
        return tmp_path

    monkeypatch.setattr(module, "run_walk_forward_experiment", capture)
    assert main([
        "walk-forward", "--synthetic", "--include-baselines", "--penalty-mode", "per_sample",
        "--ridge-alphas", "0.1", "10", "--refit-policy", "monthly", "--training-window", "756",
        "--development-only", "--final-holdout-start", "2023-01-01",
    ]) == 0
    assert captured["config"].include_baselines
    assert captured["config"].ridge_alphas == (0.1, 10.0)
    assert captured["config"].penalty_mode == "per_sample"
    assert captured["config"].refit_policy == "monthly"
    assert captured["config"].training_window == 756
    assert captured["walk_config"].development_only
