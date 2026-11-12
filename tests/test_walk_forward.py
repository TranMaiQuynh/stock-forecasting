import json
import socket

import numpy as np
import pandas as pd
import pytest

from quant_forecast_lab.experiment import sha256_file
from quant_forecast_lab.research import ResearchConfig, feature_windows, synthetic_close
from quant_forecast_lab.walk_forward import WalkForwardConfig, run_walk_forward_experiment


def test_nested_folds_match_targets_preserve_artifacts_and_isolate_final_holdout(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("Walk-forward attempted network access")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    frame = synthetic_close(42).iloc[:120].copy()
    source = tmp_path / "input.csv"
    frame.to_csv(source, index=False)
    research = ResearchConfig(asset="SYNTHETIC", lookbacks=(1, 3), ridge_alphas=(1.0,), max_analysis_lag=4)
    walk = WalkForwardConfig(
        min_train_size=40, validation_size=10, test_size=20, final_holdout_start=str(frame.Date.iloc[100])
    )
    run = run_walk_forward_experiment(
        source, output_root=tmp_path / "a", config=research, walk_config=walk, data_kind="synthetic"
    )
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert any(name.endswith("/manifest.json") for name in manifest["artifact_sha256"])
    for name, digest in manifest["artifact_sha256"].items():
        assert sha256_file(run / name) == digest
    splits = json.loads((run / "split_manifest.json").read_text())
    assert splits["development_unscored_tail"] == [90, 100]
    assert splits["trial_budget"] == 10
    predictions = pd.read_csv(run / "development_predictions.csv")
    assert not predictions.duplicated(["feature_group", "target_timestamp"]).any()
    for _, fold in predictions.groupby("fold"):
        a = fold[fold.feature_group == "returns"].reset_index(drop=True)
        b = fold[fold.feature_group == "returns_abs"].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            a[["forecast_origin", "target_timestamp"]], b[["forecast_origin", "target_timestamp"]]
        )
    for relative in predictions.child_run.unique():
        child = run / relative
        model = json.loads((child / "artifacts/model.json").read_text())
        split = json.loads((child / "split_manifest.json").read_text())
        snapshot = pd.read_csv(child / "input_snapshot.csv")
        targets = np.arange(max(research.lookbacks) + 1, split["training_series_end_exclusive"])
        x, _ = feature_windows(snapshot.Close, targets, model["lookback"], model["feature_group"])
        np.testing.assert_allclose(model["mean"], x.mean(axis=0))
    result = json.loads((run / "metrics.json").read_text())
    final = run / result["final_child_run"]
    scored = pd.read_csv(run / "final_predictions.csv")
    error = scored.actual_log_return - scored.predicted_log_return
    assert len(scored) == 20
    assert result["final"]["ridge_log_return"]["rmse"] == pytest.approx(np.sqrt(np.mean(error**2)))
    frame.loc[100:, "Close"] *= 1.5
    frame.to_csv(source, index=False)
    changed = run_walk_forward_experiment(
        source, output_root=tmp_path / "b", config=research, walk_config=walk, data_kind="synthetic"
    )
    other = json.loads((changed / "metrics.json").read_text())
    assert (run / "selection.json").read_bytes() == (changed / "selection.json").read_bytes()
    for name in ("artifacts/model.json", "trials.csv", "analysis/acf_pacf_values.csv"):
        assert (final / name).read_bytes() == (changed / other["final_child_run"] / name).read_bytes()


def test_failed_child_retains_failed_parent_and_does_not_select_or_score_final(tmp_path, monkeypatch):
    import quant_forecast_lab.walk_forward as module

    def fail_child(*args, **kwargs):
        raise RuntimeError("deliberate fold failure")

    monkeypatch.setattr(module, "run_offline_experiment", fail_child)
    with pytest.raises(RuntimeError, match="deliberate fold failure"):
        run_walk_forward_experiment(
            output_root=tmp_path, walk_config=WalkForwardConfig(min_train_size=80, validation_size=20, test_size=40)
        )
    (run,) = (tmp_path / "walk_forward_ridge").iterdir()
    assert json.loads((run / "manifest.json").read_text())["status"] == "failed"
    assert not (run / "selection.json").exists()
    assert not (run / "final_predictions.csv").exists()


@pytest.mark.parametrize("value", [True, 0, -1, 2.5])
def test_invalid_fold_sizes_are_rejected(value):
    with pytest.raises(ValueError):
        WalkForwardConfig(test_size=value).validate()


def test_cli_routes_walk_forward_settings_without_changing_the_protocol(tmp_path, monkeypatch):
    import quant_forecast_lab.walk_forward as module
    from quant_forecast_lab.cli import main

    captured = {}

    def run(input_csv, **kwargs):
        captured.update(kwargs)
        assert input_csv is None
        return tmp_path

    monkeypatch.setattr(module, "run_walk_forward_experiment", run)
    assert (
        main(
            [
                "walk-forward",
                "--synthetic",
                "--asset",
                "SMOKE",
                "--min-train-size",
                "40",
                "--validation-size",
                "10",
                "--test-size",
                "20",
                "--final-holdout-start",
                "2023-01-01",
                "--lookbacks",
                "1",
                "3",
                "--feature-groups",
                "returns",
                "--output-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert captured["walk_config"].final_holdout_start == "2023-01-01"
    assert captured["walk_config"].feature_groups == ("returns",)
    assert captured["config"].lookbacks == (1, 3)
