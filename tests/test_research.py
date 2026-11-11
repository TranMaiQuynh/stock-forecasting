import json
import socket

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")
pytest.importorskip("sklearn")
pytest.importorskip("matplotlib")

from quant_forecast_lab.cli import main
from quant_forecast_lab.experiment import sha256_file
from quant_forecast_lab.research import (
    ResearchConfig,
    predict_saved_model,
    return_windows,
    run_offline_experiment,
    synthetic_close,
    validate_close_frame,
)


def test_window_targets_and_origin_history_are_aligned():
    close = np.exp(np.cumsum([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]))
    x, y = return_windows(close, [4, 5], 2)
    np.testing.assert_allclose(x, [[0.3, 0.2], [0.4, 0.3]])
    np.testing.assert_allclose(y, [0.4, 0.5])


def test_offline_run_artifacts_metrics_and_future_isolation(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("Offline experiment attempted network access")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    frame = synthetic_close(42).iloc[:120].copy()
    csv = tmp_path / "input.csv"
    frame.to_csv(csv, index=False)
    config = ResearchConfig(asset="SYNTHETIC", lookbacks=(1, 5), ridge_alphas=(1.0, 10.0), max_analysis_lag=8)
    run = run_offline_experiment(csv, output_root=tmp_path / "log", config=config, data_kind="synthetic")
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    for name, digest in manifest["artifact_sha256"].items():
        assert sha256_file(run / name) == digest
    predictions = pd.read_csv(run / "predictions.csv")
    test = predictions[predictions.split == "test"]
    metrics = json.loads((run / "metrics.json").read_text())
    error = test.actual_log_return - test.predicted_log_return
    assert np.sqrt(np.mean(error**2)) == pytest.approx(metrics["test"]["ridge_log_return"]["rmse"])
    assert metrics["test"]["observations"] == len(test) == 24
    assert (pd.to_datetime(predictions.forecast_origin) < pd.to_datetime(predictions.target_timestamp)).all()
    assert (predictions.horizon == 1).all()
    future = json.loads((run / "future_forecast.json").read_text())
    assert future["actual"] is None and future["target_timestamp"] is None
    assert future["data_kind"] == "synthetic"
    assert np.isfinite(predict_saved_model(run / "artifacts/model.json", frame.Close))
    model_record = json.loads((run / "artifacts/model.json").read_text())
    model_record["feature_order"] = ["wrong_order"]
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(model_record))
    with pytest.raises(ValueError, match="feature order"):
        predict_saved_model(tampered, frame.Close)
    assert (run / "analysis/stationary_test.txt").stat().st_size > 0
    assert list((run / "analysis").glob("ACF_PACF_Analysis_*.txt"))
    assert list((run / "analysis").glob("z_table_*.txt"))
    assert (run / "figures/forecast.png").stat().st_size > 0

    # Alter only final-test closes. Training diagnostics, fitted parameters
    # and validation-based selection must remain unchanged.
    frame.loc[96:, "Close"] *= 1.5
    changed_csv = tmp_path / "changed.csv"
    frame.to_csv(changed_csv, index=False)
    changed = run_offline_experiment(changed_csv, output_root=tmp_path / "log", config=config, data_kind="synthetic")
    for artifact in ("artifacts/model.json", "analysis/acf_pacf_values.csv", "trials.csv"):
        assert (run / artifact).read_bytes() == (changed / artifact).read_bytes()


@pytest.mark.parametrize("defect", ["missing", "duplicate", "unsorted", "zero", "infinite"])
def test_schema_rejects_data_instead_of_repairing_history(defect):
    frame = synthetic_close(1).iloc[:80].copy()
    if defect == "missing":
        frame.loc[5, "Close"] = np.nan
    elif defect == "duplicate":
        frame.loc[5, "Date"] = frame.loc[4, "Date"]
    elif defect == "unsorted":
        frame = frame.iloc[::-1]
    elif defect == "zero":
        frame.loc[5, "Close"] = 0
    else:
        frame.loc[5, "Close"] = np.inf
    with pytest.raises(ValueError):
        validate_close_frame(frame)


def test_invalid_csv_retains_failed_manifest_and_log(tmp_path):
    source = tmp_path / "bad.csv"
    source.write_text("Date,Price\n2020-01-01,100\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Date and Close"):
        run_offline_experiment(source, output_root=tmp_path / "log")
    (run,) = (tmp_path / "log/ridge_log_return").iterdir()
    assert json.loads((run / "manifest.json").read_text())["status"] == "failed"
    assert "ValueError" in (run / "run.log").read_text()


def test_fractional_target_index_is_rejected():
    with pytest.raises(ValueError, match="integer"):
        return_windows(np.arange(1.0, 8.0), [4.5], 2)


def test_failed_trial_is_recorded_and_not_promoted(tmp_path, monkeypatch):
    from sklearn.linear_model import Ridge

    def fail_fit(*args, **kwargs):
        raise RuntimeError("deliberate training failure")

    monkeypatch.setattr(Ridge, "fit", fail_fit)
    with pytest.raises(RuntimeError, match="deliberate training failure"):
        run_offline_experiment(output_root=tmp_path, config=ResearchConfig(lookbacks=(1,), ridge_alphas=(1.0,)))
    (run,) = (tmp_path / "ridge_log_return").iterdir()
    trials = pd.read_csv(run / "trials.csv")
    assert trials.status.tolist() == ["failed"]
    assert json.loads((run / "manifest.json").read_text())["status"] == "failed"
    assert not (run / "metrics.json").exists()


def test_cli_synthetic_smoke(tmp_path, capsys):
    assert (
        main(
            [
                "experiment",
                "--synthetic",
                "--asset",
                "SMOKE",
                "--lookbacks",
                "1",
                "--ridge-alphas",
                "1",
                "--max-analysis-lag",
                "5",
                "--output-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert "Run:" in capsys.readouterr().out
