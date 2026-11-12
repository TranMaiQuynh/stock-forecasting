import json
import shutil
import socket

import numpy as np
import pandas as pd
import pytest

from quant_forecast_lab.artifacts import verify_run
from quant_forecast_lab.experiment import sha256_file
from quant_forecast_lab.forecast import run_saved_forecast
from quant_forecast_lab.research import ResearchConfig, predict_saved_model, run_offline_experiment, synthetic_close


@pytest.fixture(scope="module")
def saved_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("saved_forecast")
    source = root / "history.csv"
    synthetic_close(42).to_csv(source, index=False)
    run = run_offline_experiment(
        source,
        output_root=root,
        data_kind="synthetic",
        config=ResearchConfig(asset="DEMO", lookbacks=(1,), ridge_alphas=(1.0,), max_analysis_lag=4),
    )
    return run, source


def test_saved_forecast_is_offline_preserves_model_and_does_not_fit(saved_run, tmp_path, monkeypatch):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    model, source = saved_run
    before = {p.relative_to(model).as_posix(): sha256_file(p) for p in model.rglob("*") if p.is_file()}

    def forbidden(*args, **kwargs):
        raise AssertionError("Inference attempted network access or model fitting")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(Ridge, "fit", forbidden)
    monkeypatch.setattr(StandardScaler, "fit", forbidden)
    run = run_saved_forecast(model, source, asset="DEMO", output_root=tmp_path)
    result = json.loads((run / "forecast.json").read_text())
    close = pd.read_csv(source).Close
    assert result["predicted_log_return"] == pytest.approx(predict_saved_model(model / "artifacts/model.json", close))
    assert result["predicted_close"] == pytest.approx(close.iloc[-1] * np.exp(result["predicted_log_return"]))
    assert result["baseline_close"] == close.iloc[-1]
    assert result["actual"] is None and result["target_timestamp"] is None
    assert result["model_status"] == "research_candidate_not_promoted"
    assert result["data_kind"] == "user_supplied" and result["model_data_kind"] == "synthetic"
    assert verify_run(run)["status"] == "completed"
    assert before == {p.relative_to(model).as_posix(): sha256_file(p) for p in model.rglob("*") if p.is_file()}


@pytest.mark.parametrize("defect", ["asset", "past_origin", "corrupt_model", "duplicate_dates"])
def test_invalid_forecast_retains_failed_run_without_retraining(saved_run, tmp_path, defect):
    original, input_csv = saved_run
    model = tmp_path / "model"
    shutil.copytree(original, model)
    frame = pd.read_csv(input_csv)
    if defect == "past_origin":
        frame = frame.iloc[:80]
    elif defect == "duplicate_dates":
        frame.loc[1, "Date"] = frame.loc[0, "Date"]
    elif defect == "corrupt_model":
        (model / "artifacts/model.json").write_text("changed", encoding="utf-8")
    source = tmp_path / "input.csv"
    frame.to_csv(source, index=False)
    with pytest.raises(ValueError):
        run_saved_forecast(
            model, source, asset="WRONG" if defect == "asset" else "DEMO", output_root=tmp_path / "outputs"
        )
    (run,) = (tmp_path / "outputs/saved_ridge_forecast").iterdir()
    assert json.loads((run / "manifest.json").read_text())["status"] == "failed"
    assert not (run / "forecast.json").exists()


def test_output_cannot_modify_source_run(saved_run):
    model, source = saved_run
    with pytest.raises(ValueError, match="outside"):
        run_saved_forecast(model, source, asset="DEMO", output_root=model)


def test_prediction_and_verification_cli(saved_run, tmp_path, capsys):
    from quant_forecast_lab.cli import main

    model, source = saved_run
    assert main(["verify-run", "--run-dir", str(model)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    assert (
        main(
            [
                "predict",
                "--run-dir",
                str(model),
                "--csv",
                str(source),
                "--asset",
                "DEMO",
                "--output-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["asset"] == "DEMO" and output["actual"] is None
    assert output["run_directory"]
