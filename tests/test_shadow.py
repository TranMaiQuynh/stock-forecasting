import json
import socket

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from quant_forecast_lab.artifacts import verify_run
from quant_forecast_lab.candidate import _refit_recipe, prepare_candidate
from quant_forecast_lab.forecast_ledger import ForecastLedger
from quant_forecast_lab.research import ResearchConfig, feature_windows, predict_saved_model, run_offline_experiment
from quant_forecast_lab.shadow import (
    capture_snapshot,
    create_protocol,
    record_forecast,
    score_snapshot,
    shadow_report,
)


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    root = tmp_path_factory.mktemp("shadow_sources")
    calendar = xcals.get_calendar("XNYS", start="2025-01-01", end="2027-01-01")
    dates = calendar.sessions_in_range("2025-01-02", "2026-11-18")[-321:]
    close = 100 * np.exp(np.cumsum(np.random.default_rng(7).normal(0, 0.01, len(dates))))
    frame = pd.DataFrame({"Date": dates, "Close": close})
    source = root / "source.csv"
    frame.iloc[:160].to_csv(source, index=False)
    model = run_offline_experiment(
        source, output_root=root / "models", data_kind="synthetic",
        config=ResearchConfig(asset="DEMO", lookbacks=(1, 3), ridge_alphas=(0.1,),
                              penalty_mode="per_sample", max_analysis_lag=3),
    )
    snapshots = []
    for stop in (-2, -1, len(frame)):
        part = frame.iloc[:stop]
        path = root / f"observed_{stop}.csv"
        part.to_csv(path, index=False)
        snapshots.append(capture_snapshot(path, asset="DEMO", data_kind="synthetic", output_root=root / "receipts"))
    return root, model, snapshots, frame


def setup_protocol(sources, tmp_path, **kwargs):
    _, model, snapshots, _ = sources
    path = tmp_path / "protocol.json"
    create_protocol(model, path, protocol_id="test-v1", **kwargs)
    return model, snapshots, path, tmp_path / "ledger.sqlite"


def test_complete_replay_is_offline_never_fits_and_retains_predictions(sources, tmp_path, monkeypatch):
    model, snapshots, protocol, database = setup_protocol(sources, tmp_path, max_forecasts=2)

    def forbidden(*args, **kwargs):
        raise AssertionError("Inference/scoring attempted training or network access")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(Ridge, "fit", forbidden)
    monkeypatch.setattr(StandardScaler, "fit", forbidden)
    original, created = record_forecast(model, snapshots[0], protocol, database=database, mode="replay",
                                        as_of="2026-11-16T21:30:00Z", output_root=tmp_path)
    assert created and original["target_session"] == "2026-11-17"
    assert original["availability_evidence"] == "replay_assumption"
    again, created = record_forecast(model, snapshots[0], protocol, database=database, mode="replay",
                                     as_of="2026-11-20T21:30:00Z", output_root=tmp_path)
    assert not created and again == original
    early = score_snapshot(snapshots[1], asset="DEMO", database=database, mode="replay",
                           as_of="2026-11-17T21:10:00Z", output_root=tmp_path)
    assert json.loads((early / "score_receipt.json").read_text())[0]["status"] == "pending_label"
    scored = score_snapshot(snapshots[1], asset="DEMO", database=database, mode="replay",
                            as_of="2026-11-17T21:30:00Z", output_root=tmp_path)
    assert verify_run(scored)["status"] == "completed"
    group = shadow_report(database, mode="replay")["groups"][0]
    assert group["scored"] == 1 and group["pending"] == 0 and group["mode"] == "replay"
    assert shadow_report(database, mode="live")["groups"] == []
    assert group["collection_complete_through_last_origin"]


def test_refit_is_one_pinned_recipe_with_numerical_reload(sources, tmp_path, monkeypatch):
    _, model, snapshots, frame = sources
    import quant_forecast_lab.selection as selection

    def forbidden(*args, **kwargs):
        raise AssertionError("Candidate preparation attempted a new search")

    monkeypatch.setattr(selection, "select_return_model", forbidden)
    candidate = prepare_candidate(model, snapshots[0], mode="replay", as_of="2026-11-16T21:30:00Z", output_root=tmp_path)
    assert verify_run(candidate)["model_name"] == "shadow_model"
    recipe = json.loads((model / "artifacts/model.json").read_text())
    fitted = json.loads((candidate / "artifacts/model.json").read_text())
    history = frame.Close.iloc[:-2].to_numpy()
    targets = np.arange(4, len(history))
    x, y = feature_windows(history, targets, recipe["lookback"])
    scaler = StandardScaler().fit(x)
    expected = Ridge(alpha=len(y) * recipe["penalty"], solver="svd").fit(scaler.transform(x), y)
    assert fitted["coef"] == pytest.approx(expected.coef_)
    assert fitted["ridge_alpha"] == len(y) * recipe["penalty"]
    assert fitted["training_samples"] == len(y)
    assert fitted["fit_cutoff"].startswith("2026-11-16")
    assert np.isfinite(predict_saved_model(candidate / "artifacts/model.json", history))
    protocol = tmp_path / "candidate.json"
    create_protocol(candidate, protocol, protocol_id="candidate-v1", max_forecasts=1)
    forecast, _ = record_forecast(candidate, snapshots[0], protocol, database=tmp_path / "candidate.sqlite", mode="replay",
                                  as_of="2026-11-16T21:30:00Z", output_root=tmp_path)
    assert forecast["model_status"] == "research_candidate_not_promoted"


@pytest.mark.parametrize("defect", ["synthetic_live", "backdate_live", "before_start", "different_model", "changed_data", "changed_protocol"])
def test_invalid_record_does_not_change_existing_forecast(sources, tmp_path, defect):
    model, snapshots, protocol, database = setup_protocol(sources, tmp_path)
    original, _ = record_forecast(model, snapshots[0], protocol, database=database, mode="replay",
                                  as_of="2026-11-16T21:30:00Z", output_root=tmp_path)
    settings = {"database": database, "mode": "replay", "as_of": "2026-11-16T21:30:00Z", "output_root": tmp_path}
    snapshot = snapshots[0]
    rules = json.loads(protocol.read_text())
    if defect == "synthetic_live":
        settings.update(mode="live", as_of=None)
    elif defect == "backdate_live":
        settings["mode"] = "live"
    elif defect == "before_start":
        rules["earliest_origin"] = "2026-11-17"
    elif defect == "different_model":
        rules["model_manifest_sha256"] = "a" * 64
    elif defect == "changed_protocol":
        rules["max_forecasts"] += 1
    else:
        csv = tmp_path / "changed.csv"
        changed = sources[3].iloc[:-2].copy()
        changed.loc[5, "Close"] += 1.0
        changed.to_csv(csv, index=False)
        snapshot = capture_snapshot(csv, asset="DEMO", data_kind="synthetic", output_root=tmp_path)
    protocol.write_text(json.dumps(rules))
    with pytest.raises(ValueError):
        record_forecast(model, snapshot, protocol, **settings)
    stored = ForecastLedger(database).forecasts(mode="replay")
    assert len(stored) == 1 and stored[0]["forecast_id"] == original["forecast_id"]
    assert stored[0]["predicted_log_return"] == original["predicted_log_return"]


def test_fixed_budget_stops_new_predictions_without_retuning(sources, tmp_path):
    model, snapshots, protocol, database = setup_protocol(sources, tmp_path, max_forecasts=1)
    record_forecast(model, snapshots[0], protocol, database=database, mode="replay",
                    as_of="2026-11-16T21:30:00Z", output_root=tmp_path)
    with pytest.raises(ValueError, match="budget reached"):
        record_forecast(model, snapshots[1], protocol, database=database, mode="replay",
                        as_of="2026-11-17T21:30:00Z", output_root=tmp_path)
    assert shadow_report(database, mode="replay")["groups"][0]["decision"] == "budget_closed_waiting_for_labels"


def test_model_and_receipt_directories_cannot_hold_mutable_outputs(sources, tmp_path):
    model, snapshots, protocol, _ = setup_protocol(sources, tmp_path)
    with pytest.raises(ValueError, match="outside"):
        record_forecast(model, snapshots[0], protocol, database=model / "ledger.sqlite", mode="replay",
                        as_of="2026-11-16T21:30:00Z", output_root=tmp_path)
    with pytest.raises(FileExistsError):
        create_protocol(model, protocol, protocol_id="overwrite")


@pytest.mark.parametrize("kind", ["zero", "training_mean"])
def test_refit_baselines_respect_latest_eligible_training_window(sources, kind):
    _, source, _, frame = sources
    recipe = json.loads((source / "artifacts/model.json").read_text())
    recipe["estimator_kind"] = kind
    close = frame.iloc[:-2].set_index("Date")
    fitted = _refit_recipe(recipe, {"lookbacks": [1, 3], "training_window": 12}, close)
    expected = float(np.diff(np.log(close.Close.to_numpy()))[-12:].mean()) if kind == "training_mean" else 0.0
    assert fitted["training_samples"] == 12 and fitted["intercept"] == pytest.approx(expected)
    assert all(value == 0 for value in fitted["coef"])


def test_failed_fit_is_retained_and_has_no_completed_model(sources, tmp_path, monkeypatch):
    _, model, snapshots, _ = sources

    def fail(*args, **kwargs):
        raise RuntimeError("simulated solver failure")

    monkeypatch.setattr(Ridge, "fit", fail)
    with pytest.raises(RuntimeError, match="solver failure"):
        prepare_candidate(model, snapshots[0], mode="replay", as_of="2026-11-16T21:30:00Z", output_root=tmp_path)
    (run,) = (tmp_path / "shadow_model").iterdir()
    assert json.loads((run / "manifest.json").read_text())["status"] == "failed"
    assert not (run / "artifacts/model.json").exists()


def test_budget_decision_keeps_baseline_when_candidate_has_no_gain(sources, tmp_path):
    model, snapshots, protocol, database = setup_protocol(sources, tmp_path, max_forecasts=1)
    original, _ = record_forecast(model, snapshots[0], protocol, database=database, mode="replay",
                                  as_of="2026-11-16T21:30:00Z", output_root=tmp_path)
    # A controlled zero forecast must have exactly the same error as zero baseline.
    controlled = ForecastLedger(tmp_path / "controlled.sqlite")
    zero, _ = controlled.record({**original, "predicted_log_return": 0.0})
    controlled.score(zero["forecast_id"], actual_close=zero["origin_close"] * 1.01,
                     received_at="2026-11-17T21:25:00Z", scored_as_of="2026-11-17T21:30:00Z", input_sha256="c" * 64)
    group = shadow_report(controlled.path, mode="replay")["groups"][0]
    assert group["rmse_skill_vs_zero"] == 0
    assert group["decision"] == "stop_candidate_retain_baseline" and group["promotion"] == "disabled"


def test_missing_origin_is_visible_instead_of_selecting_only_easy_days(sources, tmp_path):
    model, snapshots, protocol, database = setup_protocol(sources, tmp_path, earliest_origin="2026-11-13")
    record_forecast(model, snapshots[0], protocol, database=database, mode="replay",
                    as_of="2026-11-16T21:30:00Z", output_root=tmp_path)
    group = shadow_report(database, mode="replay")["groups"][0]
    assert group["missing_origins"] == ["2026-11-13"]
    assert not group["collection_complete_through_last_origin"]


def test_cli_capture_init_record_score_and_report(sources, tmp_path, capsys):
    from quant_forecast_lab.cli import main

    root, model, snapshots, _ = sources
    assert main(["shadow", "capture", "--csv", str(root / "observed_-2.csv"), "--asset", "DEMO",
                 "--data-kind", "synthetic", "--output-root", str(tmp_path)]) == 0
    captured = json.loads(capsys.readouterr().out)["snapshot_run"]
    protocol, database = tmp_path / "cli.json", tmp_path / "cli.sqlite"
    assert main(["shadow", "init", "--run-dir", str(model), "--protocol-id", "cli-v1", "--output", str(protocol)]) == 0
    capsys.readouterr()
    assert main(["shadow", "record", "--run-dir", str(model), "--snapshot-run", captured,
                 "--protocol", str(protocol), "--ledger", str(database), "--mode", "replay",
                 "--as-of", "2026-11-16T21:30:00Z", "--output-root", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["created"]
    assert main(["shadow", "score", "--snapshot-run", str(snapshots[1]), "--asset", "DEMO",
                 "--ledger", str(database), "--mode", "replay", "--as-of", "2026-11-17T21:30:00Z",
                 "--output-root", str(tmp_path)]) == 0
    capsys.readouterr()
    assert main(["shadow", "report", "--ledger", str(database), "--mode", "replay"]) == 0
    assert json.loads(capsys.readouterr().out)["groups"][0]["scored"] == 1
