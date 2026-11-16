import json
from dataclasses import replace

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from quant_forecast_lab.artifacts import verify_run
from quant_forecast_lab.batch_config import current_slot, load_config, parse_config
from quant_forecast_lab.daily import BatchFailure, run_daily
from quant_forecast_lab.forecast_ledger import ForecastLedger
from quant_forecast_lab.market_data import MarketDataUnavailable
from quant_forecast_lab.operations import OperationsStore
from quant_forecast_lab.research import ResearchConfig, run_offline_experiment
from quant_forecast_lab.shadow import capture_snapshot, create_protocol


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    root = tmp_path_factory.mktemp("batch_sources")
    calendar = xcals.get_calendar("XNYS", start="2025-01-01", end="2027-01-01")
    dates = calendar.sessions_in_range("2025-01-02", "2026-11-18")[-321:]
    streams, snapshots = [], [{}, {}, {}]
    for seed, asset in enumerate(("AAA", "BBB")):
        frame = pd.DataFrame({"Date": dates, "Close": 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.01, len(dates))))})
        training = root / f"{asset}.csv"
        frame.iloc[:160].to_csv(training, index=False)
        model = run_offline_experiment(training, output_root=root, data_kind="synthetic",
                                       config=ResearchConfig(asset=asset, lookbacks=(1,), ridge_alphas=(0.1,), max_analysis_lag=3))
        protocol = root / f"{asset}.json"
        create_protocol(model, protocol, protocol_id=f"{asset}-fixed-v1", max_forecasts=2)
        streams.append({"asset": asset, "model_run": str(model), "protocol_file": str(protocol)})
        for i, stop in enumerate((len(frame) - 2, len(frame) - 1, len(frame))):
            path = root / f"{asset}_{stop}.csv"
            frame.iloc[:stop].to_csv(path, index=False)
            snapshots[i][asset] = capture_snapshot(path, asset=asset, data_kind="synthetic", output_root=root)
    return streams, snapshots


def configure(sources, tmp_path, **kwargs):
    streams, snapshots = sources
    raw = {"schema_version": 1, "deployment_id": "batch-v1", "mode": "replay",
           "database": "ledger.sqlite", "state_database": "ops.sqlite", "output_root": "runs",
           "streams": streams, "retry_delay_seconds": 0, **kwargs}
    config = parse_config(raw, base=tmp_path)
    state = OperationsStore(config.state_database)
    state.register(config.definition())
    state.activate(config.deployment_id, now="2026-11-16T21:35:00Z", reason="test")
    return config, snapshots, state


def test_three_days_complete_last_label_after_collection_budget_and_never_train(sources, tmp_path, monkeypatch):
    from sklearn.linear_model import Ridge

    def forbidden(*args, **kwargs):
        raise AssertionError("Daily inference attempted training")

    config, snapshots, state = configure(sources, tmp_path)
    monkeypatch.setattr(Ridge, "fit", forbidden)
    first = run_daily(config, as_of="2026-11-16T21:35:00Z", snapshots=snapshots[0])
    assert verify_run(first)["status"] == "completed"
    repeated = run_daily(config, as_of="2026-11-16T22:35:00Z", snapshots=snapshots[0])
    assert all(r["status"] == "already_completed" for r in json.loads((repeated / "summary.json").read_text())["assets"])
    run_daily(config, as_of="2026-11-17T21:35:00Z", snapshots=snapshots[1])
    final = run_daily(config, as_of="2026-11-18T21:35:00Z", snapshots=snapshots[2])
    assert all(r["slot"]["status"] == "collection_closed" for r in json.loads((final / "summary.json").read_text())["assets"])
    rows = ForecastLedger(config.database).forecasts(mode="replay")
    assert len(rows) == 4 and all(row["label"] is not None for row in rows)
    assert len(state.jobs(config.deployment_id)) == 6


def test_retry_after_forecast_commit_reuses_original_snapshot_without_duplicate(sources, tmp_path, monkeypatch):
    import quant_forecast_lab.daily as daily

    config, snapshots, state = configure(sources, tmp_path)
    actual = daily.record_forecast

    def interrupted(*args, **kwargs):
        actual(*args, **kwargs)
        raise RuntimeError("failure after business commit")

    monkeypatch.setattr(daily, "record_forecast", interrupted)
    with pytest.raises(BatchFailure) as failure:
        run_daily(config, as_of="2026-11-16T21:35:00Z", snapshots=snapshots[0])
    assert json.loads((failure.value.run / "manifest.json").read_text())["status"] == "failed"
    assert len(ForecastLedger(config.database).forecasts(mode="replay")) == 2
    monkeypatch.setattr(daily, "record_forecast", actual)
    # Different supplied snapshots must not replace the durable first inputs.
    resumed = run_daily(config, as_of="2026-11-16T21:40:00Z", snapshots=snapshots[1])
    results = json.loads((resumed / "summary.json").read_text())["assets"]
    assert all(not row["stages"]["record"]["created"] for row in results)
    assert len(ForecastLedger(config.database).forecasts(mode="replay")) == 2
    assert all(j["claims"] == 2 for j in state.jobs(config.deployment_id))


def test_scoring_failure_is_reported_and_does_not_skip_either_assets_forecast(sources, tmp_path, monkeypatch):
    import quant_forecast_lab.daily as daily

    config, snapshots, _ = configure(sources, tmp_path)
    actual = daily.score_snapshot

    def fail_one(*args, **kwargs):
        if kwargs["asset"] == "AAA":
            raise ValueError("revised price basis")
        return actual(*args, **kwargs)

    monkeypatch.setattr(daily, "score_snapshot", fail_one)
    with pytest.raises(BatchFailure) as failure:
        run_daily(config, as_of="2026-11-16T21:35:00Z", snapshots=snapshots[0])
    assets = failure.value.summary["assets"]
    assert assets[0]["stages"]["score"]["status"] == "failed"
    assert assets[0]["stages"]["record"]["status"] == "succeeded"
    assert assets[1]["status"] == "succeeded"
    assert len(ForecastLedger(config.database).forecasts(mode="replay")) == 2


@pytest.mark.parametrize("transient,expected_calls", [(True, 2), (False, 1)])
def test_only_transient_acquisition_failures_are_retried(sources, tmp_path, monkeypatch, transient, expected_calls):
    import quant_forecast_lab.daily as daily

    config, snapshots, _ = configure(sources, tmp_path)
    actual, calls = daily.load_snapshot, []

    def flaky(path, asset):
        if asset == "AAA":
            calls.append(asset)
            if len(calls) == 1:
                raise MarketDataUnavailable("provider timeout") if transient else ValueError("invalid schema")
        return actual(path, asset)

    monkeypatch.setattr(daily, "load_snapshot", flaky)
    if transient:
        result = run_daily(config, as_of="2026-11-16T21:35:00Z", snapshots=snapshots[0])
        row = json.loads((result / "summary.json").read_text())["assets"][0]
        assert row["stages"]["acquire"]["attempts"] == 2
    else:
        with pytest.raises(BatchFailure):
            run_daily(config, as_of="2026-11-16T21:35:00Z", snapshots=snapshots[0])
    assert len(calls) == expected_calls


def test_stale_data_exhausts_budget_and_keeps_failed_attempt(sources, tmp_path):
    config, snapshots, state = configure(sources, tmp_path, max_attempts=2)
    with pytest.raises(BatchFailure):
        run_daily(config, as_of="2026-11-17T21:35:00Z", snapshots=snapshots[0])
    assert ForecastLedger(config.database).forecasts(mode="replay") == []
    assert all(row["stages"]["acquire"]["error_type"] == "MarketDataUnavailable" for row in state.jobs(config.deployment_id))


def test_before_first_origin_skips_acquisition_and_live_cannot_backdate(sources, tmp_path):
    config, snapshots, _ = configure(sources, tmp_path)
    result = run_daily(config, as_of="2026-11-16T08:00:00Z", snapshots=snapshots[0])
    assert all(r["reason"] == "not_due" for r in json.loads((result / "summary.json").read_text())["assets"])
    assert current_slot(config, config.streams[0], "2026-11-16T21:29:00Z")["status"] == "not_due"
    assert current_slot(config, config.streams[0], "2026-11-16T21:30:00Z")["status"] == "due"


@pytest.mark.parametrize("setting,value", [("max_attempts", 100), ("schedule_delay_minutes", 10),
                                           ("retry_delay_seconds", -1), ("unexpected_setting", 1)])
def test_invalid_configuration_fails_before_state_changes(sources, tmp_path, setting, value):
    with pytest.raises((ValueError, TypeError)):
        configure(sources, tmp_path, **{setting: value})
    assert not (tmp_path / "ops.sqlite").exists()


def test_relative_config_paths_and_registered_definition_round_trip(sources, tmp_path):
    config, _, _ = configure(sources, tmp_path)
    path = tmp_path / "deployment.json"
    path.write_text(json.dumps(config.definition()))
    loaded = load_config(path)
    assert loaded.definition() == config.definition()


def test_operational_release_reuses_business_input_from_previous_release(sources, tmp_path):
    config, snapshots, state = configure(sources, tmp_path)
    run_daily(config, as_of="2026-11-16T21:35:00Z", snapshots=snapshots[0])
    next_config = replace(config, deployment_id="batch-v2")
    state.register(next_config.definition())
    state.activate(next_config.deployment_id, now="2026-11-16T21:40:00Z", reason="release")
    result = run_daily(next_config, as_of="2026-11-16T21:40:00Z", snapshots=snapshots[1])
    rows = json.loads((result / "summary.json").read_text())["assets"]
    assert all(row["stages"]["acquire"]["reused_business_forecast"] for row in rows)
    assert len(ForecastLedger(config.database).forecasts(mode="replay")) == 2


def test_monitor_detects_never_started_and_missing_origins_after_last_forecast_without_writing(sources, tmp_path):
    from quant_forecast_lab.health import check_health

    config, snapshots, state = configure(sources, tmp_path)
    before = state.path.read_bytes()
    assert check_health(config, as_of="2026-11-16T08:00:00Z")["status"] == "not_due"
    assert not (tmp_path / "ledger.sqlite").exists()
    never_started = check_health(config, as_of="2026-11-16T23:05:00Z")
    assert {a["code"] for a in never_started["alerts"]} == {"missing_forecasts", "heartbeat_missing"}
    assert state.path.read_bytes() == before
    run_daily(config, as_of="2026-11-16T23:10:00Z", snapshots=snapshots[0])
    recovered = check_health(config, as_of="2026-11-16T23:11:00Z")
    assert recovered["status"] == "healthy" and recovered["assets"][0]["late_origins"] == ["2026-11-16"]
    day_two = check_health(config, as_of="2026-11-17T23:05:00Z")
    assert day_two["assets"][0]["missing_origins"] == ["2026-11-17"]
    assert {a["code"] for a in day_two["alerts"]} >= {"overdue_labels", "heartbeat_stale"}


def test_monitor_expires_stuck_jobs_before_delivery_deadline_and_writes_local_report(sources, tmp_path):
    from quant_forecast_lab.health import check_health, write_health

    config, _, state = configure(sources, tmp_path)
    state.claim(config.deployment_id, "AAA", "2026-11-16", now="2026-11-16T21:35:00Z", lease_minutes=15)
    report = check_health(config, as_of="2026-11-16T21:51:00Z")
    assert report["status"] == "unhealthy"
    assert report["alerts"][0]["code"] == "expired_job_lease"
    path = write_health(tmp_path / "health", report)
    assert path.is_file() and "expired_job_lease" in path.read_text()
    assert json.loads(path.with_name("health.json").read_text())["status"] == "unhealthy"


def test_validated_activation_rollback_and_cli_exit_status(sources, tmp_path, capsys):
    from quant_forecast_lab.cli import main

    config, snapshots, state = configure(sources, tmp_path)
    config_file = tmp_path / "release.json"
    config_file.write_text(json.dumps(replace(config, deployment_id="batch-v2").definition()))
    assert main(["batch", "activate", "--config", str(config_file), "--reason", "release drill"]) == 0
    capsys.readouterr()
    invalid = json.loads(config_file.read_text())
    invalid["deployment_id"] = "invalid"
    invalid["streams"][0]["model_run"] = str(tmp_path / "missing_model")
    config_file.write_text(json.dumps(invalid))
    assert main(["batch", "activate", "--config", str(config_file), "--reason", "bad release"]) == 2
    capsys.readouterr()
    assert state.active("replay")["deployment_id"] == "batch-v2"
    assert main(["batch", "rollback", "--state-db", str(state.path), "--mode", "replay", "--reason", "rollback drill"]) == 0
    capsys.readouterr()
    assert state.active("replay")["deployment_id"] == "batch-v1"
    assert main(["batch", "health", "--state-db", str(state.path), "--mode", "replay", "--as-of", "2026-11-16T23:05:00Z"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "unhealthy"
    snapshot_args = [arg for asset, path in snapshots[0].items() for arg in ("--snapshot", f"{asset}={path}")]
    assert main(["batch", "run", "--state-db", str(state.path), "--mode", "replay", "--as-of", "2026-11-16T23:10:00Z", *snapshot_args]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


def test_health_command_replaces_stale_success_when_active_state_is_missing(tmp_path, capsys):
    from quant_forecast_lab.cli import main

    state = tmp_path / "missing.sqlite"
    assert main(["batch", "health", "--state-db", str(state), "--mode", "live"]) == 2
    assert not state.exists()
    report = json.loads((tmp_path / "health/live/health.json").read_text())
    assert report["alerts"][0]["code"] == "active_configuration_invalid"
    assert json.loads(capsys.readouterr().out)["status"] == "failed"


def test_report_write_failure_is_not_reported_as_success_and_retry_preserves_forecasts(sources, tmp_path, monkeypatch):
    import quant_forecast_lab.daily as daily

    config, snapshots, state = configure(sources, tmp_path)
    original = daily.write_health

    def fail(*args, **kwargs):
        raise OSError("report disk unavailable")

    monkeypatch.setattr(daily, "write_health", fail)
    with pytest.raises(OSError, match="disk unavailable"):
        run_daily(config, as_of="2026-11-16T21:35:00Z", snapshots=snapshots[0])
    assert state.deployment(config.deployment_id)["last_status"] == "failed"
    assert len(ForecastLedger(config.database).forecasts(mode="replay")) == 2
    monkeypatch.setattr(daily, "write_health", original)
    resumed = run_daily(config, as_of="2026-11-16T21:40:00Z", snapshots=snapshots[0])
    assert verify_run(resumed)["status"] == "completed"
    assert state.deployment(config.deployment_id)["last_status"] == "succeeded"
    assert len(ForecastLedger(config.database).forecasts(mode="replay")) == 2


def test_configure_preflight_are_explicit_and_refuse_overwrite(sources, tmp_path, capsys):
    from quant_forecast_lab.cli import main

    streams, _ = sources
    receipt = tmp_path / "prepared.json"
    receipt.write_text(json.dumps({"assets": streams}))
    output = tmp_path / "deployment/config.json"
    args = ["batch", "configure", "--preparation-receipt", str(receipt), "--deployment-id", "configured-v1",
            "--mode", "replay", "--output", str(output)]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "configured_not_activated"
    assert main(["batch", "preflight", "--config", str(output)]) == 0
    capsys.readouterr()
    assert not (output.parent / "operations.sqlite").exists()
    assert main(args) == 2
    assert json.loads(capsys.readouterr().out)["error_type"] == "FileExistsError"


def test_registered_deployment_rejects_a_different_runtime(sources, tmp_path):
    config, _, _ = configure(sources, tmp_path)
    changed = {**config.definition(), "runtime_source_sha256": "a" * 64}
    with pytest.raises(ValueError, match="Runtime source differs"):
        parse_config(changed, base=tmp_path)
