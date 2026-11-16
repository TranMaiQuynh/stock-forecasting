"""Watchdog CLI must replace stale status when a ledger becomes unreadable."""

import json
from types import SimpleNamespace

from quant_forecast_lab import recovery
from quant_forecast_lab.cli import main


def test_corrupt_ledger_refreshes_unhealthy_report(tmp_path, monkeypatch, capsys):
    ledger = tmp_path / "forecasts.sqlite"
    ledger.write_text("corrupt database")
    state = tmp_path / "operations.sqlite"
    config = SimpleNamespace(database=str(ledger), state_database=str(state), mode="live")
    monkeypatch.setattr(recovery, "active_config", lambda *args: config)
    report = tmp_path / "health/live/health.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"status": "healthy"}))
    assert main(["batch", "health", "--state-db", str(state), "--mode", "live"]) == 2
    assert json.loads(capsys.readouterr().out)["error_type"] == "DatabaseError"
    assert json.loads(report.read_text())["status"] == "unhealthy"
    assert "health_check_failed" in (report.parent / "health.html").read_text()
