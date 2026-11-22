"""Concurrent publisher checks for runner/watchdog shared reports."""

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from quant_forecast_lab import health
from quant_forecast_lab.health import write_health


def test_concurrent_health_publishers_leave_complete_documents(tmp_path):
    report = {"status": "unhealthy", "mode": "replay", "checked_as_of": "2026-11-16T23:00:00Z",
              "assets": [], "alerts": [{"code": "missing_forecasts"}]}
    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(lambda _: write_health(tmp_path, report), range(12)))
    assert json.loads((tmp_path / "health.json").read_text()) == report
    assert (tmp_path / "health.html").read_text().endswith("</html>")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["health.html", "health.json"]


@pytest.fixture
def delivery(tmp_path, monkeypatch):
    protocol = {"protocol_id": "AAA-v1", "calendar": "XNYS", "earliest_origin": "2026-11-16", "max_forecasts": 1}
    config = SimpleNamespace(mode="replay", deployment_id="test", database=str(tmp_path / "forecasts.sqlite"),
                             state_database=str(tmp_path / "operations.sqlite"), schedule_delay_minutes=30,
                             completion_grace_minutes=90, heartbeat_timeout_minutes=45,
                             streams=[SimpleNamespace(asset="AAA", protocol=protocol)])
    row = {"asset": "AAA", "mode": "replay", "protocol": protocol, "forecast_id": "forecast-1",
           "origin_session": "2026-11-16", "target_session": "2026-11-17", "target_close_at": "2026-11-17T21:00:00Z",
           "as_of": "2026-11-16T21:35:00Z", "predicted_log_return": 0.0,
           "label": {"scored_as_of": "2026-11-18T07:00:00Z", "actual_log_return": 0.01}}
    monkeypatch.setattr(health, "read_forecasts", lambda *args: [row])
    return config, row


def test_recovered_label_keeps_historical_lateness_and_nonzero_health(delivery, tmp_path):
    config, _ = delivery
    report = health.check_health(config, as_of="2026-11-18T08:00:00Z")
    assert report["status"] == "unhealthy"
    assert report["assets"][0]["collection_complete"]
    assert report["assets"][0]["pending_labels"] == 0
    assert report["assets"][0]["late_labels"] == ["forecast-1"]
    assert "late_labels" in {a["code"] for a in report["alerts"]}
    path = write_health(tmp_path, report)
    assert "Late labels" in path.read_text()


def test_health_replay_does_not_see_future_recovered_label(delivery):
    config, _ = delivery
    report = health.check_health(config, as_of="2026-11-17T23:01:00Z")
    assert report["assets"][0]["pending_labels"] == 1
    assert report["assets"][0]["scored"] == 0
    assert report["assets"][0]["late_labels"] == []
    assert "overdue_labels" in {a["code"] for a in report["alerts"]}


def test_late_forecast_remains_an_alert_after_successful_label(delivery):
    config, row = delivery
    row["as_of"] = "2026-11-17T07:00:00Z"
    row["label"]["scored_as_of"] = "2026-11-17T21:35:00Z"
    report = health.check_health(config, as_of="2026-11-18T08:00:00Z")
    assert report["assets"][0]["collection_complete"]
    assert report["assets"][0]["late_origins"] == ["2026-11-16"]
    assert "late_forecasts" in {a["code"] for a in report["alerts"]}


def test_health_exact_label_deadline_is_timely(delivery):
    config, row = delivery
    row["label"]["scored_as_of"] = "2026-11-17T23:00:00Z"
    report = health.check_health(config, as_of="2026-11-18T08:00:00Z")
    assert report["status"] == "collection_complete"
    assert report["assets"][0]["late_labels"] == []
