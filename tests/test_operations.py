import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from quant_forecast_lab.operations import OperationsStore

DEFINITION = {"deployment_id": "demo-v1", "mode": "replay", "streams": ["DEMO"]}
NOW = "2026-11-16T21:30:00Z"
LATER = "2026-11-16T21:46:00Z"


def store(tmp_path):
    result = OperationsStore(tmp_path / "operations.sqlite")
    result.register(DEFINITION)
    return result


def test_concurrent_claim_has_one_owner_and_completed_job_is_not_repeated(tmp_path):
    state = store(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: OperationsStore(state.path).claim("demo-v1", "DEMO", "2026-11-16",
                                                                         now=NOW, lease_minutes=15), range(8)))
    assert sum(claimed for _, claimed in results) == 1
    job = next(job for job, claimed in results if claimed)
    state.finish(job["job_id"], job["owner"], now=NOW, status="succeeded")
    terminal, claimed = state.claim("demo-v1", "DEMO", "2026-11-16", now=LATER, lease_minutes=15)
    assert not claimed and terminal["status"] == "succeeded"


def test_expired_lease_recovers_checkpoint_and_fences_old_owner(tmp_path):
    state = store(tmp_path)
    first, _ = state.claim("demo-v1", "DEMO", "2026-11-16", now=NOW, lease_minutes=15)
    snapshot = {"status": "succeeded", "snapshot_run": "original", "sha256": "a" * 64}
    state.checkpoint(first["job_id"], first["owner"], stage="acquire", result=snapshot, now=NOW)
    next_job, claimed = state.claim("demo-v1", "DEMO", "2026-11-16", now=LATER, lease_minutes=15)
    assert claimed and next_job["stages"]["acquire"] == snapshot and next_job["claims"] == 2
    with pytest.raises(ValueError, match="lease"):
        state.checkpoint(first["job_id"], first["owner"], stage="record", result={"status": "succeeded"}, now=LATER)
    with pytest.raises(ValueError, match="cannot be changed"):
        state.checkpoint(next_job["job_id"], next_job["owner"], stage="acquire", result={"status": "succeeded"}, now=LATER)
    assert state.jobs("demo-v1")[0]["stages"]["acquire"] == snapshot


def test_frozen_definition_and_monitor_do_not_reset_heartbeat(tmp_path):
    state = store(tmp_path)
    state.heartbeat("demo-v1", now=NOW, status="running", run="run-1")
    with pytest.raises(ValueError, match="changed"):
        state.register({**DEFINITION, "streams": ["OTHER"]})
    monitor = OperationsStore(state.path, create=False)
    before = state.path.read_bytes()
    assert monitor.deployment("demo-v1")["last_seen"] == NOW
    assert monitor.jobs("demo-v1") == []
    assert state.path.read_bytes() == before
    with pytest.raises(ValueError, match="cannot mutate"):
        monitor.heartbeat("demo-v1", now=LATER, status="succeeded", run="fake")


def test_activation_history_preserves_previous_version_for_rollback(tmp_path):
    state = store(tmp_path)
    state.activate("demo-v1", now=NOW, reason="initial pilot")
    state.register({**DEFINITION, "deployment_id": "demo-v2"})
    state.activate("demo-v2", now=LATER, reason="new operational release")
    assert state.active("replay")["deployment_id"] == "demo-v2"
    previous = state.previous("replay")
    assert previous["deployment_id"] == "demo-v1"
    state.activate(previous["deployment_id"], now=LATER, reason="rollback drill")
    assert state.active("replay")["deployment_id"] == "demo-v1"
    with pytest.raises(ValueError, match="Unknown"):
        state.activate("missing", now=LATER, reason="invalid")
    assert state.active("replay")["deployment_id"] == "demo-v1"


def test_forecast_database_cannot_be_used_as_operations_state(tmp_path):
    path = tmp_path / "other.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE forecasts (id TEXT)")
        connection.execute("PRAGMA user_version=1")
    with pytest.raises(ValueError, match="unrelated"):
        OperationsStore(path)


def test_expired_owner_cannot_mark_a_job_successful(tmp_path):
    state = store(tmp_path)
    job, _ = state.claim("demo-v1", "DEMO", "2026-11-16", now=NOW, lease_minutes=15)
    with pytest.raises(ValueError, match="expired"):
        state.finish(job["job_id"], job["owner"], now=LATER, status="succeeded")
    assert state.jobs("demo-v1")[0]["status"] == "running"
