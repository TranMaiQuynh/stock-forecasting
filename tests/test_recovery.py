import json
from pathlib import Path

import pytest

from quant_forecast_lab.forecast_ledger import ForecastLedger
from quant_forecast_lab.recovery import copy_database, restore_database


def ledger(tmp_path):
    state = ForecastLedger(tmp_path / "ledger.sqlite")
    row = {"mode": "replay", "protocol": {"protocol_id": "restore-demo"}, "asset": "DEMO",
           "origin_session": "2026-11-16", "target_session": "2026-11-17",
           "origin_close_at": "2026-11-16T21:00:00Z", "target_open_at": "2026-11-17T14:30:00Z",
           "target_close_at": "2026-11-17T21:00:00Z", "as_of": "2026-11-16T21:30:00Z",
           "received_at": "2026-11-16T21:25:00Z", "publication_delay_minutes": 20,
           "origin_close": 100.0, "predicted_log_return": 0.01, "model_sha256": "a" * 64, "input_sha256": "b" * 64}
    forecast, _ = state.record(row)
    state.score(forecast["forecast_id"], actual_close=101.0, received_at="2026-11-17T21:25:00Z",
                scored_as_of="2026-11-17T21:30:00Z", input_sha256="c" * 64)
    return state, row


def test_backup_restore_preserves_forecasts_labels_and_duplicate_protection(tmp_path):
    original, row = ledger(tmp_path)
    before = original.forecasts(mode="replay")
    backup = tmp_path / "backup.sqlite"
    receipt = copy_database(original.path, backup)
    assert receipt["counts"] == {"forecasts": 1, "labels": 1, "protocols": 1}
    restored = tmp_path / "recovered.sqlite"
    result = restore_database(backup, restored)
    assert result["operation"] == "restore_to_new_path"
    state = ForecastLedger(restored)
    assert state.forecasts(mode="replay") == before
    _, created = state.record(row)
    assert not created
    assert original.forecasts(mode="replay") == before


def test_corrupted_backup_and_existing_destination_are_rejected(tmp_path):
    original, _ = ledger(tmp_path)
    backup = tmp_path / "backup.sqlite"
    copy_database(original.path, backup)
    with pytest.raises(ValueError, match="must be new"):
        copy_database(original.path, backup)
    backup.write_bytes(b"corrupt")
    destination = tmp_path / "recovered.sqlite"
    with pytest.raises(ValueError, match="checksum"):
        restore_database(backup, destination)
    assert not destination.exists()


def test_backup_cannot_add_files_inside_a_finalized_run(tmp_path):
    original, _ = ledger(tmp_path)
    source_run = tmp_path / "model"
    source_run.mkdir()
    (source_run / "manifest.json").write_text(json.dumps({"schema_version": 1, "status": "completed"}))
    with pytest.raises(ValueError, match="finalized"):
        copy_database(original.path, source_run / "backup.sqlite")
    assert not Path(source_run / "backup.sqlite").exists()
