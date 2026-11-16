import sqlite3
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest

from quant_forecast_lab.forecast_ledger import ForecastLedger


def prediction():
    return {
        "mode": "replay", "protocol": {"protocol_id": "demo-v1", "stop_after": 252},
        "asset": "DEMO", "origin_session": "2026-11-16", "target_session": "2026-11-17",
        "origin_close_at": "2026-11-16T21:00:00+00:00", "target_open_at": "2026-11-17T14:30:00+00:00",
        "target_close_at": "2026-11-17T21:00:00+00:00", "publication_delay_minutes": 20,
        "as_of": "2026-11-16T21:30:00+00:00", "received_at": "2026-11-16T21:25:00+00:00",
        "origin_close": 100.0, "predicted_log_return": 0.01, "model_sha256": "a" * 64, "input_sha256": "b" * 64,
    }


def score(ledger, forecast, actual=101.0, **kwargs):
    settings = {"actual_close": actual, "received_at": "2026-11-17T21:25:00+00:00",
                "input_sha256": "c" * 64, "scored_as_of": "2026-11-17T21:30:00+00:00", **kwargs}
    return ledger.score(forecast["forecast_id"], **settings)


def test_identical_retry_retains_first_record_even_with_new_receipt(tmp_path):
    ledger = ForecastLedger(tmp_path / "ledger.sqlite")
    original, created = ledger.record(prediction())
    retry = {**prediction(), "as_of": "2026-11-16T21:40:00+00:00", "snapshot_run": "different"}
    again, created_again = ledger.record(retry)
    assert created and not created_again and original == again
    assert ForecastLedger(ledger.path).find("replay", "demo-v1", "DEMO", "2026-11-16") == original


@pytest.mark.parametrize("field,value", [("predicted_log_return", 0.02), ("input_sha256", "d" * 64),
                                          ("model_sha256", "e" * 64)])
def test_conflicting_retry_does_not_overwrite_prediction(tmp_path, field, value):
    ledger = ForecastLedger(tmp_path / "ledger.sqlite")
    original, _ = ledger.record(prediction())
    with pytest.raises(ValueError, match="Conflicting forecast"):
        ledger.record({**prediction(), field: value})
    assert ledger.forecasts(mode="replay")[0]["predicted_log_return"] == original["predicted_log_return"]


def test_protocol_changes_and_sql_edits_are_rejected(tmp_path):
    ledger = ForecastLedger(tmp_path / "ledger.sqlite")
    ledger.record(prediction())
    changed = deepcopy(prediction())
    changed["protocol"]["stop_after"] = 1000
    with pytest.raises(ValueError, match="frozen"):
        ledger.record(changed)
    with sqlite3.connect(ledger.path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("DELETE FROM forecasts")


def test_concurrent_retry_has_one_winner(tmp_path):
    ledger = ForecastLedger(tmp_path / "ledger.sqlite")
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: ForecastLedger(ledger.path).record(prediction()), range(8)))
    assert sum(created for _, created in results) == 1
    assert len({r[0]["forecast_id"] for r in results}) == 1


def test_delayed_labels_keep_first_vintage_and_metrics_use_matched_targets(tmp_path):
    ledger = ForecastLedger(tmp_path / "ledger.sqlite")
    first, _ = ledger.record(prediction())
    second, _ = ledger.record({**prediction(), "asset": "OTHER"})
    with pytest.raises(ValueError, match="before availability"):
        score(ledger, first, scored_as_of="2026-11-17T21:19:59+00:00")
    labelled, created = score(ledger, first)
    repeated, again = score(ledger, first, input_sha256="d" * 64)
    assert created and not again and repeated == labelled
    with pytest.raises(ValueError, match="first label"):
        score(ledger, first, actual=102.0)
    report = ledger.report(mode="replay", as_of="2026-11-17T22:00:00+00:00")
    a, b = report["groups"]
    assert a["asset"] == "DEMO" and a["scored"] == 1
    assert a["rmse"] == pytest.approx(abs(0.01 - labelled["actual_log_return"]))
    assert a["rmse_skill_vs_zero"] == pytest.approx(1 - a["rmse"] / abs(labelled["actual_log_return"]))
    assert b["asset"] == "OTHER" and b["overdue_unscored"] == 1 and b["rmse"] is None
    assert ledger.find("replay", "demo-v1", "OTHER", "2026-11-16")["forecast_id"] == second["forecast_id"]


def test_report_separates_live_replay_and_handles_zero_baseline_error(tmp_path):
    ledger = ForecastLedger(tmp_path / "ledger.sqlite")
    first, _ = ledger.record(prediction())
    score(ledger, first, actual=100.0)
    assert ledger.report(mode="live")["groups"] == []
    assert ledger.report(mode="replay")["groups"][0]["rmse_skill_vs_zero"] is None


def test_live_cannot_backdate_a_historical_forecast(tmp_path):
    ledger = ForecastLedger(tmp_path / "ledger.sqlite")
    old = prediction()
    for field in ("origin_session", "target_session", "origin_close_at", "target_open_at",
                  "target_close_at", "as_of", "received_at"):
        old[field] = old[field].replace("2026", "2020")
    with pytest.raises(ValueError, match="backdated"):
        ledger.record({**old, "mode": "live"})
    assert ledger.forecasts(mode="live") == []


def test_unknown_database_version_is_not_migrated_silently(tmp_path):
    path = tmp_path / "ledger.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version=99")
    with pytest.raises(ValueError, match="schema"):
        ForecastLedger(path)
