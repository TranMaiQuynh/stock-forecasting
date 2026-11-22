"""Operational windows must preserve late/missing history and delayed labels."""

from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace

import exchange_calendars as xcals
import pytest

from quant_forecast_lab.acceptance import delivery_history, evaluate_acceptance


@pytest.fixture
def observed():
    calendar = xcals.get_calendar("XNYS", start="2026-11-01", end="2027-04-01")
    dates = calendar.sessions_in_range("2026-11-16", "2027-02-01")[:25]
    streams = [SimpleNamespace(asset=a, protocol={"protocol_id": a + "-v1", "calendar": "XNYS",
                                                "earliest_origin": "2026-11-16", "max_forecasts": 60}) for a in ("AAA", "BBB")]
    config = SimpleNamespace(mode="replay", deployment_id="test-v1", streams=streams,
                             schedule_delay_minutes=30, completion_grace_minutes=90)
    rows = []
    for stream in streams:
        for origin in dates:
            target = calendar.next_session(origin)
            received = (calendar.session_close(origin) + timedelta(minutes=35)).isoformat()
            label_received = (calendar.session_close(target) + timedelta(minutes=35)).isoformat()
            rows.append({"mode": "replay", "asset": stream.asset, "protocol": stream.protocol,
                         "origin_session": origin.date().isoformat(), "forecast_id": stream.asset + str(origin.date()),
                         "as_of": received, "recorded_at": "2030-01-01T00:00:00Z",
                         "target_close_at": calendar.session_close(target).isoformat(),
                         "label": {"scored_as_of": label_received, "recorded_at": "2030-01-02T00:00:00Z"}})
    now = (calendar.session_close(calendar.next_session(dates[-1])) + timedelta(minutes=40)).isoformat()
    return config, rows, now, calendar


def test_twenty_consecutive_joint_sessions_allow_review_only(observed):
    config, rows, now, _ = observed
    report = evaluate_acceptance(config, rows, as_of=now)
    assert report["status"] == "ready_for_operational_review"
    assert report["consecutive_compliant_sessions"] == 25
    assert len(report["window"]) == 20
    assert report["model_promotion"] == "disabled"
    assert report["prospective_mode"] is False
    assert len(report["pending_newer_sessions"]) == 1
    assert "2026-11-26" not in [r["origin"] for r in report["window"]]  # exchange holiday


@pytest.mark.parametrize("kind", ["missing", "late_forecast", "late_label", "overdue_label", "duplicate"])
def test_latest_violation_cannot_be_hidden_by_old_successes(observed, kind):
    config, rows, now, _ = observed
    last = rows[-1]
    if kind == "missing":
        rows.remove(last)
    elif kind == "late_forecast":
        last["as_of"] = last["target_close_at"]
    elif kind == "late_label":
        last["label"]["scored_as_of"] = now
        # The as-of is still before the SLO; move beyond the SLO to create a late receipt.
        from quant_forecast_lab.session_contract import utc_timestamp
        now = (utc_timestamp(now) + timedelta(hours=3)).isoformat()
        last["label"]["scored_as_of"] = now
    elif kind == "overdue_label":
        last["label"] = None
        from quant_forecast_lab.session_contract import utc_timestamp
        now = (utc_timestamp(now) + timedelta(hours=3)).isoformat()
    else:
        rows.append(deepcopy(last))
    report = evaluate_acceptance(config, rows, as_of=now)
    assert report["status"] == "not_ready"
    assert report["consecutive_compliant_sessions"] == 0
    assert any(s["issues"] for s in report["assets"][-1]["sessions"])


def test_recovered_late_label_remains_a_failure(observed):
    config, rows, now, _ = observed
    row = rows[4]
    row["label"]["scored_as_of"] = rows[7]["label"]["scored_as_of"]
    before = deepcopy(rows)
    report = evaluate_acceptance(config, rows, as_of=now)
    entry = next(s for s in report["assets"][0]["sessions"] if s["origin"] == row["origin_session"])
    assert entry["issues"] == ["late_label"]
    assert report["status"] == "ready_for_operational_review"  # 20 later sessions, no erased history
    assert rows == before


def test_pending_hole_inside_window_is_not_skipped(observed):
    config, rows, now, _ = observed
    # One asset's label is still in the future relative to the report.
    rows[-2]["label"]["scored_as_of"] = "2030-01-01T00:00:00Z"
    report = evaluate_acceptance(config, rows, as_of=now)
    assert report["status"] == "not_ready"
    assert report["consecutive_compliant_sessions"] == 1


def test_live_receipts_use_actual_recorded_time_not_replay_clock(observed):
    config, rows, now, _ = observed
    config.mode = "live"
    for row in rows:
        row["mode"] = "live"
        row["recorded_at"] = row["as_of"]
        row["label"]["recorded_at"] = row["label"]["scored_as_of"]
    rows[-1]["recorded_at"] = rows[-1]["target_close_at"]
    assert evaluate_acceptance(config, rows, as_of=now)["status"] == "not_ready"


def test_future_records_are_invisible_at_historical_as_of(observed):
    config, rows, _, calendar = observed
    now = calendar.session_close("2026-11-16") + timedelta(minutes=40)
    report = evaluate_acceptance(config, rows, as_of=now)
    assert report["status"] == "not_ready"
    assert report["assets"][0]["sessions"][0]["label_received_at"] is None
    assert report["assets"][0]["sessions"][0]["outcome"] == "pending"


@pytest.mark.parametrize("required", [0, True, 61])
def test_invalid_window_is_rejected(observed, required):
    config, rows, now, _ = observed
    with pytest.raises(ValueError):
        evaluate_acceptance(config, rows, as_of=now, required_sessions=required)


def test_misaligned_schedules_are_not_silently_intersected(observed):
    config, rows, now, _ = observed
    config.streams[-1].protocol["earliest_origin"] = "2026-11-17"
    with pytest.raises(ValueError, match="aligned"):
        evaluate_acceptance(config, rows, as_of=now)


def test_deadline_equality_is_timely_and_early_close_is_used(observed):
    config, rows, now, calendar = observed
    row = next(r for r in rows if r["asset"] == "AAA" and r["origin_session"] == "2026-11-27")
    row["as_of"] = (calendar.session_close("2026-11-27") + timedelta(minutes=120)).isoformat()
    from quant_forecast_lab.acceptance import label_deadline
    row["label"]["scored_as_of"] = label_deadline(config, row).isoformat()
    entry = next(s for s in delivery_history(config, rows, as_of=now)[0]["sessions"] if s["origin"] == "2026-11-27")
    assert entry["outcome"] == "passed"
    assert "18:30:00" in entry["due_at"]  # half-day close 18:00 UTC
