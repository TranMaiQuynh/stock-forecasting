"""Read-only operational delivery criteria, separate from predictive validation."""

from __future__ import annotations

from collections import Counter
from datetime import timedelta

from .batch_config import expected_slots
from .session_contract import utc_timestamp


def forecast_time(row, mode):
    return utc_timestamp(row["recorded_at"] if mode == "live" else row["as_of"])


def label_time(label, mode):
    return utc_timestamp(label["recorded_at"] if mode == "live" else label["scored_as_of"])


def label_deadline(config, row):
    return utc_timestamp(row["target_close_at"]) + timedelta(
        minutes=config.schedule_delay_minutes + config.completion_grace_minutes
    )


def delivery_history(config, rows, *, as_of):
    """Assess durable receipt times; later recovery never erases a violation."""
    now = utc_timestamp(as_of)
    histories = []
    for stream in config.streams:
        selected = [r for r in rows if r["mode"] == config.mode and r["asset"] == stream.asset
                    and r["protocol"]["protocol_id"] == stream.protocol["protocol_id"]
                    and forecast_time(r, config.mode) <= now]
        sessions = []
        for slot in expected_slots(config, stream, now):
            matches = [r for r in selected if r["origin_session"] == slot["origin"]]
            issues = []
            forecast_due = utc_timestamp(slot["deadline_at"])
            entry = {**slot, "forecast_ids": [r["forecast_id"] for r in matches],
                     "forecast_received_at": None, "label_received_at": None, "label_deadline_at": None}
            if not matches:
                if now >= forecast_due:
                    issues.append("missing_forecast")
                pending = True
            else:
                if len(matches) != 1:
                    issues.append("duplicate_forecasts")
                row = matches[0]
                received, deadline = forecast_time(row, config.mode), label_deadline(config, row)
                entry.update(forecast_received_at=received.isoformat(), label_deadline_at=deadline.isoformat())
                if received > forecast_due:
                    issues.append("late_forecast")
                label = row["label"]
                # A historical replay report must not see labels received later.
                observed = label is not None and label_time(label, config.mode) <= now
                pending = not observed
                if observed:
                    received_label = label_time(label, config.mode)
                    entry["label_received_at"] = received_label.isoformat()
                    if received_label > deadline:
                        issues.append("late_label")
                elif now >= deadline:
                    issues.append("overdue_label")
            entry.update(issues=issues, outcome="failed" if issues else "pending" if pending else "passed")
            sessions.append(entry)
        histories.append({"asset": stream.asset, "protocol_id": stream.protocol["protocol_id"], "sessions": sessions,
                          "outcomes": dict(Counter(s["outcome"] for s in sessions))})
    return histories


def evaluate_acceptance(config, rows, *, as_of, required_sessions=20):
    """Require a recent consecutive shared-session window with timely labels.

    This gate allows independent operational review; it never promotes a model.
    Newer pending labels do not erase an established window, but an observable
    violation immediately resets it. Pending holes inside a window cannot pass.
    """
    if isinstance(required_sessions, bool) or not isinstance(required_sessions, int) or required_sessions < 1:
        raise ValueError("required_sessions must be a positive integer.")
    if not config.streams or required_sessions > min(s.protocol["max_forecasts"] for s in config.streams):
        raise ValueError("Acceptance window cannot exceed the frozen collection budget.")
    history = delivery_history(config, rows, as_of=as_of)
    origins = [s["origin"] for s in history[0]["sessions"]]
    if any([s["origin"] for s in h["sessions"]] != origins for h in history):
        raise ValueError("Joint acceptance requires aligned origin schedules; assess different calendars/start dates separately.")
    joint = []
    for i, origin in enumerate(origins):
        outcomes = [h["sessions"][i]["outcome"] for h in history]
        outcome = "failed" if "failed" in outcomes else "pending" if "pending" in outcomes else "passed"
        joint.append({"origin": origin, "outcome": outcome})
    assessed = list(joint)
    while assessed and assessed[-1]["outcome"] == "pending":
        assessed.pop()
    consecutive = 0
    for entry in reversed(assessed):
        if entry["outcome"] != "passed":
            break
        consecutive += 1
    ready = consecutive >= required_sessions
    window = assessed[-required_sessions:]
    return {
        "schema_version": 1, "deployment_id": config.deployment_id, "mode": config.mode,
        "checked_as_of": utc_timestamp(as_of).isoformat(), "required_sessions": required_sessions,
        "status": "ready_for_operational_review" if ready else "not_ready",
        "consecutive_compliant_sessions": consecutive,
        "window_start": window[0]["origin"] if window else None,
        "window_end": window[-1]["origin"] if window else None,
        "window": window, "pending_newer_sessions": joint[len(assessed):], "assets": history,
        "qualification": "delivery_and_label_timing_only", "prospective_mode": config.mode == "live",
        "model_promotion": "disabled", "production_readiness": "not_established",
        "interpretation": "A replay or a compliant delivery window establishes neither predictive advantage nor investment returns.",
    }
