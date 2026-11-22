"""Independent read-only checks of expected deliveries, leases and delayed labels."""

from __future__ import annotations

import html
import json
import math
import sqlite3
import time
import uuid
from contextlib import closing
from datetime import timedelta
from pathlib import Path

from .acceptance import forecast_time, label_deadline, label_time
from .batch_config import expected_slots
from .experiment import write_json
from .operations import OperationsStore
from .session_contract import utc_timestamp
from .shadow import _effective, _now


def read_forecasts(path, mode):
    """Monitoring never initializes, repairs or touches the business ledger."""
    path = Path(path).resolve()
    if not path.exists():
        return []
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=5)) as connection:
        connection.row_factory = sqlite3.Row
        if connection.execute("PRAGMA user_version").fetchone()[0] != 1:
            raise ValueError("Unsupported forecast ledger schema.")
        rows = connection.execute("""
            SELECT f.payload AS forecast,l.payload AS label FROM forecasts f
            LEFT JOIN labels l USING(forecast_id) WHERE f.mode=? ORDER BY f.origin_session
        """, (mode,)).fetchall()
    return [{**json.loads(r["forecast"]), "label": json.loads(r["label"]) if r["label"] else None} for r in rows]


def check_health(config, *, as_of=None):
    effective = _effective(config.mode, as_of)
    instant = utc_timestamp(effective)
    rows = [{**r, "label": r["label"] if r["label"] is not None and label_time(r["label"], config.mode) <= instant else None}
            for r in read_forecasts(config.database, config.mode) if forecast_time(r, config.mode) <= instant]
    state = OperationsStore(config.state_database, create=False) if Path(config.state_database).exists() else None
    deployment = state.deployment(config.deployment_id) if state else None
    alerts, assets = [], []
    any_due, collection_complete = False, True
    for stream in config.streams:
        slots = expected_slots(config, stream, effective)
        any_due = any_due or bool(slots)
        forecasts = [r for r in rows if r["asset"] == stream.asset and r["protocol"]["protocol_id"] == stream.protocol["protocol_id"]]
        observed = {r["origin_session"] for r in forecasts}
        missing = [s["origin"] for s in slots if utc_timestamp(s["deadline_at"]) <= instant and s["origin"] not in observed]
        if missing:
            alerts.append({"code": "missing_forecasts", "asset": stream.asset, "origins": missing})
        pending = [r for r in forecasts if r["label"] is None]
        overdue = [r["forecast_id"] for r in pending if label_deadline(config, r) <= instant]
        if overdue:
            alerts.append({"code": "overdue_labels", "asset": stream.asset, "forecast_ids": overdue})
        deadlines = {s["origin"]: utc_timestamp(s["deadline_at"]) for s in slots}
        late = [r["origin_session"] for r in forecasts if r["origin_session"] in deadlines and
                forecast_time(r, config.mode) > deadlines[r["origin_session"]]]
        late_labels = [r["forecast_id"] for r in forecasts if r["label"] is not None
                       and label_time(r["label"], config.mode) > label_deadline(config, r)]
        if late:
            alerts.append({"code": "late_forecasts", "asset": stream.asset, "origins": late})
        if late_labels:
            alerts.append({"code": "late_labels", "asset": stream.asset, "forecast_ids": late_labels})
        complete = len(forecasts) == stream.protocol["max_forecasts"] and not pending
        collection_complete = collection_complete and complete
        labelled = [r for r in forecasts if r["label"] is not None]
        actual = [r["label"]["actual_log_return"] for r in labelled]
        residuals = [r["predicted_log_return"] - y for r, y in zip(labelled, actual, strict=True)]
        error = math.sqrt(sum(x * x for x in residuals) / len(residuals)) if residuals else None
        reference = math.sqrt(sum(x * x for x in actual) / len(actual)) if actual else None
        assets.append({
            "asset": stream.asset, "protocol_id": stream.protocol["protocol_id"], "expected_mature_origins": len(slots),
            "forecasts": len(forecasts), "scored": len(labelled), "pending_labels": len(pending),
            "missing_origins": missing, "overdue_labels": len(overdue), "collection_complete": complete,
            "late_origins": late, "late_labels": late_labels,
            "latest_origin": forecasts[-1]["origin_session"] if forecasts else None,
            "latest_target_session": forecasts[-1]["target_session"] if forecasts else None,
            "candidate_log_return": forecasts[-1]["predicted_log_return"] if forecasts else None,
            "baseline_log_return": 0.0, "rmse": error,
            "rmse_skill_vs_zero": 1 - error / reference if reference else None,
            "model_status": "research_candidate_not_promoted",
        })
    if deployment and deployment["definition"] != config.definition():
        alerts.append({"code": "deployment_definition_changed"})
    if any_due and not collection_complete:
        first_due = min(utc_timestamp(s["due_at"]) for stream in config.streams for s in expected_slots(config, stream, effective))
        heartbeat_due = first_due + timedelta(minutes=config.heartbeat_timeout_minutes)
        if instant >= heartbeat_due:
            if not deployment or not deployment["last_seen"]:
                alerts.append({"code": "heartbeat_missing"})
            elif utc_timestamp(deployment["last_seen"]) + timedelta(minutes=config.heartbeat_timeout_minutes) <= instant:
                alerts.append({"code": "heartbeat_stale", "last_seen": deployment["last_seen"]})
        if deployment and deployment["last_status"] in {"degraded", "interrupted", "failed"}:
            alerts.append({"code": "last_batch_failed", "run": deployment["last_run"]})
    if state:
        for job in state.jobs(config.deployment_id):
            if job["status"] == "running" and utc_timestamp(job["lease_until"]) <= instant:
                alerts.append({"code": "expired_job_lease", "asset": job["asset"], "origin": job["origin"], "job_id": job["job_id"]})
    return {
        "schema_version": 2, "deployment_id": config.deployment_id, "mode": config.mode,
        "generated_at": _now(), "checked_as_of": effective,
        "status": "unhealthy" if alerts else ("collection_complete" if collection_complete else "healthy" if any_due else "not_due"),
        "alerts": alerts, "assets": assets,
        "last_heartbeat": deployment["last_seen"] if deployment else None,
        "interpretation": "Operational status and descriptive paired errors; no statistical, trading or profitability claim.",
    }


def write_health(directory, report):
    """Local report only; no external notifications or web service is started."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # The runner and independent watchdog can publish concurrently.
    token = uuid.uuid4().hex

    def publish(source, destination):
        # Windows can briefly deny replacement while another publisher renames.
        for attempt in range(5):
            try:
                source.replace(destination)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05)

    json_temporary = directory / f"health.{token}.json"
    write_json(json_temporary, report)
    publish(json_temporary, directory / "health.json")
    rows = []
    for asset in report["assets"]:
        forecast = f"{asset['candidate_log_return'] * 100:.3f}%" if asset["candidate_log_return"] is not None else "pending"
        skill = f"{asset['rmse_skill_vs_zero'] * 100:.3f}%" if asset["rmse_skill_vs_zero"] is not None else "pending"
        values = [asset["asset"], asset["latest_origin"], asset["latest_target_session"], forecast,
                  asset["forecasts"], asset["scored"], skill, asset["pending_labels"], len(asset["missing_origins"]),
                  len(asset.get("late_origins", [])), len(asset.get("late_labels", []))]
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in values) + "</tr>")
    document = "<!doctype html><html lang='en'><meta charset='utf-8'><title>Forecast pipeline status</title>"
    document += "<style>body{font:16px system-ui;margin:2rem;max-width:1100px}td,th{padding:.6rem;border:1px solid #ccc}table{border-collapse:collapse}pre{white-space:pre-wrap}</style>"
    document += f"<h1>Forecast pipeline: {html.escape(report['status'])}</h1>"
    document += f"<p>Mode: {html.escape(report['mode'])}; checked: {html.escape(report['checked_as_of'])}</p>"
    document += "<table><tr><th>Asset</th><th>Origin</th><th>Target session</th><th>Candidate log return</th><th>Forecasts</th><th>Scored</th><th>RMSE skill vs zero</th><th>Pending</th><th>Missing</th><th>Late forecasts</th><th>Late labels</th></tr>"
    document += "".join(rows) + "</table><h2>Active alerts</h2><pre>" + html.escape(json.dumps(report["alerts"], indent=2)) + "</pre>"
    document += "<p>Candidate forecasts are research outputs. Predictive advantage and investment returns are unverified.</p></html>"
    path = directory / "health.html"
    temporary = directory / f"health.{token}.html.tmp"
    temporary.write_text(document, encoding="utf-8")
    publish(temporary, path)
    return path
