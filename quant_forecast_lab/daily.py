"""Resumable calendar-aware batch inference; no training or model promotion."""

from __future__ import annotations

import time
from datetime import timedelta
from pathlib import Path

from .artifacts import verify_run
from .batch_config import current_slot
from .experiment import tracked_run, write_json
from .forecast_ledger import ForecastLedger
from .health import check_health, write_health
from .market_data import MarketDataUnavailable, acquire_market_snapshot
from .operations import OperationsStore
from .session_contract import utc_timestamp, validate_sessions
from .shadow import _effective, _now, load_snapshot, record_forecast, score_snapshot


class BatchFailure(RuntimeError):
    def __init__(self, run, summary):
        super().__init__(f"Batch has failed/degraded stages; inspect {run}")
        self.run, self.summary = run, summary


def _ready_labels(ledger, stream, now, mode):
    return any(row["label"] is None and utc_timestamp(row["target_close_at"]) +
               timedelta(minutes=row["publication_delay_minutes"]) <= utc_timestamp(now)
               for row in ledger.forecasts(mode=mode, protocol_id=stream.protocol["protocol_id"], asset=stream.asset))


def _run_stream(config, stream, state, ledger, *, clock, snapshots, acquire, sleep, logger):
    instant = clock()
    slot = current_slot(config, stream, instant)
    due = slot["status"] == "due"
    if not due and not _ready_labels(ledger, stream, instant, config.mode):
        return {"asset": stream.asset, "status": "skipped", "reason": slot["status"], "slot": slot}
    job, claimed = state.claim(config.deployment_id, stream.asset, slot["origin"], now=instant,
                               lease_minutes=config.lease_minutes)
    if not claimed:
        if job["status"] == "succeeded":
            recorded = job["stages"].get("record", {})
            if recorded.get("status") == "succeeded" and ledger.find(
                config.mode, stream.protocol["protocol_id"], stream.asset, slot["origin"]
            ) is None:
                raise ValueError("Operations/forecast ledger diverged; restore consistent state before continuing.")
        return {"asset": stream.asset, "status": "already_completed" if job["status"] == "succeeded" else "busy",
                "job_id": job["job_id"], "slot": slot}

    def checkpoint(stage, result):
        nonlocal job
        job = state.checkpoint(job["job_id"], job["owner"], stage=stage, result=result, now=clock(),
                               lease_minutes=config.lease_minutes)
        logger.info("asset=%s origin=%s stage=%s status=%s", stream.asset, slot["origin"], stage, result["status"])

    def stage(name, work):
        existing = job["stages"].get(name)
        if existing and existing["status"] == "succeeded":
            return existing
        try:
            value = {"status": "succeeded", **work()}
        except Exception as error:
            value = {"status": "failed", "error_type": type(error).__name__, "error": str(error)}
            logger.exception("Stage failed: %s/%s", stream.asset, name)
        checkpoint(name, value)
        return value

    def acquire_fresh():
        previous = ledger.find(config.mode, stream.protocol["protocol_id"], stream.asset, slot["origin"])
        if previous is not None:
            snapshot = previous["snapshot_run"]
            return {"snapshot_run": snapshot, "manifest_sha256": verify_run(snapshot)["manifest_sha256"],
                    "attempts": 0, "transient_failures": [], "reused_business_forecast": True}
        failures = []
        for attempt in range(1, config.max_attempts + 1):
            try:
                if config.mode == "replay":
                    snapshot = Path(snapshots[stream.asset])
                else:
                    end = (utc_timestamp(slot["origin"] + "T00:00:00Z") + timedelta(days=1)).date().isoformat()
                    snapshot = acquire(stream.asset, config.acquisition_start, end, output_root=config.output_root)
                close, _ = load_snapshot(snapshot, stream.asset)
                validate_sessions(close, stream.protocol["calendar"])
                if close.index[-1].date().isoformat() < slot["origin"]:
                    raise MarketDataUnavailable("Latest required completed session has not arrived.")
                if close.index[-1].date().isoformat() != slot["origin"]:
                    raise ValueError("Snapshot contains an unexpected/future origin.")
                return {"snapshot_run": str(Path(snapshot).resolve()), "manifest_sha256": verify_run(snapshot)["manifest_sha256"],
                        "attempts": attempt, "transient_failures": failures}
            except (MarketDataUnavailable, TimeoutError, ConnectionError) as error:
                failures.append({"attempt": attempt, "type": type(error).__name__, "error": str(error)})
                logger.warning("Acquisition unavailable: asset=%s attempt=%d error=%s", stream.asset, attempt, error)
                if attempt == config.max_attempts:
                    raise
                sleep(config.retry_delay_seconds)

    acquired = stage("acquire", acquire_fresh)
    if acquired["status"] == "succeeded":
        snapshot = acquired["snapshot_run"]

        def validate_cached():
            if verify_run(snapshot)["manifest_sha256"] != acquired["manifest_sha256"]:
                raise ValueError("Checkpointed snapshot changed; do not replace an original input silently.")
            return {"snapshot_run": snapshot}

        integrity = stage(f"snapshot_integrity_{job['claims']}", validate_cached)
        if integrity["status"] == "succeeded":
            # A label-basis failure must not silently skip today's independent forecast.
            stage("score", lambda: {"score_run": str(score_snapshot(
                snapshot, asset=stream.asset, database=config.database, mode=config.mode,
                as_of=clock() if config.mode == "replay" else None, output_root=config.output_root,
                protocol_id=stream.protocol["protocol_id"],
            ))})
            if due:
                def record():
                    result, created = record_forecast(stream.model_run, snapshot, stream.protocol_file, database=config.database,
                                                      mode=config.mode, as_of=clock() if config.mode == "replay" else None,
                                                      output_root=config.output_root)
                    return {"forecast_id": result["forecast_id"], "created": created}

                stage("record", record)
    failed = any(s["status"] == "failed" for s in job["stages"].values())
    terminal = "degraded" if failed else "succeeded"
    state.finish(job["job_id"], job["owner"], now=clock(), status=terminal)
    return {"asset": stream.asset, "status": terminal, "job_id": job["job_id"], "slot": slot, "stages": job["stages"]}


def run_daily(config, *, as_of=None, snapshots=None, acquire=acquire_market_snapshot, sleep=time.sleep):
    effective = _effective(config.mode, as_of)
    if config.mode == "live" and snapshots:
        raise ValueError("Live acquisition cannot use replay snapshot overrides.")
    if config.mode == "replay" and (snapshots is None or set(snapshots) != {s.asset for s in config.streams}):
        raise ValueError("Replay requires one explicit snapshot per configured asset.")
    clock = (lambda: effective) if config.mode == "replay" else _now
    state = OperationsStore(config.state_database)
    state.register(config.definition())
    active = state.active(config.mode)
    if active is None or active["deployment_id"] != config.deployment_id:
        raise ValueError("Deployment is not active; validate and activate it before scheduling.")
    ledger = ForecastLedger(config.database)
    summary = {"deployment_id": config.deployment_id, "mode": config.mode, "as_of": effective, "assets": [],
               "training": "not_performed", "promotion": "disabled"}
    with tracked_run(config.output_root, "daily_batch", config.definition()) as (run, logger):
        state.heartbeat(config.deployment_id, now=clock(), status="running", run=run)
        try:
            for stream in config.streams:
                try:
                    result = _run_stream(config, stream, state, ledger, clock=clock, snapshots=snapshots,
                                         acquire=acquire, sleep=sleep, logger=logger)
                except Exception as error:
                    logger.exception("Asset workflow failed: %s", stream.asset)
                    result = {"asset": stream.asset, "status": "failed", "error_type": type(error).__name__, "error": str(error)}
                summary["assets"].append(result)
            failed = any(row["status"] in {"failed", "degraded", "busy"} for row in summary["assets"])
            summary["status"] = "degraded" if failed else "succeeded"
            state.heartbeat(config.deployment_id, now=clock(), status=summary["status"], run=run)
            health = check_health(config, as_of=clock() if config.mode == "replay" else None)
            summary["health_status"] = health["status"]
            write_health(run, health)
            write_health(Path(config.state_database).parent / "health" / config.mode, health)
            write_json(run / "summary.json", summary)
            if failed:
                raise BatchFailure(run, summary)
        except BaseException as error:
            status = "degraded" if isinstance(error, BatchFailure) else "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            state.heartbeat(config.deployment_id, now=clock(), status=status, run=run)
            raise
    return run
