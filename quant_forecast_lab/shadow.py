"""Local prospective batch pilot with frozen checkpoints and delayed labels.

This operational pilot does not confirm the v2 monthly-selection research method.
No automatic model promotion, orders or external timestamp attestation is provided.
"""

from __future__ import annotations

import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .artifacts import verify_run
from .experiment import sha256_file, tracked_run, write_json
from .forecast import run_saved_forecast
from .forecast_ledger import ForecastLedger
from .research import validate_close_frame
from .session_contract import forecast_contract, label_contract, utc_timestamp, validate_sessions


def _now():
    return datetime.now(UTC).isoformat()


def _effective(mode, as_of):
    if mode not in {"live", "replay"} or (mode == "live" and as_of is not None) or (mode == "replay" and as_of is None):
        raise ValueError("Live uses the real clock without --as-of; replay requires an explicit --as-of.")
    return utc_timestamp(as_of if mode == "replay" else _now()).isoformat()


def _outside_sources(sources, *, database=None, output_root=None, run_name=None):
    for source in sources:
        root = Path(source).resolve()
        if database is not None and Path(database).resolve().is_relative_to(root):
            raise ValueError("Ledger must be outside verified source runs.")
        if output_root is not None and (Path(output_root).resolve() / run_name).is_relative_to(root):
            raise ValueError("Output must be outside verified source runs.")


def capture_snapshot(input_csv, *, asset, data_kind="user_supplied", calendar="XNYS", output_root="log"):
    """Receipt proves local observation time only; it does not certify the vendor."""
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9._-]{0,30}", asset) or data_kind not in {"market", "synthetic", "user_supplied"}:
        raise ValueError("Expected a safe single-asset identifier and explicit data kind.")
    settings = {"asset": asset, "data_kind": data_kind, "calendar": calendar}
    with tracked_run(output_root, "observed_snapshot", settings) as (run, logger):
        close = validate_close_frame(pd.read_csv(input_csv))
        validate_sessions(close, calendar)
        close.to_csv(run / "close.csv")
        write_json(run / "data_manifest.json", {
            **settings, "snapshot_sha256": sha256_file(run / "close.csv"),
            "source_file_sha256": sha256_file(input_csv), "observations": len(close),
            "availability": "Local receipt ends at enclosing manifest ended_at; vendor availability unverified",
            "point_in_time_verified": False,
        })
        logger.info("Captured %d observed closes; historical contents are not retrospective availability evidence", len(close))
    return run


def load_snapshot(directory, asset):
    root = Path(directory).resolve()
    verified = verify_run(root)
    if verified["model_name"] not in {"observed_snapshot", "market_snapshot"}:
        raise ValueError("Expected a verified observed_snapshot or market_snapshot receipt.")
    data = json.loads((root / "data_manifest.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if data.get("asset") != asset or "close.csv" not in manifest["artifact_sha256"]:
        raise ValueError("Snapshot asset/inventory does not match the model.")
    close = validate_close_frame(pd.read_csv(root / "close.csv"))
    return close, {
        "received_at": utc_timestamp(manifest["ended_at"]).isoformat(),
        "input_sha256": manifest["artifact_sha256"]["close.csv"],
        "data_kind": "market" if verified["model_name"] == "market_snapshot" else data["data_kind"],
    }


def create_protocol(model_run, output_file, *, protocol_id, calendar="XNYS", earliest_origin="2026-11-16",
                    max_forecasts=252, minimum_skill=0.01, publication_delay_minutes=20):
    """Pin one model checkpoint before collection; the threshold is not a power calculation."""
    root = Path(model_run).resolve()
    verified = verify_run(root)
    if verified["model_name"] not in {"ridge_log_return", "return_forecast", "shadow_model"}:
        raise ValueError("Expected a verified numerical model run.")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", protocol_id):
        raise ValueError("Invalid protocol_id.")
    if isinstance(max_forecasts, bool) or not isinstance(max_forecasts, int) or not 1 <= max_forecasts <= 2520:
        raise ValueError("max_forecasts must be between 1 and 2520.")
    if not math.isfinite(minimum_skill) or not 0 <= minimum_skill < 1:
        raise ValueError("minimum_skill must be a finite fraction between 0 and 1.")
    origin = pd.Timestamp(earliest_origin)
    if pd.isna(origin) or origin.tzinfo is not None or origin != origin.normalize():
        raise ValueError("earliest_origin must be a session date.")
    if calendar not in {"XNYS", "XNAS"} or isinstance(publication_delay_minutes, bool) or not isinstance(publication_delay_minutes, int) or not 0 <= publication_delay_minutes <= 240:
        raise ValueError("Unsupported calendar/publication buffer.")
    model = json.loads((root / "artifacts/model.json").read_text(encoding="utf-8"))
    protocol = {
        "schema_version": 1, "protocol_id": protocol_id, "asset": config["asset"], "calendar": calendar,
        "purpose": "fixed_checkpoint_operational_shadow_pilot", "v2_research_confirmation": False,
        "model_manifest_sha256": verified["manifest_sha256"],
        "model_sha256": sha256_file(root / "artifacts/model.json"), "estimator_kind": model.get("estimator_kind", "ridge"),
        "earliest_origin": origin.date().isoformat(), "max_forecasts": max_forecasts,
        "minimum_practical_rmse_skill": minimum_skill, "publication_delay_minutes": publication_delay_minutes,
        "update_rule": "frozen_checkpoint; a different model requires a new protocol",
        "statistical_power_calibrated": False, "promotion": "disabled", "registered_at": _now(),
        "stop_rule": "Stop collecting at budget; retain baseline if practical threshold is missed; otherwise independent review only",
    }
    path = Path(output_file)
    if path.resolve().is_relative_to(root):
        raise ValueError("Protocol must be outside the immutable source run.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return protocol


def _protocol(path):
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    if (protocol.get("schema_version") != 1 or protocol.get("purpose") != "fixed_checkpoint_operational_shadow_pilot"
            or protocol.get("promotion") != "disabled" or protocol.get("v2_research_confirmation") is not False):
        raise ValueError("Unsupported shadow protocol; no promotion or v2 confirmation is permitted.")
    if (isinstance(protocol.get("max_forecasts"), bool) or not isinstance(protocol.get("max_forecasts"), int)
            or not 1 <= protocol["max_forecasts"] <= 2520):
        raise ValueError("Invalid frozen collection budget.")
    if not math.isfinite(protocol["minimum_practical_rmse_skill"]) or not 0 <= protocol["minimum_practical_rmse_skill"] < 1:
        raise ValueError("Invalid practical skill threshold.")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", protocol["protocol_id"]):
        raise ValueError("Invalid protocol_id.")
    return protocol


def record_forecast(model_run, snapshot_run, protocol_file, *, database, mode, as_of=None, output_root="log"):
    _outside_sources((model_run, snapshot_run), database=database, output_root=output_root, run_name="shadow_record")
    effective = _effective(mode, as_of)
    protocol = _protocol(protocol_file)
    model = verify_run(model_run)
    if model["manifest_sha256"] != protocol["model_manifest_sha256"]:
        raise ValueError("Model does not match the checkpoint pinned before collection.")
    if sha256_file(Path(model_run) / "artifacts/model.json") != protocol["model_sha256"]:
        raise ValueError("Protocol's model checksum does not match the pinned model.")
    if mode == "live" and utc_timestamp(protocol["registered_at"]) > utc_timestamp(effective):
        raise ValueError("Protocol registration cannot be in the future.")
    close, receipt = load_snapshot(snapshot_run, protocol["asset"])
    data = json.loads((Path(model_run) / "data_manifest.json").read_text(encoding="utf-8"))
    if mode == "live" and "synthetic" in {receipt["data_kind"], data["data_kind"]}:
        raise ValueError("Synthetic data/models cannot enter a live prospective stream.")
    origin = close.index[-1].date().isoformat()
    if origin < protocol["earliest_origin"]:
        raise ValueError("Origin precedes the frozen protocol's earliest permitted session.")
    ledger = ForecastLedger(database)
    previous = ledger.find(mode, protocol["protocol_id"], protocol["asset"], origin)
    if previous is not None:
        if previous["protocol"] != protocol or previous["input_sha256"] != receipt["input_sha256"]:
            raise ValueError("Conflicting forecast retry; original prediction is preserved.")
        return previous, False
    if len(ledger.forecasts(mode=mode, protocol_id=protocol["protocol_id"])) >= protocol["max_forecasts"]:
        raise ValueError("Collection budget reached; stop instead of extending the experiment.")
    contract = forecast_contract(close, calendar_name=protocol["calendar"], as_of=effective,
                                 received_at=receipt["received_at"], mode=mode,
                                 publication_delay_minutes=protocol["publication_delay_minutes"])
    settings = {"asset": protocol["asset"], "mode": mode, "protocol": protocol, "data_kind": receipt["data_kind"]}
    with tracked_run(output_root, "shadow_record", settings) as (run, logger):
        inferred = run_saved_forecast(model_run, Path(snapshot_run) / "close.csv", asset=protocol["asset"], output_root=output_root)
        result = json.loads((inferred / "forecast.json").read_text(encoding="utf-8"))
        record = {
            **contract, "protocol": protocol, "asset": protocol["asset"], "data_kind": receipt["data_kind"],
            "model_data_kind": data["data_kind"], "model_sha256": protocol["model_sha256"],
            "input_sha256": receipt["input_sha256"], "snapshot_run": str(Path(snapshot_run).resolve()),
            "forecast_run": str(inferred), "origin_close": result["origin_close"],
            "predicted_log_return": result["predicted_log_return"], "baseline_log_return": 0.0,
            "model_status": "research_candidate_not_promoted",
        }
        recorded, created = ledger.record(record)
        write_json(run / "forecast_receipt.json", {"created": created, "forecast": recorded})
        logger.info("Forecast %s; created=%s; mode=%s", recorded["forecast_id"], created, mode)
    return recorded, created


def score_snapshot(snapshot_run, *, asset, database, mode, as_of=None, output_root="log", protocol_id=None):
    _outside_sources((snapshot_run,), database=database, output_root=output_root, run_name="shadow_score")
    effective = _effective(mode, as_of)
    close, receipt = load_snapshot(snapshot_run, asset)
    if mode == "live" and receipt["data_kind"] == "synthetic":
        raise ValueError("Synthetic labels cannot score live forecasts.")
    if not Path(database).is_file():
        raise ValueError("Ledger does not exist; record forecasts before scoring.")
    ledger = ForecastLedger(database)
    settings = {"asset": asset, "mode": mode, "snapshot_run": str(snapshot_run), "data_kind": receipt["data_kind"]}
    with tracked_run(output_root, "shadow_score", settings) as (run, logger):
        outcomes = []
        for forecast in ledger.forecasts(mode=mode, protocol_id=protocol_id, asset=asset):
            if forecast["label"] is not None:
                outcomes.append({"forecast_id": forecast["forecast_id"], "status": "already_scored"})
                continue
            ready = utc_timestamp(forecast["target_close_at"]) + pd.Timedelta(minutes=forecast["publication_delay_minutes"])
            if utc_timestamp(effective) < ready or pd.Timestamp(forecast["target_session"], tz="UTC") not in close.index:
                outcomes.append({"forecast_id": forecast["forecast_id"], "status": "pending_label"})
                continue
            actual = label_contract(close, forecast, received_at=receipt["received_at"], as_of=effective)
            _, created = ledger.score(forecast["forecast_id"], actual_close=actual, received_at=receipt["received_at"],
                                      input_sha256=receipt["input_sha256"], scored_as_of=effective)
            outcomes.append({"forecast_id": forecast["forecast_id"], "status": "scored" if created else "already_scored"})
        write_json(run / "score_receipt.json", outcomes)
        write_json(run / "report.json", shadow_report(database, mode=mode, protocol_id=protocol_id, as_of=effective))
        logger.info("Processed %d forecasts; mode=%s", len(outcomes), mode)
    return run


def shadow_report(database, *, mode, protocol_id=None, as_of=None):
    import exchange_calendars as xcals

    if not Path(database).is_file():
        raise ValueError("Ledger does not exist.")
    ledger = ForecastLedger(database)
    report = ledger.report(mode=mode, protocol_id=protocol_id, as_of=as_of or _now())
    for group in report["groups"]:
        rows = ledger.forecasts(mode=mode, protocol_id=group["protocol_id"], asset=group["asset"])
        protocol = rows[0]["protocol"]
        group["budget"] = protocol["max_forecasts"]
        group["minimum_practical_rmse_skill"] = protocol["minimum_practical_rmse_skill"]
        group["statistical_power_calibrated"] = False
        first, last = protocol["earliest_origin"], rows[-1]["origin_session"]
        calendar = xcals.get_calendar(protocol["calendar"], start=first, end=(pd.Timestamp(last) + pd.Timedelta(days=2)).date())
        expected = {d.date().isoformat() for d in calendar.sessions_in_range(first, last)}
        observed = {row["origin_session"] for row in rows}
        group["missing_origins"] = sorted(expected - observed)
        group["collection_complete_through_last_origin"] = not group["missing_origins"]
        if group["forecasts"] < group["budget"]:
            group["decision"] = "collecting_within_fixed_budget"
        elif group["pending"]:
            group["decision"] = "budget_closed_waiting_for_labels"
        elif group["missing_origins"]:
            group["decision"] = "incomplete_collection_no_confirmation"
        elif group["rmse_skill_vs_zero"] is None or group["rmse_skill_vs_zero"] < protocol["minimum_practical_rmse_skill"]:
            group["decision"] = "stop_candidate_retain_baseline"
        else:
            group["decision"] = "independent_review_required_no_promotion"
    return report
