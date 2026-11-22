"""Offline synthetic incident drill for daily jobs, monitoring, rollback and restore."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pandas as pd


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="log")
    parser.add_argument("--installed-package", action="store_true", help="Use an already installed wheel instead of adding the source root.")
    args = parser.parse_args(argv)
    if not args.installed_package:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import demo_shadow

    import quant_forecast_lab
    import quant_forecast_lab.daily as daily
    from quant_forecast_lab.acceptance import evaluate_acceptance
    from quant_forecast_lab.artifacts import verify_run
    from quant_forecast_lab.batch_config import parse_config
    from quant_forecast_lab.daily import BatchFailure, run_daily
    from quant_forecast_lab.experiment import sha256_file, tracked_run, write_json
    from quant_forecast_lab.forecast_ledger import ForecastLedger
    from quant_forecast_lab.health import check_health, write_health
    from quant_forecast_lab.market_data import MarketDataUnavailable
    from quant_forecast_lab.operations import OperationsStore, fingerprint
    from quant_forecast_lab.recovery import copy_database, restore_database, rollback

    if args.installed_package:
        source_root = Path(__file__).resolve().parents[1] / "quant_forecast_lab"
        installed_root = Path(quant_forecast_lab.__file__).parent
        source_hash = fingerprint({p.name: p.read_text(encoding="utf-8") for p in sorted(source_root.glob("*.py"))})
        installed_hash = fingerprint({p.name: p.read_text(encoding="utf-8") for p in sorted(installed_root.glob("*.py"))})
        if source_hash != installed_hash:
            raise RuntimeError("Installed wheel differs from this source release; rebuild before running the demo.")

    settings = {"mode": "replay", "data_kind": "synthetic", "simulated_incidents": True, "market_evidence": False}
    with tracked_run(args.output_root, "batch_demo", settings) as (run, logger):
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            demo_shadow.main(["--output-root", str(run / "sources")])
        source = Path(json.loads(captured.getvalue())["demo_run"])
        prepared = json.loads((source / "receipt.json").read_text())
        protocol = json.loads((source / "protocol.json").read_text())
        snapshots = sorted((source / "observed_snapshot").iterdir())
        config = parse_config({
            "schema_version": 1, "deployment_id": "synthetic-batch-v1", "mode": "replay",
            "database": "forecasts.sqlite", "state_database": "operations.sqlite", "output_root": "runs",
            "retry_delay_seconds": 0,
            "streams": [{"asset": "SYNTHETIC_DEMO", "model_run": prepared["model_run"], "protocol_file": str(source / "protocol.json")}],
        }, base=run)
        state = OperationsStore(config.state_database)
        state.register(config.definition())
        state.activate(config.deployment_id, now="2026-11-16T00:00:00Z", reason="synthetic incident drill")
        first = pd.Timestamp(protocol["earliest_origin"] + "T00:00:00Z")
        # The recorded source demo supplies actual exchange open/close dates.
        source_rows = ForecastLedger(source / "ledger.sqlite").forecasts(mode="replay")
        closes = [pd.Timestamp(r["origin_close_at"]) for r in source_rows]
        late = (closes[0] + pd.Timedelta(minutes=125)).isoformat()
        missing = check_health(config, as_of=late)
        if not any(a["code"] == "missing_forecasts" for a in missing["alerts"]):
            raise RuntimeError("Monitor did not detect the never-started job")
        first_run = run_daily(config, as_of=(closes[0] + pd.Timedelta(minutes=130)).isoformat(),
                              snapshots={"SYNTHETIC_DEMO": snapshots[0]})
        load, record = daily.load_snapshot, daily.record_forecast
        calls = []

        def transient(path, asset):
            calls.append(asset)
            if len(calls) == 1:
                raise MarketDataUnavailable("simulated temporary provider outage")
            return load(path, asset)

        def after_commit(*a, **kw):
            record(*a, **kw)
            raise RuntimeError("simulated failure after forecast transaction committed")

        failed_run = None
        with patch.object(daily, "load_snapshot", transient), patch.object(daily, "record_forecast", after_commit):
            try:
                run_daily(config, as_of=(closes[1] + pd.Timedelta(minutes=35)).isoformat(),
                          snapshots={"SYNTHETIC_DEMO": snapshots[1]})
            except BatchFailure as error:
                failed_run = error.run
        if failed_run is None or len(calls) != 2:
            raise RuntimeError("Fault injection did not exercise transient retry and partial commit")
        resumed = run_daily(config, as_of=(closes[1] + pd.Timedelta(minutes=40)).isoformat(),
                             snapshots={"SYNTHETIC_DEMO": snapshots[2]})
        retry = json.loads((resumed / "summary.json").read_text())["assets"][0]
        if retry["stages"]["record"]["created"]:
            raise RuntimeError("Recovery duplicated the committed forecast")
        final_time = (pd.Timestamp(source_rows[-1]["target_close_at"]) + pd.Timedelta(minutes=35)).isoformat()
        run_daily(config, as_of=final_time, snapshots={"SYNTHETIC_DEMO": snapshots[2]})
        rows = ForecastLedger(config.database).forecasts(mode="replay")
        if len(rows) != 2 or any(row["label"] is None for row in rows):
            raise RuntimeError("Final label was lost after the forecast budget closed")
        next_config = replace(config, deployment_id="synthetic-batch-v2")
        state.register(next_config.definition())
        state.activate(next_config.deployment_id, now=final_time, reason="synthetic release")
        rolled_back = rollback(config.state_database, "replay", reason="synthetic rollback drill")
        if rolled_back["deployment_id"] != config.deployment_id:
            raise RuntimeError("Rollback did not restore the previous operational definition")
        backup = run / "backups/forecasts.sqlite"
        copy_database(config.database, backup)
        recovered = run / "restored/forecasts.sqlite"
        restore_database(backup, recovered)
        if ForecastLedger(recovered).forecasts(mode="replay") != rows:
            raise RuntimeError("Restored forecast/label contents differ")
        health = check_health(config, as_of=final_time)
        acceptance = evaluate_acceptance(config, rows, as_of=final_time, required_sessions=2)
        if acceptance["status"] != "not_ready" or acceptance["consecutive_compliant_sessions"] != 1:
            raise RuntimeError("Injected late delivery was incorrectly accepted after recovery")
        write_json(run / "acceptance.json", acceptance)
        # Recovery completes the records; it cannot make the injected late
        # delivery timely. Both old and new watchdogs expose late_origins.
        historical_codes = {"late_forecasts", "late_labels"}
        if (not all(a["collection_complete"] for a in health["assets"])
                or any(a["code"] not in historical_codes for a in health["alerts"])
                or first.date().isoformat() not in health["assets"][0]["late_origins"]):
            raise RuntimeError(f"Recovered collection has unexpected failures: {health['alerts']}")
        write_health(run, health)
        checks = []
        for path in sorted(run.rglob("manifest.json")):
            if path.parent == run:
                continue
            manifest = json.loads(path.read_text())
            if manifest["status"] == "completed":
                checks.append(verify_run(path.parent))
            elif path.parent == failed_run and manifest["status"] == "failed":
                if any(sha256_file(path.parent / name) != digest for name, digest in manifest["artifact_sha256"].items()):
                    raise RuntimeError("Retained failed-run artifact changed")
            else:
                raise RuntimeError(f"Unexpected unfinished/failed run: {path.parent}")
        receipt = {**settings, "package_file": str(Path(quant_forecast_lab.__file__).resolve()),
                   "first_origin": first.date().isoformat(), "first_run": str(first_run),
                   "retained_failed_run": str(failed_run), "transient_attempts": len(calls),
                   "forecasts": 2, "scored": 2, "rollback_verified": True, "restore_verified": True,
                   "health_status": health["status"], "operational_acceptance": acceptance["status"],
                   "consecutive_compliant_sessions": acceptance["consecutive_compliant_sessions"],
                   "verified_completed_runs": len(checks), "promotion": "disabled"}
        write_json(run / "receipt.json", receipt)
        write_json(run / "verification.json", checks)
        logger.info("SYNTHETIC incident drill complete; no prospective or market-effectiveness claim")
    verified = verify_run(run)
    print(json.dumps({"demo_run": str(run), "verified_artifacts": verified["verified_artifacts"], **receipt}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
