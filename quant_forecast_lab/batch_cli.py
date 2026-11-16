"""Operational commands; active releases change deployment pointers, not ML status."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def add_batch_commands(subparsers):
    batch = subparsers.add_parser("batch", help="Operate the scheduled daily forecasting pilot.")
    commands = batch.add_subparsers(dest="batch_command", required=True)
    configure = commands.add_parser("configure", help="Create a validated deployment from an existing candidate preparation receipt.")
    configure.add_argument("--preparation-receipt", required=True)
    configure.add_argument("--deployment-id", required=True)
    configure.add_argument("--mode", choices=["live", "replay"], required=True)
    configure.add_argument("--output", required=True)
    for name, help_text in (("preflight", "Validate artifacts and calendar config without running jobs."),
                            ("activate", "Pin an operational deployment; does not promote an ML model.")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", required=True)
        if name == "activate":
            command.add_argument("--reason", required=True)
    for name, help_text in (("run", "Run the active deployment once; resume durable stage checkpoints."),
                            ("health", "Independently check deliveries and leases; nonzero exit on alerts."),
                            ("rollback", "Validate and reactivate the previous operational deployment.")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--state-db", required=True)
        command.add_argument("--mode", choices=["live", "replay"], required=True)
        if name in {"run", "health"}:
            command.add_argument("--as-of", help="Replay-only timezone-aware simulated time.")
        if name == "run":
            command.add_argument("--snapshot", action="append", default=[], help="Replay-only ASSET=verified_snapshot_run.")
        elif name == "rollback":
            command.add_argument("--reason", required=True)
    backup = commands.add_parser("backup", help="Consistent SQLite snapshot to a new path; no file overwrite.")
    backup.add_argument("--source", required=True)
    backup.add_argument("--destination", required=True)
    restore = commands.add_parser("restore", help="Verify a backup receipt and restore to a new path.")
    restore.add_argument("--backup", required=True)
    restore.add_argument("--destination", required=True)


def run_batch_command(args):
    from .batch_config import current_slot, load_config, parse_config
    from .daily import BatchFailure, run_daily
    from .health import check_health, write_health
    from .recovery import activate_config, active_config, copy_database, restore_database, rollback
    from .shadow import _effective, _now

    command, exit_code = args.batch_command, 0
    health_failure_reported = False
    try:
        if command == "configure":
            receipt = json.loads(Path(args.preparation_receipt).read_text(encoding="utf-8"))
            destination = Path(args.output).resolve()
            streams = [{"asset": row["asset"], "model_run": row["model_run"], "protocol_file": row["protocol_file"]}
                       for row in receipt["assets"]]
            config = parse_config({"schema_version": 1, "deployment_id": args.deployment_id, "mode": args.mode,
                                   "database": "forecasts.sqlite", "state_database": "operations.sqlite",
                                   "output_root": "runs", "streams": streams}, base=destination.parent)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("x", encoding="utf-8") as handle:
                json.dump(config.definition(), handle, indent=2, allow_nan=False)
                handle.write("\n")
            result = {"status": "configured_not_activated", "config": str(destination)}
        elif command == "preflight":
            config = load_config(args.config)
            result = {"status": "validated", "deployment_id": config.deployment_id, "mode": config.mode,
                      "slots_at_real_clock": {s.asset: current_slot(config, s, _now()) for s in config.streams}}
        elif command == "activate":
            result = activate_config(args.config, reason=args.reason)
        elif command == "rollback":
            result = rollback(args.state_db, args.mode, reason=args.reason)
        elif command == "backup":
            result = copy_database(args.source, args.destination)
        elif command == "restore":
            result = restore_database(args.backup, args.destination)
        else:
            # On configuration/artifact failure, refresh the health page rather than leave stale green output.
            try:
                config = active_config(args.state_db, args.mode)
            except (ValueError, OSError, KeyError, sqlite3.Error) as error:
                if command == "health":
                    failed = {"status": "unhealthy", "mode": args.mode, "checked_as_of": _now(), "assets": [],
                              "alerts": [{"code": "active_configuration_invalid", "error": str(error)}]}
                    write_health(Path(args.state_db).resolve().parent / "health" / args.mode, failed)
                    health_failure_reported = True
                raise
            if command == "run":
                snapshots = {}
                for item in args.snapshot:
                    asset, separator, path = item.partition("=")
                    if not separator or asset in snapshots:
                        raise ValueError("Snapshots must be distinct ASSET=directory values.")
                    snapshots[asset] = path
                try:
                    run = run_daily(config, as_of=args.as_of, snapshots=snapshots or None)
                    result = {"status": "completed", "run_directory": str(run)}
                except BatchFailure as error:
                    result = {"status": "degraded", "run_directory": str(error.run), "summary": error.summary}
                    exit_code = 1
            else:
                _effective(args.mode, args.as_of)
                result = check_health(config, as_of=args.as_of)
                write_health(Path(config.state_database).parent / "health" / config.mode, result)
                exit_code = int(result["status"] == "unhealthy")
    except (ValueError, OSError, KeyError, sqlite3.Error) as error:
        result, exit_code = {"status": "failed", "error_type": type(error).__name__, "error": str(error)}, 2
        if command == "health" and not health_failure_reported:
            failed = {"status": "unhealthy", "mode": args.mode, "checked_as_of": _now(), "assets": [],
                      "alerts": [{"code": "health_check_failed", "error": str(error)}]}
            write_health(Path(args.state_db).resolve().parent / "health" / args.mode, failed)
    print(json.dumps(result, indent=2, allow_nan=False))
    return exit_code
