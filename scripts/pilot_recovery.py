"""Capture and drill a local serving backup; does not activate relocated stores.

Pause the runner/watchdog first. Persistent read connections detect intervening
commits to either DB and refuse to finalize a changing capture. This is a local
recovery package, not offsite storage or an automatic cross-host deployment.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sqlite3
import sys
from contextlib import ExitStack, closing
from pathlib import Path

TABLES = {"protocols", "forecasts", "labels", "ops_deployments", "ops_jobs", "ops_events", "ops_active", "ops_activations"}


def database_contents(connection):
    from quant_forecast_lab.operations import fingerprint

    if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok" or connection.execute("PRAGMA foreign_key_check").fetchall():
        raise ValueError("Recovery database has integrity/foreign-key errors.")
    tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if connection.execute("PRAGMA user_version").fetchone()[0] != 1 or not tables.intersection({"forecasts", "ops_deployments"}):
        raise ValueError("Expected a version-1 pilot database.")
    result = {}
    for table in sorted(tables & TABLES):
        rows = sorted(connection.execute(f'SELECT * FROM "{table}"').fetchall())
        result[table] = {"rows": len(rows), "sha256": fingerprint(rows)}
    return result


def connect(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=10)


def make_backup(state_database, mode, output_root):
    import quant_forecast_lab
    from quant_forecast_lab.artifacts import verify_run
    from quant_forecast_lab.batch_config import _RUNTIME_SOURCE
    from quant_forecast_lab.experiment import sha256_file, tracked_run, write_json
    from quant_forecast_lab.operations import fingerprint
    from quant_forecast_lab.recovery import active_config, copy_database

    config = active_config(state_database, mode)
    with ExitStack() as stack:
        ledger = stack.enter_context(closing(connect(config.database)))
        operations = stack.enter_context(closing(connect(config.state_database)))
        connections = {"forecasts": ledger, "operations": operations}
        versions = {k: c.execute("PRAGMA data_version").fetchone()[0] for k, c in connections.items()}
        contents = {k: database_contents(c) for k, c in connections.items()}
        jobs = [json.loads(r[0]) for r in operations.execute("SELECT payload FROM ops_jobs")]
        if any(j["status"] == "running" for j in jobs):
            raise ValueError("A job is running; quiesce workers and resolve leases before capture.")
        definitions = [json.loads(r[0]) for r in operations.execute("SELECT definition FROM ops_deployments")]
        rows = [json.loads(r[0]) for r in ledger.execute("SELECT payload FROM forecasts")]
        roots = set()
        protocols, models = {}, {}
        for definition in definitions:
            for stream in definition["streams"]:
                roots.add(stream["model_run"])
                protocols[stream["protocol_file"]] = stream["protocol_sha256"]
                models[stream["protocol"]["protocol_id"]] = stream["model_run"]
        for row in rows:
            roots.update((row["snapshot_run"], row["forecast_run"]))
        for job in jobs:
            for stage in job["stages"].values():
                for field in ("snapshot_run", "score_run"):
                    if field in stage:
                        roots.add(stage[field])
        # Score configs identify the later snapshots that supplied labels.
        for root in list(roots):
            settings = json.loads((Path(root) / "config.json").read_text(encoding="utf-8"))
            if settings.get("snapshot_run"):
                roots.add(settings["snapshot_run"])
        available_inputs = {sha256_file(Path(p) / "close.csv") for p in roots if (Path(p) / "close.csv").is_file()}
        for row in rows:
            if (sha256_file(Path(row["snapshot_run"]) / "close.csv") != row["input_sha256"]
                    or sha256_file(Path(models[row["protocol"]["protocol_id"]]) / "artifacts/model.json") != row["model_sha256"]):
                raise ValueError("Forecast input/model hash does not match its serving sources.")
        for payload, in ledger.execute("SELECT payload FROM labels"):
            if json.loads(payload)["input_sha256"] not in available_inputs:
                raise ValueError("A label source is missing from job/snapshot lineage; refusing an incomplete serving backup.")
        destination_root = Path(output_root).resolve() / "pilot_backup"
        if any(destination_root.is_relative_to(Path(p).resolve()) for p in roots):
            raise ValueError("Recovery output must be outside source artifact directories.")
        latest = {}
        for row in sorted(rows, key=lambda r: r["origin_session"]):
            if row["mode"] == mode:
                latest[(row["asset"], row["protocol"]["protocol_id"])] = row
        settings = {"mode": mode, "state_database": str(Path(state_database).resolve()), "purpose": "local_serving_recovery"}
        with tracked_run(output_root, "pilot_backup", settings) as (run, logger):
            mapping = {}
            for i, source in enumerate(sorted(roots)):
                verify_run(source)
                relative = f"sources/{i:04d}"
                shutil.copytree(source, run / relative)
                verify_run(run / relative)
                mapping[source] = relative
            for i, (source, digest) in enumerate(sorted(protocols.items())):
                if sha256_file(source) != digest:
                    raise ValueError("Registered protocol checksum changed.")
                target = run / f"protocols/{i:04d}.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                if sha256_file(target) != digest:
                    raise ValueError("Copied protocol checksum differs.")
            for name, source in (("forecasts", config.database), ("operations", config.state_database)):
                copy_database(source, run / f"databases/{name}.sqlite")
                with closing(connect(run / f"databases/{name}.sqlite")) as captured:
                    if database_contents(captured) != contents[name]:
                        raise ValueError("Captured database differs from the initial logical state.")
            package = Path(quant_forecast_lab.__file__).parent
            code = {p.name: p.read_text(encoding="utf-8") for p in sorted(package.glob("*.py"))}
            if fingerprint(code) != _RUNTIME_SOURCE or _RUNTIME_SOURCE != config.runtime_source_sha256:
                raise ValueError("Installed runtime differs from active source pin.")
            for name in code:
                target = run / "runtime/quant_forecast_lab" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(package / name, target)
            copied_code = {p.name: p.read_text(encoding="utf-8") for p in sorted((run / "runtime/quant_forecast_lab").glob("*.py"))}
            if fingerprint(copied_code) != _RUNTIME_SOURCE:
                raise ValueError("Copied runtime source changed during capture.")
            repository = Path(__file__).resolve().parents[1]
            for name in ("uv.lock", "pyproject.toml", "LICENSE"):
                shutil.copyfile(repository / name, run / "runtime" / name)
            write_json(run / "active_definition.json", config.definition())
            probes = [{"asset": r["asset"], "model": mapping[models[r["protocol"]["protocol_id"]]],
                       "snapshot": mapping[r["snapshot_run"]], "expected_log_return": r["predicted_log_return"]}
                      for r in latest.values()]
            receipt = {"schema_version": 1, "database_contents": contents, "sources": mapping,
                       "inference_probes": probes, "active_runtime_source_sha256": _RUNTIME_SOURCE,
                       "limitations": "Local snapshot only; absolute ledger paths retained. No relocated live activation, dependency wheels, OS tasks, offsite backup or full-host recovery."}
            write_json(run / "recovery_receipt.json", receipt)
            if any(c.execute("PRAGMA data_version").fetchone()[0] != versions[k] for k, c in connections.items()):
                raise ValueError("A source database changed during capture; pause workers and retry into a new run.")
            logger.info("Captured %d verified serving sources with unchanged source database versions", len(mapping))
    verify_run(run)
    return run


def drill_backup(backup, output_root):
    from quant_forecast_lab.artifacts import verify_run
    from quant_forecast_lab.experiment import tracked_run, write_json
    from quant_forecast_lab.forecast import run_saved_forecast
    from quant_forecast_lab.recovery import restore_database

    backup = Path(backup).resolve()
    verified = verify_run(backup)
    if verified["model_name"] != "pilot_backup":
        raise ValueError("Expected a verified pilot_backup.")
    if (Path(output_root).resolve() / "pilot_recovery_drill").is_relative_to(backup):
        raise ValueError("Drill output must be outside the finalized backup.")
    receipt = json.loads((backup / "recovery_receipt.json").read_text(encoding="utf-8"))
    import re
    sources = set(receipt["sources"].values())
    if (set(receipt["database_contents"]) != {"forecasts", "operations"}
            or any(not re.fullmatch(r"sources/[0-9]{4,}", name) for name in sources)
            or any(p["model"] not in sources or p["snapshot"] not in sources for p in receipt["inference_probes"])):
        raise ValueError("Invalid recovery mapping; all outputs must remain within the new drill directory.")
    with tracked_run(output_root, "pilot_recovery_drill", {"backup": str(backup), "prospective_forecasts": "not_created"}) as (run, logger):
        for name, expected in receipt["database_contents"].items():
            target = run / f"databases/{name}.sqlite"
            restore_database(backup / f"databases/{name}.sqlite", target)
            with closing(connect(target)) as recovered:
                if database_contents(recovered) != expected:
                    raise ValueError("Recovered logical database contents differ.")
        for relative in receipt["sources"].values():
            shutil.copytree(backup / relative, run / relative)
            verify_run(run / relative)
        probes = []
        for probe in receipt["inference_probes"]:
            predicted_run = run_saved_forecast(run / probe["model"], run / probe["snapshot"] / "close.csv",
                                              asset=probe["asset"], output_root=run / "inference")
            verify_run(predicted_run)
            value = json.loads((predicted_run / "forecast.json").read_text(encoding="utf-8"))["predicted_log_return"]
            if not math.isclose(value, probe["expected_log_return"], rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError("Recovered offline inference differs from the original forecast.")
            probes.append({"asset": probe["asset"], "predicted_log_return": value, "matches": True})
        write_json(run / "drill_receipt.json", {"database_contents": receipt["database_contents"], "probes": probes,
                                               "live_database_modified": False, "relocated_activation": "not_tested",
                                               "scope": "Logical DB recovery, copied serving sources and offline numerical inference; not full-host failover."})
        logger.info("Restored both logical databases and matched %d offline inference probes", len(probes))
    verify_run(run)
    return run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed-package", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)
    backup = commands.add_parser("backup")
    backup.add_argument("--state-db", required=True)
    backup.add_argument("--mode", choices=["live", "replay"], required=True)
    backup.add_argument("--output-root", default="log")
    drill = commands.add_parser("drill")
    drill.add_argument("--backup", required=True)
    drill.add_argument("--output-root", default="log")
    args = parser.parse_args(argv)
    if not args.installed_package:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    if args.command == "backup":
        run = make_backup(args.state_db, args.mode, args.output_root)
    else:
        run = drill_backup(args.backup, args.output_root)
    print(json.dumps({"status": "completed", "operation": args.command, "run_directory": str(run)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
