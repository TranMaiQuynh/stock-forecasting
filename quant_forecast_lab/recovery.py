"""Single-database consistent backups and validated operational release changes."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from .batch_config import load_config, parse_config
from .experiment import sha256_file, write_json
from .operations import OperationsStore
from .shadow import _now


def activate_config(path, *, reason):
    config = load_config(path)
    state = OperationsStore(config.state_database)
    state.register(config.definition())
    return state.activate(config.deployment_id, now=_now(), reason=reason)


def active_config(state_database, mode):
    state = OperationsStore(state_database, create=False)
    active = state.active(mode)
    if active is None:
        raise ValueError("No active operational deployment for this mode.")
    config = parse_config(active["definition"], base=Path.cwd())
    if Path(config.state_database).resolve() != Path(state_database).resolve():
        raise ValueError("Active deployment points at a different operations store.")
    return config


def rollback(state_database, mode, *, reason):
    previous = OperationsStore(state_database, create=False).previous(mode)
    if previous is None:
        raise ValueError("No previous deployment is available for rollback.")
    # Verify artifacts and protocol before changing the pointer. No history rewrite.
    config = parse_config(previous["definition"], base=Path.cwd())
    if Path(config.state_database).resolve() != Path(state_database).resolve():
        raise ValueError("Rollback definition belongs to another operations store.")
    return OperationsStore(state_database).activate(config.deployment_id, now=_now(), reason=reason)


def copy_database(source, destination, *, operation="backup"):
    """Copy a SQLite snapshot to a new path using SQLite's online backup API.

    Does not overwrite a live database. Copying multiple DBs is not one atomic
    snapshot; pause workers for a coordinated backup of state, ledger and files.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    receipt = destination.with_suffix(destination.suffix + ".receipt.json")
    if not source.is_file() or destination.exists() or receipt.exists():
        raise ValueError("Source must exist; backup/restore destination and receipt must be new.")
    for parent in destination.parents:
        manifest = parent / "manifest.json"
        if manifest.is_file():
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
            if metadata.get("schema_version") == 1 and metadata.get("status") != "running":
                raise ValueError("Backup/restore output must be outside finalized artifact runs.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as incoming:
        version = incoming.execute("PRAGMA user_version").fetchone()[0]
        tables = {r[0] for r in incoming.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if version != 1 or not tables.intersection({"forecasts", "ops_deployments"}):
            raise ValueError("Expected a supported forecast or operations database.")
        # Exclusive file creation protects existing user data, including races.
        with destination.open("xb"):
            pass
        try:
            with closing(sqlite3.connect(destination)) as outgoing:
                incoming.backup(outgoing)
                if outgoing.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise ValueError("Backup failed SQLite integrity_check.")
                if outgoing.execute("PRAGMA foreign_key_check").fetchall():
                    raise ValueError("Backup has foreign-key violations.")
                known = {"forecasts", "labels", "protocols", "ops_deployments", "ops_jobs", "ops_events", "ops_active", "ops_activations"}
                counts = {table: outgoing.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] for table in sorted(tables & known)}
        except BaseException as error:
            write_json(receipt, {"status": "failed", "error_type": type(error).__name__, "error": str(error)})
            raise
    result = {"schema_version": 1, "status": "completed", "operation": operation, "source": str(source),
              "destination": str(destination), "recorded_at": _now(), "sha256": sha256_file(destination), "counts": counts,
              "consistency_scope": "one SQLite snapshot; cross-database/file consistency requires paused workers"}
    write_json(receipt, result)
    return result


def restore_database(backup, destination):
    backup = Path(backup).resolve()
    receipt = json.loads(backup.with_suffix(backup.suffix + ".receipt.json").read_text(encoding="utf-8"))
    if receipt.get("status") != "completed" or sha256_file(backup) != receipt.get("sha256"):
        raise ValueError("Backup receipt/checksum verification failed.")
    return copy_database(backup, destination, operation="restore_to_new_path")
