"""Durable local job leases/checkpoints; business forecasts remain in their ledger."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing, contextmanager
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from .session_contract import utc_timestamp


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


class OperationsStore:
    """Single-host leases with ownership fencing and immutable deployment definitions.

    SQLite transactions serialize claims. Leases do not promise exactly-once
    vendor requests; the forecast ledger independently enforces unique forecasts.
    """

    def __init__(self, path, *, create=True):
        self.path = Path(path).resolve()
        self.readonly = not create
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if tables:
                if version != 1 or "ops_deployments" not in tables:
                    raise ValueError("Unsupported or unrelated operations database.")
                return
            if not create:
                raise ValueError("Operations database has no initialized schema.")
            connection.executescript("""
                BEGIN IMMEDIATE;
                CREATE TABLE IF NOT EXISTS ops_deployments (
                    deployment_id TEXT PRIMARY KEY, definition TEXT NOT NULL,
                    last_seen TEXT, last_status TEXT, last_run TEXT);
                CREATE TABLE IF NOT EXISTS ops_jobs (
                    job_id TEXT PRIMARY KEY, deployment_id TEXT NOT NULL REFERENCES ops_deployments,
                    asset TEXT NOT NULL, origin TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS ops_events (
                    event_id INTEGER PRIMARY KEY, job_id TEXT NOT NULL, at TEXT NOT NULL,
                    kind TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS ops_active (mode TEXT PRIMARY KEY, deployment_id TEXT NOT NULL REFERENCES ops_deployments);
                CREATE TABLE IF NOT EXISTS ops_activations (
                    event_id INTEGER PRIMARY KEY, mode TEXT NOT NULL, deployment_id TEXT NOT NULL,
                    previous TEXT, at TEXT NOT NULL, reason TEXT NOT NULL);
                PRAGMA user_version=1;
                COMMIT;
            """)

    def _connect(self):
        target = self.path.as_uri() + "?mode=ro" if self.readonly else str(self.path)
        connection = sqlite3.connect(target, timeout=10, uri=self.readonly)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @contextmanager
    def _transaction(self):
        if self.readonly:
            raise ValueError("Monitor cannot mutate operations state.")
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            yield connection

    def register(self, definition):
        identity = definition["deployment_id"]
        encoded = encode(definition)
        with self._transaction() as connection:
            row = connection.execute("SELECT definition FROM ops_deployments WHERE deployment_id=?", (identity,)).fetchone()
            if row and row[0] != encoded:
                raise ValueError("Deployment definition changed; use a new deployment_id.")
            connection.execute("INSERT OR IGNORE INTO ops_deployments(deployment_id,definition) VALUES (?,?)", (identity, encoded))

    def deployment(self, identity):
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT * FROM ops_deployments WHERE deployment_id=?", (identity,)).fetchone()
        if row is None:
            return None
        return {**dict(row), "definition": json.loads(row["definition"])}

    def heartbeat(self, identity, *, now, status, run):
        utc_timestamp(now)
        with self._transaction() as connection:
            cursor = connection.execute("UPDATE ops_deployments SET last_seen=?,last_status=?,last_run=? WHERE deployment_id=?",
                                        (now, status, str(run), identity))
            if cursor.rowcount != 1:
                raise ValueError("Unknown deployment.")

    def claim(self, identity, asset, origin, *, now, lease_minutes):
        instant = utc_timestamp(now)
        if isinstance(lease_minutes, bool) or not isinstance(lease_minutes, int) or not 1 <= lease_minutes <= 60:
            raise ValueError("Lease must be between 1 and 60 minutes.")
        job_id = fingerprint([identity, asset, origin])
        with self._transaction() as connection:
            row = connection.execute("SELECT payload FROM ops_jobs WHERE job_id=?", (job_id,)).fetchone()
            job = json.loads(row[0]) if row else {"job_id": job_id, "deployment_id": identity,
                                                "asset": asset, "origin": origin, "stages": {}, "claims": 0}
            if row and (job["status"] == "succeeded" or (
                job["status"] == "running" and utc_timestamp(job["lease_until"]) > instant
            )):
                return job, False
            job.update(owner=uuid4().hex, status="running", updated_at=instant.isoformat(),
                       lease_until=(instant + timedelta(minutes=lease_minutes)).isoformat(), claims=job["claims"] + 1)
            connection.execute("INSERT INTO ops_jobs VALUES (?,?,?,?,?) ON CONFLICT(job_id) DO UPDATE SET payload=excluded.payload",
                               (job_id, identity, asset, origin, encode(job)))
            self._event(connection, job_id, now, "claimed", {"claims": job["claims"], "owner": job["owner"]})
        return job, True

    @staticmethod
    def _event(connection, job_id, now, kind, payload):
        connection.execute("INSERT INTO ops_events(job_id,at,kind,payload) VALUES (?,?,?,?)", (job_id, now, kind, encode(payload)))

    def checkpoint(self, job_id, owner, *, stage, result, now, lease_minutes=15):
        instant = utc_timestamp(now)
        with self._transaction() as connection:
            row = connection.execute("SELECT payload FROM ops_jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise ValueError("Unknown job.")
            job = json.loads(row[0])
            if job["owner"] != owner or job["status"] != "running" or utc_timestamp(job["lease_until"]) <= instant:
                raise ValueError("Lost or expired job lease; stale writer must stop.")
            old = job["stages"].get(stage)
            if old and old.get("status") == "succeeded" and old != result:
                raise ValueError("Completed stage checkpoint cannot be changed.")
            job["stages"][stage] = result
            job.update(updated_at=instant.isoformat(), lease_until=(instant + timedelta(minutes=lease_minutes)).isoformat())
            connection.execute("UPDATE ops_jobs SET payload=? WHERE job_id=?", (encode(job), job_id))
            self._event(connection, job_id, now, stage, result)
        return job

    def finish(self, job_id, owner, *, now, status):
        if status not in {"succeeded", "degraded", "failed"}:
            raise ValueError("Invalid terminal job status.")
        with self._transaction() as connection:
            row = connection.execute("SELECT payload FROM ops_jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise ValueError("Unknown job.")
            job = json.loads(row[0])
            if job["owner"] != owner or job["status"] != "running" or utc_timestamp(job["lease_until"]) <= utc_timestamp(now):
                raise ValueError("Lost or expired job lease.")
            job.update(status=status, updated_at=now, lease_until=now)
            connection.execute("UPDATE ops_jobs SET payload=? WHERE job_id=?", (encode(job), job_id))
            self._event(connection, job_id, now, status, {})
        return job

    def jobs(self, identity):
        with closing(self._connect()) as connection:
            rows = connection.execute("SELECT payload FROM ops_jobs WHERE deployment_id=? ORDER BY asset,origin", (identity,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def activate(self, identity, *, now, reason):
        with self._transaction() as connection:
            row = connection.execute("SELECT definition FROM ops_deployments WHERE deployment_id=?", (identity,)).fetchone()
            if row is None:
                raise ValueError("Unknown deployment.")
            mode = json.loads(row[0])["mode"]
            old = connection.execute("SELECT deployment_id FROM ops_active WHERE mode=?", (mode,)).fetchone()
            previous = old[0] if old else None
            connection.execute("INSERT INTO ops_active VALUES (?,?) ON CONFLICT(mode) DO UPDATE SET deployment_id=excluded.deployment_id",
                               (mode, identity))
            connection.execute("INSERT INTO ops_activations(mode,deployment_id,previous,at,reason) VALUES (?,?,?,?,?)",
                               (mode, identity, previous, now, reason))
        return {"deployment_id": identity, "previous": previous, "mode": mode, "at": now}

    def active(self, mode):
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT deployment_id FROM ops_active WHERE mode=?", (mode,)).fetchone()
        return self.deployment(row[0]) if row else None

    def previous(self, mode):
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT previous FROM ops_activations WHERE mode=? ORDER BY event_id DESC LIMIT 1", (mode,)).fetchone()
        return self.deployment(row[0]) if row and row[0] else None
