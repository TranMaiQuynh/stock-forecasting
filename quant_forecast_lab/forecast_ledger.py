"""Transactional, append-only local records; not an external timestamp authority."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_encode(value).encode()).hexdigest()


def _instant(value):
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        raise ValueError("Ledger instants must include a timezone.")
    return stamp.astimezone(UTC)


class ForecastLedger:
    """One forecast per mode/protocol/asset/origin; first label vintage is retained.

    Transactions serialize duplicate writers. Conflicting retries fail explicitly.
    Triggers guard normal SQL edits, not an administrator replacing the database.
    """

    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection, connection:
            current = connection.execute("PRAGMA user_version").fetchone()[0]
            if current not in {0, 1}:
                raise ValueError("Unsupported forecast ledger schema version.")
            if current == 1:
                return
            if current == 0 and connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchone():
                raise ValueError("Refusing to initialize an unrelated database.")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS protocols (
                    protocol_id TEXT PRIMARY KEY, definition TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS forecasts (
                    forecast_id TEXT PRIMARY KEY, mode TEXT NOT NULL, protocol_id TEXT NOT NULL,
                    asset TEXT NOT NULL, origin_session TEXT NOT NULL, payload TEXT NOT NULL,
                    semantic_hash TEXT NOT NULL,
                    FOREIGN KEY(protocol_id) REFERENCES protocols(protocol_id),
                    UNIQUE(mode, protocol_id, asset, origin_session));
                CREATE TABLE IF NOT EXISTS labels (
                    forecast_id TEXT PRIMARY KEY REFERENCES forecasts(forecast_id), payload TEXT NOT NULL);
                PRAGMA user_version=1;
            """)
            for table in ("protocols", "forecasts", "labels"):
                for operation in ("UPDATE", "DELETE"):
                    connection.execute(
                        f"CREATE TRIGGER IF NOT EXISTS {table}_{operation.lower()}_guard "
                        f"BEFORE {operation} ON {table} BEGIN SELECT RAISE(ABORT, 'append-only ledger'); END"
                    )

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def find(self, mode, protocol_id, asset, origin_session):
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT payload FROM forecasts WHERE mode=? AND protocol_id=? AND asset=? AND origin_session=?",
                (mode, protocol_id, asset, origin_session),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def record(self, forecast):
        record = dict(forecast)
        protocol = record["protocol"]
        protocol_id = protocol["protocol_id"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", protocol_id):
            raise ValueError("Invalid protocol_id.")
        if record["mode"] not in {"live", "replay"} or record["target_session"] <= record["origin_session"]:
            raise ValueError("Invalid forecast mode or session ordering.")
        if not math.isfinite(record["predicted_log_return"]) or not math.isfinite(record["origin_close"]) or record["origin_close"] <= 0:
            raise ValueError("Forecast values must be finite with a positive origin close.")
        for name in ("model_sha256", "input_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", record[name]):
                raise ValueError(f"Invalid {name}.")
        # New receipt/run timestamps do not make an otherwise identical retry new.
        semantic = {k: v for k, v in record.items() if k not in {
            "as_of", "received_at", "snapshot_run", "forecast_run", "recorded_at", "forecast_id"
        }}
        semantic_hash = _digest(semantic)
        identity = [record["mode"], protocol_id, record["asset"], record["origin_session"]]
        record.update(forecast_id=_digest(identity), recorded_at=datetime.now(UTC).isoformat())
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            definition = _encode(protocol)
            previous = connection.execute("SELECT definition FROM protocols WHERE protocol_id=?", (protocol_id,)).fetchone()
            if previous and previous[0] != definition:
                raise ValueError("Protocol is frozen; changed rules require a new protocol_id.")
            connection.execute("INSERT OR IGNORE INTO protocols VALUES (?, ?)", (protocol_id, definition))
            previous = connection.execute("SELECT payload, semantic_hash FROM forecasts WHERE forecast_id=?",
                                          (record["forecast_id"],)).fetchone()
            if previous:
                if previous["semantic_hash"] != semantic_hash:
                    raise ValueError("Conflicting forecast retry; the original prediction is preserved.")
                return json.loads(previous["payload"]), False
            ready = _instant(record["origin_close_at"]) + timedelta(minutes=record["publication_delay_minutes"])
            effective = _instant(record["as_of"])
            if not ready <= effective < _instant(record["target_open_at"]):
                raise ValueError("Forecast must precede target open and follow origin availability.")
            if record["mode"] == "live" and not (
                ready <= _instant(record["received_at"]) <= effective <= _instant(record["recorded_at"])
                < _instant(record["target_open_at"])
            ):
                raise ValueError("Live forecast cannot be backdated past its actual recording deadline.")
            connection.execute("INSERT INTO forecasts VALUES (?, ?, ?, ?, ?, ?, ?)",
                               (record["forecast_id"], *identity, _encode(record), semantic_hash))
        return record, True

    def forecasts(self, *, mode, protocol_id=None, asset=None):
        if mode not in {"live", "replay"}:
            raise ValueError("Reports must explicitly select live or replay mode.")
        with closing(self._connect()) as connection:
            rows = connection.execute("""
                SELECT f.payload AS forecast, l.payload AS label FROM forecasts f
                LEFT JOIN labels l USING(forecast_id)
                WHERE f.mode=? AND (? IS NULL OR f.protocol_id=?) AND (? IS NULL OR f.asset=?)
                ORDER BY f.protocol_id, f.asset, f.origin_session
            """, (mode, protocol_id, protocol_id, asset, asset)).fetchall()
        return [{**json.loads(row["forecast"]), "label": json.loads(row["label"]) if row["label"] else None} for row in rows]

    def score(self, forecast_id, *, actual_close, received_at, input_sha256, scored_as_of):
        if not math.isfinite(actual_close) or actual_close <= 0:
            raise ValueError("Actual close must be finite and positive.")
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT payload FROM forecasts WHERE forecast_id=?", (forecast_id,)).fetchone()
            if row is None:
                raise ValueError("Unknown forecast_id.")
            forecast = json.loads(row[0])
            ready = _instant(forecast["target_close_at"]) + timedelta(minutes=forecast["publication_delay_minutes"])
            if _instant(scored_as_of) < max(ready, _instant(forecast["as_of"])):
                raise ValueError("Target label cannot be scored before availability.")
            if forecast["mode"] == "live" and not ready <= _instant(received_at) <= _instant(scored_as_of) <= datetime.now(UTC):
                raise ValueError("Invalid live label receipt/scoring time.")
            if not re.fullmatch(r"[0-9a-f]{64}", input_sha256):
                raise ValueError("Invalid label input_sha256.")
            label = {
                "actual_close": actual_close,
                "actual_log_return": math.log(actual_close) - math.log(forecast["origin_close"]),
                "received_at": received_at, "input_sha256": input_sha256,
                "scored_as_of": scored_as_of, "recorded_at": datetime.now(UTC).isoformat(),
            }
            previous = connection.execute("SELECT payload FROM labels WHERE forecast_id=?", (forecast_id,)).fetchone()
            if previous:
                if json.loads(previous[0])["actual_close"] != actual_close:
                    raise ValueError("Conflicting label vintage; the first label is preserved.")
                return json.loads(previous[0]), False
            connection.execute("INSERT INTO labels VALUES (?, ?)", (forecast_id, _encode(label)))
        return label, True

    def report(self, *, mode, protocol_id=None, as_of=None):
        """Descriptive matched-target errors; neither significance nor profit."""
        groups = {}
        for row in self.forecasts(mode=mode, protocol_id=protocol_id):
            key = (row["protocol"]["protocol_id"], row["asset"])
            groups.setdefault(key, []).append(row)
        result = []
        for (protocol, asset), rows in groups.items():
            paired = [r for r in rows if r["label"] is not None]
            actual = [r["label"]["actual_log_return"] for r in paired]
            errors = [r["predicted_log_return"] - y for r, y in zip(paired, actual, strict=True)]
            model_rmse = math.sqrt(sum(e * e for e in errors) / len(errors)) if errors else None
            zero_rmse = math.sqrt(sum(y * y for y in actual) / len(actual)) if actual else None
            overdue = sum(r["label"] is None and as_of is not None and
                          _instant(r["target_close_at"]) + timedelta(minutes=r["publication_delay_minutes"])
                          <= _instant(as_of) for r in rows)
            result.append({
                "protocol_id": protocol, "asset": asset, "forecasts": len(rows), "scored": len(paired),
                "pending": len(rows) - len(paired), "overdue_unscored": overdue,
                "rmse": model_rmse, "mae": sum(abs(e) for e in errors) / len(errors) if errors else None,
                "zero_rmse": zero_rmse,
                "rmse_skill_vs_zero": 1 - model_rmse / zero_rmse if zero_rmse else None,
                "promotion": "disabled", "mode": mode,
            })
        return {"schema_version": 1, "mode": mode, "groups": result,
                "interpretation": "Descriptive paired errors; replay is not prospective evidence; no trading or significance claim."}
