"""Validated deployment definitions and calendar-aware daily due times."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path

import exchange_calendars as xcals
import pandas as pd

from .artifacts import verify_run
from .experiment import sha256_file
from .operations import fingerprint
from .session_contract import utc_timestamp
from .shadow import _outside_sources, _protocol

_RUNTIME_SOURCE = fingerprint({p.name: p.read_text(encoding="utf-8") for p in sorted(Path(__file__).parent.glob("*.py"))})


@dataclass(frozen=True)
class Stream:
    asset: str
    model_run: str
    protocol_file: str
    protocol: dict
    protocol_sha256: str


@dataclass(frozen=True)
class BatchConfig:
    deployment_id: str
    mode: str
    database: str
    state_database: str
    output_root: str
    streams: tuple[Stream, ...]
    acquisition_start: str = "2015-01-01"
    max_attempts: int = 3
    retry_delay_seconds: int = 5
    lease_minutes: int = 15
    schedule_delay_minutes: int = 30
    completion_grace_minutes: int = 90
    heartbeat_timeout_minutes: int = 45
    runtime_source_sha256: str = field(default_factory=lambda: _RUNTIME_SOURCE)

    def definition(self):
        values = asdict(self)
        values["streams"] = list(values["streams"])
        return {"schema_version": 1, **values}


def load_config(path):
    root = Path(path).resolve()
    return parse_config(json.loads(root.read_text(encoding="utf-8")), base=root.parent)


def parse_config(raw, *, base):
    """Resolve paths relative to config, never the process's working directory."""
    allowed = {f for f in BatchConfig.__dataclass_fields__} | {"schema_version"}
    if raw.get("schema_version") != 1 or set(raw) - allowed:
        raise ValueError("Unsupported batch configuration schema or unknown setting.")
    if not re.fullmatch(r"[a-zA-Z0-9_.-]{1,100}", raw.get("deployment_id", "")) or raw.get("mode") not in {"live", "replay"}:
        raise ValueError("An explicit deployment_id and live/replay mode are required.")

    def resolve(value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Paths must be nonempty strings.")
        return str((Path(base) / value).resolve())

    settings = {k: v for k, v in raw.items() if k not in {"schema_version", "streams"}}
    for name in ("database", "state_database", "output_root"):
        settings[name] = resolve(raw[name])
    if settings["database"] == settings["state_database"]:
        raise ValueError("Forecast and operations databases must be separate.")
    if not isinstance(raw.get("streams"), list) or not 1 <= len(raw["streams"]) <= 20:
        raise ValueError("A batch needs between 1 and 20 explicit streams.")
    streams = []
    for item in raw["streams"]:
        if set(item) - {"asset", "model_run", "protocol_file", "protocol", "protocol_sha256"}:
            raise ValueError("Unknown stream setting.")
        asset = item["asset"]
        if not isinstance(asset, str) or not re.fullmatch(r"[A-Z0-9][A-Z0-9._-]{0,30}", asset):
            raise ValueError("Invalid stream asset identifier.")
        model, protocol_file = resolve(item["model_run"]), resolve(item["protocol_file"])
        protocol = _protocol(protocol_file)
        verified = verify_run(model)
        if protocol["asset"] != asset or verified["manifest_sha256"] != protocol["model_manifest_sha256"]:
            raise ValueError("Stream asset/model does not match its pinned protocol.")
        if sha256_file(Path(model) / "artifacts/model.json") != protocol["model_sha256"]:
            raise ValueError("Protocol model checksum mismatch.")
        digest = sha256_file(protocol_file)
        if ("protocol" in item and item["protocol"] != protocol) or ("protocol_sha256" in item and item["protocol_sha256"] != digest):
            raise ValueError("Registered protocol changed on disk.")
        if raw["mode"] == "live":
            data = json.loads((Path(model) / "data_manifest.json").read_text(encoding="utf-8"))
            if data["data_kind"] == "synthetic":
                raise ValueError("Synthetic models cannot be deployed in live mode.")
        for mutable in (settings["database"], settings["state_database"]):
            _outside_sources((model,), database=mutable)
        _outside_sources((model,), output_root=settings["output_root"], run_name="daily_batch")
        streams.append(Stream(asset, model, protocol_file, protocol, digest))
    if len({s.asset for s in streams}) != len(streams) or len({s.protocol["protocol_id"] for s in streams}) != len(streams):
        raise ValueError("Assets and protocol IDs must be unique within a deployment.")
    config = BatchConfig(**settings, streams=tuple(streams))
    if config.runtime_source_sha256 != _RUNTIME_SOURCE:
        raise ValueError("Runtime source differs from the pinned deployment; install the matching release or register a new deployment.")
    bounds = {
        "max_attempts": (1, 5), "retry_delay_seconds": (0, 30), "lease_minutes": (5, 60),
        "schedule_delay_minutes": (0, 240), "completion_grace_minutes": (1, 720),
        "heartbeat_timeout_minutes": (5, 1440),
    }
    for name, (low, high) in bounds.items():
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer between {low} and {high}.")
    if config.schedule_delay_minutes < max(s.protocol["publication_delay_minutes"] for s in streams):
        raise ValueError("Schedule delay cannot precede the publication buffer.")
    start = pd.Timestamp(config.acquisition_start)
    if pd.isna(start) or start.tzinfo is not None or start != start.normalize():
        raise ValueError("acquisition_start must be a date.")
    if any(start >= pd.Timestamp(s.protocol["earliest_origin"]) for s in streams):
        raise ValueError("Acquisition history must start before the first origin.")
    return config


def expected_slots(config, stream, now):
    """Mature origins since registration, capped by the frozen collection window."""
    instant = utc_timestamp(now)
    first = pd.Timestamp(stream.protocol["earliest_origin"])
    last = instant.tz_localize(None).normalize()
    calendar = xcals.get_calendar(stream.protocol["calendar"], start=(min(first, last) - timedelta(days=10)).date(),
                                  end=(max(first, last) + timedelta(days=20)).date())
    if first > last:
        return []
    sessions = calendar.sessions_in_range(first, last)[:stream.protocol["max_forecasts"]]
    result = []
    for origin in sessions:
        due = calendar.session_close(origin) + timedelta(minutes=config.schedule_delay_minutes)
        if due > instant:
            continue
        target = calendar.next_session(origin)
        result.append({
            "origin": origin.date().isoformat(), "due_at": due.isoformat(),
            "deadline_at": min(due + timedelta(minutes=config.completion_grace_minutes), calendar.session_open(target)).isoformat(),
            "target_open_at": calendar.session_open(target).isoformat(), "target_session": target.date().isoformat(),
        })
    return result


def current_slot(config, stream, now):
    slot = latest_market_slot(config, stream, now)
    if slot["origin"] < stream.protocol["earliest_origin"]:
        return {**slot, "status": "not_due"}
    if slot["origin"] not in {s["origin"] for s in expected_slots(config, stream, now)}:
        return {**slot, "status": "collection_closed"}
    if utc_timestamp(now) >= utc_timestamp(slot["target_open_at"]):
        return {**slot, "status": "window_closed"}
    return {**slot, "status": "due"}


def latest_market_slot(config, stream, now):
    """Latest completed daily source session, including after collection ends."""
    instant = utc_timestamp(now)
    day = instant.tz_localize(None).normalize()
    calendar = xcals.get_calendar(stream.protocol["calendar"], start=(day - timedelta(days=20)).date(),
                                  end=(day + timedelta(days=20)).date())
    origin = calendar.date_to_session(day, direction="previous")
    due = calendar.session_close(origin) + timedelta(minutes=config.schedule_delay_minutes)
    if instant < due:
        origin = calendar.previous_session(origin)
        due = calendar.session_close(origin) + timedelta(minutes=config.schedule_delay_minutes)
    target = calendar.next_session(origin)
    return {"origin": origin.date().isoformat(), "due_at": due.isoformat(),
            "deadline_at": min(due + timedelta(minutes=config.completion_grace_minutes), calendar.session_open(target)).isoformat(),
            "target_open_at": calendar.session_open(target).isoformat(), "target_session": target.date().isoformat()}
