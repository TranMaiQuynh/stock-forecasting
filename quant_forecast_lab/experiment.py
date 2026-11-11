"""Experiment provenance helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd


def sha256_file(path) -> str:
    """Return the SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit_sha(cwd=None) -> str | None:
    """Return the current Git commit if the working tree is a Git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def runtime_metadata(cwd=None) -> dict[str, str | None]:
    """Capture versions needed to interpret a saved experiment."""

    def installed_version(name):
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "tensorflow_version": installed_version("tensorflow"),
        "sklearn_version": installed_version("scikit-learn"),
        "statsmodels_version": installed_version("statsmodels"),
        "matplotlib_version": installed_version("matplotlib"),
        "scipy_version": installed_version("scipy"),
        "git_commit": git_commit_sha(cwd=cwd),
    }


def write_json(path, value):
    """Replace a UTF-8 JSON artifact atomically and reject nonfinite numbers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def tracked_run(output_root, model_name: str, config: dict):
    """Create a unique run, retain failures, and hash artifacts after closing logs.

    Callers validate their required artifacts before leaving this context.
    Output paths are local; no tracking service or TensorFlow import is needed.
    """
    if not model_name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in model_name):
        raise ValueError("model_name must use lowercase letters, digits, underscores or hyphens.")
    started = datetime.now(UTC)
    clock_start = time.monotonic()
    run_id = started.strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid4().hex[:8]
    root = Path(output_root).resolve() / model_name / run_id
    root.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger(f"quant_forecast_lab.run.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(root / "run.log", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s")
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "model_name": model_name,
        "started_at": started.isoformat(),
        "status": "running",
    }
    try:
        write_json(root / "manifest.json", manifest)
        write_json(root / "config.json", config)
        write_json(root / "environment.json", runtime_metadata(cwd=Path(__file__).parent))
        manifest["config_sha256"] = sha256_file(root / "config.json")
        source_root = Path(__file__).parent
        manifest["source_sha256"] = {p.name: sha256_file(p) for p in sorted(source_root.glob("*.py"))}
        try:
            status = subprocess.run(
                ["git", "status", "--porcelain"], cwd=source_root, capture_output=True, text=True, check=True, timeout=5
            )
            manifest["git_dirty"] = bool(status.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            manifest["git_dirty"] = None
        write_json(root / "manifest.json", manifest)
        logger.info("Run started; data_kind=%s", config.get("data_kind", "unspecified"))
        yield root, logger
    except BaseException as exc:
        manifest.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        logger.exception("Run did not complete")
        raise
    else:
        manifest["status"] = "completed"
        logger.info("All required stages and artifact checks completed")
    finally:
        handler.close()
        logger.removeHandler(handler)
        manifest["ended_at"] = datetime.now(UTC).isoformat()
        manifest["duration_seconds"] = time.monotonic() - clock_start
        manifest["artifact_sha256"] = {
            p.relative_to(root).as_posix(): sha256_file(p)
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != "manifest.json"
        }
        write_json(root / "manifest.json", manifest)
