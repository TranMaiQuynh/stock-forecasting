"""Read-only integrity checks for completed tracked-run directories."""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath

from .experiment import sha256_file


def verify_run(directory):
    """Verify the exact recorded file inventory and checksums, not source authenticity or model skill."""
    root = Path(directory).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("status") != "completed":
        raise ValueError("Only completed schema-1 runs can be verified for inference.")
    artifacts = manifest.get("artifact_sha256")
    if not isinstance(artifacts, dict) or not {"config.json", "environment.json"} <= set(artifacts):
        raise ValueError("Run manifest is missing its required artifact inventory.")
    for name, expected in artifacts.items():
        relative = PurePosixPath(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in name
            or relative.as_posix() != name
            or name == "manifest.json"
            or not isinstance(expected, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected)
        ):
            raise ValueError("Run manifest contains an invalid artifact path/checksum.")
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Artifact is missing or has a checksum mismatch: {name}")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p != manifest_path}
    if actual != set(artifacts):
        raise ValueError("Run directory contains an unrecorded or missing artifact.")
    if manifest.get("config_sha256") != artifacts["config.json"]:
        raise ValueError("Run configuration hash is inconsistent with its artifact inventory.")
    environment = json.loads((root / "environment.json").read_text(encoding="utf-8"))
    return {
        "run_id": manifest["run_id"],
        "model_name": manifest["model_name"],
        "status": "completed",
        "verified_artifacts": len(artifacts),
        "manifest_sha256": sha256_file(manifest_path),
        "code_commit": environment.get("git_commit"),
    }
