"""Copy the five exact reference artifacts into a clean checkout, without overwriting other work."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

_FILES = {
    "model_direction.keras",
    "model_magnitude.keras",
    "model_config.json",
    "min_max_scaler.json",
    "input_scaler.json",
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def provision(source, destination, manifest_path):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Source and destination must be separate, non-nested directories.")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    files = manifest.get("required_files_sha256", {})
    if manifest.get("schema_version") != 1 or set(files) != _FILES:
        raise ValueError("Unexpected reference manifest schema/file inventory.")
    if any(not isinstance(h, str) or not re.fullmatch(r"[a-f0-9]{64}", h) for h in files.values()):
        raise ValueError("Reference manifest must contain SHA-256 digests.")
    # Complete preflight before creating or changing any destination files.
    for name, expected in files.items():
        origin = (source / name).resolve()
        if not origin.is_relative_to(source) or digest(origin) != expected:
            raise ValueError(f"Reference source checksum mismatch: {name}")
        target = (destination / name).resolve()
        if not target.is_relative_to(destination) or (target.exists() and digest(target) != expected):
            raise ValueError(f"Destination contains a different artifact: {name}")
    destination.mkdir(parents=True, exist_ok=True)
    for name, expected in files.items():
        if not (destination / name).exists():
            shutil.copy2(source / name, destination / name)
        if digest(destination / name) != expected:
            raise ValueError(f"Copied reference artifact failed checksum: {name}")
    return {"files_verified": len(files), "destination": str(destination), "manifest_sha256": digest(manifest_path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument(
        "--manifest", default=str(Path(__file__).resolve().parents[1] / "tests/fixtures/reference-v7-manifest.json")
    )
    args = parser.parse_args(argv)
    print(json.dumps(provision(args.source, args.destination, args.manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
