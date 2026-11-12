import hashlib
import json
import runpy
from pathlib import Path

import pytest

provision = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/provision_reference.py"))["provision"]


@pytest.mark.parametrize("defect", [None, "source", "destination", "traversal"])
def test_reference_provision_checks_exact_artifacts_before_writing(tmp_path, defect):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    names = (
        "model_direction.keras",
        "model_magnitude.keras",
        "model_config.json",
        "min_max_scaler.json",
        "input_scaler.json",
    )
    files = {}
    for name in names:
        data = name.encode()
        (source / name).write_bytes(data)
        files[name] = hashlib.sha256(data).hexdigest()
    if defect == "source":
        (source / names[0]).write_bytes(b"changed")
    elif defect == "destination":
        destination.mkdir()
        (destination / names[0]).write_bytes(b"unrelated work")
    elif defect == "traversal":
        files["../outside"] = files.pop(names[0])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "required_files_sha256": files}), encoding="utf-8")
    if defect:
        with pytest.raises(ValueError):
            provision(source, destination, manifest)
        if defect == "destination":
            assert (destination / names[0]).read_bytes() == b"unrelated work"
            assert len(list(destination.iterdir())) == 1
        else:
            assert not destination.exists()
    else:
        assert provision(source, destination, manifest)["files_verified"] == 5
        assert provision(source, destination, manifest)["files_verified"] == 5  # Idempotent.
