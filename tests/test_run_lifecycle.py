import json

import pytest

from quant_forecast_lab.experiment import sha256_file, tracked_run, write_json


def test_failed_run_keeps_traceback_and_artifacts(tmp_path):
    with pytest.raises(ValueError, match="invalid data"):
        with tracked_run(tmp_path, "ridge", {"data_kind": "synthetic"}) as (run, logger):
            write_json(run / "partial.json", {"stage": "ingest"})
            logger.info("Validating input")
            raise ValueError("invalid data")
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["artifact_sha256"]["partial.json"] == sha256_file(run / "partial.json")
    assert "ValueError: invalid data" in (run / "run.log").read_text()


def test_runs_are_unique_and_completed_manifest_hashes_closed_log(tmp_path):
    paths = []
    for _ in range(2):
        with tracked_run(tmp_path, "ridge", {}) as (run, _):
            paths.append(run)
        manifest = json.loads((run / "manifest.json").read_text())
        assert manifest["status"] == "completed"
        assert manifest["artifact_sha256"]["run.log"] == sha256_file(run / "run.log")
    assert paths[0] != paths[1]


def test_parent_hashes_child_manifest_but_not_its_own_manifest(tmp_path):
    with tracked_run(tmp_path, "parent", {}) as (parent, _):
        with tracked_run(parent / "folds", "child", {}) as (child, _):
            write_json(child / "metrics.json", {"rmse": 0.1})
    manifest = json.loads((parent / "manifest.json").read_text())
    child_path = (child / "manifest.json").relative_to(parent).as_posix()
    assert manifest["artifact_sha256"][child_path] == sha256_file(child / "manifest.json")
    assert "manifest.json" not in manifest["artifact_sha256"]
