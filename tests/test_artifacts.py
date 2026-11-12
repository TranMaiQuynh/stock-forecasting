import json

import pytest

from quant_forecast_lab.artifacts import verify_run
from quant_forecast_lab.experiment import tracked_run, write_json


@pytest.mark.parametrize("defect", [None, "changed", "missing", "extra", "traversal", "failed", "config_hash"])
def test_run_verification_checks_inventory_state_and_checksums(tmp_path, defect):
    with tracked_run(tmp_path, "example", {}) as (run, _):
        write_json(run / "artifacts/model.json", {"parameter": 1})
    path = run / "manifest.json"
    manifest = json.loads(path.read_text())
    if defect == "changed":
        (run / "artifacts/model.json").write_text("modified", encoding="utf-8")
    elif defect == "missing":
        (run / "artifacts/model.json").unlink()
    elif defect == "extra":
        (run / "untracked.txt").write_text("extra", encoding="utf-8")
    elif defect == "traversal":
        manifest["artifact_sha256"]["../outside"] = "0" * 64
    elif defect == "failed":
        manifest["status"] = "failed"
    elif defect == "config_hash":
        manifest["config_sha256"] = "0" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")
    if defect:
        with pytest.raises(ValueError):
            verify_run(run)
    else:
        assert verify_run(run)["verified_artifacts"] == len(manifest["artifact_sha256"])
