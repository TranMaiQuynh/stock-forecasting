import json
import runpy
from pathlib import Path

import pytest

from quant_forecast_lab.experiment import sha256_file, tracked_run, write_json

validate_snapshot = runpy.run_path(str(Path(__file__).resolve().parents[1] / "benchmarks/run_reference_research.py"))[
    "validate_snapshot"
]


@pytest.mark.parametrize("defect", [None, "asset", "adjustment", "checksum"])
def test_protocol_rejects_wrong_asset_adjustment_or_modified_snapshot(tmp_path, defect):
    source = {
        "start": "2020-01-01",
        "end_exclusive": "2021-01-01",
        "interval": "1d",
        "auto_adjust": True,
        "back_adjust": False,
        "repair": False,
        "keepna": True,
    }
    options = {**source, "end": source["end_exclusive"]}
    if defect == "adjustment":
        options["auto_adjust"] = False
    with tracked_run(tmp_path, "snapshot", {}) as (folder, _):
        (folder / "close.csv").write_text("Date,Close\n2020-01-01,100\n", encoding="utf-8")
        write_json(
            folder / "data_manifest.json",
            {
                "asset": "WRONG" if defect == "asset" else "TEST",
                "download_options": options,
                "yfinance_version": "1.0",
                "provider": "Yahoo Finance via yfinance",
                "snapshot_sha256": sha256_file(folder / "close.csv"),
            },
        )
    if defect == "checksum":
        (folder / "close.csv").write_text("tampered", encoding="utf-8")
    if defect:
        with pytest.raises(ValueError):
            validate_snapshot("TEST", folder, source)
    else:
        assert validate_snapshot("TEST", folder, source) == folder / "close.csv"
        assert json.loads((folder / "manifest.json").read_text())["status"] == "completed"
