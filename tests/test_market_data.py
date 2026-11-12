import json

import numpy as np
import pandas as pd
import pytest

from quant_forecast_lab.experiment import sha256_file
from quant_forecast_lab.market_data import acquire_market_snapshot


def _vendor_frame():
    return pd.DataFrame({"Close": 100 + np.arange(80)}, index=pd.bdate_range("2020-01-01", periods=80))


def test_snapshot_preserves_request_provenance_and_exact_validated_data(tmp_path):
    requests = []

    def download(**kwargs):
        requests.append(kwargs)
        return _vendor_frame()

    run = acquire_market_snapshot("TEST", "2020-01-01", "2021-01-01", output_root=tmp_path, downloader=download)
    assert requests[0]["auto_adjust"] is True
    assert requests[0]["keepna"] is True and requests[0]["repair"] is False
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    for name, digest in manifest["artifact_sha256"].items():
        assert sha256_file(run / name) == digest
    provenance = json.loads((run / "data_manifest.json").read_text())
    assert provenance["point_in_time_verified"] is False
    assert provenance["snapshot_sha256"] == sha256_file(run / "close.csv")
    pd.testing.assert_series_equal(pd.read_csv(run / "close.csv").Close, _vendor_frame().Close.reset_index(drop=True))


@pytest.mark.parametrize("defect", ["missing", "duplicate", "out_of_range", "empty"])
def test_invalid_provider_data_retains_failed_run_without_a_valid_snapshot(tmp_path, defect):
    frame = _vendor_frame().astype(float)
    if defect == "missing":
        frame.iloc[5, 0] = np.nan
    elif defect == "duplicate":
        frame.index = pd.DatetimeIndex([frame.index[0]] * len(frame))
    elif defect == "out_of_range":
        frame.index = frame.index + pd.DateOffset(years=3)
    else:
        frame = frame.iloc[:0]
    with pytest.raises(ValueError):
        acquire_market_snapshot("TEST", "2020-01-01", "2021-01-01", output_root=tmp_path, downloader=lambda **kw: frame)
    (run,) = (tmp_path / "market_snapshot").iterdir()
    assert json.loads((run / "manifest.json").read_text())["status"] == "failed"
    assert not (run / "close.csv").exists()
