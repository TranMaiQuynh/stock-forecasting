import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from quant_forecast_lab.scalers import load_minmax_scaler, minmax_from_record, minmax_to_record, save_minmax_scaler


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("clip", [False, True])
def test_numerical_scaler_preserves_constant_features_extrapolation_and_inverse(tmp_path, dtype, clip):
    train = np.asarray([[10, 3], [20, 3], [30, 3]], dtype=dtype)
    scaler = MinMaxScaler(feature_range=(-1, 2), clip=clip).fit(train)
    path = tmp_path / "scaler.json"
    save_minmax_scaler(scaler, path)
    loaded = load_minmax_scaler(path)
    probe = np.asarray([[0, 3], [15, 3], [40, 5]], dtype=dtype)
    np.testing.assert_array_equal(loaded.transform(probe), scaler.transform(probe))
    transformed = scaler.transform(probe)
    np.testing.assert_array_equal(loaded.inverse_transform(transformed), scaler.inverse_transform(transformed))


def test_named_features_keep_order_without_refitting():
    scaler = MinMaxScaler().fit(pd.DataFrame({"Close": [10.0, 20.0], "Volume": [100.0, 200.0]}))
    loaded = minmax_from_record(minmax_to_record(scaler))
    assert loaded.feature_names_in_.tolist() == ["Close", "Volume"]
    with pytest.raises(ValueError, match="feature names"):
        loaded.transform(pd.DataFrame({"Volume": [150.0], "Close": [15.0]}))


@pytest.mark.parametrize("defect", ["schema", "count", "shape", "nonfinite", "scale", "names", "range", "dtype"])
def test_invalid_scaler_state_is_rejected(defect):
    record = copy.deepcopy(minmax_to_record(MinMaxScaler().fit([[1.0], [2.0]])))
    if defect == "schema":
        record["schema_version"] = 2
    elif defect == "count":
        record["n_features"] = True
    elif defect == "shape":
        record["scale_"] = [1, 2]
    elif defect == "nonfinite":
        record["min_"] = [np.nan]
    elif defect == "scale":
        record["scale_"] = [-1]
    elif defect == "names":
        record["feature_names"] = []
    elif defect == "range":
        record["feature_range"] = [1, 0]
    else:
        record["dtype"] = "object"
    with pytest.raises(ValueError):
        minmax_from_record(record)
