import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

import stock_prediction_deep_learning_inference as inference
from quant_forecast_lab.scalers import save_minmax_scaler


@pytest.mark.parametrize(
    "name,loader",
    [
        ("min_max_scaler", inference._load_scaler),
        ("input_scaler", inference._load_input_scaler),
    ],
)
def test_json_preferred_and_corruption_never_falls_back_to_pickle(tmp_path, monkeypatch, name, loader):
    scaler = MinMaxScaler().fit(pd.DataFrame({"Close": [100.0, 200.0]}))
    save_minmax_scaler(scaler, tmp_path / f"{name}.json")
    (tmp_path / f"{name}.pkl").write_bytes(b"not a valid pickle")

    def no_pickle(*args):
        raise AssertionError("JSON loading attempted pickle deserialization")

    monkeypatch.setattr(pickle, "load", no_pickle)
    np.testing.assert_allclose(loader(tmp_path).transform(pd.DataFrame({"Close": [150.0]})), [[0.5]])
    (tmp_path / f"{name}.json").write_text('{"schema_version": 999}', encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        loader(tmp_path)
