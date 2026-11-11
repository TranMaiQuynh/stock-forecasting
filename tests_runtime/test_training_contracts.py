"""Exercise legacy transforms with real scalers and controlled network outputs."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import stock_prediction_deep_learning as training
from stock_prediction_class import StockPrediction
from stock_prediction_numpy import StockData


class _NoPlots:
    def __init__(self, *args):
        pass

    def __getattr__(self, name):
        return lambda *args: None


@pytest.mark.parametrize("version,use_returns,horizon", [("v1", True, 1), ("v6", False, 1), ("v6", False, 2)])
def test_historical_predictions_invert_the_actual_target_definition(tmp_path, monkeypatch, version, use_returns, horizon):
    dates = pd.bdate_range("2024-01-01", periods=45)
    frame = pd.DataFrame({"Close": 100.0 + 3 * np.arange(45)}, index=dates)
    frame.index.name = "Date"
    captured = {}

    class Data(StockData):
        def __init__(self, stock):
            super().__init__(stock)
            captured["data"] = self

    class Network:
        metrics_names = ["loss"]

        def compile(self, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            return SimpleNamespace(history={})

        def save(self, path):
            Path(path).write_text("Controlled network output; not a trained model", encoding="utf-8")

        def evaluate(self, *args, **kwargs):
            return [0.0]

        def predict(self, x):
            scaler = captured["data"].get_min_max()
            zeros = pd.DataFrame({"Close": np.zeros(len(x) * horizon)})
            return scaler.transform(zeros).reshape(len(x), horizon)

    helper = SimpleNamespace(
        create_model=lambda *args, **kwargs: Network(),
        get_optimizer=lambda *args: "adam",
        get_loss=lambda *args: "mse",
        get_defined_metrics=lambda: [],
        get_callbacks=lambda *args: [],
    )
    monkeypatch.setattr(Data, "_download_close_frame", lambda *args: frame.copy())
    monkeypatch.setattr(Data, "_get_security_info", lambda *args: {"currency": "USD"})
    monkeypatch.setattr(training, "StockData", Data)
    monkeypatch.setattr(training, "Plotter", _NoPlots)
    monkeypatch.setattr(training, "ReadmeGenerator", lambda *args: SimpleNamespace(write=lambda: None))
    monkeypatch.setattr(training, "LongShortTermMemory", lambda *args: helper)
    stock = StockPrediction("TEST", dates[0], dates[35], str(tmp_path), "", 1, 3, "", 4)
    training.train_LSTM_network(stock, use_returns=use_returns, model_version=version, forecast_horizon=horizon, trend_window=4)
    predictions = pd.read_csv(tmp_path / "predictions.csv", index_col=0)
    count = 10 - horizon + 1
    assert len(predictions) == count
    assert pd.to_datetime(predictions.index).tolist() == dates[35:35 + count].tolist()
    # Zero log return means previous observed price. Zero trend residual means
    # the extrapolated linear price, exactly the actual next close in this case.
    expected = frame.Close.iloc[34:34 + count] if use_returns else frame.Close.iloc[35:35 + count]
    np.testing.assert_allclose(predictions.iloc[:, 0], expected, rtol=1e-10, atol=1e-10)
