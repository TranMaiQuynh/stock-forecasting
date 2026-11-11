"""Exercise legacy transforms with real scalers and controlled network outputs."""

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

import stock_prediction_deep_learning as training
import stock_prediction_deep_learning_inference as inference
import stock_prediction_forecasting as forecasting
from stock_prediction_class import StockPrediction
from stock_prediction_lstm import LongShortTermMemory
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


@pytest.mark.parametrize("losses", [[3, 1, 2], [3, 1, 2, 2, 2]])
def test_best_validation_weights_survive_training_end_and_model_reload(tmp_path, losses):
    model = tf.keras.Sequential([tf.keras.Input((1,)), tf.keras.layers.Dense(1, use_bias=False)])
    model.stop_training = False  # Normally initialized by Model.fit.
    callback = LongShortTermMemory(str(tmp_path)).get_callback()
    callback.set_model(model)
    callback.on_train_begin()
    for epoch, loss in enumerate(losses):
        model.set_weights([np.array([[epoch + 1]], dtype=np.float32)])
        callback.on_epoch_end(epoch, {"val_loss": loss})
        if model.stop_training:
            break
    callback.on_train_end()
    model.save(tmp_path / "best.keras")
    loaded = tf.keras.models.load_model(tmp_path / "best.keras", compile=False)
    assert float(loaded(np.ones((1, 1))).numpy()[0, 0]) == pytest.approx(2.0)


def test_v6_recursive_inference_uses_trends_not_delta_compounding(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    source = tmp_path / "model"
    source.mkdir()
    model = tf.keras.Sequential([
        tf.keras.Input((3, 1)), tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2, kernel_initializer="zeros", bias_initializer="zeros"),
    ])
    model.save(source / "model.keras")
    scaler = MinMaxScaler().fit(pd.DataFrame({"Close": [0.0, 1.0]}))
    for name in ("min_max_scaler.pkl", "input_scaler.pkl"):
        with (source / name).open("wb") as handle:
            pickle.dump(scaler, handle)
    (source / "model_config.json").write_text(json.dumps({
        "model_version": "v6", "use_returns": False, "use_deltas": False,
        "use_trend_residual": True, "forecast_horizon": 2, "trend_window": 4,
    }), encoding="utf-8")
    frame = pd.DataFrame({"Close": 100.0 + 3 * np.arange(45)}, index=pd.bdate_range("2024-01-01", periods=45))
    frame.index.name = "Date"
    monkeypatch.setattr(StockData, "download_raw_data", lambda self: frame)
    monkeypatch.setattr(StockData, "_get_security_info", lambda self: {"currency": "USD"})
    monkeypatch.setattr(plt, "show", lambda: None)
    output = tmp_path / "forecast"
    result = forecasting.main([
        "--run-folder", str(source), "--output-folder", str(output),
        "--time-steps", "3", "--forecast-days", "3", "--ticker", "TEST",
    ])
    np.testing.assert_allclose(result.Predicted_Price_Raw, [235, 238, 241], rtol=1e-10)
    plt.close("all")


def test_residual_simulation_uses_the_same_recursive_inverse(monkeypatch):
    runner = object.__new__(inference.InferenceRunner)
    runner.stochastic_paths = 2
    runner.stochastic_seed = 42
    runner.stochastic_sigma_mult = 0.0
    runner.stochastic_lookback = 3
    runner.blend_alpha = 1.0
    monkeypatch.setattr(inference, "_estimate_noise_std", lambda *args: 1.0)
    paths = runner._build_stochastic_paths([0, 0], 120, False, False, True, pd.Series([100, 110, 120]), trend_window=3)
    np.testing.assert_allclose(paths, [[130, 140], [130, 140]])
