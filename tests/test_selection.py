import json
import socket
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_forecast_lab.research import predict_saved_model, synthetic_close, validate_close_frame
from quant_forecast_lab.selection import select_return_model


def policy(**overrides):
    return SimpleNamespace(**{
        "include_baselines": True, "lookbacks": (1, 3), "ridge_alphas": (0.1,),
        "feature_group": "returns", "penalty_mode": "per_sample", "refit_policy": "frozen",
        "training_window": None, **overrides,
    })


def read(path):
    return json.loads(path.read_text())


@pytest.mark.parametrize("return_value,expected", [(0.0, "zero"), (0.01, "training_mean")])
def test_simple_baseline_can_win_with_deterministic_ties(tmp_path, return_value, expected):
    frame = pd.DataFrame({"Date": pd.bdate_range("2020-01-01", periods=120),
                          "Close": 100 * np.exp(np.arange(120) * return_value)})
    close = validate_close_frame(frame)
    result = select_return_model(close, np.arange(4, 120), 60, 90, policy(), tmp_path)
    assert result["candidate"]["kind"] == expected
    assert result["model_record"]["intercept"] == pytest.approx(return_value, abs=1e-14)
    assert len(pd.read_csv(tmp_path / "trials.csv")) == 4
    assert read(tmp_path / "selection.json")["selection_cutoff"] == str(close.index[89])
    assert read(tmp_path / "selection.json")["tie_tolerance"] == 1e-12


def test_zero_can_displace_a_training_mean_that_fails_validation(tmp_path):
    returns = np.r_[np.full(60, 0.01), np.full(59, -0.01)]
    close = validate_close_frame(pd.DataFrame({
        "Date": pd.bdate_range("2020-01-01", periods=120), "Close": 100 * np.exp(np.r_[0, returns].cumsum()),
    }))
    result = select_return_model(close, np.arange(4, 120), 60, 90, policy(), tmp_path)
    assert result["candidate"]["kind"] == "zero"
    np.testing.assert_array_equal(result["predicted"][56:], 0.0)


@pytest.mark.parametrize("penalty_mode", ["absolute", "per_sample"])
def test_penalty_and_scaler_use_the_exact_common_training_targets(tmp_path, penalty_mode):
    close = validate_close_frame(synthetic_close().iloc[:120])
    targets = np.arange(4, 120)
    select_return_model(close, targets, 60, 90, policy(penalty_mode=penalty_mode, training_window=30), tmp_path)
    records = read(tmp_path / "fit_history.json")
    ridge = [r for r in records if r["estimator_kind"] == "ridge"]
    assert len(ridge) == 2
    for record in ridge:
        assert record["training_samples"] == 30
        assert record["first_training_target_index"] == 30
        assert record["last_training_target_index"] == 59
        assert record["ridge_alpha"] == pytest.approx(3.0 if penalty_mode == "per_sample" else 0.1)
        from quant_forecast_lab.research import feature_windows

        x, _ = feature_windows(close.Close, np.arange(30, 60), record["lookback"])
        np.testing.assert_allclose(record["mean"], x.mean(axis=0))


def test_monthly_checkpoints_are_causal_reloadable_and_rolling(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("Selection attempted network access")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    close = validate_close_frame(synthetic_close().iloc[:160])
    targets = np.arange(4, len(close))
    result = select_return_model(close, targets, 60, 95, policy(refit_policy="monthly", training_window=35), tmp_path)
    for t in range(60, len(close)):
        checkpoint = tmp_path / result["model_paths"][t]
        record = read(checkpoint)
        assert record["last_training_target_index"] < t
        assert pd.Timestamp(record["fit_cutoff"]) <= close.index[t - 1]
        assert record["training_samples"] == 35
        assert predict_saved_model(checkpoint, close.Close.iloc[:t]) == pytest.approx(result["predicted"][t - 4])
        if t > 95:
            changed = result["model_paths"][t] != result["model_paths"][t - 1]
            origin_month = close.index[t - 1].strftime("%Y-%m")
            prior_month = close.index[t - 2].strftime("%Y-%m")
            assert changed == (origin_month != prior_month)
    first_test = read(tmp_path / result["model_paths"][95])
    assert first_test["last_training_target_index"] == 94
    assert len(set(result["model_paths"][t] for t in range(95, len(close)))) >= 3


def test_future_changes_cannot_change_selection_or_earlier_refits(tmp_path):
    frame = synthetic_close().iloc[:160].copy()
    first = validate_close_frame(frame)
    targets = np.arange(4, len(frame))
    config = policy(include_baselines=False, refit_policy="monthly", training_window=35)
    original = select_return_model(first, targets, 60, 95, config, tmp_path / "a")
    frame.loc[115:, "Close"] *= 2
    other = select_return_model(validate_close_frame(frame), targets, 60, 95, config, tmp_path / "b")
    assert (tmp_path / "a/selection.json").read_bytes() == (tmp_path / "b/selection.json").read_bytes()
    assert read(tmp_path / "a/selection.json")["tie_tolerance"] == 0.0
    np.testing.assert_allclose(original["predicted"][56:112], other["predicted"][56:112])
    for relative in {original["model_paths"][t] for t in range(60, 116)}:
        assert (tmp_path / "a" / relative).read_bytes() == (tmp_path / "b" / relative).read_bytes()


def test_failed_ridge_trial_is_not_silently_replaced_with_a_baseline(tmp_path, monkeypatch):
    from sklearn.linear_model import Ridge

    def broken_fit(*args, **kwargs):
        raise RuntimeError("deliberate fit failure")

    monkeypatch.setattr(Ridge, "fit", broken_fit)
    close = validate_close_frame(synthetic_close().iloc[:120])
    with pytest.raises(RuntimeError, match="deliberate fit failure"):
        select_return_model(close, np.arange(4, 120), 60, 90, policy(), tmp_path)
    assert pd.read_csv(tmp_path / "trials.csv").status.tolist() == ["completed", "completed", "failed"]
    assert not (tmp_path / "selection.json").exists()
