"""Small offline one-step log-return experiment built on maintained utilities.

No network calls, TensorFlow, automatic trading, or claimed market edge.
The reference model is frozen after training; validation chooses its settings.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .diagnostics import write_training_diagnostics
from .evaluation import evaluate_price_forecast, mae, rmse
from .experiment import sha256_file, tracked_run, write_json


@dataclass(frozen=True)
class ResearchConfig:
    asset: str = "UNSPECIFIED"
    lookbacks: tuple[int, ...] = (1, 5, 10)
    ridge_alphas: tuple[float, ...] = (1.0, 10.0)
    train_fraction: float = 0.6
    validation_fraction: float = 0.2
    max_analysis_lag: int = 20
    diagnostic_alpha: float = 0.05
    seed: int = 42
    feature_group: str = "returns"

    def validate(self):
        if not self.asset.strip():
            raise ValueError("asset cannot be empty.")
        if self.feature_group not in {"returns", "returns_abs"}:
            raise ValueError("feature_group must be returns or returns_abs.")
        if not self.lookbacks or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in self.lookbacks):
            raise ValueError("lookbacks must be positive integers.")
        if not self.ridge_alphas or any(not np.isfinite(a) or a <= 0 for a in self.ridge_alphas):
            raise ValueError("ridge_alphas must be finite and positive.")
        if len(set(self.lookbacks)) != len(self.lookbacks) or len(set(self.ridge_alphas)) != len(self.ridge_alphas):
            raise ValueError("Candidate settings must be distinct.")
        if len(self.lookbacks) * len(self.ridge_alphas) > 64:
            raise ValueError("This small reference experiment permits at most 64 trials.")
        if not 0 < self.train_fraction < 1 or not 0 < self.validation_fraction < 1:
            raise ValueError("Split fractions must be between zero and one.")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("Reserve a final test fraction.")
        if (
            not isinstance(self.max_analysis_lag, int)
            or isinstance(self.max_analysis_lag, bool)
            or self.max_analysis_lag < 1
        ):
            raise ValueError("max_analysis_lag must be a positive integer.")
        if not 0 < self.diagnostic_alpha < 1:
            raise ValueError("diagnostic_alpha must be between zero and one.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer.")


def validate_close_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Require one asset, strictly ordered unique dates and positive closes."""
    if not {"Date", "Close"}.issubset(frame.columns):
        raise ValueError("CSV must have Date and Close columns for a single asset.")
    work = frame[["Date", "Close"]].copy()
    work["Date"] = pd.to_datetime(work.Date, errors="raise", utc=True)
    work["Close"] = pd.to_numeric(work.Close, errors="raise")
    if work.Date.isna().any() or work.Date.duplicated().any() or not work.Date.is_monotonic_increasing:
        raise ValueError("Dates must be nonmissing, unique and strictly increasing; input is never silently sorted.")
    if not np.isfinite(work.Close).all() or (work.Close <= 0).any():
        raise ValueError("Close values must be finite and positive.")
    if len(work) < 60:
        raise ValueError("At least 60 observations are required for this reference experiment.")
    return work.set_index("Date")


def synthetic_close(seed=42) -> pd.DataFrame:
    """A deterministic test signal; dates and prices are entirely synthetic."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.01, 320)
    for i in range(1, len(returns)):
        returns[i] += 0.2 * returns[i - 1]
    return pd.DataFrame(
        {"Date": pd.bdate_range("2020-01-01", periods=len(returns)), "Close": 100 * np.exp(np.cumsum(returns))}
    )


def return_windows(close, target_indices, lookback):
    """At target index t use returns ending t-1,...,t-lookback, latest first."""
    values = np.asarray(close, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Window input requires finite positive one-dimensional closes.")
    if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < 1:
        raise ValueError("lookback must be a positive integer.")
    returns = np.diff(np.log(values))
    targets = np.asarray(target_indices)
    if targets.ndim != 1 or not np.issubdtype(targets.dtype, np.integer):
        raise ValueError("Target indices must be a one-dimensional integer array.")
    if (targets < lookback + 1).any() or (targets >= len(close)).any():
        raise ValueError("Target indices require sufficient observed history.")
    x = np.column_stack([returns[targets - 1 - lag] for lag in range(1, lookback + 1)])
    return x, returns[targets - 1]


def _feature_order(lookback, group):
    names = [f"return_lag_{lag}" for lag in range(1, lookback + 1)]
    if group == "returns_abs":
        names += [f"abs_return_lag_{lag}" for lag in range(1, lookback + 1)]
    return names


def feature_windows(close, target_indices, lookback, group="returns"):
    """Build causal signed-return lags and an optional predefined magnitude group."""
    if group not in {"returns", "returns_abs"}:
        raise ValueError("Unsupported feature group.")
    x, y = return_windows(close, target_indices, lookback)
    if group == "returns_abs":
        x = np.column_stack((x, np.abs(x)))
    return x, y


def predict_saved_model(model_path, close_history) -> float:
    """Predict the next log return from JSON numerical parameters, without pickle."""
    model = json.loads(Path(model_path).read_text(encoding="utf-8"))
    if model.get("schema_version") != 1 or model.get("target") != "next_observation_log_return":
        raise ValueError("Unsupported model artifact schema/target.")
    lookback = model["lookback"]
    if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < 1:
        raise ValueError("Invalid model lookback.")
    group = model.get("feature_group", "returns")
    if group not in {"returns", "returns_abs"} or model.get("feature_order") != _feature_order(lookback, group):
        raise ValueError("Incompatible feature order in model artifact.")
    history = np.asarray(close_history, dtype=float)
    if history.ndim != 1 or len(history) < lookback + 1 or not np.isfinite(history).all() or (history <= 0).any():
        raise ValueError("Inference needs finite positive closes with enough observed history.")
    # The placeholder target is never observed by the feature builder.
    x, _ = feature_windows(np.append(history, history[-1]), [len(history)], lookback, group)
    x = x[0]
    mean, scale, coef = (np.asarray(model[name], dtype=float) for name in ("mean", "scale", "coef"))
    if any(v.shape != x.shape or not np.isfinite(v).all() for v in (mean, scale, coef)) or (scale <= 0).any():
        raise ValueError("Invalid scaler/model parameter shapes or values.")
    prediction = float(((x - mean) / scale) @ coef + model["intercept"])
    if not np.isfinite(prediction):
        raise ValueError("Model prediction must be finite.")
    return prediction


def _return_metrics(actual, predicted):
    baseline_error = rmse(actual, np.zeros(len(actual)))
    model_error = rmse(actual, predicted)
    return {
        "rmse": model_error,
        "mae": mae(actual, predicted),
        "directional_accuracy": float(np.mean(np.sign(actual) == np.sign(predicted))),
        "rmse_skill_vs_zero": 1 - model_error / baseline_error
        if baseline_error
        else (0.0 if model_error == 0 else None),
    }


def _figures(predictions, close, train_end, validation_end, directory, label):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), layout="constrained")
    fig.suptitle(label, fontsize=9)
    close.Close.plot(ax=axes[0], label="Observed close")
    for index, label in ((train_end, "Validation starts"), (validation_end, "Test starts")):
        axes[0].axvline(close.index[index], linestyle="--", label=label)
    axes[0].set(title="Observed input and chronological boundaries", ylabel="Close (input units)")
    axes[0].legend()
    test = predictions[predictions.split == "test"]
    times = pd.to_datetime(test.target_timestamp)
    axes[1].plot(times, test.actual_log_return, label="Actual")
    axes[1].plot(times, test.predicted_log_return, label="Ridge")
    axes[1].axhline(0, color="black", linewidth=0.7, label="Zero-return baseline")
    axes[1].set(
        title="One-step test forecasts from observed prior history", ylabel="Log return", xlabel="Target observation"
    )
    axes[1].legend()
    try:
        fig.savefig(directory / "forecast.png", dpi=120)
    finally:
        plt.close(fig)


def run_offline_experiment(
    input_csv=None, *, output_root="log", config=None, data_kind="user_supplied", split_boundaries=None
) -> Path:
    """Run a fixed chronological holdout with a bounded validation-only search.

    input_csv=None explicitly generates a labelled synthetic smoke experiment.
    This is a reference pipeline, not a replacement for legacy neural training.
    """
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from statsmodels.stats.diagnostic import acorr_ljungbox

    config = config or ResearchConfig()
    config.validate()
    if data_kind not in {"user_supplied", "synthetic", "market"}:
        raise ValueError("Unknown data_kind.")
    kind = "synthetic" if input_csv is None else data_kind
    settings = {
        **asdict(config),
        "data_kind": kind,
        "input_csv": str(input_csv) if input_csv else None,
        "protocol": "fixed chronological holdout; validation selects Ridge; no refit; no trading simulation",
        "explicit_split_boundaries": split_boundaries,
    }
    with tracked_run(output_root, "ridge_log_return", settings) as (run, logger):
        raw = synthetic_close(config.seed) if input_csv is None else pd.read_csv(input_csv)
        close = validate_close_frame(raw)
        if split_boundaries is None:
            train_end = int(len(close) * config.train_fraction)
            validation_end = train_end + int(len(close) * config.validation_fraction)
        else:
            if (
                len(split_boundaries) != 3
                or any(isinstance(n, bool) or not isinstance(n, int) for n in split_boundaries)
                or not 0 < split_boundaries[0] < split_boundaries[1] < split_boundaries[2] <= len(close)
            ):
                raise ValueError("split_boundaries must be ordered integer train/validation/test stops.")
            train_end, validation_end, test_end = split_boundaries
            close = close.iloc[:test_end].copy()
        close.to_csv(run / "input_snapshot.csv")
        write_json(
            run / "data_manifest.json",
            {
                "data_kind": kind,
                "observations": len(close),
                "snapshot_sha256": sha256_file(run / "input_snapshot.csv"),
                "input_sha256": sha256_file(input_csv) if input_csv else None,
                "frequency": "input observation order; exchange calendar and point-in-time validity unverified",
                "missing_policy": "reject; no imputation or missing-close removal",
                "timezone_policy": "Dates normalized to UTC",
                "asset": config.asset,
                "first": str(close.index[0]),
                "last": str(close.index[-1]),
            },
        )
        targets = np.arange(max(config.lookbacks) + 1, len(close))
        train = targets < train_end
        validation = (targets >= train_end) & (targets < validation_end)
        test = targets >= validation_end
        if train.sum() < 12 or validation.sum() < 5 or test.sum() < 5:
            raise ValueError("Insufficient common training/validation/test targets for candidate lookbacks.")
        split_record = {
            "horizon": 1,
            "target": "next_observation_log_return",
            "refit": False,
            "window_order": "return_lag_1 (latest observed) through return_lag_L",
            "common_training_targets": True,
            "training_series_end_exclusive": train_end,
        }
        for name, mask in (("train", train), ("validation", validation), ("test", test)):
            indices = targets[mask]
            split_record[name] = {
                "count": len(indices),
                "first_target": str(close.index[indices[0]]),
                "last_target": str(close.index[indices[-1]]),
            }
        write_json(run / "split_manifest.json", split_record)
        logger.info("Schema passed; train=%s validation=%s test=%s", train.sum(), validation.sum(), test.sum())
        write_training_diagnostics(
            close.Close.iloc[:train_end],
            run / "analysis",
            max_analysis_lag=config.max_analysis_lag,
            alpha=config.diagnostic_alpha,
            label=f"{kind.upper()} | {config.asset} | {run.name}",
        )
        logger.info("Training-only stationarity and correlation diagnostics saved")
        trials, best = [], None
        for lookback in config.lookbacks:
            x, y = feature_windows(close.Close, targets, lookback, config.feature_group)
            for ridge_alpha in config.ridge_alphas:
                model = make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha, solver="svd"))
                try:
                    model.fit(x[train], y[train])
                    prediction = model.predict(x[validation])
                    score = rmse(y[validation], prediction)
                except Exception as exc:
                    trials.append(
                        {
                            "lookback": lookback,
                            "ridge_alpha": ridge_alpha,
                            "status": "failed",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                    pd.DataFrame(trials).to_csv(run / "trials.csv", index=False)
                    raise
                trials.append(
                    {
                        "lookback": lookback,
                        "ridge_alpha": ridge_alpha,
                        "training_samples": int(train.sum()),
                        "validation_samples": int(validation.sum()),
                        "validation_return_rmse": score,
                        "status": "completed",
                    }
                )
                pd.DataFrame(trials).to_csv(run / "trials.csv", index=False)
                logger.info("Trial lookback=%d ridge_alpha=%g validation_return_rmse=%g", lookback, ridge_alpha, score)
                if best is None or score < best[0]:
                    best = score, lookback, ridge_alpha, model
        _, lookback, ridge_alpha, model = best
        x, y = feature_windows(close.Close, targets, lookback, config.feature_group)
        scaler, estimator = model.steps[0][1], model.steps[1][1]
        feature_order = _feature_order(lookback, config.feature_group)
        model_record = {
            "schema_version": 1,
            "target": "next_observation_log_return",
            "lookback": lookback,
            "ridge_alpha": ridge_alpha,
            "feature_group": config.feature_group,
            "feature_order": feature_order,
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "coef": estimator.coef_.tolist(),
            "intercept": float(estimator.intercept_),
        }
        artifact = run / "artifacts/model.json"
        write_json(artifact, model_record)
        write_json(
            run / "artifacts/feature_schema.json",
            {
                "feature_order": feature_order,
                "horizon": 1,
                "latest_allowed_observation": "forecast_origin",
                "units": "log return",
            },
        )
        write_json(
            run / "analysis/feature_report.json",
            {
                "base_series": 1,
                "selected_lookback": lookback,
                "scalar_features": x.shape[1],
                "training_shape": list(x[train].shape),
                "fitted_coefficients": x.shape[1] + 1,
                "selection": "validation RMSE; no ACF significance filter",
                "independent_samples": False,
                "feature_group": config.feature_group,
                "definitions": "signed return lags; optional absolute values of the identical observed lags",
            },
        )
        (run / "model_summary.txt").write_text(
            "StandardScaler -> Ridge(solver=svd)\nTarget: one-step log return\n"
            f"Input shape: {x[train].shape}\nLookback: {lookback}\nRidge alpha: {ridge_alpha}\n"
            f"Coefficients including intercept: {x.shape[1] + 1}\nFeature order: {feature_order}\n"
            "Fit scope: training only. Validation selects configuration. No refit. No epochs.\n",
            encoding="utf-8",
        )
        predicted = model.predict(x)
        values = close.Close.to_numpy()
        rows = []
        for split_name, mask in (("validation", validation), ("test", test)):
            for index in np.flatnonzero(mask):
                t = targets[index]
                rows.append(
                    {
                        "run_id": run.name,
                        "model": "ridge_log_return",
                        "asset": config.asset,
                        "split": split_name,
                        "forecast_origin": str(close.index[t - 1]),
                        "target_timestamp": str(close.index[t]),
                        "horizon": 1,
                        "actual_log_return": y[index],
                        "predicted_log_return": predicted[index],
                        "baseline_log_return": 0.0,
                        "origin_close": values[t - 1],
                        "actual_close": values[t],
                        "predicted_close": values[t - 1] * np.exp(predicted[index]),
                        "baseline_close": values[t - 1],
                    }
                )
        predictions = pd.DataFrame(rows)
        if not np.isfinite(predictions.select_dtypes(include="number")).all().all():
            raise ValueError("Predictions overflowed or contain nonfinite values.")
        predictions.to_csv(run / "predictions.csv", index=False)
        reloaded = [predict_saved_model(artifact, values[:t]) for t in targets[test]]
        np.testing.assert_allclose(reloaded, predicted[test], rtol=1e-10, atol=1e-12)
        metrics = {}
        for name in ("validation", "test"):
            subset = predictions[predictions.split == name]
            metrics[name] = {
                "observations": len(subset),
                "ridge_log_return": _return_metrics(subset.actual_log_return, subset.predicted_log_return),
                "zero_log_return": _return_metrics(subset.actual_log_return, np.zeros(len(subset))),
                "ridge_price": evaluate_price_forecast(
                    subset.actual_close, subset.predicted_close, subset.origin_close
                ).as_dict(),
            }
        write_json(run / "metrics.json", metrics)
        residuals = predictions.loc[predictions.split == "test", "actual_log_return"].to_numpy() - predicted[test]
        residual_report = {
            "mean_error": float(residuals.mean()),
            "n_obs": len(residuals),
            "interpretation": "Final test reporting only; no retuning. No independence guarantee.",
        }
        if len(residuals) >= 20 and np.ptp(residuals) > 0:
            lag = min(10, len(residuals) // 5)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                lb = acorr_ljungbox(residuals, lags=[lag], model_df=0, return_df=True)
            residual_report.update(
                ljung_box=lb.reset_index(names="lag").to_dict(orient="records"),
                model_df=0,
                warnings=[str(w.message) for w in caught],
                model_df_note="Exploratory linear-reference diagnostic, not an ARMA parameter correction.",
            )
        else:
            residual_report["ljung_box_status"] = "not_applicable_short_or_constant_residuals"
        write_json(run / "residual_diagnostics.json", residual_report)
        future = {
            "forecast_origin": str(close.index[-1]),
            "target_timestamp": None,
            "horizon": 1,
            "target_policy": "next observed session; no calendar date guessed",
            "actual": None,
            "data_kind": kind,
            "asset": config.asset,
            "predicted_log_return": predict_saved_model(artifact, values),
        }
        future["predicted_close"] = float(values[-1] * np.exp(future["predicted_log_return"]))
        write_json(run / "future_forecast.json", future)
        (run / "forecast_values.txt").write_text(
            predictions.to_string(index=False) + "\n\nFuture:\n" + json.dumps(future, indent=2), encoding="utf-8"
        )
        _figures(
            predictions,
            close,
            train_end,
            validation_end,
            run / "figures",
            label=f"{kind.upper()} | {config.asset} | {run.name}",
        )
        write_json(
            run / "capabilities.json",
            {
                "artifact_reload": "verified",
                "uncertainty": "not_implemented",
                "walk_forward_training": "not_implemented_in_this_runner",
                "backtest": "withheld_no_execution_contract",
                "learning_curve": "not_applicable_closed_form_ridge",
                "feature_group_ablation": "single group only; compare groups with the walk-forward orchestrator",
            },
        )
        (run / "summary.md").write_text(
            f"# Offline research run\n\nData kind: **{kind}**; asset: {config.asset}.\n\n"
            f"Selected lookback: {lookback}; Ridge alpha: {ridge_alpha}. No refit.\n\n"
            f"Feature group: {config.feature_group}.\n\n"
            f"Test log-return RMSE: {metrics['test']['ridge_log_return']['rmse']:.8f}.\n"
            f"Zero-return RMSE: {metrics['test']['zero_log_return']['rmse']:.8f}.\n\n"
            "Training-only diagnostics are exploratory. Validation chose the Ridge configuration; "
            "the final test was only scored after selection. No claim of market improvement or promotion.\n\n"
            "Artifact reload checked on every test origin. No calibrated uncertainty, "
            "walk-forward training, execution simulation or deployment is claimed.\n",
            encoding="utf-8",
        )
        required = [
            "data_manifest.json",
            "split_manifest.json",
            "analysis/stationary_test.txt",
            "analysis/acf_pacf_values.csv",
            "model_summary.txt",
            "trials.csv",
            "predictions.csv",
            "metrics.json",
            "artifacts/model.json",
            "future_forecast.json",
            "figures/forecast.png",
            "summary.md",
        ]
        if any(not (run / p).is_file() or not (run / p).stat().st_size for p in required):
            raise RuntimeError("Required experiment artifacts are missing or empty.")
        logger.info("Final test evaluated; saved artifact matched all test-origin predictions")
    return run
