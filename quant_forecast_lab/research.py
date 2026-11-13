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
    include_baselines: bool = False
    penalty_mode: str = "absolute"
    refit_policy: str = "frozen"
    training_window: int | None = None

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
        if len(self.lookbacks) * len(self.ridge_alphas) + (2 if self.include_baselines else 0) > 64:
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
        if not isinstance(self.include_baselines, bool):
            raise ValueError("include_baselines must be boolean.")
        if self.penalty_mode not in {"absolute", "per_sample"}:
            raise ValueError("penalty_mode must be absolute or per_sample.")
        if self.refit_policy not in {"frozen", "monthly"}:
            raise ValueError("refit_policy must be frozen or monthly.")
        if self.training_window is not None and (
            isinstance(self.training_window, bool) or not isinstance(self.training_window, int) or self.training_window < 12
        ):
            raise ValueError("training_window must be None or an integer of at least 12 target observations.")


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
    axes[1].plot(times, test.predicted_log_return, label="Selected model")
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
    from statsmodels.stats.diagnostic import acorr_ljungbox

    from .selection import select_return_model

    config = config or ResearchConfig()
    config.validate()
    if data_kind not in {"user_supplied", "synthetic", "market"}:
        raise ValueError("Unknown data_kind.")
    kind = "synthetic" if input_csv is None else data_kind
    settings = {
        **asdict(config),
        "data_kind": kind,
        "input_csv": str(input_csv) if input_csv else None,
        "protocol": "chronological validation selects a fixed candidate; explicit causal refit policy; no trading simulation",
        "explicit_split_boundaries": split_boundaries,
    }
    model_name = "return_forecast" if config.include_baselines else "ridge_log_return"
    metric_name = "selected_log_return" if config.include_baselines else "ridge_log_return"
    with tracked_run(output_root, model_name, settings) as (run, logger):
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
            "refit": config.refit_policy != "frozen",
            "refit_policy": config.refit_policy,
            "training_window": config.training_window,
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
        diagnostic_start = max(0, train_end - config.training_window - max(config.lookbacks) - 1) if config.training_window else 0
        write_training_diagnostics(
            close.Close.iloc[diagnostic_start:train_end],
            run / "analysis",
            max_analysis_lag=config.max_analysis_lag,
            alpha=config.diagnostic_alpha,
            label=f"{kind.upper()} | {config.asset} | {run.name}",
        )
        logger.info("Training-only stationarity and correlation diagnostics saved")
        selected = select_return_model(close, targets, train_end, validation_end, config, run)
        model_record = selected["model_record"]
        lookback, ridge_alpha = model_record["lookback"], model_record["ridge_alpha"]
        predicted = selected["predicted"]
        x, y = feature_windows(close.Close, targets, lookback, config.feature_group)
        feature_order = _feature_order(lookback, config.feature_group)
        split_record["model_fit_cutoff"] = model_record["fit_cutoff"]
        write_json(run / "split_manifest.json", split_record)
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
                "training_shape": [min(int(train.sum()), config.training_window or int(train.sum())), x.shape[1]],
                "fitted_coefficients": x.shape[1] + 1 if model_record["estimator_kind"] == "ridge" else int(model_record["estimator_kind"] == "training_mean"),
                "selection": "chronological validation RMSE; baselines eligible when configured; no ACF significance filter",
                "diagnostic_scope": "initial fit history only; later fit scopes in fit_history.json; diagnostics do not select refits",
                "independent_samples": False,
                "feature_group": config.feature_group,
                "definitions": "signed return lags; optional absolute values of the identical observed lags",
            },
        )
        (run / "model_summary.txt").write_text(
            f"Selected estimator: {model_record['estimator_kind']}\nTarget: one-step log return\n"
            f"Input shape: {x[train].shape}\nLookback: {lookback}\nRidge alpha: {ridge_alpha}\n"
            f"Numerical coefficients including intercept: {x.shape[1] + 1}\nFeature order: {feature_order}\n"
            f"Fit cutoff: {model_record['fit_cutoff']}; refit policy: {config.refit_policy}.\n"
            f"Training samples at exported checkpoint: {model_record['training_samples']}; penalty mode: {config.penalty_mode}.\n"
            "Every checkpoint and preprocessing state is recorded; no epochs.\n",
            encoding="utf-8",
        )
        values = close.Close.to_numpy()
        rows = []
        for split_name, mask in (("validation", validation), ("test", test)):
            for index in np.flatnonzero(mask):
                t = targets[index]
                rows.append(
                    {
                        "run_id": run.name,
                        "model": model_record["estimator_kind"],
                        "model_artifact": selected["model_paths"][int(t)],
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
        if selected["training_mean_predictions"] is not None:
            predictions["training_mean_log_return"] = selected["training_mean_predictions"]
        if not np.isfinite(predictions.select_dtypes(include="number")).all().all():
            raise ValueError("Predictions overflowed or contain nonfinite values.")
        predictions.to_csv(run / "predictions.csv", index=False)
        reloaded = [predict_saved_model(run / selected["model_paths"][int(t)], values[:t]) for t in targets[test]]
        np.testing.assert_allclose(reloaded, predicted[test], rtol=1e-10, atol=1e-12)
        metrics = {}
        for name in ("validation", "test"):
            subset = predictions[predictions.split == name]
            metrics[name] = {
                "observations": len(subset),
                metric_name: _return_metrics(subset.actual_log_return, subset.predicted_log_return),
                "zero_log_return": _return_metrics(subset.actual_log_return, np.zeros(len(subset))),
                ("selected_price" if config.include_baselines else "ridge_price"): evaluate_price_forecast(
                    subset.actual_close, subset.predicted_close, subset.origin_close
                ).as_dict(),
            }
            metrics[name]["always_up_directional_accuracy"] = float(np.mean(subset.actual_log_return > 0))
            if "training_mean_log_return" in subset:
                metrics[name]["training_mean"] = _return_metrics(subset.actual_log_return, subset.training_mean_log_return)
                baseline_error = metrics[name]["training_mean"]["rmse"]
                metrics[name][metric_name]["rmse_skill_vs_training_mean"] = (
                    1 - metrics[name][metric_name]["rmse"] / baseline_error if baseline_error else None
                )
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
            "estimator_kind": model_record["estimator_kind"],
            "model_fit_cutoff": model_record["fit_cutoff"],
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
                "walk_forward_training": config.refit_policy,
                "training_window": config.training_window,
                "fit_checkpoints": selected["fit_count"],
                "backtest": "withheld_no_execution_contract",
                "learning_curve": "not_applicable_closed_form_ridge",
                "feature_group_ablation": "single group only; compare groups with the walk-forward orchestrator",
            },
        )
        (run / "summary.md").write_text(
            f"# Offline research run\n\nData kind: **{kind}**; asset: {config.asset}.\n\n"
            f"Selected estimator: {model_record['estimator_kind']}; lookback: {lookback}; Ridge alpha: {ridge_alpha}.\n\n"
            f"Refit policy: {config.refit_policy}; training window: {config.training_window}; penalty mode: {config.penalty_mode}.\n\n"
            f"Feature group: {config.feature_group}.\n\n"
            f"Test log-return RMSE: {metrics['test'][metric_name]['rmse']:.8f}.\n"
            f"Zero-return RMSE: {metrics['test']['zero_log_return']['rmse']:.8f}.\n\n"
            "Initial-training diagnostics are exploratory. Validation chose the candidate; "
            "the final test was only scored after selection. No claim of market improvement or promotion.\n\n"
            "Each test origin was checked against its own saved checkpoint. Refer to fit_history.json for every fit cutoff. "
            "No calibrated uncertainty, execution simulation or deployment is claimed.\n",
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
