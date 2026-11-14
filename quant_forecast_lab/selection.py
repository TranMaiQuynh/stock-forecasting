"""Causal baseline-eligible selection and explicit numerical refit checkpoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import rmse
from .experiment import write_json


def select_return_model(close, targets, train_end, validation_end, config, run):
    """Select on chronological validation, then replay the chosen fixed policy.

    Monthly fits use target labels strictly before the forecast target. The
    schedule depends on the observed origin month, never the future session.
    The final exported checkpoint is for frozen inference; a consumer must not
    use it at an origin earlier than its fit/selection cutoff.
    """
    from .research import feature_windows

    run = Path(run)
    values = close.Close.to_numpy()
    first_target = int(targets[0])
    candidates = []
    if config.include_baselines:
        candidates += [{"kind": kind, "lookback": min(config.lookbacks), "penalty": None} for kind in ("zero", "training_mean")]
    candidates += [
        {"kind": "ridge", "lookback": lookback, "penalty": penalty}
        for lookback in config.lookbacks
        for penalty in config.ridge_alphas
    ]
    fits, trials, validation_results = [], [], []
    cache = {lookback: feature_windows(values, targets, lookback, config.feature_group) for lookback in config.lookbacks}

    def fit(candidate, stop, relative):
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        indices = np.flatnonzero(targets < stop)
        if config.training_window is not None:
            indices = indices[-config.training_window :]
        if len(indices) < 12:
            raise ValueError("A policy fit requires at least 12 eligible training targets.")
        x, y = cache[candidate["lookback"]]
        lookback = candidate["lookback"]
        width = x.shape[1]
        record = {
            "schema_version": 1,
            "target": "next_observation_log_return",
            "estimator_kind": candidate["kind"],
            "lookback": lookback,
            "feature_group": config.feature_group,
            "feature_order": [f"return_lag_{i}" for i in range(1, lookback + 1)]
            + ([f"abs_return_lag_{i}" for i in range(1, lookback + 1)] if config.feature_group == "returns_abs" else []),
            "mean": [0.0] * width,
            "scale": [1.0] * width,
            "coef": [0.0] * width,
            "intercept": float(y[indices].mean()) if candidate["kind"] == "training_mean" else 0.0,
            "ridge_alpha": None,
            "penalty": candidate["penalty"],
            "penalty_mode": config.penalty_mode,
            "training_samples": len(indices),
            "first_training_target_index": int(targets[indices[0]]),
            "last_training_target_index": int(targets[indices[-1]]),
            "first_training_target": str(close.index[targets[indices[0]]]),
            "last_training_target": str(close.index[targets[indices[-1]]]),
            "fit_stop_exclusive": int(stop),
            "fit_cutoff": str(close.index[stop - 1]),
        }
        if candidate["kind"] == "ridge":
            scaler = StandardScaler().fit(x[indices])
            alpha = candidate["penalty"] * (len(indices) if config.penalty_mode == "per_sample" else 1)
            estimator = Ridge(alpha=alpha, solver="svd").fit(scaler.transform(x[indices]), y[indices])
            record.update(
                mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                coef=estimator.coef_.tolist(), intercept=float(estimator.intercept_), ridge_alpha=float(alpha),
            )
        write_json(run / relative, record)
        fits.append({"model_artifact": relative, **record})
        return record

    def replay(candidate, start, stop, folder, initial=None):
        result, paths = [], []
        checkpoint, relative = initial if initial is not None else (None, None)
        previous_month = None
        for t in range(start, stop):
            origin = close.index[t - 1]
            month = (origin.year, origin.month)
            if checkpoint is None or (
                config.refit_policy == "monthly" and previous_month is not None and month != previous_month
            ):
                relative = f"{folder}/model_{t:06d}.json"
                checkpoint = fit(candidate, t, relative)
            x, _ = cache[candidate["lookback"]]
            feature = x[t - first_target]
            prediction = float(
                ((feature - np.asarray(checkpoint["mean"])) / np.asarray(checkpoint["scale"]))
                @ np.asarray(checkpoint["coef"]) + checkpoint["intercept"]
            )
            if not np.isfinite(prediction):
                raise ValueError("Policy prediction is not finite.")
            result.append(prediction)
            paths.append(relative)
            previous_month = month
        return np.asarray(result), paths, checkpoint

    best_index, best_score = None, float("inf")
    for i, candidate in enumerate(candidates):
        row = {
            "trial": i, **candidate, "penalty_mode": config.penalty_mode,
            "ridge_alpha": candidate["penalty"] if config.penalty_mode == "absolute" else None,
            "training_samples": min(train_end - first_target, config.training_window or train_end),
            "validation_samples": validation_end - train_end,
        }
        try:
            predicted, paths, checkpoint = replay(
                candidate, train_end, validation_end, f"artifacts/trials/trial_{i:03d}/validation"
            )
            score = rmse(cache[candidate["lookback"]][1][(targets >= train_end) & (targets < validation_end)], predicted)
            row.update(status="completed", validation_return_rmse=score)
        except Exception as exc:
            row.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            trials.append(row)
            pd.DataFrame(trials).to_csv(run / "trials.csv", index=False)
            raise
        trials.append(row)
        pd.DataFrame(trials).to_csv(run / "trials.csv", index=False)
        validation_results.append((predicted, paths, checkpoint))
        if score < best_score - (1e-12 if config.include_baselines else 0):
            best_index, best_score = i, score
    candidate = candidates[best_index]
    write_json(run / "selection.json", {
        "selected_trial": best_index, "selected_candidate": candidate,
        "validation_return_rmse": best_score,
        "selection_cutoff": str(close.index[validation_end - 1]),
        "tie_policy": "recorded candidate order within the stated absolute RMSE tolerance",
        "tie_tolerance": 1e-12 if config.include_baselines else 0.0,
        "refit_policy": config.refit_policy, "training_window": config.training_window,
    })
    valid_predicted, valid_paths, validation_checkpoint = validation_results[best_index]
    # Frozen policies retain the original training model. Monthly policies refit
    # once selection is complete, then use the same origin-month schedule.
    initial = (validation_checkpoint, valid_paths[-1]) if config.refit_policy == "frozen" else None
    test_predicted, test_paths, checkpoint = replay(
        candidate, validation_end, len(close), "artifacts/selected/test", initial=initial,
    )
    mean_candidate = next((c for c in candidates if c["kind"] == "training_mean"), None)
    if mean_candidate is not None:
        mean_index = candidates.index(mean_candidate)
        mean_validation, mean_paths, mean_checkpoint = validation_results[mean_index]
        mean_initial = (mean_checkpoint, mean_paths[-1]) if config.refit_policy == "frozen" else None
        mean_test, _, _ = replay(mean_candidate, validation_end, len(close), "artifacts/baseline/test", initial=mean_initial)
        baseline = np.concatenate((mean_validation, mean_test))
    else:
        baseline = None
    predictions = np.full(len(targets), np.nan)
    predictions[targets >= train_end] = np.concatenate((valid_predicted, test_predicted))
    write_json(run / "fit_history.json", fits)
    return {
        "candidate": candidate, "model_record": checkpoint, "predicted": predictions,
        "model_paths": dict(zip(targets[targets >= train_end].tolist(), valid_paths + test_paths, strict=True)),
        "training_mean_predictions": baseline, "fit_count": len(fits),
    }
