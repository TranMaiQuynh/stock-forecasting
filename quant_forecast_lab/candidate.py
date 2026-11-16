"""One bounded refit of a pinned numerical recipe; no search or promotion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .artifacts import verify_run
from .experiment import sha256_file, tracked_run, write_json
from .research import feature_windows, predict_saved_model
from .session_contract import forecast_contract, utc_timestamp
from .shadow import _effective, _outside_sources, load_snapshot


def _refit_recipe(previous, config, close):
    kind = previous.get("estimator_kind", "ridge")
    if kind not in {"ridge", "zero", "training_mean"}:
        raise ValueError("Unsupported source estimator.")
    lookback = previous["lookback"]
    targets = np.arange(max(config["lookbacks"]) + 1, len(close))
    window = config.get("training_window")
    if window is not None:
        targets = targets[-window:]
    if len(targets) < 12:
        raise ValueError("Insufficient eligible training labels for the pinned recipe.")
    x, y = feature_windows(close.Close, targets, lookback, previous.get("feature_group", "returns"))
    model = {**previous, "estimator_kind": kind, "mean": [0.0] * x.shape[1], "scale": [1.0] * x.shape[1],
             "coef": [0.0] * x.shape[1], "intercept": float(y.mean()) if kind == "training_mean" else 0.0,
             "ridge_alpha": None, "training_samples": len(targets),
             "first_training_target_index": int(targets[0]), "last_training_target_index": int(targets[-1]),
             "first_training_target": str(close.index[targets[0]]), "last_training_target": str(close.index[targets[-1]]),
             "fit_stop_exclusive": len(close), "fit_cutoff": str(close.index[-1])}
    if kind == "ridge":
        penalty_mode = previous.get("penalty_mode", "absolute")
        penalty = previous.get("penalty", previous["ridge_alpha"])
        if penalty_mode not in {"absolute", "per_sample"} or penalty is None or not np.isfinite(penalty) or penalty <= 0:
            raise ValueError("Invalid source Ridge penalty contract.")
        alpha = penalty * (len(targets) if penalty_mode == "per_sample" else 1)
        scaler = StandardScaler().fit(x)
        fitted = Ridge(alpha=alpha, solver="svd").fit(scaler.transform(x), y)
        model.update(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(), coef=fitted.coef_.tolist(),
                     intercept=float(fitted.intercept_), ridge_alpha=float(alpha), penalty=penalty, penalty_mode=penalty_mode)
    return model


def prepare_candidate(source_run, snapshot_run, *, mode, as_of=None, calendar="XNYS", output_root="log"):
    """Fit the selected estimator/lookback/penalty once on available labels.

    Source selection is inherited and disclosed, not rerun. This creates a new
    fixed-checkpoint operational candidate, not independent confirmation of v2.
    """
    _outside_sources((source_run, snapshot_run), output_root=output_root, run_name="shadow_model")
    effective = _effective(mode, as_of)
    source = Path(source_run).resolve()
    verified = verify_run(source)
    if verified["model_name"] not in {"ridge_log_return", "return_forecast"}:
        raise ValueError("Refit requires an original selected research run, not a recursive candidate.")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    data = json.loads((source / "data_manifest.json").read_text(encoding="utf-8"))
    split = json.loads((source / "split_manifest.json").read_text(encoding="utf-8"))
    previous = json.loads((source / "artifacts/model.json").read_text(encoding="utf-8"))
    close, receipt = load_snapshot(snapshot_run, config["asset"])
    if (receipt["data_kind"] == "synthetic") != (data["data_kind"] == "synthetic"):
        raise ValueError("Do not mix synthetic and market/user-supplied model lineages.")
    if mode == "live" and receipt["data_kind"] == "synthetic":
        raise ValueError("Synthetic training is replay-only.")
    contract = forecast_contract(close, calendar_name=calendar, as_of=effective,
                                 received_at=receipt["received_at"], mode=mode)
    cutoff = max(utc_timestamp(split["validation"]["last_target"]),
                 utc_timestamp(split.get("model_fit_cutoff", split["validation"]["last_target"])))
    if close.index[-1] < cutoff:
        raise ValueError("Refit data ends before the source fit/selection cutoff.")
    # Validate serialized schema, feature order and dimensions before fitting.
    predict_saved_model(source / "artifacts/model.json", close.Close)
    window = config.get("training_window")
    settings = {
        "asset": config["asset"], "data_kind": receipt["data_kind"], "mode": mode,
        "source_manifest_sha256": verified["manifest_sha256"], "source_run": str(source),
        "operation": "one_fixed_recipe_refit_no_search", "source_refit_policy": config.get("refit_policy", "frozen"),
        "serving_update_rule": "frozen_checkpoint", "training_window": window,
        "v2_research_confirmation": False,
    }
    with tracked_run(output_root, "shadow_model", settings) as (run, logger):
        model = _refit_recipe(previous, config, close)
        kind, lookback = model["estimator_kind"], model["lookback"]
        close.to_csv(run / "training_snapshot.csv")
        write_json(run / "availability_contract.json", contract)
        write_json(run / "source_model.json", previous)
        write_json(run / "artifacts/model.json", model)
        write_json(run / "data_manifest.json", {
            "asset": config["asset"], "data_kind": receipt["data_kind"], "snapshot_sha256": sha256_file(run / "training_snapshot.csv"),
            "receipt_sha256": receipt["input_sha256"], "point_in_time_verified": False,
            "price_basis": "Caller must ensure source-compatible adjusted closes; vendor revision history unverified",
        })
        write_json(run / "split_manifest.json", {
            "target": "next_observation_log_return", "horizon": 1,
            "validation": {"last_target": split["validation"]["last_target"]},
            "model_fit_cutoff": str(close.index[-1]), "evaluation": "not_performed_no_future_labels",
            "selection": "Inherited recipe; no validation or test search on the new snapshot",
        })
        prediction = predict_saved_model(run / "artifacts/model.json", close.Close)
        (run / "model_summary.txt").write_text(
            f"Estimator: {kind}\nTraining targets: {model['training_samples']}\nLookback: {lookback}\n"
            f"Ridge alpha: {model['ridge_alpha']}\nFit cutoff: {model['fit_cutoff']}\n"
            f"Feature order: {model['feature_order']}\nReloaded next-return prediction: {prediction}\n"
            "One fit of a pinned recipe; frozen serving. No performance evaluation or promotion.\n",
            encoding="utf-8",
        )
        logger.info("Prepared one %s candidate on %d labels; no parameter search or future performance claim", kind, model["training_samples"])
    return run
